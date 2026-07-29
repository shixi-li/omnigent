"""Native-subagent routing: loopback endpoint + server-side policy.

Harness ``PreToolUse`` hooks (Claude ``Task``, Codex ``spawn_agent``) call
this before a native subagent spawn so the spawn cannot proceed on a
model the router did not approve. Two halves live here:

* **Runner side** — a loopback HTTP relay (:func:`start_subagent_router`)
  on ``127.0.0.1:0``, bearer-token authenticated, advertised to hook
  scripts via ``subagent_router.json`` in the session bridge dir. Same
  rendezvous as ``tool_relay.json``.
* **Server side** — the policy (:func:`resolve_subagent_route`), which
  runs where ``RuntimeCaps.routing_client`` lives, caches per
  (session, task) with a TTL so the blocking spawn path stays fast, and
  persists every decision as a ``routing_decision`` transcript item.

The two halves are joined by the server relay route
:data:`SERVER_ROUTE_PATH` (registered in
``omnigent/server/routes/sessions/routes_hooks.py``); the runner's
handler forwards to it with :func:`make_server_relay_resolver`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

_logger = logging.getLogger(__name__)

#: Bridge-dir file that advertises the loopback endpoint to hook scripts.
ADVERTISEMENT_FILE = "subagent_router.json"

#: Path served by the loopback relay, per the frozen endpoint contract.
ROUTE_PATH_TEMPLATE = "/v1/sessions/{session_id}/route-subagent"

#: Server relay route the runner-side handler forwards to (the server
#: process is the only one holding ``RuntimeCaps.routing_client``).
SERVER_ROUTE_PATH = "/v1/sessions/{session_id}/hooks/route-subagent"

#: ``RoutingSettings`` fallbacks used when caps carry no settings yet.
DEFAULT_FAIL_MODE: Literal["open", "closed"] = "open"
DEFAULT_CACHE_TTL_S = 300.0

_SCOPE = "native_subagent"
_PROMPT_CAP = 4000

# Cross-harness counterpart used to offer the router a second family, and
# to name the redirect target when it picks from that family.
_COUNTERPART_HARNESS: dict[str, str] = {
    "claude-sdk": "codex",
    "claude-native": "codex-native",
    "codex": "claude-sdk",
    "codex-native": "claude-native",
}

Resolver = Callable[[str, "SubagentRouteRequest"], Awaitable["SubagentRouteDecision"]]


# ── Wire types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SubagentRouteRequest:
    """One native-subagent spawn awaiting a routing verdict.

    :param harness: Requesting harness id, e.g. ``"claude-native"``.
    :param task_name: Subagent type / task name from the spawn payload,
        e.g. ``"code-reviewer"``.
    :param prompt: Raw task text. ``None`` on codex, whose spawn message
        is encrypted in hook payloads.
    :param fork: ``True`` when the spawn is a fork of the parent session.
    :param parent_model: Model the parent session runs on, e.g.
        ``"databricks-claude-sonnet-4-6"``.
    """

    harness: str
    task_name: str = ""
    prompt: str | None = None
    fork: bool = False
    parent_model: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SubagentRouteRequest:
        """Parse a request body into a :class:`SubagentRouteRequest`.

        :param payload: Decoded JSON object from the hook script.
        :returns: Parsed request.
        :raises ValueError: If ``harness`` is missing or not a string.
        """
        harness = payload.get("harness")
        if not isinstance(harness, str) or not harness.strip():
            raise ValueError("route-subagent body requires a non-empty 'harness' string")
        task_name = payload.get("task_name")
        prompt = payload.get("prompt")
        parent_model = payload.get("parent_model")
        return cls(
            harness=harness.strip(),
            task_name=task_name if isinstance(task_name, str) else "",
            prompt=prompt if isinstance(prompt, str) and prompt else None,
            fork=bool(payload.get("fork")),
            parent_model=parent_model if isinstance(parent_model, str) and parent_model else None,
        )


@dataclass(frozen=True)
class SubagentRouteDecision:
    """The verdict a hook script enforces on a spawn.

    :param action: ``"allow"`` (spawn unchanged), ``"rewrite"`` (same
        harness, injected model), ``"redirect"`` (cross-harness — deny
        and tell the model to use ``sys_session_send``) or ``"deny"``.
    :param model: Servable model id; set for rewrite/redirect.
    :param harness: Target harness; set for redirect.
    :param raw_model: Router-vocabulary pick before resolution.
    :param rationale: One-line explanation, surfaced to the model and UI.
    :param decision_id: Identity shared by the response, the transcript
        item and telemetry.
    """

    action: Literal["allow", "rewrite", "redirect", "deny"]
    rationale: str
    model: str | None = None
    harness: str | None = None
    raw_model: str | None = None
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the frozen response shape.

        :returns: JSON-ready response body.
        """
        return {
            "action": self.action,
            "model": self.model,
            "harness": self.harness,
            "raw_model": self.raw_model,
            "rationale": self.rationale,
            "decision_id": self.decision_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SubagentRouteDecision:
        """Parse a response body (used by the runner-side relay).

        :param payload: Decoded JSON response from the server route.
        :returns: Parsed decision; unknown actions degrade to ``allow``.
        """
        action = payload.get("action")
        if action not in ("allow", "rewrite", "redirect", "deny"):
            action = "allow"
        rationale = payload.get("rationale")
        decision_id = payload.get("decision_id")
        return cls(
            action=action,
            rationale=rationale if isinstance(rationale, str) else "",
            model=_opt_str(payload.get("model")),
            harness=_opt_str(payload.get("harness")),
            raw_model=_opt_str(payload.get("raw_model")),
            decision_id=decision_id if isinstance(decision_id, str) else str(uuid.uuid4()),
        )


@dataclass(frozen=True)
class _Settings:
    """Subagent-relevant slice of ``RoutingSettings`` (§5.4)."""

    fail_mode: Literal["open", "closed"] = DEFAULT_FAIL_MODE
    cache_ttl_s: float = DEFAULT_CACHE_TTL_S


# P7: swap this mirror for ``RoutingDecisionData`` once P5's additive
# fields land — field names already match the frozen shape.
@dataclass(frozen=True)
class SubagentDecisionRecord:
    """Transcript payload for one native-subagent routing decision."""

    model: str
    applied: bool
    rationale: str
    decision_id: str
    harness: str | None = None
    raw_model: str | None = None
    agent: str | None = None
    attempted_override: str | None = None
    scope: str = _SCOPE

    def to_item_data(self) -> dict[str, Any]:
        """Return the ``routing_decision`` item data dict.

        :returns: Item data carrying the full field set.
        """
        return {
            "model": self.model,
            "applied": self.applied,
            "rationale": self.rationale,
            "agent": self.agent,
            "harness": self.harness,
            "scope": self.scope,
            "decision_id": self.decision_id,
            "raw_model": self.raw_model,
            "attempted_override": self.attempted_override,
        }


def decision_record(
    req: SubagentRouteRequest,
    decision: SubagentRouteDecision,
) -> SubagentDecisionRecord:
    """Build the transcript record for *decision*.

    :param req: The spawn that was routed.
    :param decision: The verdict returned to the hook.
    :returns: Record ready for :func:`persist_subagent_decision`.
    """
    model = decision.model or req.parent_model or "unrouted"
    return SubagentDecisionRecord(
        model=model,
        applied=decision.action in ("rewrite", "redirect"),
        rationale=decision.rationale,
        decision_id=decision.decision_id,
        harness=decision.harness or req.harness,
        raw_model=decision.raw_model,
        agent=req.task_name or None,
    )


# ── Decision cache ─────────────────────────────────────────────────────────


@dataclass
class _CacheEntry:
    decision: SubagentRouteDecision
    expires_at: float


_cache: dict[tuple[str, str], _CacheEntry] = {}
# Per-session pick log, shaped to populate the router's ``session_history``
# once a shipped recipe reads it.
_picks: dict[str, list[dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def task_cache_key(req: SubagentRouteRequest) -> str:
    """Hash the routing-relevant fields of a spawn.

    :param req: Spawn request.
    :returns: Hex digest identifying identical spawns.
    """
    material = "\x00".join(
        [req.harness, req.task_name, req.prompt or "", req.parent_model or ""],
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _cached(session_id: str, key: str, now: float) -> SubagentRouteDecision | None:
    with _cache_lock:
        entry = _cache.get((session_id, key))
        if entry is None:
            return None
        if entry.expires_at <= now:
            del _cache[(session_id, key)]
            return None
        return entry.decision


def _remember(
    session_id: str,
    key: str,
    req: SubagentRouteRequest,
    decision: SubagentRouteDecision,
    ttl_s: float,
    now: float,
) -> None:
    if ttl_s <= 0:
        return
    with _cache_lock:
        _cache[(session_id, key)] = _CacheEntry(decision=decision, expires_at=now + ttl_s)
        _picks.setdefault(session_id, []).append(
            {
                "task_name": req.task_name,
                "model": decision.model,
                "harness": decision.harness or req.harness,
                "decision_id": decision.decision_id,
                "at": now,
            }
        )


def session_picks(session_id: str) -> tuple[dict[str, Any], ...]:
    """Return this session's routed picks, oldest first.

    :param session_id: Session/conversation identifier.
    :returns: Pick records ``{task_name, model, harness, decision_id, at}``.
    """
    with _cache_lock:
        return tuple(_picks.get(session_id, ()))


def clear_cache(session_id: str | None = None) -> None:
    """Drop cached decisions and picks.

    :param session_id: Only this session when set, otherwise everything.
    """
    with _cache_lock:
        if session_id is None:
            _cache.clear()
            _picks.clear()
            return
        for cache_key in [k for k in _cache if k[0] == session_id]:
            del _cache[cache_key]
        _picks.pop(session_id, None)


# ── Policy ─────────────────────────────────────────────────────────────────


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _resolved_settings(caps: Any) -> _Settings:
    """Read the §5.4 settings off *caps*, falling back to defaults."""
    settings = getattr(caps, "routing_settings", None)
    fail_mode = getattr(settings, "subagent_fail_mode", None)
    if fail_mode not in ("open", "closed"):
        fail_mode = DEFAULT_FAIL_MODE
    ttl = getattr(settings, "subagent_cache_ttl_s", None)
    if not isinstance(ttl, (int, float)):
        ttl = DEFAULT_CACHE_TTL_S
    return _Settings(fail_mode=fail_mode, cache_ttl_s=float(ttl))


def _harness_family(harness: str) -> str | None:
    from omnigent.server.smart_routing import _HARNESS_FAMILY

    return _HARNESS_FAMILY.get(harness)


def candidate_models(harness: str) -> dict[str, list[str]]:
    """Build the harness → models map offered to the router.

    Offers the requesting harness plus its cross-harness counterpart so
    the router can move a task to the other family; the route-options
    seam turns this into the router's scenario menu.

    :param harness: Requesting harness id, e.g. ``"codex-native"``.
    :returns: Harness → model ids, cheapest first, empty entries dropped.
    """
    from omnigent.server.smart_routing import infer_models

    result: dict[str, list[str]] = {}
    for candidate in (harness, _COUNTERPART_HARNESS.get(harness)):
        if candidate is None or candidate in result:
            continue
        models = infer_models(candidate)
        if models:
            result[candidate] = list(models)
    return result


def _routing_prompt(req: SubagentRouteRequest) -> str:
    """Return the raw task text the router should score."""
    if req.prompt:
        return req.prompt[:_PROMPT_CAP]
    # Codex encrypts the spawn message, so ``task_name`` is the only signal.
    return req.task_name[:_PROMPT_CAP]


def _fail_mode_decision(settings: _Settings, reason: str) -> SubagentRouteDecision:
    if settings.fail_mode == "closed":
        return SubagentRouteDecision(
            action="deny",
            rationale=f"Subagent routing is required but unavailable: {reason}",
        )
    return SubagentRouteDecision(
        action="allow",
        rationale=f"Routing unavailable ({reason}); spawn allowed unchanged",
    )


def _target_harness(
    req_harness: str, picked_harness: str | None, picked_family: str | None
) -> str:
    """Name the harness a picked model should run on."""
    if picked_family is not None and picked_family == _harness_family(req_harness):
        return req_harness
    counterpart = _COUNTERPART_HARNESS.get(req_harness)
    if counterpart is not None and _harness_family(counterpart) == picked_family:
        return counterpart
    return picked_harness or counterpart or req_harness


async def resolve_subagent_route(
    session_id: str,
    req: SubagentRouteRequest,
    *,
    caps: Any = None,
    available_models: dict[str, list[str]] | None = None,
    persist: Callable[[SubagentDecisionRecord], Awaitable[None]] | None = None,
    now: float | None = None,
) -> SubagentRouteDecision:
    """Decide what happens to one native-subagent spawn.

    :param session_id: Parent session/conversation identifier.
    :param req: The spawn awaiting a verdict.
    :param caps: ``RuntimeCaps``-shaped object. ``None`` reads the
        process-global caps.
    :param available_models: Candidate harness → models map. ``None``
        derives it from the requesting harness.
    :param persist: Coroutine that records the decision in the
        transcript. ``None`` skips persistence (unit tests, dry runs).
    :param now: Monotonic-ish clock override for cache tests.
    :returns: The verdict the hook script enforces.
    """
    if caps is None:
        from omnigent.runtime import get_caps

        caps = get_caps()
    settings = _resolved_settings(caps)
    clock = time.time() if now is None else now

    decision = await _decide(session_id, req, caps, settings, available_models, clock)
    if persist is not None:
        try:
            await persist(decision_record(req, decision))
        except Exception:
            _logger.exception("route-subagent: decision persist failed for session=%s", session_id)
    return decision


async def _decide(
    session_id: str,
    req: SubagentRouteRequest,
    caps: Any,
    settings: _Settings,
    available_models: dict[str, list[str]] | None,
    clock: float,
) -> SubagentRouteDecision:
    if req.fork:
        return SubagentRouteDecision(
            action="allow",
            rationale="Fork keeps the parent's model; forks are not routed",
            model=req.parent_model,
        )

    key = task_cache_key(req)
    cached = _cached(session_id, key, clock)
    if cached is not None:
        return cached

    client = getattr(caps, "routing_client", None)
    if client is None:
        return _fail_mode_decision(settings, "no routing client configured")

    candidates = (
        available_models if available_models is not None else candidate_models(req.harness)
    )
    if not candidates:
        return _fail_mode_decision(settings, f"no candidate models for harness {req.harness}")

    try:
        result = await client.route(_routing_prompt(req), candidates)
    except Exception:  # noqa: BLE001 — router outages are a normal path here
        _logger.warning(
            "route-subagent: router call failed for session=%s", session_id, exc_info=True
        )
        result = None
    if result is None or not getattr(result, "model", None):
        detail = _opt_str(getattr(client, "last_error", None)) or "router returned no verdict"
        return _fail_mode_decision(settings, detail)

    decision = _decision_from_result(req, result, candidates)
    if decision.action != "deny":
        _remember(session_id, key, req, decision, settings.cache_ttl_s, clock)
    return decision


def _decision_from_result(
    req: SubagentRouteRequest,
    result: Any,
    candidates: Mapping[str, list[str]],
) -> SubagentRouteDecision:
    model = result.model
    rationale = getattr(result, "rationale", "") or ""
    raw_model = _opt_str(getattr(result, "raw_model", None)) or model
    offered = {m for models in candidates.values() for m in models}
    if offered and model not in offered:
        # Never let an unoffered pick through: "didn't spawn" beats "wrong model".
        return SubagentRouteDecision(
            action="deny",
            rationale=f"Router picked {model}, which this harness cannot run",
            raw_model=raw_model,
        )

    picked_harness = _opt_str(getattr(result, "harness", None))
    family = _harness_family(picked_harness) if picked_harness else None
    if family is None:
        for harness_id, models in candidates.items():
            if model in models:
                family = _harness_family(harness_id)
                break
    target = _target_harness(req.harness, picked_harness, family)

    if target == req.harness:
        if req.parent_model is not None and model == req.parent_model:
            return SubagentRouteDecision(
                action="allow",
                rationale=rationale or "Router kept the parent model",
                model=model,
                raw_model=raw_model,
            )
        return SubagentRouteDecision(
            action="rewrite",
            rationale=rationale or f"Router selected {model}",
            model=model,
            raw_model=raw_model,
        )
    return SubagentRouteDecision(
        action="redirect",
        rationale=rationale or f"Router selected {target}/{model}",
        model=model,
        harness=target,
        raw_model=raw_model,
    )


# ── Transcript persistence ─────────────────────────────────────────────────


async def persist_subagent_decision(
    session_id: str,
    conversation_store: Any,
    record: SubagentDecisionRecord,
) -> None:
    """Persist and publish *record* as a ``routing_decision`` item.

    Drops the additive §5.2 fields when the installed
    ``RoutingDecisionData`` does not accept them yet, so the chip still
    lands on a store built before the model extension.

    :param session_id: Session/conversation identifier.
    :param conversation_store: Store exposing ``append``.
    :param record: Decision payload.
    """
    from omnigent.entities.conversation import NewConversationItem, parse_item_data
    from omnigent.runtime import session_stream

    item_data = record.to_item_data()
    try:
        parsed = parse_item_data("routing_decision", item_data)
    except (ValueError, TypeError):
        legacy = {k: item_data[k] for k in ("model", "applied", "rationale", "agent")}
        try:
            parsed = parse_item_data("routing_decision", legacy)
        except (ValueError, TypeError):
            _logger.warning("route-subagent: failed to parse routing_decision data")
            return

    item = NewConversationItem(
        type="routing_decision",
        response_id=f"routing_{uuid.uuid4().hex}",
        data=parsed,
    )
    try:
        persisted = await asyncio.to_thread(conversation_store.append, session_id, [item])
        persisted_id: str | None = persisted[0].id if persisted else None
    except Exception:
        _logger.exception(
            "route-subagent: routing_decision persist failed for session=%s", session_id
        )
        persisted_id = None

    session_stream.publish(
        session_id,
        {
            "type": "response.output_item.done",
            "item": {"id": persisted_id, "type": "routing_decision", **item_data},
        },
    )


def store_persister(
    session_id: str,
    conversation_store: Any,
) -> Callable[[SubagentDecisionRecord], Awaitable[None]]:
    """Bind :func:`persist_subagent_decision` to a session and store.

    :param session_id: Session/conversation identifier.
    :param conversation_store: Store exposing ``append``.
    :returns: Coroutine function accepting a record.
    """

    async def _persist(record: SubagentDecisionRecord) -> None:
        await persist_subagent_decision(session_id, conversation_store, record)

    return _persist


# ── Runner-side loopback relay ─────────────────────────────────────────────


def write_advertisement(bridge_dir: Path, *, url: str, token: str) -> Path:
    """Advertise the endpoint to hook scripts.

    :param bridge_dir: Session bridge directory.
    :param url: Endpoint base URL, e.g. ``"http://127.0.0.1:53421"``.
    :param token: Bearer token hook scripts must present.
    :returns: Path of the written ``subagent_router.json``.
    """
    bridge_dir.mkdir(parents=True, exist_ok=True)
    path = bridge_dir / ADVERTISEMENT_FILE
    tmp = path.with_name(f"{path.name}.tmp")
    payload = {"url": url, "token": token, "pid": os.getpid(), "updated_at": time.time()}
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def read_advertisement(bridge_dir: Path) -> dict[str, str] | None:
    """Read the endpoint advertisement (used by hook scripts).

    :param bridge_dir: Session bridge directory.
    :returns: ``{"url": ..., "token": ...}``, or ``None`` when the file
        is absent or malformed.
    """
    try:
        payload = json.loads((bridge_dir / ADVERTISEMENT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    url = _opt_str(payload.get("url"))
    token = _opt_str(payload.get("token"))
    if url is None or token is None:
        return None
    return {"url": url, "token": token}


@dataclass
class SubagentRouter:
    """Handle for the running loopback router.

    :param bridge_dir: Bridge dir holding the advertisement file.
    :param url: Base URL the hook scripts POST to.
    :param token: Bearer token hook scripts present.
    :param httpd: The backing HTTP server.
    """

    bridge_dir: Path
    url: str
    token: str
    httpd: ThreadingHTTPServer

    def close(self) -> None:
        """Stop the server and remove the advertisement file."""
        self.httpd.shutdown()
        self.httpd.server_close()
        with contextlib.suppress(OSError):
            (self.bridge_dir / ADVERTISEMENT_FILE).unlink()


def start_subagent_router(
    *,
    bridge_dir: Path,
    session_id: str,
    resolver: Resolver,
    loop: asyncio.AbstractEventLoop,
    fail_mode: Literal["open", "closed"] = DEFAULT_FAIL_MODE,
    request_timeout_s: float = 60.0,
) -> SubagentRouter:
    """Start the loopback router and advertise it in *bridge_dir*.

    :param bridge_dir: Session bridge directory.
    :param session_id: Session this router serves; requests for any
        other session id are rejected.
    :param resolver: Coroutine function returning a decision.
    :param loop: Event loop that owns *resolver*.
    :param fail_mode: Behavior when the resolver errors or times out.
    :param request_timeout_s: Seconds to wait for a verdict before
        applying *fail_mode*.
    :returns: Started router handle; call :meth:`SubagentRouter.close`
        when the session ends.
    """
    token = secrets.token_urlsafe(32)
    handler_cls = _handler_factory(session_id, token, resolver, loop, fail_mode, request_timeout_s)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    host, port = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host}:{port}"
    write_advertisement(bridge_dir, url=url, token=token)
    threading.Thread(
        target=httpd.serve_forever,
        name="omnigent-subagent-router",
        daemon=True,
    ).start()
    return SubagentRouter(bridge_dir=bridge_dir, url=url, token=token, httpd=httpd)


def _handler_factory(
    session_id: str,
    token: str,
    resolver: Resolver,
    loop: asyncio.AbstractEventLoop,
    fail_mode: Literal["open", "closed"],
    request_timeout_s: float,
) -> type[BaseHTTPRequestHandler]:
    """Build the request handler class for one session's router."""
    expected_path = ROUTE_PATH_TEMPLATE.format(session_id=session_id)
    settings = _Settings(fail_mode=fail_mode)

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            if self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {"error": "unauthorized"})
                return
            if self.path.rstrip("/") != expected_path:
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("body must be a JSON object")
                req = SubagentRouteRequest.from_payload(payload)
            except (ValueError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
                return
            try:
                future = asyncio.run_coroutine_threadsafe(resolver(session_id, req), loop)
                decision = future.result(timeout=request_timeout_s)
            except Exception:  # noqa: BLE001 — never wedge the spawn path
                _logger.warning(
                    "route-subagent: resolver failed for session=%s", session_id, exc_info=True
                )
                decision = _fail_mode_decision(settings, "routing endpoint failed")
            self._send(200, decision.to_payload())

        def _send(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            _logger.debug("subagent-router: " + format, *args)

    return _Handler


def make_server_relay_resolver(
    server_client: Any,
    *,
    fail_mode: Literal["open", "closed"] = DEFAULT_FAIL_MODE,
    timeout_s: float = 60.0,
) -> Resolver:
    """Build a resolver that forwards to the server's relay route.

    The policy runs server-side because only that process holds
    ``RuntimeCaps.routing_client``. When the hop itself fails the
    server's settings are unreachable, so *fail_mode* decides locally.

    :param server_client: Async HTTP client pointed at the AP server.
    :param fail_mode: Behavior when the server hop fails — ``"open"``
        allows the spawn unchanged, ``"closed"`` denies it.
    :param timeout_s: Per-request timeout in seconds.
    :returns: Resolver for :func:`start_subagent_router`.
    """
    settings = _Settings(fail_mode=fail_mode)

    async def _resolve(session_id: str, req: SubagentRouteRequest) -> SubagentRouteDecision:
        body = {
            "harness": req.harness,
            "task_name": req.task_name,
            "prompt": req.prompt,
            "fork": req.fork,
            "parent_model": req.parent_model,
        }
        try:
            resp = await server_client.post(
                SERVER_ROUTE_PATH.format(session_id=session_id),
                json=body,
                timeout=timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:  # noqa: BLE001 — server hop failures are expected
            _logger.warning(
                "route-subagent: server relay failed for session=%s", session_id, exc_info=True
            )
            return _fail_mode_decision(settings, "routing server unreachable")
        if not isinstance(payload, dict):
            return _fail_mode_decision(settings, "unreadable verdict from routing server")
        return SubagentRouteDecision.from_payload(payload)

    return _resolve
