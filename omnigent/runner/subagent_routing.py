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
import tempfile
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import httpx

    from omnigent.entities.conversation import RoutingDecisionData

from omnigent._platform import stable_user_id
from omnigent.runtime.telemetry import ROUTING_EVENT_DECISION, emit_routing_event

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

#: Conversation label carrying the routing decision behind a session's
#: ``model_override``, so the child-sessions API can join the two without
#: a new column or a transcript scan.
ROUTING_DECISION_LABEL_KEY = "omnigent.routing.decision_id"

#: Conversation label marking a session created in auto-harness mode. The
#: ``harness_override`` sentinel is replaced the moment first-message routing
#: resolves a harness, so this label is the durable record that the router
#: may still move this session's subagents across harness families.
AUTO_HARNESS_LABEL_KEY = "omnigent.routing.auto_harness"

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


# ── Enablement gate ────────────────────────────────────────────────────────


def routing_enabled(
    cost_control_mode: str | None,
    *,
    parent_cost_control_mode: str | None = None,
    caps: Any = None,
) -> bool:
    """Report whether intelligent routing is on for one session.

    The shared gate for main-agent routing, so "routing is on" means the
    same thing everywhere the server decides to route. Two conditions:
    the per-session toggle (its own, or its parent's for a spawned child
    — the server routes children of a routed parent), and, where a
    ``RuntimeCaps`` is in reach, a configured routing client. Subagent
    spawns read :func:`subagent_routing_enabled`, which layers the
    per-session subagent override on top of this.

    :param cost_control_mode: The session's ``cost_control_mode_override``,
        e.g. ``"on"``.
    :param parent_cost_control_mode: The parent session's value for a
        spawned child. ``None`` for a top-level session.
    :param caps: ``RuntimeCaps``-shaped object whose ``routing_client``
        must be configured. ``None`` skips that check — the runner
        process holds no routing client (the server relay owns the policy
        and its fail mode), so runner-side callers pass nothing.
    :returns: ``True`` when routing applies to this session.
    """
    if cost_control_mode != "on" and parent_cost_control_mode != "on":
        return False
    if caps is None:
        return True
    return getattr(caps, "routing_client", None) is not None


def subagent_routing_enabled(
    subagent_routing_override: str | None,
    *,
    cost_control_mode: str | None,
    parent_cost_control_mode: str | None = None,
) -> bool:
    """Report whether subagent spawns are routed for one session.

    Read per spawn (not at launch) so the setting can be flipped at any
    point in a session and take effect on the next spawn. ``"on"`` /
    ``"off"`` win outright; unset inherits the session's main routing
    state, so a session started on Intelligent Routing routes its
    subagents and one started on a manual model does not.

    :param subagent_routing_override: The session's
        ``subagent_routing_override`` — ``"on"``, ``"off"``, or ``None``
        to inherit.
    :param cost_control_mode: The session's ``cost_control_mode_override``.
    :param parent_cost_control_mode: The parent session's value for a
        spawned child. ``None`` for a top-level session.
    :returns: ``True`` when subagent spawns should be routed.
    """
    if subagent_routing_override == "on":
        return True
    if subagent_routing_override == "off":
        return False
    return routing_enabled(
        cost_control_mode,
        parent_cost_control_mode=parent_cost_control_mode,
    )


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

    ``model`` is always a servable catalog id (e.g.
    ``"databricks-claude-sonnet-5"``) — never a harness's own tool
    vocabulary. Translating it is the harness hook's job: Claude Code's
    Agent/Task ``model`` parameter, for instance, only accepts the tier
    aliases ``sonnet``/``opus``/``haiku``/``fable`` and rejects a catalog id
    outright, so its hook inverse-maps before rewriting ``updatedInput``.

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


def decision_record(
    req: SubagentRouteRequest,
    decision: SubagentRouteDecision,
) -> RoutingDecisionData:
    """Build the transcript payload for *decision*.

    :param req: The spawn that was routed.
    :param decision: The verdict returned to the hook.
    :returns: Item data ready for :func:`persist_subagent_decision`.
    """
    from omnigent.entities.conversation import RoutingDecisionData

    model = decision.model or req.parent_model or "unrouted"
    return RoutingDecisionData(
        model=model,
        applied=decision.action in ("rewrite", "redirect"),
        rationale=decision.rationale,
        decision_id=decision.decision_id,
        harness=decision.harness or req.harness,
        raw_model=decision.raw_model,
        agent=req.task_name or None,
        scope=_SCOPE,
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


def candidate_models(
    harness: str,
    *,
    cross_harness: bool = False,
    catalog: Mapping[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Build the harness → models map offered to the router.

    In-family by default: a Claude Code session must not be told to move a
    subagent to Codex, so only the requesting harness's models are offered
    and the router's scenario stays that one family. Auto-harness sessions
    already let the router pick the family, so they also get the
    cross-harness counterpart — the only case where a ``redirect`` verdict
    can fire.

    :param harness: Requesting harness id, e.g. ``"codex-native"``.
    :param cross_harness: ``True`` to also offer the counterpart family.
    :param catalog: Live per-session model catalog keyed by worker name
        (:func:`omnigent.server.smart_routing.fetch_runner_models`). Preferred
        over the static table so a model generation the workspace serves today
        isn't treated as unservable; the static table fills any harness the
        catalog has no row for.
    :returns: Harness → model ids, cheapest first, empty entries dropped.
    """
    from omnigent.server.smart_routing import catalog_models_for_harness, infer_models

    offered = (harness, _COUNTERPART_HARNESS.get(harness)) if cross_harness else (harness,)
    result: dict[str, list[str]] = {}
    for candidate in offered:
        if candidate is None or candidate in result:
            continue
        models = catalog_models_for_harness(
            catalog, candidate, allow_self=candidate == harness
        ) or infer_models(candidate)
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
    catalog: Mapping[str, list[str]] | None = None,
    cross_harness: bool = False,
    persist: Callable[[RoutingDecisionData], Awaitable[None]] | None = None,
    now: float | None = None,
) -> SubagentRouteDecision:
    """Decide what happens to one native-subagent spawn.

    :param session_id: Parent session/conversation identifier.
    :param req: The spawn awaiting a verdict.
    :param caps: ``RuntimeCaps``-shaped object. ``None`` reads the
        process-global caps.
    :param available_models: Candidate harness → models map. ``None``
        derives it from the requesting harness.
    :param catalog: Live per-session model catalog, preferred over the
        static table when deriving candidates. Ignored when
        *available_models* is given.
    :param cross_harness: ``True`` when the session may move a subagent to
        the counterpart harness family (auto-harness sessions only).
        Ignored when *available_models* is given.
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

    decision = await _decide(
        session_id, req, caps, settings, available_models, catalog, cross_harness, clock
    )
    emit_routing_event(
        ROUTING_EVENT_DECISION,
        {
            "routing.scope": _SCOPE,
            "routing.session_id": session_id,
            "routing.harness": decision.harness or req.harness,
            "routing.model": decision.model,
            "routing.raw_model": decision.raw_model,
            "routing.action": decision.action,
            "routing.decision_id": decision.decision_id,
            "routing.task_name": req.task_name or None,
        },
    )
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
    catalog: Mapping[str, list[str]] | None,
    cross_harness: bool,
    clock: float,
) -> SubagentRouteDecision:
    if req.fork:
        return SubagentRouteDecision(
            action="allow",
            rationale="Fork keeps the parent's model; forks are not routed",
            model=req.parent_model,
        )

    if not req.prompt and not req.task_name:
        # Codex encrypts the spawn message, so an unnamed codex subagent
        # carries nothing to score. Calling the router anyway earns an
        # "HTTP 400: task.prompt is required" that reads on the decision chip
        # as an outage; this is a deliberate verdict, so say so. Not cached:
        # there is no router result to reuse. The spawn runs on the parent's
        # thread model, which the SubagentStart audit confirms, so naming it
        # here also keeps the audit reconciliation from flagging the spawn.
        return SubagentRouteDecision(
            action="allow",
            rationale=(
                "No routable signal (encrypted prompt, no task name); "
                "subagent inherits the session model"
            ),
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
        available_models
        if available_models is not None
        else candidate_models(req.harness, cross_harness=cross_harness, catalog=catalog)
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
    record: RoutingDecisionData,
) -> None:
    """Persist and publish *record* as a ``routing_decision`` item.

    :param session_id: Session/conversation identifier.
    :param conversation_store: Store exposing ``append``.
    :param record: Decision payload.
    """
    from omnigent.entities.conversation import NewConversationItem
    from omnigent.runtime import session_stream

    item_data = record.model_dump()
    item = NewConversationItem(
        type="routing_decision",
        response_id=f"routing_{uuid.uuid4().hex}",
        data=record,
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
) -> Callable[[RoutingDecisionData], Awaitable[None]]:
    """Bind :func:`persist_subagent_decision` to a session and store.

    :param session_id: Session/conversation identifier.
    :param conversation_store: Store exposing ``append``.
    :returns: Coroutine function accepting a record.
    """

    async def _persist(record: RoutingDecisionData) -> None:
        await persist_subagent_decision(session_id, conversation_store, record)

    return _persist


# ── Runner-side loopback relay ─────────────────────────────────────────────


def write_advertisement(
    bridge_dir: Path,
    *,
    url: str,
    token: str,
    session_id: str | None = None,
) -> Path:
    """Advertise the endpoint to hook scripts.

    :param bridge_dir: Session bridge directory.
    :param url: Endpoint base URL, e.g. ``"http://127.0.0.1:53421"``.
    :param token: Bearer token hook scripts must present.
    :param session_id: Session the endpoint serves. Included so a hook
        with no session env var of its own still knows which session to
        route for.
    :returns: Path of the written ``subagent_router.json``.
    """
    bridge_dir.mkdir(parents=True, exist_ok=True)
    path = bridge_dir / ADVERTISEMENT_FILE
    tmp = path.with_name(f"{path.name}.tmp")
    payload: dict[str, Any] = {
        "url": url,
        "token": token,
        "pid": os.getpid(),
        "updated_at": time.time(),
    }
    if session_id is not None:
        payload["session_id"] = session_id
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
    write_advertisement(bridge_dir, url=url, token=token, session_id=session_id)
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


# ── Per-session router lifecycle (runner side) ─────────────────────────────

_session_routers: dict[str, SubagentRouter] = {}
# Models the relay actually handed back, per session — the ledger the
# codex ``SubagentStart`` audit is reconciled against.
_relayed: dict[str, list[dict[str, Any]]] = {}
_lifecycle_lock = threading.Lock()


def relayed_decisions(session_id: str) -> tuple[dict[str, Any], ...]:
    """Return the verdicts this runner relayed for *session_id*.

    :param session_id: Session/conversation identifier.
    :returns: Records ``{decision_id, action, model, harness, task_name}``,
        oldest first.
    """
    with _lifecycle_lock:
        return tuple(_relayed.get(session_id, ()))


def routed_models(session_id: str) -> frozenset[str]:
    """Return every model the router approved for *session_id*.

    :param session_id: Session/conversation identifier.
    :returns: Approved model ids; empty when nothing was routed.
    """
    return frozenset(
        str(record["model"])
        for record in relayed_decisions(session_id)
        if record.get("model") and record.get("action") in ("rewrite", "allow")
    )


def _recording_resolver(resolver: Resolver) -> Resolver:
    """Wrap *resolver* so each verdict lands in the reconciliation ledger."""

    async def _resolve(session_id: str, req: SubagentRouteRequest) -> SubagentRouteDecision:
        decision = await resolver(session_id, req)
        with _lifecycle_lock:
            _relayed.setdefault(session_id, []).append(
                {
                    "decision_id": decision.decision_id,
                    "action": decision.action,
                    "model": decision.model,
                    "harness": decision.harness or req.harness,
                    "task_name": req.task_name,
                }
            )
        return decision

    return _resolve


def ensure_session_router(
    session_id: str,
    *,
    bridge_dir: Path,
    server_client: httpx.AsyncClient,
    loop: asyncio.AbstractEventLoop | None = None,
    fail_mode: Literal["open", "closed"] = DEFAULT_FAIL_MODE,
) -> SubagentRouter:
    """Start (once) the loopback router serving *session_id*.

    Idempotent: a second call for a session already served returns the
    running router after re-asserting its advertisement, so a resumed or
    re-launched terminal finds the file the hook scripts read.

    :param session_id: Session/conversation identifier.
    :param bridge_dir: Directory the advertisement is written into — the
        same directory the harness's hooks are pointed at.
    :param server_client: Runner→server client used to relay verdicts to
        the process holding ``RuntimeCaps.routing_client``.
    :param loop: Event loop owning the relay. ``None`` uses the running
        loop.
    :param fail_mode: Behavior when the server hop fails.
    :returns: The running router handle.
    """
    with _lifecycle_lock:
        existing = _session_routers.get(session_id)
    if existing is not None:
        write_advertisement(
            bridge_dir,
            url=existing.url,
            token=existing.token,
            session_id=session_id,
        )
        return existing
    router = start_subagent_router(
        bridge_dir=bridge_dir,
        session_id=session_id,
        resolver=_recording_resolver(
            make_server_relay_resolver(server_client, fail_mode=fail_mode)
        ),
        loop=loop if loop is not None else asyncio.get_running_loop(),
        fail_mode=fail_mode,
    )
    with _lifecycle_lock:
        _session_routers[session_id] = router
    _logger.info(
        "subagent router started: session=%s url=%s dir=%s", session_id, router.url, bridge_dir
    )
    return router


def shutdown_session_router(session_id: str) -> None:
    """Stop the router serving *session_id* and forget its state.

    :param session_id: Session/conversation identifier.
    """
    with _lifecycle_lock:
        router = _session_routers.pop(session_id, None)
        _relayed.pop(session_id, None)
    if router is None:
        return
    with contextlib.suppress(Exception):
        router.close()
    clear_cache(session_id)


def router_dir_for_session(session_id: str) -> Path:
    """Return the advertisement directory for a session with no bridge dir.

    The SDK harnesses (claude-agent-sdk, codex app-server) have no bridge
    directory of their own, so the router gets a private owner-only one
    beside the native bridges.

    :param session_id: Session/conversation identifier.
    :returns: Created directory, mode ``0o700``.
    """
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    root = Path(tempfile.gettempdir()) / f"omnigent-{stable_user_id()}" / "subagent-router"
    path = root / digest
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def session_router_env(session_id: str) -> dict[str, str]:
    """Return the router env for a session, or ``{}`` when it has no router.

    Read at harness-spawn time so a session whose router was started at
    session init hands every harness process the same rendezvous.

    :param session_id: Session/conversation identifier.
    :returns: Env-var overrides, empty when the session has no router.
    """
    with _lifecycle_lock:
        router = _session_routers.get(session_id)
    if router is None:
        return {}
    return router_env(session_id, router.bridge_dir)


def router_env(session_id: str, router_dir: Path) -> dict[str, str]:
    """Build the env that points a harness process at the router.

    Both harness families read the advertisement out of a directory named
    in the environment: the claude-agent-sdk executor registers its
    in-process hook from it, and the codex executor uses it as the switch
    that turns generated routing ``hooks.json`` on.

    :param session_id: Session/conversation identifier.
    :param router_dir: Directory holding ``subagent_router.json``.
    :returns: Env-var overrides for the harness process.
    """
    from omnigent.inner.codex_executor import (
        CODEX_ROUTER_DIR_ENV_VAR,
        CODEX_ROUTER_SESSION_ID_ENV_VAR,
    )
    from omnigent.inner.hook_scripts.subagent_router import (
        ROUTER_DIR_ENV_VAR,
        SESSION_ID_ENV_VAR,
    )

    return {
        ROUTER_DIR_ENV_VAR: str(router_dir),
        SESSION_ID_ENV_VAR: session_id,
        CODEX_ROUTER_DIR_ENV_VAR: str(router_dir),
        CODEX_ROUTER_SESSION_ID_ENV_VAR: session_id,
    }
