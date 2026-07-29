"""Server-side intelligent model routing.

Infers available models from the session's harness type and delegates
the routing decision to the :class:`RoutingClient` on
:attr:`RuntimeCaps.routing_client`.  The default implementation
(:class:`LLMRoutingClient`) calls the server-level LLM with a prompt
that describes each model's capabilities directly — no tier abstraction.
Managed deployments can swap in a different implementation via
``RuntimeCaps``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import httpx  # used in type annotations only; runtime import is lazy in fetch_runner_models

_logger = logging.getLogger(__name__)

# Custom-method path (Google API convention) appended to the external
# router's base URL, e.g. ``<base_url>/routes:select``.
ROUTES_SELECT_PATH = "routes:select"

# ── Model lists per harness family ──────────────────────────────────────────
#
# Ordered cheapest → most powerful within each family.

MODEL_LISTS: dict[str, list[str]] = {
    "claude": [
        "databricks-claude-haiku-4-5",
        "databricks-claude-sonnet-4-6",
        "databricks-claude-opus-4-8",
    ],
    "gpt": [
        "databricks-gpt-5-4-nano",
        "databricks-gpt-5-4-mini",
        "databricks-gpt-5-4",
        "databricks-gpt-5-5",
    ],
    # pi is multi-model: Claude and GPT both available.
    "pi": [
        "databricks-gpt-5-4-nano",
        "databricks-claude-haiku-4-5",
        "databricks-gpt-5-4-mini",
        "databricks-claude-sonnet-4-6",
        "databricks-gpt-5-4",
        "databricks-claude-opus-4-8",
        "databricks-gpt-5-5",
    ],
}

_HARNESS_FAMILY: dict[str, str] = {
    "claude-sdk": "claude",
    "claude_sdk": "claude",
    "claude-native": "claude",
    "pi": "pi",
    "codex": "gpt",
    "codex-native": "gpt",
    "openai-agents": "gpt",
    "openai-agents-sdk": "gpt",
    "agents_sdk": "gpt",
}


def infer_models(harness: str | None) -> list[str] | None:
    """Return available models for *harness*, or ``None`` if unroutable."""
    if harness is None:
        return None
    family = _HARNESS_FAMILY.get(harness)
    if family is None:
        return None
    return MODEL_LISTS.get(family)


# ── RoutingClient protocol ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RoutingResult:
    """The routing client's recommendation.

    :param model: Model id to use, e.g. ``"databricks-claude-opus-4-8"``.
    :param rationale: One-sentence explanation from the judge.
    :param harness: The harness the judge selected, e.g. ``"claude-sdk"``.
        ``None`` when the routing client does not distinguish harnesses (e.g.
        single-harness calls or custom implementations that omit it).
    :param raw_model: The router's pick in its own vocabulary, before it was
        resolved to a servable catalog id (they differ when the router names an
        arm this workspace has no endpoint for). ``None`` when the two are the
        same or the client does not distinguish them.
    """

    model: str
    rationale: str
    harness: str | None = None
    raw_model: str | None = None


class RoutingClient(Protocol):
    """Protocol for pluggable model routing implementations."""

    async def route(
        self,
        message: str,
        available_models: dict[str, list[str]],
    ) -> RoutingResult | None:
        """Pick the best model for a session's initial message.

        :param message: The user's first message text.
        :param available_models: Mapping of harness → model ids, each list
            ordered cheapest → most powerful.  A single-harness call passes
            a one-entry dict; multi-agent fan-out passes one entry per
            harness.  Implementations that only need the flat model list can
            call :func:`_flatten_models` to get a deduped ordered sequence.
        :returns: A :class:`RoutingResult`, or ``None`` to skip routing.
        """
        ...


# ── Helpers ────────────────────────────────────────────────────────────────


async def fetch_runner_models(
    session_id: str,
    runner_client: httpx.AsyncClient,
) -> dict[str, list[str]] | None:
    """Fetch live model availability from the runner's ``/v1/sessions/{id}/models`` endpoint.

    Converts the ``sys_list_models``-shaped catalog into the harness →
    model-id-list format expected by :class:`RoutingClient`.  Falls back
    to ``None`` on any HTTP/parse failure so callers can use the static
    :func:`infer_models` table instead.

    :param session_id: Session/conversation identifier.
    :param runner_client: Async HTTP client pointed at the runner.
    :returns: ``{harness: [model_id, ...]}`` ordered cheapest → most
        powerful, or ``None`` when the endpoint is unavailable or the
        response cannot be parsed.
    """
    import httpx as _httpx

    try:
        resp = await runner_client.get(f"/v1/sessions/{session_id}/models", timeout=5.0)
        resp.raise_for_status()
        payload = resp.json()
    except (_httpx.HTTPError, ValueError, KeyError):
        _logger.debug(
            "fetch_runner_models: runner request failed for session=%s", session_id, exc_info=True
        )
        return None

    workers: dict[str, Any] = payload.get("workers", {})
    if not workers:
        return None

    result: dict[str, list[str]] = {}
    for worker_name, row in workers.items():
        if not isinstance(row, dict):
            continue
        models_raw = row.get("models", [])
        if not isinstance(models_raw, list):
            continue
        ids = [m["id"] for m in models_raw if isinstance(m, dict) and isinstance(m.get("id"), str)]
        if ids:
            result[worker_name] = ids
    return result or None


def _flatten_models(available_models: dict[str, list[str]]) -> list[str]:
    """Return a deduped, ordered flat model list from a harness → models map.

    Iterates harness entries in insertion order; within each harness the
    model list is already cheapest → most powerful.  Duplicates (a model
    supported by multiple harnesses) are dropped on second occurrence so
    the first-harness ordering is preserved.
    """
    seen: set[str] = set()
    result: list[str] = []
    for models in available_models.values():
        for m in models:
            if m not in seen:
                seen.add(m)
                result.append(m)
    return result


# ── Default LLM-based implementation ───────────────────────────────────────

_JUDGE_SYSTEM_TEMPLATE = """\
You are a model router for a coding assistant. Given the user's message,
pick the harness and model best suited for the task.

Available harnesses and their models:
{harness_menu}

Harness descriptions:
- claude-sdk / claude-native: Claude Code harness; best for multi-file
  refactors, test writing, and deep reasoning chains.
- codex / codex-native: Codex harness; best for narrow, well-scoped
  code changes.
- pi: Multi-model headless harness; can run both Claude and GPT models;
  best for read-only exploration, review, and cross-vendor verification.

Model tiers (cheapest → most capable within each family):
- Claude: haiku < sonnet < opus
- GPT: *-nano < *-mini < base (e.g. gpt-5-4-nano < gpt-5-4-mini < gpt-5-4 < gpt-5-5)

Trade-off guidance — classify the task and pick the corresponding model:

  SIMPLE   → cheapest available model (haiku for Claude; nano for GPT)
             Examples: greetings, quick lookups, one-line fixes, trivial Q&A.

  MODERATE → mid-range model (sonnet for Claude; mini for GPT)
             Examples: single-file edits, debugging a known issue, brief explanations.

  COMPLEX  → most capable model (opus for Claude; newest base GPT)
             Examples: multi-file refactors, architecture decisions, security analysis,
             long reasoning chains, tasks requiring high accuracy or broad context.

The rationale field must follow this exact pattern so the explanation is consistent
with the model chosen:
  "This is a [SIMPLE/MODERATE/COMPLEX] task ([brief reason]); \
selected [cheapest/mid-range/most capable] model [model-id]."

Return **strict JSON only**:
{{"harness": "<harness-id>", "model": "<model-id>", "rationale": "<sentence>"}}
"""


def _build_rubric(available_models: dict[str, list[str]]) -> str:
    """Format the judge prompt with the harness → models structure."""
    sections: list[str] = []
    for harness, models in available_models.items():
        model_lines = "\n".join(f"    - {m}" for m in models)
        sections.append(f"  harness: {harness}\n{model_lines}")
    return _JUDGE_SYSTEM_TEMPLATE.format(harness_menu="\n".join(sections))


_VERDICT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "harness": {"type": "string"},
        "model": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["harness", "model", "rationale"],
    "additionalProperties": False,
}


class LLMRoutingClient:
    """Default routing client using the server-level PolicyLLMClient."""

    def __init__(self, llm_client: Any) -> None:  # type: ignore[explicit-any]
        self._llm = llm_client

    async def route(
        self,
        message: str,
        available_models: dict[str, list[str]],
    ) -> RoutingResult | None:
        flat = _flatten_models(available_models)
        rubric = _build_rubric(available_models)
        _logger.info("LLMRoutingClient: available_models=%s", dict(available_models))
        try:
            response = await self._llm.create(
                instructions=rubric,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": message[:4000]}],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "routing_verdict",
                        "strict": True,
                        "schema": _VERDICT_SCHEMA,
                    }
                },
            )
            text = response.output[0].content[0].text
            _logger.info("LLMRoutingClient: raw response: %s", text[:500])
            verdict = json.loads(text)
        except Exception:  # noqa: BLE001  # fail-open
            _logger.warning("LLMRoutingClient: judge call failed", exc_info=True)
            return None

        model = verdict.get("model")
        rationale = verdict.get("rationale", "")
        if not model or not isinstance(model, str):
            return None

        # Clamp hallucinated models to the cheapest available.
        if model not in flat:
            if flat:
                _logger.info(
                    "LLMRoutingClient: clamping unknown model %r to %s",
                    model,
                    flat[0],
                )
                model = flat[0]
            else:
                return None

        # Resolve the harness: use the judge's pick only when it is both a
        # known harness key AND actually contains the chosen model.  If
        # either check fails, fall back to the first harness that owns the
        # (possibly clamped) model.
        chosen_harness = verdict.get("harness")
        if (
            not isinstance(chosen_harness, str)
            or chosen_harness not in available_models
            or model not in available_models[chosen_harness]
        ):
            if isinstance(chosen_harness, str) and chosen_harness in available_models:
                _logger.info(
                    "LLMRoutingClient: harness %r does not contain model %r; re-resolving",
                    chosen_harness,
                    model,
                )
            chosen_harness = next(
                (h for h, models in available_models.items() if model in models),
                None,
            )

        return RoutingResult(model=model, rationale=str(rationale), harness=chosen_harness)


def _bearer_auth(token: str) -> Any:  # type: ignore[explicit-any]  # returns httpx.Auth
    """Build a static ``Authorization: Bearer <token>`` httpx auth.

    :param token: The bearer token, e.g. a Databricks workspace token.
    :returns: An ``httpx.Auth`` that adds the bearer header to each request.
    """
    import httpx

    class _BearerAuth(httpx.Auth):
        def auth_flow(self, request: httpx.Request):  # type: ignore[no-untyped-def]
            request.headers["Authorization"] = f"Bearer {token}"
            yield request

    return _BearerAuth()


def _router_error_detail(body: str) -> str:
    """Extract a clean, short reason from a router error response body.

    The gateway wraps the reason in nested JSON (e.g.
    ``{"error_code": 401, "message": "Credential was not sent ..."}`` or a
    doubly-encoded ``message`` holding another JSON object). Pull out the
    innermost human message when possible; otherwise return a trimmed body.

    :param body: The raw response text.
    :returns: A short reason string suitable for a UI card.
    """
    text = (body or "").strip()
    for _ in range(4):  # unwrap up to a few nested message/error layers
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            break
        if not isinstance(parsed, dict):
            break
        candidate = parsed.get("message")
        if candidate is None:
            candidate = parsed.get("error")
        # A dict ``error`` (e.g. {"error": {"message": ...}}) — descend into it.
        if isinstance(candidate, dict):
            candidate = candidate.get("message")
        if isinstance(candidate, str) and candidate:
            text = candidate.strip()
            continue
        break
    return text[:300]


# ── Route-options seam ──────────────────────────────────────────────────────
#
# Every assumption about the router's wire contract lives in this section:
# which model arms a router version expects to be offered, how a picked arm
# maps back to a servable catalog id, and which harness can actually run it.
# Callers (route_session_harness, route_turn, the subagent endpoint) never see
# router vocabulary. When the gateway starts accepting dynamic model+harness
# sets and enforcing harness constraints itself, only this section changes.

DEFAULT_ROUTER_NAME = "task_v1"

# task_v1 infers a scenario from WHICH model families appear in route_options
# and then requires that scenario's full arm menu (extra models are tolerated
# and ignored); a partial menu is rejected. Menu ids need not be servable in
# the workspace — the router requires them anyway and may select them.
_TASK_V1_CLAUDE_ARMS: tuple[str, ...] = ("claude-opus-4-8", "claude-sonnet-5")
_TASK_V1_CODEX_ARMS: tuple[str, ...] = ("glm-5-2", "gpt-5-6-sol", "gpt-5-6-luna")

# Keyed by router version: routers are frozen per version, so pinning
# ``routing.router_name`` pins the menu that goes with it. A router name with
# no entry here (e.g. the older ``task_v0``) gets no injected arms — the
# candidate set stays exactly the caller's catalog.
TASK_V1_MENUS: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        DEFAULT_ROUTER_NAME: MappingProxyType(
            {
                "cc": _TASK_V1_CLAUDE_ARMS,
                "codex": _TASK_V1_CODEX_ARMS,
                "both": _TASK_V1_CLAUDE_ARMS + _TASK_V1_CODEX_ARMS,
            }
        )
    }
)


@dataclass(frozen=True)
class RoutingSettings:
    """Deployment routing knobs, parsed from the server ``routing:`` block.

    Hung on :attr:`~omnigent.runtime.caps.RuntimeCaps.routing_settings` so
    every consumer reads one value object instead of re-parsing config.

    :param router_name: Router strategy to invoke, e.g. ``"task_v1"``.
    :param selection_model: Model the router should use for its own
        extraction call, sent as ``route_selector.config.model``. ``None``
        leaves the router's frozen default in place.
    :param scenario_menus: Router version → scenario → required arm menu.
    :param subagent_fail_mode: What a native-subagent spawn does when the
        router is unreachable — ``"open"`` allows it unchanged, ``"closed"``
        denies it so the determinism guarantee can't silently lapse.
    :param subagent_cache_ttl_s: How long a per-(session, task) subagent
        decision is reused, keeping the blocking spawn path fast.
    """

    router_name: str = DEFAULT_ROUTER_NAME
    selection_model: str | None = None
    scenario_menus: Mapping[str, Mapping[str, tuple[str, ...]]] = TASK_V1_MENUS
    subagent_fail_mode: Literal["open", "closed"] = "open"
    subagent_cache_ttl_s: float = 300.0


@dataclass(frozen=True)
class RouteOptionSpec:
    """One candidate destination to offer the router, in router vocabulary."""

    model: str
    harness: str | None = None


@dataclass(frozen=True)
class RoutePick:
    """The router's raw selection, exactly as it came off the wire."""

    model: str
    harness: str | None = None


@dataclass(frozen=True)
class ResolvedRoute:
    """A router pick translated into something this deployment can run.

    :param model: Servable catalog id to apply.
    :param harness: Harness derived from the picked arm's family.
    :param raw_model: The router's pick verbatim, kept for the decision
        payload so the UI can show what the router actually said.
    """

    model: str
    harness: str | None
    raw_model: str


class RouteOptionSource(Protocol):
    """Translates between this deployment's catalog and router vocabulary."""

    def build_route_options(
        self,
        harnesses: Sequence[str],
        catalog: dict[str, list[str]],
    ) -> list[RouteOptionSpec]:
        """Return the candidate set to offer for *harnesses*."""
        ...

    def resolve_selection(
        self,
        pick: RoutePick,
        harnesses: Sequence[str],
        catalog: dict[str, list[str]],
    ) -> ResolvedRoute | None:
        """Translate the router's *pick* into a runnable route, or ``None``."""
        ...


# Catalog prefixes stripped before comparing a model id against router
# vocabulary or the per-harness exclusion table, so the same rule matches both
# ``databricks-gpt-5-5`` and the bare ``gpt-5-5`` the router speaks.
_BARE_ID_PREFIXES: tuple[str, ...] = ("databricks-", "system.ai.")

# Router arms whose family isn't obvious from the id itself.
_ARM_FAMILY: dict[str, str] = {"glm-5-2": "gpt"}

# Harness family that serves several vendors and so never wins family matching
# outright — only used when no single-vendor harness fits.
_MULTI_MODEL_FAMILY = "pi"

# Per-harness (harness, model) incompatibilities corrected AFTER the router
# verdict. These models are NOT pruned from the candidate set: the router
# requires its full scenario menu and 400s on a partial one, so the full set
# must be offered and an incompatible pick moved to a harness that can run it.
#
# The pi harness reaches Databricks two incompatible ways for these families:
#   - Claude models ride pi's Anthropic Messages gateway, whose request path
#     adds an ``eager_input_streaming`` field the serving endpoint rejects with
#     a 400 when tools are present.
#   - The gpt-5.5 / gpt-5.6 reasoning models ride pi's openai-completions path
#     (``/chat/completions``); Databricks applies a default ``reasoning_effort``
#     there and rejects tool calls with "Function tools with reasoning_effort
#     are not supported for gpt-5.5 ... use /v1/responses or set reasoning_effort
#     to 'none'." pi's provider can't send that override, so tool turns 400.
# claude-sdk serves Claude and codex serves gpt-5.5+ (Responses API); the
# gpt-5.4 family works on pi and is left alone.
_HARNESS_EXCLUDED_MODELS: dict[str, tuple[str, ...]] = {
    "pi": (
        "databricks-claude-haiku-4-5",
        "databricks-gpt-5-5",
        "databricks-gpt-5-5-pro",
        "databricks-gpt-5-6-luna",
        "databricks-gpt-5-6-terra",
        "databricks-gpt-5-6-sol",
    ),
}


def _bare_id(model: str) -> str:
    """Strip a known catalog prefix so ids compare across vocabularies."""
    for prefix in _BARE_ID_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _model_family(model: str) -> str:
    """Tag *model* with the harness family that can serve it."""
    bare = _bare_id(model).lower()
    known = _ARM_FAMILY.get(bare)
    if known is not None:
        return known
    if "claude" in bare:
        return "claude"
    if "gpt" in bare or "codex" in bare:
        return "gpt"
    return "other"


def _shared_prefix_len(left: str, right: str) -> int:
    """Length of the longest common prefix of two model ids."""
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def _redirect_incompatible_pick(harness: str | None, model: str) -> str | None:
    """Redirect a verdict off a harness that can't serve *model*.

    A Claude model on pi moves to ``claude-sdk``; a gpt-5.5/5.6 reasoning
    model on pi moves to ``codex`` (Responses API). See
    :data:`_HARNESS_EXCLUDED_MODELS` for why.

    :param harness: The chosen harness id (may be ``None``).
    :param model: The chosen model id, in either vocabulary.
    :returns: A replacement harness id, or *harness* when already compatible.
    """
    if harness is None:
        return None
    excluded = {_bare_id(m) for m in _HARNESS_EXCLUDED_MODELS.get(harness, ())}
    if _bare_id(model) not in excluded:
        return harness
    family = _model_family(model)
    if family == "claude":
        return "claude-sdk"
    if family == "gpt":
        return "codex"
    # Unknown family on an excluding harness — leave as-is rather than guess.
    return harness


class TaskV1RouteOptionSource:
    """Route-options source for the ``task_v1`` generation of the router.

    Offers the scenario menu the router demands (injecting arms the workspace
    has no endpoint for), ignores the echoed harness tag — passthrough and
    untrusted — and maps a pick back to a servable catalog id.
    """

    def __init__(
        self,
        *,
        router_name: str = DEFAULT_ROUTER_NAME,
        scenario_menus: Mapping[str, Mapping[str, tuple[str, ...]]] = TASK_V1_MENUS,
        model_prefixes: Sequence[str] = (),
    ) -> None:
        """
        :param router_name: Router strategy name; selects the menu set.
        :param scenario_menus: Router version → scenario → required arms.
        :param model_prefixes: Catalog prefixes to strip on the way out and
            restore on the way back, e.g. ``["databricks-", "system.ai."]``.
        """
        self._router_name = router_name
        self._menus = scenario_menus
        self._model_prefixes = tuple(model_prefixes)

    def to_router_id(self, model: str) -> str:
        """Strip the first matching configured prefix from a catalog id."""
        for prefix in self._model_prefixes:
            if prefix and model.startswith(prefix):
                return model[len(prefix) :]
        return model

    def build_route_options(
        self,
        harnesses: Sequence[str],
        catalog: dict[str, list[str]],
    ) -> list[RouteOptionSpec]:
        """Offer the catalog plus whatever arms the router's menu requires.

        :param harnesses: Harnesses the decision may land on; picks the
            scenario (codex-only, claude-only, or mixed).
        :param catalog: Harness → servable model ids.
        :returns: Candidate options in router vocabulary, catalog first.
        """
        options: list[RouteOptionSpec] = []
        offered: set[str] = set()
        for harness, models in catalog.items():
            for model in models:
                router_id = self.to_router_id(model)
                options.append(RouteOptionSpec(model=router_id, harness=harness))
                offered.add(router_id)
        for arm in self.menu(harnesses or list(catalog)):
            if arm in offered:
                continue
            options.append(
                RouteOptionSpec(model=arm, harness=self._tag_harness(arm, harnesses, catalog))
            )
            offered.add(arm)
        return options

    def resolve_selection(
        self,
        pick: RoutePick,
        harnesses: Sequence[str],
        catalog: dict[str, list[str]],
    ) -> ResolvedRoute | None:
        """Translate *pick* into a servable (harness, model) pair.

        :param pick: The router's selection as received; its ``harness`` is
            ignored because the router echoes the tag verbatim without ever
            reading it.
        :param harnesses: Harnesses the decision may land on.
        :param catalog: Harness → servable model ids.
        :returns: A :class:`ResolvedRoute`, or ``None`` when the pick was
            never offered or nothing servable is close enough.
        """
        raw = (pick.model or "").strip()
        if not raw:
            return None
        harness = self._derive_harness(raw, harnesses, catalog)
        local = self._local_id(raw, harness, catalog)
        if local is None:
            # Only an arm we injected may be remapped; anything else is a
            # model the router was never offered.
            if raw not in self.menu(harnesses or list(catalog)):
                return None
            local = self._nearest_servable(raw, harness, catalog)
        if local is None:
            return None
        return ResolvedRoute(model=local, harness=harness, raw_model=raw)

    def menu(self, harnesses: Iterable[str]) -> tuple[str, ...]:
        """Return the arm menu this router requires for *harnesses*."""
        menus = self._menus.get(self._router_name)
        if not menus:
            return ()
        return tuple(menus.get(self._scenario(harnesses), ()))

    def _scenario(self, harnesses: Iterable[str]) -> str:
        """Map a harness set to a router scenario key."""
        families = {_HARNESS_FAMILY.get(h) for h in harnesses}
        if families == {"claude"}:
            return "cc"
        if families == {"gpt"}:
            return "codex"
        return "both"

    def _tag_harness(
        self,
        arm: str,
        harnesses: Sequence[str],
        catalog: dict[str, list[str]],
    ) -> str | None:
        """Pick a plausible harness tag for an injected arm.

        The tag is decoration — no router reads it — so a family match is
        enough, falling back to the first harness on offer.
        """
        order = list(catalog) or list(harnesses)
        family = _model_family(arm)
        match = next((h for h in order if _HARNESS_FAMILY.get(h) == family), None)
        if match is not None:
            return match
        return order[0] if order else None

    def _derive_harness(
        self,
        router_id: str,
        harnesses: Sequence[str],
        catalog: dict[str, list[str]],
    ) -> str | None:
        """Choose the harness that can actually run the picked arm."""
        order = list(catalog) or list(harnesses)
        serving = [
            h for h in order if any(self.to_router_id(m) == router_id for m in catalog.get(h, []))
        ]
        candidates = serving or order
        family = _model_family(router_id)
        chosen = next((h for h in candidates if _HARNESS_FAMILY.get(h) == family), None)
        if chosen is None:
            chosen = next(
                (h for h in candidates if _HARNESS_FAMILY.get(h) == _MULTI_MODEL_FAMILY), None
            )
        if chosen is None:
            chosen = candidates[0] if candidates else None
        return _redirect_incompatible_pick(chosen, router_id)

    def _local_id(
        self,
        router_id: str,
        harness: str | None,
        catalog: dict[str, list[str]],
    ) -> str | None:
        """Restore the exact catalog id behind a router id, if it is servable."""
        own = catalog.get(harness or "", [])
        exact = next((m for m in own if self.to_router_id(m) == router_id), None)
        if exact is not None:
            return exact
        for models in catalog.values():
            for model in models:
                if self.to_router_id(model) == router_id:
                    return model
        return None

    def _nearest_servable(
        self,
        arm: str,
        harness: str | None,
        catalog: dict[str, list[str]],
    ) -> str | None:
        """Map an arm with no endpoint here to the closest servable id.

        The router requires (and selects) arms this workspace may not serve,
        e.g. ``gpt-5-6-sol``. Prefer the same family, then the id sharing the
        longest prefix with the arm, breaking ties toward the more capable
        model (catalog lists run cheapest → most powerful).
        """
        pool = list(catalog.get(harness or "", []))
        if not pool:
            pool = [m for models in catalog.values() for m in models]
        same_family = [m for m in pool if _model_family(m) == _model_family(arm)]
        pool = same_family or pool
        best: str | None = None
        best_score = -1
        for model in pool:
            score = _shared_prefix_len(self.to_router_id(model), arm)
            if score >= best_score:
                best, best_score = model, score
        return best


def _routing_settings() -> RoutingSettings:
    """Read :class:`RoutingSettings` off the runtime caps, defaulted."""
    try:
        from omnigent.runtime._globals import _caps
    except ImportError:
        return RoutingSettings()
    settings = getattr(_caps, "routing_settings", None) if _caps is not None else None
    return settings if isinstance(settings, RoutingSettings) else RoutingSettings()


def route_option_source(
    settings: RoutingSettings | None = None,
    *,
    model_prefixes: Sequence[str] = (),
) -> RouteOptionSource:
    """Build the route-options source for this deployment.

    :param settings: Routing settings; read off the runtime caps when omitted.
    :param model_prefixes: Catalog prefixes the router does not expect.
    :returns: The configured :class:`RouteOptionSource`.
    """
    resolved = settings or _routing_settings()
    return TaskV1RouteOptionSource(
        router_name=resolved.router_name,
        scenario_menus=resolved.scenario_menus,
        model_prefixes=model_prefixes,
    )


class ExternalRoutingClient:
    """Routing client backed by an external ``routes:select`` service.

    Calls an external routing service (the Databricks AI-Gateway router,
    or any endpoint speaking the ``omnigent.api.routing.v1`` proto)
    instead of running a local judge. The candidate models come from
    ``available_models`` (the same live catalog the built-in judge sees),
    so no catalog plumbing changes. A failure or empty selection returns
    ``None`` so the turn proceeds on the agent's default model.
    """

    def __init__(
        self,
        *,
        base_url: str,
        router_name: str,
        auth: Any = None,  # type: ignore[explicit-any]  # httpx.Auth, imported lazily
        databricks_profile: str | None = None,
        model_prefixes: list[str] | None = None,
        request_timeout: float = 20.0,
        selection_model: str | None = None,
        scenario_menus: Mapping[str, Mapping[str, tuple[str, ...]]] = TASK_V1_MENUS,
    ) -> None:
        """
        :param base_url: Routing service base, e.g.
            ``"https://host/ai-gateway/routing/v1"``.
            ``/routes:select`` is appended.
        :param router_name: Router strategy name, e.g. ``"task_v0"``.
        :param auth: Optional static httpx auth (e.g. a bearer built from an
            explicit ``api_key``). ``None`` for an unauthenticated endpoint or
            when *databricks_profile* supplies per-call OAuth instead.
        :param databricks_profile: Optional Databricks CLI profile. When set
            (and *auth* is ``None``), a fresh bearer is minted per :meth:`route`
            call via the databricks-sdk ``Config`` — which refreshes OAuth
            tokens transparently, so a long-lived server never sends a stale
            token (the 401 an at-startup captured token hits after ~1h).
        :param model_prefixes: Optional prefixes this deployment's catalog
            attaches to model ids that the router does NOT expect. The
            first matching prefix is stripped from ids sent to the router
            and restored on its answer via the (harness, bare-id) -> local
            map. Examples: ``"databricks-"`` when serving-endpoint names are
            ``databricks-claude-opus-4-8`` but the router keys on
            ``claude-opus-4-8``; ``"system.ai."`` for Unity Catalog
            foundation-model ids like ``system.ai.claude-opus-4-8``. Empty
            or omitted (default) sends catalog ids verbatim — no provider
            assumed.
        :param request_timeout: Per-call timeout in seconds; routing
            runs once per turn so a slow router can't stall forever.
        :param selection_model: Model the router should use for its own
            extraction call, sent as ``route_selector.config.model`` so a
            deployment can pin one it has query access to. ``None`` omits
            ``config`` entirely and leaves the router's frozen default.
        :param scenario_menus: Router version → scenario → required arm menu
            (see :data:`TASK_V1_MENUS`).
        """
        self._url = base_url.rstrip("/") + "/" + ROUTES_SELECT_PATH
        self._router_name = router_name
        self._auth = auth
        self._databricks_profile = databricks_profile
        # Cached SDK Config for the profile (created lazily), reused across
        # calls; its authenticate() refreshes the OAuth token as needed.
        self._sdk_config: Any = None
        self._model_prefixes = model_prefixes or []
        self._request_timeout = request_timeout
        self._selection_model = selection_model or None
        # All router-contract knowledge (arm menus, harness derivation,
        # unservable-pick mapping) lives behind this seam.
        self._source = TaskV1RouteOptionSource(
            router_name=router_name,
            scenario_menus=scenario_menus,
            model_prefixes=self._model_prefixes,
        )
        # Human-readable reason the most recent route() returned None, for the
        # caller to surface in the UI (a bare None hides the actual cause, e.g.
        # a 401 or the task_v0 required-model-set error). Set on every failure
        # path, cleared on success.
        self.last_error: str | None = None

    def _resolve_auth(self) -> Any:  # type: ignore[explicit-any]  # httpx.Auth | None
        """Return the auth for a request, refreshing an OAuth token per call.

        A static *auth* (explicit api_key) is returned as-is. When only a
        Databricks profile is configured, mint a fresh bearer from the cached
        SDK ``Config`` so token expiry never surfaces as a router 401.
        """
        if self._auth is not None:
            return self._auth
        if self._databricks_profile is None:
            return None
        try:
            if self._sdk_config is None:
                from databricks.sdk.config import Config

                self._sdk_config = Config(profile=self._databricks_profile)
            headers = self._sdk_config.authenticate()
        except Exception:  # noqa: BLE001 — auth failure degrades to unauthenticated
            _logger.warning(
                "ExternalRoutingClient: could not resolve auth for profile %r",
                self._databricks_profile,
                exc_info=True,
            )
            return None
        token = (headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        if not token:
            return None
        return _bearer_auth(token)

    async def route(
        self,
        message: str,
        available_models: dict[str, list[str]],
    ) -> RoutingResult | None:
        import httpx
        from google.protobuf import json_format

        from omnigent.api.routing.v1 import routing_pb2 as pb

        # The seam turns the catalog into router vocabulary and injects
        # whatever arms the router's scenario menu demands.
        harnesses = list(available_models)
        specs = self._source.build_route_options(harnesses, available_models)
        if not specs:
            return None
        options = [pb.RouteOption(model=s.model, harness=s.harness) for s in specs]
        selector = pb.RouteSelector(router_name=self._router_name)
        if self._selection_model:
            # The router makes its own extraction call; ``config.model`` pins
            # the model it uses so a deployment can name one it can query.
            json_format.ParseDict({"model": self._selection_model}, selector.config)
        request = pb.SelectRouteRequest(
            route_options=options,
            task=pb.Task(prompt=message[:4000]),
            route_selector=selector,
        )
        # snake_case wire format — the router uses the proto field names.
        body = json_format.MessageToDict(request, preserving_proto_field_name=True)
        _logger.info("ExternalRoutingClient: available_models=%s", dict(available_models))
        _logger.info("ExternalRoutingClient: POST %s body=%s", self._url, body)
        # Resolve auth per call (SDK token refresh is a blocking HTTP call, so
        # run it off the event loop) — keeps a long-lived server from sending a
        # token that has expired since startup.
        import asyncio

        auth = await asyncio.to_thread(self._resolve_auth)
        self.last_error = None
        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as http:
                resp = await http.post(
                    self._url,
                    headers={"Content-Type": "application/json"},
                    json=body,
                    auth=auth,
                )
        except httpx.HTTPError as exc:
            # Transport-level failure (connect/timeout/DNS): no response body.
            _logger.warning("ExternalRoutingClient: routes:select request failed: %s", exc)
            self.last_error = f"router request failed: {exc}"
            return None
        if resp.status_code >= 400:
            # Log the response body — the gateway puts the actual reason there
            # (e.g. task_v0's required-model-set error), which the bare status
            # code from raise_for_status() omits.
            _logger.warning(
                "ExternalRoutingClient: routes:select returned %s: %s",
                resp.status_code,
                resp.text[:2000],
            )
            self.last_error = (
                f"router returned HTTP {resp.status_code}: {_router_error_detail(resp.text)}"
            )
            return None
        try:
            out = json_format.ParseDict(resp.json(), pb.SelectRouteResponse())
        except (ValueError, json_format.ParseError):
            _logger.warning(
                "ExternalRoutingClient: could not parse routes:select response: %s",
                resp.text[:2000],
            )
            self.last_error = "router returned an unparseable response"
            return None
        if not out.route_selection:
            self.last_error = "router returned no route selection"
            return None
        selected = out.route_selection[0].route_option
        if not selected.model:
            self.last_error = "router returned an empty model"
            return None
        # The seam maps the pick back to a servable catalog id and derives the
        # harness from the arm's family (the echoed tag is untrusted).
        resolved = self._source.resolve_selection(
            RoutePick(model=selected.model, harness=selected.harness or None),
            harnesses,
            available_models,
        )
        if resolved is None:
            _logger.warning(
                "ExternalRoutingClient: router returned model %r (harness %r) "
                "not in the candidate set; ignoring",
                selected.model,
                selected.harness,
            )
            self.last_error = f"router picked model {selected.model!r} outside the candidate set"
            return None
        if resolved.model != resolved.raw_model:
            _logger.info(
                "ExternalRoutingClient: router pick %r is not servable here; using %r",
                resolved.raw_model,
                resolved.model,
            )
        return RoutingResult(
            model=resolved.model,
            rationale=out.rationale,
            harness=resolved.harness,
            raw_model=resolved.raw_model,
        )


# ── Public API ──────────────────────────────────────────────────────────────

# SDK harnesses offered as candidates when the user picks "auto" harness.
# Native harnesses are excluded: they require CLI binaries that may not be
# installed, and they bake the model at terminal launch rather than per-turn.
#
# Order matters: it is the insertion order of the candidate set sent to the
# router AND the tiebreak order when a model is served by multiple harnesses
# (both the external router's id-only fallback and our own model-ownership
# fallback pick the FIRST harness owning the model). codex precedes pi so GPT
# models default to codex — which uses the Responses API and handles gpt-5.5+
# reasoning models with tools, whereas pi's openai-completions path 400s on
# them. claude-sdk owns Claude models; pi is the last-resort multi-model home.
_AUTO_ROUTING_HARNESSES: tuple[str, ...] = ("claude-sdk", "codex", "pi")

# The live runner catalog (fetch_runner_models) keys rows by WORKER name — the
# sub-agent names declared in the parent spec (e.g. "claude_code") plus "self"
# for the session's own harness — NOT by harness id. Map the common worker
# names back to their harness id so a child session's catalog still yields
# routable candidates. Unknown worker names are ignored (the static
# infer_models fallback covers them).
_WORKER_NAME_TO_HARNESS: dict[str, str] = {
    "claude_code": "claude-sdk",
    "claude-sdk": "claude-sdk",
    "codex": "codex",
    "pi": "pi",
}


async def route_session_harness(
    user_message: str,
    *,
    session_id: str | None = None,
    catalog_session_id: str | None = None,
    runner_client: httpx.AsyncClient | None = None,
) -> tuple[str | None, str | None, dict[str, Any] | None, str | None]:
    """Pick the best harness + model for a new session via the routing client.

    Builds a candidate set from the live runner catalog when *catalog_session_id*
    (defaulting to *session_id*) and *runner_client* are provided, falling back
    to the static ``infer_models`` table for any harness not represented in the
    live data.  Only harnesses in :data:`_AUTO_ROUTING_HARNESSES` are offered as
    candidates.

    :param user_message: The user's first message text, used to size the task.
    :param session_id: Session being routed (optional).
    :param catalog_session_id: Session whose catalog defines the candidate set.
        For a sub-agent this should be the PARENT session id — the parent's
        catalog enumerates the spawnable workers (claude_code/codex/pi) with
        their full model lists, whereas the child's own leaf catalog only has a
        ``"self"`` row and would force the static fallback. Defaults to
        *session_id* when unset.
    :param runner_client: HTTP client pointed at the runner (optional).
    :returns: ``(harness, model, verdict, error)`` — on success ``error`` is
        ``None``; on failure ``harness``, ``model``, and ``verdict`` are ``None``
        and ``error`` carries a human-readable reason shown in the UI.
    """
    if not user_message:
        return None, None, None, None
    try:
        from omnigent.runtime._globals import _caps
    except ImportError:
        return None, None, None, "Intelligent routing is not available."

    if _caps is None or _caps.routing_client is None:
        return None, None, None, "Intelligent routing is not configured on this server."

    # Fetch the live catalog. Its rows are keyed by worker name (sub-agent
    # names + "self"), so normalize those to harness ids before matching
    # against _AUTO_ROUTING_HARNESSES. Prefer catalog_session_id (the parent
    # for a sub-agent) so the candidate set is the full spawnable-worker map,
    # independent of whether the routed session is top-level or a sub-agent.
    _catalog_sid = catalog_session_id or session_id
    live_catalog: dict[str, list[str]] | None = None
    if _catalog_sid and runner_client is not None:
        live_catalog = await fetch_runner_models(_catalog_sid, runner_client)

    # NOTE: we do NOT filter incompatible (harness, model) pairs out of the
    # candidate set here. The external router (task_v0) enforces a required
    # model set and 400s if any required model is missing, so dropping e.g.
    # gpt-5-6-luna would break the whole request. Instead we send the full set
    # and correct an incompatible verdict afterward via
    # _redirect_incompatible_pick.
    harness_models: dict[str, list[str]] = {}
    if live_catalog:
        for worker_name, worker_models in live_catalog.items():
            harness = _WORKER_NAME_TO_HARNESS.get(worker_name)
            if harness is None or harness not in _AUTO_ROUTING_HARNESSES:
                continue
            if worker_models:
                # First worker wins for a given harness id (dedupe).
                harness_models.setdefault(harness, worker_models)

    # Fall back to the static table when the live catalog produced no
    # routable candidates (e.g. a child session whose catalog only lists
    # "self" under an unrecognized worker name, or the runner was unreachable).
    if not harness_models:
        for h in _AUTO_ROUTING_HARNESSES:
            models = infer_models(h)
            if models:
                harness_models[h] = models

    if not harness_models:
        return None, None, None, "No routable harnesses are available on this runner."

    try:
        result = await _caps.routing_client.route(user_message, harness_models)
    except Exception as exc:  # routing failures must not block session creation
        _logger.exception("smart_routing: route_session_harness failed")
        return None, None, None, f"Routing call failed: {exc}"

    if result is None:
        # Surface the client's specific failure reason (e.g. HTTP 401 with the
        # gateway's message) when it exposes one; otherwise a generic note.
        detail = getattr(_caps.routing_client, "last_error", None)
        reason = (
            f"Routing unavailable: {detail}"
            if detail
            else "The router returned no verdict; using default harness."
        )
        return None, None, None, reason

    # The seam owns harness derivation (the router's echoed harness is
    # untrusted) and moves a pick off a harness that can't serve it.
    resolved = route_option_source().resolve_selection(
        RoutePick(model=result.model, harness=result.harness),
        list(harness_models),
        harness_models,
    )
    if resolved is None:
        chosen_harness, chosen_model = None, result.model
    else:
        chosen_harness, chosen_model = resolved.harness, resolved.model

    # The UI shows what the router said, not only what we could run.
    raw_model = result.raw_model or (resolved.raw_model if resolved else None)
    verdict: dict[str, Any] = {"model": chosen_model, "rationale": result.rationale}
    if raw_model and raw_model != chosen_model:
        verdict["raw_model"] = raw_model

    _logger.info(
        "smart_routing: auto-harness harness=%s model=%s rationale=%s",
        chosen_harness,
        chosen_model,
        result.rationale,
    )
    return chosen_harness, chosen_model, verdict, None


async def route_turn(
    harness: str | None,
    user_message: str,
    *,
    session_id: str | None = None,
    runner_client: httpx.AsyncClient | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Pick the best model for a turn via :attr:`RuntimeCaps.routing_client`.

    When *session_id* and *runner_client* are provided, fetches live model
    availability from the runner's ``/v1/sessions/{id}/models`` endpoint.
    Falls back to the static :func:`infer_models` lookup table if the runner
    is unreachable or returns no data.
    """
    try:
        from omnigent.runtime._globals import _caps
    except ImportError:
        return None, None

    if _caps is None or _caps.routing_client is None:
        return None, None

    # Prefer live runner catalog — but only the "self" worker entry.
    # The catalog includes sub-agent workers (claude_code, pi, codex…);
    # for brain-turn routing we only want the models this session's own
    # harness can run, not the sub-agents' model lists.
    # Key the candidate map by the session's harness id (not the catalog's
    # "self" label) so the seam can infer the right single-harness scenario.
    available: dict[str, list[str]] | None = None
    if session_id and runner_client is not None:
        catalog = await fetch_runner_models(session_id, runner_client)
        if catalog and "self" in catalog:
            available = {harness or "self": catalog["self"]}
    if not available:
        models = infer_models(harness)
        if models is None:
            return None, None
        available = {harness or "": models}

    result = await _caps.routing_client.route(user_message, available)
    if result is None:
        return None, None

    # A turn can only change the model, so keep the harness the seam derives
    # out of it and just take the servable id.
    resolved = route_option_source().resolve_selection(
        RoutePick(model=result.model, harness=result.harness),
        list(available),
        available,
    )
    model = resolved.model if resolved is not None else result.model
    raw_model = result.raw_model or (resolved.raw_model if resolved else None)

    _logger.info(
        "smart_routing: model=%s rationale=%s",
        model,
        result.rationale,
    )
    verdict: dict[str, Any] = {"model": model, "rationale": result.rationale}
    if raw_model and raw_model != model:
        verdict["raw_model"] = raw_model
    return model, verdict
