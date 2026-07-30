"""Tests for the native-subagent routing endpoint and policy."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from omnigent.entities.conversation import RoutingDecisionData
from omnigent.runner.subagent_routing import (
    ADVERTISEMENT_FILE,
    SubagentRouteDecision,
    SubagentRouteRequest,
    candidate_models,
    clear_cache,
    ensure_session_router,
    make_server_relay_resolver,
    persist_subagent_decision,
    read_advertisement,
    relayed_decisions,
    resolve_subagent_route,
    routed_models,
    router_dir_for_session,
    routing_enabled,
    session_picks,
    session_router_env,
    shutdown_session_router,
    start_subagent_router,
    subagent_routing_enabled,
    write_advertisement,
)
from omnigent.server.smart_routing import RoutingResult

CLAUDE_MODEL = "databricks-claude-opus-4-8"
GPT_MODEL = "databricks-gpt-5-5"
PARENT_MODEL = "databricks-claude-sonnet-4-6"


# ── Stubs ───────────────────────────────────────────────────────────


class _FakeRoutingClient:
    """Records calls and returns a canned verdict (or raises)."""

    def __init__(
        self,
        result: RoutingResult | None = None,
        *,
        error: Exception | None = None,
        last_error: str | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.last_error = last_error
        self.calls: list[tuple[str, dict[str, list[str]]]] = []

    async def route(
        self, message: str, available_models: dict[str, list[str]]
    ) -> RoutingResult | None:
        self.calls.append((message, available_models))
        if self._error is not None:
            raise self._error
        return self._result


@dataclass
class _FakeSettings:
    subagent_fail_mode: str = "open"
    subagent_cache_ttl_s: float = 300.0


@dataclass
class _FakeCaps:
    routing_client: Any = None  # type: ignore[explicit-any]
    routing_settings: Any = None  # type: ignore[explicit-any]


@dataclass
class _PersistedItem:
    id: str


@dataclass
class _FakeStore:
    appended: list[Any] = field(default_factory=list)  # type: ignore[explicit-any]

    def append(self, session_id: str, items: list[Any]) -> list[_PersistedItem]:  # type: ignore[explicit-any]
        del session_id
        self.appended.extend(items)
        return [_PersistedItem(id=f"item_{len(self.appended)}")]


def _request(**overrides: Any) -> SubagentRouteRequest:
    kwargs: dict[str, Any] = {
        "harness": "claude-native",
        "task_name": "code-reviewer",
        "prompt": "review the diff",
        "parent_model": PARENT_MODEL,
    }
    kwargs.update(overrides)
    return SubagentRouteRequest(**kwargs)


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    clear_cache()
    yield
    clear_cache()


# ── Candidate set ───────────────────────────────────────────────────


def test_candidate_models_stays_in_family_by_default() -> None:
    candidates = candidate_models("claude-native")
    assert set(candidates) == {"claude-native"}
    assert CLAUDE_MODEL in candidates["claude-native"]


def test_candidate_models_prefers_the_live_catalog() -> None:
    """A model the workspace serves today must not look unservable."""
    catalog = {"self": ["databricks-claude-sonnet-5", "databricks-claude-opus-4-8"]}
    candidates = candidate_models("claude-native", catalog=catalog)
    assert candidates == {
        "claude-native": ["databricks-claude-sonnet-5", "databricks-claude-opus-4-8"]
    }


def test_candidate_models_falls_back_to_the_static_table_per_harness() -> None:
    catalog = {"self": ["databricks-claude-sonnet-5"]}
    candidates = candidate_models("claude-native", cross_harness=True, catalog=catalog)
    assert candidates["claude-native"] == ["databricks-claude-sonnet-5"]
    # No codex row in the catalog — the static table fills that harness in.
    assert GPT_MODEL in candidates["codex-native"]


def test_candidate_models_ignores_an_empty_catalog() -> None:
    candidates = candidate_models("claude-native", catalog={})
    assert CLAUDE_MODEL in candidates["claude-native"]


def test_candidate_models_offers_both_families_for_auto_sessions() -> None:
    candidates = candidate_models("claude-native", cross_harness=True)
    assert set(candidates) == {"claude-native", "codex-native"}
    assert CLAUDE_MODEL in candidates["claude-native"]
    assert GPT_MODEL in candidates["codex-native"]


# ── Policy ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_same_family_pick_rewrites() -> None:
    client = _FakeRoutingClient(
        RoutingResult(model=CLAUDE_MODEL, rationale="deep reasoning", harness="claude-sdk")
    )
    decision = await resolve_subagent_route(
        "conv_1", _request(), caps=_FakeCaps(routing_client=client)
    )
    assert decision.action == "rewrite"
    assert decision.model == CLAUDE_MODEL
    assert decision.harness is None
    assert decision.raw_model == CLAUDE_MODEL
    assert decision.rationale == "deep reasoning"
    assert len(decision.decision_id) == 36


@pytest.mark.asyncio
async def test_live_catalog_pick_is_applied_exactly() -> None:
    """A live-catalog model is offered and applied verbatim — no substitution."""
    client = _FakeRoutingClient(
        RoutingResult(model="databricks-claude-sonnet-5", rationale="r", harness="claude-sdk")
    )
    decision = await resolve_subagent_route(
        "conv_1",
        _request(),
        caps=_FakeCaps(routing_client=client),
        catalog={"self": ["databricks-claude-sonnet-5", "databricks-claude-opus-4-8"]},
    )
    assert client.calls[0][1] == {
        "claude-native": ["databricks-claude-sonnet-5", "databricks-claude-opus-4-8"]
    }
    assert decision.action == "rewrite"
    assert decision.model == "databricks-claude-sonnet-5"


@pytest.mark.asyncio
async def test_cross_family_pick_redirects_to_counterpart_harness() -> None:
    client = _FakeRoutingClient(
        RoutingResult(model=GPT_MODEL, rationale="narrow change", harness="codex")
    )
    decision = await resolve_subagent_route(
        "conv_1", _request(), caps=_FakeCaps(routing_client=client), cross_harness=True
    )
    assert decision.action == "redirect"
    assert decision.model == GPT_MODEL
    assert decision.harness == "codex-native"


@pytest.mark.asyncio
async def test_in_family_session_only_offers_its_own_harness() -> None:
    client = _FakeRoutingClient(
        RoutingResult(model=GPT_MODEL, rationale="narrow change", harness="codex")
    )
    decision = await resolve_subagent_route(
        "conv_1", _request(), caps=_FakeCaps(routing_client=client)
    )
    # Only Claude arms were offered, so a Codex pick is unrunnable — a
    # constrained session can never redirect.
    assert set(client.calls[0][1]) == {"claude-native"}
    assert decision.action == "deny"


@pytest.mark.asyncio
async def test_pick_outside_candidate_set_denies() -> None:
    client = _FakeRoutingClient(
        RoutingResult(model="databricks-kimi-k2", rationale="unavailable", harness="claude-sdk")
    )
    decision = await resolve_subagent_route(
        "conv_1", _request(), caps=_FakeCaps(routing_client=client)
    )
    assert decision.action == "deny"
    assert decision.model is None
    assert "cannot run" in decision.rationale


@pytest.mark.asyncio
async def test_parent_model_pick_allows_unchanged() -> None:
    client = _FakeRoutingClient(
        RoutingResult(model=PARENT_MODEL, rationale="parent model fits", harness="claude-sdk")
    )
    decision = await resolve_subagent_route(
        "conv_1", _request(), caps=_FakeCaps(routing_client=client)
    )
    assert decision.action == "allow"
    assert decision.model == PARENT_MODEL


@pytest.mark.asyncio
async def test_fork_is_exempt_and_never_calls_router() -> None:
    client = _FakeRoutingClient(RoutingResult(model=CLAUDE_MODEL, rationale="x"))
    decision = await resolve_subagent_route(
        "conv_1", _request(fork=True), caps=_FakeCaps(routing_client=client)
    )
    assert decision.action == "allow"
    assert decision.model == PARENT_MODEL
    assert client.calls == []


@pytest.mark.asyncio
async def test_no_routable_signal_skips_the_router_and_allows_unchanged() -> None:
    """A codex spawn with no prompt and no task name is decided locally.

    Codex encrypts the spawn message, so an unnamed subagent carries nothing
    to score. Calling the router anyway returned "HTTP 400: task.prompt is
    required", which the fail-open path then dressed up as "Routing
    unavailable (...)" on the decision chip — an outage the user does not
    have. The spawn genuinely inherits the parent's thread model (confirmed
    by the SubagentStart audit), so the verdict is allow-unchanged and the
    rationale says exactly that.
    """
    client = _FakeRoutingClient(RoutingResult(model=CLAUDE_MODEL, rationale="x"))
    decision = await resolve_subagent_route(
        "conv_1",
        _request(prompt=None, task_name=""),
        caps=_FakeCaps(routing_client=client),
    )
    assert decision.action == "allow"
    # The parent's model is named so the chip and the audit reconciliation
    # both see the model the subagent actually starts on.
    assert decision.model == PARENT_MODEL
    assert decision.rationale == (
        "No routable signal (encrypted prompt, no task name); subagent inherits the session model"
    )
    # Never reaches the router: no 400, and nothing to report as an outage.
    assert client.calls == []
    assert "unavailable" not in decision.rationale.lower()


@pytest.mark.asyncio
async def test_no_routable_signal_is_not_cached_as_an_outage() -> None:
    """The no-signal verdict is recomputed, never served from the cache.

    It is not a router result, so it must not occupy a cache slot that a
    later named spawn (or a fixed client) would be answered from.
    """
    client = _FakeRoutingClient(RoutingResult(model=CLAUDE_MODEL, rationale="picked"))
    caps = _FakeCaps(routing_client=client)
    await resolve_subagent_route("conv_1", _request(prompt=None, task_name=""), caps=caps)
    assert client.calls == []
    # A spawn that does carry a signal still reaches the router.
    routed = await resolve_subagent_route("conv_1", _request(), caps=caps)
    assert len(client.calls) == 1
    assert routed.action in ("allow", "rewrite", "redirect")


@pytest.mark.asyncio
async def test_router_outage_fails_open_by_default() -> None:
    client = _FakeRoutingClient(error=RuntimeError("router down"))
    decision = await resolve_subagent_route(
        "conv_1", _request(), caps=_FakeCaps(routing_client=client)
    )
    assert decision.action == "allow"
    assert "Routing unavailable" in decision.rationale


@pytest.mark.asyncio
async def test_router_outage_fails_closed_when_configured() -> None:
    client = _FakeRoutingClient(error=RuntimeError("router down"))
    caps = _FakeCaps(
        routing_client=client,
        routing_settings=_FakeSettings(subagent_fail_mode="closed"),
    )
    decision = await resolve_subagent_route("conv_1", _request(), caps=caps)
    assert decision.action == "deny"


@pytest.mark.asyncio
async def test_no_verdict_surfaces_last_error() -> None:
    client = _FakeRoutingClient(None, last_error="menu mismatch")
    decision = await resolve_subagent_route(
        "conv_1", _request(), caps=_FakeCaps(routing_client=client)
    )
    assert decision.action == "allow"
    assert "menu mismatch" in decision.rationale


@pytest.mark.asyncio
async def test_missing_routing_client_uses_fail_mode() -> None:
    caps = _FakeCaps(routing_settings=_FakeSettings(subagent_fail_mode="closed"))
    decision = await resolve_subagent_route("conv_1", _request(), caps=caps)
    assert decision.action == "deny"


# ── Cache ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identical_spawns_hit_cache_once() -> None:
    client = _FakeRoutingClient(
        RoutingResult(model=CLAUDE_MODEL, rationale="deep reasoning", harness="claude-sdk")
    )
    caps = _FakeCaps(routing_client=client)
    first = await resolve_subagent_route("conv_1", _request(), caps=caps)
    second = await resolve_subagent_route("conv_1", _request(), caps=caps)
    assert len(client.calls) == 1
    assert second.decision_id == first.decision_id
    picks = session_picks("conv_1")
    assert len(picks) == 1
    assert picks[0]["model"] == CLAUDE_MODEL


@pytest.mark.asyncio
async def test_cache_expires_after_ttl() -> None:
    client = _FakeRoutingClient(
        RoutingResult(model=CLAUDE_MODEL, rationale="deep reasoning", harness="claude-sdk")
    )
    caps = _FakeCaps(
        routing_client=client, routing_settings=_FakeSettings(subagent_cache_ttl_s=5.0)
    )
    await resolve_subagent_route("conv_1", _request(), caps=caps, now=1000.0)
    await resolve_subagent_route("conv_1", _request(), caps=caps, now=1006.0)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_cache_is_per_session_and_per_task() -> None:
    client = _FakeRoutingClient(
        RoutingResult(model=CLAUDE_MODEL, rationale="deep reasoning", harness="claude-sdk")
    )
    caps = _FakeCaps(routing_client=client)
    await resolve_subagent_route("conv_1", _request(), caps=caps)
    await resolve_subagent_route("conv_2", _request(), caps=caps)
    await resolve_subagent_route("conv_1", _request(task_name="tester"), caps=caps)
    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_outage_decisions_are_not_cached() -> None:
    client = _FakeRoutingClient(error=RuntimeError("router down"))
    caps = _FakeCaps(routing_client=client)
    await resolve_subagent_route("conv_1", _request(), caps=caps)
    await resolve_subagent_route("conv_1", _request(), caps=caps)
    assert len(client.calls) == 2


# ── Decision persistence ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_decision_is_persisted_with_native_subagent_scope() -> None:
    client = _FakeRoutingClient(
        RoutingResult(model=CLAUDE_MODEL, rationale="deep reasoning", harness="claude-sdk")
    )
    records: list[RoutingDecisionData] = []

    async def _persist(record: RoutingDecisionData) -> None:
        records.append(record)

    decision = await resolve_subagent_route(
        "conv_1",
        _request(),
        caps=_FakeCaps(routing_client=client),
        persist=_persist,
    )
    assert len(records) == 1
    data = records[0].model_dump()
    assert data["scope"] == "native_subagent"
    assert data["decision_id"] == decision.decision_id
    assert data["raw_model"] == CLAUDE_MODEL
    assert data["model"] == CLAUDE_MODEL
    assert data["harness"] == "claude-native"
    assert data["applied"] is True
    assert data["rationale"] == "deep reasoning"


@pytest.mark.asyncio
async def test_deny_is_persisted_unapplied() -> None:
    client = _FakeRoutingClient(error=RuntimeError("router down"))
    records: list[RoutingDecisionData] = []

    async def _persist(record: RoutingDecisionData) -> None:
        records.append(record)

    caps = _FakeCaps(
        routing_client=client,
        routing_settings=_FakeSettings(subagent_fail_mode="closed"),
    )
    await resolve_subagent_route("conv_1", _request(), caps=caps, persist=_persist)
    assert records[0].applied is False
    assert records[0].model == PARENT_MODEL


@pytest.mark.asyncio
async def test_persist_appends_routing_decision_item() -> None:
    store = _FakeStore()
    record = RoutingDecisionData(
        model=CLAUDE_MODEL,
        applied=True,
        rationale="deep reasoning",
        decision_id="dec_1",
        harness="claude-native",
        raw_model="claude-opus-4-8",
        scope="native_subagent",
    )
    await persist_subagent_decision("conv_1", store, record)
    assert len(store.appended) == 1
    item = store.appended[0]
    assert item.type == "routing_decision"
    assert item.data.model == CLAUDE_MODEL
    assert item.data.rationale == "deep reasoning"
    # The additive fields ride along once the item model carries them.
    if hasattr(item.data, "scope"):
        assert item.data.scope == "native_subagent"
        assert item.data.decision_id == "dec_1"
        assert item.data.raw_model == "claude-opus-4-8"
        assert item.data.harness == "claude-native"


@pytest.mark.asyncio
async def test_persist_failure_does_not_break_the_decision() -> None:
    class _BoomStore:
        def append(self, session_id: str, items: list[Any]) -> list[_PersistedItem]:
            raise RuntimeError("db down")

    client = _FakeRoutingClient(
        RoutingResult(model=CLAUDE_MODEL, rationale="deep reasoning", harness="claude-sdk")
    )

    async def _persist(record: RoutingDecisionData) -> None:
        await persist_subagent_decision("conv_1", _BoomStore(), record)

    decision = await resolve_subagent_route(
        "conv_1",
        _request(),
        caps=_FakeCaps(routing_client=client),
        persist=_persist,
    )
    assert decision.action == "rewrite"


# ── Advertisement + loopback endpoint ───────────────────────────────


def test_advertisement_roundtrip(tmp_path: Path) -> None:
    path = write_advertisement(tmp_path, url="http://127.0.0.1:1234", token="tok")
    assert path.name == ADVERTISEMENT_FILE
    assert read_advertisement(tmp_path) == {"url": "http://127.0.0.1:1234", "token": "tok"}


def test_read_advertisement_missing(tmp_path: Path) -> None:
    assert read_advertisement(tmp_path) is None


def _post(url: str, body: dict[str, Any], token: str | None) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


@pytest.mark.asyncio
async def test_loopback_endpoint_serves_decisions_and_checks_token(tmp_path: Path) -> None:
    canned = SubagentRouteDecision(
        action="rewrite", rationale="deep reasoning", model=CLAUDE_MODEL, raw_model=CLAUDE_MODEL
    )
    seen: list[SubagentRouteRequest] = []

    async def _resolver(session_id: str, req: SubagentRouteRequest) -> SubagentRouteDecision:
        del session_id
        seen.append(req)
        return canned

    router = start_subagent_router(
        bridge_dir=tmp_path,
        session_id="conv_1",
        resolver=_resolver,
        loop=asyncio.get_running_loop(),
    )
    try:
        advertised = read_advertisement(tmp_path)
        assert advertised is not None
        url = f"{advertised['url']}/v1/sessions/conv_1/route-subagent"
        body = {
            "harness": "claude-native",
            "task_name": "code-reviewer",
            "prompt": "review the diff",
            "fork": False,
            "parent_model": PARENT_MODEL,
        }

        status, payload = await asyncio.to_thread(_post, url, body, advertised["token"])
        assert status == 200
        assert payload == canned.to_payload()
        assert seen[0].task_name == "code-reviewer"

        status, _ = await asyncio.to_thread(_post, url, body, "wrong-token")
        assert status == 401
        status, _ = await asyncio.to_thread(_post, url, body, None)
        assert status == 401
        assert len(seen) == 1

        status, _ = await asyncio.to_thread(
            _post,
            f"{advertised['url']}/v1/sessions/other/route-subagent",
            body,
            advertised["token"],
        )
        assert status == 404

        status, _ = await asyncio.to_thread(_post, url, {"task_name": "x"}, advertised["token"])
        assert status == 400
    finally:
        router.close()
    assert not (tmp_path / ADVERTISEMENT_FILE).exists()


@pytest.mark.asyncio
async def test_server_relay_resolver_forwards_and_parses() -> None:
    posted: list[tuple[str, dict[str, Any]]] = []

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "action": "redirect",
                "model": GPT_MODEL,
                "harness": "codex-native",
                "raw_model": "gpt-5-6-sol",
                "rationale": "narrow change",
                "decision_id": "dec_9",
            }

    class _Client:
        async def post(self, path: str, *, json: dict[str, Any], timeout: float) -> _Resp:
            del timeout
            posted.append((path, json))
            return _Resp()

    resolver = make_server_relay_resolver(_Client())
    decision = await resolver("conv_1", _request())
    assert posted[0][0] == "/v1/sessions/conv_1/hooks/route-subagent"
    assert posted[0][1]["harness"] == "claude-native"
    assert decision.action == "redirect"
    assert decision.harness == "codex-native"
    assert decision.decision_id == "dec_9"


@pytest.mark.asyncio
async def test_server_relay_resolver_applies_fail_mode_on_hop_failure() -> None:
    class _DeadClient:
        async def post(self, path: str, *, json: dict[str, Any], timeout: float) -> Any:
            raise RuntimeError("connection refused")

    open_decision = await make_server_relay_resolver(_DeadClient())("conv_1", _request())
    closed_decision = await make_server_relay_resolver(_DeadClient(), fail_mode="closed")(
        "conv_1", _request()
    )
    assert open_decision.action == "allow"
    assert closed_decision.action == "deny"


@pytest.mark.asyncio
async def test_loopback_endpoint_applies_fail_mode_when_resolver_errors(tmp_path: Path) -> None:
    async def _resolver(session_id: str, req: SubagentRouteRequest) -> SubagentRouteDecision:
        raise RuntimeError("boom")

    router = start_subagent_router(
        bridge_dir=tmp_path,
        session_id="conv_1",
        resolver=_resolver,
        loop=asyncio.get_running_loop(),
        fail_mode="closed",
    )
    try:
        advertised = read_advertisement(tmp_path)
        assert advertised is not None
        status, payload = await asyncio.to_thread(
            _post,
            f"{advertised['url']}/v1/sessions/conv_1/route-subagent",
            {"harness": "claude-native"},
            advertised["token"],
        )
        assert status == 200
        assert payload["action"] == "deny"
    finally:
        router.close()


# ── Enablement gate + session router lifecycle (P7) ─────────────────


def test_routing_enabled_reads_the_session_toggle() -> None:
    assert routing_enabled("on") is True
    assert routing_enabled("off") is False
    assert routing_enabled(None) is False
    assert routing_enabled(None, parent_cost_control_mode="on") is True


def test_routing_enabled_requires_a_client_when_caps_are_given() -> None:
    assert routing_enabled("on", caps=_FakeCaps(routing_client=None)) is False
    assert routing_enabled("on", caps=_FakeCaps(routing_client=object())) is True


def test_subagent_routing_inherits_the_session_routing_state() -> None:
    assert subagent_routing_enabled(None, cost_control_mode="on") is True
    assert subagent_routing_enabled(None, cost_control_mode="off") is False
    assert subagent_routing_enabled(None, cost_control_mode=None) is False
    assert (
        subagent_routing_enabled(
            None,
            cost_control_mode=None,
            parent_cost_control_mode="on",
        )
        is True
    )


def test_subagent_routing_override_beats_the_inherited_state() -> None:
    assert subagent_routing_enabled("off", cost_control_mode="on") is False
    assert subagent_routing_enabled("on", cost_control_mode=None) is True
    assert (
        subagent_routing_enabled(
            "off",
            cost_control_mode=None,
            parent_cost_control_mode="on",
        )
        is False
    )


def test_router_dir_for_session_is_owner_only(tmp_path: Path) -> None:
    path = router_dir_for_session("conv_router_dir")
    assert path.is_dir()
    assert path.stat().st_mode & 0o777 == 0o700


def test_ensure_session_router_is_idempotent_and_advertises_everywhere(
    tmp_path: Path,
) -> None:
    class _DeadClient:
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("server down")

    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"

    async def _run() -> None:
        router = ensure_session_router(
            "conv_lifecycle",
            bridge_dir=first_dir,
            server_client=_DeadClient(),
        )
        again = ensure_session_router(
            "conv_lifecycle",
            bridge_dir=second_dir,
            server_client=_DeadClient(),
        )
        assert again is router
        # Same rendezvous advertised in both directories.
        assert read_advertisement(first_dir) == read_advertisement(second_dir)
        env = session_router_env("conv_lifecycle")
        assert env["OMNIGENT_SUBAGENT_ROUTER_SESSION_ID"] == "conv_lifecycle"
        assert env["OMNIGENT_CODEX_SUBAGENT_ROUTER_DIR"] == str(first_dir)
        shutdown_session_router("conv_lifecycle")
        assert session_router_env("conv_lifecycle") == {}

    asyncio.run(_run())


def test_relayed_decisions_start_empty() -> None:
    assert relayed_decisions("conv_never_routed") == ()
    assert routed_models("conv_never_routed") == frozenset()
