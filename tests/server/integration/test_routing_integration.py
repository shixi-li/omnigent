"""L5 integration coverage for intelligent routing with a fake router.

Drives the server's own routing paths against a stub routing client:
the ``sys_session_send`` child override precedence, the native-subagent
relay endpoint (transcript item + child-sessions join), the subagent
fail-mode knob, and the decision cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from omnigent.runner.subagent_routing import ROUTING_DECISION_LABEL_KEY, clear_cache
from omnigent.server.routes._sessions import orchestration as orchestration_module
from omnigent.server.schemas import SessionEventInput
from omnigent.server.smart_routing import RoutingResult, RoutingSettings
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from tests.server.helpers import create_test_agent

pytestmark = pytest.mark.asyncio

ROUTED_MODEL = "databricks-claude-opus-4-8"
LLM_PICKED_MODEL = "databricks-claude-sonnet-4-6"
GPT_MODEL = "databricks-gpt-5-5"


class _FakeRoutingClient:
    """Returns a canned verdict and counts calls."""

    def __init__(self, result: RoutingResult | None, *, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.last_error: str | None = None
        self.calls: list[str] = []
        self.offered: list[dict[str, list[str]]] = []

    async def route(
        self,
        message: str,
        available_models: dict[str, list[str]],
    ) -> RoutingResult | None:
        self.calls.append(message)
        self.offered.append(dict(available_models))
        if self._error is not None:
            raise self._error
        return self._result


@dataclass
class _FakeCaps:
    routing_client: Any = None  # type: ignore[explicit-any]
    routing_settings: RoutingSettings = field(default_factory=RoutingSettings)


@pytest.fixture(autouse=True)
def _clear_decision_cache() -> Any:
    clear_cache()
    yield
    clear_cache()


def _echo_runner_client() -> httpx.AsyncClient:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={"queued": True})

    return httpx.AsyncClient(
        base_url="http://runner.test",
        transport=httpx.MockTransport(_handler),
    )


async def _parent_and_child(
    client: httpx.AsyncClient,
    db_uri: str,
    *,
    agent_name: str,
    child_model_override: str | None = None,
) -> tuple[dict[str, Any], Any, SqlAlchemyConversationStore]:
    agent = await create_test_agent(client, name=agent_name)
    resp = await client.post(
        "/v1/sessions",
        json={"agent_id": agent["id"], "cost_control_mode_override": "on"},
    )
    assert resp.status_code == 201, resp.text
    parent = resp.json()
    conv_store = SqlAlchemyConversationStore(db_uri)
    child = conv_store.create_conversation(
        kind="sub_agent",
        title="reviewer:auth",
        parent_conversation_id=parent["id"],
        agent_id=agent["id"],
    )
    if child_model_override is not None:
        child = conv_store.update_conversation(child.id, model_override=child_model_override)
    return parent, child, conv_store


def _routing_decisions(conv_store: SqlAlchemyConversationStore, session_id: str) -> list[Any]:
    return [
        item
        for item in conv_store.list_items(session_id).data
        if getattr(item, "type", None) == "routing_decision"
    ]


# ── 1. Child spawn: the router wins over ``args.model`` ─────────────


async def test_router_overrides_llm_supplied_child_model(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    _parent, child, conv_store = await _parent_and_child(
        client,
        db_uri,
        agent_name="routing-precedence",
        child_model_override=LLM_PICKED_MODEL,
    )
    caps = _FakeCaps(
        routing_client=_FakeRoutingClient(
            RoutingResult(model=ROUTED_MODEL, rationale="deep refactor", harness="claude_code")
        )
    )
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "refactor auth"}]},
    )
    with patch("omnigent.runtime._globals._caps", new=caps):
        async with _echo_runner_client() as runner_client:
            await orchestration_module._forward_event_to_runner(
                child.id,
                child,
                body,
                conv_store,
                runner_client,
            )

    refreshed = conv_store.get_conversation(child.id)
    assert refreshed is not None
    # The router's pick replaced the orchestrator's ``args.model``.
    assert refreshed.model_override == ROUTED_MODEL
    decisions = _routing_decisions(conv_store, child.id)
    assert len(decisions) == 1
    data = decisions[0].data
    assert data.model == ROUTED_MODEL
    assert data.scope == "child_session"
    assert data.attempted_override == LLM_PICKED_MODEL
    assert data.decision_id
    # The decision is joined onto the child row for the sidebar.
    assert refreshed.labels.get(ROUTING_DECISION_LABEL_KEY) == data.decision_id


async def test_routed_model_publishes_session_model_event(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """A routed pick pushes ``session.model`` so open web pickers update live.

    Routing persists ``model_override`` server-side; without the SSE the
    dropdown keeps showing the launch model until a reload (the PATCH and
    ``external_model_change`` paths publish it — routing must too).
    """
    _parent, child, conv_store = await _parent_and_child(
        client,
        db_uri,
        agent_name="routing-sse",
        child_model_override=LLM_PICKED_MODEL,
    )
    caps = _FakeCaps(
        routing_client=_FakeRoutingClient(
            RoutingResult(model=ROUTED_MODEL, rationale="deep refactor", harness="claude_code")
        )
    )
    body = SessionEventInput(
        type="message",
        data={"role": "user", "content": [{"type": "input_text", "text": "refactor auth"}]},
    )
    published: list[tuple[str, dict[str, Any]]] = []
    with (
        patch("omnigent.runtime._globals._caps", new=caps),
        patch.object(
            orchestration_module.session_stream,
            "publish",
            side_effect=lambda sid, payload: published.append((sid, payload)),
        ),
    ):
        async with _echo_runner_client() as runner_client:
            await orchestration_module._forward_event_to_runner(
                child.id,
                child,
                body,
                conv_store,
                runner_client,
            )

    model_events = [
        payload
        for sid, payload in published
        if sid == child.id and payload.get("type") == "session.model"
    ]
    assert len(model_events) == 1
    assert model_events[0]["model"] == ROUTED_MODEL


# ── 2. Native-subagent relay: transcript item + child join ──────────


async def test_native_subagent_relay_persists_decision_and_joins_child_row(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    parent, child, conv_store = await _parent_and_child(
        client, db_uri, agent_name="routing-native-subagent"
    )
    caps = _FakeCaps(
        routing_client=_FakeRoutingClient(
            RoutingResult(model=ROUTED_MODEL, rationale="deep reasoning", harness="claude_code")
        )
    )
    with patch("omnigent.runtime._globals._caps", new=caps):
        resp = await client.post(
            f"/v1/sessions/{parent['id']}/hooks/route-subagent",
            json={
                "harness": "claude-native",
                "task_name": "code-reviewer",
                "prompt": "review the auth module",
                "parent_model": LLM_PICKED_MODEL,
            },
        )
    assert resp.status_code == 200, resp.text
    decision = resp.json()
    assert decision["action"] == "rewrite"
    assert decision["model"] == ROUTED_MODEL

    decisions = _routing_decisions(conv_store, parent["id"])
    assert len(decisions) == 1
    data = decisions[0].data
    assert data.scope == "native_subagent"
    assert data.decision_id == decision["decision_id"]
    assert data.harness == "claude-native"

    # A routed child row surfaces both fields to the sidebar.
    conv_store.update_conversation(child.id, model_override=ROUTED_MODEL)
    conv_store.set_labels(child.id, {ROUTING_DECISION_LABEL_KEY: decision["decision_id"]})
    rows = await client.get(f"/v1/sessions/{parent['id']}/child_sessions")
    assert rows.status_code == 200
    row = rows.json()["data"][0]
    assert row["routed_model"] == ROUTED_MODEL
    assert row["routing_decision_id"] == decision["decision_id"]


# ── 3. Fail mode on a dead router ──────────────────────────────────


@pytest.mark.parametrize(
    ("fail_mode", "expected_action"),
    [("open", "allow"), ("closed", "deny")],
)
async def test_dead_router_honours_subagent_fail_mode(
    client: httpx.AsyncClient,
    db_uri: str,
    fail_mode: str,
    expected_action: str,
) -> None:
    agent = await create_test_agent(client, name=f"routing-failmode-{fail_mode}")
    resp = await client.post(
        "/v1/sessions",
        json={"agent_id": agent["id"], "cost_control_mode_override": "on"},
    )
    assert resp.status_code == 201, resp.text
    session_id = resp.json()["id"]
    caps = _FakeCaps(
        routing_client=_FakeRoutingClient(None, error=RuntimeError("router down")),
        routing_settings=RoutingSettings(subagent_fail_mode=fail_mode),  # type: ignore[arg-type]
    )
    with patch("omnigent.runtime._globals._caps", new=caps):
        route = await client.post(
            f"/v1/sessions/{session_id}/hooks/route-subagent",
            json={"harness": "codex-native", "task_name": "explore"},
        )
    assert route.status_code == 200, route.text
    assert route.json()["action"] == expected_action

    # Both fail modes still leave a decision item behind.
    conv_store = SqlAlchemyConversationStore(db_uri)
    assert len(_routing_decisions(conv_store, session_id)) == 1


# ── 4. Decision cache ──────────────────────────────────────────────


async def test_identical_spawns_hit_the_router_once(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    agent = await create_test_agent(client, name="routing-cache")
    resp = await client.post(
        "/v1/sessions",
        json={"agent_id": agent["id"], "cost_control_mode_override": "on"},
    )
    assert resp.status_code == 201, resp.text
    session_id = resp.json()["id"]
    routing_client = _FakeRoutingClient(
        RoutingResult(model=ROUTED_MODEL, rationale="deep reasoning", harness="claude_code")
    )
    payload = {
        "harness": "claude-native",
        "task_name": "code-reviewer",
        "prompt": "review the auth module",
        "parent_model": LLM_PICKED_MODEL,
    }
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        first = await client.post(f"/v1/sessions/{session_id}/hooks/route-subagent", json=payload)
        second = await client.post(f"/v1/sessions/{session_id}/hooks/route-subagent", json=payload)

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["decision_id"] == second.json()["decision_id"]
    assert len(routing_client.calls) == 1


# ── 5. Per-session subagent-routing gate ───────────────────────────


SPAWN_PAYLOAD = {
    "harness": "claude-native",
    "task_name": "code-reviewer",
    "prompt": "review the auth module",
    "parent_model": LLM_PICKED_MODEL,
}
DISABLED_RATIONALE = "subagent routing disabled for this session"


async def _session_with_routing_flags(
    client: httpx.AsyncClient,
    *,
    agent_name: str,
    cost_control: str | None = None,
    subagent_routing: str | None = None,
) -> str:
    agent = await create_test_agent(client, name=agent_name)
    body: dict[str, Any] = {"agent_id": agent["id"]}
    if cost_control is not None:
        body["cost_control_mode_override"] = cost_control
    if subagent_routing is not None:
        body["subagent_routing_override"] = subagent_routing
    resp = await client.post("/v1/sessions", json=body)
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


@pytest.mark.parametrize(
    ("case", "cost_control", "subagent_routing", "routed"),
    [
        ("inherit-on", "on", None, True),
        ("inherit-off", None, None, False),
        ("override-off", "on", "off", False),
        ("override-on", None, "on", True),
    ],
)
async def test_subagent_gate_follows_the_session_setting(
    client: httpx.AsyncClient,
    db_uri: str,
    case: str,
    cost_control: str | None,
    subagent_routing: str | None,
    routed: bool,
) -> None:
    session_id = await _session_with_routing_flags(
        client,
        agent_name=f"routing-gate-{case}",
        cost_control=cost_control,
        subagent_routing=subagent_routing,
    )
    routing_client = _FakeRoutingClient(
        RoutingResult(model=ROUTED_MODEL, rationale="deep reasoning", harness="claude_code")
    )
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        resp = await client.post(
            f"/v1/sessions/{session_id}/hooks/route-subagent",
            json=SPAWN_PAYLOAD,
        )
    assert resp.status_code == 200, resp.text
    decision = resp.json()
    conv_store = SqlAlchemyConversationStore(db_uri)
    if routed:
        assert decision["action"] == "rewrite"
        assert decision["model"] == ROUTED_MODEL
        assert len(routing_client.calls) == 1
        assert len(_routing_decisions(conv_store, session_id)) == 1
    else:
        # Allowed unchanged, router untouched, and no transcript spam.
        assert decision["action"] == "allow"
        assert decision["model"] is None
        assert decision["rationale"] == DISABLED_RATIONALE
        assert routing_client.calls == []
        assert _routing_decisions(conv_store, session_id) == []


async def test_child_inherits_the_parents_routing_state(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    _parent, child, conv_store = await _parent_and_child(
        client, db_uri, agent_name="routing-gate-child-inherit"
    )
    routing_client = _FakeRoutingClient(
        RoutingResult(model=ROUTED_MODEL, rationale="deep reasoning", harness="claude_code")
    )
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        resp = await client.post(
            f"/v1/sessions/{child.id}/hooks/route-subagent",
            json=SPAWN_PAYLOAD,
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["action"] == "rewrite"
    assert len(routing_client.calls) == 1

    # The child's own "off" wins over the parent's routed state.
    conv_store.update_conversation(child.id, subagent_routing_override="off")
    clear_cache(child.id)
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        second = await client.post(
            f"/v1/sessions/{child.id}/hooks/route-subagent",
            json=SPAWN_PAYLOAD,
        )
    assert second.status_code == 200, second.text
    assert second.json()["action"] == "allow"
    assert second.json()["rationale"] == DISABLED_RATIONALE
    assert len(routing_client.calls) == 1


async def test_patch_round_trips_the_subagent_setting_and_rejects_junk(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    session_id = await _session_with_routing_flags(client, agent_name="routing-gate-patch")

    for value in ("on", "off"):
        patched = await client.patch(
            f"/v1/sessions/{session_id}",
            json={"subagent_routing_override": value},
        )
        assert patched.status_code == 200, patched.text
        assert patched.json()["subagent_routing_override"] == value
        snapshot = await client.get(f"/v1/sessions/{session_id}")
        assert snapshot.json()["subagent_routing_override"] == value

    # Explicit null clears it back to inheriting the main routing state.
    cleared = await client.patch(
        f"/v1/sessions/{session_id}",
        json={"subagent_routing_override": None},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["subagent_routing_override"] is None

    rejected = await client.patch(
        f"/v1/sessions/{session_id}",
        json={"subagent_routing_override": "maybe"},
    )
    assert rejected.status_code == 400, rejected.text
    assert "subagent_routing_override" in rejected.text


async def test_toggling_the_setting_on_mid_session_routes_the_next_spawn(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    session_id = await _session_with_routing_flags(client, agent_name="routing-gate-midsession")
    routing_client = _FakeRoutingClient(
        RoutingResult(model=ROUTED_MODEL, rationale="deep reasoning", harness="claude_code")
    )
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        before = await client.post(
            f"/v1/sessions/{session_id}/hooks/route-subagent",
            json=SPAWN_PAYLOAD,
        )
        assert before.json()["action"] == "allow"
        patched = await client.patch(
            f"/v1/sessions/{session_id}",
            json={"subagent_routing_override": "on"},
        )
        assert patched.status_code == 200, patched.text
        after = await client.post(
            f"/v1/sessions/{session_id}/hooks/route-subagent",
            json=SPAWN_PAYLOAD,
        )
    assert after.json()["action"] == "rewrite"
    assert after.json()["model"] == ROUTED_MODEL


# ── 6. Harness-family constraint on the candidate set ──────────────


@pytest.mark.parametrize(
    ("harness", "counterpart", "picked"),
    [
        ("claude-native", "codex-native", ROUTED_MODEL),
        ("codex-native", "claude-native", GPT_MODEL),
    ],
)
async def test_pinned_session_is_offered_only_its_own_family(
    client: httpx.AsyncClient,
    db_uri: str,
    harness: str,
    counterpart: str,
    picked: str,
) -> None:
    session_id = await _session_with_routing_flags(
        client,
        agent_name=f"routing-family-{harness}",
        subagent_routing="on",
    )
    routing_client = _FakeRoutingClient(RoutingResult(model=picked, rationale="deep reasoning"))
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        resp = await client.post(
            f"/v1/sessions/{session_id}/hooks/route-subagent",
            json={**SPAWN_PAYLOAD, "harness": harness},
        )
    assert resp.status_code == 200, resp.text
    assert set(routing_client.offered[0]) == {harness}
    assert counterpart not in routing_client.offered[0]
    assert resp.json()["action"] == "rewrite"


async def test_auto_session_and_its_children_keep_cross_harness_picks(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    agent = await create_test_agent(client, name="routing-family-auto")
    created = await client.post(
        "/v1/sessions",
        json={
            "agent_id": agent["id"],
            "cost_control_mode_override": "on",
            "harness_override": "auto",
        },
    )
    assert created.status_code == 201, created.text
    session_id = created.json()["id"]
    conv_store = SqlAlchemyConversationStore(db_uri)
    child = conv_store.create_conversation(
        kind="sub_agent",
        title="reviewer:auth",
        parent_conversation_id=session_id,
        agent_id=agent["id"],
    )

    routing_client = _FakeRoutingClient(
        RoutingResult(model=GPT_MODEL, rationale="narrow change", harness="codex")
    )
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        resp = await client.post(
            f"/v1/sessions/{session_id}/hooks/route-subagent",
            json=SPAWN_PAYLOAD,
        )
        child_resp = await client.post(
            f"/v1/sessions/{child.id}/hooks/route-subagent",
            json=SPAWN_PAYLOAD,
        )
    assert resp.status_code == 200, resp.text
    assert set(routing_client.offered[0]) == {"claude-native", "codex-native"}
    # Auto keeps the cross-family escape hatch: a Codex pick redirects.
    assert resp.json()["action"] == "redirect"
    assert resp.json()["harness"] == "codex-native"
    # The child of an auto session inherits the cross-harness allowance.
    assert child_resp.status_code == 200, child_resp.text
    assert set(routing_client.offered[1]) == {"claude-native", "codex-native"}


# ── Session warnings channel ───────────────────────────────────────


async def test_session_warning_event_lands_on_the_snapshot(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    agent = await create_test_agent(client, name="routing-warning")
    resp = await client.post("/v1/sessions", json={"agent_id": agent["id"]})
    assert resp.status_code == 201, resp.text
    session_id = resp.json()["id"]

    posted = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "type": "external_session_warning",
            "data": {
                "warnings": [
                    {
                        "code": "subagent_routing_unenforced",
                        "harness": "codex-native",
                        "reason": "SessionStart canary did not fire",
                    }
                ]
            },
        },
    )
    assert posted.status_code in (200, 201, 202), posted.text

    snapshot = await client.get(f"/v1/sessions/{session_id}")
    assert snapshot.status_code == 200
    warnings = snapshot.json()["warnings"]
    assert warnings == [
        {
            "code": "subagent_routing_unenforced",
            "harness": "codex-native",
            "reason": "SessionStart canary did not fire",
        }
    ]
