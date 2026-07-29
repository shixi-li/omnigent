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


class _FakeRoutingClient:
    """Returns a canned verdict and counts calls."""

    def __init__(self, result: RoutingResult | None, *, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.last_error: str | None = None
        self.calls: list[str] = []

    async def route(
        self,
        message: str,
        available_models: dict[str, list[str]],
    ) -> RoutingResult | None:
        self.calls.append(message)
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
