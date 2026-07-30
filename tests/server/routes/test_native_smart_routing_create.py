"""Create-time Smart Routing onto a native terminal harness.

The landing screen's top-level "Smart Routing" harness sends
``harness_override: "auto"`` with a native wrapper agent as a placeholder and
the first message as ``smart_routing_message``. A native terminal launches with
the session row, so the harness is decided during the create — not on the first
message event the way the bundle-agent auto path does — and the session is
rebound to the wrapper the router picked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from omnigent.db.utils import generate_agent_id
from omnigent.runner.subagent_routing import AUTO_HARNESS_LABEL_KEY, clear_cache
from omnigent.server.routes._sessions.orchestration import _installed_native_harnesses
from omnigent.server.smart_routing import (
    AUTO_NATIVE_ROUTING_HARNESSES,
    RoutingResult,
    RoutingSettings,
    route_session_harness,
)
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.host_store import Host
from tests.server.helpers import create_test_agent

CLAUDE_MODEL = "databricks-claude-opus-4-8"
GPT_MODEL = "databricks-gpt-5-5"
ROUTING_MESSAGE = "refactor the auth module and add tests"

SPAWN_PAYLOAD = {
    "harness": "claude-native",
    "task_name": "code-reviewer",
    "prompt": "review the auth module",
    "parent_model": CLAUDE_MODEL,
}


class _FakeRoutingClient:
    """Returns a canned verdict and records what it was offered."""

    def __init__(self, result: RoutingResult | None) -> None:
        self._result = result
        self.last_error: str | None = None
        self.offered: list[dict[str, list[str]]] = []

    async def route(
        self,
        message: str,
        available_models: dict[str, list[str]],
    ) -> RoutingResult | None:
        self.offered.append(dict(available_models))
        return self._result


@dataclass
class _FakeCaps:
    routing_client: Any = None  # type: ignore[explicit-any]
    routing_settings: RoutingSettings = field(default_factory=RoutingSettings)


@pytest.fixture(autouse=True)
def _clear_decision_cache() -> Any:  # type: ignore[explicit-any]
    clear_cache()
    yield
    clear_cache()


async def _native_wrappers(client: httpx.AsyncClient, db_uri: str) -> dict[str, str]:
    """Register both native wrapper agents and return ``harness -> agent_id``.

    The server resolves a routed harness to its wrapper by agent NAME via
    ``get_by_name``, which only sees TEMPLATE agents — so the wrappers are
    seeded as templates over a real uploaded bundle (the app fixture skips the
    lifespan that would seed the builtins).

    :param client: Test HTTP client.
    :param db_uri: Database URI for a direct store handle.
    :returns: ``{"claude-native": id, "codex-native": id}``.
    """
    source = await create_test_agent(client, name="native-wrapper-bundle-source")
    store = SqlAlchemyAgentStore(db_uri)
    bundle = store.get(str(source["id"]))
    assert bundle is not None
    wrappers: dict[str, str] = {}
    for harness, agent_name in (
        ("claude-native", "claude-native-ui"),
        ("codex-native", "codex-native-ui"),
    ):
        agent_id = generate_agent_id()
        store.create(agent_id, name=agent_name, bundle_location=bundle.bundle_location)
        wrappers[harness] = agent_id
    return wrappers


async def _create_smart_routing_session(
    client: httpx.AsyncClient,
    wrappers: dict[str, str],
    routing_client: _FakeRoutingClient | None,
) -> httpx.Response:
    """POST the landing screen's Smart Routing create payload.

    :param client: Test HTTP client.
    :param wrappers: Registered wrapper ids from :func:`_native_wrappers`.
    :param routing_client: Stub router, or ``None`` to leave routing unconfigured.
    :returns: The raw create response.
    """
    body = {
        # The Claude wrapper is the client-side placeholder; the server rebinds.
        "agent_id": wrappers["claude-native"],
        "harness_override": "auto",
        "cost_control_mode_override": "on",
        "smart_routing_message": ROUTING_MESSAGE,
    }
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        return await client.post("/v1/sessions", json=body)


@pytest.mark.parametrize(
    ("picked_model", "expected_harness"),
    [
        (CLAUDE_MODEL, "claude-native"),
        (GPT_MODEL, "codex-native"),
    ],
)
@pytest.mark.asyncio
async def test_create_binds_the_wrapper_the_router_picked(
    client: httpx.AsyncClient,
    db_uri: str,
    picked_model: str,
    expected_harness: str,
) -> None:
    wrappers = await _native_wrappers(client, db_uri)
    routing_client = _FakeRoutingClient(RoutingResult(model=picked_model, rationale="sized task"))
    created = await _create_smart_routing_session(client, wrappers, routing_client)
    assert created.status_code == 201, created.text

    conv = SqlAlchemyConversationStore(db_uri).get_conversation(created.json()["id"])
    assert conv is not None
    assert conv.agent_id == wrappers[expected_harness]
    # The wrapper's terminal launches with the routed model baked in.
    assert conv.model_override == picked_model
    # No sentinel survives: a native wrapper rejects harness_override, and a
    # leftover "auto" would re-route an already-running terminal.
    assert conv.harness_override is None
    # The auto marker is what keeps subagents cross-harness-eligible.
    assert conv.labels.get(AUTO_HARNESS_LABEL_KEY) == "1"


@pytest.mark.asyncio
async def test_router_is_offered_both_native_families(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    wrappers = await _native_wrappers(client, db_uri)
    routing_client = _FakeRoutingClient(RoutingResult(model=GPT_MODEL, rationale="narrow change"))
    created = await _create_smart_routing_session(client, wrappers, routing_client)
    assert created.status_code == 201, created.text
    assert set(routing_client.offered[0]) == {"claude-native", "codex-native"}


@pytest.mark.asyncio
async def test_routed_wrapper_gets_terminal_first_labels(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    wrappers = await _native_wrappers(client, db_uri)
    routing_client = _FakeRoutingClient(RoutingResult(model=GPT_MODEL, rationale="narrow change"))
    created = await _create_smart_routing_session(client, wrappers, routing_client)
    assert created.status_code == 201, created.text

    conv = SqlAlchemyConversationStore(db_uri).get_conversation(created.json()["id"])
    assert conv is not None
    # The routed wrapper's own presentation labels, not the placeholder's.
    assert conv.labels.get("omnigent.ui") == "terminal"
    assert conv.labels.get("omnigent.wrapper") == "codex-native-ui"


@pytest.mark.asyncio
async def test_create_falls_back_to_a_native_cli_when_routing_is_unavailable(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    wrappers = await _native_wrappers(client, db_uri)
    created = await _create_smart_routing_session(client, wrappers, None)
    assert created.status_code == 201, created.text

    conv = SqlAlchemyConversationStore(db_uri).get_conversation(created.json()["id"])
    assert conv is not None
    # Still lands on a terminal, with the CLI's own default model.
    assert conv.agent_id == wrappers["claude-native"]
    assert conv.model_override is None
    assert conv.harness_override is None
    assert conv.labels.get(AUTO_HARNESS_LABEL_KEY) == "1"


@pytest.mark.asyncio
async def test_smart_routing_session_keeps_cross_harness_subagents(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    wrappers = await _native_wrappers(client, db_uri)
    routing_client = _FakeRoutingClient(RoutingResult(model=CLAUDE_MODEL, rationale="sized task"))
    created = await _create_smart_routing_session(client, wrappers, routing_client)
    assert created.status_code == 201, created.text
    session_id = created.json()["id"]

    spawn_router = _FakeRoutingClient(
        RoutingResult(model=GPT_MODEL, rationale="narrow change", harness="codex")
    )
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=spawn_router)):
        resp = await client.post(
            f"/v1/sessions/{session_id}/hooks/route-subagent",
            json=SPAWN_PAYLOAD,
        )
    assert resp.status_code == 200, resp.text
    # The auto label survives the create, so the spawn is offered both families
    # and may leave the session's own harness family.
    assert set(spawn_router.offered[0]) == {"claude-native", "codex-native"}
    assert resp.json()["harness"] == "codex-native"


@pytest.mark.asyncio
async def test_bundle_agent_auto_path_is_unchanged(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    agent = await create_test_agent(client, name="smart-routing-bundle-agent")
    routing_client = _FakeRoutingClient(RoutingResult(model=CLAUDE_MODEL, rationale="sized task"))
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        created = await client.post(
            "/v1/sessions",
            json={
                "agent_id": agent["id"],
                "harness_override": "auto",
                "cost_control_mode_override": "on",
                "smart_routing_message": ROUTING_MESSAGE,
            },
        )
    assert created.status_code == 201, created.text

    conv = SqlAlchemyConversationStore(db_uri).get_conversation(created.json()["id"])
    assert conv is not None
    # A non-native agent keeps the sentinel for first-message resolution, and
    # the create must not have called the router.
    assert conv.harness_override == "auto"
    assert conv.model_override is None
    assert conv.agent_id == agent["id"]
    assert conv.labels.get(AUTO_HARNESS_LABEL_KEY) == "1"
    assert routing_client.offered == []


@pytest.mark.asyncio
async def test_native_candidates_impose_no_family_constraint() -> None:
    # The candidate override and the parent-family filter are orthogonal: an
    # auto session passes no allowed_family, so both native families reach the
    # router even though the family filter is applied to the same tuple.
    routing_client = _FakeRoutingClient(RoutingResult(model=GPT_MODEL, rationale="narrow change"))
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        harness, model, _verdict, error = await route_session_harness(
            ROUTING_MESSAGE,
            harness_candidates=AUTO_NATIVE_ROUTING_HARNESSES,
        )
    assert error is None
    assert set(routing_client.offered[0]) == {"claude-native", "codex-native"}
    assert harness == "codex-native"
    assert model == GPT_MODEL


@pytest.mark.asyncio
async def test_native_candidates_still_honor_an_explicit_family() -> None:
    # A family constraint (a child of a pinned parent) narrows the same tuple,
    # so the two knobs compose rather than fight.
    routing_client = _FakeRoutingClient(
        RoutingResult(model=CLAUDE_MODEL, rationale="deep reasoning")
    )
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        harness, _model, _verdict, error = await route_session_harness(
            ROUTING_MESSAGE,
            harness_candidates=AUTO_NATIVE_ROUTING_HARNESSES,
            allowed_family="claude",
        )
    assert error is None
    assert set(routing_client.offered[0]) == {"claude-native"}
    assert harness == "claude-native"


@pytest.mark.asyncio
async def test_no_installed_native_candidates_reports_the_standard_error() -> None:
    routing_client = _FakeRoutingClient(RoutingResult(model=GPT_MODEL, rationale="narrow change"))
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=routing_client)):
        harness, model, verdict, error = await route_session_harness(
            ROUTING_MESSAGE,
            harness_candidates=(),
        )
    assert (harness, model, verdict) == (None, None, None)
    assert error == "No routable harnesses are available on this runner."
    assert routing_client.offered == []


def _host(readiness: dict[str, Any] | None) -> Host:  # type: ignore[explicit-any]
    return Host(
        host_id="host_1",
        name="dev",
        user_id="alice@example.com",
        status="online",
        created_at=0,
        updated_at=0,
        configured_harnesses=readiness,
    )


@pytest.mark.parametrize(
    ("readiness", "expected"),
    [
        ({"claude-native": True, "codex-native": True}, ["claude-native", "codex-native"]),
        ({"claude-native": True, "codex-native": "binary-missing"}, ["claude-native"]),
        ({"claude-native": "needs-auth", "codex-native": True}, ["codex-native"]),
        ({"claude-native": "version-too-low", "codex-native": False}, []),
        # An unreported harness can't be assumed installed.
        ({"claude-native": True}, ["claude-native"]),
        # Fails open: a host with no readiness map doesn't disable routing.
        (None, ["claude-native", "codex-native"]),
    ],
)
def test_installed_native_harnesses_follows_host_readiness(
    readiness: dict[str, Any] | None,  # type: ignore[explicit-any]
    expected: list[str],
) -> None:
    assert _installed_native_harnesses(_host(readiness)) == expected


def test_installed_native_harnesses_without_a_host() -> None:
    assert _installed_native_harnesses(None) == ["claude-native", "codex-native"]
