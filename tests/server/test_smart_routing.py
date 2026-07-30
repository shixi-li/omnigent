"""Tests for the server-side intelligent model routing module.

Covers model inference, the RoutingClient protocol, the default
LLMRoutingClient, and the public ``route_turn`` entry point.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnigent.server.smart_routing import (
    _AUTO_ROUTING_HARNESSES,
    LLMRoutingClient,
    RoutingResult,
    _build_rubric,
    fetch_runner_models,
    infer_models,
    route_session_harness,
    route_turn,
)

# ── Stubs ───────────────────────────────────────────────────────────


@dataclass
class _FakeOutputText:
    text: str
    type: str = "output_text"


@dataclass
class _FakeMessageOutput:
    content: list[_FakeOutputText]
    type: str = "message"


@dataclass
class _FakeResponse:
    """Minimal stub matching omnigent.llms.types.Response."""

    output: list[_FakeMessageOutput]


class _FakeLLMClient:
    """Fake PolicyLLMClient that returns a canned verdict."""

    def __init__(self, verdict: dict[str, Any]) -> None:
        self._verdict = verdict

    async def create(self, **kwargs: Any) -> _FakeResponse:
        text = json.dumps(self._verdict)
        return _FakeResponse(
            output=[_FakeMessageOutput(content=[_FakeOutputText(text=text)])],
        )


class _FakeRoutingClient:
    """Stub RoutingClient for route_turn integration tests."""

    def __init__(self, result: RoutingResult | None) -> None:
        self._result = result
        self.offered: list[dict[str, list[str]]] = []

    async def route(
        self, message: str, available_models: dict[str, list[str]]
    ) -> RoutingResult | None:
        del message
        self.offered.append(dict(available_models))
        return self._result


# ── infer_models ────────────────────────────────────────────────────


def test_infer_models_claude_sdk() -> None:
    """claude-sdk returns the claude model list."""
    models = infer_models("claude-sdk")
    assert models is not None
    assert any("haiku" in m for m in models)
    assert any("opus" in m for m in models)
    # Ordered cheapest → most powerful
    haiku_idx = next(i for i, m in enumerate(models) if "haiku" in m)
    opus_idx = next(i for i, m in enumerate(models) if "opus" in m)
    assert haiku_idx < opus_idx


def test_infer_models_native_harnesses() -> None:
    assert infer_models("claude-native") is not None
    assert infer_models("codex-native") is not None


def test_infer_models_codex() -> None:
    models = infer_models("codex")
    assert models is not None
    assert any("gpt" in m for m in models)


def test_infer_models_openai_agents() -> None:
    assert infer_models("openai-agents") is not None


def test_infer_models_pi() -> None:
    """pi is multi-model — both Claude and GPT."""
    models = infer_models("pi")
    assert models is not None
    assert any("haiku" in m for m in models)
    assert any("gpt" in m for m in models)


def test_model_lists_cover_current_claude_generations() -> None:
    """A stale list turns a live model into an unservable arm (sonnet-5 → haiku)."""
    models = infer_models("claude-sdk")
    assert models is not None
    assert "databricks-claude-sonnet-5" in models


def test_model_lists_ordered_by_capability_within_each_family() -> None:
    """Curated order must agree with the capability ranking the seam uses."""
    from omnigent.server.smart_routing import MODEL_LISTS, _capability_key, _model_family

    for family, models in MODEL_LISTS.items():
        for vendor in {_model_family(m) for m in models}:
            same_vendor = [m for m in models if _model_family(m) == vendor]
            keys = [_capability_key(m) for m in same_vendor]
            assert keys == sorted(keys), f"{family}/{vendor} is not cheapest → most capable"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("databricks-claude-opus-4-8", "claude"),
        ("databricks-gpt-5-5", "gpt"),
        ("gpt-5-6-sol", "gpt"),
        # The router's glm arm and its servable spellings must all read as
        # the codex family, or a pick lands on the wrong harness.
        ("glm-5-2", "gpt"),
        ("databricks-glm-5-2", "gpt"),
        ("system.ai.glm-5-2", "gpt"),
        ("databricks-kimi-k2-6", "gpt"),
        ("databricks-meta-llama-3.3-70b-instruct", "other"),
    ],
)
def test_model_family_agrees_with_the_shared_token_rule(model: str, expected: str) -> None:
    """This file's family view must match the dispatch gate's.

    :param model: Model id under test.
    :param expected: The family it must land in.
    """
    from omnigent.model_override import model_family_mismatch
    from omnigent.server.smart_routing import _model_family

    assert _model_family(model) == expected
    if expected == "gpt":
        assert model_family_mismatch("codex", model) is None


def test_catalog_models_for_harness_matches_worker_rows() -> None:
    from omnigent.server.smart_routing import catalog_models_for_harness

    catalog = {
        "self": ["databricks-claude-sonnet-5"],
        "codex": ["databricks-gpt-5-5"],
    }
    # A worker row for the counterpart family, matched by family not id.
    assert catalog_models_for_harness(catalog, "codex-native") == ["databricks-gpt-5-5"]
    # "self" only counts for the session's own harness.
    assert catalog_models_for_harness(catalog, "claude-native") is None
    assert catalog_models_for_harness(catalog, "claude-native", allow_self=True) == [
        "databricks-claude-sonnet-5"
    ]
    assert catalog_models_for_harness(None, "claude-native", allow_self=True) is None


def test_infer_models_unknown_harness() -> None:
    assert infer_models("cursor") is None
    assert infer_models("antigravity") is None
    assert infer_models(None) is None


# ── _build_rubric ───────────────────────────────────────────────────


def test_build_rubric_includes_all_models() -> None:
    available = {
        "claude-sdk": ["databricks-claude-haiku-4-5", "databricks-claude-opus-4-8"],
    }
    rubric = _build_rubric(available)
    assert "databricks-claude-haiku-4-5" in rubric
    assert "databricks-claude-opus-4-8" in rubric
    assert "strict JSON" in rubric
    assert "haiku" in rubric and "opus" in rubric


def test_build_rubric_shows_harness_names() -> None:
    available = {
        "claude-sdk": ["databricks-claude-haiku-4-5"],
        "codex": ["databricks-gpt-5-4-nano"],
    }
    rubric = _build_rubric(available)
    assert "claude-sdk" in rubric
    assert "codex" in rubric
    assert "databricks-claude-haiku-4-5" in rubric
    assert "databricks-gpt-5-4-nano" in rubric


# ── LLMRoutingClient ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_routing_client_returns_result() -> None:
    verdict = {
        "harness": "claude-sdk",
        "model": "databricks-claude-opus-4-8",
        "rationale": "hard refactor",
    }
    client = LLMRoutingClient(_FakeLLMClient(verdict))
    models = infer_models("claude-sdk")
    assert models is not None
    result = await client.route("refactor auth", {"claude-sdk": models})
    assert result is not None
    assert result.model == "databricks-claude-opus-4-8"
    assert result.rationale == "hard refactor"
    assert result.harness == "claude-sdk"


@pytest.mark.asyncio
async def test_llm_routing_client_harness_mismatch_re_resolves() -> None:
    """If the judge picks a harness that doesn't own the model, fall back."""
    claude_models = infer_models("claude-sdk")
    assert claude_models is not None
    verdict = {
        "harness": "codex",  # codex doesn't have claude models
        "model": "databricks-claude-opus-4-8",
        "rationale": "deep reasoning",
    }
    client = LLMRoutingClient(_FakeLLMClient(verdict))
    result = await client.route(
        "hard task", {"claude-sdk": claude_models, "codex": ["databricks-gpt-5-4"]}
    )
    assert result is not None
    assert result.model == "databricks-claude-opus-4-8"
    # harness re-resolved to the one that owns the model
    assert result.harness == "claude-sdk"


@pytest.mark.asyncio
async def test_llm_routing_client_unknown_harness_re_resolves() -> None:
    """If the judge returns an unrecognised harness, fall back to model ownership."""
    models = infer_models("claude-sdk")
    assert models is not None
    verdict = {
        "harness": "hallucinated-harness",
        "model": "databricks-claude-haiku-4-5",
        "rationale": "simple task",
    }
    client = LLMRoutingClient(_FakeLLMClient(verdict))
    result = await client.route("hello", {"claude-sdk": models})
    assert result is not None
    assert result.model == "databricks-claude-haiku-4-5"
    assert result.harness == "claude-sdk"


@pytest.mark.asyncio
async def test_llm_routing_client_clamps_hallucinated_model() -> None:
    verdict = {"harness": "claude-sdk", "model": "hallucinated-model", "rationale": "hard"}
    client = LLMRoutingClient(_FakeLLMClient(verdict))
    models = infer_models("claude-sdk")
    assert models is not None
    result = await client.route("hard task", {"claude-sdk": models})
    assert result is not None
    assert result.model == models[0]  # clamped to cheapest


@pytest.mark.asyncio
async def test_llm_routing_client_rejects_empty_model() -> None:
    verdict = {"harness": "claude-sdk", "model": "", "rationale": "x"}
    client = LLMRoutingClient(_FakeLLMClient(verdict))
    models = infer_models("claude-sdk")
    assert models is not None
    result = await client.route("hello", {"claude-sdk": models})
    assert result is None


@pytest.mark.asyncio
async def test_llm_routing_client_returns_none_on_error() -> None:
    class _BrokenLLM:
        async def create(self, **kwargs: Any) -> None:
            raise TypeError("boom")

    client = LLMRoutingClient(_BrokenLLM())
    models = infer_models("claude-sdk")
    assert models is not None
    result = await client.route("hello", {"claude-sdk": models})
    assert result is None


# ── fetch_runner_models ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_runner_models_parses_catalog() -> None:
    catalog_payload = {
        "workers": {
            "self": {
                "source": "catalog",
                "verified": True,
                "models": [
                    {"id": "databricks-claude-haiku-4-5", "family": "claude"},
                    {"id": "databricks-claude-opus-4-8", "family": "claude"},
                ],
                "note": "",
            },
            "claude_code": {
                "source": "catalog",
                "verified": True,
                "models": [
                    {"id": "databricks-claude-haiku-4-5", "family": "claude"},
                    {"id": "databricks-claude-sonnet-4-6", "family": "claude"},
                ],
                "note": "",
            },
        }
    }
    mock_response = MagicMock()
    mock_response.json.return_value = catalog_payload
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await fetch_runner_models("conv_123", mock_client)
    assert result is not None
    assert "databricks-claude-haiku-4-5" in result["self"]
    assert "databricks-claude-opus-4-8" in result["self"]
    assert "databricks-claude-sonnet-4-6" in result["claude_code"]


@pytest.mark.asyncio
async def test_fetch_runner_models_returns_none_on_http_error() -> None:
    import httpx

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=httpx.HTTPError("connection refused"))

    result = await fetch_runner_models("conv_123", mock_client)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_runner_models_returns_none_on_empty_workers() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"workers": {}}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    result = await fetch_runner_models("conv_123", mock_client)
    assert result is None


# ── route_turn (integration) ───────────────────────────────────────


@dataclass
class _FakeCaps:
    routing_client: Any = None  # type: ignore[explicit-any]


@pytest.mark.asyncio
async def test_route_turn_uses_caps_routing_client() -> None:
    expected = RoutingResult(
        model="databricks-claude-haiku-4-5",
        rationale="trivial",
        harness="claude-sdk",
    )
    caps = _FakeCaps(routing_client=_FakeRoutingClient(expected))
    with patch(
        "omnigent.runtime._globals._caps",
        new=caps,
    ):
        model, v = await route_turn("claude-sdk", "hello")
    assert model == "databricks-claude-haiku-4-5"
    assert v is not None
    assert "tier" not in v


@pytest.mark.asyncio
async def test_route_turn_returns_none_when_no_client() -> None:
    caps = _FakeCaps(routing_client=None)
    with patch(
        "omnigent.runtime._globals._caps",
        new=caps,
    ):
        model, _v = await route_turn("claude-sdk", "hello")
    assert model is None


@pytest.mark.asyncio
async def test_route_turn_unknown_harness() -> None:
    model, _v = await route_turn("cursor", "hello")
    assert model is None
    assert _v is None


@pytest.mark.asyncio
async def test_route_turn_uses_runner_catalog_when_available() -> None:
    """route_turn uses live runner catalog instead of static table when provided."""
    expected = RoutingResult(
        model="databricks-claude-opus-4-8",
        rationale="complex task",
        harness="self",
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "workers": {
            "self": {
                "source": "catalog",
                "verified": True,
                "models": [
                    {"id": "databricks-claude-haiku-4-5"},
                    {"id": "databricks-claude-opus-4-8"},
                ],
                "note": "",
            }
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    caps = _FakeCaps(routing_client=_FakeRoutingClient(expected))
    with patch("omnigent.runtime._globals._caps", new=caps):
        model, _v = await route_turn(
            "claude-sdk",
            "complex task",
            session_id="conv_123",
            runner_client=mock_client,
        )
    assert model == "databricks-claude-opus-4-8"
    # Runner endpoint was called
    mock_client.get.assert_called_once()
    call_url = mock_client.get.call_args[0][0]
    assert "conv_123" in call_url and "models" in call_url


@pytest.mark.asyncio
async def test_route_turn_prefers_the_callers_session_vocabulary() -> None:
    """A pane can only switch onto its picker rows, so those win over the gateway."""
    expected = RoutingResult(
        model="databricks-claude-opus-5",
        rationale="complex task",
        harness="self",
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "workers": {
            "self": {
                "source": "catalog",
                "verified": True,
                "models": [{"id": "databricks-claude-opus-4-8"}],
                "note": "",
            }
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    routing_client = _FakeRoutingClient(expected)

    caps = _FakeCaps(routing_client=routing_client)
    with patch("omnigent.runtime._globals._caps", new=caps):
        model, _v = await route_turn(
            "claude-native",
            "complex task",
            session_id="conv_123",
            runner_client=mock_client,
            catalog=["databricks-claude-opus-5", "databricks-gpt-5-5"],
        )

    assert model == "databricks-claude-opus-5"
    # The wider runner catalog was never consulted, and the out-of-family id
    # in the vocabulary was still dropped.
    mock_client.get.assert_not_called()
    assert routing_client.offered == [{"claude-native": ["databricks-claude-opus-5"]}]


@pytest.mark.asyncio
async def test_route_turn_drops_out_of_family_catalog_models() -> None:
    """A native terminal's catalog can list other families; they aren't offered."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "workers": {
            "self": {
                "source": "catalog",
                "verified": True,
                "models": [
                    {"id": "databricks-gpt-5-5"},
                    {"id": "databricks-claude-opus-4-8"},
                    {"id": "databricks-gpt-5-4-mini"},
                ],
                "note": "",
            }
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    client = _FakeRoutingClient(
        RoutingResult(model="databricks-gpt-5-5", rationale="gpt task", harness="codex-native")
    )
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=client)):
        model, _v = await route_turn(
            "codex-native",
            "narrow fix",
            session_id="conv_123",
            runner_client=mock_client,
        )
    assert model == "databricks-gpt-5-5"
    assert client.offered == [{"codex-native": ["databricks-gpt-5-5", "databricks-gpt-5-4-mini"]}]


@pytest.mark.asyncio
async def test_route_turn_offers_and_applies_a_glm_pick_on_codex() -> None:
    """A codex session's GLM/Kimi endpoints are offered and a GLM pick applies.

    They ride the same Responses wire codex speaks, so the in-family filter
    must keep them; dropping them would leave the router unable to pick the
    delegate arm the workspace actually serves.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "workers": {
            "self": {
                "source": "catalog",
                "verified": True,
                "models": [
                    {"id": "databricks-gpt-5-5"},
                    {"id": "databricks-glm-5-2"},
                    {"id": "databricks-kimi-k2-6"},
                    {"id": "databricks-claude-opus-4-8"},
                ],
                "note": "",
            }
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    client = _FakeRoutingClient(
        RoutingResult(model="databricks-glm-5-2", rationale="delegate arm", harness="codex-native")
    )
    with patch("omnigent.runtime._globals._caps", new=_FakeCaps(routing_client=client)):
        model, _v = await route_turn(
            "codex-native",
            "refactor the parser",
            session_id="conv_glm",
            runner_client=mock_client,
        )
    assert client.offered == [
        {
            "codex-native": [
                "databricks-gpt-5-5",
                "databricks-glm-5-2",
                "databricks-kimi-k2-6",
            ]
        }
    ]
    assert model == "databricks-glm-5-2"


@pytest.mark.asyncio
async def test_route_turn_falls_back_to_static_when_runner_unavailable() -> None:
    """Falls back to infer_models when runner catalog fetch fails."""
    import httpx

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=httpx.HTTPError("runner down"))

    expected = RoutingResult(
        model="databricks-claude-haiku-4-5",
        rationale="simple",
        harness="claude-sdk",
    )
    caps = _FakeCaps(routing_client=_FakeRoutingClient(expected))
    with patch("omnigent.runtime._globals._caps", new=caps):
        model, _v = await route_turn(
            "claude-sdk",
            "hello",
            session_id="conv_123",
            runner_client=mock_client,
        )
    # Still routes — fell back to static infer_models
    assert model == "databricks-claude-haiku-4-5"


# ── ExternalRoutingClient ─────────────────────────────────────────────


def _patch_httpx(transport: Any) -> Any:
    """Patch httpx.AsyncClient to use a MockTransport."""
    import httpx

    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


@pytest.mark.asyncio
async def test_external_routing_client_sends_snake_case_and_parses() -> None:
    """available_models -> snake_case route_options; response -> RoutingResult."""
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "route_selection": [
                    {"route_option": {"model": "claude-opus-4-8", "harness": "claude"}}
                ],
                "rationale": "task_v0 matched rule 'bugfix_to_opus'.",
            },
        )

    client = ExternalRoutingClient(
        base_url="https://host/ai-gateway/routing/v1", router_name="task_v0"
    )
    with _patch_httpx(httpx.MockTransport(handler)):
        result = await client.route(
            "fix this code: x = y + 2",
            {"claude": ["claude-opus-4-8"], "codex": ["gpt-5-5"]},
        )

    assert result is not None
    assert result.model == "claude-opus-4-8"
    assert result.harness == "claude"
    assert result.rationale == "task_v0 matched rule 'bugfix_to_opus'."
    assert captured["url"] == "https://host/ai-gateway/routing/v1/routes:select"
    body = captured["body"]
    assert body["route_selector"]["router_name"] == "task_v0"  # snake_case
    assert body["task"]["prompt"] == "fix this code: x = y + 2"
    assert body["route_options"] == [
        {"model": "claude-opus-4-8", "harness": "claude"},
        {"model": "gpt-5-5", "harness": "codex"},
    ]


@pytest.mark.asyncio
async def test_external_routing_client_roundtrips_provider_prefix() -> None:
    """Send bare ids out; recover the exact catalog id from the bare answer.

    A Databricks catalog carries a ``databricks-`` prefix the router doesn't
    want, so we send bare ids and map the router's (bare) pick back to the
    local prefixed id the runner needs.
    """
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        # Router echoes the bare id it was given.
        return httpx.Response(
            200,
            json={"route_selection": [{"route_option": {"model": "claude-opus-4-8"}}]},
        )

    client = ExternalRoutingClient(
        base_url="https://host/v1", router_name="task_v0", model_prefixes=["databricks-"]
    )
    with _patch_httpx(httpx.MockTransport(handler)):
        result = await client.route(
            "hi", {"self": ["databricks-claude-opus-4-8", "databricks-gpt-5-5"]}
        )

    # Outbound: configured prefix stripped for the router's vocabulary.
    assert captured["body"]["route_options"] == [
        {"model": "claude-opus-4-8", "harness": "self"},
        {"model": "gpt-5-5", "harness": "self"},
    ]
    # Inbound: mapped back to the local (prefixed) catalog id.
    assert result is not None
    assert result.model == "databricks-claude-opus-4-8"


@pytest.mark.asyncio
async def test_external_routing_client_strips_first_matching_prefix() -> None:
    """With multiple prefixes, the first matching one is stripped per id."""
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"route_selection": [{"route_option": {"model": "claude-opus-4-8"}}]},
        )

    client = ExternalRoutingClient(
        base_url="https://host/v1",
        router_name="task_v0",
        model_prefixes=["databricks-", "system.ai."],
    )
    with _patch_httpx(httpx.MockTransport(handler)):
        result = await client.route(
            "hi", {"self": ["databricks-claude-opus-4-8", "system.ai.claude-sonnet-5"]}
        )

    # Each id has its own matching prefix stripped.
    assert captured["body"]["route_options"] == [
        {"model": "claude-opus-4-8", "harness": "self"},
        {"model": "claude-sonnet-5", "harness": "self"},
    ]
    assert result is not None
    assert result.model == "databricks-claude-opus-4-8"


@pytest.mark.asyncio
async def test_external_routing_client_maps_back_by_harness() -> None:
    """The same bare id under two harnesses maps back to distinct local ids.

    A Databricks-authed harness carries the ``databricks-`` prefix while a
    subscription harness (e.g. Codex) uses the bare id; both reduce to the
    same router id, so the (harness, router-id) key keeps them distinct.
    """
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        # Router picks the codex option (bare id, codex harness).
        return httpx.Response(
            200,
            json={"route_selection": [{"route_option": {"model": "gpt-5-5", "harness": "codex"}}]},
        )

    client = ExternalRoutingClient(
        base_url="https://host/v1", router_name="task_v0", model_prefixes=["databricks-"]
    )
    with _patch_httpx(httpx.MockTransport(handler)):
        result = await client.route(
            "hi",
            {"pi": ["databricks-gpt-5-5"], "codex": ["gpt-5-5"]},
        )

    # Both harnesses reduce to router id "gpt-5-5"; the pick maps back to the
    # codex local id, not pi's prefixed one.
    assert result is not None
    assert result.model == "gpt-5-5"
    assert result.harness == "codex"


@pytest.mark.asyncio
async def test_external_routing_client_no_prefix_sends_catalog_ids_verbatim() -> None:
    """With no model_prefix configured, catalog ids are sent verbatim.

    Provider-agnostic guarantee: core invents/strips nothing — even a
    ``databricks-`` id passes through untouched when unconfigured.
    """
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"route_selection": [{"route_option": {"model": "databricks-claude-opus-4-8"}}]},
        )

    client = ExternalRoutingClient(base_url="https://host/v1", router_name="task_v0")
    with _patch_httpx(httpx.MockTransport(handler)):
        result = await client.route("hi", {"self": ["databricks-claude-opus-4-8", "gpt-5-5"]})

    # Sent verbatim — no prefix stripped.
    assert captured["body"]["route_options"] == [
        {"model": "databricks-claude-opus-4-8", "harness": "self"},
        {"model": "gpt-5-5", "harness": "self"},
    ]
    assert result is not None
    assert result.model == "databricks-claude-opus-4-8"


@pytest.mark.asyncio
async def test_external_routing_client_empty_available_models_skips() -> None:
    """No candidates -> no HTTP call, returns None."""
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = ExternalRoutingClient(base_url="http://localhost:6767/v1", router_name="task_v0")
    with _patch_httpx(httpx.MockTransport(handler)):
        assert await client.route("hi", {}) is None
    assert called is False


@pytest.mark.asyncio
async def test_external_routing_client_swallows_http_error() -> None:
    """A router outage returns None so the turn proceeds."""
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = ExternalRoutingClient(base_url="http://localhost:6767/v1", router_name="task_v0")
    with _patch_httpx(httpx.MockTransport(handler)):
        assert await client.route("hi", {"claude": ["claude-opus-4-8"]}) is None


@pytest.mark.asyncio
async def test_external_routing_client_empty_selection_returns_none() -> None:
    """An empty route_selection (e.g. router declined) yields None."""
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"route_selection": [], "rationale": ""})

    client = ExternalRoutingClient(base_url="http://localhost:6767/v1", router_name="task_v0")
    with _patch_httpx(httpx.MockTransport(handler)):
        assert await client.route("hi", {"claude": ["claude-opus-4-8"]}) is None


@pytest.mark.asyncio
async def test_external_routing_client_rejects_out_of_set_model() -> None:
    """A model the router was never offered is rejected, not persisted.

    Parity with the built-in judge: the returned model would become the
    session's ``model_override``, so an out-of-set pick returns None and the
    turn proceeds on the agent's default model.
    """
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "route_selection": [
                    {"route_option": {"model": "hallucinated-model", "harness": "claude"}}
                ]
            },
        )

    client = ExternalRoutingClient(base_url="http://localhost:6767/v1", router_name="task_v0")
    with _patch_httpx(httpx.MockTransport(handler)):
        assert await client.route("hi", {"claude": ["claude-opus-4-8"]}) is None


@pytest.mark.asyncio
async def test_external_routing_client_sends_bearer_auth() -> None:
    """When built with auth, the request carries the bearer header."""
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient, _bearer_auth

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={"route_selection": [{"route_option": {"model": "m", "harness": "h"}}]},
        )

    client = ExternalRoutingClient(
        base_url="https://host/v1", router_name="task_v0", auth=_bearer_auth("dapi-XYZ")
    )
    with _patch_httpx(httpx.MockTransport(handler)):
        await client.route("hi", {"h": ["m"]})
    assert captured["authorization"] == "Bearer dapi-XYZ"


@pytest.mark.asyncio
async def test_external_routing_client_mints_fresh_token_per_call_from_profile() -> None:
    """With a databricks_profile, each call re-authenticates (OAuth refresh)."""
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    tokens = iter(["Bearer tok-1", "Bearer tok-2"])
    captured: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("Authorization"))
        return httpx.Response(
            200,
            json={"route_selection": [{"route_option": {"model": "m", "harness": "h"}}]},
        )

    client = ExternalRoutingClient(
        base_url="https://host/v1", router_name="task_v0", databricks_profile="agent"
    )

    # Stub the SDK Config so each authenticate() yields the next token — proving
    # the client re-resolves auth per call rather than caching a stale bearer.
    class _FakeConfig:
        def authenticate(self) -> dict[str, str]:
            return {"Authorization": next(tokens)}

    client._sdk_config = _FakeConfig()  # type: ignore[attr-defined]

    with _patch_httpx(httpx.MockTransport(handler)):
        await client.route("hi", {"h": ["m"]})
        await client.route("hi again", {"h": ["m"]})
    assert captured == ["Bearer tok-1", "Bearer tok-2"]


@pytest.mark.asyncio
async def test_external_routing_client_records_last_error_on_http_failure() -> None:
    """A 4xx/5xx sets last_error with the gateway's unwrapped message."""
    import httpx

    from omnigent.server.smart_routing import ExternalRoutingClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error_code": 401,
                "message": "Credential was not sent or was of an unsupported type for this API.",
            },
        )

    client = ExternalRoutingClient(base_url="https://host/v1", router_name="task_v0")
    with _patch_httpx(httpx.MockTransport(handler)):
        result = await client.route("hi", {"h": ["m"]})
    assert result is None
    assert client.last_error is not None
    assert "401" in client.last_error
    assert "Credential was not sent" in client.last_error


def test_router_error_detail_unwraps_nested_message() -> None:
    from omnigent.server.smart_routing import _router_error_detail

    # Doubly-encoded: outer message holds another JSON object.
    body = json.dumps(
        {
            "error_code": "BAD_REQUEST",
            "message": json.dumps({"error": {"message": "task_v0 requires [...] models"}}),
        }
    )
    assert _router_error_detail(body) == "task_v0 requires [...] models"
    # Plain body passes through (trimmed).
    assert _router_error_detail("boom") == "boom"


@pytest.mark.asyncio
async def test_route_session_harness_surfaces_router_error_detail() -> None:
    """When the client exposes last_error, route_session_harness surfaces it."""

    class _FailingClient:
        last_error = "router returned HTTP 401: Credential was not sent"

        async def route(
            self, _message: str, _available: dict[str, list[str]]
        ) -> RoutingResult | None:
            return None

    caps = _FakeCaps(routing_client=_FailingClient())
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, _verdict, error = await route_session_harness("hi")
    assert harness is None
    assert model is None
    assert error is not None
    assert "401" in error
    assert "Credential was not sent" in error


# ── route_session_harness ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_route_session_harness_picks_harness_and_model() -> None:
    """route_session_harness returns (harness, model, verdict) from the router."""
    expected = RoutingResult(
        model="databricks-claude-opus-4-8",
        rationale="complex codebase task",
        harness="claude-sdk",
    )
    caps = _FakeCaps(routing_client=_FakeRoutingClient(expected))
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, verdict, error = await route_session_harness("refactor the auth module")
    assert harness == "claude-sdk"
    assert model == "databricks-claude-opus-4-8"
    assert verdict is not None
    assert "rationale" in verdict
    assert error is None


@pytest.mark.asyncio
async def test_route_session_harness_passes_all_sdk_harnesses_static() -> None:
    """Without a runner_client, all _AUTO_ROUTING_HARNESSES appear as candidates."""
    received_harnesses: list[str] = []

    class _CapturingClient:
        async def route(
            self, _message: str, available_models: dict[str, list[str]]
        ) -> RoutingResult | None:
            received_harnesses.extend(available_models.keys())
            return RoutingResult(model="databricks-claude-haiku-4-5", rationale="x", harness="pi")

    caps = _FakeCaps(routing_client=_CapturingClient())
    with patch("omnigent.runtime._globals._caps", new=caps):
        await route_session_harness("quick task")
    for h in _AUTO_ROUTING_HARNESSES:
        assert h in received_harnesses, f"harness {h!r} missing from candidate set"


@pytest.mark.asyncio
async def test_route_session_harness_uses_catalog_session_id_for_fetch() -> None:
    """catalog_session_id (the parent) drives the catalog fetch, not session_id.

    A sub-agent's own catalog is "self"-only; routing must use the parent's
    full spawnable-worker catalog so the candidate set is stable.
    """
    from unittest.mock import MagicMock

    fetched_paths: list[str] = []

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "workers": {
            "claude_code": {
                "source": "catalog",
                "verified": True,
                "models": [{"id": "m1"}],
                "note": "",
            },
        }
    }
    mock_response.raise_for_status = MagicMock()

    async def _get(path: str, **_: Any) -> Any:
        fetched_paths.append(path)
        return mock_response

    mock_client = MagicMock()
    mock_client.get = _get

    caps = _FakeCaps(
        routing_client=_FakeRoutingClient(
            RoutingResult(model="m1", rationale="x", harness="claude-sdk")
        )
    )
    with patch("omnigent.runtime._globals._caps", new=caps):
        await route_session_harness(
            "hi",
            session_id="child_sess",
            catalog_session_id="parent_sess",
            runner_client=mock_client,
        )
    # The catalog was fetched for the PARENT, not the child.
    assert any("parent_sess" in p for p in fetched_paths)
    assert not any("child_sess" in p for p in fetched_paths)


@pytest.mark.asyncio
async def test_route_session_harness_uses_live_catalog_skips_absent_harness() -> None:
    """With a runner_client, harnesses absent from the live catalog are excluded."""
    from unittest.mock import AsyncMock, MagicMock

    # Live catalog has claude-sdk and codex but NOT pi.
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "workers": {
            "claude-sdk": {
                "source": "catalog",
                "verified": True,
                "models": [
                    {"id": "databricks-claude-haiku-4-5"},
                    {"id": "databricks-claude-opus-4-8"},
                ],
                "note": "",
            },
            "codex": {
                "source": "catalog",
                "verified": True,
                "models": [{"id": "databricks-gpt-5-4-nano"}],
                "note": "",
            },
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    received_harnesses: list[str] = []

    class _CapturingClient:
        async def route(
            self, _message: str, available_models: dict[str, list[str]]
        ) -> RoutingResult | None:
            received_harnesses.extend(available_models.keys())
            return RoutingResult(
                model="databricks-claude-haiku-4-5", rationale="simple", harness="claude-sdk"
            )

    caps = _FakeCaps(routing_client=_CapturingClient())
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, _verdict, _error = await route_session_harness(
            "hello",
            session_id="conv_test",
            runner_client=mock_client,
        )

    assert "pi" not in received_harnesses, "pi should be excluded: absent from live catalog"
    assert "claude-sdk" in received_harnesses
    assert "codex" in received_harnesses
    assert harness == "claude-sdk"
    assert model == "databricks-claude-haiku-4-5"


@pytest.mark.asyncio
async def test_route_session_harness_maps_worker_names_to_harnesses() -> None:
    """Live catalog keyed by worker names (claude_code, codex) maps to harness ids."""
    from unittest.mock import AsyncMock, MagicMock

    # Catalog uses SUB-AGENT worker names, as the real runner returns.
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "workers": {
            "claude_code": {
                "source": "catalog",
                "verified": True,
                "models": [{"id": "databricks-claude-opus-4-8"}],
                "note": "",
            },
            "codex": {
                "source": "catalog",
                "verified": True,
                "models": [{"id": "databricks-gpt-5-4-nano"}],
                "note": "",
            },
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    received: list[str] = []

    class _CapturingClient:
        async def route(
            self, _message: str, available_models: dict[str, list[str]]
        ) -> RoutingResult | None:
            received.extend(available_models.keys())
            return RoutingResult(
                model="databricks-claude-opus-4-8", rationale="complex", harness="claude-sdk"
            )

    caps = _FakeCaps(routing_client=_CapturingClient())
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, _verdict, error = await route_session_harness(
            "refactor everything",
            session_id="conv_child",
            runner_client=mock_client,
        )
    # Worker name claude_code → harness id claude-sdk in the candidate set.
    assert "claude-sdk" in received
    assert "codex" in received
    assert harness == "claude-sdk"
    assert model == "databricks-claude-opus-4-8"
    assert error is None


@pytest.mark.asyncio
async def test_route_session_harness_falls_back_when_catalog_has_only_self() -> None:
    """A catalog with only an unrecognized 'self' worker falls back to the static table."""
    from unittest.mock import AsyncMock, MagicMock

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "workers": {
            # "self" is not in _WORKER_NAME_TO_HARNESS, so live matching yields
            # nothing and the static infer_models fallback kicks in.
            "self": {
                "source": "catalog",
                "verified": True,
                "models": [{"id": "databricks-claude-opus-4-8"}],
                "note": "",
            },
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    received: list[str] = []

    class _CapturingClient:
        async def route(
            self, _message: str, available_models: dict[str, list[str]]
        ) -> RoutingResult | None:
            received.extend(available_models.keys())
            return RoutingResult(
                model="databricks-claude-opus-4-8", rationale="x", harness="claude-sdk"
            )

    caps = _FakeCaps(routing_client=_CapturingClient())
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, _model, _verdict, error = await route_session_harness(
            "hello",
            session_id="conv_child",
            runner_client=mock_client,
        )
    # Static fallback offers all _AUTO_ROUTING_HARNESSES.
    for h in _AUTO_ROUTING_HARNESSES:
        assert h in received, f"static fallback should offer {h!r}"
    assert harness == "claude-sdk"
    assert error is None


@pytest.mark.asyncio
async def test_route_session_harness_returns_none_when_no_client() -> None:
    """route_session_harness returns (None, None, None, error) when no routing client."""
    caps = _FakeCaps(routing_client=None)
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, verdict, error = await route_session_harness("hello")
    assert harness is None
    assert model is None
    assert verdict is None
    assert error is not None  # error message propagated


@pytest.mark.asyncio
async def test_route_session_harness_returns_none_for_empty_message() -> None:
    """route_session_harness returns (None, None, None) for empty user text."""
    caps = _FakeCaps(routing_client=_FakeRoutingClient(None))
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, _verdict, _error = await route_session_harness("")
    assert harness is None
    assert model is None


@pytest.mark.asyncio
async def test_route_session_harness_sends_full_candidate_set_unfiltered() -> None:
    """The candidate set sent to the router is NOT pruned of incompatible models.

    The external task_v0 router enforces a required model set and 400s if any
    required model is missing, so we must offer the full list (including
    gpt-5.5/5.6 and Claude models under pi) and correct an incompatible verdict
    afterward via the redirect, not by filtering candidates.
    """
    pi_models: list[str] = []

    class _CapturingClient:
        async def route(
            self, _message: str, available_models: dict[str, list[str]]
        ) -> RoutingResult | None:
            pi_models.extend(available_models.get("pi", []))
            return RoutingResult(model="databricks-gpt-5-4-nano", rationale="x", harness="codex")

    caps = _FakeCaps(routing_client=_CapturingClient())
    with patch("omnigent.runtime._globals._caps", new=caps):
        await route_session_harness("hello")
    # The excluded-on-pi models are still SENT (router requires the full set);
    # incompatibility is handled post-verdict by the redirect.
    assert "databricks-claude-haiku-4-5" in pi_models
    assert "databricks-gpt-5-5" in pi_models


@pytest.mark.asyncio
async def test_route_session_harness_redirects_incompatible_router_pick() -> None:
    """A router that ignores our candidate set and picks pi+gpt-5.5 is redirected.

    Some external routers return a (harness, model) pair we excluded. The
    verdict post-processing must redirect gpt-5.5 off pi to codex.
    """
    # Router returns pi + gpt-5-5 (an excluded, incompatible pair).
    expected = RoutingResult(
        model="databricks-gpt-5-5", rationale="picked despite exclusion", harness="pi"
    )
    caps = _FakeCaps(routing_client=_FakeRoutingClient(expected))
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, _verdict, error = await route_session_harness("do something")
    assert harness == "codex", f"gpt-5.5 on pi should redirect to codex, got {harness!r}"
    assert model == "databricks-gpt-5-5"
    assert error is None


@pytest.mark.asyncio
async def test_route_session_harness_redirects_claude_on_pi_to_claude_sdk() -> None:
    """A router pick of a Claude model on pi is redirected to claude-sdk."""
    expected = RoutingResult(model="databricks-claude-haiku-4-5", rationale="cheap", harness="pi")
    caps = _FakeCaps(routing_client=_FakeRoutingClient(expected))
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, _verdict, error = await route_session_harness("quick q")
    assert harness == "claude-sdk", f"claude on pi should redirect to claude-sdk, got {harness!r}"
    assert model == "databricks-claude-haiku-4-5"
    assert error is None


# ── task_v1 contract fixtures ─────────────────────────────────────────────
#
# Freeze the confirmed task_v1 behavior: the router infers a scenario from the
# model families offered, demands that scenario's full arm menu (extra models
# ignored), echoes the harness tag without reading it, and may select an arm
# this workspace has no endpoint for.

_CLAUDE_ARMS = ("claude-opus-4-8", "claude-sonnet-5")
_CODEX_ARMS = ("glm-5-2", "gpt-5-6-sol", "gpt-5-6-luna")


def _task_v1_client(**kwargs: Any) -> Any:
    from omnigent.server.smart_routing import ExternalRoutingClient

    kwargs.setdefault("base_url", "https://host/ai-gateway/routing/v1")
    kwargs.setdefault("router_name", "task_v1")
    kwargs.setdefault("model_prefixes", ["databricks-", "system.ai."])
    return ExternalRoutingClient(**kwargs)


def _capturing_handler(captured: dict[str, Any], pick: dict[str, Any]) -> Any:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"route_selection": [{"route_option": pick}], "rationale": "rule tree"},
        )

    return handler


@pytest.mark.asyncio
async def test_task_v1_codex_only_offers_codex_menu() -> None:
    import httpx

    captured: dict[str, Any] = {}
    client = _task_v1_client()
    with _patch_httpx(
        httpx.MockTransport(_capturing_handler(captured, {"model": "glm-5-2", "harness": "codex"}))
    ):
        result = await client.route("fix it", {"codex": ["databricks-gpt-5-4"]})

    offered = [o["model"] for o in captured["body"]["route_options"]]
    assert offered[0] == "gpt-5-4"  # catalog extra, tolerated by the router
    for arm in _CODEX_ARMS:
        assert arm in offered
    for arm in _CLAUDE_ARMS:
        assert arm not in offered
    assert result is not None
    assert result.harness == "codex"


@pytest.mark.asyncio
async def test_task_v1_claude_only_offers_claude_menu() -> None:
    import httpx

    captured: dict[str, Any] = {}
    client = _task_v1_client()
    with _patch_httpx(
        httpx.MockTransport(
            _capturing_handler(captured, {"model": "claude-sonnet-5", "harness": "claude-sdk"})
        )
    ):
        await client.route("hi", {"claude-sdk": ["databricks-claude-haiku-4-5"]})

    offered = [o["model"] for o in captured["body"]["route_options"]]
    for arm in _CLAUDE_ARMS:
        assert arm in offered
    for arm in _CODEX_ARMS:
        assert arm not in offered


@pytest.mark.asyncio
async def test_task_v1_mixed_harnesses_offer_all_five_arms() -> None:
    import httpx

    captured: dict[str, Any] = {}
    client = _task_v1_client()
    with _patch_httpx(
        httpx.MockTransport(
            _capturing_handler(captured, {"model": "claude-opus-4-8", "harness": "claude-sdk"})
        )
    ):
        await client.route(
            "refactor",
            {
                "claude-sdk": ["databricks-claude-haiku-4-5"],
                "codex": ["databricks-gpt-5-4"],
            },
        )

    offered = [o["model"] for o in captured["body"]["route_options"]]
    for arm in _CLAUDE_ARMS + _CODEX_ARMS:
        assert arm in offered


@pytest.mark.asyncio
async def test_task_v0_router_offers_catalog_only() -> None:
    """A router version with no menu entry gets exactly the caller's catalog."""
    import httpx

    captured: dict[str, Any] = {}
    client = _task_v1_client(router_name="task_v0")
    with _patch_httpx(
        httpx.MockTransport(_capturing_handler(captured, {"model": "gpt-5-4", "harness": "codex"}))
    ):
        await client.route("hi", {"codex": ["databricks-gpt-5-4"]})

    assert [o["model"] for o in captured["body"]["route_options"]] == ["gpt-5-4"]


@pytest.mark.asyncio
async def test_task_v1_partial_menu_error_surfaces_last_error() -> None:
    """A menu rejection leaves the session unrouted with the router's reason."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error_code": "BAD_REQUEST",
                "message": "scenario 'codex' requires its full menu; missing [gpt-5-6-luna]",
            },
        )

    client = _task_v1_client(scenario_menus={"task_v1": {"codex": ("glm-5-2",)}})
    with _patch_httpx(httpx.MockTransport(handler)):
        result = await client.route("hi", {"codex": ["databricks-gpt-5-4"]})

    assert result is None
    assert client.last_error is not None
    assert "requires its full menu" in client.last_error


@pytest.mark.asyncio
async def test_task_v1_unservable_arm_maps_to_servable_id() -> None:
    """gpt-5-6-sol has no endpoint here; the pick lands on the nearest one."""
    import httpx

    captured: dict[str, Any] = {}
    client = _task_v1_client()
    with _patch_httpx(
        httpx.MockTransport(
            _capturing_handler(captured, {"model": "gpt-5-6-sol", "harness": "codex"})
        )
    ):
        result = await client.route("hi", {"codex": ["databricks-gpt-5-4", "databricks-gpt-5-5"]})

    assert result is not None
    assert result.model == "databricks-gpt-5-5"
    assert result.raw_model == "gpt-5-6-sol"  # what the router actually said
    assert result.harness == "codex"


@pytest.mark.asyncio
async def test_task_v1_ignores_echoed_harness_and_derives_from_arm() -> None:
    """The harness tag is passthrough, so a nonsense echo never wins."""
    import httpx

    captured: dict[str, Any] = {}
    client = _task_v1_client()
    with _patch_httpx(
        httpx.MockTransport(
            _capturing_handler(
                captured, {"model": "claude-opus-4-8", "harness": "not-a-real-harness"}
            )
        )
    ):
        result = await client.route(
            "refactor",
            {
                "claude-sdk": ["databricks-claude-haiku-4-5"],
                "codex": ["databricks-gpt-5-4"],
            },
        )

    assert result is not None
    assert result.harness == "claude-sdk"
    assert result.raw_model == "claude-opus-4-8"


@pytest.mark.asyncio
async def test_task_v1_sends_selection_model_as_selector_config() -> None:
    import httpx

    captured: dict[str, Any] = {}
    client = _task_v1_client(selection_model="gpt-5-4-mini")
    with _patch_httpx(
        httpx.MockTransport(_capturing_handler(captured, {"model": "glm-5-2", "harness": "codex"}))
    ):
        await client.route("hi", {"codex": ["databricks-gpt-5-4"]})

    assert captured["body"]["route_selector"]["config"] == {"model": "gpt-5-4-mini"}


@pytest.mark.asyncio
async def test_task_v1_omits_selector_config_without_selection_model() -> None:
    import httpx

    captured: dict[str, Any] = {}
    client = _task_v1_client()
    with _patch_httpx(
        httpx.MockTransport(_capturing_handler(captured, {"model": "glm-5-2", "harness": "codex"}))
    ):
        await client.route("hi", {"codex": ["databricks-gpt-5-4"]})

    assert "config" not in captured["body"]["route_selector"]


@pytest.mark.asyncio
async def test_task_v1_keeps_raw_prompt_and_4000_char_cap() -> None:
    import httpx

    captured: dict[str, Any] = {}
    client = _task_v1_client()
    prompt = "x" * 5000
    with _patch_httpx(
        httpx.MockTransport(_capturing_handler(captured, {"model": "glm-5-2", "harness": "codex"}))
    ):
        await client.route(prompt, {"codex": ["databricks-gpt-5-4"]})

    assert captured["body"]["task"]["prompt"] == "x" * 4000


@pytest.mark.asyncio
async def test_route_session_harness_keeps_raw_pick_in_verdict() -> None:
    """An unservable pick is applied as a servable id but reported verbatim."""
    expected = RoutingResult(
        model="databricks-gpt-5-5",
        rationale="cheapest arm",
        harness="codex",
        raw_model="gpt-5-6-sol",
    )
    caps = _FakeCaps(routing_client=_FakeRoutingClient(expected))
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, verdict, error = await route_session_harness("do it")
    assert harness == "codex"
    assert model == "databricks-gpt-5-5"
    assert verdict is not None
    assert verdict["raw_model"] == "gpt-5-6-sol"
    assert error is None


# ── RouteOptionSource seam ────────────────────────────────────────────────


def test_route_option_source_scenario_menu_override() -> None:
    from omnigent.server.smart_routing import TaskV1RouteOptionSource

    source = TaskV1RouteOptionSource(
        router_name="task_v9",
        scenario_menus={"task_v9": {"codex": ("arm-a", "arm-b")}},
    )
    options = source.build_route_options(["codex"], {"codex": ["arm-a"]})
    assert [o.model for o in options] == ["arm-a", "arm-b"]


def test_route_option_source_rejects_never_offered_pick() -> None:
    from omnigent.server.smart_routing import RoutePick, TaskV1RouteOptionSource

    source = TaskV1RouteOptionSource()
    catalog = {"codex": ["databricks-gpt-5-4"]}
    assert source.resolve_selection(RoutePick(model="hallucinated"), ["codex"], catalog) is None


def test_route_option_source_redirects_excluded_pick_off_pi() -> None:
    from omnigent.server.smart_routing import RoutePick, TaskV1RouteOptionSource

    source = TaskV1RouteOptionSource()
    catalog = {"pi": ["databricks-gpt-5-5", "databricks-claude-haiku-4-5"]}
    gpt = source.resolve_selection(RoutePick(model="databricks-gpt-5-5"), ["pi"], catalog)
    claude = source.resolve_selection(
        RoutePick(model="databricks-claude-haiku-4-5"), ["pi"], catalog
    )
    assert gpt is not None and gpt.harness == "codex"
    assert claude is not None and claude.harness == "claude-sdk"


# ── Tier-aware substitution for unservable arms ────────────────────────────
#
# A live ucode workspace lists codex endpoints roughly alphabetically, so list
# position carries no cost/capability signal and no id shares more than the
# "gpt-5-" prefix with the gpt-5.6 arms.

_UCODE_CODEX_CATALOG: tuple[str, ...] = (
    "gpt-5-1-codex-max",
    "gpt-5-1-codex-mini",
    "gpt-5-2",
    "gpt-5-3-codex",
    "gpt-5-5",
    "gpt-5-mini",
    "gpt-5-nano",
)


def _substitute(arm: str, models: Sequence[str], harness: str = "codex") -> str:
    from omnigent.server.smart_routing import RoutePick, TaskV1RouteOptionSource

    source = TaskV1RouteOptionSource(model_prefixes=["databricks-", "system.ai."])
    resolved = source.resolve_selection(RoutePick(model=arm), [harness], {harness: list(models)})
    assert resolved is not None, f"{arm} resolved to nothing"
    assert resolved.raw_model == arm
    return resolved.model


def test_capable_arm_substitutes_most_capable_not_last_in_catalog() -> None:
    """sol is task_v1's anchor arm, so it must not land on a nano endpoint."""
    assert _substitute("gpt-5-6-sol", _UCODE_CODEX_CATALOG) == "gpt-5-5"


def test_cheap_arm_substitutes_cheapest_sensible_model() -> None:
    """luna is the cheap arm: the mini class, not gpt-5-5 and not nano."""
    assert _substitute("gpt-5-6-luna", _UCODE_CODEX_CATALOG) == "gpt-5-mini"


def test_cheap_arm_falls_back_to_nano_when_nothing_bigger() -> None:
    assert _substitute("gpt-5-6-luna", ("gpt-5-nano",)) == "gpt-5-nano"


def test_delegate_arm_substitutes_same_family_capable_model() -> None:
    """glm-5-2 has no local endpoint and no tier entry demoting it."""
    assert _substitute("glm-5-2", _UCODE_CODEX_CATALOG) == "gpt-5-5"


def test_delegate_arm_resolves_to_its_own_endpoint_when_served() -> None:
    """A workspace that serves GLM resolves the glm arm to that endpoint."""
    assert (
        _substitute("glm-5-2", (*_UCODE_CODEX_CATALOG, "databricks-glm-5-2"))
        == "databricks-glm-5-2"
    )


def test_substitution_is_independent_of_catalog_order() -> None:
    import random

    shuffled = list(_UCODE_CODEX_CATALOG)
    picks = set()
    for seed in range(25):
        random.Random(seed).shuffle(shuffled)
        picks.add((_substitute("gpt-5-6-sol", shuffled), _substitute("gpt-5-6-luna", shuffled)))
    assert picks == {("gpt-5-5", "gpt-5-mini")}


def test_servable_pick_keeps_exact_catalog_id() -> None:
    """The prefix-restore path is untouched: a servable arm is applied as-is."""
    assert (
        _substitute(
            "gpt-5-5", ("databricks-gpt-5-nano", "databricks-gpt-5-5", "databricks-gpt-5-mini")
        )
        == "databricks-gpt-5-5"
    )


def test_claude_arms_unaffected_when_servable() -> None:
    catalog = ("databricks-claude-sonnet-5", "databricks-claude-opus-4-8")
    assert _substitute("claude-sonnet-5", catalog, "claude-sdk") == "databricks-claude-sonnet-5"
    assert _substitute("claude-opus-4-8", catalog, "claude-sdk") == "databricks-claude-opus-4-8"


def test_unservable_opus_substitutes_most_capable_claude() -> None:
    catalog = (
        "databricks-claude-haiku-4-5",
        "databricks-claude-sonnet-4-6",
        "databricks-claude-opus-4-7",
    )
    assert _substitute("claude-opus-4-8", catalog, "claude-sdk") == "databricks-claude-opus-4-7"


def test_unservable_cheap_claude_arm_avoids_the_flagship() -> None:
    catalog = ("databricks-claude-opus-4-7", "databricks-claude-sonnet-4-6")
    assert _substitute("claude-sonnet-5", catalog, "claude-sdk") == "databricks-claude-sonnet-4-6"


def test_cheap_arm_never_falls_below_its_own_class() -> None:
    """A sonnet-class arm takes the older sonnet, not the cheapest real model."""
    catalog = (
        "databricks-claude-haiku-4-5",
        "databricks-claude-sonnet-4-6",
        "databricks-claude-opus-4-8",
    )
    assert _substitute("claude-sonnet-5", catalog, "claude-sdk") == "databricks-claude-sonnet-4-6"


def test_cheap_arm_falls_back_to_cheapest_when_nothing_at_or_below() -> None:
    catalog = ("databricks-claude-opus-4-7", "databricks-claude-opus-4-8")
    assert _substitute("claude-sonnet-5", catalog, "claude-sdk") == "databricks-claude-opus-4-7"


def test_sonnet_5_is_applied_exactly_when_servable() -> None:
    """The live-catalog case: exact prefix restore, no substitution at all."""
    catalog = (
        "databricks-claude-haiku-4-5",
        "databricks-claude-sonnet-4-6",
        "databricks-claude-sonnet-5",
        "databricks-claude-opus-4-8",
    )
    assert _substitute("claude-sonnet-5", catalog, "claude-sdk") == "databricks-claude-sonnet-5"


def test_config_supplied_menu_without_tiers_defaults_to_capable() -> None:
    from omnigent.server.smart_routing import RoutePick, TaskV1RouteOptionSource

    source = TaskV1RouteOptionSource(
        router_name="task_v9",
        scenario_menus={"task_v9": {"codex": ("mystery-arm",)}},
    )
    resolved = source.resolve_selection(
        RoutePick(model="mystery-arm"), ["codex"], {"codex": list(_UCODE_CODEX_CATALOG)}
    )
    assert resolved is not None
    assert resolved.model == "gpt-5-5"


# ── RoutingSettings parsing (cli) ─────────────────────────────────────────


def test_parse_routing_settings_defaults() -> None:
    from omnigent.cli import _parse_routing_settings
    from omnigent.server.smart_routing import TASK_V1_MENUS

    settings = _parse_routing_settings(None)
    assert settings.router_name == "task_v1"
    assert settings.selection_model is None
    assert settings.scenario_menus is TASK_V1_MENUS
    assert settings.subagent_fail_mode == "open"
    assert settings.subagent_cache_ttl_s == 300.0


def test_parse_routing_settings_reads_every_key() -> None:
    from omnigent.cli import _parse_routing_settings

    settings = _parse_routing_settings(
        {
            "provider": "external",
            "router_name": "task_v2",
            "selection_model": "gpt-5-4-mini",
            "subagent_fail_mode": "closed",
            "subagent_cache_ttl_s": 30,
            "scenario_menus": {"codex": ["arm-a", "arm-b"]},
        }
    )
    assert settings.router_name == "task_v2"
    assert settings.selection_model == "gpt-5-4-mini"
    assert settings.subagent_fail_mode == "closed"
    assert settings.subagent_cache_ttl_s == 30.0
    # The flat form is scoped to the configured router version.
    assert settings.scenario_menus == {"task_v2": {"codex": ("arm-a", "arm-b")}}


def test_parse_routing_settings_nested_scenario_menus() -> None:
    from omnigent.cli import _parse_routing_settings

    settings = _parse_routing_settings(
        {"scenario_menus": {"task_v1": {"cc": ["claude-opus-4-8"], "both": []}}}
    )
    assert settings.scenario_menus == {"task_v1": {"cc": ("claude-opus-4-8",), "both": ()}}


# ── Default-on AIGW routing ───────────────────────────────────────────────


def test_default_on_synthesizes_client_for_databricks_provider() -> None:
    from omnigent.cli import _build_default_databricks_routing_client, _parse_routing_settings
    from omnigent.server.smart_routing import ExternalRoutingClient

    cfg = {
        "providers": {
            "other": {"kind": "key"},
            "ws": {"kind": "databricks", "profile": "eng-ml-inference", "default": True},
        }
    }
    creds = MagicMock()
    creds.host = "https://eng-ml-inference.staging.cloud.databricks.com"
    with patch(
        "omnigent.runtime.credentials.databricks.resolve_databricks_workspace",
        return_value=creds,
    ):
        client = _build_default_databricks_routing_client(cfg, _parse_routing_settings(None))

    assert isinstance(client, ExternalRoutingClient)
    assert client._url == (
        "https://eng-ml-inference.staging.cloud.databricks.com/ai-gateway/routing/v1/routes:select"
    )
    assert client._router_name == "task_v1"
    assert client._databricks_profile == "eng-ml-inference"
    assert client._model_prefixes == ["databricks-", "system.ai."]


def test_default_on_skips_without_databricks_provider() -> None:
    from omnigent.cli import _build_default_databricks_routing_client, _parse_routing_settings

    settings = _parse_routing_settings(None)
    assert (
        _build_default_databricks_routing_client(
            {"providers": {"openrouter": {"kind": "gateway"}}}, settings
        )
        is None
    )
    # No providers in the server config: the global block is consulted next.
    with patch(
        "omnigent.onboarding.provider_config.load_config",
        return_value={"providers": {"openrouter": {"kind": "gateway"}}},
    ):
        assert _build_default_databricks_routing_client({}, settings) is None


def test_default_on_reads_global_providers_block() -> None:
    from omnigent.cli import _build_default_databricks_routing_client, _parse_routing_settings
    from omnigent.server.smart_routing import ExternalRoutingClient

    creds = MagicMock()
    creds.host = "https://ws.cloud.databricks.com"
    with (
        patch(
            "omnigent.onboarding.provider_config.load_config",
            return_value={"providers": {"ws": {"kind": "databricks", "profile": "oss"}}},
        ),
        patch(
            "omnigent.runtime.credentials.databricks.resolve_databricks_workspace",
            return_value=creds,
        ),
    ):
        client = _build_default_databricks_routing_client({}, _parse_routing_settings(None))
    assert isinstance(client, ExternalRoutingClient)
    assert client._databricks_profile == "oss"


def test_default_on_skips_when_workspace_host_unresolvable() -> None:
    from omnigent.cli import _build_default_databricks_routing_client, _parse_routing_settings

    cfg = {"providers": {"ws": {"kind": "databricks", "profile": "gone"}}}
    with patch(
        "omnigent.runtime.credentials.databricks.resolve_databricks_workspace",
        side_effect=OSError("no such profile"),
    ):
        assert _build_default_databricks_routing_client(cfg, _parse_routing_settings(None)) is None


@pytest.mark.asyncio
async def test_route_session_harness_falls_back_by_model_when_harness_absent() -> None:
    """When the router returns no harness, fall back to finding it by model."""
    expected = RoutingResult(
        model="databricks-gpt-5-4-nano",
        rationale="cheap task",
        harness=None,
    )
    caps = _FakeCaps(routing_client=_FakeRoutingClient(expected))
    with patch("omnigent.runtime._globals._caps", new=caps):
        harness, model, _verdict, _error = await route_session_harness("what time is it?")
    # codex precedes pi in _AUTO_ROUTING_HARNESSES, so a GPT model owned by both
    # deterministically resolves to codex.
    assert harness == "codex"
    assert model == "databricks-gpt-5-4-nano"
