"""
Tests for the routing-identity fields on ``RoutingDecisionData``.

The intelligent-routing MVP adds harness / scope / decision identity to
the routing-decision transcript item. Every field is defaulted so rows
persisted before they existed still deserialize.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnigent.entities.conversation import RoutingDecisionData, parse_item_data

_LEGACY_ROW = {
    "model": "databricks-claude-opus-4-8",
    "applied": True,
    "rationale": "Multi-file refactor needs deep reasoning.",
}


def test_legacy_row_deserializes_with_defaults() -> None:
    data = parse_item_data("routing_decision", dict(_LEGACY_ROW))
    assert isinstance(data, RoutingDecisionData)
    assert data.model == "databricks-claude-opus-4-8"
    assert data.harness is None
    assert data.scope == "turn"
    assert data.decision_id is None
    assert data.raw_model is None
    assert data.attempted_override is None


def test_full_round_trip_preserves_new_fields() -> None:
    original = RoutingDecisionData(
        model="databricks-gpt-5-6-sol",
        applied=True,
        rationale="Short prompt, cheapest arm.",
        agent="claude_code",
        harness="codex",
        scope="native_subagent",
        decision_id="dec_abc123",
        raw_model="gpt-5-6-sol",
        attempted_override="databricks-gpt-5-5",
    )
    round_tripped = RoutingDecisionData(**original.model_dump())
    assert round_tripped == original


def test_dump_carries_new_keys() -> None:
    dumped = RoutingDecisionData(
        model="databricks-claude-sonnet-5",
        applied=False,
        rationale="Advise only.",
        harness="claude-native",
        scope="session",
        decision_id="dec_1",
    ).model_dump()
    assert dumped["harness"] == "claude-native"
    assert dumped["scope"] == "session"
    assert dumped["decision_id"] == "dec_1"
    assert dumped["raw_model"] is None


@pytest.mark.parametrize("scope", ["session", "turn", "child_session", "native_subagent"])
def test_every_scope_value_validates(scope: str) -> None:
    data = RoutingDecisionData(
        model="databricks-claude-sonnet-5",
        applied=True,
        rationale="ok",
        scope=scope,  # type: ignore[arg-type]
    )
    assert data.scope == scope


def test_unknown_scope_rejected() -> None:
    with pytest.raises(ValidationError):
        RoutingDecisionData(
            model="databricks-claude-sonnet-5",
            applied=True,
            rationale="ok",
            scope="galaxy",  # type: ignore[arg-type]
        )
