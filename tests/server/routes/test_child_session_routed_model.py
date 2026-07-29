"""
Tests for the routed-model fields on child-session summaries.

``GET /v1/sessions/{id}/child_sessions`` must expose the model a
sub-agent was routed onto so the Agents rail can render it per child.
The value comes from the child's ``model_override`` — the field session
routing writes when it picks a model for a spawned child.
"""

from __future__ import annotations

from omnigent.entities import Conversation
from omnigent.server.routes._sessions.helpers import (
    _child_session_summary_from_conversation,
)


def _child(model_override: str | None) -> Conversation:
    """A minimal sub-agent conversation with the given model override."""
    return Conversation(
        id="conv_child",
        created_at=100,
        updated_at=200,
        root_conversation_id="conv_parent",
        parent_conversation_id="conv_parent",
        title="researcher:auth",
        agent_id="ag_test",
        model_override=model_override,
    )


def test_routed_model_from_model_override() -> None:
    summary = _child_session_summary_from_conversation(
        _child("databricks-claude-opus-4-8"), "conv_parent", None
    )
    assert summary.routed_model == "databricks-claude-opus-4-8"
    # Not joined to a decision row yet.
    assert summary.routing_decision_id is None


def test_routed_model_null_when_unpinned() -> None:
    summary = _child_session_summary_from_conversation(_child(None), "conv_parent", None)
    assert summary.routed_model is None
    assert summary.routing_decision_id is None


def test_payload_carries_both_fields() -> None:
    payload = _child_session_summary_from_conversation(
        _child("databricks-gpt-5-6-sol"), "conv_parent", None
    ).model_dump()
    assert payload["routed_model"] == "databricks-gpt-5-6-sol"
    assert payload["routing_decision_id"] is None
