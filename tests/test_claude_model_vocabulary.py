from __future__ import annotations

import pytest

from omnigent.claude_model_vocabulary import (
    claude_model_alias,
    claude_model_command_arg,
    normalized_model_id,
)

# A ucode-style launch pinning: each family alias mapped to a gateway id,
# plus the extra picker slot holding the newer sonnet generation.
_PINNED_ENV = {
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "databricks-claude-opus-4-8",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "databricks-claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "databricks-claude-haiku-4-5",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "databricks-claude-sonnet-5",
}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("databricks-claude-opus-4-8", "opus"),
        ("databricks-claude-sonnet-4-6", "sonnet"),
        ("databricks-claude-haiku-4-5", "haiku"),
        # Pinned to the custom slot, but the Agent tool enum has no slot for
        # it — its family alias is the closest accepted value.
        ("databricks-claude-sonnet-5", "sonnet"),
        ("claude-opus-4-8[1m]", "opus"),
        ("sonnet", "sonnet"),
        ("databricks-gpt-5-5", None),
    ],
)
def test_alias_translation_under_a_pinned_launch(model: str, expected: str | None) -> None:
    assert claude_model_alias(model, _PINNED_ENV) == expected


def test_alias_from_the_id_when_nothing_is_pinned() -> None:
    """A direct Anthropic login accepts the family alias as-is."""
    assert claude_model_alias("databricks-claude-sonnet-5", {}) == "sonnet"
    assert claude_model_alias("mystery-model", {}) is None


def test_unpinned_family_is_untranslatable() -> None:
    """An unpinned alias resolves to a vendor id the gateway rejects."""
    env = {"ANTHROPIC_DEFAULT_OPUS_MODEL": "databricks-claude-opus-4-8"}
    assert claude_model_alias("databricks-claude-sonnet-5", env) is None
    assert claude_model_alias("databricks-claude-opus-4-8", env) == "opus"


def test_command_arg_prefers_the_custom_slot_id() -> None:
    """``/model`` takes the custom slot's exact id, so no step-down."""
    assert (
        claude_model_command_arg("databricks-claude-sonnet-5", _PINNED_ENV)
        == "databricks-claude-sonnet-5"
    )
    assert claude_model_command_arg("databricks-claude-sonnet-4-6", _PINNED_ENV) == "sonnet"
    assert claude_model_command_arg("databricks-gpt-5-5", _PINNED_ENV) is None


def test_command_arg_rejects_empty_input() -> None:
    assert claude_model_command_arg("", _PINNED_ENV) is None
    assert claude_model_command_arg("   ", _PINNED_ENV) is None


def test_normalized_model_id_strips_prefix_and_context_suffix() -> None:
    assert normalized_model_id("databricks-claude-sonnet-5") == "claude-sonnet-5"
    assert normalized_model_id("system.ai.claude-sonnet-5") == "claude-sonnet-5"
    assert normalized_model_id("Claude-Opus-4-8[1M]") == "claude-opus-4-8"
