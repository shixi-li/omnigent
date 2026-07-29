"""Claude Code's model vocabulary, and how to speak it.

Omnigent routes to servable catalog ids (``databricks-claude-sonnet-5``),
but two Claude Code surfaces accept only the family *aliases*:

* the ``Agent`` / ``Task`` tool's ``model`` parameter — a closed enum
  (``sonnet``, ``opus``, ``haiku``, ``fable``), so a catalog id fails
  schema validation and the spawn dies before it starts;
* the ``/model`` slash command — an alias resolves with no validation,
  while any other value is probed/allowlist-checked and a gateway id is
  rejected, leaving the session silently on its old model.

Claude Code resolves each alias to a concrete id via the workspace's
``ANTHROPIC_DEFAULT_*_MODEL`` env (set by omnigent's launch config), so
inverting that mapping is exact; the id's own family segment is the
fallback. Both surfaces fail OPEN on an unknown id: skip the switch
rather than send something that is rejected or silently ignored.

Stdlib-only so hook subprocesses can import it on the spawn path.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

#: Family aliases both surfaces accept, longest-lived family first.
CLAUDE_MODEL_ALIASES: tuple[str, ...] = ("fable", "opus", "sonnet", "haiku")

#: Alias → env var Claude Code reads to pin that alias to one model id.
ALIAS_MODEL_ENV_VARS: dict[str, str] = {
    "fable": "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
}

#: Extra picker slot pinned to one exact id. ``/model`` accepts that id
#: verbatim (the manual picker path relies on it), but the Agent tool's
#: enum does not, so only the slash-command translation uses it.
CUSTOM_MODEL_OPTION_ENV_VAR = "ANTHROPIC_CUSTOM_MODEL_OPTION"

#: Launch-env keys that define this session's model vocabulary.
MODEL_VOCABULARY_ENV_VARS: tuple[str, ...] = (
    *ALIAS_MODEL_ENV_VARS.values(),
    CUSTOM_MODEL_OPTION_ENV_VAR,
)

_CATALOG_PREFIXES: tuple[str, ...] = ("databricks-", "system.ai.")
_SEGMENT_RE = re.compile(r"[^a-z0-9]+")


def normalized_model_id(model: str) -> str:
    """Lower-case a model id, dropping catalog prefix and ``[1m]`` suffix.

    :param model: Any model id or alias.
    :returns: The comparable bare id, e.g. ``"claude-sonnet-5"``.
    """
    bare = model.strip().lower().removesuffix("[1m]")
    for prefix in _CATALOG_PREFIXES:
        if bare.startswith(prefix):
            return bare[len(prefix) :]
    return bare


def alias_pins(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Read the session's alias → model-id pinning.

    :param env: Environment mapping. ``None`` reads :data:`os.environ`.
    :returns: Alias → pinned model id, for the aliases that are pinned.
    """
    environ = os.environ if env is None else env
    pins: dict[str, str] = {}
    for alias, env_var in ALIAS_MODEL_ENV_VARS.items():
        pinned = environ.get(env_var, "").strip()
        if pinned:
            pins[alias] = pinned
    return pins


def claude_model_alias(
    model: str,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Translate a servable model id into Claude's alias vocabulary.

    An exact hit on the pinning is authoritative. Otherwise the id's own
    family segment names the alias — but only when that alias is itself
    pinned (or nothing is pinned at all, i.e. a direct Anthropic login):
    an unpinned alias resolves to a canonical vendor id, which a gateway
    endpoint rejects on the next request.

    :param model: Model id from a routing decision, or an alias already.
    :param env: Environment mapping holding the alias pinning. ``None``
        reads :data:`os.environ` — a hook subprocess inherits the CLI's.
    :returns: An accepted alias, or ``None`` when the id maps to nothing
        Claude would accept; callers must then leave the model alone.
    """
    if not isinstance(model, str) or not model.strip():
        return None
    candidate = model.strip().lower()
    if candidate in CLAUDE_MODEL_ALIASES:
        return candidate
    pins = alias_pins(env)
    normalized = normalized_model_id(model)
    for alias, pinned in pins.items():
        if normalized_model_id(pinned) == normalized:
            return alias
    segments = set(_SEGMENT_RE.split(normalized))
    for alias in CLAUDE_MODEL_ALIASES:
        if alias in segments:
            return alias if not pins or alias in pins else None
    return None


def claude_model_command_arg(
    model: str,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Translate a model id into a ``/model`` argument.

    Same alias vocabulary as :func:`claude_model_alias`, except the extra
    picker slot: ``/model`` takes that exact id, so a routed model pinned
    there is applied precisely instead of stepping down to its family
    alias.

    :param model: Model id from a routing decision, or an alias already.
    :param env: Environment mapping holding the session's pinning.
        ``None`` reads :data:`os.environ`.
    :returns: The ``/model`` argument, or ``None`` when the id maps to
        nothing the command accepts (the caller must skip the switch —
        an unaccepted value silently keeps the current model).
    """
    if not isinstance(model, str) or not model.strip():
        return None
    environ = os.environ if env is None else env
    custom = environ.get(CUSTOM_MODEL_OPTION_ENV_VAR, "").strip()
    if custom and normalized_model_id(custom) == normalized_model_id(model):
        return custom
    return claude_model_alias(model, env)
