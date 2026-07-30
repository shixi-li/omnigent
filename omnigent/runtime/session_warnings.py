"""In-process index of session-scoped warnings.

A *warning* is a degraded-but-running condition the UI shows in the
session header while the session keeps working — today only
``subagent_routing_unenforced``, published when a harness's router hook
never fired so native sub-agent spawns are not being gated.

Same shape as the other transient recovery indexes
(:mod:`pending_elicitations`, :mod:`pending_inputs`): populated by the
route layer, replayed into the cold-load snapshot
(``GET /v1/sessions/{id}``) via :func:`snapshot_for`, in-memory only and
process-affine. Losing a warning on restart is acceptable — the
publisher re-posts it the next time it observes the condition.

Entries are deduplicated on ``(code, harness)`` so a forwarder that
re-observes the same condition every poll tick does not grow the list.
"""

from __future__ import annotations

import threading
from typing import Any

#: Warning code for "a harness ran without the router hook enforcing picks".
SUBAGENT_ROUTING_UNENFORCED = "subagent_routing_unenforced"

_warnings: dict[str, list[dict[str, Any]]] = {}
_lock = threading.Lock()


def _key(warning: dict[str, Any]) -> tuple[str, str]:
    return (str(warning.get("code") or ""), str(warning.get("harness") or ""))


def record(session_id: str, warning: dict[str, Any]) -> None:
    """
    Record one warning for *session_id*, replacing any same-key entry.

    :param session_id: Session/conversation identifier, e.g.
        ``"conv_abc123"``.
    :param warning: Warning payload, e.g. ``{"code":
        "subagent_routing_unenforced", "harness": "codex-native",
        "reason": "SessionStart canary did not fire"}``. Ignored when it
        carries no ``code``.
    """
    code = warning.get("code")
    if not isinstance(code, str) or not code:
        return
    with _lock:
        entries = _warnings.setdefault(session_id, [])
        key = _key(warning)
        for index, existing in enumerate(entries):
            if _key(existing) == key:
                entries[index] = dict(warning)
                return
        entries.append(dict(warning))


def snapshot_for(session_id: str) -> list[dict[str, Any]]:
    """
    Return the warnings to replay into a session snapshot.

    :param session_id: Session/conversation identifier.
    :returns: Warning payloads in the order they were first recorded;
        empty when nothing is wrong (the common case).
    """
    with _lock:
        return [dict(entry) for entry in _warnings.get(session_id, ())]
