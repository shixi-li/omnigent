"""Shared subagent-routing decision logic for harness hooks.

Stdlib-only on purpose: the Claude-native hook runs as a per-spawn
subprocess (``python -I -m
omnigent.inner.hook_scripts.claude_router_hook``) and blocks the spawn,
so importing anything heavier would show up as spawn latency. The
claude-agent-sdk executor imports the same functions for its in-process
``PreToolUse`` callback, so both paths map decisions identically.

The runner advertises its ``route-subagent`` endpoint by writing
``subagent_router.json`` (``{"url": ..., "token": ...}``) into the
session bridge directory. A missing or malformed file means the router
is unreachable: the hook allows the spawn unchanged and emits nothing.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Advertisement file written by the runner's subagent-routing endpoint,
# mirroring the ``tool_relay.json`` discovery pattern.
ADVERTISEMENT_FILE = "subagent_router.json"
# Claude-native bridge config, read for the session id / launch model.
_BRIDGE_CONFIG_FILE = "bridge.json"

# Explicit advertisement directory. Set for harnesses that have no
# claude-native bridge dir (e.g. the claude-agent-sdk executor).
ROUTER_DIR_ENV_VAR = "OMNIGENT_SUBAGENT_ROUTER_DIR"
# Session the spawn belongs to, when the harness knows it out of band.
SESSION_ID_ENV_VAR = "OMNIGENT_SUBAGENT_ROUTER_SESSION_ID"
# Claude-native bridge discovery, already exported to the harness.
BRIDGE_DIR_ENV_VAR = "HARNESS_CLAUDE_NATIVE_BRIDGE_DIR"
NATIVE_SESSION_ID_ENV_VAR = "HARNESS_CLAUDE_NATIVE_REQUEST_SESSION_ID"

# Claude Code's subagent-spawn tool. ``Agent`` is the current name;
# ``Task`` was renamed to it in CLI 2.1.63 and still works as an alias,
# so both are matched. Also used verbatim as the settings/SDK matcher —
# Claude Code reads a pipe-separated list as exact alternatives.
AGENT_TOOL_NAMES = ("Agent", "Task")
AGENT_TOOL_MATCHER = "|".join(AGENT_TOOL_NAMES)

# Subagent types that inherit the caller's context instead of starting
# fresh. Routing a fork would price a task whose real cost is dominated
# by inherited context, so v1 reports them and lets the server exempt
# them.
FORK_SUBAGENT_TYPES = frozenset({"fork"})
_FORK_SUFFIXES = ("-fork", "_fork", ":fork")

# The endpoint's own budget is generous (the router makes an LLM
# extraction call), but a spawn must not hang forever on a wedged
# runner.
REQUEST_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class RouterEndpoint:
    """Advertised ``route-subagent`` endpoint."""

    url: str
    token: str
    session_id: str | None = None


def discover_router_dir(bridge_dir: str | Path | None = None) -> Path | None:
    """
    Locate the directory holding the router advertisement.

    :param bridge_dir: Explicit directory, e.g. the ``--router-dir`` /
        ``--bridge-dir`` argv value. ``None`` falls back to
        :data:`ROUTER_DIR_ENV_VAR` then :data:`BRIDGE_DIR_ENV_VAR`.
    :returns: Directory path, or ``None`` when nothing advertises one.
    """
    if bridge_dir:
        return Path(bridge_dir)
    for env_var in (ROUTER_DIR_ENV_VAR, BRIDGE_DIR_ENV_VAR):
        raw = os.environ.get(env_var, "").strip()
        if raw:
            return Path(raw)
    return None


def read_router_endpoint(router_dir: str | Path | None) -> RouterEndpoint | None:
    """
    Read the advertised endpoint.

    :param router_dir: Directory containing
        :data:`ADVERTISEMENT_FILE`.
    :returns: Endpoint, or ``None`` when the advertisement is absent or
        malformed — treated as "router unreachable" by callers.
    """
    if router_dir is None:
        return None
    try:
        raw = (Path(router_dir) / ADVERTISEMENT_FILE).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    url = payload.get("url")
    token = payload.get("token")
    if not isinstance(url, str) or not url or not isinstance(token, str) or not token:
        return None
    session_id = payload.get("session_id")
    return RouterEndpoint(
        url=url.rstrip("/"),
        token=token,
        session_id=session_id if isinstance(session_id, str) and session_id else None,
    )


def resolve_session_id(
    endpoint: RouterEndpoint,
    *,
    bridge_dir: str | Path | None = None,
) -> str | None:
    """
    Resolve the Omnigent session the spawn belongs to.

    :param endpoint: Advertised endpoint, which may carry the session id.
    :param bridge_dir: Claude-native bridge directory, read as a last
        resort (``bridge.json`` tracks the active session across
        ``/clear`` rotations).
    :returns: Session id, e.g. ``"conv_abc123"``, or ``None``.
    """
    if endpoint.session_id:
        return endpoint.session_id
    for env_var in (SESSION_ID_ENV_VAR, NATIVE_SESSION_ID_ENV_VAR):
        raw = os.environ.get(env_var, "").strip()
        if raw:
            return raw
    config = _read_bridge_config(bridge_dir)
    for key in ("active_session_id", "conversation_id"):
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def resolve_parent_model(bridge_dir: str | Path | None) -> str | None:
    """
    Resolve the model the parent session runs on.

    :param bridge_dir: Claude-native bridge directory whose
        ``bridge.json`` records the launch model.
    :returns: Gateway model name, or ``None`` when unknown.
    """
    model = _read_bridge_config(bridge_dir).get("launch_model")
    return model if isinstance(model, str) and model else None


def _read_bridge_config(
    bridge_dir: str | Path | None,
) -> dict[str, Any]:  # type: ignore[explicit-any]  # opaque bridge JSON
    if bridge_dir is None:
        return {}
    try:
        config = json.loads((Path(bridge_dir) / _BRIDGE_CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def is_agent_tool(tool_name: Any) -> bool:  # type: ignore[explicit-any]  # hook payloads are untrusted JSON
    """
    Report whether a hook payload names the subagent-spawn tool.

    :param tool_name: ``tool_name`` from the hook payload.
    :returns: ``True`` for Claude Code's ``Task`` / ``Agent`` tool.
    """
    return isinstance(tool_name, str) and tool_name in AGENT_TOOL_NAMES


def is_fork_spawn(tool_input: dict[str, Any]) -> bool:  # type: ignore[explicit-any]  # hook payloads are untrusted JSON
    """
    Detect a context-inheriting (fork-typed) spawn.

    :param tool_input: ``tool_input`` from the hook payload.
    :returns: ``True`` when the requested subagent type inherits the
        caller's context.
    """
    subagent_type = tool_input.get("subagent_type")
    if not isinstance(subagent_type, str):
        return False
    normalized = subagent_type.strip().lower()
    return normalized in FORK_SUBAGENT_TYPES or normalized.endswith(_FORK_SUFFIXES)


def build_route_request(
    tool_input: dict[str, Any],  # type: ignore[explicit-any]  # hook payloads are untrusted JSON
    *,
    harness: str,
    parent_model: str | None = None,
) -> dict[str, Any]:  # type: ignore[explicit-any]  # JSON request body
    """
    Build the ``route-subagent`` request body.

    :param tool_input: ``tool_input`` from the hook payload.
    :param harness: Requesting harness, e.g. ``"claude-native"``.
    :param parent_model: Model the parent session runs on, when known.
    :returns: JSON-serializable request body.
    """
    task_name = tool_input.get("subagent_type")
    prompt = tool_input.get("prompt")
    return {
        "harness": harness,
        "task_name": task_name if isinstance(task_name, str) else "",
        "prompt": prompt if isinstance(prompt, str) and prompt else None,
        "fork": is_fork_spawn(tool_input),
        "parent_model": parent_model,
    }


def request_decision(
    endpoint: RouterEndpoint,
    session_id: str,
    body: dict[str, Any],  # type: ignore[explicit-any]  # JSON request body
    *,
    timeout: float = REQUEST_TIMEOUT_S,
) -> dict[str, Any] | None:  # type: ignore[explicit-any]  # JSON response body
    """
    POST one routing request to the runner.

    :param endpoint: Advertised endpoint.
    :param session_id: Omnigent session id.
    :param body: Request body from :func:`build_route_request`.
    :param timeout: Socket timeout in seconds.
    :returns: Decoded decision, or ``None`` on any transport / decode
        failure (callers treat that as "allow unchanged").
    """
    url = f"{endpoint.url}/v1/sessions/{urllib.parse.quote(session_id, safe='')}/route-subagent"
    # Loopback runner URL read from the owner-only bridge dir.
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {endpoint.token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    return payload if isinstance(payload, dict) else None


def _allow_with_model(
    tool_input: dict[str, Any],  # type: ignore[explicit-any]  # hook payloads are untrusted JSON
    model: str,
    reason: str,
) -> dict[str, Any]:  # type: ignore[explicit-any]  # hook output JSON
    output: dict[str, Any] = {  # type: ignore[explicit-any]
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": {**tool_input, "model": model},
    }
    if reason:
        output["permissionDecisionReason"] = reason
    return {"hookSpecificOutput": output}


def _deny(reason: str) -> dict[str, Any]:  # type: ignore[explicit-any]  # hook output JSON
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def redirect_reason(harness: str, model: str) -> str:
    """
    Build the cross-harness redirect instruction shown to the model.

    :param harness: Harness the router picked, e.g. ``"codex"``.
    :param model: Model the router picked.
    :returns: Deny reason telling the model how to respawn correctly.
    """
    return (
        f"Router selected {harness}/{model}. Use sys_session_send with "
        f"args.harness={harness}, args.model={model} instead."
    )


def decision_to_hook_output(
    decision: dict[str, Any],  # type: ignore[explicit-any]  # JSON response body
    tool_input: dict[str, Any],  # type: ignore[explicit-any]  # hook payloads are untrusted JSON
) -> dict[str, Any] | None:  # type: ignore[explicit-any]  # hook output JSON
    """
    Map a ``route-subagent`` decision to Claude ``PreToolUse`` output.

    :param decision: Decoded endpoint response.
    :param tool_input: Original ``tool_input``, preserved on rewrite.
    :returns: Hook output, or ``None`` for "no opinion" (allow the spawn
        unchanged with no emitted decision).
    """
    action = decision.get("action")
    model = decision.get("model")
    rationale = decision.get("rationale")
    rationale = rationale if isinstance(rationale, str) else ""
    if action == "rewrite" and isinstance(model, str) and model:
        return _allow_with_model(tool_input, model, rationale)
    if action == "redirect":
        harness = decision.get("harness")
        if isinstance(harness, str) and harness and isinstance(model, str) and model:
            return _deny(redirect_reason(harness, model))
        # A redirect without a target can't be followed — fail open.
        return None
    if action == "deny":
        return _deny(rationale or "Spawn denied by Omnigent intelligent routing.")
    return None


def route_pre_tool_use(
    payload: dict[str, Any],  # type: ignore[explicit-any]  # hook payloads are untrusted JSON
    *,
    harness: str,
    router_dir: str | Path | None = None,
    bridge_dir: str | Path | None = None,
    parent_model: str | None = None,
    timeout: float = REQUEST_TIMEOUT_S,
) -> dict[str, Any] | None:  # type: ignore[explicit-any]  # hook output JSON
    """
    Route one ``PreToolUse`` payload end to end.

    :param payload: Claude ``PreToolUse`` hook payload.
    :param harness: Requesting harness, e.g. ``"claude-sdk"``.
    :param router_dir: Advertisement directory; ``None`` discovers it.
    :param bridge_dir: Claude-native bridge directory for session-id
        fallback.
    :param parent_model: Model the parent session runs on, when known.
    :param timeout: Socket timeout in seconds.
    :returns: Hook output, or ``None`` for "no opinion" — every failure
        (no advertisement, unreachable router, malformed response) lands
        here so a spawn is never blocked by routing infrastructure.
    """
    if not is_agent_tool(payload.get("tool_name")):
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    endpoint = read_router_endpoint(discover_router_dir(router_dir))
    if endpoint is None:
        return None
    session_id = resolve_session_id(endpoint, bridge_dir=bridge_dir or router_dir)
    if not session_id:
        return None
    body = build_route_request(
        tool_input,
        harness=harness,
        parent_model=parent_model or resolve_parent_model(bridge_dir),
    )
    decision = request_decision(endpoint, session_id, body, timeout=timeout)
    if decision is None:
        return None
    return decision_to_hook_output(decision, tool_input)
