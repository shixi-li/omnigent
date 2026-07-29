from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnigent.inner.hook_scripts import codex_router_hook as hook

_ENCRYPTED_MESSAGE = "enc:AAAABBBBCCCC=="


def _payload(**tool_input: Any) -> dict[str, Any]:  # type: ignore[explicit-any]
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "collaborationspawn_agent",
        "model": "gpt-5-6-sol",
        "tool_input": {
            "task_name": "refactor-tests",
            "message": _ENCRYPTED_MESSAGE,
            **tool_input,
        },
    }


def _advertise(tmp_path: Path, **extra: Any) -> Path:  # type: ignore[explicit-any]
    (tmp_path / "subagent_router.json").write_text(
        json.dumps({"url": "http://127.0.0.1:1/", "token": "t0k", **extra})
    )
    return tmp_path


class _Router:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []  # type: ignore[explicit-any]
        self.response: dict[str, Any] | None = None  # type: ignore[explicit-any]

    def __call__(
        self,
        endpoint: Any,  # type: ignore[explicit-any]
        session_id: str,
        body: dict[str, Any],  # type: ignore[explicit-any]
        **kwargs: Any,  # type: ignore[explicit-any]
    ) -> dict[str, Any] | None:  # type: ignore[explicit-any]
        self.calls.append({"endpoint": endpoint, "session_id": session_id, "body": body})
        return self.response


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> _Router:
    fake = _Router()
    monkeypatch.setattr(hook, "request_decision", fake)
    return fake


def test_is_spawn_agent_tool_matches_flattened_name() -> None:
    assert hook.is_spawn_agent_tool("collaborationspawn_agent")
    assert hook.is_spawn_agent_tool("spawn_agent")
    assert not hook.is_spawn_agent_tool("Bash")
    assert not hook.is_spawn_agent_tool(None)


def test_build_route_request_never_sends_the_encrypted_prompt() -> None:
    body = hook.build_route_request(
        _payload()["tool_input"], harness="codex-native", parent_model="gpt-5-6-sol"
    )

    assert body == {
        "harness": "codex-native",
        "task_name": "refactor-tests",
        "prompt": None,
        "fork": False,
        "parent_model": "gpt-5-6-sol",
    }
    assert _ENCRYPTED_MESSAGE not in json.dumps(body)


def test_rewrite_injects_model_and_passes_message_verbatim(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path, session_id="conv_abc")
    router.response = {
        "action": "rewrite",
        "model": "claude-sonnet-5",
        "rationale": "cheapest arm",
        "decision_id": "d1",
    }

    out = hook.route_pre_tool_use(_payload(), router_dir=tmp_path)

    assert out == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": {
                "task_name": "refactor-tests",
                "message": _ENCRYPTED_MESSAGE,
                "model": "claude-sonnet-5",
            },
            "permissionDecisionReason": "cheapest arm",
        }
    }
    assert router.calls[0]["session_id"] == "conv_abc"
    assert router.calls[0]["body"]["prompt"] is None


def test_redirect_denies_with_sys_session_send_instruction(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path, session_id="conv_abc")
    router.response = {
        "action": "redirect",
        "harness": "claude-native",
        "model": "claude-opus-4-8",
    }

    out = hook.route_pre_tool_use(_payload(), router_dir=tmp_path)

    assert out == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "Router selected claude-native/claude-opus-4-8. Use sys_session_send "
                "with args.harness=claude-native, args.model=claude-opus-4-8 instead."
            ),
        }
    }


def test_deny_uses_router_rationale(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path, session_id="conv_abc")
    router.response = {"action": "deny", "rationale": "router unavailable"}

    out = hook.route_pre_tool_use(_payload(), router_dir=tmp_path)

    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert out["hookSpecificOutput"]["permissionDecisionReason"] == "router unavailable"


def test_allow_emits_no_opinion(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path, session_id="conv_abc")
    router.response = {"action": "allow", "rationale": "fork exempt"}

    assert hook.route_pre_tool_use(_payload(), router_dir=tmp_path) is None


def test_router_unreachable_allows_unchanged(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path, session_id="conv_abc")
    router.response = None

    assert hook.route_pre_tool_use(_payload(), router_dir=tmp_path) is None


def test_missing_advertisement_allows_unchanged(
    tmp_path: Path,
    router: _Router,
) -> None:
    assert hook.route_pre_tool_use(_payload(), router_dir=tmp_path) is None
    assert router.calls == []


def test_malformed_advertisement_allows_unchanged(
    tmp_path: Path,
    router: _Router,
) -> None:
    (tmp_path / "subagent_router.json").write_text("{not json")

    assert hook.route_pre_tool_use(_payload(), router_dir=tmp_path) is None
    assert router.calls == []


def test_unknown_session_allows_unchanged(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path)

    assert hook.route_pre_tool_use(_payload(), router_dir=tmp_path) is None
    assert router.calls == []


def test_baked_session_id_used_when_advertisement_has_none(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path)
    router.response = {"action": "allow"}

    hook.route_pre_tool_use(_payload(), router_dir=tmp_path, session_id="conv_baked")

    assert router.calls[0]["session_id"] == "conv_baked"


def test_non_spawn_tool_is_ignored(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path, session_id="conv_abc")
    payload = _payload()
    payload["tool_name"] = "shell"

    assert hook.route_pre_tool_use(payload, router_dir=tmp_path) is None
    assert router.calls == []


def test_fork_spawn_reported(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path, session_id="conv_abc")
    router.response = {"action": "allow"}

    hook.route_pre_tool_use(_payload(fork=True), router_dir=tmp_path)

    assert router.calls[0]["body"]["fork"] is True


def test_parent_model_falls_back_to_payload_model(
    tmp_path: Path,
    router: _Router,
) -> None:
    _advertise(tmp_path, session_id="conv_abc")
    router.response = {"action": "allow"}

    hook.route_pre_tool_use(_payload(), router_dir=tmp_path)

    assert router.calls[0]["body"]["parent_model"] == "gpt-5-6-sol"


def test_session_canary_subcommand_writes_file(tmp_path: Path) -> None:
    assert hook.main(["session-canary", "--bridge-dir", str(tmp_path), "--session-id", "c1"]) == 0

    record = json.loads(hook.canary_path(tmp_path).read_text())
    assert record["session_id"] == "c1"


def test_record_subagent_subcommand_appends_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for agent_id, model in (("a1", "claude-sonnet-5"), ("a2", "gpt-5-6-sol")):
        monkeypatch.setattr(
            hook.sys,
            "stdin",
            _Stdin(json.dumps({"agent_id": agent_id, "model": model, "task_name": "t"})),
        )
        assert hook.main(["record-subagent", "--bridge-dir", str(tmp_path)]) == 0

    lines = hook.audit_path(tmp_path).read_text().splitlines()
    assert [json.loads(line)["agent_id"] for line in lines] == ["a1", "a2"]
    assert json.loads(lines[0])["model"] == "claude-sonnet-5"


def test_unknown_subcommand_is_a_no_op(capsys: pytest.CaptureFixture[str]) -> None:
    assert hook.main(["nope"]) == 0
    assert "unknown subcommand" in capsys.readouterr().err


class _Stdin:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text
