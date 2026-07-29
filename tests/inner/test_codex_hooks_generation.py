from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import pytest

from omnigent.inner import codex_executor
from omnigent.inner.codex_executor import (
    CODEX_ROUTER_DIR_ENV_VAR,
    CODEX_ROUTER_SESSION_ID_ENV_VAR,
    SUBAGENT_ROUTING_UNENFORCED_WARNING,
    _CodexAppServerSession,
    _populate_codex_home_config,
    codex_hook_trust_bypass_args,
    codex_router_bridge_dir,
    codex_router_canary_fired,
    codex_router_hooks_generated,
    codex_router_hooks_settings,
    codex_router_session_id,
    merge_codex_user_hooks,
    read_codex_spawn_audit,
    subagent_routing_unenforced_warning,
    write_codex_router_hooks_file,
)
from omnigent.inner.hook_scripts.codex_router_hook import AUDIT_FILENAME, CANARY_FILENAME

_USER_HOOKS = {
    "hooks": {
        "PreToolUse": [{"hooks": [{"type": "command", "command": "user-pre"}]}],
        "Stop": [{"hooks": [{"type": "command", "command": "user-stop"}]}],
    }
}


def _write_user_home(tmp_path: Path, *, hooks: dict[str, object] | None = None) -> Path:
    source = tmp_path / "user-codex"
    source.mkdir()
    (source / "auth.json").write_text("{}")
    (source / "config.toml").write_text('model = "gpt-5.4-mini"\n')
    if hooks is not None:
        (source / "hooks.json").write_text(json.dumps(hooks))
    return source


def test_router_hooks_settings_registers_three_events(tmp_path: Path) -> None:
    payload = codex_router_hooks_settings(
        tmp_path / "bridge",
        session_id="conv_abc",
        python_executable="/usr/bin/python3",
    )

    hooks = payload["hooks"]
    assert set(hooks) == {"PreToolUse", "SessionStart", "SubagentStart"}
    (pre_entry,) = hooks["PreToolUse"]
    # Regex, never the flattened literal ``collaborationspawn_agent``.
    assert pre_entry["matcher"] == r".*spawn_agent"
    (pre_hook,) = pre_entry["hooks"]
    assert pre_hook["type"] == "command"
    assert pre_hook["timeout"] == 120
    assert "route-subagent" in pre_hook["command"]
    assert "--session-id conv_abc" in pre_hook["command"]
    assert "--harness codex" in pre_hook["command"]
    assert f"--bridge-dir {tmp_path / 'bridge'}" in pre_hook["command"]
    assert "session-canary" in hooks["SessionStart"][0]["hooks"][0]["command"]
    assert "record-subagent" in hooks["SubagentStart"][0]["hooks"][0]["command"]
    assert "matcher" not in hooks["SessionStart"][0]


def test_router_hooks_settings_omits_session_flag_when_unknown(tmp_path: Path) -> None:
    payload = codex_router_hooks_settings(tmp_path, python_executable="/usr/bin/python3")

    assert "--session-id" not in payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"]


def test_merge_user_hooks_preserves_user_entries_after_omnigent(tmp_path: Path) -> None:
    user_hooks = tmp_path / "hooks.json"
    user_hooks.write_text(json.dumps(_USER_HOOKS))
    payload = codex_router_hooks_settings(tmp_path, python_executable="/usr/bin/python3")

    merged = merge_codex_user_hooks(payload, user_hooks)

    pre = merged["hooks"]["PreToolUse"]
    assert len(pre) == 2
    assert pre[0]["matcher"] == r".*spawn_agent"
    assert pre[1]["hooks"][0]["command"] == "user-pre"
    assert merged["hooks"]["Stop"][0]["hooks"][0]["command"] == "user-stop"
    # The original payload is not mutated.
    assert len(payload["hooks"]["PreToolUse"]) == 1


def test_merge_user_hooks_tolerates_malformed_user_file(tmp_path: Path) -> None:
    user_hooks = tmp_path / "hooks.json"
    user_hooks.write_text("{not json")
    payload = codex_router_hooks_settings(tmp_path, python_executable="/usr/bin/python3")

    assert merge_codex_user_hooks(payload, user_hooks) == payload


def test_write_router_hooks_file_replaces_symlink_and_merges(tmp_path: Path) -> None:
    source = _write_user_home(tmp_path, hooks=_USER_HOOKS)
    codex_home = tmp_path / "private"
    codex_home.mkdir()
    _populate_codex_home_config(codex_home, source)
    assert (codex_home / "hooks.json").is_symlink()

    path = write_codex_router_hooks_file(
        codex_home,
        tmp_path / "bridge",
        session_id="conv_abc",
        python_executable="/usr/bin/python3",
    )

    assert not path.is_symlink()
    payload = json.loads(path.read_text())
    assert [entry.get("matcher") for entry in payload["hooks"]["PreToolUse"]] == [
        r".*spawn_agent",
        None,
    ]
    assert payload["hooks"]["Stop"][0]["hooks"][0]["command"] == "user-stop"
    # The user's real hooks.json is untouched.
    assert json.loads((source / "hooks.json").read_text()) == _USER_HOOKS


def test_populate_skips_hooks_symlink_when_routing_on(tmp_path: Path) -> None:
    source = _write_user_home(tmp_path, hooks=_USER_HOOKS)
    codex_home = tmp_path / "private"
    codex_home.mkdir()

    _populate_codex_home_config(codex_home, source, subagent_routing=True)

    assert not (codex_home / "hooks.json").exists()
    assert (codex_home / "auth.json").is_symlink()
    assert (codex_home / "config.toml").is_file()


def test_populate_symlinks_hooks_when_routing_off(tmp_path: Path) -> None:
    source = _write_user_home(tmp_path, hooks=_USER_HOOKS)
    codex_home = tmp_path / "private"
    codex_home.mkdir()

    _populate_codex_home_config(codex_home, source)

    assert (codex_home / "hooks.json").is_symlink()
    assert (codex_home / "hooks.json").resolve() == (source / "hooks.json").resolve()


def test_write_router_hooks_file_without_user_hooks(tmp_path: Path) -> None:
    codex_home = tmp_path / "private"
    codex_home.mkdir()

    path = write_codex_router_hooks_file(
        codex_home,
        tmp_path / "bridge",
        user_hooks_source=tmp_path / "missing" / "hooks.json",
        python_executable="/usr/bin/python3",
    )

    payload = json.loads(path.read_text())
    assert len(payload["hooks"]["PreToolUse"]) == 1


def test_router_hooks_generated_detection(tmp_path: Path) -> None:
    source = _write_user_home(tmp_path, hooks=_USER_HOOKS)
    codex_home = tmp_path / "private"
    codex_home.mkdir()
    assert not codex_router_hooks_generated(codex_home)

    _populate_codex_home_config(codex_home, source)
    assert not codex_router_hooks_generated(codex_home)

    write_codex_router_hooks_file(
        codex_home, tmp_path / "bridge", python_executable="/usr/bin/python3"
    )
    assert codex_router_hooks_generated(codex_home)


def test_hook_trust_bypass_args_requires_generated_hooks(tmp_path: Path) -> None:
    codex_home = tmp_path / "private"
    codex_home.mkdir()

    assert codex_hook_trust_bypass_args(codex_home, (0, 145, 0)) == []

    write_codex_router_hooks_file(
        codex_home, tmp_path / "bridge", python_executable="/usr/bin/python3"
    )
    assert codex_hook_trust_bypass_args(codex_home, (0, 145, 0)) == [
        "--dangerously-bypass-hook-trust"
    ]


def test_hook_trust_bypass_args_version_gated(tmp_path: Path) -> None:
    codex_home = tmp_path / "private"
    codex_home.mkdir()
    write_codex_router_hooks_file(
        codex_home, tmp_path / "bridge", python_executable="/usr/bin/python3"
    )

    assert codex_hook_trust_bypass_args(codex_home, (0, 130, 9)) == []
    assert codex_hook_trust_bypass_args(codex_home, (0, 131, 0)) == [
        "--dangerously-bypass-hook-trust"
    ]
    assert codex_hook_trust_bypass_args(codex_home, None) == []


def test_router_env_discovery(tmp_path: Path) -> None:
    env = {
        CODEX_ROUTER_DIR_ENV_VAR: str(tmp_path),
        CODEX_ROUTER_SESSION_ID_ENV_VAR: " conv_abc ",
    }

    assert codex_router_bridge_dir(env) == tmp_path
    assert codex_router_session_id(env) == "conv_abc"
    assert codex_router_bridge_dir({}) is None
    assert codex_router_session_id({}) is None


def test_canary_detection(tmp_path: Path) -> None:
    assert not codex_router_canary_fired(tmp_path)

    (tmp_path / CANARY_FILENAME).write_text(json.dumps({"session_id": "conv_abc"}))

    assert codex_router_canary_fired(tmp_path)


def test_read_spawn_audit_skips_malformed_lines(tmp_path: Path) -> None:
    assert read_codex_spawn_audit(tmp_path) == []
    (tmp_path / AUDIT_FILENAME).write_text(
        '{"agent_id": "a1", "model": "claude-sonnet-5"}\n'
        "not json\n"
        "\n"
        '{"agent_id": "a2", "model": "gpt-5-6-sol"}\n'
        "[1, 2]\n"
    )

    records = read_codex_spawn_audit(tmp_path)

    assert [(r["agent_id"], r["model"]) for r in records] == [
        ("a1", "claude-sonnet-5"),
        ("a2", "gpt-5-6-sol"),
    ]


class _HooksSnapshot:
    def __init__(self, path: Path) -> None:
        self.is_symlink = path.is_symlink()
        self.payload: dict[str, Any] | None = None  # type: ignore[explicit-any]
        if path.is_file():
            self.payload = json.loads(path.read_text())


def _start_app_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    env: dict[str, str],
    version: tuple[int, int, int] | None,
) -> tuple[tuple[str, ...], _HooksSnapshot]:
    source = _write_user_home(tmp_path, hooks=_USER_HOOKS)
    workspace = tmp_path / "work"
    workspace.mkdir()
    captured: list[tuple[tuple[str, ...], _HooksSnapshot]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> None:  # type: ignore[explicit-any]
        # The session deletes its private CODEX_HOME on the launch failure
        # below, so snapshot the hooks file while codex would have read it.
        home = Path(kwargs["env"]["CODEX_HOME"])
        captured.append((argv, _HooksSnapshot(home / "hooks.json")))
        raise RuntimeError("stop")

    async def fake_version(codex_path: str) -> tuple[int, int, int] | None:
        return version

    monkeypatch.setattr(codex_executor, "populate_codex_skills_from_bundle", lambda *a, **k: None)
    monkeypatch.setattr(codex_executor, "_codex_home_config_source_from_env", lambda: source)
    monkeypatch.setattr(codex_executor, "_codex_cli_version", fake_version)
    monkeypatch.setattr(codex_executor, "_create_subprocess_exec", fake_exec)
    session = _CodexAppServerSession(
        codex_path="/bin/echo",
        cwd=str(workspace),
        env=env,
        tool_executor=None,
    )
    with contextlib.suppress(RuntimeError):
        asyncio.run(session.start())
    assert captured, "the app-server was never launched"
    return captured[0]


def test_app_server_argv_bypasses_trust_for_generated_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge = tmp_path / "bridge"
    bridge.mkdir()

    argv, hooks = _start_app_server(
        tmp_path,
        monkeypatch,
        env={
            CODEX_ROUTER_DIR_ENV_VAR: str(bridge),
            CODEX_ROUTER_SESSION_ID_ENV_VAR: "conv_abc",
        },
        version=(0, 145, 0),
    )

    assert argv[:3] == ("/bin/echo", "--dangerously-bypass-hook-trust", "app-server")
    assert hooks.payload is not None
    payload = hooks.payload
    assert len(payload["hooks"]["PreToolUse"]) == 2
    assert "--session-id conv_abc" in payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"]


def test_app_server_argv_omits_flag_below_min_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge = tmp_path / "bridge"
    bridge.mkdir()

    argv, _ = _start_app_server(
        tmp_path,
        monkeypatch,
        env={CODEX_ROUTER_DIR_ENV_VAR: str(bridge)},
        version=(0, 130, 0),
    )

    assert argv[:2] == ("/bin/echo", "app-server")


def test_app_server_keeps_symlinked_hooks_when_routing_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, hooks = _start_app_server(tmp_path, monkeypatch, env={}, version=(0, 145, 0))

    assert argv[:2] == ("/bin/echo", "app-server")
    assert hooks.is_symlink


def test_unenforced_warning_payload() -> None:
    warning = subagent_routing_unenforced_warning("canary did not fire")

    assert warning == {
        "warning": SUBAGENT_ROUTING_UNENFORCED_WARNING,
        "harness": "codex",
        "reason": "canary did not fire",
    }
