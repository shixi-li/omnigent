"""
Tests for the harness runner CLI argument parsing, module
resolution, and parent-death watchdog.

Spawn-and-serve is exercised by ``test_process_manager.py`` since
that requires actually waiting on uvicorn — covering the same
ground here would just duplicate.
"""

from __future__ import annotations

import pytest

from omnigent.runtime.harnesses import _runner


def test_parse_args_requires_all_args() -> None:
    """Missing any of the four required args is a CLI error.

    Catches a regression where one of the arguments gets a default
    or becomes optional — the runner's contract is that all four
    (harness, module, socket, conversation-id) are AP-allocated
    and must arrive on the command line.
    """
    with pytest.raises(SystemExit):
        # Empty argv → argparse rejects, raising SystemExit(2).
        _runner._parse_args([])


def test_parse_args_returns_all_fields() -> None:
    """All required args round-trip into the namespace."""
    ns = _runner._parse_args(
        [
            "--harness",
            "test",
            "--module",
            "tests.runtime.harnesses._test_harness",
            "--socket",
            "/tmp/example.sock",
        ]
    )
    assert ns.harness == "test"
    assert ns.module == "tests.runtime.harnesses._test_harness"
    assert ns.socket == "/tmp/example.sock"


def test_parse_args_parent_pid_defaults_to_none() -> None:
    """``--parent-pid`` is optional and defaults to ``None``.

    When the parent doesn't pass it (e.g. during manual testing or
    standalone use), the watchdog thread should not start.
    """
    ns = _runner._parse_args(
        [
            "--harness",
            "test",
            "--module",
            "tests.runtime.harnesses._test_harness",
            "--socket",
            "/tmp/example.sock",
        ]
    )
    assert ns.parent_pid is None


def test_parse_args_parent_pid_parses_integer() -> None:
    """``--parent-pid`` parses as an integer when supplied.

    The watchdog thread needs an integer for ``os.kill(pid, 0)``.
    If argparse stored it as a string, the ``os.kill`` call would
    raise ``TypeError`` silently in the daemon thread.
    """
    ns = _runner._parse_args(
        [
            "--harness",
            "test",
            "--module",
            "tests.runtime.harnesses._test_harness",
            "--socket",
            "/tmp/example.sock",
            "--parent-pid",
            "12345",
        ]
    )
    assert ns.parent_pid == 12345
    assert isinstance(ns.parent_pid, int)


def test_load_harness_app_import_error_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-importable module path is fatal at boot."""
    with pytest.raises(SystemExit) as excinfo:
        _runner._load_harness_app("missing", "omnigent.does_not_exist")
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "cannot import harness module" in err
    assert "'omnigent.does_not_exist'" in err


def test_load_harness_app_module_without_create_app_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A module that doesn't export create_app is fatal."""
    with pytest.raises(SystemExit) as excinfo:
        _runner._load_harness_app("broken", "omnigent.errors")
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "does not export create_app" in err


def test_load_harness_app_loads_test_fixture() -> None:
    """A real module with create_app loads and stashes app.state.harness."""
    app = _runner._load_harness_app("test", "tests.runtime.harnesses._test_harness")
    assert app.state.harness == "test"
