"""Launch-site behaviour for the per-session subagent-routing endpoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from omnigent.inner.hook_scripts.subagent_router import read_router_endpoint
from omnigent.runner import subagent_routing
from omnigent.runner.app import _ensure_session_subagent_router
from omnigent.runner.native.orchestration import _start_subagent_router_for_native_session


class _DeadClient:
    """Stands in for the runner→server client; never actually called."""

    async def post(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[explicit-any]
        raise RuntimeError("server down")


@pytest.fixture(autouse=True)
def _cleanup_routers() -> Any:  # type: ignore[explicit-any]
    yield
    for session_id in ("conv_native_launch", "conv_sdk_launch"):
        subagent_routing.shutdown_session_router(session_id)


def test_native_launch_installs_the_router_for_an_unrouted_session(tmp_path: Path) -> None:
    async def _run() -> None:
        advertised = _start_subagent_router_for_native_session(
            "conv_native_launch",
            bridge_dir=tmp_path,
            harness="claude-native",
            server_client=_DeadClient(),  # type: ignore[arg-type]
        )
        # No session flag consulted: the hooks are installed either way and
        # the server decides per spawn.
        assert advertised == tmp_path
        assert read_router_endpoint(tmp_path) is not None

    asyncio.run(_run())


def test_native_launch_skips_without_a_server_client(tmp_path: Path) -> None:
    async def _run() -> None:
        assert (
            _start_subagent_router_for_native_session(
                "conv_native_launch",
                bridge_dir=tmp_path,
                harness="claude-native",
                server_client=None,
            )
            is None
        )

    asyncio.run(_run())


@pytest.mark.parametrize("harness", ["claude-sdk", "codex"])
def test_sdk_launch_installs_the_router_for_an_unrouted_session(harness: str) -> None:
    async def _run() -> None:
        await _ensure_session_subagent_router(
            "conv_sdk_launch",
            harness,
            server_client=_DeadClient(),  # type: ignore[arg-type]
        )
        env = subagent_routing.session_router_env("conv_sdk_launch")
        assert env["OMNIGENT_SUBAGENT_ROUTER_SESSION_ID"] == "conv_sdk_launch"

    asyncio.run(_run())


def test_sdk_launch_skips_native_harnesses() -> None:
    async def _run() -> None:
        await _ensure_session_subagent_router(
            "conv_sdk_launch",
            "claude-native",
            server_client=_DeadClient(),  # type: ignore[arg-type]
        )
        assert subagent_routing.session_router_env("conv_sdk_launch") == {}

    asyncio.run(_run())
