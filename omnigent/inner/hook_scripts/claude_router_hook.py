"""Claude Code ``PreToolUse`` hook that routes native subagent spawns.

Registered by ``build_hook_settings`` on the ``Task|Agent`` matcher and
run as a subprocess per spawn. Reads the hook payload from stdin, asks
the runner's ``route-subagent`` endpoint what to do, and writes the
decision to stdout.

Always exits ``0``: routing must never be the reason a spawn fails. When
the endpoint is unadvertised, unreachable, or answers ``allow``, the hook
emits nothing and Claude proceeds unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys

from omnigent.inner.hook_scripts.subagent_router import route_pre_tool_use

_HARNESS = "claude-native"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="omnigent-claude-router-hook")
    parser.add_argument("--bridge-dir", default=None)
    # Advertisement directory, when it isn't the bridge dir itself.
    parser.add_argument("--router-dir", default=None)
    parser.add_argument("--parent-model", default=None)
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    """
    Run the hook.

    :param argv: Command-line arguments, excluding the program name.
        ``None`` uses :data:`sys.argv`.
    :returns: Always ``0`` so a routing failure never blocks a spawn.
    """
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError as exc:
        print(f"omnigent claude router hook: malformed JSON: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        print("omnigent claude router hook: expected JSON object", file=sys.stderr)
        return 0
    output = route_pre_tool_use(
        payload,
        harness=_HARNESS,
        router_dir=args.router_dir or args.bridge_dir,
        bridge_dir=args.bridge_dir,
        parent_model=args.parent_model,
    )
    if output is not None:
        sys.stdout.write(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
