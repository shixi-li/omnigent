"""E2E: the rail's "+ New shell" affordance and typing into the shell.

The right rail's Shells tab shows by default whenever the session agent
declares a non-empty ``terminals:`` block — its empty state carries a
virtual "+ New shell" row (``NewTerminalButton`` in
``web/src/shell/NewTerminalButton.tsx``). With a single declared
terminal name the row creates the shell directly on click (no dropdown),
POSTing ``/resources/terminals`` and handing the new terminal's tab key
to ``onExpand``, which opens it in the main column via
``MainTerminalView``. None of this needs an LLM turn — the user, not the
agent, launches the shell — so these tests never send a chat message.

Three behaviors are covered:

1. **"+ New shell" launches and opens a shell.** Clicking the row creates
   a ``zsh`` shell and replaces the main session view with it: the
   chrome-light shell view (``MainTerminalView``'s ``isShellView``) shows
   a header naming the shell and a "Close shell" X, its xterm connects,
   and the X returns to the conversation surface.

2. **The user can type a command into the shell.** We type ``pwd`` into
   the connected shell and assert it keeps running — the keystrokes are
   accepted and the bridge does not error or close. We deliberately do
   NOT assert on the command's output: xterm renders to a WebGL canvas,
   so stdout is not in the DOM (the same reason
   ``files/test_right_panel.py`` only checks ``data-state``), and reading
   it back via a file side-effect proved environment-fragile (the shell's
   cwd and the filesystem-API root coincide locally but not on CI).

3. **The workspace rail preserves its redesigned top inset.** The rail floats
   beside the chat header at the 8px outer inset instead of clearing the
   header like the main-column surfaces.

Both use the function-scoped ``terminal_session`` fixture (registers the
``zsh``-declaring agent and a runner-bound session), so each test gets an
independent session.
"""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect

from tests.e2e_ui.conftest import open_right_rail


def _open_new_shell(page: Page) -> None:
    """Open the Shells tab and click the "+ New shell" row.

    Leaves the rail's Shells tab active with the create POST fired. Scopes
    every lookup to the desktop "Workspace" rail so it never matches the
    hidden mobile drawer that mirrors the same controls.

    :param page: Playwright page already navigated to ``/c/{id}``.
    """
    open_right_rail(page)
    rail = page.get_by_role("complementary", name="Workspace")
    # Shells is present by default — the agent declares a ``zsh`` terminal,
    # so the tab shows before any shell exists with the "+ New shell"
    # affordance as its whole content.
    rail.get_by_role("tab", name=re.compile("Shells")).click()
    # Single declared name → the row creates directly on click (no dropdown).
    rail.get_by_role("button", name="New shell").click()


def test_new_shell_launches_and_opens(page: Page, terminal_session: tuple[str, str]) -> None:
    """Clicking "+ New shell" launches a shell and opens it as a rail tab.

    The create is user-driven (no chat message), so the only wait is for
    the runner to spin the PTY up and the xterm to connect. The shell opens
    as a tab in the workspace rail's top strip (labeled ``zsh · u-…``) and
    its xterm renders inside the rail's content slot — the chat page is
    left undisturbed (no ``main-terminal-view`` takeover). The tab's "x"
    closes it and drops back to the Shells list.
    """
    base_url, session_id = terminal_session

    page.goto(f"{base_url}/c/{session_id}")
    _open_new_shell(page)

    rail = page.get_by_role("complementary", name="Workspace")
    # A shell tab appears in the rail strip, labeled "zsh · u-<rand>" (the
    # user-minted session key), carrying a "Close zsh · u-…" x button.
    close_tab = rail.get_by_role("button", name=re.compile(r"^Close zsh · u-"))
    expect(close_tab).to_be_visible(timeout=60_000)

    # The shell's xterm mounts INSIDE the rail (not the main column) and
    # connects. The chat surface is untouched — the composer stays visible.
    terminal_view = rail.get_by_test_id("terminal-view")
    expect(terminal_view.last).to_be_visible(timeout=20_000)
    expect(terminal_view.last).to_have_attribute("data-state", "connected", timeout=20_000)
    expect(page.get_by_test_id("main-terminal-view")).to_have_count(0)
    expect(page.get_by_placeholder("Ask the agent anything…")).to_be_visible()

    # The tab's x closes the shell — its xterm unmounts and the rail falls
    # back to the Shells list.
    close_tab.click()
    expect(terminal_view).to_have_count(0)


def test_new_shell_accepts_typed_command(page: Page, terminal_session: tuple[str, str]) -> None:
    """The user can type a command into a freshly created shell.

    Types ``pwd`` into the connected shell and asserts the bridge keeps
    running: the keystrokes are accepted and the PTY neither errors nor
    closes. We do NOT assert on the command's output — xterm renders to a
    WebGL canvas, so stdout never reaches the DOM, and capturing it via a
    file side-effect proved environment-fragile (the shell cwd and the
    filesystem-API root line up locally but not on CI). Verifying the shell
    stays healthy after input is the portable signal.
    """
    base_url, session_id = terminal_session

    page.goto(f"{base_url}/c/{session_id}")
    _open_new_shell(page)

    # Wait for the shell's xterm to connect before sending keystrokes —
    # input typed before the WS attach opens is dropped. The shell now opens
    # as a rail tab, so its xterm lives inside the Workspace rail.
    rail = page.get_by_role("complementary", name="Workspace")
    terminal_view = rail.get_by_test_id("terminal-view").last
    expect(terminal_view).to_be_visible(timeout=60_000)
    expect(terminal_view).to_have_attribute("data-state", "connected", timeout=20_000)

    # Focus xterm's hidden input (a plain container click doesn't reliably
    # focus the WebGL canvas in headless Chromium), then type a command.
    textarea = terminal_view.locator("textarea.xterm-helper-textarea")
    textarea.focus()
    page.keyboard.type("pwd")
    page.keyboard.press("Enter")

    # The shell accepted the command and stays live — no bridge error, no
    # "terminal session ended". A regression that drops user input or kills
    # the PTY on first keystroke would flip this out of ``connected``.
    expect(terminal_view).to_have_attribute("data-state", "connected")


def test_workspace_rail_preserves_outer_top_inset(
    page: Page, terminal_session: tuple[str, str]
) -> None:
    """The workspace rail starts at the shell's 8px outer inset.

    The old rail cleared the absolute chat header and aligned with expanded
    main-column surfaces. The redesign deliberately extends it beside the
    header, matching the sidebar's outer inset. Assert that geometry directly;
    shell launch behavior remains covered by the two tests above.
    """
    base_url, session_id = terminal_session

    page.goto(f"{base_url}/c/{session_id}")
    open_right_rail(page)
    rail = page.get_by_role("complementary", name="Workspace")
    header = page.get_by_role("banner")
    expect(rail).to_be_visible()
    expect(header).to_be_visible()

    rail_top = rail.evaluate("el => el.getBoundingClientRect().top")
    header_bottom = header.evaluate("el => el.getBoundingClientRect().bottom")
    assert abs(rail_top - 8) <= 2, (
        f"workspace rail top {rail_top}px — expected the 8px outer inset"
    )
    assert rail_top < header_bottom, (
        f"workspace rail top {rail_top}px vs header bottom {header_bottom}px "
        "— expected the rail to extend beside the header"
    )
