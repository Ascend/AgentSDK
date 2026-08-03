"""Comprehensive tests for all command-system registered slash commands.

Tests every command returned by ``get_builtin_commands()`` via the sync
execution path (``execute_command_sync``) with minimal mocking, verifying
that each command is registered, callable, and returns a well-formed result.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.command_system.builtins import (
    execute_command_sync,
    get_builtin_commands,
    register_builtin_commands,
)
from src.command_system.engine import create_command_context
from src.command_system.registry import CommandRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> CommandRegistry:
    """Fresh private registry with built-in commands only (no runtime)."""
    reg = CommandRegistry()
    register_builtin_commands(reg)
    return reg


@pytest.fixture
def ctx(tmp_path: Path) -> SimpleNamespace:
    """Minimal command context for sync execution."""
    conversation = MagicMock()
    conversation.messages = []
    conversation.clear = MagicMock()
    return create_command_context(
        workspace_root=tmp_path,
        conversation=conversation,
        cost_tracker=MagicMock(),
        history=MagicMock(),
        provider=SimpleNamespace(model="test-model"),
    )


def _cmd_names(reg: CommandRegistry) -> set[str]:
    """Return the set of registered command names (no aliases)."""
    return {c.name.lower() for c in reg.list_commands(include_disabled=True)}


# ---------------------------------------------------------------------------
# Registration completeness
# ---------------------------------------------------------------------------


def test_all_builtins_are_registered(registry: CommandRegistry) -> None:
    """Every command returned by ``get_builtin_commands`` must be in the registry."""
    builtins = get_builtin_commands()
    names = {c.name.lower() for c in builtins}
    # Use include_hidden=True because some commands (e.g. output-style) are hidden
    registered = {c.name.lower() for c in registry.list_commands(include_disabled=True, include_hidden=True)}
    missing = names - registered
    assert not missing, f"Commands not in registry: {missing}"


def test_all_builtins_have_call_impl(registry: CommandRegistry) -> None:
    """Every LocalCommand must have a call implementation set."""
    from src.command_system.types import LocalCommand

    for cmd in registry.list_commands(include_disabled=True):
        if isinstance(cmd, LocalCommand):
            assert cmd._call_impl is not None, f"/{cmd.name} has no call implementation (missing set_call)"


# ---------------------------------------------------------------------------
# Per-command smoke tests (sync execution)
# ---------------------------------------------------------------------------

CLEAR_ARGS: dict[str, str | None] = {}
COST_ARGS: dict[str, str | None] = {}
HELP_ARGS: dict[str, str | None] = {}
SKILLS_ARGS: dict[str, str | None] = {}
TELEMETRY_ARGS: dict[str, str | None] = {}
CONTEXT_ARGS: dict[str, str | None] = {}
COMPACT_ARGS: dict[str, str | None] = {}
RESUME_ARGS: dict[str, str | None] = {}
RESUME_WITH_ID_ARGS: dict[str, str | None] = {"args": "nonexistent-session"}
EXIT_ARGS: dict[str, str | None] = {}
CRON_LIST_ARGS: dict[str, str | None] = {}
CRON_STATUS_ARGS: dict[str, str | None] = {}
CRON_RUNS_ARGS: dict[str, str | None] = {}
CRON_DELETE_ARGS: dict[str, str | None] = {"args": "nonexistent"}
CRON_RUN_MISSING_ARGS: dict[str, str | None] = {"args": "nonexistent"}
ADVISOR_ARGS: dict[str, str | None] = {}
# InteractiveCommand / PromptCommand — not testable via sync execution
# security-review, export, output-style, statusline, theme


@pytest.mark.parametrize(
    "cmd_name,expect_success,kwargs",
    [
        # --- Simple local commands (always succeed) ---
        ("clear", True, CLEAR_ARGS),
        ("cost", True, COST_ARGS),
        ("help", True, HELP_ARGS),
        ("skills", True, SKILLS_ARGS),
        ("telemetry", True, TELEMETRY_ARGS),
        ("exit", True, EXIT_ARGS),
        # --- Commands needing specific args to not error ---
        ("cron-list", True, CRON_LIST_ARGS),
        ("cron-status", True, CRON_STATUS_ARGS),
        ("cron-runs", True, CRON_RUNS_ARGS),
        ("cron-delete", True, CRON_DELETE_ARGS),
        ("cron-run", True, CRON_RUN_MISSING_ARGS),
        # --- /resume without args (browses, may gracefully say no sessions) ---
        ("resume", True, RESUME_ARGS),
        # --- /resume with a nonexistent session id ---
        ("resume", True, RESUME_WITH_ID_ARGS),
        # --- /advisor (gated by can_user_configure_advisor) ---
        ("advisor", True, ADVISOR_ARGS),
    ],
)
def test_builtin_command_sync_execution(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx: SimpleNamespace,
    cmd_name: str,
    expect_success: bool,
    kwargs: dict[str, str | None],
) -> None:
    """Smoke-test each built-in command via execute_command_sync."""
    monkeypatch.setattr(
        "src.command_system.builtins.get_command_registry",
        lambda: registry,
    )

    args = kwargs.get("args", "")
    success, text, error = execute_command_sync(cmd_name, args, ctx)

    assert success is expect_success, (
        f"/{cmd_name}({args!r}) expected success={expect_success}, "
        f"got success={success}, text={text!r}, error={error!r}"
    )

    # No unhandled exceptions should bubble up as error text
    if expect_success:
        assert error is None, f"/{cmd_name}: unexpected error: {error}"


# ---------------------------------------------------------------------------
# Specific command behavior tests
# ---------------------------------------------------------------------------


def test_help_command_lists_all_commands(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx: SimpleNamespace,
) -> None:
    """``/help`` output contains every registered command name."""
    monkeypatch.setattr(
        "src.command_system.builtins.get_command_registry",
        lambda: registry,
    )
    success, text, error = execute_command_sync("help", "", ctx)

    assert success, f"/help failed: {error}"
    assert text is not None
    for cmd in registry.list_commands():
        assert cmd.name in text, f"/help output missing command {cmd.name}"


@pytest.mark.parametrize("alias", ["reset", "new"])
def test_clear_aliases_work(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx: SimpleNamespace,
    alias: str,
) -> None:
    """``/clear`` can be invoked via aliases ``reset`` and ``new``."""
    monkeypatch.setattr(
        "src.command_system.builtins.get_command_registry",
        lambda: registry,
    )

    success, text, error = execute_command_sync(alias, "", ctx)
    assert success, f"/{alias} failed: {error}"
    assert error is None


def test_clear_preserves_conversation_when_goal_clear_fails(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx: SimpleNamespace,
) -> None:
    """Conversation clearing is atomic with removal of the active goal."""
    monkeypatch.setattr(
        "src.command_system.builtins.get_command_registry",
        lambda: registry,
    )
    service = MagicMock()
    service.clear_goal.side_effect = RuntimeError("goal store unavailable")
    ctx.tool_context = SimpleNamespace(
        session_id="clear-session",
        goal_service=service,
    )

    success, _text, error = execute_command_sync("clear", "", ctx)

    assert success is False
    assert "goal store unavailable" in (error or "")
    ctx.conversation.clear.assert_not_called()


def test_exit_via_aliases(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx: SimpleNamespace,
) -> None:
    """``/quit`` and ``/q`` are aliases for ``/exit``."""
    monkeypatch.setattr(
        "src.command_system.builtins.get_command_registry",
        lambda: registry,
    )
    for alias in ("quit", "q"):
        success, text, error = execute_command_sync(alias, "", ctx)
        assert success, f"/{alias} failed: {error}"
        assert text is not None


@pytest.mark.parametrize("alias", ["cron-fire"])
def test_cron_run_alias(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx: SimpleNamespace,
    alias: str,
) -> None:
    """``/cron-fire`` is an alias for ``/cron-run``."""
    monkeypatch.setattr(
        "src.command_system.builtins.get_command_registry",
        lambda: registry,
    )
    success, text, error = execute_command_sync(alias, "nonexistent", ctx)
    assert success, f"/{alias} failed: {error}"


def test_unknown_command_returns_false(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx: SimpleNamespace,
) -> None:
    """An unregistered command must return ``(False, None, error)``."""
    monkeypatch.setattr(
        "src.command_system.builtins.get_command_registry",
        lambda: registry,
    )
    success, text, error = execute_command_sync("bogus-command", "", ctx)
    assert success is False
    assert error is not None
    assert "Unknown command" in error


# ---------------------------------------------------------------------------
# Permission command (REPL-native handler, not in command registry)
# ---------------------------------------------------------------------------


def test_permission_not_in_command_registry(registry: CommandRegistry) -> None:
    """``/permission`` is a REPL-native command, NOT in the command registry."""
    assert registry.get("permission") is None, (
        "/permission should not be in the command registry (it is handled directly by the REPL)"
    )


# ---------------------------------------------------------------------------
# Resume command - session id validation
# ---------------------------------------------------------------------------


def test_resume_with_valid_session_id(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx: SimpleNamespace,
    tmp_path: Path,
) -> None:
    """``/resume <session_id>`` validates the session directory exists."""
    monkeypatch.setattr(
        "src.command_system.builtins.get_command_registry",
        lambda: registry,
    )

    # Create a fake session directory
    from clawcodex_ext.services.session_storage import SESSIONS_DIR

    session_id = "test-session-123"
    fake_dir = SESSIONS_DIR / session_id
    fake_dir.mkdir(parents=True, exist_ok=True)
    try:
        success, text, error = execute_command_sync("resume", session_id, ctx)
        assert success, f"/resume {session_id} failed: {error}"
        assert error is None
        assert session_id in (text or "")
    finally:
        # Cleanup
        import shutil

        shutil.rmtree(fake_dir, ignore_errors=True)


def test_resume_with_nonexistent_session(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx: SimpleNamespace,
) -> None:
    """``/resume <nonexistent>`` returns a not-found message, not a crash."""
    monkeypatch.setattr(
        "src.command_system.builtins.get_command_registry",
        lambda: registry,
    )

    success, text, error = execute_command_sync("resume", "no-such-session", ctx)
    assert success, f"/resume no-such-session failed: {error}"
    assert error is None
    assert text is not None
    assert "not found" in text.lower() or "no session" in text.lower()
