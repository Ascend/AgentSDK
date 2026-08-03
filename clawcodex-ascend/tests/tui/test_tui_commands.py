# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Tests for TUI slash command dispatch.

Covers all commands in ``LOCAL_BUILTINS`` to ensure they are properly
dispatched by ``dispatch_local_command`` without crashing, and that
registry-backed commands go through ``dispatch_registry_command``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from src.tui.commands import (
    LOCAL_BUILTINS,
    CommandDispatchResult,
    build_command_suggestions,
    build_command_words,
    dispatch_local_command,
    dispatch_registry_command,
)
from src.tool_system.registry import ToolRegistry
from src.tool_system.context import ToolContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path)


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.conversation = MagicMock()
    session.conversation.messages = []
    return session


# ---------------------------------------------------------------------------
# LOCAL_BUILTINS completeness
# ---------------------------------------------------------------------------

_KNOWN_HANDLED_COMMANDS: set[str] = {
    # Handled by dispatch_local_command
    "/help",
    "/exit",
    "/quit",
    "/q",
    "/repl",
    "/clear",
    "/tools",
    "/stream",
    "/model",
    "/effort",
    "/history",
    "/cost",
    "/idle",
    "/theme",
    "/diff",
    "/mcp",
    "/tasks",
    "/rewind",
    "/resume",
    "/permission",
    "/forecast",
    # Handled by dispatch_registry_command (NOT by dispatch_local_command)
    "/init",
    "/provider",
    "/multimodel",
    "/auto-fix",
    "/lkb",
    "/recap",
    "/btw",
    "/advisor",
    "/buddy",
    "/compact",
    "/context",
    "/cron-list",
    "/cron-delete",
    "/cron-run",
    "/cron-runs",
    "/cron-status",
    "/goal",
    "/export",
    "/output-style",
    "/security-review",
    "/review",
    "/statusline",
    "/telemetry",
    "/copy",
    "/doctor",
    "/logo",
    "/memory",
    "/permissions",
    "/release-notes",
    "/rename",
    "/stickers",
    "/vim",
    "/voice",
    "/workflows",
    "/deep-research",
    "/dashboard",
    "/dialogue",
    "/dream",
    "/eco",
    "/render-last",
    "/skills",
    "/tts",
    "/ultraplan",
}


def test_all_local_builtins_have_handlers():
    """Every command in ``LOCAL_BUILTINS`` must have a known handler."""
    for cmd in LOCAL_BUILTINS:
        assert cmd in _KNOWN_HANDLED_COMMANDS, f"{cmd} is in LOCAL_BUILTINS but not in _KNOWN_HANDLED_COMMANDS"


def test_all_suggestion_commands_have_handlers(tmp_path: Path):
    """Every command returned by ``build_command_suggestions`` must have
    a known handler (local, registry, tool adapter, or skill).
    """
    suggestions = build_command_suggestions(tmp_path)
    missing = [
        f"/{suggestion.name} (source={suggestion.source})"
        for suggestion in suggestions
        if suggestion.source not in {"skills", "tools"} and f"/{suggestion.name}" not in _KNOWN_HANDLED_COMMANDS
    ]
    assert not missing, "suggestions have no known handler: " + ", ".join(missing)


def test_command_suggestions_include_multimodel(tmp_path: Path):
    """The REPL shares this palette, so /multimodel must be completable."""
    suggestions = build_command_suggestions(tmp_path)
    entry = next((item for item in suggestions if item.name == "multimodel"), None)

    assert entry is not None
    assert entry.slash == "/multimodel"


@pytest.mark.asyncio
async def test_registry_fork_result_is_returned_as_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clawcodex_ext.command_system.engine import CommandResult

    async def _execute_command_async(name, args, context):
        del args, context
        return CommandResult.success_assistant(name, "VERDICT: PASS")

    monkeypatch.setattr(
        "src.command_system.builtins.execute_command_async",
        _execute_command_async,
    )

    result = await dispatch_registry_command("/verify", command_context=MagicMock())

    assert result.handled is True
    assert result.assistant_text == "VERDICT: PASS"
    assert result.assistant_name == "verify"
    assert result.prompt_text is None


@pytest.mark.asyncio
async def test_registry_text_result_preserves_goal_continuation_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clawcodex_ext.command_system.engine import CommandResult

    async def _execute_command_async(name, args, context):
        del args, context
        return CommandResult(
            success=True,
            command_name=name,
            result_type="text",
            text="Goal set: verify TUI continuation",
            should_query=True,
        )

    monkeypatch.setattr(
        "src.command_system.builtins.execute_command_async",
        _execute_command_async,
    )

    result = await dispatch_registry_command(
        "/goal verify TUI continuation",
        command_context=MagicMock(),
    )

    assert result.handled is True
    assert result.should_query is True


@pytest.mark.asyncio
async def test_registry_goal_status_preserves_transient_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clawcodex_ext.command_system.engine import CommandResult

    async def _execute_command_async(name, args, context):
        del args, context
        return CommandResult(
            success=True,
            command_name=name,
            result_type="text",
            text="Goal active",
            transient=True,
        )

    monkeypatch.setattr(
        "src.command_system.builtins.execute_command_async",
        _execute_command_async,
    )

    result = await dispatch_registry_command("/goal", command_context=MagicMock())

    assert result.handled is True
    assert result.transient is True


def test_app_shows_goal_status_as_transient_screen() -> None:
    from clawcodex_ext.tui.app import ClawCodexTUI
    from clawcodex_ext.tui.screens.goal_status import GoalStatusScreen

    app = MagicMock()
    transcript = MagicMock()
    result = CommandDispatchResult(
        handled=True,
        system_text="Goal active\n\n  running 3s",
        transient=True,
    )

    ClawCodexTUI._apply_command_result(app, result, transcript)

    transcript.append_system.assert_not_called()
    screen = app.push_screen.call_args.args[0]
    assert isinstance(screen, GoalStatusScreen)
    assert "running 3s" in screen._status_text


def test_app_starts_goal_continuation_after_rendering_command_result() -> None:
    from clawcodex_ext.tui.app import ClawCodexTUI

    bridge = MagicMock()
    app = MagicMock()
    app._agent_bridge = bridge
    transcript = MagicMock()
    result = CommandDispatchResult(
        handled=True,
        system_text="Goal set: verify TUI continuation",
        should_query=True,
    )

    ClawCodexTUI._apply_command_result(app, result, transcript)

    transcript.append_system.assert_called_once()
    bridge.continue_goal_if_idle.assert_called_once_with()


def test_app_clear_removes_conversation_and_session_goal(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from clawcodex_ext.agent.conversation import Conversation
    from clawcodex_ext.goal.service import GoalService
    from clawcodex_ext.goal.store import GoalStore, goals_db_filename
    from clawcodex_ext.tui.app import ClawCodexTUI
    from clawcodex_ext.tui.state import AppState

    service = GoalService(store=GoalStore(tmp_path / goals_db_filename()))
    service.set_goal("session-1", "clear with conversation")
    conversation = Conversation()
    conversation.add_user_message("old prompt")
    app = SimpleNamespace(
        tool_context=SimpleNamespace(session_id="session-1", goal_service=service),
        session=SimpleNamespace(conversation=conversation),
        app_state=AppState(model="test", provider="test"),
        _agent_bridge=MagicMock(),
    )
    transcript = MagicMock()

    ClawCodexTUI._apply_command_result(
        app,
        CommandDispatchResult(handled=True, system_text="__clear__"),
        transcript,
    )

    assert conversation.messages == []
    assert service.get_goal("session-1") is None
    assert app.app_state.goal_status is None
    transcript.clear_transcript.assert_called_once_with()


def test_app_clear_keeps_conversation_when_goal_store_fails() -> None:
    from types import SimpleNamespace

    from clawcodex_ext.agent.conversation import Conversation
    from clawcodex_ext.tui.app import ClawCodexTUI
    from clawcodex_ext.tui.state import AppState

    service = MagicMock()
    service.clear_goal.side_effect = RuntimeError("goal store unavailable")
    conversation = Conversation()
    conversation.add_user_message("keep me")
    app = SimpleNamespace(
        tool_context=SimpleNamespace(session_id="session-1", goal_service=service),
        session=SimpleNamespace(conversation=conversation),
        app_state=AppState(model="test", provider="test"),
        _agent_bridge=MagicMock(),
        announcer=MagicMock(),
    )
    transcript = MagicMock()

    ClawCodexTUI._apply_command_result(
        app,
        CommandDispatchResult(handled=True, system_text="__clear__"),
        transcript,
    )

    assert len(conversation.messages) == 1
    transcript.clear_transcript.assert_not_called()
    transcript.append_system.assert_called_once()


def test_app_refuses_resume_dialog_while_agent_is_busy() -> None:
    from types import SimpleNamespace

    from clawcodex_ext.tui.app import ClawCodexTUI

    app = SimpleNamespace(
        _agent_bridge=SimpleNamespace(busy=True),
        _show_resume_browser=MagicMock(),
        announcer=MagicMock(),
    )
    transcript = MagicMock()

    ClawCodexTUI._open_phase2_dialog(app, "resume", transcript)

    app._show_resume_browser.assert_not_called()
    transcript.append_system.assert_called_once_with(
        "Cannot resume while the agent is running. Press Esc to interrupt first.",
        style="error",
    )
    app.announcer.announce.assert_called_once()


def test_app_keeps_current_session_if_resume_becomes_busy(monkeypatch) -> None:
    from types import SimpleNamespace

    from clawcodex_ext.tui.app import ClawCodexTUI

    current = SimpleNamespace(session_id="session-1", conversation=MagicMock())
    resumed = SimpleNamespace(session_id="session-2", conversation=MagicMock())
    monkeypatch.setattr(
        "clawcodex_ext.agent.session_ext.resume_session_with_tail",
        lambda _session_id: (resumed, None),
    )
    command_context = SimpleNamespace(
        session=current,
        conversation=current.conversation,
        tool_context=object(),
    )
    bridge = MagicMock()
    bridge.busy = False
    bridge.replace_session.return_value = False
    app = SimpleNamespace(
        session=current,
        _agent_bridge=bridge,
        _command_context=command_context,
        announcer=MagicMock(),
    )

    ClawCodexTUI._on_session_selected(app, "session-2")

    assert app.session is current
    assert command_context.session is current
    assert command_context.conversation is current.conversation
    app.announcer.announce.assert_called_once()


def test_app_keeps_current_session_when_resume_target_is_missing(monkeypatch) -> None:
    from types import SimpleNamespace

    from clawcodex_ext.tui.app import ClawCodexTUI

    current = SimpleNamespace(session_id="session-1", conversation=MagicMock())
    monkeypatch.setattr(
        "clawcodex_ext.agent.session_ext.resume_session_with_tail",
        lambda _session_id: (None, None),
    )
    command_context = SimpleNamespace(
        session=current,
        conversation=current.conversation,
        tool_context=object(),
    )
    bridge = MagicMock()
    bridge.busy = False
    app = SimpleNamespace(
        session=current,
        _agent_bridge=bridge,
        _command_context=command_context,
        announcer=MagicMock(),
    )

    ClawCodexTUI._on_session_selected(app, "missing-session")

    assert app.session is current
    assert command_context.session is current
    assert command_context.conversation is current.conversation
    bridge.replace_session.assert_not_called()
    app.announcer.announce.assert_called_once_with("Unable to resume the selected session.")


# ---------------------------------------------------------------------------
# dispatch_local_command — each LOCAL_BUILTINS command
# ---------------------------------------------------------------------------


def test_dispatch_help(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command("/help", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry)
    assert result.handled is True
    assert "Slash commands" in (result.system_text or "")


def test_dispatch_exit(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command("/exit", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry)
    assert result.handled is True
    assert result.system_text == "__exit__"


def test_dispatch_quit(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command("/quit", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry)
    assert result.handled is True
    assert result.system_text == "__exit__"


def test_dispatch_q(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command("/q", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry)
    assert result.handled is True
    assert result.system_text == "__exit__"


def test_dispatch_repl(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command("/repl", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry)
    assert result.handled is True
    assert result.system_text == "__repl__"


def test_dispatch_clear(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/clear", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.system_text == "__clear__"


def test_dispatch_tools(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/tools", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    # Should contain registered tool names (initially empty)
    assert "tools" in (result.system_text or "").lower()


def test_dispatch_stream_on(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/stream on", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.system_text == "__stream_on__"


def test_dispatch_stream_off(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/stream off", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.system_text == "__stream_off__"


def test_dispatch_stream_toggle(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/stream toggle", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.system_text == "__stream_toggle__"


def test_dispatch_stream_no_arg_shows_status(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/stream", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.system_text == "__stream_status__"


def test_dispatch_model(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/model", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "model"


def test_dispatch_model_with_argument_falls_through_to_runtime_command(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/model gpt-live",
        session=mock_session,
        workspace_root=tmp_path,
        tool_registry=tool_registry,
    )
    assert result.handled is False
    assert result.open_dialog is None


def test_dispatch_models_is_not_a_command(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/models", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is False
    assert result.open_dialog is None


def test_dispatch_effort(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/effort", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "effort"


def test_dispatch_history(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/history", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "history"


def test_dispatch_cost(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command("/cost", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry)
    assert result.handled is True
    assert result.open_dialog == "cost"


def test_dispatch_idle(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command("/idle", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry)
    assert result.handled is True
    assert result.open_dialog == "idle"


def test_dispatch_theme(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/theme", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "theme"


def test_dispatch_diff(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command("/diff", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry)
    assert result.handled is True
    assert result.open_dialog == "diff"


def test_dispatch_mcp(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command("/mcp", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry)
    assert result.handled is True
    assert result.open_dialog == "mcp"


def test_dispatch_tasks(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/tasks", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "tasks"


def test_dispatch_rewind(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/rewind", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "rewind"


def test_dispatch_resume(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/resume", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "resume"


def test_dispatch_permission(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/permission", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "permission"


def test_dispatch_forecast_dialog(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/forecast", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "forecast"


def test_dispatch_forecast_run_dialog(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/forecast run", session=mock_session, workspace_root=tmp_path, tool_registry=tool_registry
    )
    assert result.handled is True
    assert result.open_dialog == "forecast"


def test_dispatch_forecast_status_falls_through(mock_session, tmp_path, tool_registry):
    result = dispatch_local_command(
        "/forecast status",
        session=mock_session,
        workspace_root=tmp_path,
        tool_registry=tool_registry,
    )
    assert result.handled is False


# ---------------------------------------------------------------------------
# Unknown command falls through
# ---------------------------------------------------------------------------


def test_dispatch_unknown_returns_not_handled(mock_session, tmp_path, tool_registry):
    """An unknown slash command must return ``handled=False``."""
    result = dispatch_local_command(
        "/bogus-command-xyz",
        session=mock_session,
        workspace_root=tmp_path,
        tool_registry=tool_registry,
    )
    assert result.handled is False


# ---------------------------------------------------------------------------
# build_command_words consistency
# ---------------------------------------------------------------------------


def test_build_command_words_includes_local_builtins(tmp_path: Path):
    """Every item in ``LOCAL_BUILTINS`` must appear in ``build_command_words``."""
    words = build_command_words(tmp_path)
    word_set = {w.lower() for w in words}
    for cmd in LOCAL_BUILTINS:
        assert cmd.lower() in word_set, f"{cmd} is in LOCAL_BUILTINS but missing from build_command_words"


def test_build_command_suggestions_includes_local_builtins(tmp_path: Path):
    """Every item in ``LOCAL_BUILTINS`` must appear in ``build_command_suggestions``."""
    suggestions = build_command_suggestions(tmp_path)
    sugg_names = {f"/{s.name}".lower() for s in suggestions}
    for cmd in LOCAL_BUILTINS:
        assert cmd.lower() in sugg_names, f"{cmd} is in LOCAL_BUILTINS but missing from build_command_suggestions"
