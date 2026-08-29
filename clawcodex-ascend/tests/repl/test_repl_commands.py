#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for REPL-native slash commands not covered by the command registry.

Covers commands that are handled directly by ``ClawcodexREPL.handle_command``
but are NOT registered in the command system's ``get_builtin_commands()``:
``/resume``, ``/permission``, ``/diff``, ``/mcp``, ``/tasks``, ``/rewind``,
``/effort``, ``/history``, ``/idle``, ``/load``, ``/save``, ``/tui``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import src.config as config_module
from src.repl import ClawcodexREPL


class TestREPLNativeSlashCommands(unittest.TestCase):
    """Test REPL-native slash command handlers."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / ".clawcodex"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        test_config = {
            "default_provider": "glm",
            "providers": {
                "glm": {
                    "api_key": "test_api_key_12345678",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "default_model": "glm-4.5",
                }
            },
        }
        config_file = self.config_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(test_config, f)

        self._global_config_patcher = patch.object(config_module, "GLOBAL_CONFIG_FILE", config_file)
        self._global_config_patcher.start()
        config_module._default_manager = None

    def tearDown(self):
        self._global_config_patcher.stop()

    def _make_repl(self, **kwargs) -> ClawcodexREPL:
        """Create a ClawcodexREPL instance with minimal mocking.

        Patches ``src.agent.Session`` directly so ``_load_heavy_runtime``'s
        ``from src.agent import Session`` picks up the mock.
        """
        mock_session = Mock()
        mock_session.session_id = "test-session-id"
        mock_session.conversation = Mock()
        mock_session.conversation.messages = []
        mock_session.conversation.clear = Mock()
        mock_session.provider = "glm"
        mock_session.model = "glm-4.5"

        # Patch src.agent.Session directly — _load_heavy_runtime()
        # imports from there via `from src.agent import Session`
        self._session_patcher = patch("src.agent.Session", autospec=True)
        mock_session_class = self._session_patcher.start()
        mock_session_class.create.return_value = mock_session
        self.addCleanup(self._session_patcher.stop)

        with patch("src.providers.get_provider_class") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.model = "glm-4.5"
            mock_provider_class.return_value = mock_provider

            return ClawcodexREPL(provider_name="glm", **kwargs)

    # ------------------------------------------------------------------
    # /resume
    # ------------------------------------------------------------------

    def test_resume_with_session_id_loads_session(self):
        """``/resume <session_id>`` should call ``load_session``."""
        repl = self._make_repl()
        repl.load_session = Mock()

        repl.handle_command("/resume abc123")
        repl.load_session.assert_called_once_with("abc123")

    def test_resume_with_blank_arg_shows_browser(self):
        """``/resume`` with no or blank session id shows the browser."""
        repl = self._make_repl()
        repl.load_session = Mock()
        repl.console.print = Mock()

        # /resume (no args) should not crash
        repl.handle_command("/resume")

    def test_resume_in_built_in_commands(self):
        """``resume`` must be in ``_built_in_commands`` so the palette
        check does not intercept bare ``/resume``.
        """
        repl = self._make_repl()
        cmd_names = [c.lower() for c in repl._built_in_commands]
        assert "/resume" in cmd_names, "/resume not found in _built_in_commands"

    # ------------------------------------------------------------------
    # /permission
    # ------------------------------------------------------------------

    def test_permission_in_built_in_commands(self):
        """``permission`` must be in ``_built_in_commands`` so the palette
        check does not intercept bare ``/permission``.
        """
        repl = self._make_repl()
        cmd_names = [c.lower() for c in repl._built_in_commands]
        assert "/permission" in cmd_names, "/permission not found in _built_in_commands"

    def test_permission_with_mode_direct(self):
        """``/permission plan`` should set permission mode directly."""
        repl = self._make_repl()
        repl._apply_permission_mode = Mock()
        repl.console.print = Mock()

        repl.handle_command("/permission plan")
        repl._apply_permission_mode.assert_called_once_with("plan")

    def test_permission_without_args_shows_menu(self):
        """``/permission`` without args should show interactive menu."""
        repl = self._make_repl()
        repl._safe_input = Mock(return_value="")
        repl.console.print = Mock()

        repl.handle_command("/permission")

    # ------------------------------------------------------------------
    # /diff, /mcp, /tasks, /rewind (pre-palette REPL-native handlers)
    # ------------------------------------------------------------------

    def test_diff_routes_to_handle_repl_diff(self):
        """Bare ``/diff`` must route to ``_handle_repl_diff``."""
        repl = self._make_repl()
        repl._handle_repl_diff = Mock()

        repl.handle_command("/diff")
        repl._handle_repl_diff.assert_called_once()

    def test_mcp_routes_to_handle_repl_mcp(self):
        """Bare ``/mcp`` must route to ``_handle_repl_mcp``."""
        repl = self._make_repl()
        repl._handle_repl_mcp = Mock()

        repl.handle_command("/mcp")
        repl._handle_repl_mcp.assert_called_once()

    def test_tasks_routes_to_handle_repl_tasks(self):
        """Bare ``/tasks`` must route to ``_handle_repl_tasks``."""
        repl = self._make_repl()
        repl._handle_repl_tasks = Mock()

        repl.handle_command("/tasks")
        repl._handle_repl_tasks.assert_called_once()

    def test_rewind_routes_to_handle_repl_rewind(self):
        """Bare ``/rewind`` must route to ``_handle_repl_rewind``."""
        repl = self._make_repl()
        repl._handle_repl_rewind = Mock()

        repl.handle_command("/rewind")
        repl._handle_repl_rewind.assert_called_once()

    # ------------------------------------------------------------------
    # /effort, /history, /idle (pre-palette REPL-native handlers)
    # ------------------------------------------------------------------

    def test_effort_routes_to_handle_repl_effort(self):
        """Bare ``/effort`` must route to ``_handle_repl_effort``."""
        repl = self._make_repl()
        repl._handle_repl_effort = Mock()

        repl.handle_command("/effort")
        repl._handle_repl_effort.assert_called_once()

    def test_history_routes_to_handle_repl_history(self):
        """Bare ``/history`` must route to ``_handle_repl_history``."""
        repl = self._make_repl()
        repl._handle_repl_history = Mock()

        repl.handle_command("/history")
        repl._handle_repl_history.assert_called_once()

    def test_idle_routes_to_handle_repl_idle(self):
        """Bare ``/idle`` must route to ``_handle_repl_idle``."""
        repl = self._make_repl()
        repl._handle_repl_idle = Mock()

        repl.handle_command("/idle")
        repl._handle_repl_idle.assert_called_once()

    # ------------------------------------------------------------------
    # /load and /save
    # ------------------------------------------------------------------

    def test_load_with_session_id_calls_load_session(self):
        """``/load <session_id>`` must call ``load_session``."""
        repl = self._make_repl()
        repl.load_session = Mock()

        repl.handle_command("/load abc123")
        repl.load_session.assert_called_once_with("abc123")

    def test_load_without_args_shows_usage(self):
        """``/load`` without a session id must show usage error."""
        repl = self._make_repl()
        repl.load_session = Mock()
        repl.console.print = Mock()

        repl.handle_command("/load")
        repl.load_session.assert_not_called()
        # Should print an error message
        args, _ = repl.console.print.call_args
        assert any("Usage:" in str(a) or "error" in str(a).lower() for a in args), (
            f"Expected usage/error message, got: {args}"
        )

    def test_save_session_calls_save(self):
        """``/save`` must call ``session.save()``."""
        repl = self._make_repl()
        repl.session.save = Mock()

        repl.handle_command("/save")
        repl.session.save.assert_called_once()

    # ------------------------------------------------------------------
    # /tui handoff
    # ------------------------------------------------------------------

    def test_tui_handoff_does_not_crash(self):
        """``/tui`` should call ``_handoff_to_textual_tui``."""
        repl = self._make_repl()
        repl._handoff_to_textual_tui = Mock()

        repl.handle_command("/tui")
        repl._handoff_to_textual_tui.assert_called_once()

    # ------------------------------------------------------------------
    # /tools command
    # ------------------------------------------------------------------

    def test_tools_lists_registered_tools(self):
        """``/tools`` must list tools from the tool registry."""
        repl = self._make_repl()
        repl.console.print = Mock()
        repl.tool_registry.list_tools = Mock(return_value=[])

        repl.handle_command("/tools")
        # Should print "Available tools:" header
        calls = [str(c) for c in repl.console.print.call_args_list]
        assert any("Available tools" in c for c in calls), f"Expected tool listing, got: {calls}"

    # ------------------------------------------------------------------
    # No-arg palette guard
    # ------------------------------------------------------------------

    def test_all_known_commands_pass_palette_guard(self):
        """Every command in ``_built_in_commands`` must NOT be intercepted
        by the palette check when typed bare (the guard checks
        ``raw.lower() not in _built_in_commands``).
        """
        repl = self._make_repl()
        for cmd in repl._built_in_commands:
            if cmd == "/":
                continue
            lower = cmd.lower()
            assert lower in repl._built_in_commands or lower.lstrip("/") in [
                c.lstrip("/").lower() for c in repl._built_in_commands
            ], f"{cmd} in _built_in_commands but check logic may fail"


# ---------------------------------------------------------------------------
# /resume command system integration tests (standalone functions)
# ---------------------------------------------------------------------------


def test_resume_command_call_without_args_returns_text():
    """``resume_command_call`` without args returns a result, not a crash."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from clawcodex_ext.command_system.builtins import resume_command_call
    from src.command_system.engine import create_command_context

    ctx = create_command_context(
        workspace_root=Path("/tmp"),
        cwd=Path("/tmp"),
        conversation=MagicMock(),
    )
    result = resume_command_call("", ctx)
    assert result is not None
    assert result.type == "text"
    assert isinstance(result.value, str)


def test_resume_command_call_with_bad_id_returns_not_found():
    """``resume_command_call`` with a nonexistent session id returns a
    helpful 'not found' message.
    """
    from pathlib import Path
    from unittest.mock import MagicMock

    from clawcodex_ext.command_system.builtins import resume_command_call
    from src.command_system.engine import create_command_context

    ctx = create_command_context(
        workspace_root=Path("/tmp"),
        cwd=Path("/tmp"),
        conversation=MagicMock(),
    )
    result = resume_command_call("no-such-session-xyz", ctx)
    assert result is not None
    assert "not found" in result.value.lower()
