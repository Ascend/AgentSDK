#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for REPL functionality."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
import tempfile
import json
import io
import os
import threading
import time
from contextlib import redirect_stderr
from types import SimpleNamespace
from rich.markdown import Markdown

import src.config as config_module
from src.repl import ClawcodexREPL
from src.agent import Session, Conversation
from clawcodex_ext.permissions.types import (
    PermissionAskRequest,
    PermissionUpdateSetMode,
)
from clawcodex_ext.providers.base import ChatResponse

from clawcodex_ext.utils.resume_hint import reset_resume_hint_for_test_only


class TestREPL(unittest.TestCase):
    """Test REPL functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear the process-wide resume-hint latch so a print in one test
        # does not bleed into another (the latch is intentional in
        # production but must be reset between test cases).
        reset_resume_hint_for_test_only()

        # Create a temporary config directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / ".clawcodex"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create a test config
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
        with open(config_file, "w") as f:
            json.dump(test_config, f)

        # Redirect ConfigManager to the test config and drop any cached
        # singleton state. Patching ``get_config_path`` alone is a no-op
        # because the manager reads ``GLOBAL_CONFIG_FILE`` directly.
        self._global_config_patcher = patch.object(config_module, "GLOBAL_CONFIG_FILE", config_file)
        self._global_config_patcher.start()
        config_module._default_manager = None

    def tearDown(self):
        self._global_config_patcher.stop()
        config_module._default_manager = None

    def test_repl_initialization(self):
        """Test REPL initialization."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session.return_value = Mock()

                with (
                    patch("src.providers.get_provider_class") as mock_provider_class,
                    patch(
                        "clawcodex_ext.repl.core.get_provider_class",
                        mock_provider_class,
                        create=True,
                    ),
                ):
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")
                    self.assertIsNotNone(repl)
                    self.assertEqual(repl.provider_name, "glm")
                    self.assertFalse(repl.stream)

    def test_repl_initialization_with_stream_enabled(self):
        """Test REPL can start with stream mode enabled."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session.return_value = Mock()

                with (
                    patch("src.providers.get_provider_class") as mock_provider_class,
                    patch(
                        "clawcodex_ext.repl.core.get_provider_class",
                        mock_provider_class,
                        create=True,
                    ),
                ):
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm", stream=True)
                    self.assertTrue(repl.stream)

    def test_repl_uses_agent_debug_history_file(self):
        """Agent debug mode keeps prompt history out of the real home dir."""
        debug_dir = Path(self.temp_dir) / "agent-debug"

        import clawcodex_ext.repl.core as repl_core

        repl_core._load_heavy_runtime()

        with patch.dict(
            os.environ,
            {
                "CLAWCODEX_AGENT_DEBUG": "1",
                "CLAWCODEX_AGENT_DEBUG_DIR": str(debug_dir),
                "CLAWCODEX_HISTORY_FILE": "",
            },
            clear=False,
        ):
            with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
                with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                    mock_session.return_value = Mock()

                    with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                        mock_provider = Mock()
                        mock_provider.model = "glm-4.5"
                        mock_provider_class.return_value = mock_provider

                        repl = ClawcodexREPL(provider_name="glm")

        self.assertEqual(Path(repl._file_history.filename), debug_dir / "history")

    def test_repl_threads_session_id_into_tool_context(self):
        """Slash commands that persist state need the active session id."""
        expected_sid = "repl-session-for-goal"

        import clawcodex_ext.repl.core as repl_core

        repl_core._load_heavy_runtime()

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = expected_sid
                mock_session.return_value = mock_session_instance

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")

        self.assertEqual(repl.tool_context.session_id, expected_sid)

    def test_run_emits_agent_debug_ready_marker(self):
        expected_sid = "repl-ready-session"

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = expected_sid
                mock_session_instance.conversation.messages = []
                mock_session.return_value = mock_session_instance

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    repl = ClawcodexREPL(provider_name="glm", stream=True)

        stderr = io.StringIO()
        with patch.dict(os.environ, {"CLAWCODEX_AGENT_DEBUG": "1"}, clear=False):
            with patch.object(repl, "_print_startup_header"):
                with patch.object(repl, "_run_main_loop", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        with redirect_stderr(stderr):
                            repl.run()

        marker = stderr.getvalue()
        self.assertIn("CLAWCODEX_AGENT_DEBUG::repl.ready::", marker)
        self.assertIn(expected_sid, marker)

    def test_startup_header_contains_logo_and_metadata(self):
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create"):
                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = Mock(return_value=mock_provider)

                    repl = ClawcodexREPL(provider_name="glm")

                    with patch("clawcodex_ext.repl.core.Path.cwd", return_value=Path(self.temp_dir)):
                        # Capture stdout to verify fallback path output
                        import io
                        from contextlib import redirect_stdout

                        f = io.StringIO()
                        with redirect_stdout(f):
                            repl._print_startup_header()

                        rendered = f.getvalue()
                        self.assertIn("ClawCodex", rendered)
                        self.assertIn("glm-4.5", rendered)
                        self.assertIn("GLM Provider", rendered)
                        # Path may be truncated, just check start and end parts
                        self.assertTrue(self.temp_dir[:20] in rendered or self.temp_dir[-20:] in rendered)

    def test_handle_command_exit(self):
        """Test /exit command."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create"):
                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")

                    with self.assertRaises(SystemExit):
                        repl.handle_command("/exit")

    def test_handle_command_exit_prints_resume_hint_on_tty(self):
        """S-R1: /exit on a TTY should print the standard resume hint
        using the session_id, matching CCB's ``printResumeHint()``.
        """
        import io
        import sys

        expected_sid = "0123456789abcdef0123456789abcdef"

        class _FakeTTYStdout:
            """A stdout stand-in that pretends to be a TTY and buffers writes."""

            def __init__(self) -> None:
                self._buf = io.StringIO()

            def isatty(self) -> bool:
                return True

            def write(self, s: str) -> int:
                return self._buf.write(s)

            def flush(self) -> None:
                self._buf.flush()

            def getvalue(self) -> str:
                return self._buf.getvalue()

        fake_stdout = _FakeTTYStdout()

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = expected_sid
                mock_session_instance.save = Mock()
                mock_session.return_value = mock_session_instance

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")

                    # Patch sys.stdout to a TTY-pretending stream so the
                    # resume-hint gate opens and the write is captured.
                    with patch.object(sys, "stdout", fake_stdout):
                        with self.assertRaises(SystemExit):
                            repl.handle_command("/exit")

                    rendered = fake_stdout.getvalue()
                    self.assertIn("Resume this session with:", rendered)
                    self.assertIn(f"clawcodex --resume {expected_sid}", rendered)
                    mock_session_instance.save.assert_called_once()

    def test_handle_command_exit_prints_hint_exactly_once_on_tty(self):
        """S-R1: the inline ``/exit`` path must emit the hint exactly once.

        Without the process-wide latch in :func:`print_resume_hint`, this
        test would fail because the same hint could be emitted by both the
        inline ``/exit`` print and the atexit cleanup registered in
        ``frontend/repl_extensions.py:_register_signal_session_save``.
        """
        import io
        import sys

        expected_sid = "fedcba9876543210fedcba9876543210"

        class _CountingTTYStdout:
            """TTY-pretending stdout that records every flush."""

            def __init__(self) -> None:
                self._buf = io.StringIO()
                self.flushes = 0

            def isatty(self) -> bool:
                return True

            def write(self, s: str) -> int:
                return self._buf.write(s)

            def flush(self) -> None:
                self._buf.flush()
                self.flushes += 1

            def getvalue(self) -> str:
                return self._buf.getvalue()

        fake_stdout = _CountingTTYStdout()

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = expected_sid
                mock_session_instance.save = Mock()
                mock_session.return_value = mock_session_instance

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")

                    with patch.object(sys, "stdout", fake_stdout):
                        with self.assertRaises(SystemExit):
                            repl.handle_command("/exit")

        rendered = fake_stdout.getvalue()
        # Exactly-once: the latch in print_resume_hint must suppress the
        # duplicate if a second call ever fires in the same process.
        self.assertEqual(
            rendered.count("Resume this session with:"),
            1,
            f"resume hint emitted multiple times:\n{rendered}",
        )
        self.assertEqual(
            rendered.count(f"clawcodex --resume {expected_sid}"),
            1,
        )

    def test_resume_hint_latch_suppresses_double_print_across_paths(self):
        """S-R1: when BOTH the inline ``/exit`` print and the atexit
        cleanup try to print, the latch keeps the output to a single hint.

        The inline ``/exit`` path runs the print inside
        ``ClawcodexREPL.handle_command``; the atexit path is the cleanup
        callback registered by ``_register_signal_session_save``. We
        trigger the atexit callback manually after ``/exit`` to simulate
        a SIGTERM landing during shutdown.
        """
        import io
        import sys

        from clawcodex_ext.utils.resume_hint import print_resume_hint

        expected_sid = "abcdef0123456789abcdef0123456789"

        class _CountingTTYStdout:
            def __init__(self) -> None:
                self._buf = io.StringIO()

            def isatty(self) -> bool:
                return True

            def write(self, s: str) -> int:
                return self._buf.write(s)

            def flush(self) -> None:
                self._buf.flush()

            def getvalue(self) -> str:
                return self._buf.getvalue()

        fake_stdout = _CountingTTYStdout()

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = expected_sid
                mock_session_instance.save = Mock()
                mock_session.return_value = mock_session_instance

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")

                    with patch.object(sys, "stdout", fake_stdout):
                        with self.assertRaises(SystemExit):
                            repl.handle_command("/exit")
                        # Inline /exit has already emitted once. Now fire
                        # the helper a second time to simulate the atexit
                        # cleanup re-printing. The latch must no-op.
                        print_resume_hint(expected_sid, stream=fake_stdout)

        rendered = fake_stdout.getvalue()
        self.assertEqual(
            rendered.count("Resume this session with:"),
            1,
            f"resume hint emitted twice across /exit + atexit:\n{rendered}",
        )

    def test_handle_command_clear(self):
        """Test /clear command."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.conversation = Mock()
                mock_session.return_value = mock_session_instance

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")
                    repl.handle_command("/clear")

                    mock_session_instance.conversation.clear.assert_called_once()

    def test_handle_command_stream_toggle(self):
        """Test /stream command toggles stream mode safely."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create"):
                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")
                    self.assertFalse(repl.stream)

                    repl.handle_command("/stream on")
                    self.assertTrue(repl.stream)

                    repl.handle_command("/stream off")
                    self.assertFalse(repl.stream)

    def test_handle_command_render_last_renders_markdown(self):
        """Test /render-last re-renders the last assistant response."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session.conversation.add_assistant_message("## Hello\n\n- item")
                mock_session_factory.return_value = mock_session

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")
                    repl.console.print = Mock()
                    repl.handle_command("/render-last")

                    self.assertTrue(
                        any(
                            args and isinstance(args[0], Markdown)
                            for args, _kwargs in repl.console.print.call_args_list
                        )
                    )

    def test_handle_command_render_last_without_message(self):
        """Test /render-last handles empty history gracefully."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")
                    repl.console.print = Mock()
                    repl.handle_command("/render-last")

                    self.assertTrue(
                        any(
                            args and "No assistant response available to render." in str(args[0])
                            for args, _kwargs in repl.console.print.call_args_list
                        )
                    )

    def test_local_command_text_defaults_to_plain_output(self):
        """Ordinary local command output must not be markdown-rendered."""
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.console = Mock()

        repl._print_local_command_text("**plain**", command="status")

        first_arg = repl.console.print.call_args_list[0].args[0]
        self.assertEqual(first_arg, "\n**plain**")
        self.assertFalse(
            any(args and isinstance(args[0], Markdown) for args, _kwargs in repl.console.print.call_args_list)
        )

    def test_recap_local_command_text_renders_markdown(self):
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.console = Mock()

        repl._print_local_command_text("Recapitulate\n- **done**", command="recap")

        self.assertTrue(
            any(args and isinstance(args[0], Markdown) for args, _kwargs in repl.console.print.call_args_list)
        )

    def test_command_result_with_should_query_starts_goal_continuation(self):
        from clawcodex_ext.command_system.engine import CommandResult

        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.console = Mock()
        repl._print_local_command_text = Mock()
        repl._continue_goal_if_idle = Mock(return_value=True)

        handled = repl._handle_command_result(
            CommandResult(
                success=True,
                command_name="goal",
                result_type="text",
                text="Goal active\n0s",
                should_query=True,
            )
        )

        self.assertTrue(handled)
        repl._continue_goal_if_idle.assert_called_once_with()

    def test_transient_goal_status_uses_dismissible_viewer(self):
        from clawcodex_ext.command_system.engine import CommandResult

        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.console = Mock()
        repl._print_local_command_text = Mock()
        repl._print_transient_text = Mock()

        handled = repl._handle_command_result(
            CommandResult(
                success=True,
                command_name="goal",
                result_type="text",
                text="Goal active\n\n  running 3s",
                transient=True,
            )
        )

        self.assertTrue(handled)
        repl._print_transient_text.assert_called_once_with(
            "Goal active\n\n  running 3s",
            command="goal",
        )
        repl._print_local_command_text.assert_not_called()

    def test_goal_continuation_mounts_live_status_and_passes_abort_controller(self):
        import asyncio
        from contextlib import nullcontext

        from clawcodex_ext.query.agent_loop_compat import AgentLoopRunResult
        from clawcodex_ext.types.messages import create_user_message
        from clawcodex_ext.utils.abort_controller import AbortController

        class FakeLiveStatus:
            instances = []

            def __init__(self, message, **kwargs):
                self.message = message
                self.kwargs = kwargs
                self._pending_text = "queued during goal"
                self.updates = []
                self.__class__.instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def update(self, message):
                self.updates.append(message)

        previous_controller = AbortController()
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.console = Mock()
        repl.provider = Mock()
        repl.tool_registry = Mock()
        repl.tool_context = SimpleNamespace(
            abort_controller=previous_controller,
            output_style_name=None,
            output_style_dir=None,
        )
        repl.session = SimpleNamespace(
            conversation=Conversation(),
            save_transcript=Mock(),
        )
        repl.stream = True
        repl._append_system_prompt = ""
        repl._engine_messages = []
        repl._max_turns = 20
        repl._stats_turns = 0
        repl._stats_input_tokens = 0
        repl._stats_output_tokens = 0
        repl._im_active_cancel = None
        repl._active_live_status = None
        repl.completer = None
        repl._file_history = None
        repl._do_expand_last = Mock()
        repl._apply_permission_mode_cycle = Mock()
        repl._bottom_toolbar = Mock(return_value="")
        repl._status_message = Mock(return_value="Thinking…")
        repl._enqueue_prompt = Mock()
        goal_loop = asyncio.new_event_loop()
        repl._get_chat_loop = Mock(return_value=goal_loop)

        continuation = SimpleNamespace(messages=[create_user_message("continue the goal", isMeta=True)])
        goal_runtime = Mock()
        goal_runtime.continue_if_idle.return_value = continuation
        goal_runtime.claim_continuation.return_value = True

        captured = {}

        async def fake_run_query_as_agent_loop(**kwargs):
            captured.update(kwargs)
            self.assertIs(repl._active_live_status, FakeLiveStatus.instances[-1])
            self.assertIs(
                repl._im_active_cancel,
                FakeLiveStatus.instances[-1].kwargs["on_cancel"],
            )
            self.assertIs(repl.tool_context.abort_controller, kwargs["abort_controller"])
            kwargs["on_attachment"](SimpleNamespace(role="user", content="attachment payload"))
            return AgentLoopRunResult(
                response_text="goal response",
                usage={"input_tokens": 4, "output_tokens": 2},
                num_turns=1,
            )

        try:
            with (
                patch("clawcodex_ext.repl.core._load_heavy_runtime"),
                patch("clawcodex_ext.repl.core.AbortController", AbortController, create=True),
                patch("clawcodex_ext.repl.core.LiveStatus", FakeLiveStatus, create=True),
                patch("clawcodex_ext.repl.core._pt_patch_stdout", return_value=nullcontext()),
                patch(
                    "clawcodex_ext.repl.core.resolve_output_style",
                    return_value=SimpleNamespace(prompt=""),
                    create=True,
                ),
                patch(
                    "clawcodex_ext.goal.runtime.goal_runtime_for_context",
                    return_value=goal_runtime,
                ),
                patch(
                    "clawcodex_ext.query.agent_loop_compat.build_effective_system_prompt",
                    return_value="system prompt",
                ),
                patch(
                    "clawcodex_ext.query.agent_loop_compat.run_query_as_agent_loop",
                    side_effect=fake_run_query_as_agent_loop,
                ),
            ):
                completed = repl._continue_goal_if_idle()
        finally:
            goal_loop.close()

        self.assertTrue(completed)
        self.assertIsNotNone(captured.get("abort_controller"))
        self.assertEqual(captured.get("max_turns"), 0)
        self.assertTrue(callable(captured.get("on_attachment")))
        self.assertEqual(repl.session.conversation.messages[-1].content, "attachment payload")
        self.assertIs(repl.tool_context.abort_controller, previous_controller)
        self.assertIsNone(repl._im_active_cancel)
        self.assertIsNone(repl._active_live_status)
        repl._enqueue_prompt.assert_called_once_with("queued during goal")
        self.assertEqual(repl._stats_turns, 1)
        self.assertEqual(repl._stats_input_tokens, 4)
        self.assertEqual(repl._stats_output_tokens, 2)

    def test_goal_continuation_im_interrupt_aborts_without_error(self):
        import asyncio
        from contextlib import nullcontext

        from clawcodex_ext.types.messages import create_user_message
        from clawcodex_ext.utils.abort_controller import AbortController, AbortError

        class FakeLiveStatus:
            instances = []

            def __init__(self, message, **kwargs):
                self.message = message
                self.kwargs = kwargs
                self._pending_text = ""
                self.updates = []
                self.__class__.instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def update(self, message):
                self.updates.append(message)

        previous_controller = AbortController()
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.console = Mock()
        repl.provider = Mock()
        repl.tool_registry = Mock()
        repl.tool_context = SimpleNamespace(
            abort_controller=previous_controller,
            output_style_name=None,
            output_style_dir=None,
        )
        repl.session = SimpleNamespace(
            conversation=Conversation(),
            save_transcript=Mock(),
        )
        repl.stream = True
        repl._append_system_prompt = ""
        repl._engine_messages = []
        repl._max_turns = 20
        repl._stats_turns = 0
        repl._stats_input_tokens = 0
        repl._stats_output_tokens = 0
        repl._im_active_cancel = None
        repl._direct_abort_controller = None
        repl._active_live_status = None
        repl.completer = None
        repl._file_history = None
        repl._do_expand_last = Mock()
        repl._apply_permission_mode_cycle = Mock()
        repl._bottom_toolbar = Mock(return_value="")
        repl._status_message = Mock(return_value="Thinking…")
        repl._enqueue_prompt = Mock()
        goal_loop = asyncio.new_event_loop()
        repl._get_chat_loop = Mock(return_value=goal_loop)

        continuation = SimpleNamespace(messages=[create_user_message("continue the goal", isMeta=True)])
        goal_runtime = Mock()
        goal_runtime.continue_if_idle.return_value = continuation
        goal_runtime.claim_continuation.return_value = True

        async def fake_run_query_as_agent_loop(**kwargs):
            live_status = FakeLiveStatus.instances[-1]
            self.assertIs(repl._im_active_cancel, live_status.kwargs["on_cancel"])
            controller = kwargs["abort_controller"]
            live_status.kwargs["on_cancel"]()
            self.assertTrue(controller.signal.aborted)
            self.assertEqual(controller.signal.reason, "user_interrupt")
            self.assertTrue(repl._interrupt_active_chat_from_im())
            raise AbortError("user_interrupt")

        try:
            with (
                patch("clawcodex_ext.repl.core._load_heavy_runtime"),
                patch("clawcodex_ext.repl.core.AbortController", AbortController, create=True),
                patch("clawcodex_ext.repl.core.LiveStatus", FakeLiveStatus, create=True),
                patch("clawcodex_ext.repl.core._pt_patch_stdout", return_value=nullcontext()),
                patch(
                    "clawcodex_ext.repl.core.resolve_output_style",
                    return_value=SimpleNamespace(prompt=""),
                    create=True,
                ),
                patch(
                    "clawcodex_ext.goal.runtime.goal_runtime_for_context",
                    return_value=goal_runtime,
                ),
                patch(
                    "clawcodex_ext.query.agent_loop_compat.build_effective_system_prompt",
                    return_value="system prompt",
                ),
                patch(
                    "clawcodex_ext.query.agent_loop_compat.run_query_as_agent_loop",
                    side_effect=fake_run_query_as_agent_loop,
                ),
            ):
                completed = repl._continue_goal_if_idle()
        finally:
            goal_loop.close()

        self.assertFalse(completed)
        self.assertEqual(repl._last_chat_outcome, "cancelled")
        self.assertIs(repl.tool_context.abort_controller, previous_controller)
        self.assertIsNone(repl._im_active_cancel)
        self.assertIsNone(repl._active_live_status)
        self.assertIn(
            "[warning]Cancelling…[/warning]",
            FakeLiveStatus.instances[-1].updates,
        )
        self.assertFalse(any("Error:" in str(call.args[0]) for call in repl.console.print.call_args_list if call.args))
        repl.session.save_transcript.assert_not_called()

    def test_forked_skill_result_is_rendered_without_second_model_query(self):
        from clawcodex_ext.command_system.engine import CommandResult

        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.console = Mock()
        repl.chat = Mock()
        repl._engine_messages = []

        handled = repl._handle_command_result(
            CommandResult.success_assistant(
                "verify",
                "runtime evidence\nVERDICT: PASS",
            )
        )

        self.assertTrue(handled)
        repl.chat.assert_not_called()
        self.assertEqual(len(repl._engine_messages), 1)
        self.assertEqual(repl._engine_messages[0].role, "assistant")
        self.assertIn("VERDICT: PASS", repr(repl._engine_messages[0].content))
        self.assertTrue(
            any(args and isinstance(args[0], Markdown) for args, _kwargs in repl.console.print.call_args_list)
        )

    def test_handle_command_goal_starts_continuation_without_live_provider(self):
        goal_home = Path(self.temp_dir) / "goal-home"

        with patch.dict(os.environ, {"CLAWCODEX_HOME": str(goal_home)}, clear=False):
            with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
                with patch("src.agent.Session.create") as mock_session_factory:
                    mock_session = Mock()
                    mock_session.session_id = "repl-goal-session"
                    mock_session.conversation = Conversation()
                    mock_session.save_transcript = Mock()
                    mock_session_factory.return_value = mock_session

                    with (
                        patch("src.providers.get_provider_class") as mock_provider_class,
                        patch(
                            "clawcodex_ext.repl.core.get_provider_class",
                            mock_provider_class,
                            create=True,
                        ),
                    ):
                        mock_provider = Mock()
                        mock_provider.model = "glm-4.5"
                        mock_provider.chat_stream_response.side_effect = NotImplementedError()
                        mock_provider.chat.side_effect = [
                            ChatResponse(
                                content="Still working toward the active goal.",
                                model="test",
                                usage={"input_tokens": 2, "output_tokens": 2},
                                finish_reason="end_turn",
                                tool_uses=None,
                            ),
                            ChatResponse(
                                content="Goal complete.",
                                model="test",
                                usage={"input_tokens": 3, "output_tokens": 2},
                                finish_reason="end_turn",
                                tool_uses=None,
                            ),
                        ]
                        mock_provider.chat_async = AsyncMock(
                            side_effect=[
                                ChatResponse(
                                    content=('{"met": false, "reason": "Implementation is still in progress."}'),
                                    model="test",
                                    usage={},
                                    finish_reason="end_turn",
                                    tool_uses=None,
                                ),
                                ChatResponse(
                                    content=('{"met": true, "reason": "The implementation is complete."}'),
                                    model="test",
                                    usage={},
                                    finish_reason="end_turn",
                                    tool_uses=None,
                                ),
                            ]
                        )
                        mock_provider_class.return_value = Mock(return_value=mock_provider)

                        repl = ClawcodexREPL(provider_name="glm", stream=False)
                        repl.console.print = Mock()

                        repl.handle_command("/goal implement smoke")

        service = repl.tool_context.goal_service
        goal = service.get_goal("repl-goal-session")
        self.assertIsNotNone(goal)
        self.assertEqual(goal.status.value, "complete")
        self.assertEqual(mock_provider.chat.call_count, 2)
        self.assertEqual(mock_provider.chat_async.await_count, 2)
        rendered = "\n".join(
            args[0].markup if args and isinstance(args[0], Markdown) else str(args[0])
            for args, _kwargs in repl.console.print.call_args_list
            if args
        )
        self.assertIn("Goal set: implement smoke", rendered)
        self.assertIn("Goal complete.", rendered)

    def test_handle_command_goal_lifecycle_smoke_without_live_provider(self):
        goal_home = Path(self.temp_dir) / "goal-lifecycle-home"

        with patch.dict(os.environ, {"CLAWCODEX_HOME": str(goal_home)}, clear=False):
            with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
                with patch("src.agent.Session.create") as mock_session_factory:
                    mock_session = Mock()
                    mock_session.session_id = "repl-goal-lifecycle-session"
                    mock_session.conversation = Conversation()
                    mock_session.save_transcript = Mock()
                    mock_session_factory.return_value = mock_session

                    with (
                        patch("src.providers.get_provider_class") as mock_provider_class,
                        patch(
                            "clawcodex_ext.repl.core.get_provider_class",
                            mock_provider_class,
                            create=True,
                        ),
                    ):
                        mock_provider = Mock()
                        mock_provider.model = "glm-4.5"
                        mock_provider.chat_stream_response.side_effect = NotImplementedError()
                        mock_provider.chat.side_effect = [
                            ChatResponse(
                                content="First goal achieved.",
                                model="test",
                                usage={"input_tokens": 1, "output_tokens": 1},
                                finish_reason="end_turn",
                                tool_uses=None,
                            ),
                            ChatResponse(
                                content="Replacement goal achieved.",
                                model="test",
                                usage={"input_tokens": 1, "output_tokens": 1},
                                finish_reason="end_turn",
                                tool_uses=None,
                            ),
                        ]
                        mock_provider.chat_async = AsyncMock(
                            side_effect=[
                                ChatResponse(
                                    content=('{"met": true, "reason": "The first goal is complete."}'),
                                    model="test",
                                    usage={},
                                    finish_reason="end_turn",
                                    tool_uses=None,
                                ),
                                ChatResponse(
                                    content=('{"met": true, "reason": "The replacement goal is complete."}'),
                                    model="test",
                                    usage={},
                                    finish_reason="end_turn",
                                    tool_uses=None,
                                ),
                            ]
                        )
                        mock_provider_class.return_value = Mock(return_value=mock_provider)

                        repl = ClawcodexREPL(provider_name="glm", stream=False)
                        repl.console.print = Mock()
                        repl._print_transient_text = Mock(side_effect=lambda text, **_kwargs: repl.console.print(text))

                        repl.handle_command("/goal implement lifecycle smoke")
                        repl.handle_command("/goal")
                        repl.handle_command("/goal replacement condition")
                        repl.handle_command("/goal")
                        repl.handle_command("/goal stop")
                        repl.handle_command("/clear")
                        repl.handle_command("/goal")

        service = repl.tool_context.goal_service
        self.assertIsNone(service.get_goal("repl-goal-lifecycle-session"))
        self.assertEqual(mock_provider.chat.call_count, 2)
        self.assertEqual(mock_provider.chat_async.await_count, 2)
        rendered = "\n".join(
            args[0].markup if args and isinstance(args[0], Markdown) else str(args[0])
            for args, _kwargs in repl.console.print.call_args_list
            if args
        )
        self.assertIn("Goal set: implement lifecycle smoke", rendered)
        self.assertIn("First goal achieved.", rendered)
        self.assertIn("Goal set: replacement condition", rendered)
        self.assertIn("Replacement goal achieved.", rendered)
        self.assertIn("Conversation cleared.", rendered)
        self.assertIn("No goal set", rendered)

    def test_handle_command_tools_lists_registered_tools(self):
        """/tools must call ToolRegistry.list_tools() and print each name.

        Regression: the handler previously called the non-existent
        ``list_specs()`` and crashed with AttributeError on every invocation.
        """
        from types import SimpleNamespace

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create"):
                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")
                    repl.tool_registry = SimpleNamespace(
                        list_tools=lambda: [
                            SimpleNamespace(name="Bash"),
                            SimpleNamespace(name="Read"),
                        ]
                    )
                    repl.console.print = Mock()
                    repl.handle_command("/tools")

                    printed = " ".join(str(args[0]) for args, _ in repl.console.print.call_args_list if args)
                    self.assertIn("Available tools:", printed)
                    self.assertIn("Bash", printed)
                    self.assertIn("Read", printed)

    def test_handle_command_context_resolves_tool_descriptions_to_strings(self):
        """/context must populate tool_schemas with string descriptions.

        Regression: Tool.description is a Callable[[dict], str], not a string.
        The handler previously stuffed the callable straight into the
        ``tool_schemas`` payload; downstream consumers expecting a string
        would receive a function reference.
        """
        from types import SimpleNamespace
        from src.tool_system.build_tool import build_tool
        from clawcodex_ext.tool_system.protocol import ToolResult

        def _noop(_input, _ctx):
            return ToolResult(name="X", output={}, is_error=False)

        real_tool = build_tool(
            name="Bash",
            input_schema={"type": "object"},
            call=_noop,
            description=lambda _i: "Run a shell command",
        )

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create"):
                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")
                    repl.tool_registry = SimpleNamespace(list_tools=lambda: [real_tool])
                    repl._try_execute_new_command = Mock(return_value=(False, None))
                    repl.console.print = Mock()

                    repl.handle_command("/context")

                    schemas = repl.command_context.config["tool_schemas"]
                    self.assertEqual(len(schemas), 1)
                    self.assertEqual(schemas[0]["name"], "Bash")
                    self.assertIsInstance(schemas[0]["description"], str)
                    self.assertEqual(schemas[0]["description"], "Run a shell command")
                    self.assertEqual(schemas[0]["input_schema"], {"type": "object"})

    def test_chat_uses_true_api_stream_for_simple_prompt(self):
        """Simple prompts should use provider.chat_stream when stream mode is enabled."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("src.agent.Session.create") as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session

                with (
                    patch("src.providers.get_provider_class") as mock_provider_class,
                    patch(
                        "clawcodex_ext.repl.core.get_provider_class",
                        mock_provider_class,
                        create=True,
                    ),
                ):
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider.chat_stream.return_value = iter(["你", "好"])
                    mock_provider_class.return_value = Mock(return_value=mock_provider)

                    repl = ClawcodexREPL(provider_name="glm", stream=True)
                    repl.console.print = Mock()

                    repl.chat("你是谁")

                    mock_provider.chat_stream.assert_called_once()
                    self.assertFalse(
                        any(
                            args and isinstance(args[0], Markdown)
                            for args, _kwargs in repl.console.print.call_args_list
                        )
                    )
                    self.assertEqual(len(mock_session.conversation.messages), 2)
                    self.assertEqual(mock_session.conversation.messages[1].role, "assistant")
                    last_content = mock_session.conversation.messages[1].content
                    if isinstance(last_content, list):
                        self.assertEqual(last_content[0].text, "你好")
                    else:
                        self.assertEqual(last_content, "你好")

    def test_repl_goal_footer_is_visible_only_while_active(self):
        """The prompt footer mirrors Claude Code's active-goal indicator."""
        from types import SimpleNamespace

        from clawcodex_ext.goal.model import ThreadGoalStatus
        from clawcodex_ext.goal.service import GoalService
        from clawcodex_ext.goal.store import GoalStore

        service = GoalService(store=GoalStore(Path(self.temp_dir) / "footer-goals.sqlite"))
        goal = service.replace_goal("footer-session", "finish footer smoke")
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.tool_context = SimpleNamespace(
            goal_service=service,
            goal_thread_id="footer-session",
            session_id="footer-session",
        )

        with patch("clawcodex_ext.repl.core.time.monotonic", side_effect=[100.0, 113.0]):
            self.assertEqual(repl._goal_footer_status(), "◎ /goal active (0s)")
            self.assertEqual(repl._goal_footer_status(), "◎ /goal active (13s)")

        service.update_goal(
            "footer-session",
            ThreadGoalStatus.COMPLETE,
            expected_goal_id=goal.goal_id,
        )
        self.assertIsNone(repl._goal_footer_status())

    def test_direct_stream_fails_closed_when_goal_state_lookup_fails(self):
        from types import SimpleNamespace

        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.stream = True
        service = Mock()
        service.get_goal.side_effect = RuntimeError("goal database unavailable")
        repl.tool_context = SimpleNamespace(
            goal_service=service,
            goal_thread_id="footer-session",
            session_id="footer-session",
        )

        self.assertFalse(repl._should_try_direct_stream("continue working"))

    def test_chat_direct_stream_accounts_goal_usage(self):
        """Direct-stream REPL turns must report provider usage to GoalRuntime."""

        class FakeGoalRuntime:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def on_turn_start(self, *, plan_mode: bool = False):
                self.calls.append(("start", plan_mode))
                return "goal-turn-1"

            def on_token_usage(self, turn_id, usage) -> None:
                self.calls.append(("usage", turn_id, usage))

            def on_turn_stop(self, turn_id) -> None:
                self.calls.append(("stop", turn_id))

            def on_turn_abort(self, turn_id) -> None:
                self.calls.append(("abort", turn_id))

            def on_turn_error(self, turn_id, error) -> None:
                self.calls.append(("error", turn_id, type(error).__name__))

        fake_goal_runtime = FakeGoalRuntime()

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("src.agent.Session.create") as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session.save_transcript = Mock()
                mock_session_factory.return_value = mock_session

                with (
                    patch("src.providers.get_provider_class") as mock_provider_class,
                    patch(
                        "clawcodex_ext.repl.core.get_provider_class",
                        mock_provider_class,
                        create=True,
                    ),
                ):
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider.chat_stream_response.return_value = ChatResponse(
                        content="goal accounted",
                        model="test",
                        usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
                        finish_reason="stop",
                    )
                    mock_provider_class.return_value = Mock(return_value=mock_provider)

                    repl = ClawcodexREPL(provider_name="glm", stream=True)
                    repl.console.print = Mock()
                    repl._continue_goal_if_idle = Mock(return_value=True)

                    with patch(
                        "clawcodex_ext.goal.runtime.goal_runtime_for_context",
                        return_value=fake_goal_runtime,
                    ):
                        repl.chat("hello there")

        self.assertEqual(
            fake_goal_runtime.calls,
            [
                ("start", False),
                (
                    "usage",
                    "goal-turn-1",
                    {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
                ),
                ("stop", "goal-turn-1"),
            ],
        )
        repl._continue_goal_if_idle.assert_called_once_with()

    def test_goal_runtime_threads_session_id_into_tool_context(self):
        """REPL goal commands and runtime accounting must share one thread id."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("src.agent.Session.create") as mock_session_factory:
                mock_session = Mock()
                mock_session.session_id = "test-session-id"
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session

                with (
                    patch("src.providers.get_provider_class") as mock_provider_class,
                    patch(
                        "clawcodex_ext.repl.core.get_provider_class",
                        mock_provider_class,
                        create=True,
                    ),
                    patch("clawcodex_ext.repl.core.PromptSession") as mock_prompt_session,
                ):
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = Mock(return_value=mock_provider)
                    mock_prompt_session.return_value = Mock(prompt=Mock(return_value=""))

                    repl = ClawcodexREPL(provider_name="glm", stream=True)

        self.assertEqual(repl.tool_context.session_id, "test-session-id")

    def test_chat_uses_query_engine_for_code_task(self):
        """Code-like prompts use the new QueryEngine path."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider.chat_stream_response.side_effect = NotImplementedError()
                    mock_provider.chat.return_value = ChatResponse(
                        content="Done reading README.",
                        model="test",
                        usage={"input_tokens": 10, "output_tokens": 5},
                        finish_reason="end_turn",
                        tool_uses=None,
                    )
                    mock_provider_class.return_value = Mock(return_value=mock_provider)

                    repl = ClawcodexREPL(provider_name="glm", stream=True)
                    repl.console.print = Mock()
                    repl.chat("请读取 README.md 并总结")

                    mock_provider.chat.assert_called()

    def test_chat_uses_query_engine_on_stream_init_failure(self):
        """If real streaming fails, fall back to QueryEngine."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session

                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider.chat_stream.side_effect = RuntimeError("stream unavailable")
                    mock_provider.chat_stream_response.side_effect = NotImplementedError()
                    mock_provider.chat.return_value = ChatResponse(
                        content="fallback response",
                        model="test",
                        usage={"input_tokens": 10, "output_tokens": 5},
                        finish_reason="end_turn",
                        tool_uses=None,
                    )
                    mock_provider_class.return_value = Mock(return_value=mock_provider)

                    repl = ClawcodexREPL(provider_name="glm", stream=True)
                    repl.console.print = Mock()
                    repl._continue_goal_if_idle = Mock(return_value=True)
                    repl.chat("你好呀")

                    mock_provider.chat_stream.assert_called_once()
                    mock_provider.chat.assert_called()
                    repl._continue_goal_if_idle.assert_called_once_with()

    def test_chat_pumps_cron_loop_during_engine_turn(self):
        """chat() must pump the IM gateway loop during engine turns."""
        import asyncio

        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._cron_loop = None
        fallback = repl._get_chat_loop()
        try:
            self.assertIsInstance(fallback, asyncio.AbstractEventLoop)

            cron_loop = asyncio.new_event_loop()
            repl._cron_loop = cron_loop
            try:
                self.assertIs(repl._get_chat_loop(), cron_loop)
            finally:
                cron_loop.close()
        finally:
            fallback.close()
            asyncio.set_event_loop(None)

    def test_handle_command_slash_shows_commands_and_skills(self):
        skills_dir = Path(self.temp_dir) / "skills"
        (skills_dir / "hello").mkdir(parents=True, exist_ok=True)
        (skills_dir / "hello" / "SKILL.md").write_text(
            "---\ndescription: say hello\n---\nHello\n",
            encoding="utf-8",
        )
        with patch.dict("os.environ", {"CLAWCODEX_SKILLS_DIR": str(skills_dir)}):
            with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
                with patch("clawcodex_ext.repl.core.Session.create"):
                    with patch("src.providers.get_provider_class") as mock_provider_class:
                        mock_provider = Mock()
                        mock_provider.model = "glm-4.5"
                        mock_provider_class.return_value = mock_provider

                        repl = ClawcodexREPL(provider_name="glm")
                        repl.console.print = Mock()
                        repl.handle_command("/")
                        rendered = "\n".join(
                            str(args[0]) for args, _kwargs in repl.console.print.call_args_list if args
                        )
                        self.assertIn("Available commands and skills", rendered)
                        self.assertIn("/hello", rendered)

    def test_handle_command_slash_prefix_filters(self):
        skills_dir = Path(self.temp_dir) / "skills"
        (skills_dir / "hello").mkdir(parents=True, exist_ok=True)
        (skills_dir / "hello" / "SKILL.md").write_text(
            "---\ndescription: say hello\n---\nHello\n",
            encoding="utf-8",
        )
        with patch.dict("os.environ", {"CLAWCODEX_SKILLS_DIR": str(skills_dir)}):
            with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
                with patch("clawcodex_ext.repl.core.Session.create"):
                    with patch("src.providers.get_provider_class") as mock_provider_class:
                        mock_provider = Mock()
                        mock_provider.model = "glm-4.5"
                        mock_provider_class.return_value = mock_provider

                        repl = ClawcodexREPL(provider_name="glm")
                        repl.console.print = Mock()
                        repl.handle_command("/he")
                        rendered = "\n".join(
                            str(args[0]) for args, _kwargs in repl.console.print.call_args_list if args
                        )
                        self.assertIn("/help", rendered)
                        self.assertIn("/hello", rendered)

    def test_handle_command_skill_invokes_skill_tool_and_chats_with_prompt(self):
        skills_dir = Path(self.temp_dir) / "skills"
        (skills_dir / "hello").mkdir(parents=True, exist_ok=True)
        (skills_dir / "hello" / "SKILL.md").write_text(
            "---\ndescription: say hello\narguments: [name]\n---\nHello $name\n",
            encoding="utf-8",
        )
        with patch.dict("os.environ", {"CLAWCODEX_SKILLS_DIR": str(skills_dir)}):
            with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
                with patch("clawcodex_ext.repl.core.Session.create"):
                    with patch("src.providers.get_provider_class") as mock_provider_class:
                        mock_provider = Mock()
                        mock_provider.model = "glm-4.5"
                        mock_provider_class.return_value = mock_provider

                        repl = ClawcodexREPL(
                            provider_name="glm",
                            permission_mode="bypassPermissions",
                            is_bypass_permissions_mode_available=True,
                        )
                        repl.chat = Mock()
                        handled = repl._try_run_skill_slash("/hello bob")
                        self.assertTrue(handled)
                        args, _kwargs = repl.chat.call_args
                        self.assertIn("Hello bob", args[0])

    def test_forked_skill_slash_renders_result_without_second_model_query(self):
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._built_in_commands = set()
        repl.tool_context = object()
        repl.console = Mock()
        repl.chat = Mock()
        repl._engine_messages = []

        result = SimpleNamespace(
            is_error=False,
            output={
                "success": True,
                "status": "fork",
                "commandName": "verify",
                "result": "runtime evidence\nVERDICT: PASS",
            },
        )
        with patch(
            "clawcodex_ext.tool_system.tools.skill.run_user_invoked_skill",
            return_value=result,
        ) as invoke:
            handled = repl._try_run_skill_slash("/verify target.txt")

        self.assertTrue(handled)
        invoke.assert_called_once_with("verify", "target.txt", repl.tool_context)
        repl.chat.assert_not_called()
        self.assertEqual(len(repl._engine_messages), 1)
        self.assertEqual(repl._engine_messages[0].role, "assistant")
        self.assertIn("VERDICT: PASS", repr(repl._engine_messages[0].content))

    def test_save_session(self):
        """Test session saving."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = "test_session_123"
                mock_session.return_value = mock_session_instance

                with patch("src.providers.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")
                    repl.save_session()

                    mock_session_instance.save.assert_called_once()

    def test_load_session(self):
        """Test session loading."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = "current_session"
                mock_session.return_value = mock_session_instance

                with patch("src.providers.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    loaded_session = Mock()
                    loaded_session.session_id = "loaded_session_123"
                    loaded_session.provider = "glm"
                    loaded_session.model = "glm-4.5"
                    loaded_session.conversation = Mock()
                    loaded_session.conversation.messages = []
                    with patch(
                        "src.agent.Session.resume",
                        return_value=loaded_session,
                    ):
                        repl = ClawcodexREPL(provider_name="glm")
                        repl.load_session("loaded_session_123")

                        self.assertEqual(repl.session.session_id, "loaded_session_123")
                        # _engine_messages should be populated (empty list in this trivial case)
                        self.assertEqual(repl._engine_messages, [])

    def test_load_session_populates_engine_messages(self):
        """load_session must populate _engine_messages from the restored
        conversation so the next chat() call's QueryEngine sees the full
        history rather than starting with an empty mutable-message list.
        """
        from clawcodex_ext.types.content_blocks import TextBlock
        from clawcodex_ext.types.messages import AssistantMessage, UserMessage

        loaded_session = Mock()
        loaded_session.session_id = "resumed_session"
        loaded_session.provider = "glm"
        loaded_session.model = "glm-4.5"
        loaded_session.conversation = Mock()
        loaded_session.conversation.messages = [
            UserMessage(content="Hello"),
            AssistantMessage(content=[TextBlock(text="Hi there")]),
        ]

        # Mock Session.resume (the classmethod called by load_session)
        # by patching the src.agent module where the import resolves.
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("src.agent.Session") as mock_session_class:
                mock_session_instance = Mock()
                mock_session_instance.session_id = "current_session"
                mock_session_class.create.return_value = mock_session_instance
                mock_session_class.resume.return_value = loaded_session

                with patch("src.providers.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawcodexREPL(provider_name="glm")
                    # Clear default _engine_messages then load
                    repl._engine_messages = []
                    repl.load_session("resumed_session")

                    self.assertEqual(repl.session.session_id, "resumed_session")
                    self.assertEqual(repl.tool_context.session_id, "resumed_session")
                    # _engine_messages must contain the restored messages
                    self.assertEqual(len(repl._engine_messages), 2)
                    self.assertIs(
                        repl._engine_messages[0],
                        loaded_session.conversation.messages[0],
                    )
                    self.assertIs(
                        repl._engine_messages[1],
                        loaded_session.conversation.messages[1],
                    )

    def test_constructor_with_resumed_session_populates_engine_messages(self):
        """RuntimeContext passes an already-resumed session into the REPL."""
        from clawcodex_ext.repl.app import ClawCodexExtREPL
        from clawcodex_ext.types.content_blocks import TextBlock
        from clawcodex_ext.types.messages import AssistantMessage, UserMessage

        resumed_session = Mock()
        resumed_session.session_id = "resumed_session"
        resumed_session.provider = "glm"
        resumed_session.model = "glm-4.5"
        resumed_session.conversation = Mock()
        resumed_session.conversation.messages = [
            UserMessage(content="讲一个小知识"),
            AssistantMessage(content=[TextBlock(text="Python __slots__ can save memory.")]),
        ]

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("src.providers.get_provider_class") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.model = "glm-4.5"
                mock_provider_class.return_value = mock_provider

                repl = ClawCodexExtREPL(
                    provider_name="glm",
                    resume_session_id="resumed_session",
                    session=resumed_session,
                    provider=mock_provider,
                )

        self.assertEqual(repl.session.session_id, "resumed_session")
        self.assertEqual(len(repl._engine_messages), 2)
        self.assertIs(repl._engine_messages[0], resumed_session.conversation.messages[0])
        self.assertIs(repl._engine_messages[1], resumed_session.conversation.messages[1])

    def test_ext_repl_resume_resets_active_goal_progress(self):
        """An explicit resume keeps the condition but restarts live metrics."""
        from clawcodex_ext.goal.service import GoalService
        from clawcodex_ext.goal.store import GoalStore
        from clawcodex_ext.repl.app import ClawCodexExtREPL
        from clawcodex_ext.tool_system.context import ToolContext
        from clawcodex_ext.tool_system.registry import ToolRegistry

        resumed_session = Mock()
        resumed_session.session_id = "resumed-goal-session"
        resumed_session.provider = "glm"
        resumed_session.model = "glm-4.5"
        resumed_session.conversation = Conversation()
        service = GoalService(store=GoalStore(Path(self.temp_dir) / "goals.sqlite"))
        goal = service.set_goal("resumed-goal-session", "keep working")
        service.account_usage(
            "resumed-goal-session",
            expected_goal_id=goal.goal_id,
            token_delta=17,
            elapsed_seconds=6,
        )
        context = ToolContext(
            workspace_root=Path(self.temp_dir),
            goal_service=service,
        )
        provider = Mock(model="glm-4.5")

        repl = ClawCodexExtREPL(
            provider_name="glm",
            resume_session_id="resumed-goal-session",
            session=resumed_session,
            provider=provider,
            tool_registry=ToolRegistry(),
            tool_context=context,
            workspace_root=Path(self.temp_dir),
        )

        restored = service.get_goal("resumed-goal-session")
        self.assertEqual(repl.tool_context.session_id, "resumed-goal-session")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.goal_id, goal.goal_id)
        self.assertEqual(restored.objective, "keep working")
        self.assertEqual(restored.tokens_used, 0)
        self.assertEqual(restored.time_used_seconds, 0)

    def test_ext_repl_uses_structured_permission_handler_by_default(self):
        """The downstream REPL entrypoint must preserve permission suggestions."""
        from clawcodex_ext.repl.app import ClawCodexExtREPL

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("src.providers.get_provider_class") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.model = "glm-4.5"
                mock_provider_class.return_value = mock_provider
                session = Mock()
                session.session_id = "ext-permission-session"
                session.provider = "glm"
                session.model = "glm-4.5"
                session.conversation = Mock()
                session.conversation.messages = []

                repl = ClawCodexExtREPL(
                    provider_name="glm",
                    session=session,
                    provider=mock_provider,
                )

        self.assertEqual(
            repl.tool_context.permission_handler.__name__,
            "_handle_permission_ask_request",
        )
        self.assertEqual(
            repl.tool_context.default_permission_handler.__name__,
            "_handle_permission_ask_request",
        )

    def test_load_nonexistent_session(self):
        """Test loading a session that doesn't exist."""
        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("clawcodex_ext.repl.core.Session.create") as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = "current_session"
                mock_session.return_value = mock_session_instance

                with patch("src.providers.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    with patch("clawcodex_ext.repl.core.Session.load", return_value=None):
                        repl = ClawcodexREPL(provider_name="glm")
                        original_session = repl.session

                        repl.load_session("nonexistent")

                        # Session should not change
                        self.assertEqual(repl.session, original_session)

    def test_permission_prompt_is_serialized(self):
        """Concurrent permission checks should not open overlapping prompts."""
        with (
            patch(
                "clawcodex_ext.repl.core.get_provider_config",
                return_value={
                    "api_key": "test_api_key_12345678",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "default_model": "glm-4.5",
                },
            ),
            patch("clawcodex_ext.repl.core.PromptSession") as mock_prompt_session,
        ):
            mock_prompt_session.return_value = Mock(prompt=Mock(return_value=""))
            with patch("clawcodex_ext.repl.core.Session.create"):
                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    repl = ClawcodexREPL(provider_name="glm")
                    repl.console.print = Mock()

                    in_prompt = 0
                    max_in_prompt = 0
                    counter_lock = threading.Lock()

                    def fake_input(_prompt: str) -> str:
                        nonlocal in_prompt, max_in_prompt
                        with counter_lock:
                            in_prompt += 1
                            if in_prompt > max_in_prompt:
                                max_in_prompt = in_prompt
                        time.sleep(0.03)
                        with counter_lock:
                            in_prompt -= 1
                        return "1"

                    repl._safe_input = fake_input  # type: ignore[assignment]

                    with patch("clawcodex_ext.repl.core.get_selection_mode", return_value="number"):
                        t1 = threading.Thread(
                            target=repl._handle_permission_request,
                            args=("Grep", "Claude wants to use Grep. Allow?", None),
                        )
                        t2 = threading.Thread(
                            target=repl._handle_permission_request,
                            args=("Read", "Claude wants to use Read. Allow?", None),
                        )
                        t1.start()
                        t2.start()
                        t1.join()
                        t2.join()

                    self.assertEqual(max_in_prompt, 1)

    def test_permission_prompt_forwards_options_to_im_controller(self):
        """IM-driven turns should mirror permission options to the gateway."""
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._permission_prompt_lock = threading.Lock()
        repl._permission_decision_cache = {}
        repl._current_status = None
        repl.console = Mock()
        repl.tool_context = Mock(allow_docs=False)
        repl._safe_input = Mock(return_value="1")

        calls = []

        class _FakeImReply:
            def send_permission_prompt(self, **kwargs):
                calls.append(kwargs)
                return True

        repl._im_reply_controller = _FakeImReply()

        with patch("clawcodex_ext.repl.core.get_selection_mode", return_value="number"):
            allowed, cached = repl._handle_permission_request("Bash", "Claude wants to delete files. Allow?", None)

        self.assertTrue(allowed)
        self.assertFalse(cached)
        self.assertEqual(calls[0]["message"], "Claude wants to delete files. Allow?")
        self.assertEqual(
            calls[0]["options"],
            [("y", "Yes, allow this action"), ("n", "No, deny this action")],
        )

    def test_permission_prompt_plain_allow_is_not_cached_by_tool_name(self):
        """Plain "allow this action" must not authorize later calls implicitly."""
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._permission_prompt_lock = threading.Lock()
        repl._permission_decision_cache = {}
        repl._current_status = None
        repl.console = Mock()
        repl.tool_context = Mock(allow_docs=False)
        repl._safe_input = Mock(side_effect=["1", "2"])
        repl._im_reply_controller = None

        with patch("clawcodex_ext.repl.core.get_selection_mode", return_value="number"):
            first_allowed, first_cached = repl._handle_permission_request(
                "Read",
                "Claude wants to use Read for /private/etc/hosts. Allow?",
                None,
            )
            second_allowed, second_cached = repl._handle_permission_request(
                "Read",
                "Claude wants to use Read for /private/etc/protocols. Allow?",
                None,
            )

        self.assertTrue(first_allowed)
        self.assertFalse(first_cached)
        self.assertFalse(second_allowed)
        self.assertFalse(second_cached)
        self.assertEqual(repl._safe_input.call_count, 2)

    def test_permission_ask_request_returns_chosen_session_updates(self):
        """Session suggestion choices must reach the tool registry."""
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._permission_prompt_lock = threading.Lock()
        repl._permission_decision_cache = {}
        repl._current_status = None
        repl.console = Mock()
        repl.tool_context = Mock(allow_docs=True)
        repl._safe_input = Mock(return_value="2")
        repl._im_reply_controller = None

        update = PermissionUpdateSetMode(destination="session", mode="acceptEdits")
        request = PermissionAskRequest(
            tool_name="Write",
            message="Claude wants to use Write. Allow?",
            tool_input={"file_path": "/tmp/example.md", "content": "x"},
            suggestions=(update,),
        )

        with patch("clawcodex_ext.repl.core.get_selection_mode", return_value="number"):
            reply = repl._handle_permission_ask_request(request)

        self.assertEqual(reply.behavior, "allow")
        self.assertEqual(reply.chosen_updates, (update,))

    def test_permission_ask_request_marks_session_choice_allowed_for_im(self):
        """Rich IM channels receive stable allow/deny semantics for session choices."""
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._permission_prompt_lock = threading.Lock()
        repl._current_status = None
        repl.console = Mock()
        repl.tool_context = Mock(allow_docs=True)

        class _FakeClient:
            def peek_reply_origin(self):
                return "feishu:dm:cli_app:ou_allowed"

        repl._im_reply_controller = Mock(_client=_FakeClient())
        repl._wait_im_permission_choice = Mock(return_value="s")
        update = PermissionUpdateSetMode(destination="session", mode="acceptEdits")
        request = PermissionAskRequest(
            tool_name="Bash",
            message="Claude wants to use Bash. Allow?",
            tool_input={"command": "pwd"},
            suggestions=(update,),
        )

        reply = repl._handle_permission_ask_request(request)

        self.assertEqual(reply.behavior, "allow")
        self.assertEqual(reply.chosen_updates, (update,))
        self.assertEqual(
            repl._wait_im_permission_choice.call_args.kwargs["allow_choices"],
            {"y", "s"},
        )

    def test_permission_ask_request_ignores_hidden_enable_alias(self):
        """Typing hidden aliases must not allow options absent from the menu."""
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._permission_prompt_lock = threading.Lock()
        repl._permission_decision_cache = {}
        repl._current_status = None
        repl.console = Mock()
        repl.tool_context = Mock(allow_docs=True)
        repl._safe_input = Mock(return_value="enable")
        repl._im_reply_controller = None

        request = PermissionAskRequest(
            tool_name="Bash",
            message="Claude wants to use Bash. Allow?",
            tool_input={"command": "pwd"},
            suggestions=(),
        )

        with patch("clawcodex_ext.repl.core.get_selection_mode", return_value="number"):
            reply = repl._handle_permission_ask_request(request)

        self.assertEqual(reply.behavior, "deny")

    def test_permission_ask_request_ignores_hidden_session_alias(self):
        """The session alias only works when session updates are available."""
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._permission_prompt_lock = threading.Lock()
        repl._permission_decision_cache = {}
        repl._current_status = None
        repl.console = Mock()
        repl.tool_context = Mock(allow_docs=True)
        repl._safe_input = Mock(return_value="session")
        repl._im_reply_controller = None

        request = PermissionAskRequest(
            tool_name="Bash",
            message="Claude wants to use Bash. Allow?",
            tool_input={"command": "pwd"},
            suggestions=(),
        )

        with patch("clawcodex_ext.repl.core.get_selection_mode", return_value="number"):
            reply = repl._handle_permission_ask_request(request)

        self.assertEqual(reply.behavior, "deny")

    def test_permission_prompt_im_branch_uses_wechat_reply_choice(self):
        """1b: when the turn is IM-driven (peek_reply_origin non-None), the
        permission decision comes from the WeChat reply via
        ``_wait_im_permission_choice`` and feeds the SAME parser as the
        keyboard path — ``1``/``y`` allow, ``2``/``n`` deny.
        """
        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl._permission_prompt_lock = threading.Lock()
        repl._permission_decision_cache = {}
        repl._current_status = None
        repl.console = Mock()
        repl.tool_context = Mock(allow_docs=False)

        class _FakeClient:
            def peek_reply_origin(self):
                return "wechat:direct:a:u1"

        class _FakeImReply:
            _client = _FakeClient()

            def send_permission_prompt(self, **kwargs):
                return True

        repl._im_reply_controller = _FakeImReply()

        # allow via WeChat reply "1"
        repl._wait_im_permission_choice = Mock(return_value="1")
        allowed, cached = repl._handle_permission_request("Bash", "Claude wants to delete files. Allow?", None)
        self.assertTrue(allowed)
        self.assertFalse(cached)

        # deny via WeChat reply "2"
        repl._permission_decision_cache.clear()
        repl._wait_im_permission_choice = Mock(return_value="2")
        allowed, cached = repl._handle_permission_request("Bash", "Claude wants to delete files. Allow?", None)
        self.assertFalse(allowed)
        self.assertFalse(cached)

    def test_wait_im_permission_choice_lazy_inits_lock_and_resolves_via_probe(self):
        """_wait_im_permission_choice must work even when subclass init skipped the lock."""
        import os
        import threading
        import time as _time

        from clawcodex_ext.frontend.repl_extensions import _handle_im_permission_reply

        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.console = Mock()

        class _FakeImReply:
            def send_permission_prompt(self, **kwargs):
                return True

        repl._im_reply_controller = _FakeImReply()

        os.environ["CLAWCODEX_IM_PERMISSION_TIMEOUT"] = "3"
        try:
            result: dict = {}

            def _waiter():
                try:
                    result["choice"] = repl._wait_im_permission_choice(
                        message="Bash wants to delete files. Allow?",
                        options=[("y", "Yes, allow"), ("n", "No, deny")],
                    )
                except Exception as exc:  # noqa: BLE001
                    result["error"] = exc

            t = threading.Thread(target=_waiter, daemon=True)
            t.start()
            _time.sleep(0.3)
            _handle_im_permission_reply(repl, "1")
            t.join(timeout=3.0)
        finally:
            os.environ.pop("CLAWCODEX_IM_PERMISSION_TIMEOUT", None)

        self.assertNotIn("error", result, f"_wait_im_permission_choice raised: {result.get('error')}")
        self.assertEqual(result.get("choice"), "1")
        self.assertIsNotNone(getattr(repl, "_im_permission_lock", None))

    def test_interrupt_active_chat_from_im_invokes_engine_cancel(self):
        """IM /stop should use the active cancel hook or abort-controller fallback."""
        from clawcodex_ext.utils.abort_controller import AbortController

        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        engine_interrupts: list[str] = []

        def _cancel_engine():
            engine_interrupts.append("interrupt")

        repl._im_active_cancel = _cancel_engine
        repl._direct_abort_controller = None
        self.assertTrue(repl._interrupt_active_chat_from_im())
        self.assertEqual(engine_interrupts, ["interrupt"])

        repl2 = ClawcodexREPL.__new__(ClawcodexREPL)
        repl2._im_active_cancel = None
        controller = AbortController()
        repl2._direct_abort_controller = controller
        self.assertFalse(controller.signal.aborted)
        self.assertTrue(repl2._interrupt_active_chat_from_im())
        self.assertTrue(controller.signal.aborted)

        repl3 = ClawcodexREPL.__new__(ClawcodexREPL)
        repl3._im_active_cancel = None
        repl3._direct_abort_controller = None
        self.assertFalse(repl3._interrupt_active_chat_from_im())

    def test_permission_prompt_plain_allow_prompts_each_time(self):
        """Plain allow does not cache later prompts for the same tool."""
        with (
            patch(
                "clawcodex_ext.repl.core.get_provider_config",
                return_value={
                    "api_key": "test_api_key_12345678",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "default_model": "glm-4.5",
                },
            ),
            patch("clawcodex_ext.repl.core.PromptSession") as mock_prompt_session,
        ):
            mock_prompt_session.return_value = Mock(prompt=Mock(return_value=""))
            with patch("clawcodex_ext.repl.core.Session.create"):
                with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    repl = ClawcodexREPL(provider_name="glm")
                    repl.console.print = Mock()

                    prompt_calls = 0

                    def fake_input(_prompt: str) -> str:
                        nonlocal prompt_calls
                        prompt_calls += 1
                        return "1"

                    repl._safe_input = fake_input  # type: ignore[assignment]

                    with patch("clawcodex_ext.repl.core.get_selection_mode", return_value="number"):
                        first = repl._handle_permission_request(
                            "Grep",
                            "Claude wants to use Grep. Allow?",
                            None,
                        )
                        second = repl._handle_permission_request(
                            "Grep",
                            "Claude wants to use Grep. Allow?",
                            None,
                        )

                    self.assertEqual(first, (True, False))
                    self.assertEqual(second, (True, False))
                    self.assertEqual(prompt_calls, 2)

    def test_ask_user_questions_other_branch_uses_safe_input(self):
        """Regression: the 'Other' follow-up must read through ``_safe_input``.

        Bug: ``src/repl/core.py:1037`` used bare ``input("Other > ")``. When
        ``chat()`` had mounted a ``LiveStatus`` spinner (its
        ``prompt_toolkit.Application`` runs on a background thread and reads
        from the same TTY), the foreground ``input()`` raced the spinner —
        keystrokes for the custom text were eaten by the spinner's buffer
        and the session appeared stuck, with Ctrl+C / Ctrl+D unreachable
        because the spinner captured them first.

        The fix routes the 'Other' follow-up through ``_safe_input`` (same
        path used by the surrounding ``Select >`` and the permission
        prompt), which pauses ``LiveStatus`` around the read.
        """
        with (
            patch(
                "clawcodex_ext.repl.core.get_provider_config",
                return_value={
                    "api_key": "test_api_key_12345678",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "default_model": "glm-4.5",
                },
            ),
            patch("clawcodex_ext.repl.core.PromptSession") as mock_prompt_session,
            patch("clawcodex_ext.repl.core.Session.create"),
            patch("src.providers.runtime.build_provider_from_config") as mock_build_provider,
        ):
            mock_prompt_session.return_value = Mock(prompt=Mock(return_value=""))
            mock_provider = Mock()
            mock_provider.model = "glm-4.5"
            mock_build_provider.return_value = mock_provider
            repl = ClawcodexREPL(provider_name="glm")
            repl.console.print = Mock()

            prompts_seen: list[str] = []

            def fake_safe_input(prompt: str) -> str:
                prompts_seen.append(prompt)
                if len(prompts_seen) == 1:
                    return "4"  # synthetic "Other" option
                return "心跳一下 ⏰"

            repl._safe_input = fake_safe_input  # type: ignore[assignment]

            # If the bug regresses, bare input() is reached for the 'Other'
            # follow-up and the AssertionError makes the test fail loudly.
            with (
                patch("clawcodex_ext.repl.core.get_selection_mode", return_value="number"),
                patch(
                    "builtins.input",
                    side_effect=AssertionError(
                        "bare input() reached the 'Other' branch — regression; "
                        "'Other' must read through _safe_input so LiveStatus "
                        "can pause the spinner."
                    ),
                ),
            ):
                answers = repl._ask_user_questions(
                    [
                        {
                            "question": "每分钟发送的消息内容应该是什么？",
                            "header": "msg",
                            "options": [
                                {
                                    "label": "简单心跳提醒",
                                    "description": "每分钟发送一条简短的状态消息",
                                },
                                {
                                    "label": "健康检查报告",
                                    "description": "汇报工作树与运行中的任务",
                                },
                                {"label": "自定义内容", "description": "我会告诉你具体想发的内容"},
                            ],
                            "multiSelect": False,
                        }
                    ]
                )

            self.assertEqual(
                answers,
                {
                    "每分钟发送的消息内容应该是什么？": "心跳一下 ⏰",
                },
            )
            # Both prompts went through _safe_input, in order.
            self.assertEqual(prompts_seen, ["Select > ", "Other > "])


class TestConversation(unittest.TestCase):
    """Test conversation management."""

    def test_add_message(self):
        """Test adding messages to conversation."""
        conv = Conversation()
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")

        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].role, "user")
        self.assertEqual(conv.messages[0].content, "Hello")
        self.assertEqual(conv.messages[1].role, "assistant")

    def test_max_history(self):
        """Test max history limit."""
        conv = Conversation(max_history=3)

        # Add 5 messages
        for i in range(5):
            conv.add_message("user", f"Message {i}")

        # Should only keep last 3
        self.assertEqual(len(conv.messages), 3)
        self.assertEqual(conv.messages[0].content, "Message 2")
        self.assertEqual(conv.messages[2].content, "Message 4")

    def test_get_messages(self):
        """Test getting messages in API format."""
        conv = Conversation()
        conv.add_message("user", "Test")
        conv.add_message("assistant", "Response")

        messages = conv.get_messages()

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "user", "content": "Test"})
        self.assertEqual(messages[1]["role"], "assistant")
        content = messages[1]["content"]
        if isinstance(content, list):
            self.assertEqual(content[0]["text"], "Response")
        else:
            self.assertEqual(content, "Response")

    def test_clear(self):
        """Test clearing conversation."""
        conv = Conversation()
        conv.add_message("user", "Test")
        conv.clear()

        self.assertEqual(len(conv.messages), 0)

    def test_serialization(self):
        """Test conversation serialization."""
        conv = Conversation()
        conv.add_message("user", "Test")
        conv.add_message("assistant", "Response")

        # Serialize
        data = conv.to_dict()
        self.assertIn("messages", data)
        self.assertEqual(len(data["messages"]), 2)

        # Deserialize
        conv2 = Conversation.from_dict(data)
        self.assertEqual(len(conv2.messages), 2)
        self.assertEqual(conv2.messages[0].content, "Test")


class TestSession(unittest.TestCase):
    """Test session management."""

    def test_create_session(self):
        """Test session creation."""
        session = Session.create("glm", "glm-4.5")

        self.assertIsNotNone(session.session_id)
        self.assertEqual(session.provider, "glm")
        self.assertEqual(session.model, "glm-4.5")
        self.assertEqual(len(session.conversation.messages), 0)

    def test_session_save_load(self):
        """Test session save and load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("clawcodex_ext.agent.session.Path.home", return_value=Path(temp_dir)):
                # Create and save
                session = Session.create("glm", "glm-4.5")
                session.conversation.add_message("user", "Test message")
                session.save()

                # Load
                loaded = Session.load(session.session_id)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.session_id, session.session_id)
                self.assertEqual(len(loaded.conversation.messages), 1)
                self.assertEqual(loaded.conversation.messages[0].content, "Test message")


class TestREPLConversationSanitization(unittest.TestCase):
    """Pins the REPL-side mirror of the engine's image-strip recovery.

    The bug: an image-bearing UserMessage that triggered an
    `image_unsupported` API error stays in `session.conversation` after
    the engine strips its own `_mutable_messages`. The direct-stream
    path (_build_direct_stream_payload) reads
    `session.conversation.messages` directly — so without this mirror,
    a short text-only follow-up routed through `_stream_direct_response`
    would re-trigger the same 404 against Anthropic/Minimax providers.

    Tests `_sanitize_conversation_for_api_error` directly (extracted
    from the engine-loop handler for testability) so a regression
    that removes the strip call or breaks the tag check is caught.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / ".clawcodex"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        test_config = {
            "default_provider": "openrouter",
            "providers": {
                "openrouter": {
                    "api_key": "sk-or-test-key-12345678",
                    "base_url": "https://openrouter.ai/api/v1",
                    "default_model": "deepseek/deepseek-v4-pro",
                }
            },
        }
        with open(self.config_file, "w") as f:
            json.dump(test_config, f)
        self._global_config_patcher = patch.object(config_module, "GLOBAL_CONFIG_FILE", self.config_file)
        self._global_config_patcher.start()
        config_module._default_manager = None

    def tearDown(self):
        self._global_config_patcher.stop()
        config_module._default_manager = None

    def _make_repl(self):
        # Build the smallest valid REPL fixture for unit-level testing
        # of the sanitization helper. We need a real `session.conversation`
        # (so the helper has something to mutate); everything else is mocked.
        with patch("clawcodex_ext.repl.core.Session.create") as mock_session_create:
            session = Mock()
            session.conversation = Conversation()
            session.session_id = "test-session"
            session.provider = "openrouter"
            session.model = "deepseek/deepseek-v4-pro"
            mock_session_create.return_value = session

            with patch("clawcodex_ext.repl.core.get_provider_class") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.model = "deepseek/deepseek-v4-pro"
                mock_provider_class.return_value = mock_provider

                repl = ClawcodexREPL(provider_name="openrouter")
                repl.session = session
                return repl

    def test_image_unsupported_strips_images_from_conversation(self):
        """When an image_unsupported AssistantMessage is passed to the
        sanitizer, image blocks must be removed from
        session.conversation.messages — direct-stream path correctness
        depends on this.
        """
        from src.types.content_blocks import ImageBlock, TextBlock
        from src.types.messages import AssistantMessage

        repl = self._make_repl()

        repl.session.conversation.add_user_message(
            [
                TextBlock(text="describe this image"),
                ImageBlock(
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA",
                    }
                ),
            ]
        )

        err_msg = AssistantMessage(
            content="image not supported text",
            isApiErrorMessage=True,
        )
        err_msg._api_error = "image_unsupported"  # type: ignore[attr-defined]

        repl._sanitize_conversation_for_api_error(err_msg)

        # User's text intent must survive; the image bytes get the
        # "[image]" placeholder (matches strip_images_from_typed_messages
        # contract).
        user_msgs = [m for m in repl.session.conversation.messages if m.role == "user"]
        self.assertEqual(len(user_msgs), 1)
        content = user_msgs[0].content
        self.assertIsInstance(content, list)
        texts = [b.text for b in content if isinstance(b, TextBlock)]
        self.assertIn("describe this image", texts)
        self.assertIn("[image]", texts)
        for block in content:
            self.assertNotIsInstance(block, ImageBlock)

    def test_no_strip_when_error_tag_absent(self):
        """A regular assistant message (or one with a different
        _api_error tag) must NOT strip images — the strip is gated on
        the specific image_unsupported tag, so adjacent errors
        (prompt_too_long, etc.) keep their own recovery semantics.
        """
        from src.types.content_blocks import ImageBlock, TextBlock
        from src.types.messages import AssistantMessage

        repl = self._make_repl()
        repl.session.conversation.add_user_message(
            [
                TextBlock(text="describe"),
                ImageBlock(
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA",
                    }
                ),
            ]
        )

        # Non-error success message: no-op.
        ok_msg = AssistantMessage(content="here is the description")
        repl._sanitize_conversation_for_api_error(ok_msg)
        user_content = repl.session.conversation.messages[0].content
        self.assertTrue(
            any(isinstance(b, ImageBlock) for b in user_content),
            "image must survive when no _api_error tag is set",
        )

        # Different error tag: no-op too (prompt_too_long has its own
        # recovery via reactive_compact and must not trip image strip).
        ptl_msg = AssistantMessage(
            content="prompt too long",
            isApiErrorMessage=True,
        )
        ptl_msg._api_error = "prompt_too_long"  # type: ignore[attr-defined]
        repl._sanitize_conversation_for_api_error(ptl_msg)
        user_content = repl.session.conversation.messages[0].content
        self.assertTrue(
            any(isinstance(b, ImageBlock) for b in user_content),
            "image must survive when a non-image_unsupported tag is set",
        )

    def test_chat_invokes_sanitization_for_image_unsupported(self):
        """Wiring test: the engine-loop handler in REPL.chat MUST call
        ``_sanitize_conversation_for_api_error`` when the engine yields
        an image_unsupported AssistantMessage. Without this test, the
        helper could exist + be unit-tested while the call site is
        silently removed — and the bug would re-appear only at runtime
        on the direct-stream path.
        """
        from src.types.content_blocks import ImageBlock, TextBlock
        from src.types.messages import AssistantMessage

        repl = self._make_repl()

        # Seed the conversation with an image-bearing user message so
        # if the strip is invoked, we have something to strip.
        repl.session.conversation.add_user_message(
            [
                TextBlock(text="describe"),
                ImageBlock(
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA",
                    }
                ),
            ]
        )

        # Mock the engine's submit_message to yield exactly one
        # AssistantMessage tagged image_unsupported. This is the
        # message-shape the production query() function yields for the
        # OpenRouter 404.
        async def fake_submit_message(content, **kwargs):
            err_msg = AssistantMessage(
                content="image not supported text",
                isApiErrorMessage=True,
            )
            err_msg._api_error = "image_unsupported"  # type: ignore[attr-defined]
            yield err_msg

        sanitize_spy = Mock(wraps=repl._sanitize_conversation_for_api_error)
        repl._sanitize_conversation_for_api_error = sanitize_spy  # type: ignore[method-assign]

        # Patch QueryEngine.__init__ -> .submit_message to use our fake.
        # ``chat()`` imports QueryEngine lazily from the engine module, so
        # patch its definition there and return our fake_submit_message.
        with patch("clawcodex_ext.query.engine.QueryEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.submit_message = fake_submit_message
            mock_engine.reset_abort_controller = Mock()
            mock_engine.get_messages = Mock(return_value=[])
            mock_engine_class.return_value = mock_engine

            # Suppress console output so the test runs silently.
            repl.console.print = Mock()
            # Ensure the test's prompt routes through the QueryEngine
            # path (the direct-stream path doesn't go through the
            # handler we're testing). A long-enough prompt with a
            # code keyword forces _should_try_direct_stream to return
            # False; see core.py:2199-2238.
            repl.chat("please read README.md and summarize it for me carefully")

        sanitize_spy.assert_called()
        # And it must have been called with the tagged AssistantMessage,
        # not some other message — pinning the call-site code path.
        call_args_list = sanitize_spy.call_args_list
        self.assertTrue(
            any(
                len(call.args) >= 1
                and isinstance(call.args[0], AssistantMessage)
                and getattr(call.args[0], "_api_error", None) == "image_unsupported"
                for call in call_args_list
            ),
            "sanitization must be called with the image_unsupported AssistantMessage; "
            f"got call_args_list={call_args_list!r}",
        )

    def test_chat_renders_goal_evaluator_failure(self):
        """The canonical REPL path must not hide evaluator failures."""
        from src.types.messages import SystemMessage

        repl = self._make_repl()

        async def fake_submit_message(content, **kwargs):
            del content, kwargs
            notice = SystemMessage(
                content="Goal evaluator failed: provider unavailable",
                subtype="goal_evaluator_error",
                level="warning",
            )
            notice.usage = {"input_tokens": 5, "output_tokens": 2}  # type: ignore[attr-defined]
            yield notice

        with patch("clawcodex_ext.query.engine.QueryEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.submit_message = fake_submit_message
            mock_engine.reset_abort_controller = Mock()
            mock_engine.get_messages = Mock(return_value=[])
            mock_engine_class.return_value = mock_engine
            repl.console.print = Mock()
            repl._continue_goal_if_idle = Mock()

            repl.chat("please inspect README.md and verify the implementation carefully")

        self.assertTrue(
            any(
                "Goal evaluator failed: provider unavailable" in str(call.args[0])
                for call in repl.console.print.call_args_list
                if call.args
            )
        )
        self.assertEqual(repl._last_chat_outcome, "goal_evaluator_error")
        self.assertEqual(repl._stats_input_tokens, 5)
        self.assertEqual(repl._stats_output_tokens, 2)
        repl._continue_goal_if_idle.assert_not_called()

    def test_chat_accounts_goal_evaluator_usage(self):
        """REPL session stats include the evaluator side-call tokens."""
        from src.types.messages import SystemMessage

        repl = self._make_repl()

        async def fake_submit_message(content, **kwargs):
            del content, kwargs
            notice = SystemMessage(
                content="✓ Goal achieved",
                subtype="goal_achieved",
                level="info",
            )
            notice.usage = {"input_tokens": 7, "output_tokens": 3}  # type: ignore[attr-defined]
            yield notice

        with patch("clawcodex_ext.query.engine.QueryEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.submit_message = fake_submit_message
            mock_engine.reset_abort_controller = Mock()
            mock_engine.get_messages = Mock(return_value=[])
            mock_engine_class.return_value = mock_engine
            repl.console.print = Mock()

            repl.chat("please inspect README.md and verify the implementation carefully")

        self.assertEqual(repl._stats_input_tokens, 7)
        self.assertEqual(repl._stats_output_tokens, 3)


class TestREPLResumeReplay(unittest.TestCase):
    """Pins the rendering behaviour of ``_replay_resume_history()``.

    Tests that:
    * Tool-only assistant messages do NOT print a lonely "Assistant" label
    * String-content assistant messages do NOT produce a duplicate label
    * Text-bearing assistant messages still print "Assistant" as expected
    """

    def setUp(self):
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
        with open(config_file, "w") as f:
            json.dump(test_config, f)
        # Ensure get_provider_config can find the API key even when the
        # ConfigManager singleton bypasses get_config_path patching.
        self._old_glm_api_key = os.environ.get("GLM_API_KEY")
        os.environ["GLM_API_KEY"] = "test_api_key_12345678"

    def tearDown(self):
        if self._old_glm_api_key is not None:
            os.environ["GLM_API_KEY"] = self._old_glm_api_key
        else:
            os.environ.pop("GLM_API_KEY", None)

    def _make_repl(self) -> ClawcodexREPL:
        """Create a minimal ClawcodexREPL instance for testing replay."""
        # _load_heavy_runtime() must be called first so module-level
        # globals (Session, get_provider_config, etc.) exist before
        # patches try to resolve them.
        from clawcodex_ext.repl.core import _load_heavy_runtime

        _load_heavy_runtime()

        with patch("src.config.get_config_path", return_value=self.config_dir / "config.json"):
            with patch("src.agent.Session") as mock_session_class:
                mock_session_instance = Mock()
                mock_session_instance.session_id = "test-session"
                mock_session_class.create.return_value = mock_session_instance

                with patch("src.providers.get_provider_class") as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    return ClawcodexREPL(provider_name="glm")

    def test_replay_tool_only_assistant_suppresses_label(self):
        """Tool-only assistant messages must NOT print the Assistant label,
        avoiding the visual bug where multiple lonely 'Assistant' lines
        stack up with no content underneath.
        """
        from clawcodex_ext.types.content_blocks import TextBlock, ToolUseBlock
        from clawcodex_ext.types.messages import AssistantMessage, UserMessage

        repl = self._make_repl()

        # Two assistant messages: text-bearing and tool-only.
        # The tool-only one should NOT produce an "Assistant" print.
        user_msg = UserMessage(content="do something")
        text_assistant = AssistantMessage(content=[TextBlock(text="I will help")])
        tool_assistant = AssistantMessage(
            content=[
                ToolUseBlock(id="call-1", name="Bash", input={"command": "echo hello"}),
                ToolUseBlock(id="call-2", name="Read", input={"path": "/tmp/x"}),
            ]
        )
        repl.session.conversation.messages = [
            user_msg,
            text_assistant,
            tool_assistant,
        ]

        repl.console.print = Mock()
        repl._resume_session_id = "test-session"
        repl._replay_resume_history()

        # Collect all console.print call arguments (ignore the header/footer lines)
        printed_lines = []
        for call_args in repl.console.print.call_args_list:
            arg = call_args[0][0] if call_args[0] else ""
            printed_lines.append(str(arg))

        # The text assistant should produce exactly one "Assistant" print
        assistant_lines = [ln for ln in printed_lines if "Assistant" in ln]
        self.assertEqual(
            len(assistant_lines),
            1,
            f"Expected exactly 1 'Assistant' print (text message), got {len(assistant_lines)}: {assistant_lines}",
        )

        # Both tool-use headers should be rendered.
        tool_headers = [ln for ln in printed_lines if "●" in ln]
        self.assertEqual(
            len(tool_headers),
            2,
            f"Expected 2 tool-use headers, got {len(tool_headers)}",
        )

    def test_replay_unmatched_tool_call_stays_before_later_recap(self):
        """An unmatched persisted tool call must not move past a later recap."""
        from clawcodex_ext.away_summary.messages import create_away_summary_message
        from clawcodex_ext.types.content_blocks import ToolUseBlock
        from clawcodex_ext.types.messages import AssistantMessage, UserMessage

        repl = self._make_repl()
        recap = create_away_summary_message(
            "finished editing",
            trigger="auto",
            fingerprint="fp",
            message_count=2,
        )
        repl.session.conversation.messages = [
            UserMessage(content="edit the file"),
            AssistantMessage(content=[ToolUseBlock(id="call-1", name="Edit", input={"path": "a.md"})]),
            recap,
        ]
        repl.console.print = Mock()

        repl._replay_resume_history()

        printed = [
            getattr(call.args[0], "markup", str(call.args[0]))
            for call in repl.console.print.call_args_list
            if call.args
        ]
        tool_index = next(i for i, line in enumerate(printed) if "Edit" in line)
        recap_index = next(i for i, line in enumerate(printed) if "Recapitulate" in line)
        self.assertLess(tool_index, recap_index)

    def test_replay_tool_results_and_suppresses_whitespace_assistant_labels(self):
        """Persisted tool results render without empty Assistant headings."""
        from clawcodex_ext.types.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
        from clawcodex_ext.types.messages import AssistantMessage, UserMessage

        repl = self._make_repl()
        repl.session.conversation.messages = [
            AssistantMessage(
                content=[
                    TextBlock(text="\n\n"),
                    ToolUseBlock(
                        id="call-edit",
                        name="Edit",
                        input={"file_path": "a.md"},
                    ),
                ]
            ),
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="call-edit",
                        content="old_string not found in file",
                        is_error=True,
                    )
                ]
            ),
            AssistantMessage(
                content=[
                    TextBlock(text="\n"),
                    ToolUseBlock(
                        id="call-grep",
                        name="Grep",
                        input={"pattern": "中间件", "path": "a.md"},
                    ),
                ]
            ),
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="call-grep",
                        content='{"mode":"content","numLines":1}',
                    )
                ]
            ),
        ]
        repl.console.print = Mock()

        repl._replay_resume_history()

        printed = [
            getattr(call.args[0], "markup", str(call.args[0]))
            for call in repl.console.print.call_args_list
            if call.args
        ]
        self.assertFalse(any("Assistant" in line for line in printed))
        self.assertTrue(any("old_string not found in file" in line for line in printed))
        self.assertTrue(any("Found 1 line" in line for line in printed))

    def test_engine_tool_result_is_recorded_for_resume(self):
        from clawcodex_ext.types.content_blocks import ToolResultBlock
        from clawcodex_ext.types.messages import UserMessage

        repl = self._make_repl()
        repl.session.conversation = Conversation()
        result = UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="call-1",
                    content="Unchanged since last read",
                )
            ]
        )

        self.assertTrue(repl._record_tool_result_message(result))
        self.assertEqual(repl.session.conversation.messages, [result])

    def test_replay_string_content_no_duplicate_label(self):
        """Assistant messages with string content must not produce a
        duplicate 'Assistant' label (bug: the unconditional print at the
        top of the assistant branch plus the string-content branch both
        printed it).
        """
        from clawcodex_ext.types.messages import AssistantMessage, UserMessage

        repl = self._make_repl()
        repl.session.conversation.messages = [
            UserMessage(content="hi"),
            AssistantMessage(content="Hello there"),
        ]

        repl.console.print = Mock()
        repl._resume_session_id = "test-session"
        repl._replay_resume_history()

        printed_lines = []
        for call_args in repl.console.print.call_args_list:
            arg = call_args[0][0] if call_args[0] else ""
            printed_lines.append(str(arg))

        assistant_lines = [ln for ln in printed_lines if "Assistant" in ln]
        self.assertEqual(
            len(assistant_lines),
            1,
            f"Expected exactly 1 'Assistant' print for string content, got {len(assistant_lines)}: {assistant_lines}",
        )

    def test_replay_text_content_prints_assistant_label(self):
        """Text-bearing assistant messages must still print the Assistant
        label so the user sees the turn boundary.
        """
        from clawcodex_ext.types.content_blocks import TextBlock
        from clawcodex_ext.types.messages import AssistantMessage, UserMessage

        repl = self._make_repl()
        repl.session.conversation.messages = [
            UserMessage(content="hello"),
            AssistantMessage(content=[TextBlock(text="Hi, how can I help?")]),
        ]

        repl.console.print = Mock()
        repl._resume_session_id = "test-session"
        repl._replay_resume_history()

        printed_lines = []
        for call_args in repl.console.print.call_args_list:
            arg = call_args[0][0] if call_args[0] else ""
            printed_lines.append(str(arg))

        assistant_lines = [ln for ln in printed_lines if "Assistant" in ln]
        self.assertGreaterEqual(
            len(assistant_lines),
            1,
            "Expected at least 1 'Assistant' print for text-bearing message, got none",
        )


class TestEchoUserInput(unittest.TestCase):
    """_echo_user_input must render text without background and without
    leaking Rich markup tags like ``[bold ...]`` to the screen.
    """

    def setUp(self):
        from clawcodex_ext.repl.core import ClawcodexREPL

        repl = ClawcodexREPL.__new__(ClawcodexREPL)
        repl.console = Mock()
        # The palette is needed by _echo_user_input for the primary colour.
        from clawcodex_ext.repl.color_scheme import DARK

        repl._repl_palette = DARK
        self.repl = repl

    def _call(self, text: str) -> list:
        """Invoke ``_echo_user_input`` and return the ``Text`` args from
        each ``console.print`` call.
        """
        self.repl.console.print.reset_mock()
        self.repl._echo_user_input(text)
        return [
            call.args[0]  # first positional arg (the Text renderable)
            for call in self.repl.console.print.call_args_list
            if call.args
        ]

    def assert_no_markup_leak(self, text_objects: list) -> None:
        """Fail if any Text object's plain text contains Rich markup tags."""
        for t in text_objects:
            plain = t.plain if hasattr(t, "plain") else str(t)
            self.assertNotIn(
                "[bold",
                plain,
                f"Rich markup leaked to screen: {plain!r}",
            )
            self.assertNotIn(
                "[/bold",
                plain,
                f"Rich markup close-tag leaked to screen: {plain!r}",
            )

    def assert_no_background_style(self, text_objects: list) -> None:
        """Fail if any Text object carries a background colour."""
        from rich.style import Style

        for t in text_objects:
            styled_spans = getattr(t, "spans", [])
            for span in styled_spans:
                style: Style = span.style
                self.assertIsNone(
                    style.bgcolor,
                    f"Text span has non-transparent background: {style.bgcolor}",
                )

    # ── tests ─────────────────────────────────────────────────────────

    def test_single_line_no_markup_leak(self):
        """Single-line input must not leak Rich markup tags."""
        texts = self._call("/btw 123")
        self.assert_no_markup_leak(texts)

    def test_no_background(self):
        """Output must be fully transparent (no bgcolor on any span)."""
        texts = self._call("/btw 123")
        self.assert_no_background_style(texts)

    def test_multi_line_no_markup_leak(self):
        """Multi-line input must not leak Rich markup tags."""
        texts = self._call("line1\nline2\nline3")
        self.assert_no_markup_leak(texts)

    def test_multi_line_no_background(self):
        """Multi-line output must also have transparent background."""
        texts = self._call("line1\nline2\nline3")
        self.assert_no_background_style(texts)

    def test_text_with_square_brackets_is_safe(self):
        """User input containing ``[bold]`` literals must render literally
        (``markup=False`` prevents Rich from interpreting them as markup
        tags). No unintended Rich styles should be injected.
        """
        texts = self._call("[bold]raw[/bold]")
        # The literal brackets appear in plain text because markup=False.
        self.assertIn("[bold]", texts[0].plain)
        self.assertIn("[/bold]", texts[0].plain)
        # But no unintended background or markup-leak styles.
        self.assert_no_background_style(texts)

    def test_text_with_square_brackets_no_extra_styles(self):
        """Literal square brackets in user input must not introduce
        unintended Rich styles.
        """
        texts = self._call("[red]alert[/red]")
        self.assert_no_markup_leak(texts)
        # The plain text should contain exactly the brackets as-is.
        self.assertIn("[red]", texts[0].plain)
        self.assertIn("[/red]", texts[0].plain)

    def test_prefix_is_bold_with_primary_color(self):
        """The ``❯ `` prefix must be styled with bold + primary colour."""
        from rich.text import Text

        texts = self._call("hello")
        self.assertGreater(len(texts), 0)
        t: Text = texts[0]
        # The first span should be the "❯ " prefix.
        self.assertTrue(
            t.plain.startswith("❯ "),
            f"Expected ❯ prefix, got: {t.plain!r}",
        )

    def test_returns_single_rich_text(self):
        """_echo_user_input should produce a single Text object for a given
        input line.
        """
        texts = self._call("hello")
        self.assertEqual(len(texts), 1)
        from rich.text import Text

        self.assertIsInstance(texts[0], Text)

    def test_empty_string(self):
        """Empty input produces a ``❯ `` with no trailing text."""
        texts = self._call("")
        # The method always echoes the prefix even for empty text.
        # The caller decides what is meaningful input.
        self.assertGreaterEqual(len(texts), 1)
        plain = texts[0].plain.rstrip("\n")
        self.assertIn("❯", plain)
        self.assert_no_markup_leak(texts)
        self.assert_no_background_style(texts)

    def test_blank_string(self):
        """Whitespace-only input should produce output (preserves the
        existing behaviour — the caller decides what is meaningful).
        """
        texts = self._call("   ")
        self.assert_no_markup_leak(texts)
        self.assert_no_background_style(texts)

    def test_console_markup_flag_is_false(self):
        """console.print must always be called with ``markup=False`` so
        user input like ``[bold]`` renders literally.
        """
        self.repl.console.print.reset_mock()
        self.repl._echo_user_input("hello")
        for call in self.repl.console.print.call_args_list:
            kwargs = call.kwargs or {}
            self.assertIs(
                kwargs.get("markup", True),
                False,
                f"console.print called without markup=False: {kwargs}",
            )

    def test_short_text_no_trailing_padding(self):
        """Output must NOT be padded to terminal width (the old behaviour
        did this to force a full-row background highlight). Since we no
        longer apply a background, the extra padding is unnecessary and
        would waste horizontal space.
        """
        # With background removal, we no longer ljust the text — short
        # input should remain short.
        texts = self._call("hi")
        plain = texts[0].plain.rstrip("\n")
        # The plain text should not be padded to 80+ chars.
        self.assertLess(len(plain), 10)


if __name__ == "__main__":
    unittest.main()
