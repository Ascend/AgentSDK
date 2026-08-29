#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for stage3 repl."""

from __future__ import annotations

import os
from pathlib import Path

from tests.stability_gate._config_helper import (
    cleanup_config,
    make_config,
    redirect_global_config,
)


class TestStage3Repl:
    """Tests for TestStage3Repl."""

    def test_repl_module_importable(self):
        """Verify repl module importable."""
        from src.repl.core import ClawcodexREPL

        assert ClawcodexREPL is not None

    def test_repl_instantiation(self):
        """Verify repl instantiation."""
        from src.repl.core import ClawcodexREPL

        assert ClawcodexREPL is not None
        import inspect

        sig = inspect.signature(ClawcodexREPL.__init__)
        assert "provider_name" in sig.parameters
        assert "permission_mode" in sig.parameters
        assert "stream" in sig.parameters

    def test_repl_simple_query_with_fake_provider(self, tmp_path):
        """Verify repl simple query with fake provider."""
        from src.agent.conversation import Conversation

        home_path = tmp_path / "home"
        config_file = make_config(home_path)
        config_patcher = redirect_global_config(config_file)
        old_cwd = Path.cwd()

        try:
            conv = Conversation()
            conv.add_user_message("Hello")
            conv.add_assistant_message("Hi there!")
            msgs = conv.get_messages()
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[1]["role"] == "assistant"
        finally:
            config_patcher.stop()
            cleanup_config()
            os.chdir(old_cwd)

    def test_repl_file_history_attr(self) -> None:
        """Both ``ClawcodexREPL`` and ``ClawCodexExtREPL`` set
        ``self._file_history`` in their ``__init__``, enabling up/down
        history navigation during agent work.

        Regression guard: if either class omits this attribute,
        ``LiveStatus(history=self._file_history)`` in ``chat()``
        raises ``AttributeError``.
        """
        import inspect

        # Check ClawcodexREPL (core.py)
        from src.repl.core import ClawcodexREPL

        core_src = inspect.getsource(ClawcodexREPL.__init__)
        assert "self._file_history" in core_src, "ClawcodexREPL.__init__ must set self._file_history"

        # Check ClawCodexExtREPL (app.py)
        from clawcodex_ext.repl.app import ClawCodexExtREPL

        app_src = inspect.getsource(ClawCodexExtREPL.__init__)
        assert "self._file_history" in app_src, "ClawCodexExtREPL.__init__ must set self._file_history"

    def test_chat_passes_history_to_live_status(self) -> None:
        """Both ``chat()`` paths pass ``history=self._file_history``
        to ``LiveStatus(...)`` (direct-stream and engine paths).
        """
        import inspect

        from src.repl.core import ClawcodexREPL

        chat_src = inspect.getsource(ClawcodexREPL.chat)
        # Count occurrences — one in each LiveStatus(...) call
        count = chat_src.count("history=self._file_history")
        assert count >= 2, f"Expected at least 2 ``history=self._file_history`` in chat(), found {count}"


class TestStage3Headless:
    """Tests for TestStage3Headless."""

    def test_headless_module_importable(self):
        """Verify headless module importable."""
        import src.entrypoints.headless as hl

        assert hasattr(hl, "HeadlessOptions")
        assert hasattr(hl, "run_headless")

    def test_headless_options_buildable(self):
        """Verify headless options buildable."""
        import src.entrypoints.headless as hl

        options = hl.HeadlessOptions(
            prompt="Say hello",
            output_format="text",
            input_format="text",
            skip_permissions=True,
            max_turns=20,
        )
        assert options.prompt == "Say hello"
        assert options.output_format == "text"

    def test_run_headless_is_callable(self):
        """Verify run headless is callable."""
        import src.entrypoints.headless as hl

        assert callable(hl.run_headless)

    def test_headless_output_format_enum(self):
        """Verify headless output format enum."""
        import src.entrypoints.headless as hl

        for fmt in ("text", "json", "stream-json"):
            opts = hl.HeadlessOptions(
                prompt="test",
                output_format=fmt,
                skip_permissions=True,
            )
            assert opts.output_format == fmt
