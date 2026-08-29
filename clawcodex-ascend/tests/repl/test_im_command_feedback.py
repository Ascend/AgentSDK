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

from __future__ import annotations

from unittest.mock import Mock

from src.repl import ClawcodexREPL


def _make_clear_repl() -> ClawcodexREPL:
    repl = ClawcodexREPL.__new__(ClawcodexREPL)
    repl._built_in_commands = ["/clear"]
    repl._try_execute_new_command = Mock(side_effect=RuntimeError("fallback"))
    repl._run_command_async_with_status = Mock(side_effect=RuntimeError("fallback"))
    repl.command_registry = Mock()
    repl.command_registry.get.return_value = None
    repl.console = Mock()
    repl.session = Mock()
    repl.session.conversation = Mock()
    repl._engine_messages = ["old"]
    return repl


def test_handle_command_clear_sends_im_command_feedback() -> None:
    """IM-driven /clear should get a visible command completion notice."""
    repl = _make_clear_repl()
    calls: list[tuple[str, bool]] = []

    class _FakeImReply:
        def send_command_feedback(self, command: str, *, success: bool) -> bool:
            calls.append((command, success))
            return True

    repl._im_reply_controller = _FakeImReply()

    repl.handle_command("/clear")

    assert calls == [("/clear", True)]


def test_handle_command_exception_sends_failed_im_command_feedback() -> None:
    repl = _make_clear_repl()
    repl._handle_command = Mock(side_effect=RuntimeError("command failed"))
    calls: list[tuple[str, bool]] = []

    class _FakeImReply:
        def send_command_feedback(self, command: str, *, success: bool) -> bool:
            calls.append((command, success))
            return True

    repl._im_reply_controller = _FakeImReply()

    try:
        repl.handle_command("/clear")
    except RuntimeError as exc:
        assert str(exc) == "command failed"
    else:
        raise AssertionError("command failure must propagate")

    assert calls == [("/clear", False)]
