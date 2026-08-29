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

"""P0 P1 P2 Tests for stage3b repl resilience."""

from __future__ import annotations


def _is_recoverable_tool_error(tool_name: str, tool_output) -> bool:
    """Inline equivalent of ClawcodexREPL._is_recoverable_tool_error."""
    if not isinstance(tool_name, str):
        return False
    if not isinstance(tool_output, dict):
        return False
    name = tool_name.strip().lower()
    err = tool_output.get("error")
    if not isinstance(err, str):
        return False
    e = err.lower()
    if name == "read" and e.startswith("file not found:"):
        p = err.split(":", 2)[-1].strip()
        if (
            "/.clawcodex/skills/" in p
            or "\\.clawcodex\\skills\\" in p
            or "/.claude/skills/" in p
            or "\\.claude\\skills\\" in p
        ):
            return True
    return False


class TestStage3bToolErrorClassification:
    """P1 Tests for TestStage3bToolErrorClassification."""

    def test_read_file_not_found_not_recoverable(self):
        """Verify read file not found not recoverable."""
        result = _is_recoverable_tool_error("Read", {"error": "File not found: /tmp/x.txt"})
        assert result is False

    def test_read_skill_path_is_recoverable(self):
        """Verify read skill path is recoverable."""
        result = _is_recoverable_tool_error(
            "Read",
            {"error": "File not found: /home/x/.clawcodex/skills/my-skill.md"},
        )
        assert result is True

    def test_is_recoverable_non_dict_output(self):
        """Verify is recoverable non dict output."""
        result = _is_recoverable_tool_error("Read", None)
        assert result is False

        result = _is_recoverable_tool_error("Read", "just a string")
        assert result is False

    def test_is_recoverable_missing_error_key(self):
        """Verify is recoverable missing error key."""
        result = _is_recoverable_tool_error("Write", {"ok": True})
        assert result is False

    def test_is_recoverable_windows_style_path(self):
        """Verify is recoverable windows style path."""
        result = _is_recoverable_tool_error(
            "Read",
            {"error": "File not found: C:\\Users\\x\\.clawcodex\\skills\\foo.md"},
        )
        assert result is True

    def test_is_recoverable_case_insensitive_tool_name(self):
        """Verify is recoverable case insensitive tool name."""
        result = _is_recoverable_tool_error("read", {"error": "File not found: /x/.clawcodex/skills/s.md"})
        assert result is True

        result = _is_recoverable_tool_error("READ", {"error": "File not found: /x/.clawcodex/skills/s.md"})
        assert result is True


class TestStage3bConversationBoundary:
    """P2 Tests for TestStage3bConversationBoundary."""

    def test_conversation_emoji_content(self):
        """Verify conversation emoji content."""
        from src.agent.conversation import Conversation

        conv = Conversation()
        conv.add_user_message("Hello 👋 世界 🌍 测试 test")
        conv.add_assistant_message("回复: 你好！🎉")

        msgs = conv.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert "👋" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        if isinstance(msgs[1]["content"], str):
            assert "🎉" in msgs[1]["content"]
        else:
            assert any(isinstance(b, dict) and "🎉" in b.get("text", "") for b in msgs[1]["content"])

    def test_conversation_empty_get_messages(self):
        """Verify conversation empty get messages."""
        from src.agent.conversation import Conversation

        conv = Conversation()
        assert conv.get_messages() == []

    def test_conversation_tool_result_with_error(self):
        """Verify conversation tool result with error."""
        from src.agent.conversation import Conversation

        conv = Conversation()
        conv.add_user_message("run test")
        conv.add_tool_result_message(
            tool_use_id="tu_test_001",
            content="Command failed with exit code 1",
            is_error=True,
            duration_ms=1500,
        )
        assert len(conv.messages) == 2
        api_msgs = conv.get_messages()
        assert len(api_msgs) >= 1
        assert api_msgs[-1]["role"] == "user"

    def test_conversation_from_dict_empty(self):
        """Verify conversation from dict empty."""
        from src.agent.conversation import Conversation

        conv = Conversation.from_dict({})
        assert conv.get_messages() == []

    def test_conversation_from_dict_none_messages(self):
        """Verify conversation from dict none messages."""
        from src.agent.conversation import Conversation

        conv = Conversation.from_dict({"messages": None})
        assert conv.get_messages() == []

    def test_conversation_add_large_message(self):
        """Verify conversation add large message."""
        from src.agent.conversation import Conversation

        conv = Conversation(max_history=10)
        large = "A" * 50_000
        conv.add_user_message(large)
        msgs = conv.get_messages()
        assert len(msgs) == 1
        assert len(msgs[0]["content"]) == 50_000

    def test_conversation_to_dict_from_dict_with_emoji(self):
        """Verify conversation to dict from dict with emoji."""
        from src.agent.conversation import Conversation

        conv = Conversation()
        conv.add_user_message("Hello 你好 🎯")
        data = conv.to_dict()
        reloaded = Conversation.from_dict(data)
        msgs = reloaded.get_messages()
        assert len(msgs) == 1
        assert "🎯" in msgs[0]["content"]

    def test_conversation_clear_then_get_messages(self):
        """Verify conversation clear then get messages."""
        from src.agent.conversation import Conversation

        conv = Conversation()
        conv.add_user_message("temp")
        conv.clear()
        assert conv.get_messages() == []


class TestStage3bRichMarkupEscape:
    """Tests for TestStage3bRichMarkupEscape."""

    _MARKUP_LIKE_ERRORS = [  # noqa: RUF012
        "unexpected token [/bold] found",
        "[/color] without matching [color]",
        "provider response: [bold]ERROR[/bold] occurred",
        "malformed [link=xxx] without close",
        "nested [bold][italic]test[/bold][/italic] misorder",
    ]

    def test_escape_marks_up_brackets(self):
        """Verify escape marks up brackets."""
        from rich.markup import escape

        raw = "[/bold]"
        escaped = escape(raw)
        assert escaped == r"\[/bold]", f"escape() should escape `[`, got: {escaped!r}"

    def test_escape_idempotent_on_clean_text(self):
        """Verify escape idempotent on clean text."""
        from rich.markup import escape

        clean = "Error 401: authentication failed"
        assert escape(clean) == clean

    def test_console_print_escaped_error_no_markup_error(self):
        """Verify console print escaped error no markup error."""
        from io import StringIO

        from rich.console import Console
        from rich.markup import escape

        for msg in self._MARKUP_LIKE_ERRORS:
            buf = StringIO()
            c = Console(file=buf, force_terminal=False, safe_box=False)
            c.print(f"\n[error]Error: {escape(msg)}[/error]")
            output = buf.getvalue()
            assert msg in output, f"Expected original message in output, msg={msg!r}, output={output!r}"

    def test_console_print_raises_without_escape(self):
        """Verify console print raises without escape."""
        from io import StringIO

        import pytest
        from rich.console import Console
        from rich.errors import MarkupError

        buf = StringIO()
        c = Console(file=buf, force_terminal=False)
        with pytest.raises(MarkupError):
            c.print("\n[error]Error: [/bold][/error]")

    def test_all_console_print_in_chat_use_escape(self):
        """Verify all console print in chat use escape."""
        import inspect

        from clawcodex_ext.repl.core import ClawcodexREPL

        src = inspect.getsource(ClawcodexREPL.chat)
        lines = src.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.startswith("self.console.print"):
                contains_untrusted = "{e" in stripped or "{err_text" in stripped
                if contains_untrusted:
                    assert "escape(" in stripped, (
                        f"chat() line {i} embeds untrusted content without escape(): {stripped!r}"
                    )

            if "call_args =" in stripped and "{summary}" in stripped:
                assert "escape(summary)" in stripped, (
                    f"chat() line {i} constructs call_args with summary without escape(): {stripped!r}"
                )
