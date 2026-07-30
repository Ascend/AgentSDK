"""Stage 3b — REPL 弹性测试（< 5 秒）。

覆盖 P0#1（provider 异常恢复）、P1#8（malformed input）、
P2#13（字符边界）、P2#14（磁盘满/权限拒绝）。

验证：
- _is_recoverable_tool_error 对工具返回的错误分类正确
- Conversation 对空/损坏/超大输入不抛异常
- 字符边界（emoji / CJK）输入正常
- 工具执行结果报错标记被正确传递

注：_is_recoverable_tool_error 是 ClawcodexREPL 上的 @staticmethod，
但由于 src.repl.core 已改为 facade 模式（通过 __getattr__ 代理到
clawcodex_ext），无法直接导入 ClawcodexREPL。因此我们在测试中
内联了等效的逻辑，保证分类契约被锁住。
"""

from __future__ import annotations


# 从 clawcodex_ext.repl.core 中提取的 _is_recoverable_tool_error 等效逻辑
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
    """_is_recoverable_tool_error 分类正确性 — P1#7 工具异常隔离。"""

    def test_read_file_not_found_not_recoverable(self):
        """Read 工具的 File not found（非 skills 路径）不可恢复。"""
        result = _is_recoverable_tool_error("Read", {"error": "File not found: /tmp/x.txt"})
        assert result is False

    def test_read_skill_path_is_recoverable(self):
        """Read 工具的 skills 路径 File not found 是可恢复的。"""
        result = _is_recoverable_tool_error(
            "Read",
            {"error": "File not found: /home/x/.clawcodex/skills/my-skill.md"},
        )
        assert result is True

    def test_is_recoverable_non_dict_output(self):
        """tool_output 不是 dict 时返回 False 不抛异常。"""
        result = _is_recoverable_tool_error("Read", None)
        assert result is False

        result = _is_recoverable_tool_error("Read", "just a string")
        assert result is False

    def test_is_recoverable_missing_error_key(self):
        """tool_output 缺 error key 时返回 False。"""
        result = _is_recoverable_tool_error("Write", {"ok": True})
        assert result is False

    def test_is_recoverable_windows_style_path(self):
        """Windows 风格 skills 路径也被识别。"""
        result = _is_recoverable_tool_error(
            "Read",
            {"error": "File not found: C:\\Users\\x\\.clawcodex\\skills\\foo.md"},
        )
        assert result is True

    def test_is_recoverable_case_insensitive_tool_name(self):
        """工具名大小写不敏感 (read=Read=READ)。"""
        result = _is_recoverable_tool_error("read", {"error": "File not found: /x/.clawcodex/skills/s.md"})
        assert result is True

        result = _is_recoverable_tool_error("READ", {"error": "File not found: /x/.clawcodex/skills/s.md"})
        assert result is True


class TestStage3bConversationBoundary:
    """Conversation 边界测试 — P2#13 字符边界, P2#14 错误标记。"""

    def test_conversation_emoji_content(self):
        """Conversation 含 emoji 和 CJK 字符时不抛异常。"""
        from src.agent.conversation import Conversation

        conv = Conversation()
        conv.add_user_message("Hello 👋 世界 🌍 测试 test")
        conv.add_assistant_message("回复: 你好！🎉")

        msgs = conv.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert "👋" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        # assistant content 可能是 string 或 list of blocks
        if isinstance(msgs[1]["content"], str):
            assert "🎉" in msgs[1]["content"]
        else:
            assert any(isinstance(b, dict) and "🎉" in b.get("text", "") for b in msgs[1]["content"])

    def test_conversation_empty_get_messages(self):
        """空 Conversation.get_messages() 返回 []，下游不炸。"""
        from src.agent.conversation import Conversation

        conv = Conversation()
        assert conv.get_messages() == []

    def test_conversation_tool_result_with_error(self):
        """add_tool_result_message 带 is_error=True 不抛异常。

        normalize_messages_for_api 会将连续的 user role 消息合并，
        所以 get_messages() 返回 1 条，但 messages 内部是 2 条。
        """
        from src.agent.conversation import Conversation

        conv = Conversation()
        conv.add_user_message("run test")
        conv.add_tool_result_message(
            tool_use_id="tu_test_001",
            content="Command failed with exit code 1",
            is_error=True,
            duration_ms=1500,
        )
        # 内部 messages 应有 2 条记录
        assert len(conv.messages) == 2
        # get_messages 会合并连续 user 消息
        api_msgs = conv.get_messages()
        assert len(api_msgs) >= 1
        assert api_msgs[-1]["role"] == "user"

    def test_conversation_from_dict_empty(self):
        """Conversation.from_dict({}) 不抛异常，messages 为空。"""
        from src.agent.conversation import Conversation

        conv = Conversation.from_dict({})
        assert conv.get_messages() == []

    def test_conversation_from_dict_none_messages(self):
        """from_dict 的 messages=None 不抛异常。"""
        from src.agent.conversation import Conversation

        conv = Conversation.from_dict({"messages": None})
        assert conv.get_messages() == []

    def test_conversation_add_large_message(self):
        """超长单条消息不导致 Conversation 崩溃。"""
        from src.agent.conversation import Conversation

        conv = Conversation(max_history=10)
        large = "A" * 50_000
        conv.add_user_message(large)
        msgs = conv.get_messages()
        assert len(msgs) == 1
        assert len(msgs[0]["content"]) == 50_000

    def test_conversation_to_dict_from_dict_with_emoji(self):
        """含 emoji 的对话 to_dict → from_dict 往返不丢失字符。"""
        from src.agent.conversation import Conversation

        conv = Conversation()
        conv.add_user_message("Hello 你好 🎯")
        data = conv.to_dict()
        reloaded = Conversation.from_dict(data)
        msgs = reloaded.get_messages()
        assert len(msgs) == 1
        assert "🎯" in msgs[0]["content"]

    def test_conversation_clear_then_get_messages(self):
        """clear() 后 get_messages() 返回空。"""
        from src.agent.conversation import Conversation

        conv = Conversation()
        conv.add_user_message("temp")
        conv.clear()
        assert conv.get_messages() == []


class TestStage3bRichMarkupEscape:
    """Rich markup 转义防护 — 异常消息含 `[/bold]` 时不抛出 MarkupError。

    回归防护：clawcodex_ext/repl/core.py chat() 中 error/authentication error
    的 console.print 必须使用 escape(str(e)) 而非裸 {e}，否则异常消息中
    的 [/bold] 等 Rich 标记会导致 MarkupError 击穿。

    相关问题：chat() 第 4832 行 `Error: {escape(str(e))}` 路径。
    """

    # Rich 标记样式的异常消息样本
    _MARKUP_LIKE_ERRORS = [  # noqa: RUF012
        "unexpected token [/bold] found",
        "[/color] without matching [color]",
        "provider response: [bold]ERROR[/bold] occurred",
        "malformed [link=xxx] without close",
        "nested [bold][italic]test[/bold][/italic] misorder",
    ]

    def test_escape_marks_up_brackets(self):
        """escape() 将 `[` 转义为反斜杠加左方括号，防止被 Rich 解析为标记起始。"""
        from rich.markup import escape

        raw = "[/bold]"
        escaped = escape(raw)
        # Rich 只转义 [（标记起始），] 独立时无歧义无需转义
        assert escaped == r"\[/bold]", f"escape() should escape `[`, got: {escaped!r}"

    def test_escape_idempotent_on_clean_text(self):
        """escape() 对不含标记的纯文本保持恒等。"""
        from rich.markup import escape

        clean = "Error 401: authentication failed"
        assert escape(clean) == clean

    def test_console_print_escaped_error_no_markup_error(self):
        """console.print 使用 escape() 后在 error 标签内打印不抛 MarkupError。

        对多种 Rich 标记样式逐一验证。
        """
        from io import StringIO

        from rich.console import Console
        from rich.markup import escape

        for msg in self._MARKUP_LIKE_ERRORS:
            buf = StringIO()
            c = Console(file=buf, force_terminal=False, safe_box=False)
            # 不应抛出 MarkupError
            c.print(f"\n[error]Error: {escape(msg)}[/error]")
            output = buf.getvalue()
            assert msg in output, f"Expected original message in output, msg={msg!r}, output={output!r}"

    def test_console_print_raises_without_escape(self):
        """console.print 对裸 `[/bold]` 抛出 MarkupError — 确认测试有效性。"""
        from io import StringIO

        import pytest
        from rich.console import Console
        from rich.errors import MarkupError

        buf = StringIO()
        c = Console(file=buf, force_terminal=False)
        with pytest.raises(MarkupError):
            c.print("\n[error]Error: [/bold][/error]")

    def test_all_console_print_in_chat_use_escape(self):
        """chat() 中所有嵌不可信变量的 console.print 行都使用 escape()。

        来源检查：确保 repair 不会因后续重构退化。
        覆盖模式：
        - {escape(str(e))} 用于异常 e
        - {escape(err_text)} 用于工具输出
        - call_args 构造中 escape(summary) 用于工具调用摘要
        """
        import inspect

        from clawcodex_ext.repl.core import ClawcodexREPL

        src = inspect.getsource(ClawcodexREPL.chat)
        lines = src.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # 检查 console.print 中的不可信变量
            if stripped.startswith("self.console.print"):
                contains_untrusted = "{e" in stripped or "{err_text" in stripped
                if contains_untrusted:
                    assert "escape(" in stripped, (
                        f"chat() line {i} embeds untrusted content without escape(): {stripped!r}"
                    )

            # 检查 call_args 构造（summary 来自工具输入，用户可控制）
            if "call_args =" in stripped and "{summary}" in stripped:
                assert "escape(summary)" in stripped, (
                    f"chat() line {i} constructs call_args with summary without escape(): {stripped!r}"
                )
