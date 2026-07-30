"""Stage 3e — REPL 配色 / Rich Theme Markup 渲染测试（< 2 秒）。

验证使用 OKLCH 调色板的 Rich Theme 不会抛出 MarkupError。
每种自定义语义名称（error/success/warning/info/等）都独立测试，
确保其作为独立标签和嵌套标签（如 [bold][error]）都能正确渲染。

注意：此测试不依赖 Console 的终端宽度或颜色支持，使用
``force_terminal=True`` 确保 ANSI 序列始终生成，从而触发
完整的 Rich markup 解析路径。输出写入 ``io.StringIO`` 而非
真实终端，避免测试噪音。
"""

from __future__ import annotations

import inspect
import io

import pytest
from clawcodex_ext.repl.color_scheme import DARK, REPLPalette, build_rich_theme
from rich.console import Console
from rich.theme import Theme

# ── 共享 Theme 和 Console ──────────────────────────────────────────────

_OKLCH_THEME = Theme(build_rich_theme(DARK))


def _console() -> Console:
    """Return a Console with the OKLCH Theme, writing to a StringIO buffer."""
    return Console(
        file=io.StringIO(),
        force_terminal=True,
        width=80,
        theme=_OKLCH_THEME,
    )


# 每个语义名都要测试的 markup 模式
_SEMANTIC_NAMES = [
    "error",
    "success",
    "warning",
    "info",
    "primary",
    "secondary",
    "muted",
    "agent",
    "tool",
    "call",
    "result",
    "spinner",
    "spinner_highlight",
    "user_bg",
    "diff_add",
    "diff_remove",
]


class TestStage3eReplColors:
    """REPL 配色渲染 — 各类 semantic 标签不抛 MarkupError。"""

    def test_official_398_brand_and_neutral_tokens(self) -> None:
        assert DARK.primary == "#d77757"
        assert DARK.text == "#ffffff"
        assert DARK.text_muted == "#999999"
        assert DARK.border == "#505050"
        assert DARK.spinner_highlight == "#eb9f7f"
        assert DARK.diff_add == "#225c2b"
        assert DARK.diff_remove == "#7a2936"

    def test_spinner_highlight_keeps_old_palette_constructor_compatible(self) -> None:
        parameter = inspect.signature(REPLPalette).parameters["spinner_highlight"]
        assert parameter.default == "#eb9f7f"

    # ── 独立标签 ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("name", _SEMANTIC_NAMES)
    def test_standalone_tag(self, name: str) -> None:
        """``[name]text[/name]`` 渲染无异常。"""
        c = _console()
        c.print(f"[{name}]sample text[/{name}]")

    @pytest.mark.parametrize("name", _SEMANTIC_NAMES)
    def test_bold_nested(self, name: str) -> None:
        """``[bold][name]text[/name][/bold]`` 渲染无异常。"""
        c = _console()
        c.print(f"[bold][{name}]bolded text[/{name}][/bold]")

    # ── 复合场景 ──────────────────────────────────────────────────────

    def test_error_message_like_exception(self) -> None:
        """模拟 ``chat()`` 中异常输出的 markup 模式。"""
        c = _console()
        c.print("\n[error]Error: something went wrong[/error]")

    def test_success_status(self) -> None:
        c = _console()
        c.print("[success]✓ Task completed[/success]")

    def test_warning_banner(self) -> None:
        c = _console()
        c.print("[bold][warning]⚠ Warning[/warning][/bold]")

    def test_info_label(self) -> None:
        c = _console()
        c.print("[bold][info]Info label[/info][/bold]")

    def test_tool_call_line(self) -> None:
        """模拟官方工具调用行 ``⏺ ToolName(args)``。"""
        c = _console()
        c.print("[success]⏺[/success] [bold][tool]Read[/tool][/bold] [call](file.txt)[/call]")

    def test_tool_result_preview(self) -> None:
        c = _console()
        c.print("[muted]  ⎿  some result[/muted]")

    def test_tool_error(self) -> None:
        c = _console()
        c.print("[error]  ⎿  command failed[/error]")

    def test_muted_text_with_bold(self) -> None:
        c = _console()
        c.print("[bold][muted]dim header[/muted][/bold]")

    def test_agent_label(self) -> None:
        c = _console()
        c.print("[bold][agent]Assistant[/agent][/bold]")

    def test_mixed_standalone_and_nested(self) -> None:
        """混合场景：同一 ``print()`` 中既有独立标签也有嵌套标签。"""
        c = _console()
        c.print("[bold][agent]Assistant[/agent][/bold]\n[muted]  ⎿  done[/muted]")

    # ── 边界情况 ──────────────────────────────────────────────────────

    def test_long_text_no_markup_error(self) -> None:
        """长文本中嵌入语义标签不应导致 MarkupError。"""
        c = _console()
        c.print(
            "[error]Error: This is a very long error message that might "
            "trigger edge cases in the markup parser when combined with "
            "other text elements and nested formatting.[/error]"
        )

    def test_multiple_tags_same_line(self) -> None:
        c = _console()
        c.print("[success]✓[/success] [bold][info]Task[/info][/bold] [muted](2 done, 1 in progress)[/muted]")

    def test_tag_inside_fstring_interpolation(self) -> None:
        """模拟 chat() 中 ``f\"\\n[error]Error: {e}[/error]\"`` 的模式。"""
        c = _console()
        err_msg = "permission denied"
        c.print(f"\n[error]Error: {err_msg}[/error]")
        c.print(f"\n[warning]Warning: {err_msg}[/warning]")
        c.print(f"\n[success]Success: {err_msg}[/success]")


# ── prompt_toolkit 路径回归（与上面 Rich Console 路径并列）────────────
# ``LiveStatus._parse_rich_markup`` 把 ``[warning]text[/warning]`` 转成
# prompt_toolkit ``FormattedText`` 元组。prompt_toolkit 的样式解析器
# 不识别 OKLCH 语义名，遇到 ``warning`` 会抛
# ``ValueError: Wrong color format 'warning'`` —— 这一组测试守住
# ESC/Ctrl+C 时 ``status.update("[warning]Cancelling…[/warning]")``
# 的 redraw 路径，避免回归到那一帧崩溃的状态。
#
# 不真正启动 prompt_toolkit Application（CI 终端环境不稳），直接调用
# 静态解析函数并断言产物里没有裸语义名，只有 ``fg:#xxxxxx`` 或基础类名。

import re as _re  # noqa: E402

from clawcodex_ext.repl.live_status import LiveStatus  # noqa: E402

_PTK_BASE_STYLE = "class:status"
_OKLCH_SEMANTIC_TAGS = [
    "error",
    "success",
    "warning",
    "info",
    "primary",
    "secondary",
    "muted",
    "agent",
    "tool",
    "call",
    "result",
    "spinner",
    "spinner_highlight",
    "diff_add",
    "diff_remove",
]
_ANSI_TAGS_FOR_PTK = ["red", "green", "yellow", "blue", "cyan", "magenta", "white"]

_HEX_RE = _re.compile(r"^#[0-9a-fA-F]{6}$")


class TestStage3ePromptToolkitMarkup:
    """``LiveStatus._parse_rich_markup`` —— prompt_toolkit 路径不抛
    ``ValueError: Wrong color format``，所有 OKLCH 语义名映射到 hex。
    """

    @pytest.mark.parametrize("name", _OKLCH_SEMANTIC_TAGS)
    def test_oklch_semantic_tag_maps_to_hex(self, name: str) -> None:
        """每个 OKLCH 语义名都映射成 ``fg:#xxxxxx``，不再出现裸语义名。"""
        parts = LiveStatus._parse_rich_markup(f"[{name}]sample text[/{name}]", _PTK_BASE_STYLE)
        styled = [s for s, _ in parts if s != _PTK_BASE_STYLE]
        assert styled, f"[{name}] 没有产生任何 styled 行"
        for style in styled:
            assert "fg:#" in style, f"[{name}] 样式里缺 fg:#xxxxxx：{style!r} —— 可能是裸语义名泄露到 prompt_toolkit"
            for tok in style.split():
                if tok.startswith(("fg:", "bg:")):
                    value = tok.split(":", 1)[1]
                    assert _HEX_RE.match(value), (
                        f"[{name}] 颜色值 {value!r} 不是 hex —— prompt_toolkit 会抛 Wrong color format"
                    )

    @pytest.mark.parametrize("name", _ANSI_TAGS_FOR_PTK)
    def test_ansi_alias_tag_maps_to_hex(self, name: str) -> None:
        """``[red]`` / ``[yellow]`` 等 ANSI 别名也映射到 hex。"""
        parts = LiveStatus._parse_rich_markup(f"[{name}]text[/{name}]", _PTK_BASE_STYLE)
        styled = [s for s, _ in parts if s != _PTK_BASE_STYLE]
        assert styled
        for style in styled:
            assert "fg:#" in style

    def test_cancelling_message_used_by_repl_core(self) -> None:
        """回归测试：repl/core.py cancel 路径使用的
        ``[warning]Cancelling…[/warning]`` 必须能干净地解析，不抛
        ``ValueError: Wrong color format 'warning'``。

        这正是 ESC/Ctrl+C 时跑出 ``Unhandled exception in event loop``
        那条 traceback 的来源。
        """
        parts = LiveStatus._parse_rich_markup("[warning]Cancelling…[/warning]", _PTK_BASE_STYLE)
        cancel_styles = [s for s, txt in parts if txt and s != _PTK_BASE_STYLE]
        assert cancel_styles, "未找到 Cancelling 文本对应的样式行"
        for style in cancel_styles:
            assert "fg:#" in style, f"cancel 样式缺 fg:#xxxxxx：{style!r}"
            assert "warning" not in style.split(), f"裸语义名 'warning' 泄露到 prompt_toolkit 样式：{style!r}"

    def test_unknown_tag_falls_back_to_base_style(self) -> None:
        """未知 tag 退回 base_style，绝不裸传（防止未来再回归）。"""
        parts = LiveStatus._parse_rich_markup("[not_a_real_tag]text[/not_a_real_tag]", _PTK_BASE_STYLE)
        for style, _ in parts:
            assert "not_a_real_tag" not in style.split(), f"未知 tag 被裸传到样式：{style!r}"

    def test_plain_text_unchanged(self) -> None:
        """无 markup 的纯文本保持 base_style。"""
        parts = LiveStatus._parse_rich_markup("hello world", _PTK_BASE_STYLE)
        assert parts == [(_PTK_BASE_STYLE, "hello world")]

    def test_mixed_markup_and_plain_text(self) -> None:
        """普通文本与 markup 混合时，普通段保持 base_style，标记段带 fg:。"""
        parts = LiveStatus._parse_rich_markup("before [success]ok[/success] after", _PTK_BASE_STYLE)
        assert (_PTK_BASE_STYLE, "before ") in parts
        assert (_PTK_BASE_STYLE, " after") in parts
        ok_styles = [s for s, txt in parts if txt == "ok"]
        assert ok_styles
        for style in ok_styles:
            assert "fg:#" in style
