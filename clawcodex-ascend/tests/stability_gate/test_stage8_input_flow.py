"""Stage 8 — PromptInput input + paste 行为契约门禁。

锁住 :class:`PromptInput` 的三类用户面契约：

1. **Input 事件流** — ``on_input_submitted`` 在非弹层场景下必须
   发出 :class:`PromptSubmitted` 并清空 draft；弹层高亮时 Enter
   优先接受 option，不发 ``PromptSubmitted``。``on_input_changed``
   在 ``/`` 前缀时打开 slash 弹层，在非 ``/`` 内容时关闭。

2. **按键绑定** — ``Shift+Tab`` 必须发 :class:`PermissionModeCycleRequested`
   （即使用户在 Input 子组件里）。``Ctrl+L`` 必须清空 draft。

3. **Bracketed paste** — :meth:`PromptInput.handle_paste` 的三种
   输入必须正确分类：空粘贴不改 input 但仍发 :class:`PromptPasted`；
   文本粘贴插入到光标位置；image 拖拽必须标记为 ``is_image_drag``。

不覆盖 :class:`PromptPasted` 业务层（图像 attach），只保证 **用户面
的"门"不出 bug** —— 也就是 ``handle_paste`` 的分类正确性。

实现说明：Textual 通过 ``on_<MessageType>`` 方法把消息路由到 app /
screen / widget。本门禁用 ``_Host`` 上的 ``on_PromptSubmitted`` /
``on_PermissionModeCycleRequested`` / ``on_PromptPasted`` 收集所有
``post_message`` 行为 —— 这是 Textual 官方推荐的可观察模式，比
monkey-patch 内部 ``_handle_message`` 更稳。
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from clawcodex_ext.tui.messages import (
    PermissionModeCycleRequested,
    PromptPasted,
)
from clawcodex_ext.tui.widgets.prompt_input import PromptInput, PromptSubmitted
from textual.app import App, ComposeResult
from textual.widgets.option_list import Option

# ---------------------------------------------------------------------------
# Host app — captures posted messages via Textual's on_X dispatch
# ---------------------------------------------------------------------------


class _Host(App):
    """Minimal host app that mounts one :class:`PromptInput` and records
    every message of the types we care about via ``on_<Type>`` handlers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[PromptSubmitted] = []
        self.cycle_requests: list[PermissionModeCycleRequested] = []
        self.pasted: list[PromptPasted] = []

    def compose(self) -> ComposeResult:
        yield PromptInput(words_provider=lambda: ["/repl", "/exit"])

    def on_prompt_submitted(self, message: PromptSubmitted) -> None:
        self.submitted.append(message)

    def on_permission_mode_cycle_requested(self, message: PermissionModeCycleRequested) -> None:
        self.cycle_requests.append(message)

    def on_prompt_pasted(self, message: PromptPasted) -> None:
        self.pasted.append(message)


# ---------------------------------------------------------------------------
# Section 1 — Input 事件流（submit + change）
# ---------------------------------------------------------------------------


class TestStage8InputSubmit:
    """``on_input_submitted`` 的两条主路径：纯 submit / 弹层高亮 accept。"""

    pytestmark = pytest.mark.asyncio

    async def test_submit_posts_prompt_submitted_and_clears_input(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            host: _Host = pilot.app  # type: ignore[assignment]
            pi = host.query_one(PromptInput)

            # 写入文本（直接绕过 keystroke，模拟用户输入）
            pi._input.value = "hello world"
            await pilot.pause()

            # 模拟 Enter 提交
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            # 必须发出 PromptSubmitted 且 input 已被清空
            assert len(host.submitted) == 1
            assert host.submitted[0].text == "hello world"
            assert pi._input.value == ""

    async def test_submit_with_highlighted_popup_accepts_instead_of_posting(self):
        """弹层高亮时 Enter 接受 option，不发 PromptSubmitted。"""
        async with _Host().run_test() as pilot:
            await pilot.pause()
            host: _Host = pilot.app  # type: ignore[assignment]
            pi = host.query_one(PromptInput)

            # 模拟用户在弹层打开且高亮的状态
            pi._input.value = "/re"
            pi._suggestions.add_option(Option("/repl", id="/repl"))
            pi._suggestions.highlighted = 0
            pi._suggestions.remove_class("-hidden")
            await pilot.pause()

            # Enter 应当接受弹层 option
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            # input 被替换为完整 option id；不应发出 PromptSubmitted
            assert pi._input.value == "/repl"
            assert host.submitted == [], "弹层高亮时 Enter 优先 accept，不应发出 PromptSubmitted"


class TestStage8InputChange:
    """``on_input_changed`` 驱动 slash 弹层的开 / 关。"""

    pytestmark = pytest.mark.asyncio

    async def test_slash_token_opens_command_popup(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            pi._input.value = "/re"
            await pilot.pause()
            await pilot.pause()

            # 弹层应打开（无 -hidden class）
            assert not pi._suggestions.has_class("-hidden"), "输入 / 前缀时应打开 slash 弹层"

    async def test_non_slash_closes_slash_popup(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            # 先打开弹层
            pi._input.value = "/re"
            await pilot.pause()
            await pilot.pause()
            assert not pi._suggestions.has_class("-hidden")

            # 再输入非 / 内容
            pi._input.value = "regular text"
            await pilot.pause()
            await pilot.pause()

            # 弹层应被关闭
            assert pi._suggestions.has_class("-hidden"), "输入非 / 内容时应关闭 slash 弹层"


# ---------------------------------------------------------------------------
# Section 2 — 按键绑定（Shift+Tab / Ctrl+L）
# ---------------------------------------------------------------------------


class TestStage8KeyBindings:
    """PromptInput 容器级别的按键绑定契约。"""

    pytestmark = pytest.mark.asyncio

    async def test_shift_tab_posts_permission_mode_cycle_requested(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            host: _Host = pilot.app  # type: ignore[assignment]
            host.query_one(PromptInput)

            # 即使 Input 有焦点，Shift+Tab 也必须被 PromptInput.on_key
            # 拦截并转发为 PermissionModeCycleRequested
            await pilot.press("shift+tab")
            await pilot.pause()
            await pilot.pause()

            assert len(host.cycle_requests) == 1, "Shift+Tab 必须发出 PermissionModeCycleRequested"

    async def test_ctrl_l_clears_draft(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            pi._input.value = "some draft text"
            await pilot.pause()
            assert pi._input.value == "some draft text"

            await pilot.press("ctrl+l")
            await pilot.pause()
            await pilot.pause()

            assert pi._input.value == "", "Ctrl+L 必须清空 draft (action_clear_draft → clear)"


# ---------------------------------------------------------------------------
# Section 3 — Bracketed paste 分类
# ---------------------------------------------------------------------------


class TestStage8Paste:
    """``handle_paste`` 的三种 payload 契约。"""

    pytestmark = pytest.mark.asyncio

    async def test_empty_paste_does_not_modify_input_but_posts_message(self):
        """空粘贴（macOS Cmd+V 图像 sentinel）必须不改 input 但仍发消息。"""
        async with _Host().run_test() as pilot:
            await pilot.pause()
            host: _Host = pilot.app  # type: ignore[assignment]
            pi = host.query_one(PromptInput)

            pi._input.value = "existing draft"
            await pilot.pause()

            info = pi.handle_paste("")
            await pilot.pause()
            await pilot.pause()

            # 分类正确
            assert info.is_empty is True
            assert info.is_image_drag is False
            assert info.length == 0
            # input 不被改写
            assert pi._input.value == "existing draft"
            # last_paste 测试接缝已记录
            assert pi.last_paste is info
            # PromptPasted 仍被发出 —— 上层据此查询剪贴板图像
            assert len(host.pasted) == 1
            assert host.pasted[0].info.is_empty is True

    async def test_text_paste_inserts_at_cursor(self):
        """文本粘贴必须插入到光标位置。"""
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            pi._input.value = "hello "
            pi._input.cursor_position = len("hello ")
            await pilot.pause()

            info = pi.handle_paste("world")
            await pilot.pause()
            await pilot.pause()

            # 分类正确
            assert info.is_empty is False
            assert info.is_image_drag is False
            assert info.text == "world"
            assert info.line_count == 1
            # input 在光标处插入文本
            assert pi._input.value == "hello world"
            # 光标位置移到插入末尾
            assert pi._input.cursor_position == len("hello world")

    async def test_image_drag_paste_classified_correctly(self):
        """图像文件拖拽路径必须被标记为 is_image_drag。"""
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            # 模拟一个图像文件绝对路径的 paste
            info = pi.handle_paste("/tmp/screenshot.png")

            assert info.is_empty is False
            assert info.is_image_drag is True
            assert info.text == "/tmp/screenshot.png"
            # 文本依然会被插入（host 决定是否替换为 image attach）
            assert "/tmp/screenshot.png" in pi._input.value
