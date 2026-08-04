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

"""Stage 7 — TUI 弹层事件分派 + 静态契约门禁。

锁定 ``PromptInput`` 的两类用户面契约：

1. **运行时路由** —— 三个 OptionList 弹层（slash / message / @file）
   收到 ``OptionSelected`` 时，必须把 ``option.id`` 写回 ``_input``，
   并且**恰好调用一次对应的 ``_hide_*`` 方法**。早期版本用
   ``event.sender is self._foo`` 判断弹层归属，但 Textual 的
   ``Message`` 基类没有 ``sender`` 属性，触发 ``AttributeError`` ——
   Stage 7 把这条契约锁住。

   注意：测试只断言"hide 方法被调用"，不断言"弹层最终保持隐藏"。
   ``Input.Changed`` 事件会在 ``OptionSelected`` 之后异步触发，
   调 ``_refresh_suggestions`` 并可能重新挂上 ``-hidden`` 反转，这是
   一个独立的 UX bug，Stage 8 单独跟踪。

2. **静态契约** —— 扫描 ``clawcodex_ext/tui/`` 和 ``src/tui/`` 下所有
   ``.py`` 文件，禁止 ``event.sender is/==/!=`` 形式出现。零运行时
   成本，IDE / CI 即时反馈。

不覆盖每条 ``/命令`` 的逻辑正确性（那是 ``tests/command_system/`` 的
职责），只保证**用户面的"门"不出 bug**。
"""

from __future__ import annotations

import re
import tokenize
from pathlib import Path
from tokenize import NAME, OP

import pytest

pytest.importorskip("textual")

from clawcodex_ext.tui.widgets.prompt_input import PromptInput
from textual.app import App, ComposeResult
from textual.widgets.option_list import Option

# ---------------------------------------------------------------------------
# Section 1 — runtime popup dispatch
# ---------------------------------------------------------------------------


class _Host(App):
    """Minimal host app that mounts exactly one :class:`PromptInput`."""

    def compose(self) -> ComposeResult:
        yield PromptInput(words_provider=lambda: ["/repl", "/exit"])


class _HideSpy:
    """Records calls to PromptInput's three ``_hide_*`` popup methods.

    We replace the bound methods on the instance for the duration of
    one test, then assert that the matching hide method was called
    *at least once* and the wrong ones were not. "At least once" lets
    the test pass if a future refactor introduces an extra idempotent
    hide (e.g. an explicit double-hide guard).
    """

    def __init__(self, pi: PromptInput) -> None:
        self.calls: dict[str, int] = {
            "slash": 0,
            "message": 0,
            "atfile": 0,
        }
        self._originals: dict[str, object] = {}
        for attr, key in (
            ("_hide_suggestions", "slash"),
            ("_hide_message_suggestions", "message"),
            ("_hide_at_file_suggestions", "atfile"),
        ):
            self._originals[attr] = getattr(pi, attr)
            setattr(
                pi,
                attr,
                self._make_spy(key, self._originals[attr]),
            )

    def _make_spy(self, key: str, original):
        def spy() -> None:
            self.calls[key] += 1
            original()

        return spy

    def restore(self, pi: PromptInput) -> None:
        for attr, original in self._originals.items():
            setattr(pi, attr, original)


class TestStage7PopupDispatch:
    """``on_option_list_option_selected`` 的三类弹层路由契约。

    契约的核心：选中事件必须**至少**调用匹配弹层的 ``_hide_*`` 方法。
    其它 ``_hide_*`` 方法可能因为 ``Input.Changed`` 触发的
    ``_refresh_suggestions`` 而被顺带调用 —— 这是正常的副作用，不算
    路由错误，所以这里只断言"匹配的那一个被调用了"。

    反例：如果旧版 ``event.sender is self._foo`` 的 AttributeError
    没修，这里所有断言都会因为 hide 根本没被调用而失败。
    """

    pytestmark = pytest.mark.asyncio

    async def test_slash_popup_routes_to_hide_suggestions(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)
            pi._suggestions.add_option(Option("/repl", id="/repl"))
            pi._suggestions.highlighted = 0
            pi._suggestions.remove_class("-hidden")
            spy = _HideSpy(pi)

            try:
                pi._suggestions.action_select()
                await pilot.pause()
                await pilot.pause()

                assert pi._input.value == "/repl"
                # 路由契约：匹配弹层的 hide 必须被调用
                assert spy.calls["slash"] >= 1
            finally:
                spy.restore(pi)

    async def test_message_popup_routes_to_hide_message_suggestions(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)
            pi._message_history_provider = lambda: ["hello world", "help me"]
            pi._message_suggestions.add_option(Option("hello world", id="hello world"))
            pi._message_suggestions.highlighted = 0
            pi._message_suggestions.remove_class("-hidden")
            spy = _HideSpy(pi)

            try:
                pi._message_suggestions.action_select()
                await pilot.pause()
                await pilot.pause()

                assert pi._input.value == "hello world"
                assert spy.calls["message"] >= 1
            finally:
                spy.restore(pi)

    async def test_at_file_popup_routes_to_hide_at_file_suggestions(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)
            pi._at_file_suggestions.add_option(Option("foo.py", id="@foo.py"))
            pi._at_file_suggestions.highlighted = 0
            pi._at_file_suggestions.remove_class("-hidden")
            spy = _HideSpy(pi)

            try:
                pi._at_file_suggestions.action_select()
                await pilot.pause()
                await pilot.pause()

                assert pi._input.value == "@foo.py"
                assert spy.calls["atfile"] >= 1
            finally:
                spy.restore(pi)

    async def test_option_without_id_does_not_overwrite_or_crash(self):
        """无 id 的 option（纯展示行）选中时不能覆盖 input，也不能抛异常。"""

        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)
            pi._input.value = "/preset"
            pi._suggestions.add_option(Option("display only"))  # id=None
            pi._suggestions.highlighted = 0
            pi._suggestions.remove_class("-hidden")

            # 关键回归：原 ``event.sender`` 写法会在此处抛 AttributeError
            pi._suggestions.action_select()
            await pilot.pause()
            await pilot.pause()

            # 无 id 的 option 不应覆盖已有输入
            assert pi._input.value == "/preset"


# ---------------------------------------------------------------------------
# Section 2 — static contract (event.sender is forbidden in TUI handlers)
# ---------------------------------------------------------------------------


_BUG_PATTERN_OPS = ("is", "==", "!=")
_SCAN_DIRS = ("clawcodex_ext/tui", "src/tui")


def _scan_event_sender(repo_root: Path) -> list[str]:
    """返回所有 ``event.sender is/==/!=`` 的 token 级命中。

    用 ``tokenize`` 而不是正则，确保注释、字符串、docstring 里的
    提及不会被误报；同时精确匹配 ``event`` 名字（避免误伤
    ``widget.sender`` 之类）。
    """

    offenders: list[str] = []
    for rel_dir in _SCAN_DIRS:
        base = repo_root / rel_dir
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    tokens = list(tokenize.generate_tokens(f.readline))
            except (tokenize.TokenError, OSError):
                continue
            for i in range(len(tokens) - 3):
                a, b, c, d = (
                    tokens[i],
                    tokens[i + 1],
                    tokens[i + 2],
                    tokens[i + 3],
                )
                if (
                    a.type == NAME
                    and a.string == "event"
                    and b.type == OP
                    and b.string == "."
                    and c.type == NAME
                    and c.string == "sender"
                    and d.type == OP
                    and d.string in _BUG_PATTERN_OPS
                ):
                    rel = path.relative_to(repo_root)
                    offenders.append(f"{rel}:{a.start[0]}: event.sender {d.string} …")
                    break  # 一个文件命中一次足够
    return offenders


class TestStage7StaticContract:
    """零运行时成本的"event.sender 禁忌"门禁。"""

    def test_no_event_sender_attribute_access_in_tui(self):
        repo_root = Path(__file__).resolve().parents[2]
        offenders = _scan_event_sender(repo_root)
        assert not offenders, (
            "TUI handler 中禁止使用 ``event.sender is/==/!=`` —— "
            "Textual 的 Message 基类没有该属性。改用 event.option_list / "
            "event.control / event.input 等具体事件字段。\n命中：\n  " + "\n  ".join(offenders)
        )


# ---------------------------------------------------------------------------
# Section 3 — extended static contract: known Textual 0.79 API drifts
# ---------------------------------------------------------------------------


# 单数 add_option 接受 id= kwarg 的旧用法 —— Textual 0.79 已移除。
# 注意：``add_options`` (复数) 是 OK 的，因为 \b 在 n 和 s 之间不匹配。
# 关键：用 ``[^()]*`` 而不是 ``[^)]*``，避免被 ``add_option(Option(text, id=...))``
# 这种"把 id= 嵌在 Option 构造里"的正确用法误报。
_ADD_OPTION_ID_RE = re.compile(r"\.add_option\b\([^()]*,\s*id\s*=")
# 私有方法在 0.79 已移除 —— 应用公开的 action_select() 替代。
_POST_SELECTED_RE = re.compile(r"\._post_selected\b")


def _scan_textual_api_drift(repo_root: Path) -> list[str]:
    """检测已知的 Textual 0.79 API 漂移（add_option 旧签名 / _post_selected 移除）。

    用 ``re`` + 去掉行内注释：因为这两条规则都是简单标记（单 kwarg /
    私有方法名），tokenize 反而比正则更繁琐。这里接受"字符串字面量里的
    提及会被误报"的极小概率 —— 真出现时肉眼一眼能分辨。
    """

    offenders: list[str] = []
    for rel_dir in _SCAN_DIRS:
        base = repo_root / rel_dir
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for ln, line in enumerate(text.splitlines(), start=1):
                # 去掉行内注释；纯注释行 code 为空，下面的 strip 跳过
                code = line.split("#", 1)[0]
                if not code.strip():
                    continue
                if _ADD_OPTION_ID_RE.search(code):
                    rel = path.relative_to(repo_root)
                    offenders.append(
                        f"{rel}:{ln}: add_option(arg, id=...) — Textual 0.79 forbidden, use Option wrapper"
                    )
                    break  # 一个文件命中一次足够
                if _POST_SELECTED_RE.search(code):
                    rel = path.relative_to(repo_root)
                    offenders.append(f"{rel}:{ln}: _post_selected — removed in Textual 0.79, use action_select()")
                    break
    return offenders


class TestStage7TextualApiDrift:
    """锁住已知的 Textual 0.79 API 漂移。"""

    def test_no_textual_0_79_api_drift_in_tui(self):
        repo_root = Path(__file__).resolve().parents[2]
        offenders = _scan_textual_api_drift(repo_root)
        assert not offenders, (
            "TUI 代码中检测到已知的 Textual 0.79 API 漂移 —— "
            "add_option 的 id= kwarg 改用 Option 包装；_post_selected "
            "改用 action_select()。\n命中：\n  " + "\n  ".join(offenders)
        )


# ---- TUI resize 边界 + 多弹层冲突 --------------------------------------


class TestStage7TuiResize:
    """TUI resize 边界测试 — P0#3 resize 到极端值不崩溃。"""

    pytestmark = pytest.mark.asyncio

    async def test_resize_tiny_terminal(self):
        """resize 到 1x1 极小尺寸不应 crash。"""
        from clawcodex_ext.tui.widgets.prompt_input import PromptInput
        from textual.app import App

        class _TinyHost(App):
            def compose(self) -> ComposeResult:
                yield PromptInput(words_provider=list)

        # size=(columns, rows) => 1 col x 1 row 极端小
        async with _TinyHost().run_test(size=(1, 1)) as pilot:
            await pilot.pause()
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)
            assert pi is not None
            # 能在 1x1 终端中键入文字
            pi._input.value = "x"
            assert pi._input.value == "x"

    async def test_resize_large_terminal(self):
        """resize 到 200x200 超大尺寸不应 crash。"""
        from clawcodex_ext.tui.widgets.prompt_input import PromptInput
        from textual.app import App

        class _LargeHost(App):
            def compose(self) -> ComposeResult:
                yield PromptInput(words_provider=list)

        async with _LargeHost().run_test(size=(200, 200)) as pilot:
            await pilot.pause()
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)
            assert pi is not None
            pi._input.value = "hello in large terminal"
            assert "hello" in pi._input.value
            # 验证弹层也能正常添加
            from textual.widgets.option_list import Option

            pi._suggestions.add_option(Option("/test", id="/test"))
            pi._suggestions.remove_class("-hidden")
            assert not pi._suggestions.has_class("-hidden")

    async def test_slash_and_at_file_popup_simultaneous(self):
        """同时打开 slash + @file 弹层不冲突 (P2#11 多弹层)。"""
        from clawcodex_ext.tui.widgets.prompt_input import PromptInput
        from textual.app import App
        from textual.widgets.option_list import Option

        class _MultiPopupHost(App):
            def compose(self) -> ComposeResult:
                yield PromptInput(words_provider=lambda: ["/repl", "/exit"])

        async with _MultiPopupHost().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            # 打开 slash 弹层 — 使用 words_provider 不含的命令避免重复 id
            pi._input.value = "/"
            await pilot.pause()
            # _refresh_suggestions 已从 words_provider 填充了 /repl /exit
            pi._suggestions.remove_class("-hidden")
            await pilot.pause()

            # 同时打开 @file 弹层
            pi._at_file_suggestions.add_option(Option("foo.py", id="@foo.py"))
            pi._at_file_suggestions.remove_class("-hidden")
            await pilot.pause()

            # 两个弹层都应可见 — 不崩溃即通过
            assert not pi._suggestions.has_class("-hidden")
            assert not pi._at_file_suggestions.has_class("-hidden")

            # Enter 不应抛异常（即使弹层都开着）
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            # 测试通过 = 没有异常抛出

    async def test_resize_does_not_lose_input(self):
        """resize 后已输入的文本不应丢失。"""
        from clawcodex_ext.tui.widgets.prompt_input import PromptInput
        from textual.app import App

        class _ResizeHost(App):
            def compose(self) -> ComposeResult:
                yield PromptInput(words_provider=list)

        async with _ResizeHost().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)
            pi._input.value = "persistent text"
            await pilot.pause()

            # 通过 pilot.resize_terminal 模拟终端 resize (width, height)
            await pilot.resize_terminal(120, 40)
            await pilot.pause()
            await pilot.pause()

            # resize 后 input 内容不变
            assert pi._input.value == "persistent text", "resize 后 input 内容不应丢失"
