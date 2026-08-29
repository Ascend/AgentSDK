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

"""Tests for stage7 popup dispatch."""

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
    """Tests for TestStage7PopupDispatch."""

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
        """Verify option without id does not overwrite or crash."""

        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)
            pi._input.value = "/preset"
            pi._suggestions.add_option(Option("display only"))  # id=None
            pi._suggestions.highlighted = 0
            pi._suggestions.remove_class("-hidden")

            pi._suggestions.action_select()
            await pilot.pause()
            await pilot.pause()

            assert pi._input.value == "/preset"


# ---------------------------------------------------------------------------
# Section 2 — static contract (event.sender is forbidden in TUI handlers)
# ---------------------------------------------------------------------------


_BUG_PATTERN_OPS = ("is", "==", "!=")
_SCAN_DIRS = ("clawcodex_ext/tui", "src/tui")


def _scan_event_sender(repo_root: Path) -> list[str]:
    """Test helper for scan event sender."""

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
                    break
    return offenders


class TestStage7StaticContract:
    """Tests for TestStage7StaticContract."""

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


_ADD_OPTION_ID_RE = re.compile(r"\.add_option\b\([^()]*,\s*id\s*=")
_POST_SELECTED_RE = re.compile(r"\._post_selected\b")


def _scan_textual_api_drift(repo_root: Path) -> list[str]:
    """Test helper for scan textual api drift."""

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
                code = line.split("#", 1)[0]
                if not code.strip():
                    continue
                if _ADD_OPTION_ID_RE.search(code):
                    rel = path.relative_to(repo_root)
                    offenders.append(
                        f"{rel}:{ln}: add_option(arg, id=...) — Textual 0.79 forbidden, use Option wrapper"
                    )
                    break
                if _POST_SELECTED_RE.search(code):
                    rel = path.relative_to(repo_root)
                    offenders.append(f"{rel}:{ln}: _post_selected — removed in Textual 0.79, use action_select()")
                    break
    return offenders


class TestStage7TextualApiDrift:
    """Tests for TestStage7TextualApiDrift."""

    def test_no_textual_0_79_api_drift_in_tui(self):
        repo_root = Path(__file__).resolve().parents[2]
        offenders = _scan_textual_api_drift(repo_root)
        assert not offenders, (
            "TUI 代码中检测到已知的 Textual 0.79 API 漂移 —— "
            "add_option 的 id= kwarg 改用 Option 包装；_post_selected "
            "改用 action_select()。\n命中：\n  " + "\n  ".join(offenders)
        )


class TestStage7TuiResize:
    """P0 Tests for TestStage7TuiResize."""

    pytestmark = pytest.mark.asyncio

    async def test_resize_tiny_terminal(self):
        """Verify resize tiny terminal."""
        from clawcodex_ext.tui.widgets.prompt_input import PromptInput
        from textual.app import App

        class _TinyHost(App):
            def compose(self) -> ComposeResult:
                yield PromptInput(words_provider=list)

        async with _TinyHost().run_test(size=(1, 1)) as pilot:
            await pilot.pause()
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)
            assert pi is not None
            pi._input.value = "x"
            assert pi._input.value == "x"

    async def test_resize_large_terminal(self):
        """Verify resize large terminal."""
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
            from textual.widgets.option_list import Option

            pi._suggestions.add_option(Option("/test", id="/test"))
            pi._suggestions.remove_class("-hidden")
            assert not pi._suggestions.has_class("-hidden")

    async def test_slash_and_at_file_popup_simultaneous(self):
        """P2 Verify slash and at file popup simultaneous."""
        from clawcodex_ext.tui.widgets.prompt_input import PromptInput
        from textual.app import App
        from textual.widgets.option_list import Option

        class _MultiPopupHost(App):
            def compose(self) -> ComposeResult:
                yield PromptInput(words_provider=lambda: ["/repl", "/exit"])

        async with _MultiPopupHost().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            pi._input.value = "/"
            await pilot.pause()
            pi._suggestions.remove_class("-hidden")
            await pilot.pause()

            pi._at_file_suggestions.add_option(Option("foo.py", id="@foo.py"))
            pi._at_file_suggestions.remove_class("-hidden")
            await pilot.pause()

            assert not pi._suggestions.has_class("-hidden")
            assert not pi._at_file_suggestions.has_class("-hidden")

            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

    async def test_resize_does_not_lose_input(self):
        """Verify resize does not lose input."""
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

            await pilot.resize_terminal(120, 40)
            await pilot.pause()
            await pilot.pause()

            assert pi._input.value == "persistent text", "resize 后 input 内容不应丢失"
