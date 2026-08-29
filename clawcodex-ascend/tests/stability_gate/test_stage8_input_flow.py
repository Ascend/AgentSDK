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

"""Tests for stage8 input flow."""

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
# ---------------------------------------------------------------------------


class TestStage8InputSubmit:
    """Tests for TestStage8InputSubmit."""

    pytestmark = pytest.mark.asyncio

    async def test_submit_posts_prompt_submitted_and_clears_input(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            host: _Host = pilot.app  # type: ignore[assignment]
            pi = host.query_one(PromptInput)

            pi._input.value = "hello world"
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            assert len(host.submitted) == 1
            assert host.submitted[0].text == "hello world"
            assert pi._input.value == ""

    async def test_submit_with_highlighted_popup_accepts_instead_of_posting(self):
        """Verify submit with highlighted popup accepts instead of posting."""
        async with _Host().run_test() as pilot:
            await pilot.pause()
            host: _Host = pilot.app  # type: ignore[assignment]
            pi = host.query_one(PromptInput)

            pi._input.value = "/re"
            pi._suggestions.add_option(Option("/repl", id="/repl"))
            pi._suggestions.highlighted = 0
            pi._suggestions.remove_class("-hidden")
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            assert pi._input.value == "/repl"
            assert host.submitted == [], "弹层高亮时 Enter 优先 accept，不应发出 PromptSubmitted"


class TestStage8InputChange:
    """Tests for TestStage8InputChange."""

    pytestmark = pytest.mark.asyncio

    async def test_slash_token_opens_command_popup(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            pi._input.value = "/re"
            await pilot.pause()
            await pilot.pause()

            assert not pi._suggestions.has_class("-hidden"), "输入 / 前缀时应打开 slash 弹层"

    async def test_non_slash_closes_slash_popup(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            pi._input.value = "/re"
            await pilot.pause()
            await pilot.pause()
            assert not pi._suggestions.has_class("-hidden")

            pi._input.value = "regular text"
            await pilot.pause()
            await pilot.pause()

            assert pi._suggestions.has_class("-hidden"), "输入非 / 内容时应关闭 slash 弹层"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestStage8KeyBindings:
    """Tests for TestStage8KeyBindings."""

    pytestmark = pytest.mark.asyncio

    async def test_shift_tab_posts_permission_mode_cycle_requested(self):
        async with _Host().run_test() as pilot:
            await pilot.pause()
            host: _Host = pilot.app  # type: ignore[assignment]
            host.query_one(PromptInput)

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
# ---------------------------------------------------------------------------


class TestStage8Paste:
    """Tests for TestStage8Paste."""

    pytestmark = pytest.mark.asyncio

    async def test_empty_paste_does_not_modify_input_but_posts_message(self):
        """Verify empty paste does not modify input but posts message."""
        async with _Host().run_test() as pilot:
            await pilot.pause()
            host: _Host = pilot.app  # type: ignore[assignment]
            pi = host.query_one(PromptInput)

            pi._input.value = "existing draft"
            await pilot.pause()

            info = pi.handle_paste("")
            await pilot.pause()
            await pilot.pause()

            assert info.is_empty is True
            assert info.is_image_drag is False
            assert info.length == 0
            assert pi._input.value == "existing draft"
            assert pi.last_paste is info
            assert len(host.pasted) == 1
            assert host.pasted[0].info.is_empty is True

    async def test_text_paste_inserts_at_cursor(self):
        """Verify text paste inserts at cursor."""
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            pi._input.value = "hello "
            pi._input.cursor_position = len("hello ")
            await pilot.pause()

            info = pi.handle_paste("world")
            await pilot.pause()
            await pilot.pause()

            assert info.is_empty is False
            assert info.is_image_drag is False
            assert info.text == "world"
            assert info.line_count == 1
            assert pi._input.value == "hello world"
            assert pi._input.cursor_position == len("hello world")

    async def test_image_drag_paste_classified_correctly(self):
        """Verify image drag paste classified correctly."""
        async with _Host().run_test() as pilot:
            await pilot.pause()
            pi = pilot.app.query_one(PromptInput)

            info = pi.handle_paste("/tmp/screenshot.png")

            assert info.is_empty is False
            assert info.is_image_drag is True
            assert info.text == "/tmp/screenshot.png"
            assert "/tmp/screenshot.png" in pi._input.value
