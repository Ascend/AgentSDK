# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Unit tests for :class:`SelectList`.

Select list reactives only work when the widget is mounted inside an
active Textual app, so each test boots a tiny harness app that mounts
the widget before driving it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from textual.app import App, ComposeResult

from src.tui.widgets.select_list import SelectList, SelectOption


class _Host(App):
    def __init__(self, select_list: SelectList) -> None:
        super().__init__()
        self._select_list = select_list

    def compose(self) -> ComposeResult:
        yield self._select_list


def _options(*labels: str, disabled: set[str] = frozenset()) -> list[SelectOption]:
    return [SelectOption(label=label, disabled=(label in disabled)) for label in labels]


@pytest.mark.asyncio
async def test_initial_cursor_is_first_option():
    sl = SelectList(_options("a", "b", "c"))
    async with _Host(sl).run_test() as pilot:
        await pilot.pause()
        assert sl.cursor == 0
        assert sl.current is not None and sl.current.label == "a"


@pytest.mark.asyncio
async def test_move_wraps_around():
    sl = SelectList(_options("a", "b", "c"))
    async with _Host(sl).run_test() as pilot:
        await pilot.pause()
        sl.action_move(-1)
        assert sl.cursor == 2
        sl.action_move(1)
        assert sl.cursor == 0


@pytest.mark.asyncio
async def test_move_skips_disabled_rows():
    sl = SelectList(_options("a", "b", "c", disabled={"b"}))
    async with _Host(sl).run_test() as pilot:
        await pilot.pause()
        sl.action_move(1)
        assert sl.current is not None and sl.current.label == "c"


@pytest.mark.asyncio
async def test_set_options_resets_cursor_by_default():
    sl = SelectList(_options("a", "b", "c"))
    async with _Host(sl).run_test() as pilot:
        await pilot.pause()
        sl.action_move(1)
        sl.set_options([SelectOption(label="x"), SelectOption(label="y")])
        assert sl.cursor == 0


@pytest.mark.asyncio
async def test_set_options_can_keep_cursor():
    sl = SelectList(_options("a", "b", "c"))
    async with _Host(sl).run_test() as pilot:
        await pilot.pause()
        sl.action_move(1)
        sl.set_options(
            [
                SelectOption(label="x"),
                SelectOption(label="y"),
                SelectOption(label="z"),
            ],
            keep_cursor=True,
        )
        assert sl.cursor == 1


@pytest.mark.asyncio
async def test_empty_options_is_safe():
    sl = SelectList([])
    async with _Host(sl).run_test() as pilot:
        await pilot.pause()
        assert sl.current is None
        sl.action_move(1)
        sl.action_select()
        sl.action_cancel()


@pytest.mark.asyncio
async def test_move_to_edge():
    sl = SelectList(_options("a", "b", "c", "d"))
    async with _Host(sl).run_test() as pilot:
        await pilot.pause()
        sl.action_move_to_edge(1)
        assert sl.cursor == 3
        sl.action_move_to_edge(-1)
        assert sl.cursor == 0
