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

"""Plan 3: context-aware Tab in :class:`PromptInput`.

Upstream ``useTypeahead`` binds ``tab -> autocomplete:accept`` only
inside the ``Autocomplete`` context. The Python port replicates the
same shape on top of Textual's ``on_key`` dispatch:

* Ghost-text visible -> Tab accepts the suggestion and stops the event.
* Ghost-text hidden -> Tab does NOT stop, so it bubbles up to the
  App's default ``focus_next`` binding and standard widget focus
  navigation continues to work.

These tests pin the two halves of that contract.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from textual.app import App, ComposeResult
from textual.widgets import Button

from clawcodex_ext.tui.widgets.prompt_input import PromptInput


class _Host(App):
    """Host that mounts a PromptInput next to a Button (focusable)."""

    def __init__(self, prompt: PromptInput) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        yield self._prompt
        yield Button("next", id="after-prompt")


def _make_prompt() -> PromptInput:
    return PromptInput(
        words_provider=lambda: ["/help", "/exit", "/repl"],
        vim_mode=False,
    )


@pytest.mark.asyncio
async def test_tab_with_visible_ghost_accepts_suggestion():
    """Tab accepts the ghost-text suggestion when one is shown."""
    prompt = _make_prompt()
    # Seed history so typing "git" matches a longer previous entry.
    prompt._history = ["git status", "ls"]

    host = _Host(prompt)
    async with host.run_test() as pilot:
        await pilot.pause()
        prompt._input.value = "git"
        prompt._input.cursor_position = 3
        # Force the on_input_changed handler to refresh ghost.
        # ``Input.Changed`` only takes (input, value); cursor_position
        # is read off the input widget itself in the handler.
        prompt.on_input_changed(type(prompt._input).Changed(prompt._input, prompt._input.value))
        await pilot.pause()
        # Ghost is now visible.
        assert not prompt._ghost_suggestion.has_class("-hidden")

        # Tab -> accept the suggestion.
        await pilot.press("tab")
        await pilot.pause()
        assert prompt._input.value == "git status"
        assert prompt._input.cursor_position == len("git status")
        # Ghost hidden after accept.
        assert prompt._ghost_suggestion.has_class("-hidden")


@pytest.mark.asyncio
async def test_tab_with_no_ghost_falls_through_to_focus_next():
    """Tab does NOT consume the event when no ghost is visible.

    We verify this indirectly by checking that focus moves to the next
    focusable widget (the Button). If the prompt input consumed Tab,
    the button would never receive focus.
    """
    prompt = _make_prompt()
    # No history match: typing "xyz" should not produce a ghost.
    prompt._history = ["git status"]

    host = _Host(prompt)
    async with host.run_test() as pilot:
        await pilot.pause()
        prompt._input.value = "xyz"
        prompt._input.cursor_position = 3
        prompt.on_input_changed(type(prompt._input).Changed(prompt._input, prompt._input.value))
        await pilot.pause()
        # No ghost visible — input is still "xyz" and ghost is hidden.
        assert prompt._ghost_suggestion.has_class("-hidden")
        assert prompt._input.value == "xyz"

        prompt._input.focus()
        await pilot.pause()
        # Tab should let the App's focus_next fire.
        await pilot.press("tab")
        await pilot.pause()
        # Focus moved off the input.
        assert prompt._input.has_focus is False
        # And landed on the Button (the next focusable widget).
        assert host.query_one("#after-prompt", Button).has_focus is True


@pytest.mark.asyncio
async def test_ghost_hint_mentions_tab():
    """The ghost-text hint must mention ``TAB``."""
    prompt = _make_prompt()
    prompt._history = ["git status"]

    host = _Host(prompt)
    async with host.run_test() as pilot:
        await pilot.pause()
        prompt._input.value = "git"
        prompt._input.cursor_position = 3
        prompt.on_input_changed(type(prompt._input).Changed(prompt._input, prompt._input.value))
        await pilot.pause()
        # Render the ghost widget to a string to inspect the hint.
        ghost_text = prompt._ghost_suggestion.renderable
        plain = ghost_text.plain
        assert "TAB" in plain
        # CTRL+E must NOT appear — only Tab completion is supported.
        assert "CTRL + E" not in plain
