#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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


"""Idle-return confirmation dialog.

Port of ``components/IdleReturnDialog.tsx``. Surfaced when the user
returns to the REPL after an idle period so they can explicitly
decide whether to continue the existing conversation, clear it, or
suppress future prompts. Resolves with the chosen action string;
``None`` on Esc means "dismiss" (restore pending input only).
"""

from __future__ import annotations
# pylint: disable=W0201

from typing import Callable, Iterator, Literal

from textual.widget import Widget

from ..widgets.select_list import SelectList, SelectOption
from .dialog_base import DialogScreen


IdleAction = Literal["continue", "clear", "never", "dismiss"]


class IdleReturnScreen(DialogScreen["IdleAction"]):
    """Blocking dialog asking how to resume after idle."""

    title_text = "Welcome back"
    footer_hint = "Enter to choose · Esc to dismiss"

    def __init__(
        self,
        *,
        idle_minutes: int,
        total_input_tokens: int = 0,
        on_choice: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._idle = idle_minutes
        self._tokens = total_input_tokens
        self._on_choice = on_choice
        tok_line = ""
        if total_input_tokens > 0:
            tok_line = f" · {_fmt_tokens(total_input_tokens)} tokens so far"
        self.subtitle_text = f"You've been idle for {idle_minutes} min{tok_line}. How would you like to continue?"

    def build_body(self) -> Iterator[Widget]:
        self._select = SelectList(
            [
                SelectOption(
                    label="Continue this conversation",
                    value="continue",
                    description="Pick up where you left off",
                ),
                SelectOption(
                    label="Start a new conversation",
                    value="clear",
                    description="Clear history and send as new",
                ),
                SelectOption(
                    label="Don't ask again",
                    value="never",
                    description="Suppress idle-return prompts",
                ),
            ],
            allow_cancel=True,
        )
        yield self._select

    def _post_mount(self) -> None:
        self._select.focus()

    def on_select_list_option_selected(self, event: SelectList.OptionSelected) -> None:
        value = str(event.option.value)
        if self._on_choice is not None:
            try:
                self._on_choice(value)
            except Exception:  # nosec B110
                pass  # The optional callback is isolated so it cannot interrupt the owning event loop.
        self.dismiss(value)  # type: ignore[arg-type]

    def on_select_list_selection_cancelled(self, _: SelectList.SelectionCancelled) -> None:
        if self._on_choice is not None:
            try:
                self._on_choice("dismiss")
            except Exception:  # nosec B110
                pass  # The optional callback is isolated so it cannot interrupt the owning event loop.
        self.dismiss("dismiss")


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)
