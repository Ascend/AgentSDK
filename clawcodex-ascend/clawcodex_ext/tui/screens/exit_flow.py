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


"""Exit confirmation dialog.

Port of ``components/ExitFlow.tsx`` + ``components/WorktreeExitDialog.tsx``.
Phase 2 models the user-facing behaviour (confirm before leaving, with
an option to discard in-flight conversation) but intentionally skips
the git-worktree aware branch — worktree cleanup belongs in Phase 3
alongside the diff/MCP work that already reads git state.
"""

from __future__ import annotations
# pylint: disable=W0201

from typing import Callable, Iterator, Literal

from textual.widget import Widget

from ..widgets.select_list import SelectList, SelectOption
from .dialog_base import DialogScreen


ExitAction = Literal["quit", "quit-clear", "cancel"]


class ExitFlowScreen(DialogScreen["ExitAction"]):
    """Confirm-exit dialog pushed by ``/exit``.

    (Ctrl+C / Ctrl+D exit directly via the double-press flow in
    ``ClawCodexTUI._request_exit`` and do NOT push this dialog.)

    Resolves with:
      * ``"quit"``       — leave, keep conversation in session history.
      * ``"quit-clear"`` — leave and clear the current conversation.
      * ``"cancel"``     — stay (Esc).
    """

    title_text = "Leave ClawCodex?"
    footer_hint = "Enter to choose · Esc to stay"
    border_variant = "warning"

    def __init__(
        self,
        *,
        has_inflight_work: bool = False,
        on_choice: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._inflight = has_inflight_work
        self._on_choice = on_choice
        self.subtitle_text = (
            "Work is still running — quitting will abort it."
            if has_inflight_work
            else "Your conversation will be saved to the session history."
        )

    def build_body(self) -> Iterator[Widget]:
        self._select = SelectList(
            [
                SelectOption(
                    label="Quit",
                    value="quit",
                    description="save and exit",
                ),
                SelectOption(
                    label="Quit & clear conversation",
                    value="quit-clear",
                    description="discard history then exit",
                ),
                SelectOption(
                    label="Stay",
                    value="cancel",
                    description="cancel exit",
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
                self._on_choice("cancel")
            except Exception:  # nosec B110
                pass  # The optional callback is isolated so it cannot interrupt the owning event loop.
        self.dismiss("cancel")
