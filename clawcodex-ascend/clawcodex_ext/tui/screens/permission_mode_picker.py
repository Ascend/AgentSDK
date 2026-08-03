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

"""Permission mode picker dialog.

Allows the user to switch between permission modes (Default, Plan,
Accept edits, Bypass permissions, Don't Ask) via a modal screen.
"""

from __future__ import annotations

from typing import Callable, Iterator

from textual.widget import Widget

from ..widgets.select_list import SelectList, SelectOption
from .dialog_base import DialogScreen


class PermissionModePickerScreen(DialogScreen[str | None]):
    """Modal picker that resolves with the selected permission mode.

    The result is ``None`` when the user cancels (Esc).
    """

    title_text = "Permission mode"
    subtitle_text = "Choose the permission level for this session."
    footer_hint = "Enter to select · Esc to cancel"

    # Ordered cycle as defined in permissions/cycle.py:get_next_permission_mode
    # Auto mode is not part of the Shift+Tab cycle but can be selected manually.
    MODES = [
        ("default", "Default", "Ask before running each tool"),
        ("acceptEdits", "Accept edits", "Auto-approve file edit operations"),
        ("plan", "Plan mode", "Auto-approve read-only operations"),
        ("bypassPermissions", "Bypass permissions", "Run all tools without prompting"),
        ("dontAsk", "Don't ask", "Never prompt, auto-approve everything"),
        ("auto", "Auto mode", "LLM-based intelligent auto-approval"),
    ]

    def __init__(
        self,
        *,
        current_mode: str | None = None,
        is_bypass_available: bool = False,
        on_select: Callable[[str | None], None] | None = None,
    ) -> None:
        super().__init__()
        self._current = current_mode or "default"
        # Filter out bypass if not available
        self._modes = [
            (key, title, desc) for key, title, desc in self.MODES if key != "bypassPermissions" or is_bypass_available
        ]
        self._on_select = on_select
        self._select: SelectList | None = None

    def build_body(self) -> Iterator[Widget]:
        options: list[SelectOption] = []
        current_index = 0
        for idx, (key, title, desc) in enumerate(self._modes):
            label = f"{title} ({key})"
            options.append(SelectOption(label=label, value=key, description=desc))
            if key == self._current:
                current_index = idx
        self._select = SelectList(
            options or [SelectOption(label="(no modes available)", disabled=True)],
            initial_index=current_index,
            allow_cancel=True,
        )
        yield self._select

    def _post_mount(self) -> None:
        if self._select is not None:
            self._select.focus()

    def on_select_list_option_selected(self, event: SelectList.OptionSelected) -> None:
        mode = str(event.option.value)
        if self._on_select is not None:
            try:
                self._on_select(mode)
            except Exception:  # nosec B110
                pass
        self.dismiss(mode)

    def on_select_list_selection_cancelled(self, _: SelectList.SelectionCancelled) -> None:
        if self._on_select is not None:
            try:
                self._on_select(None)
            except Exception:  # nosec B110
                pass
        self.dismiss(None)
