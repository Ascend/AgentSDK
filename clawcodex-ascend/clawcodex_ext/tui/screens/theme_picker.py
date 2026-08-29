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

"""Theme picker dialog.

Port of ``components/ThemePicker.tsx``. Lists the available palettes
(``auto``, ``dark``, ``light``, ``claude``) and resolves with the
selected theme id. ``auto`` resolves to the OS-derived palette at
boot time via :func:`src.tui.theme.resolve_auto_theme`.
"""

from __future__ import annotations
# pylint: disable=W0201

from typing import Callable, Iterator, Sequence

from textual.widget import Widget

from ..widgets.select_list import SelectList, SelectOption
from .dialog_base import DialogScreen


class ThemePickerScreen(DialogScreen[str | None]):
    """Modal picker resolving with the selected theme id."""

    title_text = "Select theme"
    footer_hint = "Enter to apply · Esc to cancel"

    def __init__(
        self,
        *,
        themes: Sequence[str],
        current: str | None = None,
        on_persist: Callable[[str], None] | None = None,
        on_preview: Callable[[str | None], None] | None = None,
    ) -> None:
        """Create the theme picker.

        ``on_preview(name)`` mirrors the TS ``usePreviewTheme`` hook:
        it fires every time the cursor highlights a different theme,
        letting the host swap the palette live. When the user cancels
        with Esc we call ``on_preview(None)`` so the host can restore
        the previously-active theme.
        """

        super().__init__()
        self._themes = list(themes)
        self._current = (current or "").lower()
        self._on_persist = on_persist
        self._on_preview = on_preview

    def build_body(self) -> Iterator[Widget]:
        options: list[SelectOption] = []
        current_index = 0
        for idx, name in enumerate(self._themes):
            desc = "current" if name == self._current else None
            options.append(SelectOption(label=name, value=name, description=desc))
            if name == self._current:
                current_index = idx
        self._select = SelectList(
            options,
            initial_index=current_index,
            allow_cancel=True,
        )
        yield self._select

    def _post_mount(self) -> None:
        self._select.focus()

    def on_select_list_option_highlighted(self, event: SelectList.OptionHighlighted) -> None:
        if self._on_preview is None:
            return
        try:
            self._on_preview(str(event.option.value))
        except Exception:  # nosec B110
            pass  # Intentional best-effort path; the surrounding fallback remains valid.

    def on_select_list_option_selected(self, event: SelectList.OptionSelected) -> None:
        value = str(event.option.value)
        if self._on_persist is not None:
            try:
                self._on_persist(value)
            except Exception:  # nosec B110
                pass  # Intentional best-effort path; the surrounding fallback remains valid.
        self.dismiss(value)

    def on_select_list_selection_cancelled(self, _: SelectList.SelectionCancelled) -> None:
        if self._on_preview is not None:
            try:
                self._on_preview(None)
            except Exception:  # nosec B110
                pass  # Intentional best-effort path; the surrounding fallback remains valid.
        self.dismiss(None)
