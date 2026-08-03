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

"""Intent Forecast picker dialog."""

from __future__ import annotations
# pylint: disable=E0611

from typing import Iterator

from textual.widget import Widget

from clawcodex_ext.intent_forecast.messages import ForecastResult

from ..widgets.select_list import SelectList, SelectOption
from .dialog_base import DialogScreen


class ForecastPickerScreen(DialogScreen[str | None]):
    """Keyboard picker for accepting or dismissing forecast suggestions."""

    title_text = "Forecast"
    subtitle_text = "Choose a next step to submit."
    footer_hint = "Enter to accept - Esc to ignore"

    def __init__(self, result: ForecastResult) -> None:
        super().__init__()
        self._result = result
        self._select: SelectList | None = None

    def build_body(self) -> Iterator[Widget]:
        rows: list[SelectOption] = []
        for idx, suggestion in enumerate(self._result.suggestions, 1):
            confidence = int(round(suggestion.confidence * 100))
            detail = suggestion.reason or f"{confidence}% confidence"
            if suggestion.reason:
                detail = f"{detail} ({confidence}%)"
            rows.append(
                SelectOption(
                    label=f"{idx}. {suggestion.title}",
                    value=str(idx),
                    description=detail,
                )
            )
        rows.append(
            SelectOption(
                label="Ignore these suggestions",
                value=None,
                description="Dismiss without submitting a prompt",
            )
        )
        self._select = SelectList(rows, allow_cancel=True)
        yield self._select

    def _post_mount(self) -> None:
        if self._select is not None:
            self._select.focus()

    def on_select_list_option_selected(self, event: SelectList.OptionSelected) -> None:
        value = event.option.value
        self.dismiss(str(value) if value is not None else None)

    def on_select_list_selection_cancelled(self, _: SelectList.SelectionCancelled) -> None:
        self.dismiss(None)


__all__ = ["ForecastPickerScreen"]
