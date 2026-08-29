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

"""Cost-threshold acknowledgement dialog.

Port of ``components/CostThresholdDialog.tsx``. Shown once per
process after session spend crosses the configured threshold so the
user is aware of API billing. Resolves with ``True`` when the user
acknowledges (Enter / Got it), ``False`` on Esc.
"""

from __future__ import annotations
# pylint: disable=W0201

from typing import Callable, Iterator

from textual.widget import Widget

from ..widgets.select_list import SelectList, SelectOption
from .dialog_base import DialogScreen


class CostThresholdScreen(DialogScreen[bool]):
    """Dialog surfaced once per session when spend crosses a threshold."""

    title_text = "You've crossed the cost threshold"
    border_variant = "warning"
    footer_hint = "Enter to acknowledge · Esc to dismiss"

    def __init__(
        self,
        *,
        provider: str,
        amount_usd: float,
        on_acknowledge: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._amount = amount_usd
        self._on_acknowledge = on_acknowledge
        self.subtitle_text = (
            f"This session has used ~${amount_usd:.2f} of {provider} credits. "
            "Check your provider dashboard for details."
        )

    def build_body(self) -> Iterator[Widget]:
        self._select = SelectList(
            [SelectOption(label="Got it, thanks!", value="ack")],
            allow_cancel=True,
        )
        yield self._select

    def _post_mount(self) -> None:
        self._select.focus()

    def on_select_list_option_selected(self, _: SelectList.OptionSelected) -> None:
        if self._on_acknowledge is not None:
            try:
                self._on_acknowledge()
            except Exception:  # nosec B110
                pass  # Intentional best-effort path; the surrounding fallback remains valid.
        self.dismiss(True)

    def on_select_list_selection_cancelled(self, _: SelectList.SelectionCancelled) -> None:
        self.dismiss(False)
