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

from __future__ import annotations
# pylint: disable=E0611

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from clawcodex_ext.multimodel.display.protocol import ModelDisplayState

from .result_card import ModelResultCard


class MultiModelSummaryPanel(Vertical):
    """Completed-results panel; cards retain independent expanded state."""

    def __init__(self, states: list[ModelDisplayState] | None = None) -> None:
        super().__init__()
        self._states = states or []

    def compose(self) -> ComposeResult:
        yield Static("✅ 全部完成", classes="-multimodel-title")
        for index, state in enumerate(self._states):
            card = ModelResultCard()
            card.set_result(state, selected=index == 0)
            yield card

    def set_results(self, states: list[ModelDisplayState], selected: int) -> None:
        self._states = states
        cards = list(self.query(ModelResultCard))
        for index, state in enumerate(states):
            if index < len(cards):
                cards[index].set_result(state, selected=index == selected)
