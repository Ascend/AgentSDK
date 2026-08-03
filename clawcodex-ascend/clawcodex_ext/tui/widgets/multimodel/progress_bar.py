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

from rich.text import Text
from textual.widgets import Static


class ModelProgressBars(Static):
    def set_progress(self, states: list[object], selected: int) -> None:
        out = Text()
        for index, state in enumerate(states):
            percent = getattr(state, "progress_percent", 0)
            fill = "█" * round(percent / 5) + "░" * (20 - round(percent / 5))
            marker = "●" if index == selected else "○"
            out.append(
                f"{marker} {getattr(state, 'slot')}  {fill} {percent}%\n", style="cyan" if index == selected else "dim"
            )
        self.update(out)
