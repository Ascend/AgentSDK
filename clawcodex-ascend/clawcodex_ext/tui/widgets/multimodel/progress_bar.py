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
