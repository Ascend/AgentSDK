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

"""Pure key-routing state machine for the two multi-model display phases."""

from __future__ import annotations

from .protocol import DisplayPhase


class MultiModelKeyboard:
    """Translate physical keys into stable, UI-independent actions."""

    def action_for(self, phase: DisplayPhase, key: str) -> str | None:
        key = key.lower()
        if phase is DisplayPhase.STREAMING:
            return {
                "left": "previous_tab",
                "right": "next_tab",
                "up": "scroll_up",
                "down": "scroll_down",
                "enter": "waiting",
                "f3": "toggle_columns",
            }.get(key)
        if phase is DisplayPhase.SELECTION:
            return {
                "up": "previous_result",
                "down": "next_result",
                "right": "expand",
                "left": "collapse",
                "enter": "adopt",
                "f2": "toggle_diff",
                "f3": "toggle_columns",
                "escape": "cancel",
                "q": "cancel",
            }.get(key)
        return None
