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

"""Tabbed state model used while model slots stream in parallel."""

from __future__ import annotations

from .keyboard import MultiModelKeyboard
from .protocol import DisplayPhase, ModelDisplayState


class TabbedDisplay:
    def __init__(self, slots: list[str]) -> None:
        self.results = [ModelDisplayState(slot=slot) for slot in slots]
        self.selected_index = 0
        self.scroll_offset = 0
        self.phase = DisplayPhase.STREAMING
        self.keyboard = MultiModelKeyboard()

    @property
    def selected(self) -> ModelDisplayState | None:
        return self.results[self.selected_index] if self.results else None

    def on_progress(self, slot: str, chunk: str, *, status: str = "streaming") -> None:
        result = self._slot(slot)
        result.content += chunk
        result.status = status

    def complete_all(self) -> None:
        self.phase = DisplayPhase.SELECTION

    def handle_key(self, key: str) -> str | None:
        action = self.keyboard.action_for(self.phase, key)
        if not action or not self.results:
            return action
        if action == "previous_tab":
            self.selected_index = (self.selected_index - 1) % len(self.results)
        elif action == "next_tab":
            self.selected_index = (self.selected_index + 1) % len(self.results)
        elif action == "scroll_up":
            self.scroll_offset = max(0, self.scroll_offset - 1)
        elif action == "scroll_down":
            self.scroll_offset += 1
        elif action == "previous_result":
            self.selected_index = (self.selected_index - 1) % len(self.results)
        elif action == "next_result":
            self.selected_index = (self.selected_index + 1) % len(self.results)
        elif action == "expand":
            self.selected.expanded = True
        elif action == "collapse":
            self.selected.expanded = False
        elif action == "adopt":
            self.phase = DisplayPhase.ADOPTED
        elif action == "cancel":
            self.phase = DisplayPhase.CANCELLED
        return action

    def _slot(self, slot: str) -> ModelDisplayState:
        for result in self.results:
            if result.slot == slot:
                return result
        result = ModelDisplayState(slot=slot)
        self.results.append(result)
        return result
