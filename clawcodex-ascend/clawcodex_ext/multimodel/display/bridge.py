#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Bridge between a multi-model scheduler and any display surface."""

from __future__ import annotations

import json
from collections.abc import Callable

from clawcodex_ext.capabilities.multimodel_protocol import MultiModelResult  # pylint: disable=no-name-in-module

from .protocol import ModelDisplayState
from .diff_display import DiffDisplay
from .side_by_side import SideBySideDisplay
from .summary import SummaryBuilder
from .tab_display import TabbedDisplay


class MultiModelBridge:
    """Owns display state and calls the adoption callback exactly once."""

    def __init__(
        self,
        slots: list[str],
        *,
        on_adopt: Callable[[MultiModelResult], None] | None = None,
        terminal_width: int = 0,
    ) -> None:
        self.display = TabbedDisplay(slots)
        self.side_by_side = SideBySideDisplay(terminal_width)
        self.diff: DiffDisplay | None = None
        self._on_adopt = on_adopt
        self._completed: dict[str, MultiModelResult] = {}

    def on_progress(self, slot: str, chunk: str, *, status: str = "streaming") -> None:
        self.display.on_progress(slot, chunk, status=status)

    def on_complete(self, result: MultiModelResult) -> None:
        self._completed[result.slot_name] = result
        state = self.display._slot(result.slot_name)
        complete = ModelDisplayState.from_result(result)
        state.content, state.duration_ms, state.tokens = complete.content, complete.duration_ms, complete.tokens
        state.status, state.error = complete.status, complete.error
        if len(self._completed) >= len(self.display.results):
            self.display.complete_all()

    def handle_key(self, key: str) -> str | None:
        action = self.display.handle_key(key)
        if action == "toggle_columns":
            self.side_by_side.toggle()
        elif action == "toggle_diff":
            expanded = [state.slot for state in self.display.results if state.expanded]
            if len(expanded) < 2:
                return "diff_unavailable"
            self.diff = DiffDisplay(expanded)
        if action == "adopt" and self.display.selected is not None:
            result = self._completed.get(self.display.selected.slot)
            if result is not None and self._on_adopt is not None:
                self._on_adopt(result)
        return action

    def render_text(self) -> str:
        return SummaryBuilder.build_text(self.display.results)

    def render_json(self, *, strategy: str = "parallel") -> str:
        return SummaryBuilder.build_json(self.display.results, strategy=strategy)

    def stream_json_event(self, slot: str, *, chunk: str = "", status: str = "streaming") -> str:
        """Serialize one NDJSON event for a non-TUI streaming caller."""
        if status == "complete":
            result = self._completed.get(slot)
            payload: dict[str, object] = {"type": "multimodel_complete", "slot": slot}
            if result is not None:
                payload["duration_ms"] = result.duration_ms
        else:
            payload = {"type": "multimodel_progress", "slot": slot, "status": status, "chunk": chunk}
        return json.dumps(payload, ensure_ascii=False)
