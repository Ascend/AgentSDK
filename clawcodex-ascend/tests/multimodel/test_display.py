#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Behavioural coverage for the display-only half."""

from __future__ import annotations

import json

from clawcodex_ext.capabilities.multimodel_protocol import MultiModelResult
from clawcodex_ext.multimodel.display import (
    DiffDisplay,
    DisplayPhase,
    MultiModelBridge,
    SideBySideDisplay,
    SummaryBuilder,
    TabbedDisplay,
)
from clawcodex_ext.multimodel.display.protocol import ModelDisplayState
from clawcodex_ext.providers.base import ChatResponse


def result(slot: str, content: str = "answer") -> MultiModelResult:
    return MultiModelResult(slot, ChatResponse(content, slot, {}, "stop"), 1800, {"input": 10, "output": 22})


def test_streaming_tabs_wrap_and_do_not_finish_early() -> None:
    display = TabbedDisplay(["one", "two"])
    display.on_progress("one", "hello")
    assert display.selected.content == "hello"
    assert display.handle_key("left") == "previous_tab"
    assert display.selected.slot == "two"
    assert display.phase is DisplayPhase.STREAMING
    assert display.handle_key("enter") == "waiting"


def test_selection_keys_expand_adopt_and_cancel() -> None:
    display = TabbedDisplay(["one", "two"])
    display.complete_all()
    display.handle_key("right")
    assert display.selected.expanded
    display.handle_key("down")
    assert display.selected.slot == "two"
    display.handle_key("enter")
    assert display.phase is DisplayPhase.ADOPTED

    cancelled = TabbedDisplay(["one"])
    cancelled.complete_all()
    cancelled.handle_key("escape")
    assert cancelled.phase is DisplayPhase.CANCELLED


def test_bridge_transitions_only_when_every_slot_completes_and_adopts_selected() -> None:
    adopted: list[str] = []
    bridge = MultiModelBridge(["one", "two"], on_adopt=lambda item: adopted.append(item.slot_name))
    bridge.on_progress("one", "partial")
    bridge.on_complete(result("one", "first"))
    assert bridge.display.phase is DisplayPhase.STREAMING
    bridge.on_complete(result("two", "second"))
    assert bridge.display.phase is DisplayPhase.SELECTION
    bridge.handle_key("down")
    bridge.handle_key("enter")
    assert adopted == ["two"]


def test_headless_text_and_json_preserve_required_result_fields() -> None:
    states = [ModelDisplayState("one", "text", 2300, {"input": 1, "output": 9}, "complete")]
    assert "one (2.3s, 9 tok)" in SummaryBuilder.build_text(states)
    payload = json.loads(SummaryBuilder.build_json(states))
    assert payload == {
        "multimodel": True,
        "strategy": "parallel",
        "results": [
            {
                "slot": "one",
                "duration_ms": 2300,
                "tokens": {"input": 1, "output": 9},
                "content": "text",
                "status": "complete",
            }
        ],
    }


def test_wide_mode_and_diff_pair_are_safe_for_small_slot_counts() -> None:
    columns = SideBySideDisplay(179)
    assert not columns.toggle()
    columns.terminal_width = 180
    assert columns.toggle()
    assert columns.scroll(-1) == 0
    diff = DiffDisplay(["one", "two"])
    assert diff.pair == ("one", "two")
    assert any(line.startswith("-") for line in diff.lines("old", "new"))


def test_bridge_exposes_ndjson_events_and_requires_two_expanded_results_for_diff() -> None:
    bridge = MultiModelBridge(["one", "two"], terminal_width=180)
    assert json.loads(bridge.stream_json_event("one", chunk="hello"))["type"] == "multimodel_progress"
    bridge.display.complete_all()
    assert bridge.handle_key("f2") == "diff_unavailable"
    bridge.handle_key("right")
    bridge.handle_key("down")
    bridge.handle_key("right")
    assert bridge.handle_key("f2") == "toggle_diff"
    assert bridge.diff is not None
    bridge.handle_key("f3")
    assert bridge.side_by_side.enabled
