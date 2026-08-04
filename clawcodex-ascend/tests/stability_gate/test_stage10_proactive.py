#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Stage 10 proactive mode smoke tests.

These keep the F-89 public integration points importable and verify the
lightweight state transitions without starting a long-running scheduler.
"""

from __future__ import annotations


class TestStage10Proactive:
    def test_proactive_package_exports_controller_and_emitter(self):
        from clawcodex_ext.services import proactive

        assert callable(proactive.ProactiveController)
        assert callable(proactive.TickEmitter)
        assert proactive.TICK_INTERVAL_MS == 30_000
        assert proactive.TICK_TAG == "tick"

    def test_proactive_state_machine_smoke(self):
        from clawcodex_ext.services.proactive import ProactiveController

        ctrl = ProactiveController(clock_ms=lambda: 1_000.0)
        ctrl.activate("stage10", focus="minimal")
        assert ctrl.state.phase == "active"
        assert ctrl.state.focus == "minimal"

        ctrl.pause()
        assert ctrl.state.phase == "paused"

        ctrl.resume()
        assert ctrl.should_tick() is True

        ctrl.set_context_blocked(True)
        assert ctrl.state.phase == "blocked"

    def test_proactive_prompt_event_round_trip(self):
        from clawcodex_ext.query.outbox_types import (
            ProactivePromptEvent,
            outbox_event_from_dict,
        )

        event = ProactivePromptEvent(prompt="<tick>12:00:00</tick>", source="tick")
        assert event.get("type") == "proactive_prompt"
        assert event["prompt"] == "<tick>12:00:00</tick>"

        restored = outbox_event_from_dict({"type": "proactive_prompt", "prompt": event.prompt, "source": event.source})
        assert isinstance(restored, ProactivePromptEvent)
        assert restored.prompt == event.prompt

    def test_sleep_tool_registered(self):
        from clawcodex_ext.tool_system.tools import SleepTool

        assert SleepTool.name == "Sleep"
