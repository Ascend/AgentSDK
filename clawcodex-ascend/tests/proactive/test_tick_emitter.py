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

from __future__ import annotations

from clawcodex_ext.services.kairos import TickEvent
from clawcodex_ext.services.proactive import ProactiveController, TickEmitter


def test_tick_emitter_injects_tick_to_outbox() -> None:
    ctrl = ProactiveController()
    outbox = []
    emitter = TickEmitter(controller=ctrl, outbox=outbox)
    ctrl.activate("test")

    text = emitter._on_tick_event(
        TickEvent(
            scheduler_id="test",
            tick_number=1,
            scheduled_at=1.0,
            actual_at=1.0,
        )
    )

    assert text is not None
    assert text.startswith("<tick>")
    assert outbox[0].get("type") == "proactive_prompt"
    assert outbox[0].prompt == text
    assert ctrl.state.tick_count == 1


def test_tick_emitter_blocks_when_compact_fails() -> None:
    ctrl = ProactiveController()
    ctrl.activate("test")

    def compact() -> None:
        raise RuntimeError("boom")

    emitter = TickEmitter(
        controller=ctrl,
        outbox=[],
        should_compact_first=lambda: True,
        compact_callback=compact,
    )

    assert emitter.emit_now() is None
    assert ctrl.state.phase == "blocked"
