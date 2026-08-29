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

from clawcodex_ext.services.proactive import ProactiveController


def test_controller_lifecycle_and_block_ttl() -> None:
    now = 1_000.0
    ctrl = ProactiveController(clock_ms=lambda: now)
    seen = []
    ctrl.subscribe(seen.append)

    ctrl.activate("test", focus="full")
    assert ctrl.state.phase == "active"
    assert ctrl.state.focus == "full"
    assert ctrl.should_tick()

    ctrl.pause()
    assert ctrl.state.phase == "paused"
    assert not ctrl.should_tick()

    ctrl.resume()
    ctrl.set_context_blocked(True)
    assert ctrl.state.phase == "blocked"
    assert not ctrl.should_tick()

    now += 61_000
    assert ctrl.state.phase == "active"
    assert seen


def test_sleep_and_tick_summary() -> None:
    now = 2_000.0
    ctrl = ProactiveController(clock_ms=lambda: now)
    ctrl.activate("test")
    ctrl.enter_sleep(now + 500)
    assert ctrl.state.phase == "sleeping"

    now += 600
    assert ctrl.state.phase == "active"

    state = ctrl.record_tick(summary="  did   something useful  ")
    assert state.tick_count == 1
    assert state.last_tick_summary == "did something useful"
