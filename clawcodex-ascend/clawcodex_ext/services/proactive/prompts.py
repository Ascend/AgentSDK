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

from __future__ import annotations

import time
from typing import Literal

from .constants import TICK_TAG
from .controller import get_default_controller

TerminalFocus = Literal["full", "medium", "minimal", "off"]

_FULL = """<proactive-mode phase="{phase}" focus="{focus}">
You are in proactive mode. When a <{tag}>HH:MM:SS</{tag}> message arrives, continue useful work without waiting for the user when the next step is clear.
Current goal:
{goal_block}
Tick count: {tick_count}
Next tick: {next_tick_at_str}
Last tick: {last_tick}
</proactive-mode>"""

_MEDIUM = """<proactive-mode phase="{phase}" focus="{focus}">
On <{tag}> ticks, make bounded progress on the active goal when the next step is clear; ask briefly when blocked.
{goal_block}
Ticks: {tick_count}; next: {next_tick_at_str}; last: {last_tick}
</proactive-mode>"""

_MINIMAL = """<proactive-mode phase="{phase}" focus="{focus}">
<{tag}> ticks may resume concise progress on the active goal.
{goal_block}
</proactive-mode>"""


def get_proactive_section(
    terminal_focus: TerminalFocus = "medium",
    *,
    session_id: str | None = None,
) -> str | None:
    if terminal_focus == "off":
        return None
    ctrl = get_default_controller()
    state = ctrl.state
    if not state.is_active:
        return None

    goal_block = _active_goal_block(session_id)
    next_tick = (
        time.strftime("%H:%M:%S", time.localtime(state.next_tick_at / 1000))
        if state.next_tick_at is not None
        else "(not scheduled)"
    )
    template = {"full": _FULL, "medium": _MEDIUM, "minimal": _MINIMAL}[terminal_focus]
    return template.format(
        tag=TICK_TAG,
        phase=state.phase,
        focus=state.focus,
        goal_block=goal_block or "No active goal is set; ask the user briefly on first wake-up.",
        tick_count=state.tick_count,
        next_tick_at_str=next_tick,
        last_tick=state.last_tick_summary or "(none)",
    )


def _active_goal_block(session_id: str | None) -> str | None:
    try:
        from clawcodex_ext.goal.store import GoalStore  # pylint: disable=no-name-in-module

        store = GoalStore()
        if session_id is None:
            return None
        goal = store.get_thread_goal(session_id)
        if goal is None:
            return None
        status = getattr(goal.status, "value", str(goal.status))
        return f'<active-goal status="{status}">\n{goal.objective}\n</active-goal>'
    except Exception:
        return None
