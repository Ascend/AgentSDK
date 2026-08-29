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

# pylint: disable=no-name-in-module

from __future__ import annotations

from clawcodex_ext.services.ultraplan import Plan, PlanExecutor, Step, StepStatus, SubPlan


def test_executor_transition_hook_receives_plan_id_and_transition() -> None:
    plan = Plan(
        id="p1",
        title="Hooked",
        goal="Hooked",
        sub_plans=[
            SubPlan(
                id="sp1",
                title="Work",
                description="Work",
                steps=[Step(id="s1", title="Step", description="Step")],
            )
        ],
    )
    seen = []
    executor = PlanExecutor(plan, transition_hooks=[lambda plan_id, tr: seen.append((plan_id, tr))])

    executor.mark_in_progress("s1")

    assert seen[0][0] == "p1"
    assert seen[0][1].step_id == "s1"
    assert seen[0][1].new_status is StepStatus.IN_PROGRESS
