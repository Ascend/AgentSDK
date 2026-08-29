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

import asyncio

from clawcodex_ext.services.ultraplan import (
    AcceptanceCriteria,
    CheckKind,
    Plan,
    PlanStatus,
    Step,
    StepKind,
    StepStatus,
    SubPlan,
)
from clawcodex_ext.services.ultraplan.audit import AuditLogger
from clawcodex_ext.services.ultraplan.controller import UltraplanController
from clawcodex_ext.services.ultraplan.store import PlanStore


def test_run_plan_advances_all_runnable_steps(tmp_path) -> None:
    marker = tmp_path / "marker.txt"
    marker.write_text("ok", encoding="utf-8")
    plan = Plan(
        id="p1",
        title="Run all",
        goal="Run all",
        sub_plans=[
            SubPlan(
                id="sp1",
                title="Work",
                description="Work",
                steps=[
                    Step(
                        id="s1",
                        title="Check marker",
                        description="Check marker",
                        kind=StepKind.VERIFY,
                        criteria=[
                            AcceptanceCriteria(
                                id="c1",
                                description="marker exists",
                                kind=CheckKind.FILE_EXISTS,
                                target=str(marker),
                            )
                        ],
                    ),
                    Step(
                        id="s2",
                        title="Dependent",
                        description="Dependent",
                        depends_on=["s1"],
                    ),
                ],
            )
        ],
    )
    store = PlanStore(tmp_path / "plans")
    store.save(plan)
    controller = UltraplanController(
        planner=None,
        store=store,
        audit=AuditLogger(tmp_path / "audit"),
    )

    progress = asyncio.run(controller.run_plan("p1"))
    saved = store.load("p1")

    assert progress.completed == 2
    assert saved.status is PlanStatus.COMPLETED
    assert [step.status for step in saved.all_steps()] == [
        StepStatus.COMPLETED,
        StepStatus.COMPLETED,
    ]
    assert (tmp_path / "audit" / "p1.ndjson").exists()


def test_run_plan_stops_on_required_criteria_failure(tmp_path) -> None:
    plan = Plan(
        id="p1",
        title="Run fail",
        goal="Run fail",
        sub_plans=[
            SubPlan(
                id="sp1",
                title="Work",
                description="Work",
                steps=[
                    Step(
                        id="s1",
                        title="Check missing",
                        description="Check missing",
                        criteria=[
                            AcceptanceCriteria(
                                id="c1",
                                description="missing file",
                                kind=CheckKind.FILE_EXISTS,
                                target=str(tmp_path / "missing.txt"),
                            )
                        ],
                    ),
                    Step(id="s2", title="After", description="After"),
                ],
            )
        ],
    )
    store = PlanStore(tmp_path / "plans")
    store.save(plan)
    controller = UltraplanController(planner=None, store=store)

    progress = asyncio.run(controller.run_plan("p1"))
    saved = store.load("p1")

    assert progress.failed == 1
    assert saved.find_step("s1")[1].status is StepStatus.FAILED  # type: ignore[index]
    assert saved.find_step("s2")[1].status is StepStatus.PENDING  # type: ignore[index]
