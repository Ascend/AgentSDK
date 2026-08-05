#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""PlanAdjuster tests: add/replace/remove step and sub_plan primitives."""
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import pytest

from src.services.ultraplan import (
    DuplicateStepIdError,
    DuplicateSubPlanIdError,
    Plan,
    PlanAdjuster,
    PlanStatus,
    Step,
    StepHasDependentsError,
    StepKind,
    StepNotFoundError,
    StepStatus,
    SubPlan,
    SubPlanNotFoundError,
)


def _plan() -> Plan:
    sp1 = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[
            Step(id="s1", title="T1", description="D1", kind=StepKind.IMPLEMENT),
            Step(
                id="s2",
                title="T2",
                description="D2",
                kind=StepKind.VERIFY,
                depends_on=["s1"],
            ),
        ],
    )
    sp2 = SubPlan(
        id="sp2",
        title="B",
        description="d",
        steps=[Step(id="s3", title="T3", description="D3", kind=StepKind.OTHER)],
    )
    return Plan(id="p1", title="My plan", goal="Goal", sub_plans=[sp1, sp2])


def test_adjuster_add_step_appends_by_default() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    new_step = Step(id="s4", title="T4", description="D4")
    adj.add_step("sp1", new_step)
    assert plan.sub_plans[0].steps[-1] is new_step


def test_adjuster_add_step_at_position() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    new_step = Step(id="s4", title="T4", description="D4")
    adj.add_step("sp1", new_step, position=0)
    assert plan.sub_plans[0].steps[0] is new_step


def test_adjuster_add_step_rejects_duplicate_id() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    dup = Step(id="s1", title="T", description="D")
    with pytest.raises(DuplicateStepIdError):
        adj.add_step("sp1", dup)


def test_adjuster_add_step_rejects_unknown_sub_plan() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(SubPlanNotFoundError):
        adj.add_step("missing", Step(id="s4", title="T", description="D"))


def test_adjuster_add_step_rejects_unknown_dep() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    bad = Step(id="s4", title="T", description="D", depends_on=["s9"])
    with pytest.raises(ValueError):
        adj.add_step("sp1", bad)


def test_adjuster_replace_step_keeps_id() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    new = Step(id="s1", title="T1'", description="D1'", status=StepStatus.IN_PROGRESS)
    adj.replace_step("s1", new)
    assert plan.sub_plans[0].steps[0] is new


def test_adjuster_replace_step_with_rename() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    new = Step(id="s1-renamed", title="T", description="D")
    adj.replace_step("s1", new)
    ids = [s.id for s in plan.sub_plans[0].steps]
    assert "s1-renamed" in ids
    assert "s1" not in ids


def test_adjuster_replace_step_rejects_rename_to_existing() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    bad = Step(id="s2", title="T", description="D")
    with pytest.raises(DuplicateStepIdError):
        adj.replace_step("s1", bad)


def test_adjuster_replace_step_unknown() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(StepNotFoundError):
        adj.replace_step("missing", Step(id="missing", title="T", description="D"))


def test_adjuster_remove_step() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    # s3 in sp2 has no dependents, so it can be removed directly.
    adj.remove_step("s3")
    sp2 = plan.find_sub_plan("sp2")
    assert sp2 is not None
    assert all(s.id != "s3" for s in sp2.steps)


def test_adjuster_remove_step_with_dependents_rejected() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(StepHasDependentsError):
        adj.remove_step("s1")


def test_adjuster_remove_step_unknown() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(StepNotFoundError):
        adj.remove_step("missing")


def test_adjuster_reorder_steps() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    adj.reorder_steps("sp1", ["s2", "s1"])
    assert [s.id for s in plan.sub_plans[0].steps] == ["s2", "s1"]


def test_adjuster_reorder_steps_rejects_partial() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(ValueError):
        adj.reorder_steps("sp1", ["s1"])


def test_adjuster_reorder_steps_rejects_extra() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(ValueError):
        adj.reorder_steps("sp1", ["s1", "s2", "extra"])


def test_adjuster_set_step_status() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    adj.set_step_status("s1", StepStatus.SKIPPED, note="manual override")
    assert plan.sub_plans[0].steps[0].status is StepStatus.SKIPPED
    assert plan.sub_plans[0].steps[0].notes == "manual override"


def test_adjuster_add_sub_plan() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    new_sp = SubPlan(id="sp3", title="C", description="d")
    adj.add_sub_plan(new_sp)
    assert plan.sub_plans[-1] is new_sp


def test_adjuster_add_sub_plan_duplicate_rejected() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    dup = SubPlan(id="sp1", title="X", description="d")
    with pytest.raises(DuplicateSubPlanIdError):
        adj.add_sub_plan(dup)


def test_adjuster_replace_sub_plan() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    new_sp = SubPlan(id="sp1", title="A'", description="d'")
    adj.replace_sub_plan("sp1", new_sp)
    assert plan.sub_plans[0] is new_sp
    assert plan.sub_plans[0].title == "A'"


def test_adjuster_replace_sub_plan_unknown() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(SubPlanNotFoundError):
        adj.replace_sub_plan("missing", SubPlan(id="missing", title="X", description="d"))


def test_adjuster_remove_sub_plan() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    adj.remove_sub_plan("sp2")
    assert all(sp.id != "sp2" for sp in plan.sub_plans)


def test_adjuster_remove_sub_plan_unknown() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(SubPlanNotFoundError):
        adj.remove_sub_plan("missing")


def test_adjuster_set_status() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    adj.set_status(PlanStatus.ACTIVE)
    assert plan.status is PlanStatus.ACTIVE


def test_adjuster_set_status_rejects_non_enum() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(TypeError):
        adj.set_status("active")  # type: ignore[arg-type]


def test_adjuster_set_metadata() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    adj.set_metadata("priority", "high")
    assert plan.metadata["priority"] == "high"


def test_adjuster_set_metadata_rejects_empty_key() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    with pytest.raises(ValueError):
        adj.set_metadata("", "x")


def test_adjuster_add_step_at_negative_position() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    new_step = Step(id="s4", title="T4", description="D4")
    adj.add_step("sp1", new_step, position=-1)
    # -1 inserts before the last step.
    assert plan.sub_plans[0].steps[-2] is new_step


def test_adjuster_replace_sub_plan_with_rename() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    new_sp = SubPlan(id="sp2-renamed", title="B'", description="d")
    adj.replace_sub_plan("sp2", new_sp)
    assert any(sp.id == "sp2-renamed" for sp in plan.sub_plans)
    assert not any(sp.id == "sp2" for sp in plan.sub_plans)


def test_adjuster_replace_sub_plan_rename_to_existing_rejected() -> None:
    plan = _plan()
    adj = PlanAdjuster(plan)
    bad = SubPlan(id="sp1", title="A", description="d")
    with pytest.raises(DuplicateSubPlanIdError):
        adj.replace_sub_plan("sp2", bad)


def test_adjuster_requires_plan_instance() -> None:
    with pytest.raises(TypeError):
        PlanAdjuster({"id": "p1", "title": "x", "goal": "x"})  # type: ignore[arg-type]


def test_adjuster_concurrent_adds() -> None:
    """10 threads each add a unique step to the same sub-plan."""
    import threading

    plan = _plan()
    adj = PlanAdjuster(plan)
    n = 10
    barrier = threading.Barrier(n)
    errors: list[Exception] = []

    def fire(i: int) -> None:
        try:
            barrier.wait()
            adj.add_step("sp2", Step(id=f"new{i}", title="T", description="D"))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    sp2 = plan.find_sub_plan("sp2")
    assert sp2 is not None
    assert len(sp2.steps) == 1 + n  # original s3 + n new steps
