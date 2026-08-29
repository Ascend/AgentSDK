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

"""Plan / SubPlan / Step / AcceptanceCriteria validation tests."""
# pylint: disable=redefined-builtin

from __future__ import annotations

import pytest

from src.services.ultraplan import (
    AcceptanceCriteria,
    CheckKind,
    Plan,
    PlanStatus,
    Step,
    StepKind,
    StepStatus,
    SubPlan,
)


def _step(id: str = "s1", **overrides) -> Step:
    defaults: dict = {
        "id": id,
        "title": "Step title",
        "description": "Step description",
        "kind": StepKind.OTHER,
    }
    defaults.update(overrides)
    return Step(**defaults)


def test_plan_defaults() -> None:
    plan = Plan(id="p1", title="My plan", goal="Achieve X")
    assert plan.id == "p1"
    assert plan.title == "My plan"
    assert plan.goal == "Achieve X"
    assert plan.sub_plans == []
    assert plan.status is PlanStatus.DRAFT
    assert plan.metadata == {}


def test_plan_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        Plan(id="", title="x", goal="x")


def test_plan_rejects_bad_id_chars() -> None:
    with pytest.raises(ValueError):
        Plan(id="has spaces", title="x", goal="x")
    with pytest.raises(ValueError):
        Plan(id="a" * 65, title="x", goal="x")


def test_plan_rejects_empty_title() -> None:
    with pytest.raises(ValueError):
        Plan(id="p1", title="", goal="x")


def test_plan_rejects_empty_goal() -> None:
    with pytest.raises(ValueError):
        Plan(id="p1", title="x", goal="")


def test_plan_rejects_oversized_goal() -> None:
    with pytest.raises(ValueError):
        Plan(id="p1", title="x", goal="x" * 30_001)


def test_plan_rejects_non_dict_metadata() -> None:
    with pytest.raises(TypeError):
        Plan(id="p1", title="x", goal="x", metadata=[1, 2])  # type: ignore[arg-type]


def test_plan_rejects_duplicate_sub_plan_id() -> None:
    sp1 = SubPlan(id="sp1", title="Sub 1", description="d")
    sp2 = SubPlan(id="sp1", title="Sub 2", description="d")
    with pytest.raises(ValueError):
        Plan(id="p1", title="x", goal="x", sub_plans=[sp1, sp2])


def test_plan_rejects_cross_sub_plan_step_ids() -> None:
    sp1 = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[_step("shared")],
    )
    sp2 = SubPlan(
        id="sp2",
        title="B",
        description="d",
        steps=[_step("shared")],
    )
    with pytest.raises(ValueError):
        Plan(id="p1", title="x", goal="x", sub_plans=[sp1, sp2])


def test_plan_rejects_cross_sub_plan_dependency() -> None:
    sp1 = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[_step("s1"), _step("s2", depends_on=["s1"])],
    )
    sp2 = SubPlan(
        id="sp2",
        title="B",
        description="d",
        steps=[_step("s3", depends_on=["s1"])],  # s1 is in sp1, not sp2
    )
    with pytest.raises(ValueError):
        Plan(id="p1", title="x", goal="x", sub_plans=[sp1, sp2])


def test_plan_accepts_in_sub_plan_dependency() -> None:
    sp = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[_step("s1"), _step("s2", depends_on=["s1"])],
    )
    plan = Plan(id="p1", title="x", goal="x", sub_plans=[sp])
    assert plan.sub_plans[0].steps[1].depends_on == ["s1"]


def test_plan_round_trip() -> None:
    sp = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[_step("s1"), _step("s2", depends_on=["s1"])],
    )
    plan = Plan(id="p1", title="My plan", goal="Goal", sub_plans=[sp])
    data = plan.to_dict()
    assert Plan.from_dict(data) == plan


def test_sub_plan_rejects_duplicate_step_id() -> None:
    with pytest.raises(ValueError):
        SubPlan(
            id="sp1",
            title="A",
            description="d",
            steps=[_step("s1"), _step("s1", title="t2", description="d2")],
        )


def test_step_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        _step(id="")


def test_step_rejects_empty_title() -> None:
    with pytest.raises(ValueError):
        _step(id="s1", title="")


def test_step_rejects_empty_description() -> None:
    with pytest.raises(ValueError):
        _step(id="s1", description="")


def test_step_rejects_duplicate_depends_on() -> None:
    with pytest.raises(ValueError):
        _step(id="s1", depends_on=["s2", "s2"])


def test_step_rejects_non_list_depends_on() -> None:
    with pytest.raises(TypeError):
        _step(id="s1", depends_on="s2")  # type: ignore[arg-type]


def test_step_is_terminal() -> None:
    s = _step()
    assert s.is_terminal() is False
    s.status = StepStatus.COMPLETED
    assert s.is_terminal() is True
    s.status = StepStatus.SKIPPED
    assert s.is_terminal() is True
    s.status = StepStatus.FAILED
    assert s.is_terminal() is True


def test_step_round_trip() -> None:
    s = _step(
        id="s1",
        title="T",
        description="D",
        kind=StepKind.IMPLEMENT,
        status=StepStatus.IN_PROGRESS,
        depends_on=["s0"],
        notes="hello",
    )
    assert Step.from_dict(s.to_dict()) == s


def test_acceptance_criteria_rejects_empty_target() -> None:
    with pytest.raises(ValueError):
        AcceptanceCriteria(
            id="c1",
            description="d",
            kind=CheckKind.FILE_EXISTS,
            target="",
        )


def test_acceptance_criteria_rejects_non_dict_args() -> None:
    with pytest.raises(TypeError):
        AcceptanceCriteria(
            id="c1",
            description="d",
            kind=CheckKind.FILE_EXISTS,
            target="/tmp/x",
            args=[1, 2],  # type: ignore[arg-type]
        )


def test_acceptance_criteria_rejects_non_bool_required() -> None:
    with pytest.raises(TypeError):
        AcceptanceCriteria(
            id="c1",
            description="d",
            kind=CheckKind.FILE_EXISTS,
            target="/tmp/x",
            required="yes",  # type: ignore[arg-type]
        )


def test_acceptance_criteria_round_trip() -> None:
    c = AcceptanceCriteria(
        id="c1",
        description="check file",
        kind=CheckKind.FILE_CONTAINS,
        target="/tmp/x",
        args={"substring": "ok"},
        required=False,
    )
    assert AcceptanceCriteria.from_dict(c.to_dict()) == c


def test_plan_find_sub_plan() -> None:
    sp = SubPlan(id="sp1", title="A", description="d")
    plan = Plan(id="p1", title="x", goal="x", sub_plans=[sp])
    assert plan.find_sub_plan("sp1") is sp
    assert plan.find_sub_plan("missing") is None


def test_plan_find_step_returns_sub_plan_and_step() -> None:
    s = _step("s1")
    sp = SubPlan(id="sp1", title="A", description="d", steps=[s])
    plan = Plan(id="p1", title="x", goal="x", sub_plans=[sp])
    result = plan.find_step("s1")
    assert result is not None
    found_sp, found_step = result
    assert found_sp is sp
    assert found_step is s
    assert plan.find_step("missing") is None


def test_plan_all_terminal_when_empty() -> None:
    plan = Plan(id="p1", title="x", goal="x")
    assert plan.all_terminal() is True
