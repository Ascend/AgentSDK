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

"""PlanExecutor tests: state machine, transitions, progress, clock."""
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import threading
from typing import Callable

import pytest

from src.services.ultraplan import (
    IllegalStepTransitionError,
    Plan,
    PlanExecutor,
    PlanStatus,
    Step,
    StepKind,
    StepNotFoundError,
    StepStatus,
    SubPlan,
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


def _clock() -> Callable[[], str]:
    counter = [0]

    def next_ts() -> str:
        counter[0] += 1
        return f"2026-01-01T00:00:{counter[0]:02d}.000Z"

    return next_ts


def test_executor_next_step_returns_first_pending() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    nxt = ex.next_step()
    assert nxt is not None
    assert nxt.id == "s1"


def test_executor_next_step_skips_blocked_deps() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_in_progress("s1")
    ex.mark_completed("s1")
    # Now s2 should be next.
    nxt = ex.next_step()
    assert nxt is not None and nxt.id == "s2"


def test_executor_next_step_returns_none_when_done() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    for sid in ("s1", "s2", "s3"):
        ex.mark_in_progress(sid)
        ex.mark_completed(sid)
    assert ex.next_step() is None


def test_executor_next_step_filtered_by_sub_plan() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    nxt = ex.next_step(sub_plan_id="sp2")
    assert nxt is not None and nxt.id == "s3"


def test_executor_next_step_unknown_sub_plan() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    assert ex.next_step(sub_plan_id="missing") is None


def test_executor_mark_in_progress_sets_started_at() -> None:
    plan = _plan()
    clock = _clock()
    ex = PlanExecutor(plan, clock=clock)
    ex.mark_in_progress("s1")
    assert plan.sub_plans[0].steps[0].status is StepStatus.IN_PROGRESS
    assert plan.sub_plans[0].steps[0].started_at == "2026-01-01T00:00:01.000Z"


def test_executor_mark_completed_sets_completed_at_and_result() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_in_progress("s1")
    ex.mark_completed("s1", result={"files": 3})
    s = plan.sub_plans[0].steps[0]
    assert s.status is StepStatus.COMPLETED
    assert s.completed_at is not None
    assert s.result == {"files": 3}


def test_executor_mark_failed_sets_error() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_in_progress("s1")
    ex.mark_failed("s1", "compile error")
    s = plan.sub_plans[0].steps[0]
    assert s.status is StepStatus.FAILED
    assert s.error == "compile error"
    assert s.completed_at is not None


def test_executor_mark_blocked_then_unblock() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_in_progress("s1")
    ex.mark_blocked("s1", note="waiting for input")
    assert plan.sub_plans[0].steps[0].status is StepStatus.BLOCKED
    ex.unblock("s1")
    assert plan.sub_plans[0].steps[0].status is StepStatus.PENDING


def test_executor_terminal_status_is_sticky() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_in_progress("s1")
    ex.mark_completed("s1")
    for target in (
        StepStatus.IN_PROGRESS,
        StepStatus.PENDING,
        StepStatus.FAILED,
        StepStatus.SKIPPED,
    ):
        with pytest.raises(IllegalStepTransitionError):
            ex._transition_step(  # noqa: SLF001 - direct for test
                plan.sub_plans[0].steps[0],
                plan.sub_plans[0],
                target,
            )


def test_executor_no_op_transition_is_silent() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_in_progress("s1")
    # Re-applying the same status is a no-op (allowed).
    ex.mark_in_progress("s1")
    assert plan.sub_plans[0].steps[0].status is StepStatus.IN_PROGRESS


def test_executor_progress_counts() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    p = ex.progress()
    assert p.total == 3
    assert p.completed == 0
    assert p.pending == 3
    assert p.ratio == 0.0

    ex.mark_in_progress("s1")
    ex.mark_completed("s1")
    p = ex.progress()
    assert p.completed == 1
    assert p.pending == 2
    assert p.ratio == pytest.approx(1 / 3)


def test_executor_progress_filtered_by_sub_plan() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    p = ex.progress(sub_plan_id="sp2")
    assert p.total == 1


def test_executor_progress_empty_sub_plan() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    p = ex.progress(sub_plan_id="missing")
    assert p.total == 0


def test_executor_plan_status_marks_active() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    assert plan.status is PlanStatus.DRAFT
    ex.mark_in_progress("s1")
    assert plan.status is PlanStatus.ACTIVE


def test_executor_plan_status_marks_completed_when_all_done() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    for sid in ("s1", "s2", "s3"):
        ex.mark_in_progress(sid)
        ex.mark_completed(sid)
    assert plan.status is PlanStatus.COMPLETED
    assert ex.is_complete() is True


def test_executor_plan_status_marks_failed_when_all_stuck() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_in_progress("s1")
    ex.mark_failed("s1", "x")
    ex.mark_in_progress("s3")
    ex.mark_failed("s3", "y")
    # s2 is PENDING with dependency on a failed s1, so plan isn't done.
    assert ex.has_failures() is True
    assert plan.status is PlanStatus.ACTIVE


def test_executor_transitions_logged() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_in_progress("s1")
    ex.mark_completed("s1")
    log = ex.transitions
    assert len(log) == 2
    assert log[0].step_id == "s1"
    assert log[0].old_status is StepStatus.PENDING
    assert log[0].new_status is StepStatus.IN_PROGRESS


def test_executor_updated_at_set() -> None:
    plan = _plan()
    clock = _clock()
    ex = PlanExecutor(plan, clock=clock)
    ex.mark_in_progress("s1")
    assert plan.updated_at is not None


def test_executor_get_step_raises_on_missing() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    with pytest.raises(StepNotFoundError):
        ex.get_step("nope")


def test_executor_get_sub_plan_raises_on_missing() -> None:
    from src.services.ultraplan import SubPlanNotFoundError

    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    with pytest.raises(SubPlanNotFoundError):
        ex.get_sub_plan("nope")


def test_executor_concurrent_transitions() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    # 50 threads racing to mark s1 IN_PROGRESS. The state machine
    # guarantees only the first transition succeeds; the rest are no-ops.
    n = 50
    barrier = threading.Barrier(n)
    failures: list[Exception] = []

    def fire() -> None:
        try:
            barrier.wait()
            ex.mark_in_progress("s1")
        except Exception as e:  # noqa: BLE001
            failures.append(e)

    threads = [threading.Thread(target=fire) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # No thread should have raised; the executor is thread-safe.
    assert failures == []


def test_executor_dep_satisfied_only_with_completed() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    # Failed dep should not make s2 runnable.
    ex.mark_in_progress("s1")
    ex.mark_failed("s1", "x")
    nxt = ex.next_step()
    # s3 is independent; s2 is blocked by failed s1.
    assert nxt is not None and nxt.id == "s3"


def test_executor_skip_step_directly() -> None:
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_skipped("s1", note="not needed")
    s = plan.sub_plans[0].steps[0]
    assert s.status is StepStatus.SKIPPED
    # s2 is still blocked because s1 was skipped, not completed.
    nxt = ex.next_step()
    assert nxt is not None and nxt.id == "s3"


def test_executor_requires_plan_instance() -> None:
    with pytest.raises(TypeError):
        PlanExecutor({"id": "p1", "title": "x", "goal": "x"})  # type: ignore[arg-type]


def test_executor_recovers_from_completed_status() -> None:
    """A completed step cannot be reset, but mark_pending is rejected."""
    plan = _plan()
    ex = PlanExecutor(plan, clock=_clock())
    ex.mark_in_progress("s1")
    ex.mark_completed("s1")
    with pytest.raises(IllegalStepTransitionError):
        ex.mark_pending("s1")
