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

"""Plan executor: step state machine and progress tracking.

The executor owns the in-memory :class:`Plan` and advances it through
the :class:`StepStatus` state machine. It records every transition in
an audit log so the verifier and the CLI can later reconstruct the
history of a plan's execution. The clock is injected so tests can
produce deterministic timestamps.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .exceptions import (
    IllegalStepTransitionError,
    StepNotFoundError,
    SubPlanNotFoundError,
)
from .models import Plan, PlanStatus, Step, StepStatus, SubPlan

# Allowed forward transitions in the step state machine. ``BLOCKED`` can
# be set from any non-terminal state and from ``BLOCKED`` itself we
# allow moving back to ``PENDING`` (unblock) or to ``SKIPPED``. Once a
# step is in a terminal state (``COMPLETED``, ``FAILED``, ``SKIPPED``)
# it cannot leave.
_ALLOWED_STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset({StepStatus.IN_PROGRESS, StepStatus.SKIPPED, StepStatus.BLOCKED}),
    StepStatus.IN_PROGRESS: frozenset(
        {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.BLOCKED, StepStatus.SKIPPED}
    ),
    StepStatus.BLOCKED: frozenset({StepStatus.PENDING, StepStatus.SKIPPED, StepStatus.IN_PROGRESS}),
    StepStatus.COMPLETED: frozenset(),
    StepStatus.FAILED: frozenset(),
    StepStatus.SKIPPED: frozenset(),
}


@dataclass
class StepTransition:
    step_id: str
    sub_plan_id: str
    old_status: StepStatus
    new_status: StepStatus
    timestamp: str
    note: str | None = None


@dataclass
class Progress:
    completed: int
    failed: int
    skipped: int
    pending: int
    in_progress: int
    blocked: int
    total: int

    @property
    def done(self) -> int:
        return self.completed + self.failed + self.skipped

    @property
    def ratio(self) -> float:
        if self.total == 0:
            return 1.0
        return self.done / self.total


ClockFn = Callable[[], str]
TransitionHook = Callable[[str, StepTransition], None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _is_terminal(status: StepStatus) -> bool:
    return status in (
        StepStatus.COMPLETED,
        StepStatus.FAILED,
        StepStatus.SKIPPED,
    )


class PlanExecutor:
    """Advance a :class:`Plan` through its step state machine."""

    def __init__(
        self,
        plan: Plan,
        *,
        clock: ClockFn | None = None,
        transition_hooks: list[TransitionHook] | None = None,
    ) -> None:
        if not isinstance(plan, Plan):
            raise TypeError("PlanExecutor requires a Plan instance")
        self._plan = plan
        self._lock = threading.RLock()
        self._clock: ClockFn = clock or _utc_now_iso
        self._transitions: list[StepTransition] = []
        self._transition_hooks: list[TransitionHook] = list(transition_hooks or [])

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def transitions(self) -> list[StepTransition]:
        with self._lock:
            return list(self._transitions)

    def add_transition_hook(self, hook: TransitionHook) -> None:
        if not callable(hook):
            raise TypeError("transition hook must be callable")
        with self._lock:
            self._transition_hooks.append(hook)

    def get_step(self, step_id: str) -> tuple[SubPlan, Step]:
        result = self._plan.find_step(step_id)
        if result is None:
            raise StepNotFoundError(f"no step {step_id!r} in plan {self._plan.id!r}")
        return result

    def get_sub_plan(self, sub_plan_id: str) -> SubPlan:
        sp = self._plan.find_sub_plan(sub_plan_id)
        if sp is None:
            raise SubPlanNotFoundError(f"no sub_plan {sub_plan_id!r} in plan {self._plan.id!r}")
        return sp

    # ------------------------------------------------------------------
    # Step selection
    # ------------------------------------------------------------------

    def next_step(self, *, sub_plan_id: str | None = None) -> Step | None:
        """Return the next runnable step, or ``None`` if the plan is done.

        A step is "runnable" when its status is ``PENDING`` and all
        steps it depends on are ``COMPLETED`` (failed/skipped deps
        block the dependent step — the caller must explicitly unblock
        via :meth:`mark_blocked` / :meth:`mark_pending`).

        Steps are visited in ``sub_plans`` order, then by index within
        the sub-plan, so the executor is deterministic.
        """
        with self._lock:
            sub_plans = self._plan.sub_plans
            if sub_plan_id is not None:
                match = self._plan.find_sub_plan(sub_plan_id)
                if match is None:
                    return None
                sub_plans = [match]
            for sp in sub_plans:
                for step in sp.steps:
                    if step.status != StepStatus.PENDING:
                        continue
                    if self._deps_satisfied(step, sp):
                        return step
            return None

    def _deps_satisfied(self, step: Step, owner: SubPlan) -> bool:
        owner_steps = {s.id: s for s in owner.steps}
        for dep in step.depends_on:
            target = owner_steps.get(dep)
            if target is None:
                # Cross-sub-plan or missing deps are rejected at
                # construction time; treat as not satisfied defensively.
                return False
            if target.status != StepStatus.COMPLETED:
                return False
        return True

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _transition_step(
        self,
        step: Step,
        sub_plan: SubPlan,
        new_status: StepStatus,
        *,
        note: str | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> Step:
        with self._lock:
            old = step.status
            if new_status == old:
                return step
            allowed = _ALLOWED_STEP_TRANSITIONS[old]
            if new_status not in allowed:
                raise IllegalStepTransitionError(
                    f"step {step.id!r}: cannot move from {old.value} to {new_status.value}"
                )
            now = self._clock()
            step.status = new_status
            if new_status == StepStatus.IN_PROGRESS and step.started_at is None:
                step.started_at = now
            if _is_terminal(new_status):
                step.completed_at = now
            if new_status == StepStatus.FAILED:
                step.error = error or step.error
            if new_status == StepStatus.COMPLETED:
                step.result = result if result is not None else step.result
            if note is not None:
                step.notes = note
            transition = StepTransition(
                step_id=step.id,
                sub_plan_id=sub_plan.id,
                old_status=old,
                new_status=new_status,
                timestamp=now,
                note=note,
            )
            self._transitions.append(transition)
            self._refresh_plan_status()
            self._plan.updated_at = now
            for hook in list(self._transition_hooks):
                hook(self._plan.id, transition)
            return step

    def mark_in_progress(self, step_id: str) -> Step:
        sp, step = self.get_step(step_id)
        return self._transition_step(step, sp, StepStatus.IN_PROGRESS)

    def mark_completed(
        self,
        step_id: str,
        *,
        result: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> Step:
        sp, step = self.get_step(step_id)
        return self._transition_step(step, sp, StepStatus.COMPLETED, result=result, note=note)

    def mark_failed(self, step_id: str, error: str, *, note: str | None = None) -> Step:
        sp, step = self.get_step(step_id)
        return self._transition_step(step, sp, StepStatus.FAILED, error=error, note=note)

    def mark_skipped(self, step_id: str, *, note: str | None = None) -> Step:
        sp, step = self.get_step(step_id)
        return self._transition_step(step, sp, StepStatus.SKIPPED, note=note)

    def mark_blocked(self, step_id: str, *, note: str | None = None) -> Step:
        sp, step = self.get_step(step_id)
        return self._transition_step(step, sp, StepStatus.BLOCKED, note=note)

    def unblock(self, step_id: str) -> Step:
        """Move a blocked step back to ``PENDING`` so it can run again."""
        sp, step = self.get_step(step_id)
        return self._transition_step(step, sp, StepStatus.PENDING)

    def mark_pending(self, step_id: str) -> Step:
        """Reset a step to ``PENDING`` (only allowed from ``BLOCKED``)."""
        sp, step = self.get_step(step_id)
        return self._transition_step(step, sp, StepStatus.PENDING)

    # ------------------------------------------------------------------
    # Plan-level progress and status
    # ------------------------------------------------------------------

    def _refresh_plan_status(self) -> None:
        steps = self._plan.all_steps()
        if not steps:
            return
        if all(s.status == StepStatus.COMPLETED for s in steps):
            self._plan.status = PlanStatus.COMPLETED
            return
        if any(s.status == StepStatus.IN_PROGRESS for s in steps):
            self._plan.status = PlanStatus.ACTIVE
            return
        if all(s.status in (StepStatus.SKIPPED, StepStatus.COMPLETED) for s in steps):
            self._plan.status = PlanStatus.COMPLETED
            return
        if any(s.status == StepStatus.FAILED for s in steps):
            # Mark plan as FAILED only if all non-failed steps are
            # terminal (no more work can rescue it).
            non_failed = [s for s in steps if s.status != StepStatus.FAILED]
            if all(s.is_terminal() for s in non_failed):
                self._plan.status = PlanStatus.FAILED
                return
        # Default: any non-terminal step means the plan is active.
        if any(not s.is_terminal() for s in steps):
            self._plan.status = PlanStatus.ACTIVE
        else:
            self._plan.status = PlanStatus.COMPLETED

    def progress(self, sub_plan_id: str | None = None) -> Progress:
        with self._lock:
            steps: list[Step]
            if sub_plan_id is not None:
                sp = self._plan.find_sub_plan(sub_plan_id)
                if sp is None:
                    return Progress(0, 0, 0, 0, 0, 0, 0)
                steps = list(sp.steps)
            else:
                steps = self._plan.all_steps()
            completed = sum(1 for s in steps if s.status == StepStatus.COMPLETED)
            failed = sum(1 for s in steps if s.status == StepStatus.FAILED)
            skipped = sum(1 for s in steps if s.status == StepStatus.SKIPPED)
            pending = sum(1 for s in steps if s.status == StepStatus.PENDING)
            in_progress = sum(1 for s in steps if s.status == StepStatus.IN_PROGRESS)
            blocked = sum(1 for s in steps if s.status == StepStatus.BLOCKED)
            return Progress(
                completed=completed,
                failed=failed,
                skipped=skipped,
                pending=pending,
                in_progress=in_progress,
                blocked=blocked,
                total=len(steps),
            )

    def is_complete(self) -> bool:
        with self._lock:
            return self._plan.all_terminal()

    def has_failures(self) -> bool:
        with self._lock:
            return any(s.status == StepStatus.FAILED for s in self._plan.all_steps())


@dataclass
class ExecutorOptions:
    auto_advance: bool = False
    recorder: Any = field(default=None, repr=False)
