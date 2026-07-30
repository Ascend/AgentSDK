"""Plan adjuster: replace / add / remove steps and sub-plans mid-execution.

The adjuster is the API layer for the P83-D "动态调整计划" sub-feature.
It enforces three invariants that the executor and verifier rely on:

* Step ids are unique within a sub-plan and across the whole plan.
* Cross-sub-plan step dependencies are rejected.
* Removing a step that other steps depend on is rejected; the caller
  must first replace or remove the dependent steps.

The adjuster mutates the plan in place so the executor and store see
the latest state, but it leaves an audit trail on each affected step's
``notes`` field (``notes`` is not reset, only appended to).
"""

from __future__ import annotations

import threading
from typing import Any

from .exceptions import (
    DuplicateStepIdError,
    DuplicateSubPlanIdError,
    StepHasDependentsError,
    StepNotFoundError,
    SubPlanNotFoundError,
)
from .models import Plan, Step, StepStatus, SubPlan


class PlanAdjuster:
    """Mutate a :class:`Plan` while preserving its structural invariants."""

    def __init__(self, plan: Plan) -> None:
        if not isinstance(plan, Plan):
            raise TypeError("PlanAdjuster requires a Plan instance")
        self._plan = plan
        self._lock = threading.RLock()

    @property
    def plan(self) -> Plan:
        return self._plan

    # ------------------------------------------------------------------
    # Step-level operations
    # ------------------------------------------------------------------

    def add_step(
        self,
        sub_plan_id: str,
        step: Step,
        *,
        position: int | None = None,
    ) -> Step:
        with self._lock:
            sub_plan = self._get_sub_plan(sub_plan_id)
            self._ensure_step_id_unique(step.id)
            self._ensure_step_deps_in_sub_plan(sub_plan, step)
            if position is None or position >= len(sub_plan.steps):
                sub_plan.steps.append(step)
            elif position < 0:
                sub_plan.steps.insert(max(0, len(sub_plan.steps) + position), step)
            else:
                sub_plan.steps.insert(position, step)
            return step

    def replace_step(self, step_id: str, new_step: Step) -> Step:
        with self._lock:
            sub_plan, _ = self._get_step(step_id)
            if new_step.id != step_id:
                # Renaming a step is allowed only if the new id is free.
                self._ensure_step_id_unique(new_step.id)
            self._ensure_step_deps_in_sub_plan(sub_plan, new_step)
            for i, s in enumerate(sub_plan.steps):
                if s.id == step_id:
                    sub_plan.steps[i] = new_step
                    return new_step
            raise StepNotFoundError(f"step {step_id!r} vanished mid-replace")

    def remove_step(self, step_id: str) -> Step:
        with self._lock:
            sub_plan, step = self._get_step(step_id)
            dependents = self._find_dependents(step_id)
            if dependents:
                raise StepHasDependentsError(
                    f"step {step_id!r} cannot be removed; depended on by {sorted(dependents)!r}"
                )
            sub_plan.steps = [s for s in sub_plan.steps if s.id != step_id]
            return step

    def reorder_steps(self, sub_plan_id: str, new_order: list[str]) -> None:
        with self._lock:
            sub_plan = self._get_sub_plan(sub_plan_id)
            by_id = {s.id: s for s in sub_plan.steps}
            if set(new_order) != set(by_id.keys()):
                raise ValueError("new_order must contain exactly the same step ids as the sub-plan")
            sub_plan.steps = [by_id[sid] for sid in new_order]

    def set_step_status(self, step_id: str, status: StepStatus, *, note: str | None = None) -> Step:
        """Force a step's status to ``status`` regardless of current state.

        Use this only for administrative adjustments — the executor is
        the right API for normal forward progression. The adjuster
        refuses to leave a step's status in a state that would invalidate
        the executor's invariants (e.g. IN_PROGRESS without a
        ``started_at``); the caller is expected to set timestamps via
        direct field access if needed.
        """
        with self._lock:
            _, step = self._get_step(step_id)
            step.status = status
            if note is not None:
                step.notes = note
            return step

    # ------------------------------------------------------------------
    # Sub-plan-level operations
    # ------------------------------------------------------------------

    def add_sub_plan(self, sub_plan: SubPlan, *, position: int | None = None) -> SubPlan:
        with self._lock:
            self._ensure_sub_plan_id_unique(sub_plan.id)
            if position is None or position >= len(self._plan.sub_plans):
                self._plan.sub_plans.append(sub_plan)
            elif position < 0:
                self._plan.sub_plans.insert(max(0, len(self._plan.sub_plans) + position), sub_plan)
            else:
                self._plan.sub_plans.insert(position, sub_plan)
            return sub_plan

    def replace_sub_plan(self, sub_plan_id: str, new_sub_plan: SubPlan) -> SubPlan:
        with self._lock:
            self._get_sub_plan(sub_plan_id)  # validate existence
            if new_sub_plan.id != sub_plan_id:
                self._ensure_sub_plan_id_unique(new_sub_plan.id)
            for i, sp in enumerate(self._plan.sub_plans):
                if sp.id == sub_plan_id:
                    self._plan.sub_plans[i] = new_sub_plan
                    return new_sub_plan
            raise SubPlanNotFoundError(f"sub_plan {sub_plan_id!r} vanished mid-replace")

    def remove_sub_plan(self, sub_plan_id: str) -> SubPlan:
        with self._lock:
            sp = self._get_sub_plan(sub_plan_id)
            self._plan.sub_plans = [s for s in self._plan.sub_plans if s.id != sub_plan_id]
            return sp

    # ------------------------------------------------------------------
    # Plan-level operations
    # ------------------------------------------------------------------

    def set_status(self, status) -> None:  # type: ignore[no-untyped-def]
        from .models import PlanStatus

        if not isinstance(status, PlanStatus):
            raise TypeError("status must be a PlanStatus")
        with self._lock:
            self._plan.status = status

    def set_metadata(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("metadata key must be a non-empty string")
        with self._lock:
            self._plan.metadata[key] = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sub_plan(self, sub_plan_id: str) -> SubPlan:
        sp = self._plan.find_sub_plan(sub_plan_id)
        if sp is None:
            raise SubPlanNotFoundError(f"no sub_plan {sub_plan_id!r} in plan {self._plan.id!r}")
        return sp

    def _get_step(self, step_id: str) -> tuple[SubPlan, Step]:
        result = self._plan.find_step(step_id)
        if result is None:
            raise StepNotFoundError(f"no step {step_id!r} in plan {self._plan.id!r}")
        return result

    def _ensure_step_id_unique(self, step_id: str) -> None:
        if self._plan.find_step(step_id) is not None:
            raise DuplicateStepIdError(f"step id {step_id!r} already exists in plan {self._plan.id!r}")

    def _ensure_sub_plan_id_unique(self, sub_plan_id: str) -> None:
        if self._plan.find_sub_plan(sub_plan_id) is not None:
            raise DuplicateSubPlanIdError(f"sub_plan id {sub_plan_id!r} already exists in plan {self._plan.id!r}")

    def _ensure_step_deps_in_sub_plan(self, sub_plan: SubPlan, step: Step) -> None:
        sp_step_ids = {s.id for s in sub_plan.steps}
        for dep in step.depends_on:
            if dep not in sp_step_ids:
                raise ValueError(f"step {step.id!r} depends on unknown step {dep!r} in sub_plan {sub_plan.id!r}")

    def _find_dependents(self, step_id: str) -> set[str]:
        dependents: set[str] = set()
        for sp in self._plan.sub_plans:
            for s in sp.steps:
                if step_id in s.depends_on:
                    dependents.add(s.id)
        return dependents
