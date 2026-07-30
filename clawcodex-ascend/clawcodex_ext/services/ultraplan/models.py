"""Ultraplan data models.

The package models a hierarchical plan (Plan → SubPlan → Step) with
acceptance criteria attached to each step. The structure intentionally
mirrors the CCB ``/ultraplan`` template: a plan is a goal expressed as a
list of sub-plans, each sub-plan breaks the goal into ordered steps, and
each step carries an explicit acceptance list that the verifier can run
automatically once the step completes.

Persistence is JSON; every dataclass provides ``to_dict`` / ``from_dict``
helpers and ``__post_init__`` validation so malformed plans fail loudly
instead of silently dropping fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_TITLE_MAX = 200
_TEXT_MAX = 30_000


class PlanStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    FAILED = "failed"


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class StepKind(str, Enum):
    RESEARCH = "research"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    REVIEW = "review"
    OTHER = "other"


class CheckKind(str, Enum):
    FILE_EXISTS = "file_exists"
    FILE_CONTAINS = "file_contains"
    PYTHON_PREDICATE = "python_predicate"
    SHELL_COMMAND = "shell_command"
    CUSTOM = "custom"


_TERMINAL_STEP_STATUSES = frozenset({StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED})


def _validate_id(value: str, *, kind: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} id must be a non-empty string")
    if not _ID_RE.match(value):
        raise ValueError(f"{kind} id must match [A-Za-z0-9._-]{{1,64}}; got: {value!r}")
    return value


def _validate_text(value: str, *, name: str, cap: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > cap:
        raise ValueError(f"{name} exceeds {cap} character safety cap")
    return value


@dataclass
class AcceptanceCriteria:
    id: str
    description: str
    kind: CheckKind
    target: str
    args: dict[str, Any] = field(default_factory=dict)
    required: bool = True

    def __post_init__(self) -> None:
        _validate_id(self.id, kind="AcceptanceCriteria")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("AcceptanceCriteria.description must be a non-empty string")
        if not isinstance(self.kind, CheckKind):
            raise TypeError("AcceptanceCriteria.kind must be a CheckKind")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("AcceptanceCriteria.target must be a non-empty string")
        if self.args is not None and not isinstance(self.args, dict):
            raise TypeError("AcceptanceCriteria.args must be a dict or None")
        if not isinstance(self.required, bool):
            raise TypeError("AcceptanceCriteria.required must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "kind": self.kind.value,
            "target": self.target,
            "args": dict(self.args),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcceptanceCriteria:
        if not isinstance(data, dict):
            raise TypeError("AcceptanceCriteria data must be a dict")
        return cls(
            id=str(data["id"]),
            description=str(data["description"]),
            kind=CheckKind(str(data["kind"])),
            target=str(data["target"]),
            args=dict(data.get("args") or {}),
            required=bool(data.get("required", True)),
        )


@dataclass
class Step:
    id: str
    title: str
    description: str
    kind: StepKind = StepKind.OTHER
    criteria: list[AcceptanceCriteria] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.id, kind="Step")
        _validate_text(self.title, name="Step.title", cap=_TITLE_MAX)
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("Step.description must be a non-empty string")
        if len(self.description) > _TEXT_MAX:
            raise ValueError("Step.description exceeds 30000 character safety cap")
        if not isinstance(self.kind, StepKind):
            raise TypeError("Step.kind must be a StepKind")
        if not isinstance(self.status, StepStatus):
            raise TypeError("Step.status must be a StepStatus")
        if self.criteria is not None and not isinstance(self.criteria, list):
            raise TypeError("Step.criteria must be a list or None")
        if self.depends_on is not None and not isinstance(self.depends_on, list):
            raise TypeError("Step.depends_on must be a list or None")
        if self.depends_on:
            seen: set[str] = set()
            for dep in self.depends_on:
                if not isinstance(dep, str) or not dep:
                    raise ValueError("Step.depends_on entries must be non-empty strings")
                if dep in seen:
                    raise ValueError(f"Step.depends_on contains duplicate {dep!r}")
                seen.add(dep)
        # Result must be a dict (or None) to round-trip cleanly.
        if self.result is not None and not isinstance(self.result, dict):
            raise TypeError("Step.result must be a dict or None")
        # Terminal status implies completed_at is set.
        if self.status in _TERMINAL_STEP_STATUSES and not self.completed_at:
            # Allow callers to construct terminal steps without a timestamp;
            # the executor will fill it in. We only validate type here.
            pass
        if self.started_at is not None and not isinstance(self.started_at, str):
            raise TypeError("Step.started_at must be a string or None")
        if self.completed_at is not None and not isinstance(self.completed_at, str):
            raise TypeError("Step.completed_at must be a string or None")
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("Step.notes must be a string or None")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("Step.error must be a string or None")

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STEP_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "kind": self.kind.value,
            "criteria": [c.to_dict() for c in self.criteria],
            "depends_on": list(self.depends_on),
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": dict(self.result) if self.result is not None else None,
            "error": self.error,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        if not isinstance(data, dict):
            raise TypeError("Step data must be a dict")
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            description=str(data["description"]),
            kind=StepKind(str(data.get("kind", StepKind.OTHER.value))),
            criteria=[AcceptanceCriteria.from_dict(c) for c in (data.get("criteria") or [])],
            depends_on=list(data.get("depends_on") or []),
            status=StepStatus(str(data.get("status", StepStatus.PENDING.value))),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error"),
            notes=data.get("notes"),
        )


@dataclass
class SubPlan:
    id: str
    title: str
    description: str
    steps: list[Step] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    notes: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.id, kind="SubPlan")
        _validate_text(self.title, name="SubPlan.title", cap=_TITLE_MAX)
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("SubPlan.description must be a non-empty string")
        if len(self.description) > _TEXT_MAX:
            raise ValueError("SubPlan.description exceeds 30000 character safety cap")
        if not isinstance(self.status, PlanStatus):
            raise TypeError("SubPlan.status must be a PlanStatus")
        if self.steps is not None and not isinstance(self.steps, list):
            raise TypeError("SubPlan.steps must be a list or None")
        # Validate step id uniqueness within the sub-plan.
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"SubPlan {self.id!r} has duplicate step id {step.id!r}")
            seen.add(step.id)
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("SubPlan.notes must be a string or None")

    def find_step(self, step_id: str) -> Step | None:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubPlan:
        if not isinstance(data, dict):
            raise TypeError("SubPlan data must be a dict")
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            description=str(data["description"]),
            steps=[Step.from_dict(s) for s in (data.get("steps") or [])],
            status=PlanStatus(str(data.get("status", PlanStatus.DRAFT.value))),
            notes=data.get("notes"),
        )


@dataclass
class Plan:
    id: str
    title: str
    goal: str
    sub_plans: list[SubPlan] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.id, kind="Plan")
        _validate_text(self.title, name="Plan.title", cap=_TITLE_MAX)
        if not isinstance(self.goal, str) or not self.goal:
            raise ValueError("Plan.goal must be a non-empty string")
        if len(self.goal) > _TEXT_MAX:
            raise ValueError("Plan.goal exceeds 30000 character safety cap")
        if not isinstance(self.status, PlanStatus):
            raise TypeError("Plan.status must be a PlanStatus")
        if self.sub_plans is not None and not isinstance(self.sub_plans, list):
            raise TypeError("Plan.sub_plans must be a list or None")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("Plan.metadata must be a dict or None")
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("Plan.notes must be a string or None")
        # Sub-plan id uniqueness.
        seen: set[str] = set()
        for sp in self.sub_plans:
            if sp.id in seen:
                raise ValueError(f"Plan {self.id!r} has duplicate sub_plan id {sp.id!r}")
            seen.add(sp.id)
        # Cross-sub-plan step id uniqueness.
        all_step_ids: set[str] = set()
        for sp in self.sub_plans:
            for step in sp.steps:
                if step.id in all_step_ids:
                    raise ValueError(f"Plan {self.id!r} has duplicate step id {step.id!r} across sub_plans")
                all_step_ids.add(step.id)
        # Validate depends_on references: a step may only depend on steps
        # in the same sub-plan; cross-sub-plan deps are not supported.
        for sp in self.sub_plans:
            sp_step_ids = {s.id for s in sp.steps}
            for step in sp.steps:
                for dep in step.depends_on:
                    if dep not in sp_step_ids:
                        raise ValueError(
                            f"Step {step.id!r} in sub_plan {sp.id!r} depends on unknown or cross-sub-plan step {dep!r}"
                        )

    def find_sub_plan(self, sub_plan_id: str) -> SubPlan | None:
        for sp in self.sub_plans:
            if sp.id == sub_plan_id:
                return sp
        return None

    def find_step(self, step_id: str) -> tuple[SubPlan, Step] | None:
        for sp in self.sub_plans:
            step = sp.find_step(step_id)
            if step is not None:
                return sp, step
        return None

    def all_steps(self) -> list[Step]:
        return [step for sp in self.sub_plans for step in sp.steps]

    def all_terminal(self) -> bool:
        steps = self.all_steps()
        return all(s.is_terminal() for s in steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "goal": self.goal,
            "sub_plans": [sp.to_dict() for sp in self.sub_plans],
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        if not isinstance(data, dict):
            raise TypeError("Plan data must be a dict")
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            goal=str(data["goal"]),
            sub_plans=[SubPlan.from_dict(sp) for sp in (data.get("sub_plans") or [])],
            status=PlanStatus(str(data.get("status", PlanStatus.DRAFT.value))),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=dict(data.get("metadata") or {}),
            notes=data.get("notes"),
        )
