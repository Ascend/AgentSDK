# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
"""Deterministic seed planner and coordinator prompt for F-118."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable

from .models import Subtask, TaskPlan

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..issue import Issue

    # Optional async LLM client: prompt → response.
    LLMClient = Callable[[str], str | Awaitable[str]]


_LIST_ITEM = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)(.+?)\s*$")
_SEQUENTIAL_MARKERS = (" then ", "after ", "once ", "然后", "之后", "完成后", "再")
_DISCOVERY_MARKERS = (
    "analyze",
    "audit",
    "inspect",
    "investigate",
    "map ",
    "梳理",
    "分析",
    "调查",
    "定位",
)
_VERIFICATION_MARKERS = (
    "test",
    "verify",
    "validation",
    "regression",
    "测试",
    "验证",
    "回归",
    "检查",
)
_FILE_REF = re.compile(
    r"(?:^|[\s`,;:])"
    r"("
    r"(?:src|tests?|docs?|lib|app|extensions|clawcodex_ext|"
    r"scripts?|config|tools?|data|assets?|static|public|"
    r"views?|models?|controllers?|services?|utils?|helpers?|"
    r"middleware|routes?|api|migrations?|seeds?|fixtures?|"
    r"hooks?|layouts?|components?|pages?|stores?|composables?|"
    r"plugins?|providers?|composables?)"
    r"(?:/[\w.\-]+)+"
    r")"
    r"(?:[\s`,;:]|$)",
    re.VERBOSE,
)
_CONTROL_NAMESPACE_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


class TaskDecomposer:
    """Create a bounded seed graph that the coordinator may refine at runtime.

    Parameters
    ----------
    max_subtasks
        Hard upper bound on the number of subtasks (default 8).
    max_parallel
        Max workers allowed in a single wave (default 3).
    max_waves
        Max execution waves (default 6).
    llm_client
        Optional async callable ``(prompt: str) -> str`` for LLM-based plan
        restructuring. When ``None`` (default), the decomposer is purely
        deterministic (regex extraction + three-stage fallback).
    planner_strategy
        Controls how the LLM rewrites the seed plan:
        - ``"none"`` (default): skip LLM rewriting entirely.
        - ``"refine"``: improve task descriptions, dependencies, and
          ``affected_files`` without changing the task count or wave structure.
        - ``"restructure"``: allow the LLM to reorganize tasks, dependencies,
          and waves within the configured bounds.
    """

    def __init__(
        self,
        *,
        max_subtasks: int = 8,
        max_parallel: int = 3,
        max_waves: int = 6,
        llm_client: "LLMClient | None" = None,
        planner_strategy: str = "none",
    ) -> None:
        self.max_subtasks = max(1, int(max_subtasks))
        self.max_parallel = max(1, int(max_parallel))
        self.max_waves = max(1, int(max_waves))
        self._llm_client = llm_client
        if planner_strategy not in ("none", "refine", "restructure"):
            raise ValueError(f"planner_strategy={planner_strategy!r}; expected one of: 'none', 'refine', 'restructure'")
        self._planner_strategy = planner_strategy

    async def decompose_issue(self, issue: "Issue") -> TaskPlan:
        title = str(issue.title or issue.identifier or issue.id or "Complex task").strip()
        description = str(issue.description or "").strip()
        explicit = self._extract_explicit_tasks(description)
        if len(explicit) >= 2:
            subtasks = self._from_explicit_tasks(explicit)
        else:
            subtasks = self._bounded_fallback_tasks(title, description)
        subtasks = tuple(subtasks[: self.max_subtasks])
        waves = _build_waves(subtasks, max_parallel=self.max_parallel)
        fallback_reason: str | None = None
        if len(waves) > self.max_waves:
            # A long sequential list can legitimately need more waves than
            # the configured budget. Fall back to a compact phased plan
            # instead of crashing the entire swarm dispatch.
            logger.warning(
                "dropping explicit task list: %d waves exceed max_waves=%d",
                len(waves),
                self.max_waves,
            )
            fallback_reason = f"explicit plan required {len(waves)} waves; configured maximum is {self.max_waves}"
            subtasks = tuple(self._bounded_fallback_tasks(title, description))
            waves = _build_waves(subtasks, max_parallel=self.max_parallel)
        plan = TaskPlan(
            goal=title,
            subtasks=subtasks,
            waves=waves,
            max_parallel=self.max_parallel,
            fallback_reason=fallback_reason,
        )
        plan = await self._llm_rewrite_plan(plan, issue)
        plan.validate(max_subtasks=self.max_subtasks, max_waves=self.max_waves)
        return plan

    async def _llm_rewrite_plan(self, plan: TaskPlan, issue: "Issue") -> TaskPlan:
        """Optionally refine or restructure the seed plan via the configured LLM client.

        Returns the (possibly modified) plan. When no LLM client is configured
        or the strategy is ``"none"``, the plan is returned unchanged.
        """
        if self._llm_client is None or self._planner_strategy == "none":
            return plan

        instruction = (
            "Refine the task descriptions, dependencies, and affected_files "
            "to better match the issue. Keep the same task count and wave structure."
            if self._planner_strategy == "refine"
            else (
                "Reorganize tasks, dependencies, and waves to best solve the issue. "
                f"Stay within {self.max_subtasks} subtasks, {self.max_waves} waves, "
                f"and at most {self.max_parallel} parallel tasks per wave."
            )
        )
        prompt = (
            f"You are a task planner. Given this issue:\n\n"
            f"Title: {issue.title}\n"
            f"Description: {issue.description}\n\n"
            f"Current plan:\n{json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)}\n\n"
            f"{instruction}\n\n"
            f"Respond with a JSON object matching the TaskPlan schema exactly: "
            f'{{"goal": str, "subtasks": [{{"id", "title", "description", '
            f'"depends_on": [str], "affected_files": [str], "verification": str}}], '
            f'"waves": [[str]], "max_parallel": int, "version": int}}'
        )
        result = self._llm_client(prompt)
        if isinstance(result, Awaitable):
            result = await result
        try:
            data = json.loads(result)
            if not isinstance(data, dict):
                raise TypeError("LLM planner response must be a JSON object")
            subtasks = tuple(
                Subtask(
                    id=s["id"],
                    title=s.get("title", "")[:120],
                    description=s.get("description", ""),
                    depends_on=tuple(s.get("depends_on", [])),
                    verification=s.get("verification", ""),
                    affected_files=tuple(s.get("affected_files", [])),
                )
                for s in data.get("subtasks", [])
            )
            if not subtasks:
                logger.warning("LLM planner returned empty subtask list; using seed plan")
                return plan
            revised = TaskPlan(
                goal=data.get("goal", plan.goal),
                subtasks=subtasks,
                waves=tuple(tuple(w) for w in data.get("waves", [])),
                max_parallel=min(
                    max(1, int(data.get("max_parallel", plan.max_parallel))),
                    self.max_parallel,
                ),
                fallback_reason=plan.fallback_reason,
            )
            # Validate the revised plan; fall back to seed on failure.
            revised.validate(
                max_subtasks=self.max_subtasks,
                max_waves=self.max_waves,
            )
            if self._planner_strategy == "refine":
                seed_ids = tuple(task.id for task in plan.subtasks)
                revised_ids = tuple(task.id for task in revised.subtasks)
                if revised_ids != seed_ids:
                    raise ValueError("refine must preserve subtask ids, count, and order")
                if revised.waves != plan.waves:
                    raise ValueError("refine must preserve execution waves")
                if revised.max_parallel != plan.max_parallel:
                    raise ValueError("refine must preserve max_parallel")
            logger.info(
                "LLM planner (%s) revised plan: %d subtasks, %d waves",
                self._planner_strategy,
                len(revised.subtasks),
                len(revised.waves),
            )
            return revised
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "LLM planner (%s) failed; using seed plan. Reason: %s",
                self._planner_strategy,
                exc,
            )
            return plan

    def _extract_explicit_tasks(self, description: str) -> list[str]:
        tasks: list[str] = []
        for line in description.splitlines():
            match = _LIST_ITEM.match(line)
            if match:
                text = match.group(1).strip()
                if text and len(text) >= 4:
                    tasks.append(text[:600])
            if len(tasks) >= self.max_subtasks:
                break
        return tasks

    def _from_explicit_tasks(self, rows: list[str]) -> list[Subtask]:
        tasks: list[Subtask] = []
        discovery_ids: list[str] = []
        for index, row in enumerate(rows[: self.max_subtasks], start=1):
            normalized = f" {row.lower()} "
            task_id = f"task-{index}"
            is_discovery = any(marker in normalized for marker in _DISCOVERY_MARKERS)
            is_verification = any(marker in normalized for marker in _VERIFICATION_MARKERS)
            dependencies: list[str] = []
            if is_verification:
                # Tests and final verification must observe completed code.
                dependencies.extend(task.id for task in tasks)
            elif not is_discovery:
                # Independent implementation tasks may run together after the
                # shared discovery phase has established the change surface.
                dependencies.extend(discovery_ids)
            if index > 1 and any(marker in normalized for marker in _SEQUENTIAL_MARKERS):
                dependencies.append(f"task-{index - 1}")
            affected_files = tuple(sorted(set(m for m in _FILE_REF.findall(row) if m)))
            for previous in tasks:
                if set(affected_files) & set(previous.affected_files):
                    dependencies.append(previous.id)
            depends_on = tuple(dict.fromkeys(dependencies))
            tasks.append(
                Subtask(
                    id=task_id,
                    title=row[:120],
                    description=row,
                    depends_on=depends_on,
                    verification="Report concrete evidence and changed files.",
                    affected_files=affected_files,
                )
            )
            if is_discovery:
                discovery_ids.append(task_id)
        return tasks

    @staticmethod
    def _fallback_tasks(title: str, description: str) -> list[Subtask]:
        context = description[:1200] or title
        return [
            Subtask(
                id="task-1",
                title="Investigate and bound the change",
                description=f"Inspect the repository and identify the exact implementation surface for: {context}",
                verification="Return verified facts, affected files, and risks.",
            ),
            Subtask(
                id="task-2",
                title="Implement the change",
                description=f"Implement the smallest complete solution for: {title}",
                depends_on=("task-1",),
                verification="Return changed files and explain how the root cause is addressed.",
            ),
            Subtask(
                id="task-3",
                title="Verify and challenge the result",
                description="Run focused tests, then check for regressions and incomplete edge cases.",
                depends_on=("task-2",),
                verification="Return commands, exit codes, failures, and residual risks.",
            ),
        ]

    def _bounded_fallback_tasks(self, title: str, description: str) -> list[Subtask]:
        limit = min(self.max_subtasks, self.max_waves, 3)
        if limit >= 3:
            return self._fallback_tasks(title, description)
        context = description[:1200] or title
        if limit == 1:
            return [
                Subtask(
                    id="task-1",
                    title="Investigate, implement, and verify",
                    description=f"Complete the bounded change end to end: {context}",
                    verification="Report changed files, tests, exit codes, and residual risks.",
                )
            ]
        return [
            Subtask(
                id="task-1",
                title="Investigate and bound the change",
                description=f"Identify the exact implementation surface for: {context}",
                verification="Return verified facts, affected files, and risks.",
            ),
            Subtask(
                id="task-2",
                title="Implement and verify the change",
                description=f"Implement the solution and run focused regression checks for: {title}",
                depends_on=("task-1",),
                verification="Report changed files, tests, exit codes, and residual risks.",
            ),
        ]


def _build_waves(
    subtasks: tuple[Subtask, ...],
    *,
    max_parallel: int,
) -> tuple[tuple[str, ...], ...]:
    remaining = {task.id: set(task.depends_on) for task in subtasks}
    completed: set[str] = set()
    waves: list[tuple[str, ...]] = []
    while remaining:
        ready = sorted(task_id for task_id, deps in remaining.items() if deps <= completed)
        if not ready:
            raise ValueError("task plan contains a dependency cycle")
        for start in range(0, len(ready), max_parallel):
            wave = tuple(ready[start : start + max_parallel])
            waves.append(wave)
            completed.update(wave)
            for task_id in wave:
                remaining.pop(task_id, None)
    return tuple(waves)


def write_task_plan(
    plan: TaskPlan,
    workspace: Path,
    *,
    namespace: str | None = None,
) -> Path:
    control_dir = Path(workspace) / ".orchestrator_control"
    if namespace:
        safe_namespace = _CONTROL_NAMESPACE_SAFE.sub("_", namespace).strip("_")[:80]
        if not safe_namespace:
            raise ValueError("task plan namespace must contain a safe character")
        control_dir /= safe_namespace
    target = control_dir / "task_decomposition.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def validate_task_execution(plan_path: Path, seed_plan: TaskPlan) -> None:
    """Validate coordinator-reported execution evidence against the seed graph."""
    try:
        payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"task execution plan is unreadable: {exc}") from exc
    rows = payload.get("subtasks") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("task execution plan must contain a subtasks list")
    expected_ids = {task.id for task in seed_plan.subtasks}
    if any(not isinstance(row, dict) or row.get("id") is None for row in rows):
        raise ValueError("each task execution evidence row must have an id")
    by_id = {str(row["id"]): row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != expected_ids:
        raise ValueError("task execution evidence must cover exactly the seed task ids")

    intervals: dict[str, tuple[float, float]] = {}
    for task in seed_plan.subtasks:
        row = _load_task_execution_row(Path(plan_path), task.id, by_id[task.id])
        if str(row.get("status", "")).strip().lower() != "completed":
            raise ValueError(f"subtask {task.id} is not marked completed")
        evidence = row.get("evidence")
        if not _has_completion_evidence(evidence):
            raise ValueError(f"subtask {task.id} has no completion evidence")
        started_at = _parse_execution_time(row.get("started_at"), task.id, "started_at")
        completed_at = _parse_execution_time(row.get("completed_at"), task.id, "completed_at")
        if completed_at < started_at:
            raise ValueError(f"subtask {task.id} completed before it started")
        intervals[task.id] = (started_at, completed_at)

    for task in seed_plan.subtasks:
        started_at, _ = intervals[task.id]
        for dependency in task.depends_on:
            if intervals[dependency][1] > started_at:
                raise ValueError(f"subtask {task.id} started before dependency {dependency} completed")

    events: list[tuple[float, int]] = []
    for started_at, completed_at in intervals.values():
        events.append((started_at, 1))
        events.append((completed_at, -1))
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    if peak > seed_plan.max_parallel:
        raise ValueError(f"task execution peak parallelism {peak} exceeds limit {seed_plan.max_parallel}")


def _load_task_execution_row(
    plan_path: Path,
    task_id: str,
    inline_row: dict[str, object],
) -> dict[str, object]:
    """Overlay a worker's per-task evidence file onto the seed-plan row."""
    evidence_dir = (plan_path.parent / "task_evidence").resolve()
    evidence_path = (evidence_dir / f"{task_id}.json").resolve()
    if evidence_path.parent != evidence_dir:
        raise ValueError(f"subtask {task_id} resolves outside the evidence directory")
    if not evidence_path.exists():
        return inline_row
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"subtask {task_id} evidence file is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or str(payload.get("id", "")) != task_id:
        raise ValueError(f"subtask {task_id} evidence file has the wrong id")
    merged = dict(inline_row)
    for field in ("status", "evidence", "started_at", "completed_at"):
        if field in payload:
            merged[field] = payload[field]
    # Worker-authored timestamps are claims, not trustworthy execution
    # facts. The atomic evidence file's mtime is recorded by the host and
    # gives deterministic dependency ordering without letting a model invent
    # a future timestamp. Inline legacy evidence keeps its original semantics.
    recorded_at = evidence_path.stat().st_mtime
    merged["started_at"] = recorded_at
    merged["completed_at"] = recorded_at
    return merged


def _has_completion_evidence(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(_has_completion_evidence(item) for item in value)
    if isinstance(value, dict):
        return any(_has_completion_evidence(item) for item in value.values())
    return False


def _parse_execution_time(value: object, task_id: str, field: str) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, (int, float)):
            result = float(value)
        elif isinstance(value, str):
            result = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        else:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"subtask {task_id} has invalid {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"subtask {task_id} has invalid {field}")
    return result


def build_swarm_prompt(issue: "Issue", plan: TaskPlan, plan_path: Path) -> str:
    plan_json = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)
    description = str(issue.description or "").strip()
    evidence_dir = plan_path.parent / "task_evidence"
    return f"""You are the coordinator for a dynamically decomposed task.

Issue: {issue.title or issue.identifier or issue.id}
Description:
{description}

The seed plan below is stored at {plan_path}. Review it before acting. You may refine task descriptions, dependencies, or affected_files, but stay within {len(plan.subtasks)} subtasks, {len(plan.waves)} waves, and at most {plan.max_parallel} active workers. The plan has been statically checked for file conflicts: no two parallel subtasks claim the same file. If you add new tasks, preserve this constraint.

{plan_json}

Execution rules:
1. Work wave by wave. Spawn workers only for tasks whose dependencies are complete.
2. Give each worker one bounded task and require: verified facts, changed files, commands/tests, and residual risks.
3. Do not let two workers edit the same file concurrently. Serialize overlapping work.
4. You do not have write tools, so evidence recording must be delegated. For every planned task, use a write-capable worker (not an Explore-only worker) and include the exact task id plus these instructions in its Agent prompt:
   - Record only this task in {evidence_dir}/<task-id>.json. Never edit task_decomposition.json. Unique per-task files avoid write conflicts between parallel workers.
   - Before substantive work, atomically write id, status="in_progress", and started_at. Use the current output of `date +%s`; never invent or estimate timestamps.
   - After successful work, atomically replace it with id, status="completed", non-empty evidence, started_at, and completed_at from a fresh `date +%s`. Evidence must name changed files and exact verification commands/results. Do not mark a failed command as completed. The validator uses the evidence file's host mtime for dependency ordering.
   - Do not commit files under .orchestrator_control.
5. Run the repository's focused verification after implementation. If verification fails, create a bounded repair task instead of declaring success.
6. Finish with a structured summary: completed tasks, changes, verification evidence, unresolved risks.
"""


__all__ = [
    "TaskDecomposer",
    "build_swarm_prompt",
    "validate_task_execution",
    "write_task_plan",
]
