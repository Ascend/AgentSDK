#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""F-118 dynamic decomposition backed by the existing coordinator runtime.

Checkpoint / crash recovery
---------------------------
Before each run, ``SwarmModeRunner.run()`` checks for an existing
``task_decomposition.json`` with partial execution data. If found, it
reuses the plan instead of re-decomposing, and the coordinator resumes
from the last incomplete wave. This avoids full re-execution when the
daemon is restarted mid-run (F-39 retry / daemon crash).

The checkpoint file is written at ``.orchestrator_control/swarm_checkpoint.json``
after the coordinator finishes. On the next invocation, the runner reads it
to detect the last completed wave and tells the coordinator to skip completed
tasks.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..git_sync import VerificationFailed
from ..task_decomposition import (
    TaskDecomposer,
    build_swarm_prompt,
    validate_task_execution,
    write_task_plan,
)
from .coordinator import CoordinatorModeRunner

if TYPE_CHECKING:
    from ..agent_runner import AgentRunner, AgentSession
    from ..config.schema import WorkflowConfig
    from ..task_decomposition import TaskPlan
    from ..workflow_engine.cost import CostTracker

logger = logging.getLogger(__name__)


class SwarmModeRunner:
    def __init__(
        self,
        agent_runner: "AgentRunner",
        *,
        max_subtasks: int = 8,
        max_parallel: int = 3,
        max_waves: int = 6,
        cost_tracker: "CostTracker | None" = None,
    ) -> None:
        self._agent_runner = agent_runner
        self._coordinator = CoordinatorModeRunner(agent_runner)
        self._decomposer = TaskDecomposer(
            max_subtasks=max_subtasks,
            max_parallel=max_parallel,
            max_waves=max_waves,
        )
        self._cost_tracker = cost_tracker
        self.max_subtasks = max_subtasks
        self.max_parallel = max_parallel
        self.max_waves = max_waves

    async def run(
        self,
        session: "AgentSession",
        workflow: "WorkflowConfig",
        **hooks: Any,
    ) -> Any:
        workspace_path = Path(session.workspace.path)
        control_dir = workspace_path / ".orchestrator_control"
        checkpoint_path = control_dir / "swarm_checkpoint.json"
        plan_path = control_dir / "task_decomposition.json"

        # Checkpoint recovery: if a task graph with partial evidence exists,
        # skip decomposition and resume from the last incomplete wave.
        plan, resumed = await self._resolve_or_resume(session, plan_path, checkpoint_path)
        if not resumed:
            plan = await self._decomposer.decompose_issue(session.issue)
            plan_path = write_task_plan(plan, workspace_path)
        # Clean up any stale evidence from previous runs (F-39 retry safety).
        evidence_dir = Path(plan_path.parent) / "task_evidence"
        if evidence_dir.is_dir():
            shutil.rmtree(evidence_dir)
            logger.info("Cleaned stale task_evidence/ at %s", evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        original_prompt = session.prompt_override
        original_kind = session.run_kind
        session.task_decomposition_path = str(plan_path)
        session.task_decomposition = plan.to_dict()
        session.prompt_override = build_swarm_prompt(session.issue, plan, plan_path)
        session.run_kind = "swarm"
        logger.info(
            "F-118 swarm plan issue=%s tasks=%d waves=%d max_parallel=%d path=%s",
            session.issue.id,
            len(plan.subtasks),
            len(plan.waves),
            plan.max_parallel,
            plan_path,
        )
        try:
            result = await self._coordinator.run(session, workflow, **hooks)
            if getattr(session, "status", None) == "completed":
                try:
                    validate_task_execution(plan_path, plan)
                    # Backfill evidence from task_evidence/*.json into the
                    # persisted task graph and session snapshot.
                    _backfill_evidence(plan_path, evidence_dir, session)
                except ValueError as exc:
                    raise VerificationFailed(
                        "Swarm execution evidence validation failed",
                        output=str(exc),
                    ) from exc
            return result
        finally:
            session.prompt_override = original_prompt
            session.run_kind = original_kind
            # Write checkpoint so daemon crash recovery can resume from
            # the last completed wave on the next invocation.
            _write_checkpoint(plan_path, checkpoint_path, plan)

    async def _resolve_or_resume(
        self,
        session: "AgentSession",
        plan_path: Path,
        checkpoint_path: Path,
    ) -> tuple["TaskPlan | None", bool]:
        """Check for an existing checkpoint and return ``(plan, resumed)``.

        When ``resumed`` is ``True``, the caller should skip decomposition
        and use the returned ``plan`` directly. Otherwise, normal
        decomposition should proceed.
        """
        if not plan_path.is_file() or not checkpoint_path.is_file():
            return None, False

        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None, False

        completed_wave = checkpoint.get("last_completed_wave", -1)
        if completed_wave < 0:
            return None, False

        # Reconstruct the plan from the persisted payload.
        from ..task_decomposition.models import Subtask, TaskPlan

        waves = [tuple(w) for w in payload.get("waves", [])]
        subtasks = []
        for s in payload.get("subtasks", []):
            subtasks.append(
                Subtask(
                    id=s["id"],
                    title=s.get("title", ""),
                    description=s.get("description", ""),
                    depends_on=tuple(s.get("depends_on", [])),
                    verification=s.get("verification", ""),
                    affected_files=tuple(s.get("affected_files", [])),
                    token_cost=s.get("token_cost", 0.0),
                    budget=s.get("budget", 0.0),
                )
            )
        plan = TaskPlan(
            goal=payload.get("goal", ""),
            subtasks=tuple(subtasks),
            waves=tuple(waves),
            max_parallel=payload.get("max_parallel", self.max_parallel),
        )
        plan.validate(
            max_subtasks=self.max_subtasks,
            max_waves=self.max_waves,
        )

        # Set session attributes so the coordinator sees the resumed state.
        session.task_decomposition_path = str(plan_path)
        session.task_decomposition = payload
        session.prompt_override = build_swarm_prompt(session.issue, plan, plan_path)
        session.run_kind = "swarm"
        logger.info(
            "F-118 swarm checkpoint resume: issue=%s resumed after wave %d/%d",
            session.issue.id,
            completed_wave,
            len(plan.waves),
        )
        return plan, True


__all__ = ["SwarmModeRunner"]


def _backfill_evidence(
    plan_path: Path,
    evidence_dir: Path,
    session: "AgentSession",
) -> None:
    """Read ``task_evidence/*.json`` and update ``task_decomposition.json``.

    Also updates ``session.task_decomposition`` so the orchestration layer
    (report_writer, dashboard) sees the final execution state.
    """
    if not evidence_dir.is_dir():
        return

    plan_path = Path(plan_path)
    if not plan_path.is_file():
        return

    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("_backfill_evidence: cannot read %s", plan_path)
        return

    updated = False
    for ev_file in sorted(evidence_dir.iterdir()):
        if ev_file.suffix != ".json":
            continue
        try:
            ev = json.loads(ev_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        task_id = ev.get("id")
        if not task_id:
            continue
        for sub in payload.get("subtasks", []):
            if sub["id"] == task_id:
                sub["status"] = ev.get("status", sub.get("status", "completed"))
                sub["evidence"] = ev.get("evidence", sub.get("evidence", ""))
                sub["started_at"] = ev.get("started_at", sub.get("started_at"))
                sub["completed_at"] = ev.get("completed_at", sub.get("completed_at"))
                sub["token_cost"] = ev.get("token_cost", sub.get("token_cost", 0.0))
                updated = True
                break

    if updated:
        plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        session.task_decomposition = payload
        logger.info("_backfill_evidence: updated %s with execution evidence", plan_path)


def _write_checkpoint(
    plan_path: Path,
    checkpoint_path: Path,
    plan: "TaskPlan | None",
) -> None:
    """Write a checkpoint file recording the last completed wave.

    The checkpoint is used by ``_resolve_or_resume`` on the next invocation
    to skip already-completed waves after a daemon crash or restart.
    """
    if plan is None:
        return

    if not plan_path.is_file():
        return

    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("_write_checkpoint: cannot read %s", plan_path)
        return

    # Determine the last completed wave index by scanning subtask statuses.
    last_completed_wave = -1
    status_by_id: dict[str, str] = {}
    for sub in payload.get("subtasks", []):
        status_by_id[sub["id"]] = sub.get("status", "pending")

    for wave_idx, wave in enumerate(plan.waves):
        # Only count a wave as completed if ALL its subtasks are completed.
        all_done = all(status_by_id.get(task_id, "pending") == "completed" for task_id in wave)
        if all_done:
            last_completed_wave = wave_idx
        else:
            break

    checkpoint = {
        "last_completed_wave": last_completed_wave,
        "total_waves": len(plan.waves),
        "total_subtasks": len(plan.subtasks),
    }
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Checkpoint: wave %d/%d completed for %s",
            last_completed_wave + 1,
            len(plan.waves),
            plan_path.name,
        )
    except OSError as exc:
        logger.warning("_write_checkpoint: cannot write %s: %s", checkpoint_path, exc)
