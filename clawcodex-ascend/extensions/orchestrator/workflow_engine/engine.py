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

# pylint: disable=relative-beyond-top-level

"""Declarative workflow engine core.

Reads workflow.yaml, schedules agents in DAG order, manages GATE/DECISION loops,
and provides workflow-level error recovery and cost tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .cost import CostBudget, CostTracker
from .decision_handler import DecisionHandler
from .errors import (
    ConvergenceError,
    CostExceededError,
    DecisionExhaustedError,
    RollbackError,
    StageFailureError,
    StageTimeoutError,
    WorkflowEngineError,
    WorkflowSchemaError,
)
from .checkpoint import CheckpointManager, WorkflowResumer  # noqa: F401
from .event_bus import EventBus
from .rollback import RollbackManager
from .gate_rollback import GateRollbackHandler
from .validators import ContractValidator
from .workflow_state import (
    StageKind,
    StageNode,
    StageResult,
    StageStatus,
    WorkflowState,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Engine runtime configuration."""

    cost_budget: CostBudget = field(default_factory=CostBudget)
    max_concurrent_stages: int = 1
    default_timeout_seconds: int = 600
    workspace_dir: str = ""
    run_dir: str = ""
    run_id: str = ""
    enable_snapshots: bool = False  # Enable stage snapshots (for rollback)
    llm_client: Any | None = None  # For the llm_judge validator


@dataclass
class WorkflowSchema:
    """Parsed result of workflow.yaml."""

    name: str
    version: str = "1.0"
    description: str = ""
    stages: list[StageNode] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowSchema":
        """Build WorkflowSchema from a dict."""
        name = data.get("name", "unnamed")
        version = str(data.get("version", "1.0"))
        description = data.get("description", "")

        stages_raw = data.get("stages", [])
        if not isinstance(stages_raw, list):
            raise WorkflowSchemaError("workflow.yaml: 'stages' must be a list")

        stages = [_parse_stage_node(i, s) for i, s in enumerate(stages_raw)]
        seen_ids: set[int] = set()
        for s in stages:
            if s.id in seen_ids:
                raise WorkflowSchemaError(f"workflow.yaml: duplicate stage id '{s.id}'")
            seen_ids.add(s.id)
        config = data.get("config", {})
        if not isinstance(config, dict):
            config = {}

        return cls(
            name=name,
            version=version,
            description=description,
            stages=stages,
            config=config,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkflowSchema":
        """Load WorkflowSchema from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise WorkflowSchemaError(f"Workflow file not found: {path}")
        content = path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise WorkflowSchemaError(f"Invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise WorkflowSchemaError("workflow.yaml root must be a mapping")
        return cls.from_dict(data)

    def get_stage(self, stage_id: int) -> StageNode | None:
        for s in self.stages:
            if s.id == stage_id:
                return s
        return None

    def build_dag_order(self) -> list[int]:
        """Return stage IDs in DAG topological order."""
        stage_ids = {s.id for s in self.stages}
        in_degree: dict[int, int] = {s.id: len(s.depends_on) for s in self.stages}
        adj: dict[int, list[int]] = {s.id: [] for s in self.stages}

        for s in self.stages:
            for dep in s.depends_on:
                if dep in adj:
                    adj[dep].append(s.id)

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        order: list[int] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(stage_ids):
            raise WorkflowSchemaError("Workflow DAG contains a cycle")

        return order


def _parse_stage_node(index: int, raw: dict[str, Any]) -> StageNode:
    """Parse a single stage node."""
    kind_str = str(raw.get("kind", "agent")).lower()
    try:
        kind = StageKind(kind_str)
    except ValueError:
        raise WorkflowSchemaError(f"Stage[{index}]: unknown kind '{kind_str}'")

    return StageNode(
        id=raw.get("id", index + 1),
        name=raw.get("name", f"stage-{index + 1}"),
        kind=kind,
        phase=raw.get("phase", ""),
        prompt=raw.get("prompt", ""),
        depends_on=_normalize_int_list(raw.get("depends_on", [])),
        agent_config=raw.get("agent_config", {}),
        validators=raw.get("validators", []),
        gate_mode=raw.get("gate_mode", "manual"),
        gate_threshold=float(raw.get("gate_threshold", 0.8)),
        gate_rollback_to=raw.get("gate_rollback_to"),
        decision_outcomes=raw.get("decision_outcomes", {}),
        timeout_seconds=int(raw.get("timeout_seconds", 0)),
        max_retries=int(raw.get("max_retries", 0)),
        on_error=raw.get("on_error", "fail"),
    )


def _normalize_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    return []


@dataclass
class WorkflowResult:
    """Workflow execution result."""

    success: bool
    workflow_name: str
    completed_stages: int
    total_stages: int
    total_cost_usd: float
    total_duration_seconds: float
    error: str | None = None
    stage_results: dict[int, StageResult] = field(default_factory=dict)


class DeclarativeWorkflowEngine:
    """Declarative workflow engine -- interprets workflow.yaml."""

    def __init__(
        self,
        workflow: WorkflowSchema,
        config: EngineConfig | None = None,
    ) -> None:
        self.workflow = workflow
        self.config = config or EngineConfig()
        self.state = WorkflowState(
            workflow_name=workflow.name,
            workflow_version=workflow.version,
        )
        self.cost_tracker = CostTracker(budget=self.config.cost_budget)
        self.event_bus = EventBus()
        self._dag_order: list[int] = []
        self._stage_runner = None  # Injected lazily

        self._validator = ContractValidator(
            workspace_dir=self.config.workspace_dir,
            llm_client=self.config.llm_client,
        )

        self._rollback_manager: RollbackManager | None = None
        self._gate_rollback_handler: GateRollbackHandler | None = None
        self._init_rollback()

        self._checkpoint_manager: CheckpointManager | None = None

        # GATE/DECISION handlers
        self._decision_handler = DecisionHandler()
        self._decision_count: dict[
            int, int
        ] = {}  # Kept for compatibility; actual counting is handled by DecisionHandler.history

    def set_stage_runner(self, runner: Any) -> None:
        """Inject the StageRunner adapter."""
        self._stage_runner = runner

    def _init_rollback(self) -> None:
        """Initialize the rollback system."""
        if self.config.enable_snapshots and self.config.workspace_dir:
            self._rollback_manager = RollbackManager(
                workspace_dir=self.config.workspace_dir,
            )
            self._gate_rollback_handler = GateRollbackHandler(
                rollback_manager=self._rollback_manager,
            )

    def set_checkpoint_manager(self, checkpoint_manager: CheckpointManager) -> None:
        """Inject a checkpoint manager."""
        self._checkpoint_manager = checkpoint_manager

    def _effective_timeout(self, stage: StageNode) -> int:
        """Resolve the effective timeout for a stage.

        stage.timeout_seconds == 0 means no explicit config; use the engine default.
        """
        return stage.timeout_seconds or self.config.default_timeout_seconds

    def _save_checkpoint(self, current_stage_id: int) -> None:
        """Save a checkpoint."""
        if self._checkpoint_manager is None:
            return
        self.state.current_stage = current_stage_id
        try:
            self._checkpoint_manager.save(
                self.state,
                decision_history=self._decision_handler.history.to_dict(),
            )
        except Exception:
            logger.debug("Failed to save checkpoint at stage %s", current_stage_id, exc_info=True)

    async def execute(self, from_stage: int | None = None) -> WorkflowResult:
        """Execute the workflow.

        Args:
            from_stage: resume from the given stage (for checkpoint recovery).

        Returns:
            WorkflowResult: execution result.
        """
        start_time = time.time()

        try:
            self._dag_order = self.workflow.build_dag_order()
        except WorkflowSchemaError as exc:
            return WorkflowResult(
                success=False,
                workflow_name=self.workflow.name,
                completed_stages=0,
                total_stages=len(self.workflow.stages),
                total_cost_usd=0.0,
                total_duration_seconds=0.0,
                error=str(exc),
            )

        # Initialize all stage statuses
        for stage in self.workflow.stages:
            self.state.stage_statuses[stage.id] = StageStatus.PENDING

        self.event_bus.emit_workflow_start(
            workflow_name=self.workflow.name,
            total_stages=len(self.workflow.stages),
        )

        start_index = 0
        if from_stage is not None:
            for i, sid in enumerate(self._dag_order):
                if sid == from_stage:
                    start_index = i
                    break
            else:
                raise WorkflowEngineError(
                    f"from_stage {from_stage} not found in workflow DAG order",
                    stage_id=from_stage,
                )
            for i in range(start_index):
                sid = self._dag_order[i]
                if sid not in self.state.completed_stages:
                    self.state.completed_stages.append(sid)
                    self.state.stage_statuses[sid] = StageStatus.COMPLETED

        # Execution loop
        error_msg: str | None = None
        idx = start_index
        while idx < len(self._dag_order):
            stage_id = self._dag_order[idx]
            stage = self.workflow.get_stage(stage_id)
            if stage is None:
                idx += 1
                continue

            # Check whether dependencies are satisfied
            if not self._dependencies_satisfied(stage):
                self.event_bus.emit_stage_skipped(
                    stage_id=stage.id,
                    stage_name=stage.name,
                    reason="dependencies not satisfied",
                )
                self.state.stage_statuses[stage.id] = StageStatus.SKIPPED
                idx += 1
                continue

            try:
                result = await self._execute_stage(stage)
                self.state.mark_stage_completed(stage.id, result)

                # Save a checkpoint after each stage completes
                self._save_checkpoint(stage.id)

                self.event_bus.emit_stage_complete(
                    stage_id=stage.id,
                    stage_name=stage.name,
                    cost=result.cost_usd,
                    duration=result.duration_seconds,
                )

                # GATE rejection handling
                if stage.is_gate_stage and result.status == StageStatus.GATE_REJECTED:
                    if stage.on_error == "rollback" or stage.gate_rollback_to is not None:
                        error_msg = result.error or f"GATE stage {stage.id} rejected"
                        idx = await self._handle_gate_rejection(stage, result)
                        continue
                    if stage.on_error == "skip":
                        logger.info("GATE stage %s rejected, skipping (on_error=skip)", stage.id)
                        # Mark as COMPLETED so downstream dependencies are satisfied
                        result.status = StageStatus.COMPLETED
                        idx += 1
                        continue

                    # on_error == "fail" (default)
                    error_msg = result.error or f"GATE stage {stage.id} rejected"
                    break

                # DECISION stage: compute the next stage
                if stage.is_decision_stage and result.decision_next_stage is not None:
                    try:
                        next_idx = self._dag_order.index(result.decision_next_stage)
                        idx = next_idx
                        continue
                    except ValueError:
                        logger.warning("Decision next_stage %s not in DAG order", result.decision_next_stage)

                idx += 1

            except StageTimeoutError as exc:
                result = self._handle_stage_error(stage, exc, "timeout")
                self.state.mark_stage_failed(stage.id, str(exc))
                error_msg = str(exc)
                if stage.on_error == "fail":
                    break
                if stage.on_error == "rollback":
                    idx = await self._rollback_to_stage(stage)
                    continue
                idx += 1

            except StageFailureError as exc:
                result = self._handle_stage_error(stage, exc, "failure")
                self.state.mark_stage_failed(stage.id, str(exc))
                error_msg = str(exc)
                if stage.on_error == "fail":
                    break
                if stage.on_error == "rollback":
                    idx = await self._rollback_to_stage(stage)
                    continue
                idx += 1

            except CostExceededError as exc:
                self.event_bus.emit_workflow_error(error=str(exc), stage_id=stage.id)
                error_msg = str(exc)
                break

            except WorkflowEngineError as exc:
                self.event_bus.emit_workflow_error(error=str(exc), stage_id=stage.id)
                error_msg = str(exc)
                break

        self.state.mark_workflow_finished()
        total_duration = time.time() - start_time

        if error_msg:
            self.event_bus.emit_workflow_error(error=error_msg)
        else:
            self.event_bus.emit_workflow_complete(
                total_cost=self.cost_tracker.total_usd,
                total_duration=total_duration,
            )

        return WorkflowResult(
            success=error_msg is None,
            workflow_name=self.workflow.name,
            completed_stages=self.state.completed_count,
            total_stages=self.state.total_stages,
            total_cost_usd=self.cost_tracker.total_usd,
            total_duration_seconds=total_duration,
            error=error_msg,
            stage_results=dict(self.state.stage_results),
        )

    # -- Stage execution --─────────────────────────────────────────────────

    async def _execute_stage(self, stage: StageNode) -> StageResult:
        """Execute a single stage.

        Dispatch to the appropriate handler based on stage kind.
        """
        self.state.mark_stage_running(stage.id)
        self.event_bus.emit_stage_start(
            stage_id=stage.id,
            stage_name=stage.name,
            phase=stage.phase,
        )

        # Save snapshot (for rollback)
        self._save_stage_snapshot(stage)

        stage_start = time.time()

        if stage.is_agent_stage:
            result = await self._run_agent_stage(stage)
        elif stage.is_gate_stage:
            result = await self._run_gate_stage(stage)
        elif stage.is_decision_stage:
            result = await self._run_decision_stage(stage)
        else:
            result = StageResult(
                stage_id=stage.id,
                status=StageStatus.COMPLETED,
            )

        result.duration_seconds = time.time() - stage_start
        return result

    async def _run_agent_stage(self, stage: StageNode) -> StageResult:
        """Execute an agent stage.

        Invoke AgentRunner through the StageRunner adapter.
        """
        self.cost_tracker.reset_stage()

        if self._stage_runner is None:
            raise WorkflowEngineError(
                "StageRunner not injected. Call set_stage_runner() before execute().",
                stage_id=stage.id,
            )

        effective_timeout = self._effective_timeout(stage)
        try:
            run_result = await asyncio.wait_for(
                self._stage_runner.run(stage, self.state),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            raise StageTimeoutError(
                f"Stage {stage.id} ({stage.name}) timed out after {effective_timeout}s",
                stage_id=stage.id,
            )

        cost = getattr(run_result, "cost_usd", 0.0)
        self.cost_tracker.add(cost)

        # Budget check
        warnings = self.cost_tracker.check_budget()
        for w in warnings:
            self.event_bus.emit_cost_warning(message=w)

        # Check StageRunner execution result
        if not getattr(run_result, "success", False):
            error_msg = getattr(run_result, "error", "Stage execution failed")
            raise StageFailureError(
                f"Stage {stage.id} ({stage.name}) failed: {error_msg}",
                stage_id=stage.id,
            )

        # Output validation
        if stage.validators:
            validation_errors = await self._validate_stage_output(stage, run_result)
            if validation_errors:
                raise StageFailureError(
                    f"Stage {stage.id} validation failed: {validation_errors}",
                    stage_id=stage.id,
                )

        return StageResult(
            stage_id=stage.id,
            status=StageStatus.COMPLETED,
            outputs=getattr(run_result, "outputs", []),
            artifacts=getattr(run_result, "artifacts", {}),
            cost_usd=cost,
        )

    async def _run_gate_stage(self, stage: StageNode) -> StageResult:
        """Execute a GATE stage."""
        if self._stage_runner is None:
            raise WorkflowEngineError("StageRunner not injected", stage_id=stage.id)

        self.event_bus.emit_gate_request(
            stage_id=stage.id,
            stage_name=stage.name,
            mode=stage.gate_mode,
        )

        effective_timeout = self._effective_timeout(stage)
        try:
            gate_result = await asyncio.wait_for(
                self._stage_runner.run_gate(stage, self.state),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            raise StageTimeoutError(
                f"GATE stage {stage.id} timed out",
                stage_id=stage.id,
            )

        approved = getattr(gate_result, "approved", False)
        if approved:
            self.event_bus.emit_gate_approved(
                stage_id=stage.id,
                stage_name=stage.name,
            )
            return StageResult(
                stage_id=stage.id,
                status=StageStatus.GATE_APPROVED,
            )
        else:
            self.event_bus.emit_gate_rejected(
                stage_id=stage.id,
                stage_name=stage.name,
                reason=getattr(gate_result, "reason", "unknown"),
            )
            return StageResult(
                stage_id=stage.id,
                status=StageStatus.GATE_REJECTED,
                error=getattr(gate_result, "reason", "GATE rejected"),
            )

    async def _run_decision_stage(self, stage: StageNode) -> StageResult:
        """Execute a DECISION stage.

        Uses DecisionHandler for loop-count checks and convergence detection.
        """
        if self._stage_runner is None:
            raise WorkflowEngineError("StageRunner not injected", stage_id=stage.id)

        effective_timeout = self._effective_timeout(stage)
        try:
            decision_result = await asyncio.wait_for(
                self._stage_runner.run_decision(stage, self.state),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            raise StageTimeoutError(
                f"DECISION stage {stage.id} timed out",
                stage_id=stage.id,
            )

        outcome = getattr(decision_result, "outcome", "proceed")
        next_stage = getattr(decision_result, "next_stage", None)

        stage_result = StageResult(
            stage_id=stage.id,
            status=StageStatus.COMPLETED,
            decision_outcome=outcome,
            decision_next_stage=next_stage,
        )

        decision = self._decision_handler.resolve(stage, stage_result)

        if decision.exhausted:
            self.event_bus.emit_decision(
                stage_id=stage.id,
                outcome=outcome,
                next_stage=None,
            )
            raise DecisionExhaustedError(
                decision.reason,
                stage_id=stage.id,
            )

        if decision.converged:
            self.event_bus.emit_decision(
                stage_id=stage.id,
                outcome=outcome,
                next_stage=None,
            )
            raise ConvergenceError(
                decision.reason,
                stage_id=stage.id,
            )

        resolved_next = decision.next_stage or next_stage
        self.event_bus.emit_decision(
            stage_id=stage.id,
            outcome=decision.outcome,
            next_stage=resolved_next,
        )

        return StageResult(
            stage_id=stage.id,
            status=StageStatus.COMPLETED,
            decision_outcome=decision.outcome,
            decision_next_stage=resolved_next,
        )

    async def _validate_stage_output(self, stage: StageNode, run_result: Any) -> list[str]:
        """Validate stage output.

        Delegates to the injected ``ContractValidator`` instance, supporting all 7 validator types.
        """
        results = await self._validator.validate_all(stage.validators)
        return [r.message for r in results if not r.passed]

    # -- Error handling --─────────────────────────────────────────────────

    def _handle_stage_error(self, stage: StageNode, exc: WorkflowEngineError, error_type: str) -> StageResult:
        """Handle a stage error."""
        self.event_bus.emit_stage_failed(
            stage_id=stage.id,
            stage_name=stage.name,
            error=str(exc),
        )
        return StageResult(
            stage_id=stage.id,
            status=StageStatus.FAILED,
            error=str(exc),
        )

    async def _rollback_to_stage(self, stage: StageNode) -> int:
        """Roll back to a given stage (via RollbackManager).

        Priority:
        1. Restore the snapshot via RollbackManager
        2. Fall back to rolling back to a dependency stage
        """
        if self._rollback_manager is not None:
            try:
                target = self._rollback_manager.resolve_rollback_target(stage)
                self._rollback_manager.restore_snapshot(target.stage_id)
                self._rollback_manager.update_state_on_rollback(
                    self.state,
                    target.stage_id,
                    stage.id,
                )
                return self._dag_order.index(target.stage_id)
            except (RollbackError, ValueError) as exc:
                logger.warning("Rollback failed: %s, falling back to simple rollback", exc)

        # Fallback: simple rollback to a dependency stage
        if stage.depends_on:
            target = stage.depends_on[0]
        else:
            target = self._dag_order[0] if self._dag_order else 0
        try:
            return self._dag_order.index(target)
        except ValueError:
            return 0

    def _save_stage_snapshot(self, stage: StageNode) -> None:
        """Save a snapshot before the stage executes."""
        if self._rollback_manager is not None:
            try:
                self._rollback_manager.save_snapshot(stage)
            except Exception as exc:
                logger.debug("Failed to save snapshot for stage %s: %s", stage.id, exc)

    async def _handle_gate_rejection(self, stage: StageNode, result: StageResult) -> int:
        """Handle a GATE rejection by rolling back.

        Returns:
            The rollback target's index in the DAG
        """
        if self._gate_rollback_handler is not None:
            try:
                new_idx, gate_result = await self._gate_rollback_handler.handle_gate_rejection(
                    stage=stage,
                    state=self.state,
                    dag_order=self._dag_order,
                    rejection_reason=result.error or "GATE rejected",
                )
                self.event_bus.emit_gate_rejected(
                    stage_id=stage.id,
                    stage_name=stage.name,
                    reason=gate_result.reason,
                )
                return new_idx
            except Exception as exc:
                logger.warning("Gate rollback failed: %s", exc)

        # Fallback: simple rollback
        return await self._rollback_to_stage(stage)

    # -- Dependency checking --─────────────────────────────────────────────────

    def _dependencies_satisfied(self, stage: StageNode) -> bool:
        """Check whether all dependency stages are completed."""
        for dep_id in stage.depends_on:
            if dep_id not in self.state.completed_stages:
                return False
            dep_result = self.state.get_stage_result(dep_id)
            if dep_result is None or dep_result.status not in (
                StageStatus.COMPLETED,
                StageStatus.GATE_APPROVED,
            ):
                return False
        return True
