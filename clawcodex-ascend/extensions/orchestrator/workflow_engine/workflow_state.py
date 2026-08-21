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

"""Workflow runtime state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    """Stage execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    GATE_PENDING = "gate_pending"
    GATE_APPROVED = "gate_approved"
    GATE_REJECTED = "gate_rejected"
    ROLLED_BACK = "rolled_back"


class StageKind(str, Enum):
    """Stage kind."""

    AGENT = "agent"
    GATE = "gate"
    DECISION = "decision"


@dataclass
class StageNode:
    """A stage node in the DAG."""

    id: int
    name: str
    kind: StageKind
    phase: str = ""
    prompt: str = ""
    depends_on: list[int] = field(default_factory=list)
    # Agent configuration
    agent_config: dict[str, Any] = field(default_factory=dict)
    # Validators
    validators: list[dict[str, Any]] = field(default_factory=list)
    # GATE configuration
    gate_mode: str = ""  # manual | auto | threshold
    gate_threshold: float = 0.8
    gate_rollback_to: int | None = None
    # DECISION configuration
    decision_outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Timeout (seconds)
    timeout_seconds: int = 600
    # Maximum retries
    max_retries: int = 0
    # Error handling strategy
    on_error: str = "fail"  # fail | skip | retry | rollback

    @property
    def is_agent_stage(self) -> bool:
        return self.kind == StageKind.AGENT

    @property
    def is_gate_stage(self) -> bool:
        return self.kind == StageKind.GATE

    @property
    def is_decision_stage(self) -> bool:
        return self.kind == StageKind.DECISION


@dataclass
class StageResult:
    """Execution result of a single stage."""

    stage_id: int
    status: StageStatus
    outputs: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    decision_outcome: str | None = None
    decision_next_stage: int | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class WorkflowState:
    """Global runtime state of the workflow.

    Owned by DeclarativeWorkflowEngine and passed between stages.
    """

    workflow_name: str
    workflow_version: str = "1.0"
    current_stage: int = 0
    completed_stages: list[int] = field(default_factory=list)
    stage_results: dict[int, StageResult] = field(default_factory=dict)
    stage_statuses: dict[int, StageStatus] = field(default_factory=dict)
    cost_accumulated_usd: float = 0.0
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    finished_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rollback_events: list[dict[str, Any]] = field(default_factory=list)
    issue_context: dict[str, Any] | None = None  # Issue context from the Orchestrator
    decision_history: Any = field(default=None)  # DecisionHistory instance, injected on checkpoint recovery

    def is_stage_completed(self, stage_id: int) -> bool:
        return stage_id in self.completed_stages

    def get_stage_result(self, stage_id: int) -> StageResult | None:
        return self.stage_results.get(stage_id)

    def mark_stage_running(self, stage_id: int) -> None:
        self.current_stage = stage_id
        self.stage_statuses[stage_id] = StageStatus.RUNNING

    def mark_stage_completed(self, stage_id: int, result: StageResult) -> None:
        if stage_id in self.completed_stages:
            return
        self.completed_stages.append(stage_id)
        self.stage_results[stage_id] = result
        self.stage_statuses[stage_id] = result.status
        self.cost_accumulated_usd += result.cost_usd

    def mark_stage_failed(self, stage_id: int, error: str) -> None:
        self.stage_statuses[stage_id] = StageStatus.FAILED
        if stage_id in self.stage_results:
            self.stage_results[stage_id].status = StageStatus.FAILED
            self.stage_results[stage_id].error = error
        else:
            self.stage_results[stage_id] = StageResult(
                stage_id=stage_id,
                status=StageStatus.FAILED,
                error=error,
            )

    def add_rollback_event(self, from_stage: int, to_stage: int, reason: str = "") -> None:
        """Record a rollback event."""
        self.rollback_events.append(
            {
                "from_stage": from_stage,
                "to_stage": to_stage,
                "reason": reason,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    def mark_workflow_finished(self) -> None:
        self.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @property
    def total_stages(self) -> int:
        return len(self.stage_statuses)

    @property
    def completed_count(self) -> int:
        return len(self.completed_stages)

    @property
    def progress_pct(self) -> float:
        if self.total_stages == 0:
            return 0.0
        return (self.completed_count / self.total_stages) * 100.0
