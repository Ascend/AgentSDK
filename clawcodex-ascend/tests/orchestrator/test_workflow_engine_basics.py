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

"""Unit tests for workflow engine basics: errors, workflow_state, cost."""

import pytest

from extensions.orchestrator.workflow_engine.cost import CostBudget, CostTracker
from extensions.orchestrator.workflow_engine.errors import (
    CheckpointError,
    ConvergenceError,
    CostExceededError,
    DecisionExhaustedError,
    GateRejectedError,
    ResumeError,
    RollbackError,
    StageFailureError,
    StageTimeoutError,
    ValidationError,
    WorkflowEngineError,
    WorkflowSchemaError,
)
from extensions.orchestrator.workflow_engine.workflow_state import (
    StageKind,
    StageNode,
    StageResult,
    StageStatus,
    WorkflowState,
)


# ── errors.py ─────────────────────────────────────────────────────────


class TestWorkflowEngineError:
    """WorkflowEngineError base class and derived exceptions."""

    def test_base_error_carries_stage_id_and_details(self) -> None:
        err = WorkflowEngineError("boom", stage_id=3, details={"k": "v"})
        assert err.stage_id == 3
        assert err.details == {"k": "v"}

    def test_base_error_defaults(self) -> None:
        err = WorkflowEngineError("boom")
        assert err.stage_id is None
        assert err.details == {}

    def test_all_derived_errors_inherit_base(self) -> None:
        for cls in [
            StageTimeoutError,
            StageFailureError,
            CostExceededError,
            ValidationError,
            GateRejectedError,
            DecisionExhaustedError,
            ConvergenceError,
            CheckpointError,
            WorkflowSchemaError,
            ResumeError,
            RollbackError,
        ]:
            assert issubclass(cls, WorkflowEngineError)
            assert isinstance(cls("msg"), WorkflowEngineError)

    def test_error_message_preserved(self) -> None:
        err = StageFailureError("stage failed", stage_id=5)
        assert str(err) == "stage failed"
        assert err.stage_id == 5


# ── workflow_state.py ─────────────────────────────────────────────────


class TestEnums:
    """StageStatus / StageKind enums."""

    def test_stage_status_values(self) -> None:
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.RUNNING.value == "running"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.FAILED.value == "failed"
        assert StageStatus.TIMED_OUT.value == "timed_out"
        assert StageStatus.SKIPPED.value == "skipped"
        assert StageStatus.GATE_PENDING.value == "gate_pending"
        assert StageStatus.GATE_APPROVED.value == "gate_approved"
        assert StageStatus.GATE_REJECTED.value == "gate_rejected"
        assert StageStatus.ROLLED_BACK.value == "rolled_back"

    def test_stage_kind_values(self) -> None:
        assert StageKind.AGENT.value == "agent"
        assert StageKind.GATE.value == "gate"
        assert StageKind.DECISION.value == "decision"


class TestStageNode:
    """StageNode DAG node."""

    def test_kind_properties(self) -> None:
        agent = StageNode(id=1, name="a", kind=StageKind.AGENT)
        gate = StageNode(id=2, name="g", kind=StageKind.GATE)
        decision = StageNode(id=3, name="d", kind=StageKind.DECISION)
        assert agent.is_agent_stage and not agent.is_gate_stage
        assert gate.is_gate_stage and not gate.is_decision_stage
        assert decision.is_decision_stage

    def test_defaults(self) -> None:
        node = StageNode(id=1, name="a", kind=StageKind.AGENT)
        assert node.phase == ""
        assert node.prompt == ""
        assert node.depends_on == []
        assert node.gate_threshold == 0.8
        assert node.timeout_seconds == 600
        assert node.max_retries == 0
        assert node.on_error == "fail"


class TestStageResult:
    """StageResult for a single stage."""

    def test_defaults(self) -> None:
        result = StageResult(stage_id=1, status=StageStatus.COMPLETED)
        assert result.outputs == []
        assert result.artifacts == {}
        assert result.error is None
        assert result.cost_usd == 0.0
        assert result.decision_next_stage is None
        assert result.timestamp != ""

    def test_decision_next_stage_carried(self) -> None:
        result = StageResult(stage_id=1, status=StageStatus.COMPLETED, decision_next_stage=5)
        assert result.decision_next_stage == 5


class TestWorkflowState:
    """WorkflowState global runtime state."""

    def test_mark_stage_running(self) -> None:
        state = WorkflowState(workflow_name="wf")
        state.mark_stage_running(1)
        assert state.current_stage == 1
        assert state.stage_statuses[1] == StageStatus.RUNNING

    def test_mark_stage_completed_accumulates_cost(self) -> None:
        state = WorkflowState(workflow_name="wf")
        result = StageResult(stage_id=1, status=StageStatus.COMPLETED, cost_usd=3.5)
        state.mark_stage_completed(1, result)
        assert state.completed_stages == [1]
        assert state.stage_results[1] is result
        assert state.cost_accumulated_usd == 3.5
        assert state.is_stage_completed(1)

    def test_mark_stage_failed_creates_fallback_result(self) -> None:
        state = WorkflowState(workflow_name="wf")
        state.mark_stage_failed(2, "oops")
        assert state.stage_statuses[2] == StageStatus.FAILED
        assert state.stage_results[2].error == "oops"
        assert state.stage_results[2].status == StageStatus.FAILED

    def test_rollback_event_recorded(self) -> None:
        state = WorkflowState(workflow_name="wf")
        state.add_rollback_event(from_stage=3, to_stage=1, reason="gate rejected")
        assert len(state.rollback_events) == 1
        evt = state.rollback_events[0]
        assert evt["from_stage"] == 3
        assert evt["to_stage"] == 1
        assert evt["reason"] == "gate rejected"
        assert "timestamp" in evt

    def test_progress_pct(self) -> None:
        state = WorkflowState(workflow_name="wf")
        assert state.progress_pct == 0.0
        state.mark_stage_running(1)
        state.mark_stage_completed(1, StageResult(stage_id=1, status=StageStatus.COMPLETED))
        state.mark_stage_running(2)
        state.mark_stage_completed(2, StageResult(stage_id=2, status=StageStatus.COMPLETED))
        state.mark_stage_running(3)
        # total_stages = len(stage_statuses) = 3 (stages touched by mark_*)
        assert state.completed_count == 2
        assert state.progress_pct == pytest.approx(66.67, abs=0.01)

    def test_mark_workflow_finished(self) -> None:
        state = WorkflowState(workflow_name="wf")
        assert state.finished_at is None
        state.mark_workflow_finished()
        assert state.finished_at is not None


# ── cost.py ──────────────────────────────────────────────────────────


class TestCostBudget:
    """CostBudget configuration."""

    def test_defaults(self) -> None:
        budget = CostBudget()
        assert budget.max_total_usd == 50.0
        assert budget.max_per_stage_usd == 10.0
        assert budget.warn_threshold_usd == 40.0  # 50 * 0.8


class TestCostTracker:
    """CostTracker cost tracking."""

    def test_add_accumulates(self) -> None:
        tracker = CostTracker()
        tracker.add(5.0)
        tracker.add(3.0)
        assert tracker.total_usd == 8.0
        assert tracker.stage_usd == 8.0

    def test_reset_stage(self) -> None:
        tracker = CostTracker()
        tracker.add(5.0)
        tracker.reset_stage()
        assert tracker.stage_usd == 0.0
        assert tracker.total_usd == 5.0

    def test_check_budget_ok_no_warnings(self) -> None:
        tracker = CostTracker()
        tracker.add(1.0)
        assert tracker.check_budget() == []

    def test_check_budget_total_exceeded_raises(self) -> None:
        tracker = CostTracker()
        tracker.add(60.0)
        with pytest.raises(CostExceededError):
            tracker.check_budget()

    def test_check_budget_stage_exceeded_raises(self) -> None:
        tracker = CostTracker()
        tracker.add(12.0)
        with pytest.raises(CostExceededError):
            tracker.check_budget()

    def test_check_budget_warns_once_at_threshold(self) -> None:
        tracker = CostTracker()
        # Accumulate total cost to 40 (80% of 50 budget); reset per stage to stay under per-stage budget
        for _ in range(5):
            tracker.add(8.0)
            tracker.reset_stage()
        assert tracker.total_usd == 40.0
        warnings = tracker.check_budget()
        assert any("Cost warning" in w for w in warnings)
        # Second call must not warn again
        assert tracker.check_budget() == []

    def test_load_state_restores_accumulated(self) -> None:
        tracker = CostTracker()
        tracker.load_state(total_usd=12.5, stage_usd=2.0, warned_total=True, warned_stage=True)
        assert tracker.total_usd == 12.5
        assert tracker.stage_usd == 2.0
        # Restored warn flags suppress new warnings
        assert tracker.check_budget() == []
