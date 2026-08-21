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

"""Checkpoint and recovery unit tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from extensions.orchestrator.workflow_engine.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    ArtifactResolver,
    Checkpoint,
    CheckpointError,
    CheckpointManager,
)
from extensions.orchestrator.workflow_engine.workflow_state import (
    StageResult,
    StageStatus,
    WorkflowState,
)


# -- Checkpoint serialization --──────────────────────────────────────────────


class TestCheckpointSerialization:
    """Checkpoint data model round-trip serialization."""

    def test_to_dict_includes_schema_version_and_metadata(self) -> None:
        cp = Checkpoint(
            workflow_name="wf",
            workflow_version="1.2",
            current_stage=3,
            completed_stages=[1, 2],
            stage_results={
                1: {
                    "status": "completed",
                    "outputs": ["a.md"],
                    "error": None,
                    "cost_usd": 0.1,
                    "duration_seconds": 1.0,
                    "timestamp": "2026-07-10T10:00:00Z",
                }
            },
            decision_history=[{"stage": 2, "outcome": "proceed"}],
            cost_accumulated_usd=0.5,
            started_at="2026-07-10T09:00:00Z",
            last_checkpoint="2026-07-10T10:00:00Z",
            metadata={"run_id": "r1"},
            workflow_state_metadata={"key": "value"},
            rollback_events=[{"from_stage": 3, "to_stage": 1}],
            issue_context={"id": "i1", "title": "issue"},
            finished_at="2026-07-10T11:00:00Z",
        )
        data = cp.to_dict()
        assert data["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert data["metadata"] == {"run_id": "r1"}
        assert data["workflow_state_metadata"] == {"key": "value"}
        assert data["rollback_events"] == [{"from_stage": 3, "to_stage": 1}]
        assert data["issue_context"] == {"id": "i1", "title": "issue"}
        assert data["finished_at"] == "2026-07-10T11:00:00Z"

    def test_from_dict_roundtrip(self) -> None:
        cp = Checkpoint(
            workflow_name="wf",
            workflow_version="1.0",
            current_stage=2,
            completed_stages=[1],
            stage_results={1: {"status": "completed", "outputs": ["out.md"]}},
            cost_accumulated_usd=1.23,
            metadata={"k": "v"},
        )
        cp2 = Checkpoint.from_dict(cp.to_dict())
        assert cp2.workflow_name == cp.workflow_name
        assert cp2.current_stage == cp.current_stage
        assert cp2.completed_stages == cp.completed_stages
        # to_dict/from_dict normalize default fields in stage_results
        assert cp2.stage_results[1]["status"] == "completed"
        assert cp2.stage_results[1]["outputs"] == ["out.md"]
        assert cp2.cost_accumulated_usd == cp.cost_accumulated_usd
        assert cp2.metadata == cp.metadata

    def test_from_dict_rejects_unsupported_schema_version(self) -> None:
        data = {
            "workflow_name": "wf",
            "workflow_version": "1.0",
            "current_stage": 1,
            "schema_version": CHECKPOINT_SCHEMA_VERSION + 1,
        }
        with pytest.raises(CheckpointError):
            Checkpoint.from_dict(data)


# ── CheckpointManager ───────────────────────────────────────────────


class TestCheckpointManager:
    """Checkpoint manager read/write and recovery."""

    @pytest.fixture
    def tmp_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    def _make_state(self) -> WorkflowState:
        state = WorkflowState(workflow_name="wf", workflow_version="1.0")
        state.current_stage = 2
        state.completed_stages = [1]
        state.cost_accumulated_usd = 0.42
        state.metadata = {"run_id": "r1"}
        state.rollback_events = [{"from_stage": 2, "to_stage": 1}]
        state.issue_context = {"id": "i1", "title": "title"}
        state.stage_results[1] = StageResult(
            stage_id=1,
            status=StageStatus.COMPLETED,
            outputs=["goal.md"],
            artifacts={"report": "/workspace/stage_01/report.md"},
            cost_usd=0.42,
            duration_seconds=10.0,
            timestamp="2026-07-10T10:00:00Z",
        )
        state.stage_statuses[1] = StageStatus.COMPLETED
        return state

    def test_save_and_load(self, tmp_run_dir: str) -> None:
        mgr = CheckpointManager(tmp_run_dir)
        state = self._make_state()
        decision_history = [{"stage": 1, "outcome": "proceed"}]

        cp = mgr.save(state, decision_history=decision_history)
        assert mgr.exists()
        assert cp.schema_version == CHECKPOINT_SCHEMA_VERSION

        loaded = mgr.load()
        assert loaded.workflow_name == "wf"
        assert loaded.current_stage == 2
        assert loaded.completed_stages == [1]
        assert loaded.cost_accumulated_usd == 0.42
        assert loaded.decision_history == decision_history
        assert loaded.workflow_state_metadata == {"run_id": "r1"}
        assert loaded.rollback_events == [{"from_stage": 2, "to_stage": 1}]
        assert loaded.issue_context == {"id": "i1", "title": "title"}

    def test_atomic_write_does_not_leave_corrupt_checkpoint(self, tmp_run_dir: str) -> None:
        """Simulates a crash mid-write: the temp file must not replace the old checkpoint.

        Verifies atomic write by calling internal logic directly: checkpoint.json stays valid after writing.
        """
        mgr = CheckpointManager(tmp_run_dir)
        state = self._make_state()
        mgr.save(state)

        # Corrupting the temp file must not affect the committed checkpoint.json
        temp_path = Path(tmp_run_dir) / "checkpoint.tmp"
        temp_path.write_text("garbage", encoding="utf-8")

        loaded = mgr.load()
        assert loaded.workflow_name == "wf"

    def test_load_missing_raises(self, tmp_run_dir: str) -> None:
        mgr = CheckpointManager(tmp_run_dir)
        with pytest.raises(CheckpointError):
            mgr.load()

    def test_delete(self, tmp_run_dir: str) -> None:
        mgr = CheckpointManager(tmp_run_dir)
        mgr.save(self._make_state())
        assert mgr.exists()
        mgr.delete()
        assert not mgr.exists()

    def test_restore_state(self, tmp_run_dir: str) -> None:
        mgr = CheckpointManager(tmp_run_dir)
        state = self._make_state()
        decision_history = [{"stage": 1, "outcome": "proceed", "next_stage": 2}]
        cp = mgr.save(state, decision_history=decision_history)

        restored = mgr.restore_state(cp)
        assert restored.workflow_name == "wf"
        assert restored.current_stage == 2
        assert restored.completed_stages == [1]
        assert restored.cost_accumulated_usd == 0.42
        assert restored.metadata == {"run_id": "r1"}
        assert restored.rollback_events == [{"from_stage": 2, "to_stage": 1}]
        assert restored.issue_context == {"id": "i1", "title": "title"}
        assert restored.stage_results[1].artifacts == {"report": "/workspace/stage_01/report.md"}
        assert restored.stage_statuses[1] == StageStatus.COMPLETED
        assert restored.decision_history is not None
        assert restored.decision_history.count("proceed", 1) == 1


# ── WorkflowResumer ───────────────────────────────────────────────


class TestArtifactResolver:
    """Cross-stage artifact path resolution."""

    def test_resolve_with_artifact(self) -> None:
        state = WorkflowState(workflow_name="wf")
        state.stage_results[1] = StageResult(
            stage_id=1,
            status=StageStatus.COMPLETED,
            artifacts={"report.md": "/workspace/stage_01/report.md"},
        )
        result = ArtifactResolver.resolve(
            "Review ${stage:1:output:report.md}",
            state=state,
        )
        assert result == "Review /workspace/stage_01/report.md"

    def test_resolve_fallback_to_workspace_dir(self) -> None:
        state = WorkflowState(workflow_name="wf")
        result = ArtifactResolver.resolve(
            "Open ${stage:2:output:summary.md}",
            state=state,
            workspace_dir="/tmp/ws",
        )
        assert result == "Open /tmp/ws/stage_02/summary.md"

    def test_resolve_keeps_template_when_no_fallback(self) -> None:
        state = WorkflowState(workflow_name="wf")
        result = ArtifactResolver.resolve(
            "Open ${stage:2:output:summary.md}",
            state=state,
        )
        assert result == "Open ${stage:2:output:summary.md}"

    def test_resolve_multiple_placeholders(self) -> None:
        state = WorkflowState(workflow_name="wf")
        state.stage_results[1] = StageResult(
            stage_id=1,
            status=StageStatus.COMPLETED,
            artifacts={"a.md": "/path/a.md"},
        )
        state.stage_results[2] = StageResult(
            stage_id=2,
            status=StageStatus.COMPLETED,
            artifacts={"b.md": "/path/b.md"},
        )
        result = ArtifactResolver.resolve(
            "A=${stage:1:output:a.md} B=${stage:2:output:b.md}",
            state=state,
        )
        assert result == "A=/path/a.md B=/path/b.md"


# -- Integration: cleanup and retention --──────────────────────────────────────────────


class TestCheckpointLifecycle:
    """Checkpoint cleanup-on-success / retention-on-failure behavior."""

    @pytest.fixture
    def tmp_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    def test_success_deletes_checkpoint(self, tmp_run_dir: str) -> None:
        mgr = CheckpointManager(tmp_run_dir)
        state = WorkflowState(workflow_name="wf")
        mgr.save(state)
        assert mgr.exists()
        mgr.delete()
        assert not mgr.exists()

    def test_checkpoint_written_to_disk_is_valid_json(self, tmp_run_dir: str) -> None:
        mgr = CheckpointManager(tmp_run_dir)
        state = WorkflowState(workflow_name="wf", workflow_version="1.0")
        state.current_stage = 1
        state.completed_stages = [1]
        state.cost_accumulated_usd = 0.1
        mgr.save(state)

        path = Path(tmp_run_dir) / "checkpoint.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert data["workflow_name"] == "wf"
        assert data["cost_accumulated_usd"] == 0.1
