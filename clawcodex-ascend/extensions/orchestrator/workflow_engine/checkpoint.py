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

"""Checkpoint persistence and recovery.

Workflow-level persistence with resume from any stage (ARC atomic write).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decision_handler import DecisionHistory
from .errors import CheckpointError, ResumeError
from .workflow_state import StageResult, StageStatus, WorkflowState

logger = logging.getLogger(__name__)


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class Checkpoint:
    """Workflow checkpoint."""

    workflow_name: str
    workflow_version: str
    current_stage: int
    completed_stages: list[int] = field(default_factory=list)
    stage_results: dict[int, dict[str, Any]] = field(default_factory=dict)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    cost_accumulated_usd: float = 0.0
    started_at: str = ""
    last_checkpoint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    # Extended state context
    workflow_state_metadata: dict[str, Any] = field(default_factory=dict)
    rollback_events: list[dict[str, Any]] = field(default_factory=list)
    issue_context: dict[str, Any] | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "stage_results": {
                str(k): {
                    "status": v.get("status", "unknown"),
                    "outputs": v.get("outputs", []),
                    "artifacts": v.get("artifacts", {}),
                    "error": v.get("error"),
                    "cost_usd": v.get("cost_usd", 0.0),
                    "duration_seconds": v.get("duration_seconds", 0.0),
                    "timestamp": v.get("timestamp", ""),
                }
                for k, v in self.stage_results.items()
            },
            "decision_history": self.decision_history,
            "cost_accumulated_usd": self.cost_accumulated_usd,
            "started_at": self.started_at,
            "last_checkpoint": self.last_checkpoint,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
            "workflow_state_metadata": self.workflow_state_metadata,
            "rollback_events": self.rollback_events,
            "issue_context": self.issue_context,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        schema_version = int(data.get("schema_version", 1))
        if schema_version > CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(
                f"Unsupported checkpoint schema version: {schema_version} (expected <= {CHECKPOINT_SCHEMA_VERSION})"
            )

        stage_results = {}
        for k, v in data.get("stage_results", {}).items():
            stage_results[int(k)] = v

        issue_context = data.get("issue_context")
        # Avoid serializing the raw Issue object (not trivially restorable); from_dict keeps dict form
        if issue_context is not None and not isinstance(issue_context, dict):
            issue_context = {"_raw": issue_context}

        return cls(
            workflow_name=data.get("workflow_name", ""),
            workflow_version=str(data.get("workflow_version", "1.0")),
            current_stage=int(data.get("current_stage", 0)),
            completed_stages=[int(s) for s in data.get("completed_stages", [])],
            stage_results=stage_results,
            decision_history=data.get("decision_history", []),
            cost_accumulated_usd=float(data.get("cost_accumulated_usd", 0.0)),
            started_at=data.get("started_at", ""),
            last_checkpoint=data.get("last_checkpoint", ""),
            metadata=data.get("metadata", {}),
            schema_version=schema_version,
            workflow_state_metadata=data.get("workflow_state_metadata", {}),
            rollback_events=data.get("rollback_events", []),
            issue_context=issue_context,
            finished_at=data.get("finished_at"),
        )


class CheckpointManager:
    """Checkpoint manager (ARC atomic write: temp file + rename)."""

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self._run_dir / "checkpoint.json"

    def save(
        self,
        state: WorkflowState,
        decision_history: list[dict[str, Any]] | None = None,
    ) -> Checkpoint:
        """Save a checkpoint (atomic write: temp file first, then rename)."""
        checkpoint = Checkpoint(
            workflow_name=state.workflow_name,
            workflow_version=state.workflow_version,
            current_stage=state.current_stage,
            completed_stages=list(state.completed_stages),
            stage_results={
                sid: {
                    "status": sr.status.value,
                    "outputs": sr.outputs,
                    "artifacts": sr.artifacts,
                    "error": sr.error,
                    "cost_usd": sr.cost_usd,
                    "duration_seconds": sr.duration_seconds,
                    "timestamp": sr.timestamp,
                }
                for sid, sr in state.stage_results.items()
            },
            decision_history=decision_history or [],
            cost_accumulated_usd=state.cost_accumulated_usd,
            started_at=state.started_at,
            last_checkpoint=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata=getattr(state, "metadata", {}),
            workflow_state_metadata=getattr(state, "metadata", {}),
            rollback_events=list(getattr(state, "rollback_events", [])),
            issue_context=_serialize_issue_context(state.issue_context),
            finished_at=state.finished_at,
        )

        try:
            data = checkpoint.to_dict()
            json_text = json.dumps(data, ensure_ascii=False, indent=2, default=str)

            # Atomic write: temp file + rename
            temp_path = self._checkpoint_path.with_suffix(".tmp")
            temp_path.write_text(json_text, encoding="utf-8")
            temp_path.replace(self._checkpoint_path)

            logger.info(
                "Checkpoint saved: stage %s, %s stages completed",
                checkpoint.current_stage,
                len(checkpoint.completed_stages),
            )
        except Exception as exc:
            raise CheckpointError(f"Failed to save checkpoint: {exc}") from exc

        return checkpoint

    def load(self) -> Checkpoint:
        """Load a checkpoint."""
        if not self._checkpoint_path.exists():
            raise CheckpointError(f"Checkpoint file not found: {self._checkpoint_path}")

        try:
            text = self._checkpoint_path.read_text(encoding="utf-8")
            data = json.loads(text)
            return Checkpoint.from_dict(data)
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"Invalid checkpoint JSON: {exc}") from exc
        except Exception as exc:
            raise CheckpointError(f"Failed to load checkpoint: {exc}") from exc

    def exists(self) -> bool:
        """Check whether a checkpoint file exists."""
        return self._checkpoint_path.exists()

    def restore_state(self, checkpoint: Checkpoint) -> WorkflowState:
        """Restore WorkflowState from a checkpoint."""
        state = WorkflowState(
            workflow_name=checkpoint.workflow_name,
            workflow_version=checkpoint.workflow_version,
        )
        state.current_stage = checkpoint.current_stage
        state.completed_stages = list(checkpoint.completed_stages)
        state.cost_accumulated_usd = checkpoint.cost_accumulated_usd
        state.started_at = checkpoint.started_at

        state.workflow_state_metadata = dict(checkpoint.workflow_state_metadata)
        state.rollback_events = list(checkpoint.rollback_events)
        state.issue_context = checkpoint.issue_context
        state.finished_at = checkpoint.finished_at
        state.decision_history = DecisionHistory.from_dict_list(checkpoint.decision_history)
        # Backward-compat: workflow_state_metadata is synonymous with metadata
        if checkpoint.workflow_state_metadata:
            state.metadata = dict(checkpoint.workflow_state_metadata)

        for sid, sr_data in checkpoint.stage_results.items():
            status_str = sr_data.get("status", "completed")
            try:
                status = StageStatus(status_str)
            except ValueError as exc:
                raise CheckpointError(f"Unknown stage status {status_str!r} for stage {sid} in checkpoint") from exc

            result = StageResult(
                stage_id=sid,
                status=status,
                outputs=sr_data.get("outputs", []),
                artifacts=sr_data.get("artifacts", {}),
                error=sr_data.get("error"),
                cost_usd=sr_data.get("cost_usd", 0.0),
                duration_seconds=sr_data.get("duration_seconds", 0.0),
                timestamp=sr_data.get("timestamp", ""),
            )
            state.stage_results[sid] = result
            state.stage_statuses[sid] = status

        return state

    def delete(self) -> None:
        """Delete the checkpoint file."""
        try:
            self._checkpoint_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete checkpoint %s: %s", self._checkpoint_path, exc)


def _serialize_issue_context(issue_context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Serialize issue_context, filtering out non-serializable raw object references.

    The raw Issue object stays in memory; only serializable fields are persisted in the checkpoint.
    """
    if issue_context is None:
        return None
    if not isinstance(issue_context, dict):
        return None
    # Keep all fields except raw object references
    return {k: v for k, v in issue_context.items() if not k.startswith("_")}


class WorkflowResumer:
    """Workflow resume executor.

    Resume workflow execution from a checkpoint.
    """

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        self._checkpoint_manager = checkpoint_manager

    async def resume(self, engine: Any) -> Any:
        """Resume execution from a checkpoint.

        Args:
            engine: a DeclarativeWorkflowEngine instance

        Returns:
            WorkflowResult: execution result.
        """
        if not self._checkpoint_manager.exists():
            raise ResumeError("No checkpoint found to resume from")

        checkpoint = self._checkpoint_manager.load()
        logger.info(
            "Resuming workflow '%s' from stage %s (%s stages completed)",
            checkpoint.workflow_name,
            checkpoint.current_stage,
            len(checkpoint.completed_stages),
        )

        # Restore state
        state = self._checkpoint_manager.restore_state(checkpoint)
        engine.state = state
        engine.cost_tracker.load_state(total_usd=checkpoint.cost_accumulated_usd)

        # Continue from the current stage
        return await engine.execute(from_stage=checkpoint.current_stage)


class ArtifactResolver:
    """Cross-stage artifact path resolver.

    Resolves inter-stage artifact references such as ${stage:3:output:goal.md}.
    """

    _PATTERN = r"\$\{stage:(\d+):output:([^}]+)\}"

    @classmethod
    def resolve(cls, path_template: str, state: WorkflowState, workspace_dir: str = "") -> str:
        """Resolve an artifact path template.

        Args:
            path_template: path template containing references
            state: workflow state
            workspace_dir: workspace root

        Returns:
            The resolved path.
        """
        import re

        def _replace(match: re.Match) -> str:
            stage_id = int(match.group(1))
            artifact_name = match.group(2)
            result = state.get_stage_result(stage_id)
            if result and artifact_name in result.artifacts:
                return result.artifacts[artifact_name]
            if workspace_dir:
                return str(Path(workspace_dir) / f"stage_{stage_id:02d}" / artifact_name)
            return match.group(0)

        return re.sub(cls._PATTERN, _replace, path_template)
