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

"""Workflow observability integration.

Integrates workflow execution events into ClawCodex's visualization and audit systems.
Integration points:
- State Journal NDJSON event writes
- WorkflowProgressSink progress reporting
- Workflow-level audit events
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..state_journal import StateJournalWriter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# -- Workflow observability writer --──────────────────────────────────────────


class WorkflowObservability:
    """Workflow observability integration.

    Writes workflow-level events to the State Journal for the Visualizer.
    """

    def __init__(
        self,
        journal: StateJournalWriter | None = None,
        tool_events_path: str | None = None,
    ) -> None:
        self._journal = journal
        self._tool_events_path = tool_events_path

    def write_stage_start(self, stage_id: int, stage_name: str, phase: str = "", **kwargs: Any) -> None:
        """Write a stage_start event."""
        self._emit(
            "workflow_stage_start",
            {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "phase": phase,
                **kwargs,
            },
        )

    def write_stage_complete(
        self,
        stage_id: int,
        stage_name: str,
        cost: float = 0.0,
        duration: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Write a stage_complete event."""
        self._emit(
            "workflow_stage_complete",
            {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "cost_usd": cost,
                "duration_seconds": duration,
                **kwargs,
            },
        )

    def write_stage_failed(self, stage_id: int, stage_name: str, error: str = "", **kwargs: Any) -> None:
        """Write a stage_failed event."""
        self._emit(
            "workflow_stage_failed",
            {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "error": error,
                **kwargs,
            },
        )

    def write_gate_request(self, stage_id: int, stage_name: str, mode: str = "", **kwargs: Any) -> None:
        """Write a gate_request event."""
        self._emit(
            "workflow_gate_request",
            {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "gate_mode": mode,
                **kwargs,
            },
        )

    def write_gate_result(
        self, stage_id: int, stage_name: str, approved: bool, reason: str = "", **kwargs: Any
    ) -> None:
        """Write a gate_result event."""
        self._emit(
            "workflow_gate_result",
            {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "approved": approved,
                "reason": reason,
                **kwargs,
            },
        )

    def write_decision(self, stage_id: int, outcome: str, next_stage: int | None = None, **kwargs: Any) -> None:
        """Write a decision event."""
        self._emit(
            "workflow_decision",
            {
                "stage_id": stage_id,
                "outcome": outcome,
                "next_stage": next_stage,
                **kwargs,
            },
        )

    def write_workflow_complete(
        self,
        total_cost: float,
        total_duration: float,
        completed_stages: int,
        total_stages: int,
        **kwargs: Any,
    ) -> None:
        """Write a workflow_complete event."""
        self._emit(
            "workflow_complete",
            {
                "total_cost_usd": total_cost,
                "total_duration_seconds": total_duration,
                "completed_stages": completed_stages,
                "total_stages": total_stages,
                **kwargs,
            },
        )

    def write_workflow_error(self, error: str, stage_id: int | None = None, **kwargs: Any) -> None:
        """Write a workflow_error event."""
        self._emit(
            "workflow_error",
            {
                "error": error,
                "stage_id": stage_id,
                **kwargs,
            },
        )

    def write_cost_event(self, total_usd: float, stage_usd: float, budget_max: float, **kwargs: Any) -> None:
        """Write a cost-tracking event."""
        self._emit(
            "workflow_cost",
            {
                "total_usd": total_usd,
                "stage_usd": stage_usd,
                "budget_max": budget_max,
                "usage_pct": round((total_usd / budget_max * 100), 1) if budget_max > 0 else 0,
                **kwargs,
            },
        )

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event to the State Journal and the audit log."""
        event = {
            "type": event_type,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **data,
        }

        # Write to State Journal
        if self._journal is not None:
            try:
                self._journal.write_event(event)
            except Exception as exc:
                logger.warning("Observability journal write failed: %s", exc)

        # Write to audit log
        if self._tool_events_path:
            try:
                self._append_audit_event(event)
            except Exception as exc:
                logger.warning("Audit event write failed: %s", exc)

    def _append_audit_event(self, event: dict[str, Any]) -> None:
        """Append an audit event to tool-events NDJSON."""
        if not self._tool_events_path:
            return
        audit_path = Path(self._tool_events_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(line)


# ── WorkflowProgressSink ─────────────────────────────────────────────


@dataclass
class WorkflowProgressSink:
    """Workflow progress reporter.

    Implements the ProgressSink protocol, reporting stage completion percentage.
    Reuses the ProgressSink protocol.
    """

    workflow_name: str = ""
    task_id: str = ""
    total_stages: int = 0
    completed_stages: int = 0
    current_stage: str = ""
    _progress_sinks: list[Any] = field(default_factory=list)

    def add_sink(self, sink: Any) -> None:
        """Add a ProgressSink."""
        self._progress_sinks.append(sink)

    def on_stage_start(self, stage_id: int, stage_name: str, phase: str = "") -> None:
        """Stage started."""
        self.current_stage = f"{stage_name}"

    def on_stage_complete(self, stage_id: int, stage_name: str = "", cost: float = 0.0, duration: float = 0.0) -> None:
        """Update progress when a stage completes."""
        self.completed_stages += 1
        progress = (self.completed_stages / self.total_stages * 100) if self.total_stages > 0 else 0.0

        for sink in self._progress_sinks:
            try:
                if hasattr(sink, "on_phase_complete"):
                    from ...api.query import PhaseComplete

                    event = PhaseComplete(
                        phase=self.completed_stages,
                        progress=progress,
                        message=f"Stage {stage_id}: {stage_name} completed",
                    )
                    sink.on_phase_complete(event, None)
            except Exception as exc:
                logger.warning("Progress sink update failed: %s", exc)

    def on_stage_failed(self, stage_id: int, error: str) -> None:
        """Stage failed."""
        self.current_stage = f"FAILED: stage {stage_id}"

    def on_workflow_complete(self, total_cost: float = 0.0, total_duration: float = 0.0) -> None:
        """Workflow completed."""
        for sink in self._progress_sinks:
            try:
                if hasattr(sink, "on_session_complete"):
                    from ...api.query import SessionComplete

                    event = SessionComplete(reason="success")
                    sink.on_session_complete(event, None)
            except Exception as exc:
                logger.warning("Progress sink complete failed: %s", exc)

    def snapshot(self) -> dict[str, Any]:
        """Get the current progress snapshot."""
        return {
            "workflow_name": self.workflow_name,
            "total_stages": self.total_stages,
            "completed_stages": self.completed_stages,
            "current_stage": self.current_stage,
            "progress_pct": self.progress_pct,
        }

    @property
    def progress_pct(self) -> float | None:
        if self.total_stages == 0:
            return None
        return (self.completed_stages / self.total_stages) * 100.0
