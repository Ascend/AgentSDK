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

"""Workflow audit event writer.

Writes workflow-level events to the audit log in NDJSON format,
for the Visualizer and audit systems.

Complementary to the per-tool audit in tool_event_log.py:
- tool_event_log: per-tool-call audit
- audit: workflow-stage-level audit

Output path: ~/.clawcodex/workflow-events/{workflow_name}/events.ndjson
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# -- Audit event schema --────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkflowAuditEvent:
    """Workflow-level audit event.

    Schema (fixed field order so NDJSON stays greppable):
        ts:              float  -- time.time()
        event_type:      str    -- event type
        workflow_name:   str    -- workflow name
        stage_id:        int|None
        stage_name:      str|None
        outcome:         str|None
        cost_usd:        float
        duration_seconds: float
        error:           str|None
        metadata:        dict   -- extra fields
    """

    event_type: str
    workflow_name: str
    ts: float = field(default_factory=time.time)
    stage_id: int | None = None
    stage_name: str | None = None
    phase: str | None = None
    outcome: str | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "event_type": self.event_type,
            "workflow_name": self.workflow_name,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "phase": self.phase,
            "outcome": self.outcome,
            "cost_usd": self.cost_usd,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# -- Audit writer --─────────────────────────────────────────────────────


class WorkflowAuditWriter:
    """Workflow audit event writer.

    Appends audit events in NDJSON format.
    Reuses the ARC atomic-write pattern (temp file + rename) only when needed;
    daily appends use plain append mode (performance-first; audit logs allow eventual consistency).
    """

    def __init__(
        self,
        workflow_name: str,
        events_dir: str | Path | None = None,
    ) -> None:
        self._workflow_name = workflow_name

        if not events_dir:
            events_dir = Path.home() / ".clawcodex" / "workflow-events" / workflow_name
        self._events_dir = Path(events_dir)
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._events_dir / "events.ndjson"

    def write_event(self, event: WorkflowAuditEvent) -> None:
        """Append one audit event (an NDJSON line)."""
        line = event.to_json() + "\n"
        with open(self._events_path, "a", encoding="utf-8") as f:
            f.write(line)

    def write_stage_start(
        self,
        stage_id: int,
        stage_name: str,
        phase: str = "",
        **kwargs: Any,
    ) -> None:
        self.write_event(
            WorkflowAuditEvent(
                event_type="stage_start",
                workflow_name=self._workflow_name,
                stage_id=stage_id,
                stage_name=stage_name,
                phase=phase,
                metadata=kwargs,
            )
        )

    def write_stage_complete(
        self,
        stage_id: int,
        stage_name: str,
        cost: float = 0.0,
        duration: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.write_event(
            WorkflowAuditEvent(
                event_type="stage_complete",
                workflow_name=self._workflow_name,
                stage_id=stage_id,
                stage_name=stage_name,
                outcome="success",
                cost_usd=cost,
                duration_seconds=duration,
                metadata=kwargs,
            )
        )

    def write_stage_failed(
        self,
        stage_id: int,
        stage_name: str,
        error: str = "",
        cost: float = 0.0,
        duration: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.write_event(
            WorkflowAuditEvent(
                event_type="stage_failed",
                workflow_name=self._workflow_name,
                stage_id=stage_id,
                stage_name=stage_name,
                outcome="failed",
                error=error,
                cost_usd=cost,
                duration_seconds=duration,
                metadata=kwargs,
            )
        )

    def write_gate_result(
        self,
        stage_id: int,
        stage_name: str,
        approved: bool,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        self.write_event(
            WorkflowAuditEvent(
                event_type="gate_result",
                workflow_name=self._workflow_name,
                stage_id=stage_id,
                stage_name=stage_name,
                outcome="approved" if approved else "rejected",
                error=reason if not approved else None,
                metadata=kwargs,
            )
        )

    def write_decision(
        self,
        stage_id: int,
        outcome: str,
        next_stage: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.write_event(
            WorkflowAuditEvent(
                event_type="decision",
                workflow_name=self._workflow_name,
                stage_id=stage_id,
                outcome=outcome,
                metadata={"next_stage": next_stage, **kwargs},
            )
        )

    def write_rollback(
        self,
        stage_id: int,
        rollback_to: int,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        self.write_event(
            WorkflowAuditEvent(
                event_type="rollback",
                workflow_name=self._workflow_name,
                stage_id=stage_id,
                outcome="rollback",
                error=reason,
                metadata={"rollback_to": rollback_to, **kwargs},
            )
        )

    def write_workflow_complete(
        self,
        total_cost: float,
        total_duration: float,
        completed_stages: int,
        total_stages: int,
        **kwargs: Any,
    ) -> None:
        self.write_event(
            WorkflowAuditEvent(
                event_type="workflow_complete",
                workflow_name=self._workflow_name,
                outcome="success",
                cost_usd=total_cost,
                duration_seconds=total_duration,
                metadata={
                    "completed_stages": completed_stages,
                    "total_stages": total_stages,
                    **kwargs,
                },
            )
        )

    def write_workflow_error(
        self,
        error: str,
        stage_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.write_event(
            WorkflowAuditEvent(
                event_type="workflow_error",
                workflow_name=self._workflow_name,
                stage_id=stage_id,
                outcome="error",
                error=error,
                metadata=kwargs,
            )
        )

    def write_cost_event(
        self,
        total_usd: float,
        stage_usd: float,
        budget_max: float,
        **kwargs: Any,
    ) -> None:
        self.write_event(
            WorkflowAuditEvent(
                event_type="cost_update",
                workflow_name=self._workflow_name,
                cost_usd=total_usd,
                metadata={
                    "stage_usd": stage_usd,
                    "budget_max": budget_max,
                    "usage_pct": round(total_usd / budget_max * 100, 1) if budget_max > 0 else 0,
                    **kwargs,
                },
            )
        )

    @property
    def events_path(self) -> Path:
        return self._events_path
