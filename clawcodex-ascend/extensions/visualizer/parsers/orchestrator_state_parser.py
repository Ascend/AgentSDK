#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Parse Orchestrator state journals for the Visualizer.

Reads ``{workspace}/.reports/run_*/state_journal.ndjson`` files produced by
the Orchestrator's :class:`StateJournalWriter` and produces structured
state snapshots for the dashboard API and WebSocket push.

Design principle: the parser lives in the Visualizer process and has
zero imports from the orchestrator package.  All data flows through
NDJSON files on disk — the only shared contract is the event schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IssueState:
    """Aggregated state of one issue from the journal."""

    issue_id: str = ""
    status: str = "unknown"
    current_phase: str = ""
    progress: float | None = None
    verification_status: str = ""
    pr_url: str = ""
    pr_number: str | None = None
    session_path: str = ""
    session_id: str = ""
    error: str = ""
    started_at: str = ""
    last_updated: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowStage:
    """Aggregated state of one workflow stage."""

    stage_id: int = 0
    name: str = ""
    phase: str = ""
    status: str = "pending"
    started_at: str = ""
    completed_at: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""


@dataclass
class RunState:
    """Aggregated state of one orchestrator run."""

    run_id: str = ""
    workflow: str = ""
    started_at: str = ""
    completed_at: str = ""
    total_stages: int = 0
    completed_stages: int = 0
    workflow_status: str = "unknown"
    stages: dict[int, WorkflowStage] = field(default_factory=dict)
    issues: dict[str, IssueState] = field(default_factory=dict)
    event_count: int = 0


class OrchestratorStateParser:
    """Parse state_journal.ndjson files into structured RunState objects.

    Parameters
    ----------
    reports_dir:
        Path to the ``.reports`` directory (e.g. ``{workspace}/.reports``).
    """

    def __init__(self, reports_dir: Path | None = None) -> None:
        # In the new ClawCodeX format, the orchestrator state journals
        # live under ``~/.clawcodex/reports/run_*/`` (per-user, not
        # per-workspace). The parser is path-agnostic but the default
        # points at the new location.
        self.reports_dir = reports_dir or (Path.home() / ".clawcodex" / "reports")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a summary list of all runs in the reports directory."""
        runs: list[dict[str, Any]] = []
        if not self.reports_dir.exists():
            return runs
        for run_dir in sorted(self.reports_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            journal = run_dir / "state_journal.ndjson"
            runs.append(
                {
                    "run_id": run_dir.name,
                    "path": str(run_dir),
                    "journal_exists": journal.exists(),
                    "event_count": self._count_ndjson_lines(journal) if journal.exists() else 0,
                }
            )
        return runs

    def parse_run(self, run_id: str) -> RunState | None:
        """Parse a single run's state journal into a RunState."""
        run_dir = self.reports_dir / run_id
        if not run_dir.is_dir():
            return None
        journal = run_dir / "state_journal.ndjson"
        if not journal.exists():
            return RunState(run_id=run_id)

        state = RunState(run_id=run_id)
        events = self._read_ndjson(journal)
        state.event_count = len(events)
        for event in events:
            self._apply_event(state, event)
        return state

    def parse_latest_run(self) -> RunState | None:
        """Parse the most recent run's state journal."""
        runs = self.list_runs()
        if not runs:
            return None
        # Sort by name (run_YYYYMMDD_HHMMSS) — latest last
        latest = runs[-1]
        return self.parse_run(latest["run_id"])

    def get_issue_timeline(self, run_id: str, issue_id: str) -> list[dict[str, Any]]:
        """Return the timeline of events for a specific issue in a run."""
        run_state = self.parse_run(run_id)
        if run_state is None:
            return []
        issue = run_state.issues.get(issue_id)
        if issue is None:
            return []
        return issue.events

    def get_current_snapshot(self) -> dict[str, Any]:
        """Return a dashboard-friendly snapshot of the latest run state.

        This is the primary API for the ``/api/viz/orchestrator/state`` endpoint.
        """
        run_state = self.parse_latest_run()
        if run_state is None:
            return {"status": "no_runs", "issues": [], "run_id": None}

        issues_list = []
        for issue_id, issue in run_state.issues.items():
            issues_list.append(
                {
                    "issue_id": issue_id,
                    "status": issue.status,
                    "current_phase": issue.current_phase,
                    "progress": issue.progress,
                    "verification_status": issue.verification_status,
                    "pr_url": issue.pr_url,
                    "pr_number": issue.pr_number,
                    "session_path": issue.session_path,
                    "session_id": issue.session_id,
                    "error": issue.error,
                    "started_at": issue.started_at,
                    "last_updated": issue.last_updated,
                }
            )

        return {
            "status": "active",
            "run_id": run_state.run_id,
            "workflow": run_state.workflow,
            "started_at": run_state.started_at,
            "completed_at": run_state.completed_at,
            "issue_count": len(run_state.issues),
            "workflow_status": run_state.workflow_status,
            "total_stages": run_state.total_stages,
            "completed_stages": run_state.completed_stages,
            "stages": {
                str(sid): {
                    "stage_id": s.stage_id,
                    "name": s.name,
                    "phase": s.phase,
                    "status": s.status,
                    "started_at": s.started_at,
                    "completed_at": s.completed_at,
                    "cost_usd": s.cost_usd,
                    "duration_seconds": s.duration_seconds,
                    "error": s.error,
                }
                for sid, s in run_state.stages.items()
            },
            "issues": issues_list,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_stage_event(state: RunState, event_type: str, event: dict[str, Any]) -> None:
        """Fold a workflow stage event into the run state."""
        stage_id = event.get("stage_id", 0)
        if stage_id and stage_id not in state.stages:
            state.stages[stage_id] = WorkflowStage(stage_id=stage_id)

        stage = state.stages.get(stage_id)
        if stage is None:
            return
        stage.name = event.get("stage_name", stage.name)
        stage.phase = event.get("phase", stage.phase)
        ts = event.get("timestamp", "")

        # Normalize event type: strip "workflow_" prefix for uniform handling
        normalized = event_type
        if normalized.startswith("workflow_"):
            normalized = normalized[len("workflow_") :]

        if normalized in ("stage_start",):
            stage.status = "running"
            stage.started_at = stage.started_at or ts
        elif normalized in ("stage_complete",):
            stage.status = "completed"
            stage.completed_at = ts
            stage.cost_usd = event.get("cost_usd", event.get("cost", 0.0))
            stage.duration_seconds = event.get("duration_seconds", event.get("duration", 0.0))
        elif normalized in ("stage_failed",):
            stage.status = "failed"
            stage.error = event.get("error", "")
        elif normalized in ("stage_skipped",):
            stage.status = "skipped"
        elif normalized in ("gate_approved", "gate_result"):
            stage.status = "gate_approved" if event.get("approved", True) else "gate_rejected"
            stage.error = event.get("reason", "")
        elif normalized in ("gate_rejected",):
            stage.status = "gate_rejected"
            stage.error = event.get("reason", "")

    @staticmethod
    def _read_ndjson(path: Path) -> list[dict[str, Any]]:
        """Read an NDJSON file into a list of dicts."""
        events: list[dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            pass
        return events

    @staticmethod
    def _count_ndjson_lines(path: Path) -> int:
        """Count non-empty lines in an NDJSON file."""
        try:
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        except Exception:
            return 0

    @staticmethod
    def _apply_event(state: RunState, event: dict[str, Any]) -> None:
        """Fold one NDJSON event into the run state."""
        event_type = event.get("type", "")
        issue_id = event.get("issue_id", "")
        ts = event.get("timestamp", "")

        # Run-level events
        if event_type == "orchestrator_start":
            state.workflow = event.get("workflow", "")
            state.started_at = ts
            return

        # Workflow-level events
        if event_type == "workflow_start":
            state.workflow_status = "running"
            state.total_stages = event.get("total_stages", 0)
            return

        if event_type == "workflow_complete":
            state.workflow_status = "success"
            state.completed_at = ts
            state.completed_stages = state.total_stages
            return

        if event_type == "workflow_error":
            state.workflow_status = "error"
            return

        if event_type.startswith(
            (
                "workflow_stage_",
                "workflow_gate_",
                "workflow_decision",
                "stage_",
                "gate_",
                "decision_",
            )
        ):
            OrchestratorStateParser._apply_stage_event(state, event_type, event)
            return

        # Issue events without an issue identifier are malformed input. Ignore
        # them instead of referencing an unbound local below and aborting the
        # complete journal fold.
        if not issue_id:
            return

        # Issue-level events — ensure the IssueState exists
        if issue_id and issue_id not in state.issues:
            state.issues[issue_id] = IssueState(issue_id=issue_id, started_at=ts)

        if issue_id:
            issue = state.issues[issue_id]
            issue.last_updated = ts
            issue.events.append(event)

        if event_type == "issue_status":
            issue.status = event.get("status", issue.status)
            issue.started_at = issue.started_at or ts

        elif event_type == "phase":
            if issue_id:
                issue.current_phase = event.get("phase", "")
                if event.get("progress") is not None:
                    issue.progress = event["progress"]

        elif event_type == "verification":
            if issue_id:
                issue.verification_status = event.get("verification_status", "")

        elif event_type == "pr_status":
            if issue_id:
                issue.pr_url = event.get("pr_url", "")
                issue.pr_number = event.get("pr_number")

        elif event_type == "session_ref":
            if issue_id:
                issue.session_path = event.get("session_path", "")
                issue.session_id = event.get("session_id", "")

        elif event_type == "error":
            if issue_id:
                issue.error = event.get("error", "")
                issue.status = "error"

        elif event_type == "complete":
            if issue_id:
                overall = event.get("overall_status", "completed")
                if overall in ("completed", "success"):
                    issue.status = "completed"
                elif overall in ("failed", "error"):
                    issue.status = "failed"
                else:
                    issue.status = overall
