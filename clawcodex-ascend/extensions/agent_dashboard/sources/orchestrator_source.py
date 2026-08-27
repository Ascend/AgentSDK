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

"""OrchestratorDashboardSource — exposes orchestrator state to the dashboard.

The source is provider-based so it can be registered before the
:class:`Orchestrator` instance exists (the orchestrator is constructed
inside :meth:`OrchestrationSubsystem.run`).  On each :meth:`pull` it
reads the current orchestrator's ``status_dashboard`` and ``_registry``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from extensions.capabilities.dashboard_entry import (
    DASHBOARD_STATUS_BLOCKED,
    DASHBOARD_STATUS_COMPLETED,
    DASHBOARD_STATUS_FAILED,
    DASHBOARD_STATUS_IN_PROGRESS,
    DASHBOARD_STATUS_PENDING,
    DashboardEntry,
)

logger = logging.getLogger(__name__)

__all__ = ["OrchestratorDashboardSource"]


_ISSUE_STATUS_MAP: dict[str, str] = {
    "queued": DASHBOARD_STATUS_PENDING,
    "pending": DASHBOARD_STATUS_PENDING,
    "running": DASHBOARD_STATUS_IN_PROGRESS,
    "synced": DASHBOARD_STATUS_COMPLETED,
    "pending_review": DASHBOARD_STATUS_IN_PROGRESS,
    "completed": DASHBOARD_STATUS_COMPLETED,
    "failed": DASHBOARD_STATUS_FAILED,
    "abandoned": DASHBOARD_STATUS_FAILED,
    "verification_failed": DASHBOARD_STATUS_FAILED,
}


_STAGE_STATUS_MAP: dict[str, str] = {
    "pending": DASHBOARD_STATUS_PENDING,
    "running": DASHBOARD_STATUS_IN_PROGRESS,
    "completed": DASHBOARD_STATUS_COMPLETED,
    "failed": DASHBOARD_STATUS_FAILED,
    "timed_out": DASHBOARD_STATUS_FAILED,
    "skipped": DASHBOARD_STATUS_COMPLETED,
    "gate_pending": DASHBOARD_STATUS_BLOCKED,
    "gate_approved": DASHBOARD_STATUS_COMPLETED,
    "gate_rejected": DASHBOARD_STATUS_FAILED,
    "rolled_back": DASHBOARD_STATUS_FAILED,
}


class OrchestratorDashboardSource:
    """Read-only dashboard source backed by the Orchestrator runtime."""

    source_name = "orchestrator"

    def __init__(
        self,
        orchestrator_provider: Callable[[], Any],
        *,
        cache_ttl_ms: int = 1_000,
    ) -> None:
        self._orchestrator_provider = orchestrator_provider
        self._cache_ttl_ms = int(cache_ttl_ms)

    @property
    def cache_ttl_ms(self) -> int:
        return self._cache_ttl_ms

    def pull(self, **filters: Any) -> list[DashboardEntry]:
        orchestrator = self._orchestrator_provider()
        if orchestrator is None:
            return []
        entries: list[DashboardEntry] = []
        try:
            entries.extend(self._issue_entries(orchestrator))
            entries.extend(self._live_session_entries(orchestrator))
            entries.extend(self._workflow_stage_entries(orchestrator))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("OrchestratorDashboardSource pull failed: %s", exc)
        return entries

    def _issue_entries(self, orchestrator: Any) -> list[DashboardEntry]:
        registry = getattr(orchestrator, "_registry", None)
        if registry is None:
            return []
        dashboard = getattr(orchestrator, "status_dashboard", None)
        issue_ids: set[str] = set()
        if dashboard is not None:
            state_fn = getattr(dashboard, "state", None)
            if callable(state_fn):
                state = state_fn()
                issue_ids.update(str(k) for k in getattr(state, "running", {}))
                issue_ids.update(str(k) for k in getattr(state, "completed", set()))
                issue_ids.update(str(k) for k in getattr(state, "failed", set()))
        running_records_fn = getattr(registry, "running_records", None)
        if callable(running_records_fn):
            for record in running_records_fn():
                issue_id = str(getattr(record, "issue_id", "") or "")
                if issue_id:
                    issue_ids.add(issue_id)
        entries: list[DashboardEntry] = []
        for issue_id in issue_ids:
            record = getattr(registry, "get", lambda _id: None)(issue_id)
            if record is None:
                continue
            entries.append(self._record_to_entry(record))
        return entries

    def _record_to_entry(self, record: Any) -> DashboardEntry:
        issue_id = str(getattr(record, "issue_id", "") or "")
        identifier = str(getattr(record, "issue_identifier", "") or issue_id)
        status_obj = getattr(record, "status", None)
        status_value = getattr(status_obj, "value", None) or str(status_obj or "")
        status = _ISSUE_STATUS_MAP.get(status_value, DASHBOARD_STATUS_PENDING)
        detail_parts: list[str] = []
        verification = getattr(record, "verification_status", None)
        if verification:
            detail_parts.append(f"verification={verification}")
        pr_url = getattr(record, "pr_url", None)
        if pr_url:
            detail_parts.append(f"pr={pr_url}")
        error = getattr(record, "error", None) or getattr(record, "last_hook_error", None)
        if error:
            detail_parts.append(f"error={str(error)[:80]}")
        if not detail_parts:
            detail_parts.append(status_value)
        updated_at = int(getattr(record, "updated_at", 0) or time.time() * 1000)
        if updated_at < 1e12:
            updated_at = int(updated_at * 1000)
        return DashboardEntry(
            id=f"orchestrator:{issue_id}",
            source="orchestrator",
            title=identifier,
            status=status,
            detail=" · ".join(detail_parts),
            source_session_id=issue_id,
            progress_pct=None,
            tags=["orchestrator"],
            owner=None,
            updated_at_ms=updated_at,
        )

    def _live_session_entries(self, orchestrator: Any) -> list[DashboardEntry]:
        dashboard = getattr(orchestrator, "status_dashboard", None)
        if dashboard is None:
            return []
        state_fn = getattr(dashboard, "state", None)
        if not callable(state_fn):
            return []
        state = state_fn()
        entries: list[DashboardEntry] = []
        for issue_id, sess in getattr(state, "running", {}).items():
            issue_id = str(issue_id)
            identifier = str(getattr(sess, "issue_identifier", "") or issue_id)
            detail_parts: list[str] = []
            last_event = getattr(sess, "last_event", None)
            if last_event:
                detail_parts.append(str(last_event))
            total_tokens = getattr(sess, "total_tokens", 0)
            if total_tokens:
                detail_parts.append(f"tokens={total_tokens}")
            seconds = getattr(sess, "seconds_running", 0)
            if seconds:
                detail_parts.append(f"age={seconds}s")
            entries.append(
                DashboardEntry(
                    id=f"orchestrator:live:{issue_id}",
                    source="orchestrator",
                    title=f"live: {identifier}",
                    status=DASHBOARD_STATUS_IN_PROGRESS,
                    detail=" · ".join(detail_parts) or "running",
                    source_session_id=issue_id,
                    parent_id=f"orchestrator:{issue_id}",
                    progress_pct=None,
                    tags=["orchestrator", "live"],
                    owner=getattr(sess, "worker_host", None),
                    updated_at_ms=int(time.time() * 1000),
                )
            )
        return entries

    def _workflow_stage_entries(self, orchestrator: Any) -> list[DashboardEntry]:
        wf = getattr(orchestrator, "_workflow_orchestrator", None)
        if wf is None:
            return []
        engine = getattr(wf, "engine", None)
        if engine is None:
            return []
        wf_state = getattr(engine, "state", None)
        if wf_state is None:
            return []
        schema = getattr(wf, "schema", None)
        stages: list[Any] = []
        if schema is not None:
            stages = list(getattr(schema, "stages", []) or ())
        stage_statuses = getattr(wf_state, "stage_statuses", {}) or {}
        stage_results = getattr(wf_state, "stage_results", {}) or {}
        workflow_name = getattr(wf_state, "workflow_name", "workflow") or "workflow"
        entries: list[DashboardEntry] = []
        for stage in stages:
            stage_id = getattr(stage, "id", None)
            if stage_id is None:
                continue
            stage_id_str = str(stage_id)
            stage_name = str(getattr(stage, "name", "") or f"stage-{stage_id}")
            phase = str(getattr(stage, "phase", "") or "")
            status_obj = stage_statuses.get(stage_id)
            status_value = getattr(status_obj, "value", None) or str(status_obj or "pending")
            status = _STAGE_STATUS_MAP.get(status_value, DASHBOARD_STATUS_PENDING)
            result = stage_results.get(stage_id)
            detail_parts: list[str] = []
            if phase:
                detail_parts.append(phase)
            if result is not None:
                duration = getattr(result, "duration_seconds", 0.0) or 0.0
                cost = getattr(result, "cost_usd", 0.0) or 0.0
                if duration:
                    detail_parts.append(f"duration={duration:.1f}s")
                if cost:
                    detail_parts.append(f"cost=${cost:.4f}")
            progress = getattr(wf_state, "progress_pct", 0.0) or 0.0
            entries.append(
                DashboardEntry(
                    id=f"orchestrator:wf:{stage_id_str}",
                    source="orchestrator",
                    title=f"{workflow_name}: {stage_name}",
                    status=status,
                    detail=" · ".join(detail_parts) or status_value,
                    source_session_id=None,
                    parent_id="orchestrator:workflow",
                    progress_pct=min(progress / 100.0 if progress > 1.0 else progress, 1.0),
                    tags=["orchestrator", "workflow"],
                    owner=None,
                    updated_at_ms=int(time.time() * 1000),
                )
            )
        return entries
