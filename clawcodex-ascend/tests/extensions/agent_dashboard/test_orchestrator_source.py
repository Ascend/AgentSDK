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

"""Tests for OrchestratorDashboardSource."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from extensions.agent_dashboard.sources.orchestrator_source import OrchestratorDashboardSource
from extensions.capabilities.dashboard_entry import (
    DASHBOARD_STATUS_COMPLETED,
    DASHBOARD_STATUS_FAILED,
    DASHBOARD_STATUS_IN_PROGRESS,
    DASHBOARD_STATUS_PENDING,
)
from extensions.orchestrator.workflow_engine.workflow_state import StageStatus


def _make_issue_record(
    issue_id: str = "i1",
    identifier: str = "#1 fix bug",
    status: str = "running",
    verification_status: str | None = None,
    pr_url: str | None = None,
    error: str | None = None,
    updated_at: float = 1_700_000_000.0,
) -> Any:
    return SimpleNamespace(
        issue_id=issue_id,
        issue_identifier=identifier,
        status=SimpleNamespace(value=status),
        verification_status=verification_status,
        pr_url=pr_url,
        error=error,
        last_hook_error=None,
        updated_at=updated_at,
    )


def _make_registry(records: list[Any]) -> Any:
    by_id = {r.issue_id: r for r in records}

    def get(issue_id: str) -> Any:
        return by_id.get(issue_id)

    return SimpleNamespace(get=get, running_records=lambda: list(records))


def _make_status_dashboard(running: dict[str, Any] | None = None) -> Any:
    state = SimpleNamespace(
        running=running or {},
        completed=set(),
        failed=set(),
        retry_queue=[],
    )
    return SimpleNamespace(state=lambda: state)


def test_orchestrator_source_returns_empty_when_provider_returns_none() -> None:
    src = OrchestratorDashboardSource(orchestrator_provider=lambda: None)
    assert src.pull() == []


def test_orchestrator_source_emits_issue_entries_from_registry() -> None:
    record = _make_issue_record("i1", "#1 bug", status="running")
    orchestrator = SimpleNamespace(
        _registry=_make_registry([record]),
        status_dashboard=_make_status_dashboard(),
    )
    src = OrchestratorDashboardSource(orchestrator_provider=lambda: orchestrator)
    out = src.pull()
    ids = [e.id for e in out]
    assert "orchestrator:i1" in ids
    entry = next(e for e in out if e.id == "orchestrator:i1")
    assert entry.source == "orchestrator"
    assert entry.status == DASHBOARD_STATUS_IN_PROGRESS
    assert entry.title == "#1 bug"


def test_orchestrator_source_maps_issue_statuses() -> None:
    cases = [
        ("queued", DASHBOARD_STATUS_PENDING),
        ("pending", DASHBOARD_STATUS_PENDING),
        ("running", DASHBOARD_STATUS_IN_PROGRESS),
        ("synced", DASHBOARD_STATUS_COMPLETED),
        ("pending_review", DASHBOARD_STATUS_IN_PROGRESS),
        ("completed", DASHBOARD_STATUS_COMPLETED),
        ("failed", DASHBOARD_STATUS_FAILED),
        ("abandoned", DASHBOARD_STATUS_FAILED),
        ("verification_failed", DASHBOARD_STATUS_FAILED),
    ]
    for raw, expected in cases:
        record = _make_issue_record("i1", status=raw)
        orchestrator = SimpleNamespace(
            _registry=_make_registry([record]),
            status_dashboard=_make_status_dashboard(),
        )
        src = OrchestratorDashboardSource(orchestrator_provider=lambda: orchestrator)
        [entry] = [e for e in src.pull() if e.id == "orchestrator:i1"]
        assert entry.status == expected, f"{raw} -> {entry.status}"


def test_orchestrator_source_includes_verification_and_pr_detail() -> None:
    record = _make_issue_record(
        "i1",
        status="synced",
        verification_status="passed",
        pr_url="https://example.com/pr/1",
    )
    orchestrator = SimpleNamespace(
        _registry=_make_registry([record]),
        status_dashboard=_make_status_dashboard(),
    )
    src = OrchestratorDashboardSource(orchestrator_provider=lambda: orchestrator)
    [entry] = [e for e in src.pull() if e.id == "orchestrator:i1"]
    assert "verification=passed" in entry.detail
    assert "pr=https://example.com/pr/1" in entry.detail


def test_orchestrator_source_emits_live_session_entries() -> None:
    sess = SimpleNamespace(
        issue_id="i1",
        issue_identifier="#1 bug",
        last_event="tool: Read",
        total_tokens=1234,
        seconds_running=42,
        worker_host="worker-1",
    )
    dashboard = _make_status_dashboard(running={"i1": sess})
    orchestrator = SimpleNamespace(
        _registry=_make_registry([]),
        status_dashboard=dashboard,
    )
    src = OrchestratorDashboardSource(orchestrator_provider=lambda: orchestrator)
    out = src.pull()
    entry = next((e for e in out if e.id == "orchestrator:live:i1"), None)
    assert entry is not None
    assert entry.status == DASHBOARD_STATUS_IN_PROGRESS
    assert entry.parent_id == "orchestrator:i1"
    assert entry.owner == "worker-1"
    assert "tool: Read" in entry.detail
    assert "tokens=1234" in entry.detail
    assert "age=42s" in entry.detail


def test_orchestrator_source_includes_registry_records_from_status_sets() -> None:
    """Completed/failed ids in status_dashboard drive registry lookups."""
    record = _make_issue_record("i1", status="completed")
    registry = _make_registry([record])
    dashboard_state = SimpleNamespace(
        running={},
        completed={"i1"},
        failed=set(),
        retry_queue=[],
    )
    dashboard = SimpleNamespace(state=lambda: dashboard_state)
    orchestrator = SimpleNamespace(
        _registry=registry,
        status_dashboard=dashboard,
    )
    src = OrchestratorDashboardSource(orchestrator_provider=lambda: orchestrator)
    out = src.pull()
    ids = {e.id for e in out}
    assert "orchestrator:i1" in ids
    assert all(e.id != "orchestrator:live:i1" for e in out)


def test_orchestrator_source_workflow_stage_entries() -> None:
    stage = SimpleNamespace(id=1, name="compile", phase="build")
    wf_state = SimpleNamespace(
        workflow_name="ci",
        stage_statuses={1: StageStatus.RUNNING},
        stage_results={},
        progress_pct=25.0,
    )
    engine = SimpleNamespace(state=wf_state)
    schema = SimpleNamespace(stages=[stage])
    wf = SimpleNamespace(engine=engine, schema=schema)
    orchestrator = SimpleNamespace(
        _registry=_make_registry([]),
        status_dashboard=_make_status_dashboard(),
        _workflow_orchestrator=wf,
    )
    src = OrchestratorDashboardSource(orchestrator_provider=lambda: orchestrator)
    out = src.pull()
    entry = next((e for e in out if e.id == "orchestrator:wf:1"), None)
    assert entry is not None
    assert entry.title == "ci: compile"
    assert entry.status == DASHBOARD_STATUS_IN_PROGRESS
    assert entry.progress_pct == pytest.approx(0.25)


def test_orchestrator_source_default_ttl_is_1s() -> None:
    src = OrchestratorDashboardSource(orchestrator_provider=lambda: None)
    assert src.cache_ttl_ms == 1_000


def test_orchestrator_source_survives_pull_exception() -> None:
    class _BadOrchestrator:
        @property
        def status_dashboard(self) -> Any:
            raise RuntimeError("boom")

    src = OrchestratorDashboardSource(orchestrator_provider=lambda: _BadOrchestrator())
    assert src.pull() == []
