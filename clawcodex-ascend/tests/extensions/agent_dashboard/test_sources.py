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

"""Tests for GoalDashboardSource + TasksDashboardSource."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from extensions.agent_dashboard.sources.goal_source import GoalDashboardSource
from extensions.agent_dashboard.sources.tasks_source import TasksDashboardSource
from extensions.capabilities.dashboard_entry import (
    DASHBOARD_STATUS_BLOCKED,
    DASHBOARD_STATUS_COMPLETED,
    DASHBOARD_STATUS_IN_PROGRESS,
    DASHBOARD_STATUS_PENDING,
)


# ---------------------------------------------------------------------------
# GoalDashboardSource
# ---------------------------------------------------------------------------


def _make_goal(
    thread_id: str = "t1",
    goal_id: str = "g1",
    objective: str = "Ship the dashboard",
    status: str = "active",
    tokens_used: int = 100,
    token_budget: int | None = 1000,
    time_used: int = 12,
) -> SimpleNamespace:
    return SimpleNamespace(
        thread_id=thread_id,
        goal_id=goal_id,
        objective=objective,
        status=SimpleNamespace(value=status),
        tokens_used=tokens_used,
        token_budget=token_budget,
        time_used_seconds=time_used,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )


def _make_service(goals_by_id: dict[str, Any]) -> Any:
    """Tiny GoalService stand-in (duck-typed)."""

    def get_goal(tid: str):
        return goals_by_id.get(tid)

    return SimpleNamespace(get_goal=get_goal)


def test_goal_source_emits_one_entry_per_thread() -> None:
    goals = {"t1": _make_goal("t1"), "t2": _make_goal("t2")}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1", "t2"))
    out = src.pull()
    assert [e.id for e in out] == ["goal:t1", "goal:t2"]
    assert all(e.source == "goal" for e in out)
    assert all(e.source_session_id in {"t1", "t2"} for e in out)


def test_goal_source_maps_status_enum_to_dashboard_status() -> None:
    cases = [
        ("active", DASHBOARD_STATUS_IN_PROGRESS),
        ("paused", DASHBOARD_STATUS_PENDING),
        ("blocked", DASHBOARD_STATUS_BLOCKED),
        ("usage_limited", DASHBOARD_STATUS_BLOCKED),
        ("budget_limited", DASHBOARD_STATUS_BLOCKED),
        ("complete", DASHBOARD_STATUS_COMPLETED),
    ]
    for raw, expected in cases:
        goals = {"t1": _make_goal("t1", status=raw)}
        src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1",))
        [entry] = src.pull()
        assert entry.status == expected, f"{raw} -> {entry.status}"


def test_goal_source_computes_progress_pct() -> None:
    goals = {"t1": _make_goal("t1", tokens_used=250, token_budget=1000)}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1",))
    [entry] = src.pull()
    assert entry.progress_pct == pytest.approx(0.25)


def test_goal_source_progress_pct_none_without_budget() -> None:
    goals = {"t1": _make_goal("t1", token_budget=None)}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1",))
    [entry] = src.pull()
    assert entry.progress_pct is None


def test_goal_source_progress_pct_clamps_over_budget() -> None:
    goals = {"t1": _make_goal("t1", tokens_used=2000, token_budget=1000)}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1",))
    [entry] = src.pull()
    assert entry.progress_pct == 1.0


def test_goal_source_skips_missing_goals() -> None:
    goals = {"t1": _make_goal("t1")}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1", "t-missing"))
    out = src.pull()
    assert [e.id for e in out] == ["goal:t1"]


def test_goal_source_dedupes_thread_ids() -> None:
    goals = {"t1": _make_goal("t1")}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1", "t1", "t1"))
    out = src.pull()
    assert len(out) == 1


def test_goal_source_handles_provider_exception() -> None:
    def _bad() -> Any:
        raise RuntimeError("boom")

    src = GoalDashboardSource(_make_service({}), thread_ids_provider=_bad)
    out = src.pull()
    assert out == []


def test_goal_source_handles_get_goal_exception() -> None:
    def _bad_get(_tid: str) -> Any:
        raise RuntimeError("boom")

    svc = SimpleNamespace(get_goal=_bad_get)
    src = GoalDashboardSource(svc, thread_ids_provider=lambda: ("t1",))
    out = src.pull()
    assert out == []


def test_goal_source_uses_goal_id_when_thread_id_empty() -> None:
    goals = {"t1": _make_goal(thread_id="", goal_id="gid-abc")}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1",))
    [entry] = src.pull()
    assert entry.id == "goal:gid-abc"


def test_goal_source_default_ttl_is_5s() -> None:
    src = GoalDashboardSource(_make_service({}))
    assert src.cache_ttl_ms == 5_000


def test_goal_source_objective_uses_fallback_when_blank() -> None:
    goals = {"t1": _make_goal("t1", objective="  ")}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1",))
    [entry] = src.pull()
    assert entry.title == "(no objective)"


def test_goal_source_format_detail_uses_budget() -> None:
    goals = {"t1": _make_goal("t1", tokens_used=250, token_budget=1000, time_used=42)}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1",))
    [entry] = src.pull()
    assert "250 / 1000 tokens" in entry.detail
    assert "42s" in entry.detail


def test_goal_source_format_detail_without_budget() -> None:
    goals = {"t1": _make_goal("t1", tokens_used=99, token_budget=None, time_used=7)}
    src = GoalDashboardSource(_make_service(goals), thread_ids_provider=lambda: ("t1",))
    [entry] = src.pull()
    assert "99 tokens" in entry.detail
    assert "7s" in entry.detail


# ---------------------------------------------------------------------------
# TasksDashboardSource
# ---------------------------------------------------------------------------


def _make_tasks() -> dict[str, dict[str, Any]]:
    return {
        "1": {
            "id": "1",
            "subject": "first",
            "description": "the first task",
            "status": "in_progress",
            "owner": "researcher",
            "updated_at_ms": 1_700_000_000_000,
            "tags": ["alpha"],
        },
        "2": {
            "id": "2",
            "subject": "second",
            "description": "the second task",
            "status": "completed",
            "owner": None,
            "updated_at_ms": 1_700_000_000_001,
            "tags": [],
        },
        "3": {
            "id": "3",
            "subject": "blocked one",
            "description": "blocked",
            "status": "pending",
            "owner": None,
            "updated_at_ms": 1_700_000_000_002,
            "blockedBy": ["1"],
        },
    }


def test_tasks_source_emits_entries_for_each_task() -> None:
    ctx = SimpleNamespace(tasks=_make_tasks())
    src = TasksDashboardSource(tool_context_provider=lambda: ctx)
    out = src.pull()
    assert len(out) == 3
    ids = sorted(e.id for e in out)
    assert ids == ["task:1", "task:2", "task:3"]


def test_tasks_source_maps_status() -> None:
    ctx = SimpleNamespace(tasks=_make_tasks())
    src = TasksDashboardSource(tool_context_provider=lambda: ctx)
    out = src.pull()
    by_id = {e.id: e for e in out}
    assert by_id["task:1"].status == DASHBOARD_STATUS_IN_PROGRESS
    assert by_id["task:2"].status == DASHBOARD_STATUS_COMPLETED
    assert by_id["task:3"].status == DASHBOARD_STATUS_PENDING


def test_tasks_source_unknown_status_defaults_to_pending() -> None:
    ctx = SimpleNamespace(tasks={"1": {"id": "1", "subject": "x", "status": "mystery", "description": ""}})
    src = TasksDashboardSource(tool_context_provider=lambda: ctx)
    [entry] = src.pull()
    assert entry.status == DASHBOARD_STATUS_PENDING


def test_tasks_source_handles_missing_provider() -> None:
    src = TasksDashboardSource()
    assert src.pull() == []


def test_tasks_source_handles_provider_exception() -> None:
    def _bad() -> Any:
        raise RuntimeError("boom")

    src = TasksDashboardSource(tool_context_provider=_bad)
    assert src.pull() == []


def test_tasks_source_handles_missing_tasks_attr() -> None:
    src = TasksDashboardSource(tool_context_provider=lambda: SimpleNamespace())
    assert src.pull() == []


def test_tasks_source_uses_id_for_title_when_subject_blank() -> None:
    ctx = SimpleNamespace(tasks={"42": {"id": "42", "subject": "", "description": "x", "status": "pending"}})
    src = TasksDashboardSource(tool_context_provider=lambda: ctx)
    [entry] = src.pull()
    assert entry.title == "42"


def test_tasks_source_includes_blockedBy_tag() -> None:
    ctx = SimpleNamespace(tasks=_make_tasks())
    src = TasksDashboardSource(tool_context_provider=lambda: ctx)
    out = src.pull()
    by_id = {e.id: e for e in out}
    assert any(t.startswith("blocked_by:") for t in by_id["task:3"].tags)


def test_tasks_source_default_ttl_is_1s() -> None:
    src = TasksDashboardSource()
    assert src.cache_ttl_ms == 1_000


def test_tasks_source_skips_none_entries() -> None:
    ctx = SimpleNamespace(tasks={"1": None, "2": {"id": "2", "subject": "x", "status": "pending"}})
    src = TasksDashboardSource(tool_context_provider=lambda: ctx)
    out = src.pull()
    assert [e.id for e in out] == ["task:2"]


def test_tasks_source_handles_dict_like_via_attributes() -> None:
    """A SimpleNamespace with task-shaped attrs should also work."""

    class _Taskish:
        id = "9"
        subject = "from obj"
        description = "yep"
        status = "in_progress"
        owner = "x"
        updated_at_ms = 1
        tags = ["a", "b"]

    ctx = SimpleNamespace(tasks={"9": _Taskish()})
    src = TasksDashboardSource(tool_context_provider=lambda: ctx)
    [entry] = src.pull()
    assert entry.title == "from obj"
    assert entry.owner == "x"
    assert "a" in entry.tags


def test_tasks_source_accepts_dict_directly() -> None:
    src = TasksDashboardSource(tool_context_provider=lambda: _make_tasks())
    out = src.pull()
    assert len(out) == 3
