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

"""Tests for DashboardGet / DashboardList model tools."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from extensions.agent_dashboard import (
    DashboardEntry,
    DashboardSourceRegistry,
    DashboardStore,
    reset_default_store,
)
from extensions.agent_dashboard.tools import (
    DashboardGetTool,
    DashboardListTool,
)
from clawcodex_ext.tool_system.errors import ToolInputError


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _sample_entries() -> list[DashboardEntry]:
    return [
        DashboardEntry(id="goal:t1", source="goal", title="g1", status="in_progress"),
        DashboardEntry(id="goal:t2", source="goal", title="g2", status="completed"),
        DashboardEntry(id="task:1", source="task", title="t1", status="pending"),
        DashboardEntry(id="task:2", source="task", title="t2", status="failed"),
    ]


class _StaticSource:
    def __init__(self, name: str, entries: list[DashboardEntry]):
        self._name = name
        self._entries = entries

    @property
    def source_name(self) -> str:
        return self._name

    @property
    def cache_ttl_ms(self) -> int:
        return 5_000

    def pull(self, **filters: Any) -> list[DashboardEntry]:
        return list(self._entries)


@pytest.fixture
def tool_context_with_store(tmp_path) -> Any:
    """Build a context duck-typed as a ToolContext with a private DashboardStore.

    The real :class:`ToolContext` is ``@dataclass(slots=True)`` so
    we can't add ``dashboard_store`` to an instance. We use
    ``SimpleNamespace`` and the tool's ``getattr(ctx, ...)`` call
    still works.
    """
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("goal", _sample_entries()[:2]))
    reg.register(_StaticSource("task", _sample_entries()[2:]))
    store = DashboardStore(registry=reg, archive_dir=None)
    ctx = SimpleNamespace(workspace_root=tmp_path, dashboard_store=store)
    return ctx


# ---------------------------------------------------------------------------
# DashboardList
# ---------------------------------------------------------------------------


def test_dashboard_list_default_returns_all_sources(tool_context_with_store: Any) -> None:
    res = DashboardListTool.call({"source": "all"}, tool_context_with_store)
    assert res.is_error is False
    assert res.output["count"] == 4
    assert {e["id"] for e in res.output["entries"]} == {"goal:t1", "goal:t2", "task:1", "task:2"}


def test_dashboard_list_filters_by_source(tool_context_with_store: Any) -> None:
    res = DashboardListTool.call({"source": "goal"}, tool_context_with_store)
    assert res.output["count"] == 2
    assert {e["source"] for e in res.output["entries"]} == {"goal"}


def test_dashboard_list_filters_by_status(tool_context_with_store: Any) -> None:
    res = DashboardListTool.call({"source": "all", "status": "failed"}, tool_context_with_store)
    assert res.output["count"] == 1
    assert res.output["entries"][0]["id"] == "task:2"


def test_dashboard_list_rejects_invalid_source(tool_context_with_store: Any) -> None:
    with pytest.raises(ToolInputError):
        DashboardListTool.call({"source": "nope"}, tool_context_with_store)


def test_dashboard_list_rejects_non_string_source(tool_context_with_store: Any) -> None:
    with pytest.raises(ToolInputError):
        DashboardListTool.call({"source": 42}, tool_context_with_store)


def test_dashboard_list_rejects_non_string_status(tool_context_with_store: Any) -> None:
    with pytest.raises(ToolInputError):
        DashboardListTool.call({"status": 5}, tool_context_with_store)


def test_dashboard_list_serializes_entries(tool_context_with_store: Any) -> None:
    res = DashboardListTool.call({"source": "goal"}, tool_context_with_store)
    [entry] = [e for e in res.output["entries"] if e["id"] == "goal:t1"]
    assert entry == {
        "id": "goal:t1",
        "source": "goal",
        "title": "g1",
        "status": "in_progress",
        "detail": "",
        "source_session_id": None,
        "progress_pct": None,
        "parent_id": None,
        "order": 0,
        "tags": [],
        "owner": None,
        "updated_at_ms": 0,
    }


def test_dashboard_list_uses_default_store(tmp_path: Any) -> None:
    """When context.dashboard_store is None, the tool falls back to the default."""
    from extensions.agent_dashboard import get_default_store

    reset_default_store()
    reg = get_default_store().registry
    reg.clear()
    reg.register(_StaticSource("x", [DashboardEntry(id="x:1", source="x", title="x")]))
    ctx = SimpleNamespace(workspace_root=tmp_path)
    res = DashboardListTool.call({"source": "all"}, ctx)
    assert res.output["count"] == 1
    reset_default_store()


def test_dashboard_list_handles_store_error(tmp_path: Any) -> None:
    class _BoomStore:
        def snapshot(self, *, filters=None):
            raise RuntimeError("kaboom")

    ctx = SimpleNamespace(workspace_root=tmp_path, dashboard_store=_BoomStore())
    res = DashboardListTool.call({"source": "all"}, ctx)
    assert res.output["entries"] == []
    assert res.output["count"] == 0
    assert "kaboom" in res.output["error"]


# ---------------------------------------------------------------------------
# DashboardGet
# ---------------------------------------------------------------------------


def test_dashboard_get_returns_entry(tool_context_with_store: Any) -> None:
    res = DashboardGetTool.call({"entry_id": "goal:t1"}, tool_context_with_store)
    assert res.output["entry"]["id"] == "goal:t1"


def test_dashboard_get_returns_none_when_missing(tool_context_with_store: Any) -> None:
    res = DashboardGetTool.call({"entry_id": "missing"}, tool_context_with_store)
    assert res.output["entry"] is None


def test_dashboard_get_with_source_hint_finds_entry(tool_context_with_store: Any) -> None:
    res = DashboardGetTool.call(
        {"entry_id": "goal:t1", "source": "goal"},
        tool_context_with_store,
    )
    assert res.output["entry"] is not None
    assert res.output["entry"]["id"] == "goal:t1"


def test_dashboard_get_with_source_hint_and_missing(tool_context_with_store: Any) -> None:
    res = DashboardGetTool.call(
        {"entry_id": "missing", "source": "goal"},
        tool_context_with_store,
    )
    assert res.output["entry"] is None
    assert "no entry" in res.output["hint"]


def test_dashboard_get_rejects_blank_id(tool_context_with_store: Any) -> None:
    with pytest.raises(ToolInputError):
        DashboardGetTool.call({"entry_id": ""}, tool_context_with_store)
    with pytest.raises(ToolInputError):
        DashboardGetTool.call({"entry_id": 5}, tool_context_with_store)  # type: ignore[arg-type]


def test_dashboard_get_rejects_non_string_source(tool_context_with_store: Any) -> None:
    with pytest.raises(ToolInputError):
        DashboardGetTool.call(
            {"entry_id": "x:1", "source": 5},
            tool_context_with_store,
        )


def test_dashboard_get_handles_store_error(tmp_path: Any) -> None:
    class _BoomStore:
        def get_by_id(self, entry_id: str):
            raise RuntimeError("kaboom")

    ctx = SimpleNamespace(workspace_root=tmp_path, dashboard_store=_BoomStore())
    res = DashboardGetTool.call({"entry_id": "x:1"}, ctx)
    assert res.output["entry"] is None
    assert "kaboom" in res.output["error"]


# ---------------------------------------------------------------------------
# Read-only invariants
# ---------------------------------------------------------------------------


def test_dashboard_tools_are_marked_read_only() -> None:
    assert DashboardGetTool.is_read_only({}) is True
    assert DashboardListTool.is_read_only({}) is True
    assert DashboardGetTool.is_concurrency_safe({}) is True
    assert DashboardListTool.is_concurrency_safe({}) is True
    assert DashboardGetTool.is_destructive({}) is False
    assert DashboardListTool.is_destructive({}) is False
