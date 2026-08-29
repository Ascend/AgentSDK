#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for the ``/dashboard`` slash command."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from extensions.agent_dashboard import (
    DashboardEntry,
    DashboardSourceRegistry,
    DashboardStore,
    reset_default_store,
)
from clawcodex_ext.command_system.dashboard_command import (
    DASHBOARD_COMMAND,
    _format_section,
    _format_snapshot,
    _parse_args,
    dashboard_command_call,
)
from clawcodex_ext.command_system.types import (
    CommandContext,
    InteractiveCommand,
    InteractiveOutcome,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


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
def store() -> DashboardStore:
    reg = DashboardSourceRegistry()
    reg.register(
        _StaticSource(
            "goal",
            [
                DashboardEntry(
                    id="goal:t1",
                    source="goal",
                    title="ship X",
                    status="in_progress",
                    progress_pct=0.4,
                    detail="50 / 100 tokens",
                ),
                DashboardEntry(id="goal:t2", source="goal", title="ship Y", status="completed"),
            ],
        )
    )
    reg.register(
        _StaticSource(
            "task",
            [
                DashboardEntry(id="task:1", source="task", title="write tests", status="pending"),
                DashboardEntry(id="task:2", source="task", title="fix bug", status="failed"),
            ],
        )
    )
    return DashboardStore(registry=reg, archive_dir=None)


@pytest.fixture
def ctx(store: DashboardStore, tmp_path: Path) -> CommandContext:
    ctx = CommandContext(workspace_root=tmp_path, cwd=tmp_path)
    # Inject our private store via the ``app_state_store`` slot
    # (which the command resolves via ``getattr``).
    ctx.app_state_store = SimpleNamespace(dashboard_store=store)
    return ctx


@pytest.fixture(autouse=True)
def _reset_default_store() -> None:
    reset_default_store()


# ---------------------------------------------------------------------------
# Smoke / registration
# ---------------------------------------------------------------------------


def test_dashboard_command_is_registered_with_call_impl() -> None:
    assert isinstance(DASHBOARD_COMMAND, InteractiveCommand)
    assert DASHBOARD_COMMAND.name == "dashboard"
    assert "dash" in DASHBOARD_COMMAND.aliases


@pytest.mark.asyncio
async def test_dashboard_command_run_returns_interactive_outcome(ctx: CommandContext) -> None:
    result = await DASHBOARD_COMMAND.run("", ctx)
    assert isinstance(result, InteractiveOutcome)
    assert "Dashboard" in (result.message or "")
    # A populated snapshot is long enough to be scrollable.
    assert result.scrollable is True


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def test_parse_args_extracts_flags() -> None:
    flags = _parse_args("--status failed --id task:1")
    assert flags == {"status": "failed", "id": "task:1"}


def test_parse_args_handles_empty() -> None:
    assert _parse_args("") == {}


def test_parse_args_ignores_unknown_flags() -> None:
    assert _parse_args("--weird foo --status ok") == {"status": "ok"}


# ---------------------------------------------------------------------------
# Snapshot rendering
# ---------------------------------------------------------------------------


def test_format_snapshot_includes_each_source_section(store: DashboardStore) -> None:
    out = _format_snapshot(store.snapshot())
    assert "■ goal" in out
    assert "■ task" in out
    assert "ship X" in out
    assert "write tests" in out


def test_format_snapshot_handles_empty_store() -> None:
    reg = DashboardSourceRegistry()
    store = DashboardStore(registry=reg, archive_dir=None)
    out = _format_snapshot(store.snapshot())
    assert "no entries match" in out


def test_format_snapshot_source_filter(store: DashboardStore) -> None:
    out = _format_snapshot(store.snapshot(), source="goal")
    assert "■ goal" in out
    assert "■ task" not in out


def test_format_section_uses_status_icons() -> None:
    entries = [
        DashboardEntry(id="x:1", source="x", title="p", status="pending"),
        DashboardEntry(id="x:2", source="x", title="i", status="in_progress"),
        DashboardEntry(id="x:3", source="x", title="c", status="completed"),
        DashboardEntry(id="x:4", source="x", title="f", status="failed"),
        DashboardEntry(id="x:5", source="x", title="b", status="blocked"),
    ]
    out = _format_section("x", entries)
    assert "◻" in out  # pending
    assert "◼" in out  # in_progress
    assert "✓" in out  # completed
    assert "✕" in out  # failed
    assert "◆" in out  # blocked


def test_format_section_handles_empty() -> None:
    out = _format_section("x", [])
    assert "(no entries)" in out


# ---------------------------------------------------------------------------
# dashboard_command_call (integration)
# ---------------------------------------------------------------------------


def test_dashboard_command_call_with_no_args_returns_all(ctx: CommandContext) -> None:
    res = dashboard_command_call("", ctx)
    assert res.type == "text"
    assert "■ goal" in res.value
    assert "■ task" in res.value


def test_dashboard_command_call_with_source_shortcut(ctx: CommandContext) -> None:
    res = dashboard_command_call("goal", ctx)
    assert "■ goal" in res.value
    assert "■ task" not in res.value


def test_dashboard_command_call_with_status_flag(ctx: CommandContext) -> None:
    res = dashboard_command_call("--status failed", ctx)
    assert "■ task" in res.value
    assert "fix bug" in res.value
    assert "write tests" not in res.value


def test_dashboard_command_call_with_id_flag_missing(ctx: CommandContext) -> None:
    res = dashboard_command_call("--id missing", ctx)
    assert "No dashboard entry" in res.value


def test_dashboard_command_call_with_id_flag_found(ctx: CommandContext) -> None:
    res = dashboard_command_call("--id goal:t1", ctx)
    assert "ship X" in res.value


def test_dashboard_command_call_source_flag_takes_priority(ctx: CommandContext) -> None:
    res = dashboard_command_call("goal --source task", ctx)
    # --source wins over positional, so only task should appear.
    assert "■ task" in res.value
    assert "■ goal" not in res.value


@pytest.mark.asyncio
async def test_dashboard_command_run_with_no_args_returns_all(ctx: CommandContext) -> None:
    result = await DASHBOARD_COMMAND.run("", ctx)
    assert isinstance(result, InteractiveOutcome)
    assert "■ goal" in (result.message or "")
    assert "■ task" in (result.message or "")


@pytest.mark.asyncio
async def test_dashboard_command_run_scrollable_when_long(ctx: CommandContext) -> None:
    result = await DASHBOARD_COMMAND.run("", ctx)
    assert isinstance(result, InteractiveOutcome)
    assert result.scrollable is True


@pytest.mark.asyncio
async def test_dashboard_command_run_not_scrollable_when_empty(
    tmp_path: Path,
) -> None:
    from extensions.agent_dashboard import get_default_store

    get_default_store().registry.clear()
    ctx = CommandContext(workspace_root=tmp_path, cwd=tmp_path)
    result = await DASHBOARD_COMMAND.run("", ctx)
    assert isinstance(result, InteractiveOutcome)
    assert result.scrollable is False


def test_dashboard_command_call_uses_fallback_store_when_no_app_state(tmp_path: Path, store: DashboardStore) -> None:
    """A context with no app_state should still resolve a store."""
    ctx = CommandContext(workspace_root=tmp_path, cwd=tmp_path)
    ctx.dashboard_store = store
    res = dashboard_command_call("", ctx)
    assert "■ goal" in res.value


def test_dashboard_command_call_handles_no_sources(tmp_path: Path) -> None:
    from extensions.agent_dashboard import get_default_store

    get_default_store().registry.clear()
    ctx = CommandContext(workspace_root=tmp_path, cwd=tmp_path)
    res = dashboard_command_call("", ctx)
    assert "No data sources" in res.value
