#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for the current /lkb board host integration.

Covers the flag-off rejection and the flag-on board render through the
real Graph Store.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clawcodex_ext.command_system import lkb_command  # noqa: F401  (ensures import side effects)
from clawcodex_ext.command_system.lkb_command import (
    LkbCommandOutcome,
    _lkb_audit,
    _lkb_board,
    _lkb_call,
    _lkb_plan,
)


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HOME + CLAWCODEX_HOME to a temp dir (mirrors lkb conftest)."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CLAWCODEX_HOME", str(home))
    return home


def _tool_ctx_with_tasks(tasks: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        tasks=tasks or {},
        session_id="command-session",
        lkb_plan_id=None,
    )


def test_lkb_board_rejects_when_plan_graph_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With LKB_PLAN_GRAPH off, no LKB-owned projection is rendered."""
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: False)
    ctx = _tool_ctx_with_tasks({})
    out = _lkb_board(ctx, "")
    assert "requires the LKB_PLAN_GRAPH feature flag" in out.text
    assert "No tasks found" not in out.text
    assert out.success is False


def test_lkb_board_renders_when_enabled(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With LKB_PLAN_GRAPH on, /lkb board renders the ASCII board."""
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: True)
    from lkb.application import LkbApplicationService
    from lkb.commands import GraphCommand
    from lkb.plan_graph import plan_command_dispatcher
    from lkb.repository import JsonFileLkbRepository

    # Point get_repository at our temp home and seed a board with one task.
    test_repo = JsonFileLkbRepository(home=tmp_home)
    monkeypatch.setattr("lkb.repository.get_repository", lambda *, home=None: test_repo)
    board_id = test_repo.resolve_board(explicit_id="cmd-board").board_id
    svc = LkbApplicationService(repository=test_repo)
    dispatcher = plan_command_dispatcher()
    svc.execute(
        GraphCommand(
            command_id="c1",
            board_id=board_id,
            actor="a",
            kind="create_task",
            payload={"task_id": "T-1", "subject": "Hello"},
        ),
        validate=dispatcher.validate,
        apply=dispatcher.apply,
    )
    # Make _lkb_board resolve the same board deterministically.
    monkeypatch.setattr(test_repo, "resolve_board", lambda *a, **kw: SimpleNamespace(board_id=board_id))

    ctx = _tool_ctx_with_tasks({})
    ctx.lkb_plan_id = "plan"
    out = _lkb_board(ctx, "")
    assert "LKB BOARD" in out.text
    assert "T-1" in out.text
    assert "READY" in out.text
    assert out.success is True


def test_lkb_call_dispatches_board_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """The /lkb board subcommand is wired into the top-level dispatch."""
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: True)
    # With the merged single flag, the flag-on guard passes; stub the board
    # handler to prove the "board" subcommand routes to it.
    monkeypatch.setattr(
        "clawcodex_ext.command_system.lkb_command._lkb_board",
        lambda tool_ctx, args: LkbCommandOutcome(text="BOARD-HANDLER-CALLED", success=True),
    )
    cmd_ctx = SimpleNamespace(tool_context=_tool_ctx_with_tasks({}))
    result = _lkb_call("board", cmd_ctx)
    assert result.success is True
    assert result.value == "BOARD-HANDLER-CALLED"


def test_lkb_board_compact_invokes_manual_audit_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: True)
    repo = SimpleNamespace(
        compact_audit=Mock(
            return_value=SimpleNamespace(
                segment_id="audit-1",
                file="audit-1.jsonl.gz",
                event_count=10,
                processed_command_count=4,
                validation_run_count=3,
                bytes_before=1000,
                bytes_after=400,
            )
        )
    )
    monkeypatch.setattr(
        "clawcodex_ext.command_system.lkb_command._scope",
        lambda tool_ctx: (repo, "board", "session", SimpleNamespace(plan_id="plan")),
    )
    context = _tool_ctx_with_tasks({})

    output = _lkb_board(context, "compact")

    assert "Compacted 10 events" in output.text
    assert "1000 -> 400 bytes" in output.text
    repo.compact_audit.assert_called_once_with("board", actor="command-session")


def test_lkb_audit_matches_canonical_affected_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = SimpleNamespace(
        events=[
            {
                "type": "command_executed",
                "subject_ref": "plan:task:other",
                "affected_refs": ["plan:task:T-001"],
                "timestamp": "2026-01-01T00:00:00Z",
                "actor": "agent",
                "store_revision": 1,
            }
        ]
    )
    monkeypatch.setattr(
        "lkb.clawcodex_commands._load_envelope",
        lambda _context: (envelope, "board", "plan"),
    )

    outcome = _lkb_audit(_tool_ctx_with_tasks({}), "T-001")

    assert outcome.success is True
    assert "command_executed" in outcome.text


def test_lkb_audit_does_not_match_legacy_short_affected_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = SimpleNamespace(
        events=[
            {
                "type": "command_executed",
                "subject_ref": "plan:task:other",
                "affected_refs": ["T-001"],
            }
        ]
    )
    monkeypatch.setattr(
        "lkb.clawcodex_commands._load_envelope",
        lambda _context: (envelope, "board", "plan"),
    )

    outcome = _lkb_audit(_tool_ctx_with_tasks({}), "T-001")

    assert outcome.success is True
    assert outcome.text == "Task T-001 has no LKB audit events."


def test_lkb_plan_commands_manage_session_binding(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lkb.repository import JsonFileLkbRepository

    repo = JsonFileLkbRepository(home=tmp_home)
    board_id = repo.resolve_board(explicit_id="plan-command-board").board_id
    monkeypatch.setattr("lkb.repository.get_repository", lambda *, home=None: repo)
    monkeypatch.setattr(
        repo,
        "resolve_board",
        lambda *args, **kwargs: SimpleNamespace(board_id=board_id),
    )
    ctx = _tool_ctx_with_tasks({})

    current = _lkb_plan(ctx, "current")
    first_plan_id = ctx.lkb_plan_id
    created = _lkb_plan(ctx, "new Release plan")
    second_plan_id = ctx.lkb_plan_id
    listed = _lkb_plan(ctx, "list")
    suspended = _lkb_plan(ctx, "suspend")
    reopened = _lkb_plan(ctx, f"reopen {second_plan_id}")
    rebound = _lkb_plan(ctx, f"use {first_plan_id}")

    assert first_plan_id in current.text
    assert second_plan_id != first_plan_id
    assert "Release plan" in created.text
    assert first_plan_id in listed.text and second_plan_id in listed.text
    assert "[suspended]" in suspended.text
    assert "[active]" in reopened.text
    assert first_plan_id in rebound.text
