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

"""Tests for the LKB Read Model + ASCII board (Phase 7, spec §8).

Covers summary counts, badge priority (needs_recheck over blocked),
display-width handling for CJK, width degradation, and a no-ANSI golden
snapshot (LKB-VIEW-002/005/006/007/011).
"""

from __future__ import annotations

from pathlib import Path

from lkb.application import LkbApplicationService
from lkb.ascii_board import display_width, render_board, truncate
from lkb.commands import GraphCommand
from lkb.plan_graph import plan_command_dispatcher
from lkb.read_model import LkbBadge, build_board_view
from lkb.repository import JsonFileLkbRepository


def _setup(tmp_home: Path):
    repo = JsonFileLkbRepository(home=tmp_home)
    board_id = repo.resolve_board(explicit_id="view-test").board_id
    svc = LkbApplicationService(repository=repo)
    dispatcher = plan_command_dispatcher()
    return svc, repo, board_id, dispatcher


_n = 0


def _cid() -> str:
    global _n
    _n += 1
    return f"v-{_n}"


def _exec(svc, dispatcher, kind, board_id, *, payload=None, actor="agent-a", reason=None):
    return svc.execute(
        GraphCommand(
            command_id=_cid(),
            board_id=board_id,
            actor=actor,
            kind=kind,
            payload=payload or {},
            reason=reason,
        ),
        validate=dispatcher.validate,
        apply=dispatcher.apply,
    )


def _complete(svc, dispatcher, board_id, task_id):
    _exec(svc, dispatcher, "claim_task", board_id, payload={"task_id": task_id})
    _exec(svc, dispatcher, "start_task", board_id, payload={"task_id": task_id})
    _exec(svc, dispatcher, "complete_task", board_id, payload={"task_id": task_id})


def test_summary_counts_and_badges(tmp_home: Path) -> None:
    svc, repo, board_id, dispatcher = _setup(tmp_home)
    # T-1 ready, T-2 ready, T-3 blocked (depends on T-1,T-2), T-4 running.
    _exec(svc, dispatcher, "create_task", board_id, payload={"task_id": "T-1", "subject": "API spec"})
    _exec(svc, dispatcher, "create_task", board_id, payload={"task_id": "T-2", "subject": "DB schema"})
    _exec(svc, dispatcher, "create_task", board_id, payload={"task_id": "T-3", "subject": "Service"})
    _exec(svc, dispatcher, "create_task", board_id, payload={"task_id": "T-4", "subject": "Tests"})
    _exec(svc, dispatcher, "add_dependency", board_id, payload={"task_id": "T-3", "depends_on": "T-1"})
    _exec(svc, dispatcher, "add_dependency", board_id, payload={"task_id": "T-3", "depends_on": "T-2"})
    _exec(svc, dispatcher, "claim_task", board_id, payload={"task_id": "T-4"})
    _exec(svc, dispatcher, "start_task", board_id, payload={"task_id": "T-4"})

    env = repo._get_store(board_id).load()
    view = build_board_view(env)
    assert view.summary.ready == 2  # T-1, T-2
    assert view.summary.running == 1  # T-4
    assert view.summary.blocked == 1  # T-3
    assert view.summary.needs_recheck == 0
    badges = {r.task_id: r.badge for r in view.rows}
    assert badges["T-3"] is LkbBadge.BLOCKED
    assert badges["T-4"] is LkbBadge.RUNNING
    # T-3 active blockers include T-1 and T-2.
    t3 = next(r for r in view.rows if r.task_id == "T-3")
    assert set(t3.active_blockers) == {"T-1", "T-2"}


def test_needs_recheck_badge_overrides_blocked(tmp_home: Path) -> None:
    """A completed task reopened upstream shows NEEDS_RECHECK even if blocked."""
    svc, repo, board_id, dispatcher = _setup(tmp_home)
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-1", "subject": "up"},
    )
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-2", "subject": "down"},
    )
    _exec(svc, dispatcher, "add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"})
    _complete(svc, dispatcher, board_id, "T-1")
    _complete(svc, dispatcher, board_id, "T-2")
    _exec(svc, dispatcher, "reopen_task", board_id, payload={"task_id": "T-1"}, reason="changed")

    env = repo._get_store(board_id).load()
    view = build_board_view(env)
    t2 = next(r for r in view.rows if r.task_id == "T-2")
    assert t2.badge is LkbBadge.NEEDS_RECHECK
    assert t2.base_status == "completed"
    assert view.summary.needs_recheck == 1


def test_latest_failed_validation_has_highest_badge_priority(tmp_home: Path) -> None:
    svc, repo, board_id, dispatcher = _setup(tmp_home)
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-1", "subject": "review me"},
    )
    env = repo._get_store(board_id).load()
    env.nodes["T-1"]["payload"]["derived_status"] = "needs_review"
    env.validation_runs["V-failed"] = {
        "validationRunId": "V-failed",
        "subjectRef": {"graph": "plan", "kind": "task", "id": "T-1"},
        "result": "fail",
    }

    row = build_board_view(env).rows[0]

    assert row.badge is LkbBadge.VALIDATION_FAILED


def test_display_width_and_truncate() -> None:
    assert display_width("hello") == 5
    assert display_width("中文") == 4  # two full-width chars
    assert display_width("a中") == 3
    assert truncate("hello world", 8) == "hello..."
    assert truncate("短", 4) == "短"
    assert display_width(truncate("中文测试很长", 6)) <= 6


def test_render_board_golden_no_ansi(tmp_home: Path) -> None:
    svc, repo, board_id, dispatcher = _setup(tmp_home)
    _exec(svc, dispatcher, "create_task", board_id, payload={"task_id": "T-1", "subject": "API spec"})
    _exec(svc, dispatcher, "create_task", board_id, payload={"task_id": "T-2", "subject": "DB schema"})
    _exec(svc, dispatcher, "add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"})
    _exec(svc, dispatcher, "claim_task", board_id, payload={"task_id": "T-1"})

    env = repo._get_store(board_id).load()
    view = build_board_view(env)
    out = render_board(view, width=100)
    # No ANSI escape codes.
    assert "\x1b[" not in out
    # Header + summary + table present.
    assert "LKB BOARD" in out
    assert "Ready" in out
    assert "T-1" in out and "T-2" in out
    assert "READY" in out
    assert "BLOCKED" in out
    # Stable: rendering twice yields identical output.
    assert render_board(view, width=100) == out


def test_render_board_narrow_degrades_to_cards(tmp_home: Path) -> None:
    svc, repo, board_id, dispatcher = _setup(tmp_home)
    _exec(svc, dispatcher, "create_task", board_id, payload={"task_id": "T-1", "subject": "API spec"})
    env = repo._get_store(board_id).load()
    view = build_board_view(env)
    out = render_board(view, width=40)
    # Card mode uses the bracket header form.
    assert "[T-1]" in out
    assert "lkb:" in out
