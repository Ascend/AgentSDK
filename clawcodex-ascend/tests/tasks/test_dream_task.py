#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for ``src.tasks.dream`` — F-100.

Covers the state machine + lifecycle helpers in
:mod:`src.tasks.dream.dream_task`. Mirrors the shape of
``tests/tasks/test_task_registry.py`` (registry round-trip) and the
``local_agent`` lifecycle tests (mutator returns + status flip).
"""

from __future__ import annotations

import asyncio
import time


from clawcodex_ext.task_registry import RuntimeTaskRegistry
from src.tasks.dream import (
    MAX_DREAM_TURNS,
    DreamTask,
    DreamTaskState,
    add_dream_turn,
    complete_dream_task,
    fail_dream_task,
    is_dream_task,
    is_dream_task_terminal,
    register_dream_task,
    rollback_dream_lock_after_kill,
)
from clawcodex_ext.tasks_core import is_terminal_task_status


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_register_dream_task_upserts_with_id_and_running_status() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(
        sessions_reviewing=7,
        prior_mtime=12345,
        registry=reg,
    )
    assert task_id.startswith("d")  # TaskType "dream" → "d" prefix
    assert len(task_id) == 9  # "d" + 8 base36 chars
    state = reg.get(task_id)
    assert state is not None
    assert isinstance(state, DreamTaskState)
    assert state.type == "dream"
    assert state.status == "running"
    assert state.phase == "starting"
    assert state.sessions_reviewing == 7
    assert state.prior_mtime == 12345
    assert state.files_touched == []
    assert state.turns == []


def test_register_dream_task_id_is_unique_under_load() -> None:
    reg = RuntimeTaskRegistry()
    ids = {register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg) for _ in range(50)}
    # 50 random 36^8 ids — collision probability is ~ 50^2 / (2 * 36^8)
    # ≈ 4e-13. Assert no duplicates.
    assert len(ids) == 50


def test_is_dream_task_type_guard() -> None:
    state = DreamTaskState(
        id="d12345678",
        type="dream",
        status="running",
        description="d",
        start_time=0.0,
        output_file="",
    )
    assert is_dream_task(state) is True
    assert is_dream_task(None) is False
    assert is_dream_task({"type": "dream"}) is False


# ---------------------------------------------------------------------------
# add_dream_turn — phase flip + dedup + cap
# ---------------------------------------------------------------------------


def test_add_dream_turn_flips_phase_on_first_touch() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    # No path touched — phase stays "starting".
    add_dream_turn(
        task_id,
        text="orient",
        tool_use_count=1,
        touched_paths=[],
        registry=reg,
    )
    state = reg.get(task_id)
    assert state.phase == "starting"
    assert state.turns[-1] == {"text": "orient", "tool_use_count": 1}

    # First real touch flips phase to "updating".
    add_dream_turn(
        task_id,
        text="write MEMORY.md",
        tool_use_count=1,
        touched_paths=["MEMORY.md"],
        registry=reg,
    )
    state = reg.get(task_id)
    assert state.phase == "updating"
    assert "MEMORY.md" in state.files_touched


def test_add_dream_turn_dedupes_touched_paths() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    add_dream_turn(
        task_id,
        text="",
        tool_use_count=1,
        touched_paths=["a.md"],
        registry=reg,
    )
    add_dream_turn(
        task_id,
        text="",
        tool_use_count=1,
        touched_paths=["a.md", "b.md"],
        registry=reg,
    )
    state = reg.get(task_id)
    assert state.files_touched == ["a.md", "b.md"]


def test_add_dream_turn_skips_pure_noop() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    add_dream_turn(task_id, text="", tool_use_count=0, touched_paths=[], registry=reg)
    state = reg.get(task_id)
    assert state.turns == []  # noop dropped
    assert state.phase == "starting"


def test_add_dream_turn_caps_at_max_dream_turns() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    for i in range(MAX_DREAM_TURNS + 10):
        add_dream_turn(
            task_id,
            text=f"t{i}",
            tool_use_count=0,
            touched_paths=[f"file_{i}.md"],
            registry=reg,
        )
    state = reg.get(task_id)
    assert len(state.turns) == MAX_DREAM_TURNS
    # Most-recent N turns kept — first one dropped.
    assert state.turns[0]["text"] == f"t{10}"
    assert state.turns[-1]["text"] == f"t{MAX_DREAM_TURNS + 9}"


def test_add_dream_turn_noop_on_terminal() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    complete_dream_task(task_id, reg)
    add_dream_turn(
        task_id,
        text="late",
        tool_use_count=1,
        touched_paths=["x.md"],
        registry=reg,
    )
    state = reg.get(task_id)
    assert state.turns == []
    assert "x.md" not in state.files_touched


# ---------------------------------------------------------------------------
# Terminal transitions
# ---------------------------------------------------------------------------


def test_complete_dream_task_flips_status_and_stamps_end_time() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    complete_dream_task(task_id, reg)
    state = reg.get(task_id)
    assert state.status == "completed"
    assert state.end_time is not None
    assert state.abort_event is None
    assert is_dream_task_terminal(state) is True
    assert is_terminal_task_status(state.status) is True


def test_complete_dream_task_is_idempotent() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    complete_dream_task(task_id, reg)
    first_end = reg.get(task_id).end_time
    time.sleep(0.001)
    complete_dream_task(task_id, reg)
    second_end = reg.get(task_id).end_time
    assert first_end == second_end  # second call is a no-op


def test_fail_dream_task_flips_status() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    fail_dream_task(task_id, reg)
    state = reg.get(task_id)
    assert state.status == "failed"
    assert is_dream_task_terminal(state) is True


# ---------------------------------------------------------------------------
# Kill / lock rollback
# ---------------------------------------------------------------------------


def test_rollback_dream_lock_after_kill_returns_prior_mtime() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=99999, registry=reg)
    captured = rollback_dream_lock_after_kill(task_id, reg)
    assert captured == 99999
    state = reg.get(task_id)
    assert state.status == "killed"
    assert state.end_time is not None


def test_rollback_dream_lock_after_kill_returns_none_on_already_terminal() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    complete_dream_task(task_id, reg)
    captured = rollback_dream_lock_after_kill(task_id, reg)
    assert captured is None  # already terminal — no state change


def test_dream_task_adapter_kill_dispatches_to_rollback() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=42, registry=reg)
    adapter = DreamTask()
    assert adapter.name == "DreamTask"
    assert adapter.type == "dream"
    # Async kill — drive the coroutine to completion.
    asyncio.run(adapter.kill(task_id, reg))
    state = reg.get(task_id)
    assert state.status == "killed"


def test_dream_task_adapter_kill_no_op_on_terminal() -> None:
    reg = RuntimeTaskRegistry()
    task_id = register_dream_task(sessions_reviewing=0, prior_mtime=0, registry=reg)
    fail_dream_task(task_id, reg)
    adapter = DreamTask()
    asyncio.run(adapter.kill(task_id, reg))
    state = reg.get(task_id)
    assert state.status == "failed"  # unchanged from prior fail


# ---------------------------------------------------------------------------
# Centralized registration
# ---------------------------------------------------------------------------


def test_dream_task_registered_via_src_tasks_init() -> None:
    from clawcodex_ext.task_registry import get_task_by_type
    from src.tasks import dream  # noqa: F401  (trigger registration)

    impl = get_task_by_type("dream")
    assert impl is not None
    assert impl.name == "DreamTask"


def test_dream_task_idempotent_registration() -> None:
    from clawcodex_ext.task_registry import get_all_tasks, register_task
    from src.tasks.dream import DreamTask

    before = len(get_all_tasks())
    register_task(DreamTask())
    register_task(DreamTask())
    after = len(get_all_tasks())
    assert after == before
