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

"""Tests for ``clawcodex_ext.dreaming.service``.

Covers the gate chain: enabled → time → scan throttle → session →
lock. Uses the built-in stub runner and a tmp-path memory dir to
keep the tests hermetic.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from clawcodex_ext.dreaming import (
    DreamConfig,
    execute_auto_dream,
    init_auto_dream,
    is_auto_dream_enabled,
    kill_dream_task,
    record_consolidation,
    set_dream_config,
)
from clawcodex_ext.dreaming.lock import HOLDER_STALE_MS, LOCK_FILE_NAME
from clawcodex_ext.dreaming.runner import (
    DreamRunResult,
    set_dream_runner_factory,
)
from clawcodex_ext.task_registry import RuntimeTaskRegistry


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Wipe config + runner factory + memory dir between tests.

    Also pins both ``project_transcript_dir`` references (one in
    ``service.py``, one in ``lock.py``) to a tmp dir so tests don't
    read the user's real ``~/.clawcodex/sessions/`` directory.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    monkeypatch.setenv("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", str(tmp_path))
    monkeypatch.delenv("CLAWCODEX_DISABLE_AUTO_DREAM", raising=False)
    monkeypatch.delenv("CLAWCODEX_KAIROS", raising=False)
    set_dream_config(DreamConfig(min_hours=24.0, min_sessions=5))
    set_dream_runner_factory(None)
    monkeypatch.setattr(
        "clawcodex_ext.dreaming.service.project_transcript_dir",
        lambda *_a, **_kw: str(sessions_dir),
    )
    monkeypatch.setattr(
        "clawcodex_ext.dreaming.lock.project_transcript_dir",
        lambda *_a, **_kw: str(sessions_dir),
    )
    # Wipe the lock file between tests so the time gate state is clean.
    lock_path = tmp_path / LOCK_FILE_NAME
    if lock_path.exists():
        lock_path.unlink()
    yield  # type: ignore[misc]
    set_dream_runner_factory(None)


def _init(*, force_min_hours: float = 0.0, force_min_sessions: int = 0) -> RuntimeTaskRegistry:
    """Initialize the service with permissive gates for testing."""
    reg = RuntimeTaskRegistry()
    init_auto_dream(
        config=DreamConfig(min_hours=force_min_hours, min_sessions=force_min_sessions),
        registry=reg,
    )
    return reg


# ---------------------------------------------------------------------------
# Gate chain
# ---------------------------------------------------------------------------


def test_disabled_gate_short_circuits(tmp_path: Path) -> None:
    os.environ["CLAWCODEX_DISABLE_AUTO_DREAM"] = "1"
    reg = _init(force_min_hours=0, force_min_sessions=0)
    asyncio.run(execute_auto_dream(registry=reg))
    # No task registered.
    assert reg.by_type("dream") == []


def test_kairos_active_short_circuits(tmp_path: Path) -> None:
    os.environ["CLAWCODEX_KAIROS"] = "1"
    reg = _init(force_min_hours=0, force_min_sessions=0)
    asyncio.run(execute_auto_dream(registry=reg))
    assert reg.by_type("dream") == []


def test_time_gate_blocks_when_recent(tmp_path: Path) -> None:
    # Lock file written 1h ago — less than the default 24h.
    lock_path = tmp_path / LOCK_FILE_NAME
    lock_path.write_text(str(os.getpid()))
    one_hour_ago = time.time() - 3600
    os.utime(lock_path, (one_hour_ago, one_hour_ago))
    reg = _init(force_min_hours=24.0, force_min_sessions=0)
    asyncio.run(execute_auto_dream(registry=reg))
    assert reg.by_type("dream") == []


def test_session_gate_blocks_when_too_few(tmp_path: Path) -> None:
    # No sessions touched since last consolidation.
    reg = _init(force_min_hours=0, force_min_sessions=5)
    asyncio.run(execute_auto_dream(registry=reg))
    assert reg.by_type("dream") == []


def test_lock_already_held_blocks(tmp_path: Path) -> None:
    # Pre-acquire the lock with a FOREIGN live PID, fresh mtime → blocked.
    # ``os.getppid()`` is the test runner's parent — guaranteed alive
    # and never our own PID (a self-PID would self-acquire per /100.4).
    lock_path = tmp_path / LOCK_FILE_NAME
    lock_path.write_text(str(os.getppid()))
    now = time.time()
    os.utime(lock_path, (now, now))
    reg = _init(force_min_hours=0, force_min_sessions=0)
    asyncio.run(execute_auto_dream(registry=reg))
    assert reg.by_type("dream") == []


# ---------------------------------------------------------------------------
# Happy path + post-conditions
# ---------------------------------------------------------------------------


def test_happy_path_registers_dream_task_and_completes(tmp_path: Path) -> None:
    seen: list[str] = []

    def factory():
        def runner(prompt, on_message):
            seen.append(prompt)
            if on_message is not None:
                on_message(text="consolidated", tool_use_count=0, touched_paths=[])
            return DreamRunResult(
                files_touched=["MEMORY.md"],
                usage={"output_tokens": 17},
                summary="updated MEMORY.md",
            )

        return runner

    set_dream_runner_factory(factory)
    reg = _init(force_min_hours=0, force_min_sessions=0)
    asyncio.run(execute_auto_dream(registry=reg))
    tasks = reg.by_type("dream")
    assert len(tasks) == 1
    state = tasks[0]
    assert state.status == "completed"
    assert state.phase == "starting"  # no real touch in the stub
    assert state.sessions_reviewing == 0
    assert state.turns == [{"text": "consolidated", "tool_use_count": 0}]
    # Prompt was built and forwarded.
    assert len(seen) == 1
    assert "Memory Consolidation" in seen[0]


def test_force_skips_all_gates(tmp_path: Path) -> None:
    os.environ["CLAWCODEX_DISABLE_AUTO_DREAM"] = "1"
    reg = _init(force_min_hours=24.0, force_min_sessions=99)
    asyncio.run(execute_auto_dream(registry=reg, force=True))
    assert len(reg.by_type("dream")) == 1


def test_exclude_current_session_from_session_gate(tmp_path: Path) -> None:
    # The autouse fixture pins project_transcript_dir to
    # tmp_path / "sessions" — read it back to create fixtures there.
    from clawcodex_ext.dreaming.service import project_transcript_dir

    proj = Path(project_transcript_dir())
    current = proj / "current_session"
    current.mkdir()
    now = time.time()
    os.utime(current, (now, now))

    # If we exclude "current_session" but it's the only one, we need
    # at least 2 more sessions to satisfy min_sessions=2.
    for i in range(3):
        other = proj / f"other_{i}"
        other.mkdir()
        os.utime(other, (now, now))

    reg = _init(force_min_hours=0, force_min_sessions=2)
    asyncio.run(
        execute_auto_dream(
            registry=reg,
            current_session_id="current_session",
        )
    )
    # Without exclusion, count would be 4 (≥2) — task fires either way.
    # With exclusion, count is 3 (still ≥2) — task fires.
    # The test asserts the fire (the exclusion logic is exercised in
    # the "count == 1" case below).
    assert len(reg.by_type("dream")) == 1


def test_exclude_current_session_keeps_gate_closed(tmp_path: Path) -> None:
    """If the only fresh session is the current one, the gate stays closed."""
    from clawcodex_ext.dreaming.service import project_transcript_dir

    proj = Path(project_transcript_dir())
    current = proj / "current_session"
    current.mkdir()
    now = time.time()
    os.utime(current, (now, now))

    reg = _init(force_min_hours=0, force_min_sessions=1)
    asyncio.run(
        execute_auto_dream(
            registry=reg,
            current_session_id="current_session",
        )
    )
    # min_sessions=1 not met (only "current_session" present, excluded).
    assert reg.by_type("dream") == []


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_runner_unavailable_fails_task_and_rewinds_lock(tmp_path: Path) -> None:
    def bad_factory():
        def runner(_prompt, _on_message):
            raise RuntimeError("LLM down")

        return runner

    set_dream_runner_factory(bad_factory)
    reg = _init(force_min_hours=0, force_min_sessions=0)
    asyncio.run(execute_auto_dream(registry=reg))
    tasks = reg.by_type("dream")
    assert len(tasks) == 1
    assert tasks[0].status == "failed"
    # Lock was rewound — no lock file should exist (prior mtime was 0).
    assert not (tmp_path / LOCK_FILE_NAME).exists()


def test_kill_during_run_marks_killed_no_double_rewind(tmp_path: Path) -> None:
    """Direct test of kill_dream_task: register a task with a known
    prior mtime, then call kill_dream_task. Verifies the task flips
    to ``killed`` and the lock is rewound (mtime → prior).
    """
    from src.tasks.dream import register_dream_task

    reg = RuntimeTaskRegistry()
    init_auto_dream(registry=reg)

    # Pre-stage the lock file as if the runner had acquired it.
    lock_path = tmp_path / LOCK_FILE_NAME
    lock_path.write_text(str(os.getpid()))
    now = time.time()
    os.utime(lock_path, (now, now))

    task_id = register_dream_task(
        sessions_reviewing=0,
        prior_mtime=int(now * 1000),
        registry=reg,
    )
    kill_dream_task(task_id, reg)

    state = reg.get(task_id)
    assert state.status == "killed"
    # The lock file's mtime should have been rewound to the prior mtime
    # (the same one we stashed on the state).
    rewound_ms = int(lock_path.stat().st_mtime * 1000)
    assert abs(rewound_ms - int(now * 1000)) < 1500


def test_kill_dream_task_no_op_on_completed(tmp_path: Path) -> None:
    """kill_dream_task on an already-terminal task is a no-op
    (no double rewind).
    """
    from src.tasks.dream import register_dream_task

    reg = RuntimeTaskRegistry()
    init_auto_dream(registry=reg)

    lock_path = tmp_path / LOCK_FILE_NAME
    lock_path.write_text(str(os.getpid()))
    now = time.time()
    os.utime(lock_path, (now, now))

    task_id = register_dream_task(
        sessions_reviewing=0,
        prior_mtime=int(now * 1000),
        registry=reg,
    )
    # Mark the task completed BEFORE kill.
    from src.tasks.dream import complete_dream_task

    complete_dream_task(task_id, reg)

    kill_dream_task(task_id, reg)  # must not raise, must not change anything

    state = reg.get(task_id)
    assert state.status == "completed"  # still completed, not killed
    # The lock mtime must NOT have been touched (kill is a no-op on
    # terminal state). Allow 1s slop for filesystem precision.
    mtime_after = int(lock_path.stat().st_mtime * 1000)
    assert abs(mtime_after - int(now * 1000)) < 1500


# ---------------------------------------------------------------------------
# Manual /dream entry
# ---------------------------------------------------------------------------


def test_record_consolidation_optimistic_stamp(tmp_path: Path) -> None:
    record_consolidation()
    lock_path = tmp_path / LOCK_FILE_NAME
    assert lock_path.exists()
    assert int(lock_path.read_text()) == os.getpid()


def test_is_auto_dream_enabled_default_true(tmp_path: Path) -> None:
    # No env override, auto-memory enabled by default.
    assert is_auto_dream_enabled() is True


def test_is_auto_dream_enabled_env_off(tmp_path: Path) -> None:
    os.environ["CLAWCODEX_DISABLE_AUTO_DREAM"] = "1"
    assert is_auto_dream_enabled() is False


# ---------------------------------------------------------------------------
# Phase B — TTL active cleanup wired into the service gate chain
# ---------------------------------------------------------------------------


def test_stale_lock_is_force_released_before_lock_gate(tmp_path: Path) -> None:
    """When a previous consolidator crashed and left a stale lock,
    the service's gate chain must ``force_release_if_stale`` before
    the in-band lock acquire. A successful dream run is observable
    proof that the cleanup path actually ran.
    """
    seen: list[str] = []

    def factory():
        def runner(prompt, on_message):
            seen.append(prompt)
            if on_message is not None:
                on_message(text="consolidated", tool_use_count=0, touched_paths=[])
            return DreamRunResult(
                files_touched=["MEMORY.md"],
                usage={"output_tokens": 17},
                summary="updated MEMORY.md",
            )

        return runner

    set_dream_runner_factory(factory)
    # Stage a stale lock: live holder PID + TTL+60s old mtime. Without
    # Phase B this would block the lock gate forever; with Phase B
    # ``force_release_if_stale`` unlinks it on entry.
    lock_path = tmp_path / LOCK_FILE_NAME
    lock_path.write_text(str(os.getppid()), encoding="utf-8")
    stale_seconds = HOLDER_STALE_MS / 1000 + 60
    stale_mtime = time.time() - stale_seconds
    os.utime(lock_path, (stale_mtime, stale_mtime))

    reg = _init(force_min_hours=0, force_min_sessions=0)
    asyncio.run(execute_auto_dream(registry=reg))

    # Dream task was registered + completed — the TTL-sweep path
    # successfully reclaimed the stale lock for the next pass.
    tasks = reg.by_type("dream")
    assert tasks, "stale lock should have been force-released; expected a dream task"
    assert tasks[0].status == "completed"


def test_fresh_lock_alive_pid_blocks_unchanged(tmp_path: Path) -> None:
    """Phase A path is preserved: fresh mtime + alive holder → blocked,
    even with the new force_release_if_stale call wired in.
    """
    lock_path = tmp_path / LOCK_FILE_NAME
    lock_path.write_text(str(os.getppid()), encoding="utf-8")
    now = time.time()
    os.utime(lock_path, (now, now))

    reg = _init(force_min_hours=0, force_min_sessions=0)
    asyncio.run(execute_auto_dream(registry=reg))

    assert reg.by_type("dream") == []
