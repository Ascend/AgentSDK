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

"""End-to-end tests for the dreaming subsystem.

Covers the *full* integration paths that a real session would take:

* ``/dream run`` slash command → dispatch → ``manual_dream`` →
  ``execute_auto_dream`` → runner → task registry (manual trigger).
* Permanent cron install + wire handler + manual fire → dream runs
  (scheduled trigger).
* ``install_and_wire_dream`` is idempotent on startup: a missing recurring
  dream task is registered as a permanent cron, and later starts do not
  register duplicates.
* Real :class:`CronScheduler` tick with a backdated ``next_fire_at``
  fires the dream task via the wire handler.
* A custom runner factory records calls — proves the runner is
  reached end-to-end, not stubbed out by an early gate or exception.

Hermetic via autouse fixture: pins ``project_transcript_dir`` to a
tmp path so session scans in the gate chain never read the real
``~/.clawcodex/sessions/`` directory.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from clawcodex_ext.cron_system.scheduler import CronScheduler
from clawcodex_ext.cron_system.tasks import read_cron_tasks, write_cron_tasks
from clawcodex_ext.dreaming import (
    DREAM_PERMANENT_TASK_ID,
    init_auto_dream,
    install_and_wire_dream,
    manual_dream,
)
from clawcodex_ext.dreaming.config import DreamConfig
from clawcodex_ext.dreaming.lock import LOCK_FILE_NAME
from clawcodex_ext.dreaming.runner import (
    DreamRunResult,
    set_dream_runner_factory,
)
from clawcodex_ext.task_registry import RuntimeTaskRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reset state between tests."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    monkeypatch.setenv("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", str(tmp_path))
    monkeypatch.delenv("CLAWCODEX_DISABLE_AUTO_DREAM", raising=False)
    monkeypatch.delenv("CLAWCODEX_KAIROS", raising=False)

    from clawcodex_ext.dreaming import service as _service
    from clawcodex_ext.dreaming.config import DEFAULT_DREAM_CONFIG, set_dream_config

    set_dream_config(DEFAULT_DREAM_CONFIG)
    set_dream_runner_factory(None)

    monkeypatch.setattr(
        "clawcodex_ext.dreaming.service.project_transcript_dir",
        lambda *_a, **_kw: str(sessions_dir),
    )
    monkeypatch.setattr(
        "clawcodex_ext.dreaming.lock.project_transcript_dir",
        lambda *_a, **_kw: str(sessions_dir),
    )

    _service._runner = None

    lock_path = tmp_path / LOCK_FILE_NAME
    if lock_path.exists():
        lock_path.unlink()

    yield  # type: ignore[misc]

    set_dream_runner_factory(None)
    _service._runner = None


class _FakeLock:
    """Stand-in for ``CronTaskLock`` — the real one tries to acquire
    a file lock; in tests we already own the workspace.
    """

    def __init__(self, acquired: bool = True) -> None:
        self._acquired = acquired

    @property
    def is_acquired(self) -> bool:
        return self._acquired


def _recording_factory() -> tuple[list[str], Any]:
    """Return a ``(seen_prompts, factory)`` pair that records every
    prompt the runner is called with. Drives the test end-to-end
    (real prompt build + real gate chain) without going through the
    LLM.
    """
    seen: list[str] = []

    def factory():
        def runner(prompt: str, on_message):
            seen.append(prompt)
            if on_message is not None:
                on_message(text="recorded", tool_use_count=0, touched_paths=[])
            return DreamRunResult(
                files_touched=["MEMORY.md"],
                usage={"output_tokens": 1},
                summary="e2e recorded",
            )

        return runner

    return seen, factory


def _wait_for(predicate, timeout: float = 2.0, step: float = 0.02) -> bool:
    """Poll ``predicate()`` until True or timeout. Returns the final value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


# ---------------------------------------------------------------------------
# E2E #2 — manual_dream end-to-end (force=True path)
# ---------------------------------------------------------------------------


def test_e2e_manual_dream_force_bypasses_all_gates(tmp_path: Path) -> None:
    """``manual_dream`` end-to-end with force=True bypasses the time
    gate (locked 1h ago) and the session gate (no sessions) — proves
    the fire path reaches the runner regardless of state.
    """
    seen, factory = _recording_factory()
    set_dream_runner_factory(factory)
    reg = RuntimeTaskRegistry()
    # Deliberately tight min_hours + high min_sessions so force is
    # the only way through.
    init_auto_dream(
        config=DreamConfig(min_hours=24.0, min_sessions=99),
        registry=reg,
    )

    # Stamp the lock to a recent time so the time gate would close.
    lock_path = tmp_path / LOCK_FILE_NAME
    lock_path.write_text(str(os.getpid()))
    one_hour_ago = time.time() - 3600
    os.utime(lock_path, (one_hour_ago, one_hour_ago))

    manual_dream()
    assert seen, "manual_dream force=True should have called the runner"
    tasks = reg.by_type("dream")
    assert len(tasks) == 1
    assert tasks[0].status == "completed"


# ---------------------------------------------------------------------------
# E2E #3 — permanent cron install + wire + manual fire
# ---------------------------------------------------------------------------


def test_e2e_cron_fire_triggers_dream_via_wire_handler(tmp_path: Path) -> None:
    """Install the permanent dream task, wire the fire handler, and
    manually fire the cron task. The wire handler intercepts and
    calls ``execute_auto_dream``; the runner is reached end-to-end.
    """
    seen, factory = _recording_factory()
    set_dream_runner_factory(factory)
    reg = RuntimeTaskRegistry()
    init_auto_dream(
        config=DreamConfig(min_hours=0.0, min_sessions=0),
        registry=reg,
    )

    # Original outbox-style handler — must NOT be called for the dream task.
    outbox: list[dict] = []

    def original(task, run) -> None:
        outbox.append({"id": task.id, "run": run})

    scheduler = CronScheduler(
        tmp_path,
        on_fire=lambda p: outbox.append({"prompt": p}),
        on_fire_task=original,
    )
    task, created = install_and_wire_dream(tmp_path, scheduler, registry=reg)
    assert created is True
    assert task.id == DREAM_PERMANENT_TASK_ID

    # Build a minimal CronRun stand-in (the wire handler only reads .id).
    @dataclass
    class _FakeRun:
        id: str = "run-e2e-3"

    scheduler.on_fire_task(task, _FakeRun())

    # Runner reached end-to-end.
    assert seen, "wire handler should have invoked the dream runner"
    # Original handler was NOT called (wire handler swallowed the fire).
    assert outbox == [], f"original on_fire_task should not be called for dream task; got {outbox!r}"
    # Task on the registry, completed.
    tasks = reg.by_type("dream")
    assert len(tasks) == 1
    assert tasks[0].status == "completed"


# ---------------------------------------------------------------------------
# E2E #4 — install_and_wire_dream is idempotent on startup
# ---------------------------------------------------------------------------


def test_e2e_install_is_idempotent_across_starts(tmp_path: Path) -> None:
    """Simulate two app starts. The first installs the dream
    permanent task; the second sees it and is a no-op. The task list
    ends up with exactly one permanent dream task.
    """
    # Start 1 — install.
    s1 = CronScheduler(tmp_path, on_fire=lambda p: None)
    task1, created1 = install_and_wire_dream(tmp_path, s1)
    assert created1 is True

    # Start 2 — same workspace, different scheduler. The install
    # is a no-op (same cron + same prompt → same id).
    s2 = CronScheduler(tmp_path, on_fire=lambda p: None)
    task2, created2 = install_and_wire_dream(tmp_path, s2)
    assert created2 is False
    assert task2.id == task1.id

    # Exactly one permanent dream task in the workspace.
    tasks = read_cron_tasks(tmp_path)
    dream_tasks = [t for t in tasks if t.id == DREAM_PERMANENT_TASK_ID]
    assert len(dream_tasks) == 1
    assert dream_tasks[0].permanent is True


# ---------------------------------------------------------------------------
# E2E #5 — real CronScheduler tick fires the dream task
# ---------------------------------------------------------------------------


def test_e2e_real_scheduler_tick_fires_dream(tmp_path: Path) -> None:
    """Real CronScheduler tick (with a backdated ``next_fire_at``)
    fires the dream task. The wire handler intercepts and runs dream
    end-to-end. The test is async on the tick but deterministic
    thanks to the short poll interval + the backdated next_fire_at.
    """
    seen, factory = _recording_factory()
    set_dream_runner_factory(factory)
    reg = RuntimeTaskRegistry()
    init_auto_dream(
        config=DreamConfig(min_hours=0.0, min_sessions=0),
        registry=reg,
    )

    # Install the dream task with the real installer.
    outbox: list[Any] = []
    scheduler = CronScheduler(
        tmp_path,
        on_fire=outbox.append,
        on_fire_task=lambda t, r: outbox.append(t),
    )
    task, _created = install_and_wire_dream(tmp_path, scheduler, registry=reg)

    # Backdate the dream task's next_fire_at to NOW-1s so the very
    # next check_once() tick picks it up.
    tasks = read_cron_tasks(tmp_path)
    backdated = [replace(t, next_fire_at=int(time.time() * 1000) - 1000) if t.id == task.id else t for t in tasks]
    write_cron_tasks(tmp_path, backdated)

    # Hand the scheduler a fake lock (the real CronTaskLock would
    # try to acquire a file lock; in tests we already own the workspace).
    scheduler._lock = _FakeLock(acquired=True)  # type: ignore[attr-defined]

    # One tick — should fire the backdated dream task.
    scheduler.check_once()

    # Runner reached.
    assert seen, "real scheduler tick should have fired the dream task"
    # Outbox untouched (wire handler swallowed the fire).
    assert outbox == []
    # Task on the registry, completed.
    tasks = reg.by_type("dream")
    assert len(tasks) == 1
    assert tasks[0].status == "completed"
