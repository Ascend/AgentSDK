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

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from clawcodex_ext.cron_system.models import CronJitterConfig
from clawcodex_ext.cron_system.notifications import build_missed_task_notification
from clawcodex_ext.cron_system.runs import read_cron_runs
from clawcodex_ext.cron_system.scheduler import CronScheduler
from clawcodex_ext.cron_system.tasks import (
    add_cron_task,
    read_cron_tasks,
    read_session_cron_tasks,
    write_cron_tasks,
)


def test_check_once_fires_due_one_shot_and_deletes_it(tmp_path) -> None:
    fired: list[str] = []
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="once", recurring=False, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])
    scheduler = CronScheduler(tmp_path, on_fire=fired.append)
    due = scheduler.check_once(at_ms=3_000)
    assert [task.prompt for task in due] == ["once"]
    assert fired == ["once"]
    assert read_cron_tasks(tmp_path) == []


def test_check_once_fires_recurring_and_updates_last_fire(tmp_path) -> None:
    fired: list[str] = []
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])
    scheduler = CronScheduler(tmp_path, on_fire=fired.append)
    scheduler.check_once(at_ms=3_000)
    tasks = read_cron_tasks(tmp_path)
    assert fired == ["ping"]
    assert len(tasks) == 1
    assert tasks[0].last_fired_at == 3_000


def test_check_once_prefers_task_callback_over_prompt_callback(tmp_path) -> None:
    prompts: list[str] = []
    fired = []
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", recurring=False, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])
    scheduler = CronScheduler(
        tmp_path,
        on_fire=prompts.append,
        on_fire_task=lambda task, run: fired.append((task, run)),
    )

    scheduler.check_once(at_ms=3_000)

    assert not prompts
    assert [task.prompt for task, _run in fired] == ["ping"]
    assert fired[0][1].task_id == task.id


def test_check_once_keeps_run_queued_until_external_finalize(tmp_path) -> None:
    """After fire, the run stays in "queued" status. D1 dedup blocks
    subsequent ticks for the same task until the run is externally
    finalized (by the REPL after chat() completes).
    """
    from clawcodex_ext.cron_system.runs import finalize_cron_run

    fired: list[str] = []
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])
    scheduler = CronScheduler(tmp_path, on_fire=fired.append)

    first_due = scheduler.check_once(at_ms=3_000)
    assert [item.id for item in first_due] == [task.id]
    assert fired == ["ping"]
    runs = read_cron_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].status == "queued"

    write_cron_tasks(tmp_path, [replace(read_cron_tasks(tmp_path)[0], next_fire_at=4_000)])
    second_due = scheduler.check_once(at_ms=5_000)
    assert second_due == []
    assert fired == ["ping"]

    finalize_cron_run(tmp_path, runs[0].id, "completed")

    write_cron_tasks(tmp_path, [replace(read_cron_tasks(tmp_path)[0], next_fire_at=6_000)])
    third_due = scheduler.check_once(at_ms=7_000)
    assert [item.id for item in third_due] == [task.id]
    assert fired == ["ping", "ping"]


def test_notify_missed_once_reports_and_removes_due_one_shot(tmp_path) -> None:
    notifications: list[str] = []
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="once", recurring=False, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])
    scheduler = CronScheduler(
        tmp_path,
        on_fire=lambda _prompt: None,
        on_missed=lambda _tasks, message: notifications.append(message),
    )

    missed = scheduler.notify_missed_once(at_ms=3_000)

    assert [task.id for task in missed] == [task.id]
    assert task.id in notifications[0]
    assert read_cron_tasks(tmp_path) == []


def test_missed_notification_uses_safe_fence(tmp_path) -> None:
    task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="contains ``` fence",
        recurring=False,
        created_at=1_000,
    )
    notification = build_missed_task_notification([task])
    assert "````" in notification
    assert task.id in notification


# ---------------------------------------------------------------------------
# Phase D-2: dual-durable scheduler coverage.
# ---------------------------------------------------------------------------


class _FakeLock:
    """Stand-in for CronTaskLock used in tests. The scheduler consults
    ``is_acquired`` (Phase B-2) to gate session-task firing.
    """

    def __init__(self, acquired: bool) -> None:
        self._acquired = acquired

    @property
    def is_acquired(self) -> bool:
        return self._acquired


def _make_scheduler(tmp_path: Any, *, session_store: Any | None, on_fire: Any, lock_acquired: bool) -> CronScheduler:
    scheduler = CronScheduler(tmp_path, on_fire=on_fire, session_store=session_store)
    # Inject a stand-in for CronTaskLock so check_once's lock gate works
    # without going through real file lock I/O.
    scheduler._lock = _FakeLock(lock_acquired)  # type: ignore[attr-defined]
    return scheduler


def test_check_once_fires_due_session_recurring(tmp_path) -> None:
    fired: list[str] = []
    session_store: dict[str, object] = {}
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="session-rec",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    rec = read_session_cron_tasks(session_store)[0]
    # Force the next_fire_at to a past timestamp.
    session_store[rec.id] = replace(rec, next_fire_at=2_000)
    scheduler = _make_scheduler(tmp_path, session_store=session_store, on_fire=fired.append, lock_acquired=True)
    due = scheduler.check_once(at_ms=3_000)
    assert [t.prompt for t in due] == ["session-rec"]
    assert fired == ["session-rec"]
    # run is recorded on disk regardless of task durability
    runs = read_cron_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].status == "queued"
    # session_store's recurring task is kept and rescheduled
    survivors = read_session_cron_tasks(session_store)
    assert len(survivors) == 1
    assert survivors[0].last_fired_at == 3_000
    assert survivors[0].next_fire_at is not None and survivors[0].next_fire_at > 3_000


def test_check_once_fires_due_session_one_shot_and_deletes(tmp_path) -> None:
    fired: list[str] = []
    session_store: dict[str, object] = {}
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="session-once",
        recurring=False,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    task = read_session_cron_tasks(session_store)[0]
    session_store[task.id] = replace(task, next_fire_at=2_000)
    scheduler = _make_scheduler(tmp_path, session_store=session_store, on_fire=fired.append, lock_acquired=True)
    due = scheduler.check_once(at_ms=3_000)
    assert [t.prompt for t in due] == ["session-once"]
    assert fired == ["session-once"]
    # one-shot session task is dropped from the in-memory store
    assert read_session_cron_tasks(session_store) == []


def test_check_once_mixed_durable_and_session(tmp_path) -> None:
    fired: list[str] = []
    session_store: dict[str, object] = {}
    file_task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="file", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(file_task, next_fire_at=2_000)])
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="session",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    sess = read_session_cron_tasks(session_store)[0]
    session_store[sess.id] = replace(sess, next_fire_at=2_000)
    scheduler = _make_scheduler(tmp_path, session_store=session_store, on_fire=fired.append, lock_acquired=True)
    due = scheduler.check_once(at_ms=3_000)
    assert sorted(t.prompt for t in due) == ["file", "session"]
    assert sorted(fired) == ["file", "session"]
    # both rescheduled
    assert read_cron_tasks(tmp_path)[0].last_fired_at == 3_000
    assert read_session_cron_tasks(session_store)[0].last_fired_at == 3_000


def test_session_task_not_fired_when_lock_not_owned(tmp_path) -> None:
    fired: list[str] = []
    session_store: dict[str, object] = {}
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="no-lock-session",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    sess = read_session_cron_tasks(session_store)[0]
    session_store[sess.id] = replace(sess, next_fire_at=2_000)
    # lock_acquired=False -> scheduler should skip session tasks
    scheduler = _make_scheduler(tmp_path, session_store=session_store, on_fire=fired.append, lock_acquired=False)
    due = scheduler.check_once(at_ms=3_000)
    assert due == []
    assert not fired
    # session task is still present, unmodified
    assert read_session_cron_tasks(session_store)[0].last_fired_at is None
    # but file tasks (if any) are still picked up
    file_task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="file", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(file_task, next_fire_at=2_000)])
    due = scheduler.check_once(at_ms=3_000)
    assert [t.prompt for t in due] == ["file"]


def test_fire_callback_exception_marks_run_failed(tmp_path) -> None:
    def boom(prompt: str) -> None:
        raise RuntimeError("upstream glitch")

    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="x", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])
    scheduler = _make_scheduler(tmp_path, session_store=None, on_fire=boom, lock_acquired=True)
    scheduler.check_once(at_ms=3_000)
    runs = read_cron_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].error is not None
    assert "upstream glitch" in runs[0].error
    # Recurring task still rescheduled
    assert read_cron_tasks(tmp_path)[0].last_fired_at == 3_000


def test_get_next_fire_time_includes_session(tmp_path) -> None:
    session_store: dict[str, object] = {}
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="later-session",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    later = read_session_cron_tasks(session_store)[0]
    session_store[later.id] = replace(later, next_fire_at=1_000_000)
    # also a file task with an earlier fire
    file_task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="sooner-file",
        recurring=True,
        created_at=1_000,
    )
    write_cron_tasks(tmp_path, [replace(file_task, next_fire_at=500_000)])
    scheduler = _make_scheduler(
        tmp_path,
        session_store=session_store,
        on_fire=lambda p: None,
        lock_acquired=True,
    )
    assert scheduler.get_next_fire_time() == 500_000


def test_in_flight_dedup_across_file_and_session(tmp_path) -> None:
    """In-flight set must prevent double-fire even when the same task
    id appears in both file and session_store (a defensive check).
    """
    fired: list[str] = []
    session_store: dict[str, object] = {}
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="dup",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    sess = read_session_cron_tasks(session_store)[0]
    session_store[sess.id] = replace(sess, next_fire_at=2_000)
    # also write a file task with the same id
    file_task = replace(sess, next_fire_at=2_000, durable=True)
    write_cron_tasks(tmp_path, [file_task])
    scheduler = _make_scheduler(tmp_path, session_store=session_store, on_fire=fired.append, lock_acquired=True)
    scheduler.check_once(at_ms=3_000)
    # Should fire once, not twice
    assert fired == ["dup"]


def test_daemon_mode_dir_override(tmp_path: Path) -> None:
    """Verify that dir_override redirects all file I/O."""
    import time

    alt_root = tmp_path / "daemon-workspace"
    alt_root.mkdir()

    fired: list[str] = []
    scheduler = CronScheduler(
        workspace_root=tmp_path,  # original root — should be overridden
        on_fire=fired.append,
        dir_override=alt_root,
        lock_identity="daemon-001",
    )
    # __post_init__ should have replaced workspace_root
    assert scheduler.workspace_root == alt_root
    assert scheduler.lock_identity == "daemon-001"

    # Add a task via the normal file path (using alt_root)
    now = int(time.time() * 1000)
    task = add_cron_task(alt_root, cron="* * * * *", prompt="daemon-ping", created_at=now)
    due_task = replace(task, next_fire_at=now - 1000)
    write_cron_tasks(alt_root, [due_task])

    # check_once should find the task in alt_root, not tmp_path
    fired.clear()
    scheduler.check_once()
    assert fired == ["daemon-ping"]

    # tmp_path should have no tasks (all I/O went to alt_root)
    from clawcodex_ext.cron_system.tasks import read_all_cron_tasks

    assert read_all_cron_tasks(tmp_path, None) == []


def test_daemon_mode_no_override_falls_back(tmp_path: Path) -> None:
    """When dir_override and lock_identity are None, existing
    behaviour is preserved.
    """
    fired: list[str] = []
    scheduler = CronScheduler(
        workspace_root=tmp_path,
        on_fire=fired.append,
        dir_override=None,
        lock_identity=None,
    )
    assert scheduler.workspace_root == tmp_path
    assert scheduler.lock_identity is None
    # session_id should still be auto-generated
    assert scheduler.session_id is not None


def test_daemon_mode_async_scheduler(tmp_path: Path) -> None:
    """Verify that AsyncCronScheduler respects dir_override and lock_identity."""
    alt_root = tmp_path / "async-daemon-workspace"
    alt_root.mkdir()

    from clawcodex_ext.cron_system.scheduler import AsyncCronScheduler

    fired: list[str] = []
    scheduler = AsyncCronScheduler(
        workspace_root=tmp_path,
        on_fire=fired.append,
        dir_override=alt_root,
        lock_identity="async-daemon",
    )
    assert scheduler.workspace_root == alt_root
    assert scheduler.lock_identity == "async-daemon"


def test_check_once_skips_when_is_loading(tmp_path) -> None:
    """is_loading=True → check_once returns empty, no fire."""
    fired: list[str] = []
    task = add_cron_task(tmp_path, cron="* * * * *", prompt="ping", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])

    scheduler = CronScheduler(tmp_path, on_fire=fired.append, is_loading=lambda: True)
    due = scheduler.check_once(at_ms=3_000)

    assert due == []
    assert not fired


def test_check_once_fires_after_is_loading_clears(tmp_path) -> None:
    """is_loading flips True→False → overdue task fires on next tick."""
    fired: list[str] = []
    task = add_cron_task(tmp_path, cron="* * * * *", prompt="ping", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])

    busy = True
    scheduler = CronScheduler(tmp_path, on_fire=fired.append, is_loading=lambda: busy)

    due1 = scheduler.check_once(at_ms=3_000)
    assert due1 == []
    assert not fired

    busy = False
    due2 = scheduler.check_once(at_ms=3_000)
    assert len(due2) == 1
    assert fired == ["ping"]


def test_assistant_mode_bypasses_is_loading(tmp_path) -> None:
    """is_loading=True + assistant_mode=True → fire proceeds normally."""
    fired: list[str] = []
    task = add_cron_task(tmp_path, cron="* * * * *", prompt="ping", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])

    scheduler = CronScheduler(
        tmp_path,
        on_fire=fired.append,
        is_loading=lambda: True,
        assistant_mode=True,
    )
    due = scheduler.check_once(at_ms=3_000)

    assert len(due) == 1
    assert fired == ["ping"]


def test_is_loading_exception_treated_as_not_loading(tmp_path) -> None:
    """is_loading callback raises → treated as False (defensive)."""
    fired: list[str] = []
    task = add_cron_task(tmp_path, cron="* * * * *", prompt="ping", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])

    def broken_loading():
        raise RuntimeError("boom")

    scheduler = CronScheduler(tmp_path, on_fire=fired.append, is_loading=broken_loading)
    due = scheduler.check_once(at_ms=3_000)

    assert len(due) == 1
    assert fired == ["ping"]


def test_is_loading_none_means_never_busy(tmp_path) -> None:
    """is_loading=None (default) → gate never activates."""
    fired: list[str] = []
    task = add_cron_task(tmp_path, cron="* * * * *", prompt="ping", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])

    scheduler = CronScheduler(tmp_path, on_fire=fired.append)
    due = scheduler.check_once(at_ms=3_000)

    assert len(due) == 1
    assert fired == ["ping"]
