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

"""Test: two 1-minute recurring tasks firing simultaneously, one with slow execution."""

from __future__ import annotations

import time
from dataclasses import replace

from clawcodex_ext.cron_system.scheduler import CronScheduler
from clawcodex_ext.cron_system.tasks import add_cron_task, read_cron_tasks, write_cron_tasks


def test_both_tasks_due_same_tick_both_fire(tmp_path) -> None:
    """A and B both due at same time -> both fire in one check_once call."""
    fired: list[str] = []
    task_a = add_cron_task(tmp_path, cron="* * * * *", prompt="task_a", recurring=True, created_at=1_000)
    task_b = add_cron_task(tmp_path, cron="* * * * *", prompt="task_b", recurring=True, created_at=1_000)
    fire_time = 60_000
    write_cron_tasks(
        tmp_path,
        [
            replace(task_a, next_fire_at=fire_time),
            replace(task_b, next_fire_at=fire_time),
        ],
    )

    scheduler = CronScheduler(tmp_path, on_fire=fired.append)
    due = scheduler.check_once(at_ms=fire_time + 1000)

    assert len(due) == 2
    assert {t.prompt for t in due} == {"task_a", "task_b"}
    assert set(fired) == {"task_a", "task_b"}


def test_slow_callback_blocks_other_task_in_same_tick(tmp_path) -> None:
    """check_once() iterates due tasks sequentially in a for-loop.
    The fire callback runs synchronously. Whichever task fires first
    (by id sort order) blocks the other until its callback returns.
    """
    fire_order: list[tuple[str, float]] = []

    def slow_fire_task(task, run):
        fire_order.append((f"{task.prompt}_start", time.monotonic()))
        time.sleep(0.3)
        fire_order.append((f"{task.prompt}_end", time.monotonic()))

    task_a = add_cron_task(tmp_path, cron="* * * * *", prompt="task_a", recurring=True, created_at=1_000)
    task_b = add_cron_task(tmp_path, cron="* * * * *", prompt="task_b", recurring=True, created_at=1_000)
    fire_time = 60_000
    write_cron_tasks(
        tmp_path,
        [
            replace(task_a, next_fire_at=fire_time),
            replace(task_b, next_fire_at=fire_time),
        ],
    )

    scheduler = CronScheduler(tmp_path, on_fire=lambda p: None, on_fire_task=slow_fire_task)
    due = scheduler.check_once(at_ms=fire_time + 1000)

    assert len(due) == 2
    assert len(fire_order) == 4

    first_task = fire_order[0][0].replace("_start", "")
    second_task = fire_order[2][0].replace("_start", "")
    assert first_task != second_task

    first_end = fire_order[1][1]
    second_start = fire_order[2][1]
    assert second_start >= first_end, (
        f"second task ({second_task}) should not start until first task ({first_task}) callback finishes"
    )
    gap_ms = (second_start - first_end) * 1000
    assert gap_ms < 50, f"second task should start promptly after first finishes, gap={gap_ms:.1f}ms"


def test_fast_outbox_callback_both_fire_near_simultaneously(tmp_path) -> None:
    """Default attach_cron_runtime callback just appends to outbox (fast).
    Both tasks fire within the same check_once, negligible delay between them.
    """
    outbox: list[dict] = []

    def fast_fire_task(task, run):
        outbox.append({"prompt": task.prompt, "run_id": run.id, "time": time.monotonic()})

    task_a = add_cron_task(tmp_path, cron="* * * * *", prompt="task_a", recurring=True, created_at=1_000)
    task_b = add_cron_task(tmp_path, cron="* * * * *", prompt="task_b", recurring=True, created_at=1_000)
    fire_time = 60_000
    write_cron_tasks(
        tmp_path,
        [
            replace(task_a, next_fire_at=fire_time),
            replace(task_b, next_fire_at=fire_time),
        ],
    )

    scheduler = CronScheduler(tmp_path, on_fire=lambda p: None, on_fire_task=fast_fire_task)
    due = scheduler.check_once(at_ms=fire_time + 1000)

    assert len(due) == 2
    assert len(outbox) == 2
    delay_ms = (outbox[1]["time"] - outbox[0]["time"]) * 1000
    assert delay_ms < 10, f"both tasks should fire near-instantly, got {delay_ms:.1f}ms gap"


def test_long_running_a_delays_next_check_for_b(tmp_path) -> None:
    """After a slow check_once, the next tick is delayed.
    The first task's callback sleeps, blocking the second task's fire
    and delaying the overall check_once return.
    """
    fire_log: list[str] = []
    first_call = True

    def slow_fire_task(task, run):
        nonlocal first_call
        fire_log.append(f"{task.prompt}@{run.queued_at}")
        if first_call:
            first_call = False
            time.sleep(0.5)

    task_a = add_cron_task(tmp_path, cron="* * * * *", prompt="task_a", recurring=True, created_at=1_000)
    task_b = add_cron_task(tmp_path, cron="* * * * *", prompt="task_b", recurring=True, created_at=1_000)
    fire_time = 60_000
    write_cron_tasks(
        tmp_path,
        [
            replace(task_a, next_fire_at=fire_time),
            replace(task_b, next_fire_at=fire_time),
        ],
    )

    scheduler = CronScheduler(tmp_path, on_fire=lambda p: None, on_fire_task=slow_fire_task)

    t0 = time.monotonic()
    scheduler.check_once(at_ms=fire_time + 1000)
    elapsed = time.monotonic() - t0

    assert len(fire_log) == 2
    assert elapsed >= 0.4, f"check_once should take >=0.4s due to slow callback, got {elapsed:.2f}s"

    tasks = read_cron_tasks(tmp_path)
    for t in tasks:
        assert t.last_fired_at == fire_time + 1000
        assert t.next_fire_at is not None and t.next_fire_at > fire_time + 1000


def test_in_flight_does_not_block_different_task(tmp_path) -> None:
    """in_flight set is per-task-id. Task A being in-flight does NOT
    prevent task B from firing in the same tick.
    """
    fired: list[str] = []

    task_a = add_cron_task(tmp_path, cron="* * * * *", prompt="task_a", recurring=True, created_at=1_000)
    task_b = add_cron_task(tmp_path, cron="* * * * *", prompt="task_b", recurring=True, created_at=1_000)
    fire_time = 60_000
    write_cron_tasks(
        tmp_path,
        [
            replace(task_a, next_fire_at=fire_time),
            replace(task_b, next_fire_at=fire_time),
        ],
    )

    scheduler = CronScheduler(tmp_path, on_fire=fired.append)
    scheduler._in_flight_add(task_a.id)

    due = scheduler.check_once(at_ms=fire_time + 1000)

    assert len(due) == 1
    assert due[0].prompt == "task_b"
    assert "task_b" in fired
    assert "task_a" not in fired
