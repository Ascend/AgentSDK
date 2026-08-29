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

"""Cron task accumulation guard — ~D4 integration tests.

Verifies that when a recurring task's interval is shorter than its
execution time, the scheduler does NOT pile up prompts in the outbox.

The 4-layer defence:
  D1: sourceId-level active-run dedup in ``create_queued_run``
  D2: PID liveness check for stale runs
  D3: inFlight set prevents double-fire during async IO
  D4: filesystem lock for cross-process mutual exclusion

These tests focus on D1: after ``check_once`` fires a task, the run
stays in "queued" status. Subsequent ticks see the active run and
skip firing. Only after the run is externally finalized (simulating
the REPL completing ``chat()``) does the next tick fire again.
"""

from __future__ import annotations

from dataclasses import replace

from clawcodex_ext.cron_system.runs import (
    finalize_cron_run,
    read_cron_runs,
)
from clawcodex_ext.cron_system.scheduler import CronScheduler
from clawcodex_ext.cron_system.tasks import (
    add_cron_task,
    read_cron_tasks,
    write_cron_tasks,
)


def test_recurring_task_no_outbox_accumulation(tmp_path) -> None:
    """Simulate 3 scheduler ticks at 1-minute intervals with NO outbox
    drain (REPL is busy processing the first prompt). The outbox
    should contain exactly 1 prompt, not 3.
    """
    outbox: list[str] = []
    task = add_cron_task(
        tmp_path,
        cron="*/1 * * * *",
        prompt="long-running-task",
        recurring=True,
        created_at=1_000,
    )
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])
    scheduler = CronScheduler(tmp_path, on_fire=outbox.append)

    scheduler.check_once(at_ms=3_000)
    assert len(outbox) == 1

    write_cron_tasks(tmp_path, [replace(read_cron_tasks(tmp_path)[0], next_fire_at=4_000)])
    scheduler.check_once(at_ms=63_000)
    assert len(outbox) == 1

    write_cron_tasks(tmp_path, [replace(read_cron_tasks(tmp_path)[0], next_fire_at=64_000)])
    scheduler.check_once(at_ms=123_000)
    assert len(outbox) == 1


def test_run_stays_active_until_finalized(tmp_path) -> None:
    """After ``check_once`` fires, the run stays in "queued" status.
    It does NOT auto-transition to "completed".
    """
    outbox: list[str] = []
    task = add_cron_task(
        tmp_path,
        cron="*/1 * * * *",
        prompt="ping",
        recurring=True,
        created_at=1_000,
    )
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])
    scheduler = CronScheduler(tmp_path, on_fire=outbox.append)

    scheduler.check_once(at_ms=3_000)

    runs = read_cron_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].status == "queued"


def test_new_trigger_fires_after_previous_completes(tmp_path) -> None:
    """After the first run is finalized to "completed", the next tick
    fires normally.
    """
    outbox: list[str] = []
    task = add_cron_task(
        tmp_path,
        cron="*/1 * * * *",
        prompt="ping",
        recurring=True,
        created_at=1_000,
    )
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])
    scheduler = CronScheduler(tmp_path, on_fire=outbox.append)

    scheduler.check_once(at_ms=3_000)
    assert len(outbox) == 1
    runs = read_cron_runs(tmp_path)
    assert runs[0].status == "queued"

    finalize_cron_run(tmp_path, runs[0].id, "completed")

    write_cron_tasks(tmp_path, [replace(read_cron_tasks(tmp_path)[0], next_fire_at=4_000)])
    scheduler.check_once(at_ms=5_000)
    assert len(outbox) == 2
    runs = read_cron_runs(tmp_path)
    active_runs = [r for r in runs if r.status == "queued"]
    assert len(active_runs) == 1


def test_failed_callback_finalizes_run_as_failed(tmp_path) -> None:
    """If the fire callback raises, the run is finalized as "failed"
    (not left in "queued" forever).
    """
    task = add_cron_task(
        tmp_path,
        cron="*/1 * * * *",
        prompt="boom",
        recurring=True,
        created_at=1_000,
    )
    write_cron_tasks(tmp_path, [replace(task, next_fire_at=2_000)])

    def _bad_fire(prompt: str) -> None:
        raise RuntimeError("callback exploded")

    scheduler = CronScheduler(tmp_path, on_fire=_bad_fire)
    scheduler.check_once(at_ms=3_000)

    runs = read_cron_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert "callback exploded" in (runs[0].error or "")
