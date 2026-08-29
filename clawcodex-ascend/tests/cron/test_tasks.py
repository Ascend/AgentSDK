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

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from clawcodex_ext.cron_system.models import (
    SCHEDULED_TASKS_RELATIVE_PATH,
    CronJitterConfig,
    CronTask,
)
from clawcodex_ext.cron_system.parser import parse_cron_expression
from clawcodex_ext.cron_system.tasks import (
    add_cron_task,
    find_due_tasks,
    find_missed_tasks,
    mark_cron_tasks_fired,
    prune_expired_recurring_tasks,
    read_all_cron_tasks,
    read_cron_tasks,
    read_session_cron_tasks,
    remove_cron_tasks,
    write_cron_tasks,
)


def test_add_list_delete_persisted_tasks(tmp_path) -> None:
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", created_at=1_000)
    assert len(task.id) == 8
    assert (tmp_path / SCHEDULED_TASKS_RELATIVE_PATH).exists()
    assert read_cron_tasks(tmp_path) == [task]
    assert remove_cron_tasks(tmp_path, task.id) is True
    assert read_cron_tasks(tmp_path) == []


def test_invalid_persisted_entries_are_skipped(tmp_path) -> None:
    path = tmp_path / SCHEDULED_TASKS_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "bad"},
                    {"id": "ok", "cron": "*/5 * * * *", "prompt": "ping"},
                ]
            }
        ),
        encoding="utf-8",
    )
    tasks = read_cron_tasks(tmp_path)
    assert [task.id for task in tasks] == ["ok"]


def test_session_tasks_do_not_write_persisted_file(tmp_path) -> None:
    session_store = {}
    task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="session ping",
        durable=False,
        session_store=session_store,
        created_at=1_000,
    )

    assert read_session_cron_tasks(session_store) == [task]
    assert read_cron_tasks(tmp_path) == []
    assert not (tmp_path / SCHEDULED_TASKS_RELATIVE_PATH).exists()


def test_read_all_cron_tasks_merges_session_and_persisted_tasks(tmp_path) -> None:
    session_store = {}
    session_task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="session ping",
        durable=False,
        session_store=session_store,
        created_at=1_000,
    )
    durable_task = add_cron_task(
        tmp_path,
        cron="*/10 * * * *",
        prompt="durable ping",
        durable=True,
        created_at=1_000,
    )

    assert {task.id for task in read_all_cron_tasks(tmp_path, session_store)} == {
        session_task.id,
        durable_task.id,
    }


def test_remove_cron_tasks_deletes_session_before_persisted(tmp_path) -> None:
    session_store = {}
    task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="session ping",
        durable=False,
        session_store=session_store,
        created_at=1_000,
    )

    assert remove_cron_tasks(tmp_path, task.id, session_store) is True
    assert read_session_cron_tasks(session_store) == []


def test_mark_fired_updates_recurring_and_removes_one_shot(tmp_path) -> None:
    recurring = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", recurring=True, created_at=1_000)
    one_shot = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="once", recurring=False, created_at=1_000)
    mark_cron_tasks_fired(tmp_path, [recurring, one_shot], fired_at=10_000)
    tasks = read_cron_tasks(tmp_path)
    assert [task.id for task in tasks] == [recurring.id]
    assert tasks[0].last_fired_at == 10_000


def test_prune_expired_recurring_tasks(tmp_path) -> None:
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", recurring=True, created_at=1_000)
    write_cron_tasks(tmp_path, [replace(task, expires_at=2_000)])
    removed = prune_expired_recurring_tasks(tmp_path, at_ms=3_000)
    assert [task.id for task in removed] == [task.id]
    assert read_cron_tasks(tmp_path) == []


def test_add_cron_task_serializes_concurrent_writes(tmp_path) -> None:
    def create_task(index: int) -> str:
        task = add_cron_task(
            tmp_path,
            cron="* * * * *",
            prompt=f"ping {index}",
            jitter=CronJitterConfig(enabled=False),
            created_at=1_000,
        )
        return task.id

    with ThreadPoolExecutor(max_workers=16) as pool:
        task_ids = list(pool.map(create_task, range(40)))

    tasks = read_cron_tasks(tmp_path)
    assert len(task_ids) == 40
    assert len(tasks) == 40
    assert {task.id for task in tasks} == set(task_ids)


# ---------------------------------------------------------------------------
# Phase D-1: dual-durable support tests (session_store parameter coverage).
# ---------------------------------------------------------------------------


def test_find_due_tasks_includes_session_tasks(tmp_path) -> None:
    session_store: dict[str, object] = {}
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="file-task",
        recurring=True,
        durable=True,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="session-task",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    # both tasks have next_fire_at computed in the future; query at a
    # timestamp that lies after both -> both should be reported.
    due = find_due_tasks(tmp_path, at_ms=1_000_000, session_store=session_store)
    prompts = {task.prompt for task in due}
    assert "file-task" in prompts
    assert "session-task" in prompts


def test_find_missed_tasks_includes_session_tasks(tmp_path) -> None:
    session_store: dict[str, object] = {}
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="once-file",
        recurring=False,
        durable=True,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="once-session",
        recurring=False,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    # Query at 1_000_000 which is after both have fired-time.
    missed = find_missed_tasks(tmp_path, at_ms=1_000_000, session_store=session_store)
    prompts = {task.prompt for task in missed}
    assert "once-file" in prompts
    assert "once-session" in prompts


def test_mark_fired_updates_recurring_session_and_removes_one_shot_session(
    tmp_path,
) -> None:
    session_store: dict[str, object] = {}
    recurring = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="recurring-session",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    one_shot = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="one-shot-session",
        recurring=False,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    mark_cron_tasks_fired(tmp_path, [recurring, one_shot], fired_at=2_000, session_store=session_store)
    remaining = read_session_cron_tasks(session_store)
    ids = {task.id for task in remaining}
    # one-shot removed, recurring kept
    assert one_shot.id not in ids
    assert recurring.id in ids
    # recurring's last_fired_at is updated
    kept = next(task for task in remaining if task.id == recurring.id)
    assert kept.last_fired_at == 2_000
    assert kept.next_fire_at is not None and kept.next_fire_at > 2_000


def test_prune_expired_recurring_tasks_handles_session(tmp_path) -> None:
    session_store: dict[str, object] = {}
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="stale",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="permanent",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    # Mark the "permanent" task as permanent=True.
    tasks_now = read_session_cron_tasks(session_store)
    permanent_task = next(t for t in tasks_now if t.prompt == "permanent")
    session_store[permanent_task.id] = replace(permanent_task, permanent=True)

    removed = prune_expired_recurring_tasks(tmp_path, at_ms=1_000_000_000, session_store=session_store)
    assert {task.prompt for task in removed} == {"stale"}
    survivors = read_session_cron_tasks(session_store)
    assert {task.prompt for task in survivors} == {"permanent"}


def test_remove_missed_tasks_handles_session(tmp_path) -> None:
    from clawcodex_ext.cron_system.tasks import remove_missed_tasks

    session_store: dict[str, object] = {}
    one_shot_file = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="file",
        recurring=False,
        durable=True,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    one_shot_session = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="session",
        recurring=False,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    assert one_shot_session.id in session_store
    remove_missed_tasks(tmp_path, [one_shot_file, one_shot_session], session_store=session_store)
    assert one_shot_session.id not in session_store
    # file task was also removed from disk
    assert one_shot_file.id not in {task.id for task in read_cron_tasks(tmp_path)}


def test_session_store_none_safe(tmp_path) -> None:
    """Backward-compat: passing session_store=None must NOT regress to
    file-only behaviour and must NOT raise.
    """
    # Build a permanent task directly so prune leaves it alone.
    fields = parse_cron_expression("*/5 * * * *")
    assert fields is not None
    task = CronTask(
        id="perm1",
        cron="*/5 * * * *",
        prompt="file-only",
        recurring=True,
        durable=True,
        permanent=True,
        created_at=1_000,
        updated_at=1_000,
        next_fire_at=2_000,
        last_fired_at=None,
        jitter=CronJitterConfig(enabled=False),
    )
    write_cron_tasks(tmp_path, [task])
    # No session_store at all -> file path only.
    due = find_due_tasks(tmp_path, at_ms=10_000)
    assert [t.id for t in due] == [task.id]
    updated = mark_cron_tasks_fired(tmp_path, [task], fired_at=10_000)
    assert updated[0].last_fired_at == 10_000
    # No session_store passed to prune -> still works on file. permanent
    # tasks are exempt from auto-expiry so nothing is pruned.
    assert prune_expired_recurring_tasks(tmp_path, at_ms=10_000) == []


def test_concurrent_session_mark_fired_does_not_duplicate(tmp_path) -> None:
    """The session_store list view is copied before mutation so a
    concurrent mark_fired + add doesn't corrupt the dict.
    """
    import threading

    session_store: dict[str, object] = {}
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="rec",
        recurring=True,
        durable=False,
        session_store=session_store,
        jitter=CronJitterConfig(enabled=False),
        created_at=1_000,
    )
    rec = read_session_cron_tasks(session_store)[0]

    def fire() -> None:
        mark_cron_tasks_fired(tmp_path, [rec], fired_at=2_000, session_store=session_store)

    def add_more(index: int) -> None:
        add_cron_task(
            tmp_path,
            cron="*/5 * * * *",
            prompt=f"x{index}",
            recurring=True,
            durable=False,
            session_store=session_store,
            jitter=CronJitterConfig(enabled=False),
            created_at=1_000,
        )

    t1 = threading.Thread(target=fire)
    t2 = threading.Thread(target=add_more, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    survivors = read_session_cron_tasks(session_store)
    # rec must still exist with an updated last_fired_at, x1 must exist
    prompts = {t.prompt for t in survivors}
    assert "rec" in prompts
    assert "x1" in prompts
