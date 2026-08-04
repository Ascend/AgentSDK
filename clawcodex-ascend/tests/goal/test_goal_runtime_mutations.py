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
# pylint: disable=no-name-in-module
"""Goal runtime mutation integration unit test."""

from __future__ import annotations

import threading
from pathlib import Path

from clawcodex_ext.goal.model import ThreadGoalStatus
from clawcodex_ext.goal.runtime import GoalRuntime
from clawcodex_ext.goal.service import GoalService
from clawcodex_ext.goal.store import GoalStore, goals_db_filename


def make_runtime(tmp_path: Path) -> tuple[GoalService, GoalRuntime]:
    service = GoalService(store=GoalStore(tmp_path / goals_db_filename()))
    runtime = GoalRuntime(thread_id="thread-1", service=service)
    service.register_runtime(runtime)
    return service, runtime


def test_registered_runtime_service_mutations_control_idle_continuation(
    tmp_path: Path,
) -> None:
    service, runtime = make_runtime(tmp_path)

    goal = service.create_goal("thread-1", "external create")
    active_request = runtime.continue_if_idle()
    blocked = service.update_goal(
        "thread-1",
        ThreadGoalStatus.BLOCKED,
        expected_goal_id=goal.goal_id,
    )
    blocked_request = runtime.continue_if_idle()
    resumed = service.resume_goal("thread-1")
    resumed_request = runtime.continue_if_idle()
    cleared = service.clear_goal("thread-1")

    assert active_request is not None
    assert active_request.expected_goal_id == goal.goal_id
    assert blocked is not None
    assert blocked.status is ThreadGoalStatus.BLOCKED
    assert blocked_request is None
    assert resumed is not None
    assert resumed.status is ThreadGoalStatus.ACTIVE
    assert resumed_request is not None
    assert resumed_request.expected_goal_id == resumed.goal_id
    assert cleared is True
    assert runtime.continue_if_idle() is None


def test_turn_stop_waits_for_goal_state_permit(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    service.create_goal("thread-1", "serialize stop")
    runtime.on_turn_start("turn-1")
    started = threading.Event()
    finished = threading.Event()

    def stop_turn() -> None:
        started.set()
        runtime.on_turn_stop("turn-1")
        finished.set()

    thread = threading.Thread(target=stop_turn)
    with runtime.goal_state_permit():
        thread.start()
        assert started.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert finished.is_set()


def test_stale_turn_error_preserves_current_turn_steering(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    service.create_goal("thread-1", "old objective")
    runtime.on_turn_start("turn-stale")
    runtime.on_turn_start("turn-current")
    service.replace_goal("thread-1", "new objective")

    runtime.on_turn_error("turn-stale", RuntimeError("late failure"))

    assert len(runtime.consume_pending_steering_messages()) == 1
    assert runtime.accounting_state.current_turn_id() == "turn-current"


def test_unrelated_quota_error_blocks_goal(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    service.create_goal("thread-1", "handle provider failure")
    runtime.on_turn_start("turn-1")

    runtime.on_turn_error("turn-1", RuntimeError("The quota parameter is invalid"))

    goal = service.get_goal("thread-1")
    assert goal is not None
    assert goal.status is ThreadGoalStatus.BLOCKED


def test_provider_quota_variants_mark_goal_usage_limited(tmp_path: Path) -> None:
    for index, message in enumerate(("quota depleted", "monthly quota reached")):
        service, runtime = make_runtime(tmp_path / str(index))
        service.create_goal("thread-1", "handle provider quota")
        runtime.on_turn_start("turn-1")

        runtime.on_turn_error("turn-1", RuntimeError(message))

        goal = service.get_goal("thread-1")
        assert goal is not None
        assert goal.status is ThreadGoalStatus.USAGE_LIMITED


def test_continuation_claim_is_atomic_across_threads(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    service.create_goal("thread-1", "continue exactly once")
    request = runtime.continue_if_idle()
    assert request is not None

    barrier = threading.Barrier(3)
    results: list[bool] = []

    def claim() -> None:
        barrier.wait()
        results.append(runtime.claim_continuation(request))

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=1)

    assert sorted(results) == [False, True]
    runtime.on_turn_start("continuation-turn")
    next_request = runtime.continue_if_idle()
    assert next_request is None
