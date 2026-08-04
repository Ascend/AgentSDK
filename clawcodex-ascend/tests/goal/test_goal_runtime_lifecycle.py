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
# pylint: disable=no-name-in-module,use-implicit-booleaness-not-comparison
"""Goal runtime continuation and lifecycle tests for F-122 Spec 5."""

from __future__ import annotations

import asyncio
from pathlib import Path


from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
from clawcodex_ext.goal.runtime import (
    BUDGET_LIMIT_STEERING_MARKER,
    CONTINUATION_STEERING_MARKER,
    OBJECTIVE_UPDATED_STEERING_MARKER,
    GoalRuntime,
    restore_goal_runtime_after_session_resume,
)
from clawcodex_ext.goal.accounting import (
    BudgetLimitedGoalDisposition,
    GoalAccountingState,
)
from clawcodex_ext.goal.service import GoalService
from clawcodex_ext.goal.store import GoalStore, goals_db_filename
from clawcodex_ext.tool_system.context import ToolContext


class FakeClock:
    def __init__(self) -> None:
        self._now = 1_000.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _run(coro):
    try:
        previous = asyncio.get_event_loop()
    except RuntimeError:
        previous = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        if previous is not None and not previous.is_closed():
            asyncio.set_event_loop(previous)
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())


def make_service(tmp_path: Path) -> GoalService:
    return GoalService(store=GoalStore(tmp_path / goals_db_filename()))


def make_runtime(
    tmp_path: Path,
    *,
    thread_id: str = "thread-1",
    accounting_state: GoalAccountingState | None = None,
) -> tuple[GoalService, GoalRuntime]:
    service = make_service(tmp_path)
    runtime = GoalRuntime(
        thread_id=thread_id,
        service=service,
        accounting_state=accounting_state,
    )
    service.register_runtime(runtime)
    return service, runtime


def test_restore_after_resume_accounts_idle_active_goal_time(tmp_path: Path) -> None:
    clock = FakeClock()
    service, runtime = make_runtime(
        tmp_path,
        accounting_state=GoalAccountingState(clock=clock),
    )
    goal = service.replace_goal("thread-1", "resume accounting")

    runtime.restore_after_resume()
    clock.advance(7)
    progress = runtime.account_idle_goal_progress(BudgetLimitedGoalDisposition.KEEP_ACTIVE)

    stored = service.get_goal("thread-1")
    assert progress is not None
    assert progress.goal_id == goal.goal_id
    assert stored is not None
    assert stored.time_used_seconds == 7
    assert stored.tokens_used == 0
    assert stored.status is ThreadGoalStatus.ACTIVE


def test_session_resume_keeps_condition_but_resets_progress(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    goal = service.replace_goal("thread-1", "keep this condition")
    service.account_usage(
        "thread-1",
        expected_goal_id=goal.goal_id,
        token_delta=17,
        elapsed_seconds=6,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )

    runtime = restore_goal_runtime_after_session_resume(context)

    restored = service.get_goal("thread-1")
    assert runtime is not None
    assert runtime.thread_id == "thread-1"
    assert restored is not None
    assert restored.goal_id == goal.goal_id
    assert restored.objective == "keep this condition"
    assert restored.status is ThreadGoalStatus.ACTIVE
    assert restored.tokens_used == 0
    assert restored.time_used_seconds == 0


def test_session_resume_does_not_restore_achieved_evaluator_goal(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    goal = service.replace_goal(
        "thread-1",
        "already achieved",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    service.update_goal(
        "thread-1",
        ThreadGoalStatus.COMPLETE,
        expected_goal_id=goal.goal_id,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )

    runtime = restore_goal_runtime_after_session_resume(context)

    assert runtime is not None
    assert service.get_goal("thread-1") is None


def test_completion_accounts_triggering_turn_before_status_change(tmp_path: Path) -> None:
    clock = FakeClock()
    service, runtime = make_runtime(
        tmp_path,
        accounting_state=GoalAccountingState(clock=clock),
    )
    goal = service.replace_goal("thread-1", "finish with accounting")
    runtime.on_turn_start("turn-1", plan_mode=False)
    runtime.on_token_usage("turn-1", {"input_tokens": 3, "output_tokens": 2})
    clock.advance(7)

    completed = service.update_goal(
        "thread-1",
        ThreadGoalStatus.COMPLETE,
        expected_goal_id=goal.goal_id,
    )
    runtime.on_turn_stop("turn-1")

    assert completed is not None
    assert completed.status is ThreadGoalStatus.COMPLETE
    assert completed.tokens_used == 5
    assert completed.time_used_seconds == 7


def test_continue_if_idle_starts_only_for_idle_active_goal(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    goal = service.replace_goal("thread-1", "ship runtime")

    request = runtime.continue_if_idle()

    assert request is not None
    assert request.expected_goal_id == goal.goal_id
    assert request.messages[0].isMeta is True
    assert CONTINUATION_STEERING_MARKER in str(request.messages[0].content)

    runtime.on_turn_start("turn-1", plan_mode=False)

    assert runtime.continue_if_idle() is None


def test_pause_and_clear_invalidate_pending_continuation(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    service.replace_goal("thread-1", "do not continue after pause")
    paused_request = runtime.continue_if_idle()

    service.pause_goal("thread-1")

    assert paused_request is not None
    assert runtime.claim_continuation(paused_request) is False
    assert runtime.continue_if_idle() is None

    service.resume_goal("thread-1")
    cleared_request = runtime.continue_if_idle()
    service.clear_goal("thread-1")

    assert cleared_request is not None
    assert runtime.claim_continuation(cleared_request) is False
    assert runtime.continue_if_idle() is None


def test_replace_invalidates_old_pending_continuation(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    service.replace_goal("thread-1", "old objective")
    old_request = runtime.continue_if_idle()

    new_goal = service.replace_goal("thread-1", "new objective")
    new_request = runtime.continue_if_idle()

    assert old_request is not None
    assert runtime.claim_continuation(old_request) is False
    assert new_request is not None
    assert new_request.expected_goal_id == new_goal.goal_id
    assert "new objective" in str(new_request.messages[0].content)
    assert "old objective" not in str(new_request.messages[0].content)


def test_tool_finish_accounts_usage_and_reports_budget_limit_once(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    goal = service.replace_goal("thread-1", "budget", token_budget=10)
    runtime.on_turn_start("turn-1", plan_mode=False)
    runtime.on_token_usage(
        "turn-1",
        {"input_tokens": 6, "cache_read_input_tokens": 1, "output_tokens": 5},
    )

    first = runtime.on_tool_finish(
        "turn-1",
        tool_name="Bash",
        call_id="call-1",
        handler_executed=True,
    )
    second = runtime.on_tool_finish(
        "turn-1",
        tool_name="Read",
        call_id="call-2",
        handler_executed=True,
    )

    limited = service.get_goal("thread-1")
    assert limited is not None
    assert limited.goal_id == goal.goal_id
    assert limited.status is ThreadGoalStatus.BUDGET_LIMITED
    assert limited.tokens_used == 10
    assert len(first) == 1
    assert BUDGET_LIMIT_STEERING_MARKER in str(first[0].content)
    assert second == []
    assert runtime.continue_if_idle() is None


def test_expected_goal_id_prevents_old_turn_usage_from_writing_new_goal(
    tmp_path: Path,
) -> None:
    service, runtime = make_runtime(tmp_path)
    service.replace_goal("thread-1", "old", token_budget=10)
    runtime.on_turn_start("turn-1", plan_mode=False)
    runtime.on_token_usage("turn-1", {"input_tokens": 10, "output_tokens": 1})

    new_goal = service.replace_goal("thread-1", "new", token_budget=100)
    runtime.on_tool_finish(
        "turn-1",
        tool_name="Bash",
        call_id="call-1",
        handler_executed=True,
    )

    assert service.get_goal("thread-1") == new_goal


def test_objective_update_during_turn_queues_objective_updated_steering(
    tmp_path: Path,
) -> None:
    service, runtime = make_runtime(tmp_path)
    original = service.replace_goal("thread-1", "old objective")
    runtime.on_turn_start("turn-1", plan_mode=False)

    replacement = service.replace_goal("thread-1", "new <objective>")
    pending = runtime.consume_pending_steering_messages()

    assert replacement.goal_id != original.goal_id
    assert runtime.goal_id_at_turn_start("turn-1") == original.goal_id
    assert len(pending) == 1
    assert OBJECTIVE_UPDATED_STEERING_MARKER in str(pending[0].content)
    assert "new &lt;objective&gt;" in str(pending[0].content)
    assert runtime.consume_pending_steering_messages() == []


def test_turn_abort_keeps_active_goal_for_later_resume(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    service.replace_goal("thread-1", "abort me")
    runtime.on_turn_start("turn-1", plan_mode=False)
    runtime.on_token_usage("turn-1", {"input_tokens": 2, "output_tokens": 3})

    runtime.on_turn_abort("turn-1")

    goal = service.get_goal("thread-1")
    assert goal is not None
    assert goal.status is ThreadGoalStatus.ACTIVE
    assert goal.tokens_used == 5
    assert runtime.continue_if_idle() is not None


def test_turn_error_and_usage_limit_stop_active_goal(tmp_path: Path) -> None:
    service, runtime = make_runtime(tmp_path)
    service.replace_goal("thread-1", "error")
    runtime.on_turn_start("turn-1", plan_mode=False)
    runtime.on_token_usage("turn-1", {"input_tokens": 2, "output_tokens": 3})

    runtime.on_turn_error("turn-1", RuntimeError("model crashed"))

    blocked = service.get_goal("thread-1")
    assert blocked is not None
    assert blocked.status is ThreadGoalStatus.BLOCKED
    assert runtime.continue_if_idle() is None

    service.replace_goal("thread-1", "usage")
    runtime.on_turn_start("turn-2", plan_mode=False)
    runtime.on_turn_error("turn-2", RuntimeError("usage limit exceeded"))

    limited = service.get_goal("thread-1")
    assert limited is not None
    assert limited.status is ThreadGoalStatus.USAGE_LIMITED
    assert runtime.continue_if_idle() is None


def test_turn_error_keeps_evaluator_goal_active_for_retry(tmp_path: Path) -> None:
    clock = FakeClock()
    service, runtime = make_runtime(
        tmp_path,
        accounting_state=GoalAccountingState(clock=clock),
    )
    service.replace_goal(
        "thread-1",
        "retry after provider error",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    runtime.on_turn_start("turn-1", plan_mode=False)
    runtime.on_token_usage("turn-1", {"input_tokens": 2, "output_tokens": 3})

    runtime.on_turn_error("turn-1", RuntimeError("model crashed"))

    goal = service.get_goal("thread-1")
    assert goal is not None
    assert goal.status is ThreadGoalStatus.ACTIVE
    assert goal.tokens_used == 5
    clock.advance(4)
    runtime.account_idle_goal_progress(BudgetLimitedGoalDisposition.KEEP_ACTIVE)
    goal = service.get_goal("thread-1")
    assert goal is not None
    assert goal.time_used_seconds == 4
    assert runtime.continue_if_idle() is not None


def test_usage_limit_error_promotes_budget_limited_goal_to_usage_limited(
    tmp_path: Path,
) -> None:
    service, runtime = make_runtime(tmp_path)
    service.replace_goal("thread-1", "budget then usage limit", token_budget=1)
    runtime.on_turn_start("turn-1", plan_mode=False)
    runtime.on_token_usage("turn-1", {"input_tokens": 1, "output_tokens": 0})

    budget_prompt = runtime.on_tool_finish(
        "turn-1",
        tool_name="Bash",
        call_id="call-1",
        handler_executed=True,
    )
    runtime.on_turn_error("turn-1", RuntimeError("usage limit exceeded"))

    goal = service.get_goal("thread-1")
    assert budget_prompt
    assert goal is not None
    assert goal.status is ThreadGoalStatus.USAGE_LIMITED
    assert runtime.continue_if_idle() is None
