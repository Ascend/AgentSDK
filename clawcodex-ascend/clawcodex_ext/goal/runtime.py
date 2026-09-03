#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

# pylint: disable=relative-beyond-top-level
"""Goal runtime lifecycle hooks and idle continuation."""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

# pylint: disable-next=no-name-in-module
from clawcodex_ext.types.messages import UserMessage

from .accounting import BudgetLimitedGoalDisposition, GoalAccountingState
from .gate import goal_enabled
from .model import GoalCompletionMode, ThreadGoal, ThreadGoalStatus
from .observability import (
    GoalObserver,
    record_continuation_skipped,
)
from .service import GoalService
from .steering import (
    BUDGET_LIMIT_STEERING_MARKER,
    CONTINUATION_STEERING_MARKER,
    EVALUATOR_START_MARKER,
    OBJECTIVE_UPDATED_STEERING_MARKER,
    budget_limit_steering_message,
    continuation_steering_message,
    evaluator_start_message,
    objective_updated_steering_message,
)
from .tools import (
    UPDATE_GOAL_TOOL_NAME,
    _is_review_subagent_context,
    _thread_id_from_context,
)

logger = logging.getLogger(__name__)

_USAGE_LIMIT_ERROR_MARKERS = (
    "usage limit",
    "rate limit",
    "insufficient_quota",
    "quota exceeded",
    "quota depleted",
    "quota reached",
    "exceeded your current quota",
    "billing hard limit",
)


@dataclass(frozen=True)
class AccountedGoalProgress:
    goal: ThreadGoal
    goal_id: str


@dataclass(frozen=True)
class GoalContinuationRequest:
    request_id: str
    expected_goal_id: str
    messages: list[UserMessage]


class GoalRuntime:
    """Per-thread goal runtime state and lifecycle hooks."""

    def __init__(
        self,
        *,
        thread_id: str,
        service: GoalService,
        tools_available_for_thread: bool = True,
        is_enabled: Any = goal_enabled,
        accounting_state: GoalAccountingState | None = None,
        observer: GoalObserver | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.service = service
        self.tools_available_for_thread = tools_available_for_thread
        self._is_enabled = is_enabled
        self.accounting_state = accounting_state or GoalAccountingState()
        self._goal_state_lock = threading.RLock()
        self._plan_mode = False
        self._pending_steering_messages: list[UserMessage] = []
        self._continuation_reservation: tuple[str, str] | None = None
        self.observer = observer if observer is not None else getattr(service, "observer", None)

    def is_enabled(self) -> bool:
        return bool(self._is_enabled())

    def tools_visible(self) -> bool:
        return self.is_enabled() and self.tools_available_for_thread

    @contextmanager
    def goal_state_permit(self) -> Iterator[None]:
        self._goal_state_lock.acquire()
        try:
            yield
        finally:
            self._goal_state_lock.release()

    def set_plan_mode(self, plan_mode: bool) -> None:
        self._plan_mode = bool(plan_mode)

    def next_turn_id(self) -> str:
        return f"{self.thread_id}:turn:{uuid.uuid4().hex}"

    def restore_after_resume(self) -> None:
        if not self.is_enabled():
            return
        goal = self.service.get_goal(self.thread_id)
        if goal is not None and goal.status is ThreadGoalStatus.ACTIVE:
            self.accounting_state.mark_idle_goal_active(goal.goal_id)
        else:
            self.accounting_state.clear_active_goal()

    def prepare_external_goal_mutation(self) -> None:
        if not self.is_enabled():
            return
        self._continuation_reservation = None
        turn_id = self.accounting_state.current_turn_id()
        if turn_id is not None:
            self.account_active_goal_progress(
                turn_id,
                BudgetLimitedGoalDisposition.CLEAR_ACTIVE,
            )
            return
        self.account_idle_goal_progress(BudgetLimitedGoalDisposition.CLEAR_ACTIVE)

    def apply_external_goal_set(
        self,
        goal: ThreadGoal,
        previous_goal: ThreadGoal | None,
    ) -> None:
        if not self.is_enabled():
            return
        current_turn_id = self.accounting_state.current_turn_id()
        if (
            current_turn_id is not None
            and previous_goal is not None
            and previous_goal.objective != goal.objective
            and goal.status is ThreadGoalStatus.ACTIVE
        ):
            self._pending_steering_messages.append(objective_updated_steering_message(goal))
        if goal.status is ThreadGoalStatus.ACTIVE:
            if current_turn_id is not None:
                self.accounting_state.mark_current_turn_goal_active(goal.goal_id)
            else:
                self.accounting_state.mark_idle_goal_active(goal.goal_id)
            return
        if goal.status is ThreadGoalStatus.BUDGET_LIMITED:
            if self.accounting_state.current_turn_id() is None:
                self.accounting_state.clear_active_goal()
            return
        self.accounting_state.clear_active_goal()

    def apply_external_goal_clear(self, goal: ThreadGoal) -> None:
        del goal
        if self.is_enabled():
            self.accounting_state.clear_active_goal()

    def on_turn_start(self, turn_id: str | None = None, *, plan_mode: bool = False) -> str:
        turn_id = turn_id or self.next_turn_id()
        with self.goal_state_permit():
            self.set_plan_mode(plan_mode)
            if not self.is_enabled():
                return turn_id
            self.accounting_state.start_turn(turn_id, plan_mode=plan_mode)
            if plan_mode:
                self._continuation_reservation = None
                self.accounting_state.clear_current_turn_goal()
                return turn_id
            goal = self.service.get_goal(self.thread_id)
            if goal is not None and goal.status in {
                ThreadGoalStatus.ACTIVE,
                ThreadGoalStatus.BUDGET_LIMITED,
            }:
                self.accounting_state.bind_goal_to_turn(turn_id, goal.goal_id)
            self._continuation_reservation = None
        return turn_id

    def on_token_usage(self, turn_id: str, usage: dict[str, Any] | None) -> None:
        if self.is_enabled():
            self.accounting_state.record_token_usage(turn_id, usage or {})

    def goal_id_at_turn_start(self, turn_id: str) -> str | None:
        """Return the immutable goal identity captured for a running turn."""

        if not self.is_enabled():
            return None
        return self.accounting_state.turn_started_goal_id(turn_id)

    def on_tool_finish(
        self,
        turn_id: str,
        *,
        tool_name: str,
        call_id: str,
        handler_executed: bool,
    ) -> list[UserMessage]:
        del call_id
        if not self.is_enabled() or not handler_executed or tool_name == UPDATE_GOAL_TOOL_NAME:
            return []
        progress = self.account_active_goal_progress(
            turn_id,
            BudgetLimitedGoalDisposition.KEEP_ACTIVE,
        )
        if progress is None or progress.goal.status is not ThreadGoalStatus.BUDGET_LIMITED:
            return []
        if not self.accounting_state.mark_budget_limit_reported_if_new(progress.goal_id):
            return []
        return [budget_limit_steering_message(progress.goal)]

    def on_turn_stop(self, turn_id: str) -> None:
        if not self.is_enabled():
            return
        with self.goal_state_permit():
            self._finish_stopped_turn(turn_id)

    def on_turn_abort(self, turn_id: str) -> None:
        if not self.is_enabled():
            return
        with self.goal_state_permit():
            self._finish_stopped_turn(turn_id)

    def _finish_stopped_turn(self, turn_id: str) -> None:
        self._continuation_reservation = None
        self.account_active_goal_progress(
            turn_id,
            BudgetLimitedGoalDisposition.CLEAR_ACTIVE,
        )
        self._pending_steering_messages.clear()
        self.accounting_state.finish_turn(turn_id)

    def on_turn_error(self, turn_id: str, error: BaseException) -> None:
        if not self.is_enabled():
            return
        with self.goal_state_permit():
            if not self.accounting_state.turn_is_current_active_goal(turn_id):
                self.accounting_state.finish_turn(turn_id)
                return
            self.account_active_goal_progress(
                turn_id,
                BudgetLimitedGoalDisposition.CLEAR_ACTIVE,
            )
            goal = self.service.get_goal(self.thread_id)
            if goal is None:
                self.accounting_state.clear_active_goal()
                self.accounting_state.finish_turn(turn_id)
                return
            if goal.completion_mode is GoalCompletionMode.EVALUATOR and goal.status is ThreadGoalStatus.ACTIVE:
                # Claude-style /goal conditions survive provider/model
                # failures. Stop this run, but keep the idle wall-clock
                # baseline live so `/goal` and the footer agree while the
                # user decides whether to retry or clear it.
                self.accounting_state.mark_idle_goal_active(goal.goal_id)
                self._pending_steering_messages.clear()
                self.accounting_state.finish_turn(turn_id)
                return
            status = ThreadGoalStatus.USAGE_LIMITED if _is_usage_limit_error(error) else ThreadGoalStatus.BLOCKED
            if goal.status is ThreadGoalStatus.ACTIVE or (
                goal.status is ThreadGoalStatus.BUDGET_LIMITED and status is ThreadGoalStatus.USAGE_LIMITED
            ):
                self.service.update_goal_from_runtime(
                    self.thread_id,
                    status,
                    expected_goal_id=goal.goal_id,
                )
            self.accounting_state.clear_active_goal()
            self._pending_steering_messages.clear()
            self.accounting_state.finish_turn(turn_id)

    def account_active_goal_progress(
        self,
        turn_id: str,
        budget_limited_goal_disposition: BudgetLimitedGoalDisposition,
    ) -> AccountedGoalProgress | None:
        with self.accounting_state.progress_accounting_permit():
            snapshot = self.accounting_state.progress_snapshot(turn_id)
            if snapshot is None:
                return None
            goal = self.service.account_usage(
                self.thread_id,
                expected_goal_id=snapshot.expected_goal_id,
                token_delta=snapshot.token_delta,
                elapsed_seconds=snapshot.time_delta_seconds,
            )
            if goal is None:
                self.accounting_state.clear_active_goal()
                return None
            self.accounting_state.mark_progress_accounted_for_status(
                turn_id,
                snapshot,
                goal.status,
                budget_limited_goal_disposition,
            )
            return AccountedGoalProgress(goal=goal, goal_id=goal.goal_id)

    def account_idle_goal_progress(
        self,
        budget_limited_goal_disposition: BudgetLimitedGoalDisposition,
    ) -> AccountedGoalProgress | None:
        with self.accounting_state.progress_accounting_permit():
            snapshot = self.accounting_state.idle_progress_snapshot()
            if snapshot is None:
                return None
            goal = self.service.account_usage(
                self.thread_id,
                expected_goal_id=snapshot.expected_goal_id,
                token_delta=0,
                elapsed_seconds=snapshot.time_delta_seconds,
            )
            if goal is None:
                self.accounting_state.reset_idle_progress_baseline_and_clear_active_goal()
                return None
            self.accounting_state.mark_idle_progress_accounted_for_status(
                snapshot,
                goal.status,
                budget_limited_goal_disposition,
            )
            return AccountedGoalProgress(goal=goal, goal_id=goal.goal_id)

    def continue_if_idle(self) -> GoalContinuationRequest | None:
        if not self.tools_visible():
            reason = "feature_disabled" if not self.is_enabled() else "no_live_session"
            record_continuation_skipped(
                self.observer,
                thread_id=self.thread_id,
                reason=reason,
            )
            self.accounting_state.clear_active_goal()
            return None
        with self.goal_state_permit():
            if self._plan_mode or self.accounting_state.current_turn_id() is not None:
                record_continuation_skipped(
                    self.observer,
                    thread_id=self.thread_id,
                    reason="plan_mode" if self._plan_mode else "not_idle",
                )
                return None
            goal = self.service.get_goal(self.thread_id)
            if goal is None:
                record_continuation_skipped(
                    self.observer,
                    thread_id=self.thread_id,
                    reason="no_active_goal",
                )
                self.accounting_state.clear_active_goal()
                return None
            if goal.status is not ThreadGoalStatus.ACTIVE:
                record_continuation_skipped(
                    self.observer,
                    thread_id=self.thread_id,
                    reason="paused" if goal.status is ThreadGoalStatus.PAUSED else goal.status.value,
                    goal=goal,
                )
                self.accounting_state.clear_active_goal()
                return None
            self.accounting_state.mark_idle_goal_active(goal.goal_id)
            return GoalContinuationRequest(
                request_id=uuid.uuid4().hex,
                expected_goal_id=goal.goal_id,
                messages=[
                    evaluator_start_message(goal)
                    if goal.completion_mode is GoalCompletionMode.EVALUATOR
                    else continuation_steering_message(goal)
                ],
            )

    def claim_continuation(self, request: GoalContinuationRequest) -> bool:
        with self.goal_state_permit():
            if (
                self._plan_mode
                or self.accounting_state.current_turn_id() is not None
                or self._continuation_reservation is not None
            ):
                return False
            goal = self.service.get_goal(self.thread_id)
            if goal is None or goal.status is not ThreadGoalStatus.ACTIVE or goal.goal_id != request.expected_goal_id:
                return False
            self._continuation_reservation = (
                request.expected_goal_id,
                request.request_id,
            )
            return True

    def release_continuation(self, request: GoalContinuationRequest) -> None:
        """Release a claimed request when dispatch fails before turn start."""

        with self.goal_state_permit():
            expected = (request.expected_goal_id, request.request_id)
            if self._continuation_reservation == expected:
                self._continuation_reservation = None

    def consume_pending_steering_messages(self) -> list[UserMessage]:
        with self.goal_state_permit():
            messages = list(self._pending_steering_messages)
            self._pending_steering_messages.clear()
            return messages


def goal_runtime_for_context(context: Any) -> GoalRuntime | None:
    if not goal_enabled() or context is None:
        return None
    thread_id = _thread_id_from_context(context)
    if thread_id is None:
        return None
    runtime = getattr(context, "goal_runtime", None)
    if isinstance(runtime, GoalRuntime) and runtime.thread_id == thread_id:
        runtime.set_plan_mode(bool(getattr(context, "plan_mode", False)))
        return runtime
    service = getattr(context, "goal_service", None)
    if service is None:
        service = GoalService()
        try:
            context.goal_service = service
        except Exception:
            logger.debug("Goal service context attachment failed", exc_info=True)
    runtime = GoalRuntime(
        thread_id=thread_id,
        service=service,
        tools_available_for_thread=not _is_review_subagent_context(context),
    )
    runtime.set_plan_mode(bool(getattr(context, "plan_mode", False)))
    service.register_runtime(runtime)
    try:
        context.goal_runtime = runtime
    except Exception:
        logger.debug("Goal runtime context attachment failed", exc_info=True)
    return runtime


def restore_goal_runtime_after_session_resume(context: Any) -> GoalRuntime | None:
    """Rebind an active goal to a resumed session with fresh metrics.

    Goal elapsed time and token counters describe the current live run.  A
    persisted active condition survives session resume, but its counters and
    wall-clock baseline start over, matching Claude Code's resume behaviour.
    """

    if not goal_enabled() or context is None:
        return None

    previous = getattr(context, "goal_runtime", None)
    if isinstance(previous, GoalRuntime):
        previous.service.unregister_runtime(previous)
        try:
            context.goal_runtime = None
        except Exception:
            logger.debug("Goal runtime context clearing failed", exc_info=True)

    runtime = goal_runtime_for_context(context)
    if runtime is None:
        return None

    goal = runtime.service.get_goal(runtime.thread_id)
    if goal is not None and goal.status is ThreadGoalStatus.ACTIVE:
        runtime.service.reset_progress_for_resume(
            runtime.thread_id,
            expected_goal_id=goal.goal_id,
        )
    elif (
        goal is not None
        and goal.status is ThreadGoalStatus.COMPLETE
        and goal.completion_mode is GoalCompletionMode.EVALUATOR
    ):
        # Achieved /goal state is represented by its transcript entry, not a
        # live goal restored into a later session run.
        runtime.service.clear_goal(runtime.thread_id)
    runtime.restore_after_resume()
    return runtime


def _is_usage_limit_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(needle in text for needle in _USAGE_LIMIT_ERROR_MARKERS)


__all__ = [
    "BUDGET_LIMIT_STEERING_MARKER",
    "CONTINUATION_STEERING_MARKER",
    "EVALUATOR_START_MARKER",
    "OBJECTIVE_UPDATED_STEERING_MARKER",
    "AccountedGoalProgress",
    "GoalContinuationRequest",
    "GoalRuntime",
    "goal_runtime_for_context",
    "restore_goal_runtime_after_session_resume",
]
