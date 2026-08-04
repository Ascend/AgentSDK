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
"""GoalService boundary for upstream-compatible thread goals."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from clawcodex_ext.goal.evaluator import GoalEvaluation
from clawcodex_ext.goal.files import (
    materialize_goal_objective,
    remove_materialized_goal_objective,
)
from clawcodex_ext.goal.gate import goal_enabled
from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoal, ThreadGoalStatus
from clawcodex_ext.goal.observability import (
    GoalObserver,
    record_goal_cleared,
    record_goal_created,
    record_status_transition,
    record_usage_accounted,
)
from clawcodex_ext.goal.store import GoalStore, GoalUpdate, current_goal_thread_id

KEEP_TOKEN_BUDGET = object()

_log = logging.getLogger(__name__)

GoalSubscriber = Callable[[ThreadGoal | None], None]


class GoalServiceError(RuntimeError):
    """Raised when a goal service operation is invalid or disabled."""


class GoalService:
    """Coordinates goal state mutations behind one service boundary.

    Runtime side effects are intentionally not implemented in Spec 3, but
    command/API callers still route through this class so those effects can be
    added later without giving callers direct store access.
    """

    def __init__(
        self,
        *,
        store: GoalStore | None = None,
        is_enabled: Any = goal_enabled,
        codex_home: Any = None,
        observer: GoalObserver | None = None,
    ) -> None:
        self.store = store if store is not None else GoalStore()
        self._is_enabled = is_enabled
        self.codex_home = codex_home
        self.observer = observer
        self._runtime_lock = threading.RLock()
        self._runtimes: dict[str, Any] = {}
        self._subscriber_lock = threading.RLock()
        self._subscribers: dict[str, list[GoalSubscriber]] = {}

    def register_runtime(self, runtime: Any) -> None:
        thread_id = str(getattr(runtime, "thread_id"))
        with self._runtime_lock:
            self._runtimes[thread_id] = runtime

    def unregister_runtime(self, runtime: Any) -> None:
        thread_id = str(getattr(runtime, "thread_id"))
        with self._runtime_lock:
            if self._runtimes.get(thread_id) is runtime:
                self._runtimes.pop(thread_id, None)

    def subscribe(
        self,
        thread_id: str,
        callback: GoalSubscriber,
        *,
        emit_current: bool = False,
    ) -> Callable[[], None]:
        """Subscribe to committed goal snapshots for one persisted thread."""

        normalized_thread_id = str(thread_id)
        with self._subscriber_lock:
            self._subscribers.setdefault(normalized_thread_id, []).append(callback)

        if emit_current:
            self._notify_subscriber(
                callback,
                self.store.get_thread_goal(normalized_thread_id),
            )

        def unsubscribe() -> None:
            with self._subscriber_lock:
                callbacks = self._subscribers.get(normalized_thread_id)
                if callbacks is None:
                    return
                try:
                    callbacks.remove(callback)
                except ValueError:
                    return
                if not callbacks:
                    self._subscribers.pop(normalized_thread_id, None)

        return unsubscribe

    def get_goal(self, thread_id: str) -> ThreadGoal | None:
        self._ensure_enabled()
        return self.store.get_thread_goal(thread_id)

    def create_goal(
        self,
        thread_id: str,
        objective: str,
        token_budget: int | None = None,
        completion_mode: GoalCompletionMode | str = GoalCompletionMode.TOOL,
    ) -> ThreadGoal:
        """Create a fresh active goal without replacing unfinished work."""
        self._ensure_enabled()
        normalized_objective = self._prepare_objective(objective)
        normalized_budget = _validate_token_budget(token_budget)
        normalized_completion_mode = _coerce_completion_mode(completion_mode)
        runtime = self._runtime_for_thread(thread_id)
        if runtime is None:
            previous = self.store.get_thread_goal(thread_id)
            try:
                goal = self.store.insert_thread_goal(
                    thread_id,
                    normalized_objective,
                    ThreadGoalStatus.ACTIVE,
                    normalized_budget,
                    normalized_completion_mode,
                )
            except Exception:
                self._discard_objective(normalized_objective)
                raise
            if goal is None:
                self._discard_objective(normalized_objective)
                raise _unfinished_goal_error()
            self._discard_replaced_objective(previous, goal)
            record_goal_created(self.observer, goal)
            record_status_transition(self.observer, None, goal)
            self._publish_goal(goal)
            return goal

        with runtime.goal_state_permit():
            previous = self.store.get_thread_goal(thread_id)
            if previous is not None and previous.status is not ThreadGoalStatus.COMPLETE:
                raise _unfinished_goal_error()
            runtime.prepare_external_goal_mutation()
            try:
                goal = self.store.insert_thread_goal(
                    thread_id,
                    normalized_objective,
                    ThreadGoalStatus.ACTIVE,
                    normalized_budget,
                    normalized_completion_mode,
                )
            except Exception:
                self._discard_objective(normalized_objective)
                raise
            if goal is None:
                self._discard_objective(normalized_objective)
                raise _unfinished_goal_error()
            runtime.apply_external_goal_set(goal, previous)
            self._discard_replaced_objective(previous, goal)
            record_goal_created(self.observer, goal)
            record_status_transition(self.observer, None, goal)
        self._publish_goal(goal)
        return goal

    def set_goal(
        self,
        thread_id: str,
        objective: str | None,
        status: ThreadGoalStatus | str | None = ThreadGoalStatus.ACTIVE,
        token_budget: int | None | object = KEEP_TOKEN_BUDGET,
        completion_mode: GoalCompletionMode | str | None = None,
    ) -> ThreadGoal:
        self._ensure_enabled()
        normalized_objective = self._prepare_objective(objective) if objective is not None else None
        normalized_status = None if status is None else _coerce_status(status)
        normalized_budget = None if token_budget is KEEP_TOKEN_BUDGET else _normalize_token_budget_update(token_budget)
        normalized_completion_mode = None if completion_mode is None else _coerce_completion_mode(completion_mode)
        runtime = self._runtime_for_thread(thread_id)
        if runtime is not None:
            with runtime.goal_state_permit():
                goal = self._set_goal_locked(
                    runtime,
                    thread_id,
                    normalized_objective,
                    normalized_status,
                    normalized_budget,
                    normalized_completion_mode,
                    token_budget_is_keep=token_budget is KEEP_TOKEN_BUDGET,
                )
            self._publish_goal(goal)
            return goal

        existing = self.store.get_thread_goal(thread_id)

        if existing is None:
            if normalized_objective is None:
                raise GoalServiceError(f"cannot update goal for thread {thread_id}: no goal exists")
            try:
                created_goal = self.store.insert_thread_goal(
                    thread_id,
                    normalized_objective,
                    normalized_status or ThreadGoalStatus.ACTIVE,
                    None if token_budget is KEEP_TOKEN_BUDGET else normalized_budget,
                    normalized_completion_mode or GoalCompletionMode.TOOL,
                )
            except Exception:
                self._discard_objective(normalized_objective)
                raise
            if created_goal is None:
                self._discard_objective(normalized_objective)
                raise GoalServiceError(f"cannot create goal for thread {thread_id}: unfinished goal exists")
            record_goal_created(self.observer, created_goal)
            record_status_transition(self.observer, None, created_goal)
            self._publish_goal(created_goal)
            return created_goal

        update = (
            GoalUpdate(
                objective=normalized_objective,
                status=normalized_status,
                completion_mode=normalized_completion_mode,
            )
            if token_budget is KEEP_TOKEN_BUDGET
            else GoalUpdate(
                objective=normalized_objective,
                status=normalized_status,
                token_budget=normalized_budget,
                completion_mode=normalized_completion_mode,
            )
        )
        try:
            updated_goal = self.store.update_thread_goal(
                thread_id,
                update,
                expected_goal_id=existing.goal_id,
            )
        except Exception:
            if normalized_objective is not None:
                self._discard_objective(normalized_objective)
            raise
        if updated_goal is None:
            if normalized_objective is not None:
                self._discard_objective(normalized_objective)
            raise GoalServiceError(f"cannot update goal for thread {thread_id}")
        self._discard_replaced_objective(existing, updated_goal)
        record_status_transition(self.observer, existing.status, updated_goal)
        self._publish_goal(updated_goal)
        return updated_goal

    def replace_goal(
        self,
        thread_id: str,
        objective: str,
        token_budget: int | None = None,
        completion_mode: GoalCompletionMode | str = GoalCompletionMode.TOOL,
    ) -> ThreadGoal:
        self._ensure_enabled()
        normalized_objective = self._prepare_objective(objective)
        normalized_budget = _validate_token_budget(token_budget)
        normalized_completion_mode = _coerce_completion_mode(completion_mode)
        runtime = self._runtime_for_thread(thread_id)
        if runtime is None:
            previous = self.store.get_thread_goal(thread_id)
            try:
                goal = self.store.replace_thread_goal(
                    thread_id,
                    normalized_objective,
                    ThreadGoalStatus.ACTIVE,
                    normalized_budget,
                    normalized_completion_mode,
                )
            except Exception:
                self._discard_objective(normalized_objective)
                raise
            self._discard_replaced_objective(previous, goal)
            record_goal_created(self.observer, goal)
            record_status_transition(self.observer, None, goal)
            self._publish_goal(goal)
            return goal
        with runtime.goal_state_permit():
            previous = self.store.get_thread_goal(thread_id)
            runtime.prepare_external_goal_mutation()
            try:
                goal = self.store.replace_thread_goal(
                    thread_id,
                    normalized_objective,
                    ThreadGoalStatus.ACTIVE,
                    normalized_budget,
                    normalized_completion_mode,
                )
            except Exception:
                self._discard_objective(normalized_objective)
                raise
            runtime.apply_external_goal_set(goal, previous)
            self._discard_replaced_objective(previous, goal)
            record_goal_created(self.observer, goal)
            record_status_transition(self.observer, None, goal)
        self._publish_goal(goal)
        return goal

    def clear_goal(self, thread_id: str) -> bool:
        self._ensure_enabled()
        runtime = self._runtime_for_thread(thread_id)
        if runtime is None:
            deleted = self.store.delete_thread_goal(thread_id)
            if deleted is not None:
                self._discard_objective(deleted.objective)
                record_goal_cleared(self.observer, deleted)
                self._publish_thread_goal(thread_id, None)
            return deleted is not None
        with runtime.goal_state_permit():
            previous = self.store.get_thread_goal(thread_id)
            if previous is None:
                return False
            runtime.prepare_external_goal_mutation()
            deleted = self.store.delete_thread_goal(thread_id)
            if deleted is None:
                return False
            runtime.apply_external_goal_clear(deleted)
            self._discard_objective(deleted.objective)
            record_goal_cleared(self.observer, deleted)
        self._publish_thread_goal(thread_id, None)
        return True

    def pause_goal(self, thread_id: str) -> ThreadGoal | None:
        return self.update_goal(thread_id, ThreadGoalStatus.PAUSED)

    def resume_goal(self, thread_id: str) -> ThreadGoal | None:
        return self.update_goal(thread_id, ThreadGoalStatus.ACTIVE)

    def update_goal(
        self,
        thread_id: str,
        status: ThreadGoalStatus | str,
        expected_goal_id: str | None = None,
    ) -> ThreadGoal | None:
        self._ensure_enabled()
        normalized_status = _coerce_status(status)
        runtime = self._runtime_for_thread(thread_id)
        if runtime is None:
            previous = self.store.get_thread_goal(thread_id)
            goal = self.store.update_thread_goal(
                thread_id,
                GoalUpdate(status=normalized_status),
                expected_goal_id=expected_goal_id,
            )
            if goal is not None:
                record_status_transition(
                    self.observer,
                    previous.status if previous is not None else None,
                    goal,
                )
                self._publish_goal(goal)
            return goal
        with runtime.goal_state_permit():
            previous = self.store.get_thread_goal(thread_id)
            if previous is None:
                return None
            runtime.prepare_external_goal_mutation()
            goal = self.store.update_thread_goal(
                thread_id,
                GoalUpdate(status=normalized_status),
                expected_goal_id=expected_goal_id,
            )
            if goal is not None:
                runtime.apply_external_goal_set(goal, previous)
                record_status_transition(self.observer, previous.status, goal)
        if goal is not None:
            self._publish_goal(goal)
        return goal

    def update_goal_from_runtime(
        self,
        thread_id: str,
        status: ThreadGoalStatus | str,
        expected_goal_id: str | None = None,
    ) -> ThreadGoal | None:
        self._ensure_enabled()
        previous = self.store.get_thread_goal(thread_id)
        goal = self.store.update_thread_goal(
            thread_id,
            GoalUpdate(status=_coerce_status(status)),
            expected_goal_id=expected_goal_id,
        )
        if goal is not None:
            record_status_transition(
                self.observer,
                previous.status if previous is not None else None,
                goal,
            )
            self._publish_goal(goal)
        return goal

    def account_usage(
        self,
        thread_id: str,
        expected_goal_id: str | None,
        token_delta: int,
        elapsed_seconds: int,
    ) -> ThreadGoal | None:
        self._ensure_enabled()
        previous = self.store.get_thread_goal(thread_id)
        goal = self.store.account_thread_goal_usage(
            thread_id,
            time_delta=elapsed_seconds,
            token_delta=token_delta,
            expected_goal_id=expected_goal_id,
        )
        if goal is not None:
            record_usage_accounted(
                self.observer,
                goal,
                token_delta=token_delta,
                time_delta_seconds=elapsed_seconds,
            )
            record_status_transition(
                self.observer,
                previous.status if previous is not None else None,
                goal,
            )
            self._publish_goal(goal)
        return goal

    def record_evaluation(
        self,
        thread_id: str,
        evaluation: GoalEvaluation,
        expected_goal_id: str,
        expected_evaluation_count: int,
    ) -> ThreadGoal | None:
        """Record one independent evaluator decision for an active goal."""

        self._ensure_enabled()
        previous = self.store.get_thread_goal(thread_id)
        goal = self.store.record_thread_goal_evaluation(
            thread_id,
            evaluation,
            expected_goal_id=expected_goal_id,
            expected_evaluation_count=expected_evaluation_count,
        )
        if goal is not None:
            record_status_transition(
                self.observer,
                previous.status if previous is not None else None,
                goal,
            )
            self._publish_goal(goal)
        return goal

    def reset_progress_for_resume(
        self,
        thread_id: str,
        *,
        expected_goal_id: str | None = None,
    ) -> ThreadGoal | None:
        """Reset active goal metrics when a persisted session is resumed."""

        self._ensure_enabled()
        goal = self.store.reset_thread_goal_progress_for_resume(
            thread_id,
            expected_goal_id=expected_goal_id,
        )
        if goal is not None:
            self._publish_goal(goal)
        return goal

    def _ensure_enabled(self) -> None:
        if not bool(self._is_enabled()):
            raise GoalServiceError("goals feature is disabled")

    def _runtime_for_thread(self, thread_id: str) -> Any | None:
        with self._runtime_lock:
            return self._runtimes.get(thread_id)

    def _publish_goal(self, goal: ThreadGoal) -> None:
        self._publish_thread_goal(goal.thread_id, goal)

    def _publish_thread_goal(self, thread_id: str, goal: ThreadGoal | None) -> None:
        with self._subscriber_lock:
            callbacks = tuple(self._subscribers.get(str(thread_id), ()))
        for callback in callbacks:
            self._notify_subscriber(callback, goal)

    @staticmethod
    def _notify_subscriber(callback: GoalSubscriber, goal: ThreadGoal | None) -> None:
        try:
            callback(goal)
        except Exception:
            _log.exception("Goal subscriber failed")

    def _prepare_objective(self, objective: str) -> str:
        return materialize_goal_objective(
            _validate_objective(objective),
            codex_home=self.codex_home,
        ).objective

    def _discard_objective(self, objective: str) -> None:
        remove_materialized_goal_objective(
            objective,
            codex_home=self.codex_home,
        )

    def _discard_replaced_objective(
        self,
        previous: ThreadGoal | None,
        current: ThreadGoal,
    ) -> None:
        if previous is not None and previous.objective != current.objective:
            self._discard_objective(previous.objective)

    def _set_goal_locked(
        self,
        runtime: Any,
        thread_id: str,
        normalized_objective: str | None,
        normalized_status: ThreadGoalStatus | None,
        normalized_budget: int | None,
        normalized_completion_mode: GoalCompletionMode | None,
        *,
        token_budget_is_keep: bool,
    ) -> ThreadGoal:
        existing = self.store.get_thread_goal(thread_id)
        if existing is None:
            if normalized_objective is None:
                raise GoalServiceError(f"cannot update goal for thread {thread_id}: no goal exists")
            runtime.prepare_external_goal_mutation()
            try:
                goal = self.store.insert_thread_goal(
                    thread_id,
                    normalized_objective,
                    normalized_status or ThreadGoalStatus.ACTIVE,
                    None if token_budget_is_keep else normalized_budget,
                    normalized_completion_mode or GoalCompletionMode.TOOL,
                )
            except Exception:
                self._discard_objective(normalized_objective)
                raise
            if goal is None:
                self._discard_objective(normalized_objective)
                raise GoalServiceError(f"cannot create goal for thread {thread_id}: unfinished goal exists")
            runtime.apply_external_goal_set(goal, existing)
            record_goal_created(self.observer, goal)
            record_status_transition(self.observer, None, goal)
            return goal

        update = (
            GoalUpdate(
                objective=normalized_objective,
                status=normalized_status,
                completion_mode=normalized_completion_mode,
            )
            if token_budget_is_keep
            else GoalUpdate(
                objective=normalized_objective,
                status=normalized_status,
                token_budget=normalized_budget,
                completion_mode=normalized_completion_mode,
            )
        )
        runtime.prepare_external_goal_mutation()
        try:
            goal = self.store.update_thread_goal(
                thread_id,
                update,
                expected_goal_id=existing.goal_id,
            )
        except Exception:
            if normalized_objective is not None:
                self._discard_objective(normalized_objective)
            raise
        if goal is None:
            if normalized_objective is not None:
                self._discard_objective(normalized_objective)
            raise GoalServiceError(f"cannot update goal for thread {thread_id}")
        runtime.apply_external_goal_set(goal, existing)
        self._discard_replaced_objective(existing, goal)
        record_status_transition(self.observer, existing.status, goal)
        return goal


def goal_thread_id_from_context(context: Any) -> str:
    explicit = getattr(context, "goal_thread_id", None)
    if explicit:
        return str(explicit)
    tool_context = getattr(context, "tool_context", None)
    session_id = getattr(tool_context, "session_id", None)
    if session_id:
        return str(session_id)
    return current_goal_thread_id()


def clear_goal_for_context(context: Any) -> bool:
    """Remove the goal owned by a live session context when one is bound."""

    tool_context = getattr(context, "tool_context", None) or context
    service = getattr(tool_context, "goal_service", None) or getattr(
        context,
        "goal_service",
        None,
    )
    if service is None:
        return False
    thread_id = getattr(tool_context, "goal_thread_id", None) or getattr(
        tool_context,
        "session_id",
        None,
    )
    if not thread_id:
        return False
    return bool(service.clear_goal(str(thread_id)))


def _validate_objective(objective: str) -> str:
    text = objective.strip()
    if not text:
        raise GoalServiceError("goal objective cannot be empty")
    return text


def _unfinished_goal_error() -> GoalServiceError:
    return GoalServiceError(
        "cannot create a new goal because this thread has an unfinished "
        "goal; complete, replace, or clear the existing goal first"
    )


def _coerce_status(status: ThreadGoalStatus | str) -> ThreadGoalStatus:
    if isinstance(status, ThreadGoalStatus):
        return status
    return ThreadGoalStatus.from_wire(str(status))


def _coerce_completion_mode(
    completion_mode: GoalCompletionMode | str,
) -> GoalCompletionMode:
    if isinstance(completion_mode, GoalCompletionMode):
        return completion_mode
    return GoalCompletionMode.from_wire(str(completion_mode))


def _normalize_token_budget_update(value: int | None | object) -> int | None:
    if value is KEEP_TOKEN_BUDGET:
        raise RuntimeError("token budget keep sentinel must be handled by caller")
    return _validate_token_budget(value)


def _validate_token_budget(value: int | None | object) -> int | None:
    if value is None:
        return None
    budget = int(value)
    if budget < 0:
        raise GoalServiceError("token budget must be non-negative")
    return budget


__all__ = [
    "KEEP_TOKEN_BUDGET",
    "GoalService",
    "GoalServiceError",
    "clear_goal_for_context",
    "goal_thread_id_from_context",
]
