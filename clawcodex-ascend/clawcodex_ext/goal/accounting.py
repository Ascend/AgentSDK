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
"""Runtime accounting state for upstream-compatible thread goals."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator

from clawcodex_ext.goal.model import ThreadGoalStatus


class BudgetLimitedGoalDisposition(Enum):
    """Whether a budget-limited goal remains active for steering."""

    KEEP_ACTIVE = "keep_active"
    CLEAR_ACTIVE = "clear_active"


@dataclass(frozen=True)
class GoalProgressSnapshot:
    current_token_total: int
    expected_goal_id: str
    time_delta_seconds: int
    token_delta: int


@dataclass(frozen=True)
class IdleGoalProgressSnapshot:
    expected_goal_id: str
    time_delta_seconds: int


@dataclass
class _GoalTurnAccounting:
    current_token_total: int = 0
    last_accounted_token_total: int = 0
    active_goal_id: str | None = None
    started_goal_id: str | None = None
    account_tokens: bool = True

    def token_delta_since_last_accounting(self) -> int:
        return max(self.current_token_total - self.last_accounted_token_total, 0)


class _GoalWallClockAccounting:
    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.last_accounted_at = clock()
        self.active_goal_id: str | None = None

    def time_delta_since_last_accounting(self) -> int:
        return max(int(self._clock() - self.last_accounted_at), 0)

    def mark_accounted(self, accounted_seconds: int) -> None:
        if accounted_seconds <= 0:
            return
        self.last_accounted_at += accounted_seconds

    def reset_baseline(self) -> None:
        self.last_accounted_at = self._clock()

    def mark_active_goal(self, goal_id: str) -> None:
        if self.active_goal_id != goal_id:
            self.active_goal_id = goal_id
            self.reset_baseline()

    def clear_active_goal(self) -> None:
        self.active_goal_id = None
        self.reset_baseline()


class GoalAccountingState:
    """Per-thread runtime accounting state.

    Token usage arrives as per-provider deltas in this Python codebase, so
    this state stores a running per-turn total and flushes only the unaccounted
    portion at safe lifecycle points.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._progress_accounting_lock = threading.Lock()
        self._current_turn_id: str | None = None
        self._turns: dict[str, _GoalTurnAccounting] = {}
        self._wall_clock = _GoalWallClockAccounting(self._clock)
        self._budget_limit_reported_goal_id: str | None = None

    def start_turn(self, turn_id: str, *, plan_mode: bool) -> None:
        with self._lock:
            previous_turn_id = self._current_turn_id
            if previous_turn_id is not None and previous_turn_id != turn_id:
                self._turns.pop(previous_turn_id, None)
            self._current_turn_id = turn_id
            self._turns[turn_id] = _GoalTurnAccounting(account_tokens=not plan_mode)

    def current_turn_id(self) -> str | None:
        with self._lock:
            return self._current_turn_id

    @contextmanager
    def progress_accounting_permit(self) -> Iterator[None]:
        self._progress_accounting_lock.acquire()
        try:
            yield
        finally:
            self._progress_accounting_lock.release()

    def turn_is_current_active_goal(self, turn_id: str) -> bool:
        with self._lock:
            turn = self._turns.get(turn_id)
            return (
                self._current_turn_id == turn_id
                and turn is not None
                and turn.account_tokens
                and turn.active_goal_id is not None
            )

    def turn_started_goal_id(self, turn_id: str) -> str | None:
        """Return the goal bound when ``turn_id`` began.

        ``active_goal_id`` may change when the user replaces a goal while a
        provider request is in flight.  Completion evaluation must use this
        immutable start-of-turn binding so the old response cannot satisfy
        the replacement goal.
        """

        with self._lock:
            turn = self._turns.get(turn_id)
            return None if turn is None else turn.started_goal_id

    def record_token_usage(self, turn_id: str, usage: dict[str, Any] | None) -> int | None:
        token_delta = goal_token_delta_for_usage(usage or {})
        if token_delta <= 0:
            return None
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None or not turn.account_tokens:
                return None
            turn.current_token_total += token_delta
            return turn.token_delta_since_last_accounting()

    def bind_goal_to_turn(self, turn_id: str, goal_id: str) -> None:
        """Bind the goal active at turn start without resetting its baseline."""

        with self._lock:
            if self._budget_limit_reported_goal_id != goal_id:
                self._budget_limit_reported_goal_id = None
            turn = self._turns.get(turn_id)
            if turn is None:
                return
            turn.active_goal_id = goal_id
            if turn.started_goal_id is None:
                turn.started_goal_id = goal_id
            if self._current_turn_id == turn_id and turn.account_tokens:
                self._wall_clock.mark_active_goal(goal_id)

    def mark_current_turn_goal_active(self, goal_id: str) -> str | None:
        with self._lock:
            turn_id = self._current_turn_id
            if turn_id is None:
                return None
            turn = self._turns.get(turn_id)
            if turn is None or not turn.account_tokens:
                return None
            if self._budget_limit_reported_goal_id != goal_id:
                self._budget_limit_reported_goal_id = None
            turn.active_goal_id = goal_id
            turn.last_accounted_token_total = turn.current_token_total
            self._wall_clock.mark_active_goal(goal_id)
            return turn_id

    def mark_idle_goal_active(self, goal_id: str) -> None:
        with self._lock:
            if self._budget_limit_reported_goal_id != goal_id:
                self._budget_limit_reported_goal_id = None
            self._wall_clock.mark_active_goal(goal_id)

    def clear_current_turn_goal(self) -> str | None:
        with self._lock:
            turn_id = self._current_turn_id
            if turn_id is None:
                return None
            turn = self._turns.get(turn_id)
            if turn is not None:
                turn.active_goal_id = None
            self._wall_clock.clear_active_goal()
            self._budget_limit_reported_goal_id = None
            return turn_id

    def clear_active_goal(self) -> None:
        with self._lock:
            turn_id = self._current_turn_id
            if turn_id is not None and turn_id in self._turns:
                self._turns[turn_id].active_goal_id = None
            self._wall_clock.clear_active_goal()
            self._budget_limit_reported_goal_id = None

    def progress_snapshot(self, turn_id: str) -> GoalProgressSnapshot | None:
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None or not turn.account_tokens or turn.active_goal_id is None:
                return None
            token_delta = turn.token_delta_since_last_accounting()
            expected_goal_id = turn.active_goal_id
            time_delta_seconds = (
                self._wall_clock.time_delta_since_last_accounting()
                if self._wall_clock.active_goal_id == expected_goal_id
                else 0
            )
            if token_delta <= 0 and time_delta_seconds <= 0:
                return None
            return GoalProgressSnapshot(
                current_token_total=turn.current_token_total,
                expected_goal_id=expected_goal_id,
                time_delta_seconds=time_delta_seconds,
                token_delta=token_delta,
            )

    def idle_progress_snapshot(self) -> IdleGoalProgressSnapshot | None:
        with self._lock:
            expected_goal_id = self._wall_clock.active_goal_id
            if expected_goal_id is None:
                return None
            time_delta_seconds = self._wall_clock.time_delta_since_last_accounting()
            if time_delta_seconds <= 0:
                return None
            return IdleGoalProgressSnapshot(
                expected_goal_id=expected_goal_id,
                time_delta_seconds=time_delta_seconds,
            )

    def mark_progress_accounted_for_status(
        self,
        turn_id: str,
        snapshot: GoalProgressSnapshot,
        status: ThreadGoalStatus,
        budget_limited_goal_disposition: BudgetLimitedGoalDisposition,
    ) -> None:
        clear_active_goal = _should_clear_active_goal(status, budget_limited_goal_disposition)
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is not None:
                turn.last_accounted_token_total = snapshot.current_token_total
                if clear_active_goal:
                    turn.active_goal_id = None
            self._wall_clock.mark_accounted(snapshot.time_delta_seconds)
            if clear_active_goal:
                self._wall_clock.clear_active_goal()
            if status is not ThreadGoalStatus.BUDGET_LIMITED:
                self._budget_limit_reported_goal_id = None

    def mark_idle_progress_accounted_for_status(
        self,
        snapshot: IdleGoalProgressSnapshot,
        status: ThreadGoalStatus,
        budget_limited_goal_disposition: BudgetLimitedGoalDisposition,
    ) -> None:
        clear_active_goal = _should_clear_active_goal(status, budget_limited_goal_disposition)
        with self._lock:
            self._wall_clock.mark_accounted(snapshot.time_delta_seconds)
            if clear_active_goal:
                self._wall_clock.clear_active_goal()
            if status is not ThreadGoalStatus.BUDGET_LIMITED:
                self._budget_limit_reported_goal_id = None

    def reset_idle_progress_baseline_and_clear_active_goal(self) -> None:
        with self._lock:
            self._wall_clock.reset_baseline()
            self._wall_clock.clear_active_goal()
            self._budget_limit_reported_goal_id = None

    def finish_turn(self, turn_id: str) -> None:
        """Discard turn-local counters while preserving thread goal state.

        The wall-clock baseline and budget-limit notification marker belong to
        the thread's active goal, which may continue across turn boundaries.
        Lifecycle callers must clear or rebind that state when the goal itself
        stops or changes.
        """

        with self._lock:
            self._turns.pop(turn_id, None)
            if self._current_turn_id == turn_id:
                self._current_turn_id = None

    def mark_budget_limit_reported_if_new(self, goal_id: str) -> bool:
        with self._lock:
            if self._budget_limit_reported_goal_id == goal_id:
                return False
            self._budget_limit_reported_goal_id = goal_id
            return True


def goal_token_delta_for_usage(usage: dict[str, Any]) -> int:
    input_tokens = _non_negative_int(usage.get("input_tokens", 0))
    cached_input_tokens = max(
        _non_negative_int(usage.get("cached_input_tokens", 0)),
        _non_negative_int(usage.get("cache_read_input_tokens", 0)),
    )
    output_tokens = _non_negative_int(usage.get("output_tokens", 0))
    return max(input_tokens - cached_input_tokens, 0) + output_tokens


def _non_negative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _should_clear_active_goal(
    status: ThreadGoalStatus,
    budget_limited_goal_disposition: BudgetLimitedGoalDisposition,
) -> bool:
    if status is ThreadGoalStatus.ACTIVE:
        return False
    if status is ThreadGoalStatus.BUDGET_LIMITED:
        return budget_limited_goal_disposition is BudgetLimitedGoalDisposition.CLEAR_ACTIVE
    return True


__all__ = [
    "BudgetLimitedGoalDisposition",
    "GoalAccountingState",
    "GoalProgressSnapshot",
    "IdleGoalProgressSnapshot",
    "goal_token_delta_for_usage",
]
