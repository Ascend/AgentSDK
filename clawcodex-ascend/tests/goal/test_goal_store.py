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
"""SQLite GoalStore parity tests for F-122 Spec 2."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
from clawcodex_ext.goal.store import (
    GoalStore,
    GoalUpdate,
    current_goal_thread_id,
    goals_db_filename,
    goals_db_path,
)
from src.bootstrap.state import SessionId, reset_state_for_tests, switch_session


def make_store(tmp_path: Path) -> GoalStore:
    return GoalStore(tmp_path / goals_db_filename())


def _create_legacy_goal_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE thread_goals (
                thread_id TEXT PRIMARY KEY NOT NULL,
                goal_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                status TEXT NOT NULL,
                token_budget INTEGER,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                time_used_seconds INTEGER NOT NULL DEFAULT 0,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO thread_goals VALUES
            ('thread-1', 'goal-1', 'legacy', 'active', NULL, 5, 2, 1, 1)
            """
        )


def test_goals_db_path_uses_independent_goal_store_file(tmp_path: Path) -> None:
    assert goals_db_filename() == "goals_1.sqlite"
    assert goals_db_path(home=tmp_path) == tmp_path / ".clawcodex" / "goals_1.sqlite"


def test_current_goal_thread_id_uses_recoverable_bootstrap_session_id() -> None:
    reset_state_for_tests()
    try:
        switch_session(SessionId("session-123"))
        assert current_goal_thread_id() == "session-123"
    finally:
        reset_state_for_tests()


def test_schema_bootstrap_matches_upstream_columns_and_constraints(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.close()

    with sqlite3.connect(tmp_path / goals_db_filename()) as conn:
        columns = conn.execute("PRAGMA table_info(thread_goals)").fetchall()
        create_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'thread_goals'"
        ).fetchone()[0]
    completed_field = "completed" + "_at"

    assert [column[1] for column in columns] == [
        "thread_id",
        "goal_id",
        "objective",
        "status",
        "token_budget",
        "tokens_used",
        "time_used_seconds",
        "completion_mode",
        "evaluation_count",
        "last_evaluation_reason",
        "created_at_ms",
        "updated_at_ms",
    ]
    assert columns[0][5] == 1
    assert completed_field not in create_sql
    for status in ThreadGoalStatus:
        assert f"'{status.value}'" in create_sql
    assert "tokens_used INTEGER NOT NULL DEFAULT 0" in create_sql
    assert "time_used_seconds INTEGER NOT NULL DEFAULT 0" in create_sql
    assert "completion_mode TEXT NOT NULL DEFAULT 'tool'" in create_sql
    assert "evaluation_count INTEGER NOT NULL DEFAULT 0" in create_sql


def test_schema_bootstrap_migrates_existing_goal_database(tmp_path: Path) -> None:
    db_path = tmp_path / goals_db_filename()
    _create_legacy_goal_database(db_path)

    store = GoalStore(db_path)
    goal = store.get_thread_goal("thread-1")

    assert goal is not None
    assert goal.evaluation_count == 0
    assert goal.last_evaluation_reason is None
    assert goal.completion_mode is GoalCompletionMode.TOOL


def test_schema_migration_is_idempotent_across_two_store_instances(tmp_path: Path) -> None:
    db_path = tmp_path / goals_db_filename()
    _create_legacy_goal_database(db_path)

    first = GoalStore(db_path)
    second = GoalStore(db_path)
    try:
        assert first.get_thread_goal("thread-1") == second.get_thread_goal("thread-1")
    finally:
        second.close()
        first.close()


def test_schema_migration_is_safe_for_concurrent_first_open(tmp_path: Path) -> None:
    db_path = tmp_path / goals_db_filename()
    _create_legacy_goal_database(db_path)
    ready = Barrier(2)

    def open_store() -> GoalStore:
        ready.wait(timeout=5)
        return GoalStore(db_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(open_store) for _ in range(2)]
        stores = [future.result(timeout=10) for future in futures]

    try:
        goals = [store.get_thread_goal("thread-1") for store in stores]
        assert goals[0] == goals[1]
        assert goals[0] is not None
        assert goals[0].completion_mode is GoalCompletionMode.TOOL
        assert goals[0].evaluation_count == 0
    finally:
        for store in stores:
            store.close()


def test_insert_rejects_unfinished_goal_but_allows_after_complete(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    first = store.insert_thread_goal(
        "thread-1",
        "first objective",
        ThreadGoalStatus.ACTIVE,
        token_budget=100,
    )
    duplicate = store.insert_thread_goal(
        "thread-1",
        "second objective",
        ThreadGoalStatus.ACTIVE,
        token_budget=200,
    )
    assert duplicate is None
    assert store.get_thread_goal("thread-1") == first

    completed = store.update_thread_goal(
        "thread-1",
        GoalUpdate(status=ThreadGoalStatus.COMPLETE),
        expected_goal_id=first.goal_id,
    )
    replacement = store.insert_thread_goal(
        "thread-1",
        "second objective",
        ThreadGoalStatus.ACTIVE,
        token_budget=200,
    )

    assert completed is not None
    assert replacement.goal_id != first.goal_id
    assert replacement.objective == "second objective"
    assert replacement.tokens_used == 0
    assert replacement.time_used_seconds == 0
    assert replacement.evaluation_count == 0
    assert replacement.last_evaluation_reason is None


def test_record_evaluation_uses_goal_id_cas_and_completes_atomically(tmp_path: Path) -> None:
    from clawcodex_ext.goal.evaluator import GoalEvaluation

    store = make_store(tmp_path)
    stale = store.replace_thread_goal("thread-1", "old", ThreadGoalStatus.ACTIVE, token_budget=None)
    current = store.replace_thread_goal(
        "thread-1",
        "new",
        ThreadGoalStatus.ACTIVE,
        token_budget=None,
        completion_mode=GoalCompletionMode.EVALUATOR,
    )

    stale_result = store.record_thread_goal_evaluation(
        "thread-1",
        GoalEvaluation(met=True, reason="stale", usage={}),
        expected_goal_id=stale.goal_id,
        expected_evaluation_count=0,
    )
    first = store.record_thread_goal_evaluation(
        "thread-1",
        GoalEvaluation(met=False, reason="tests still running", usage={}),
        expected_goal_id=current.goal_id,
        expected_evaluation_count=0,
    )
    completed = store.record_thread_goal_evaluation(
        "thread-1",
        GoalEvaluation(met=True, reason="all tests pass", usage={}),
        expected_goal_id=current.goal_id,
        expected_evaluation_count=1,
    )

    assert stale_result is None
    assert first is not None
    assert first.status is ThreadGoalStatus.ACTIVE
    assert first.evaluation_count == 1
    assert first.last_evaluation_reason == "tests still running"
    assert completed is not None
    assert completed.status is ThreadGoalStatus.COMPLETE
    assert completed.evaluation_count == 2
    assert completed.last_evaluation_reason == "all tests pass"


def test_record_evaluation_rejects_concurrent_result_from_same_snapshot(
    tmp_path: Path,
) -> None:
    from clawcodex_ext.goal.evaluator import GoalEvaluation

    db_path = tmp_path / goals_db_filename()
    first_store = GoalStore(db_path)
    second_store = GoalStore(db_path)
    goal = first_store.replace_thread_goal(
        "thread-1",
        "concurrent",
        ThreadGoalStatus.ACTIVE,
        token_budget=None,
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    ready = Barrier(2)

    def record(store: GoalStore, reason: str):
        ready.wait(timeout=5)
        return store.record_thread_goal_evaluation(
            "thread-1",
            GoalEvaluation(met=False, reason=reason, usage={}),
            expected_goal_id=goal.goal_id,
            expected_evaluation_count=0,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(record, first_store, "first result"),
                executor.submit(record, second_store, "second result"),
            ]
            results = [future.result(timeout=10) for future in futures]

        persisted = first_store.get_thread_goal("thread-1")
        assert sum(result is not None for result in results) == 1
        assert persisted is not None
        assert persisted.evaluation_count == 1
        assert persisted.last_evaluation_reason in {"first result", "second result"}
    finally:
        second_store.close()
        first_store.close()


def test_record_evaluation_ignores_tool_completed_goal(tmp_path: Path) -> None:
    from clawcodex_ext.goal.evaluator import GoalEvaluation

    store = make_store(tmp_path)
    goal = store.replace_thread_goal("thread-1", "tool-owned", ThreadGoalStatus.ACTIVE, token_budget=None)

    result = store.record_thread_goal_evaluation(
        "thread-1",
        GoalEvaluation(met=True, reason="would complete", usage={}),
        expected_goal_id=goal.goal_id,
        expected_evaluation_count=0,
    )

    assert result is None
    assert store.get_thread_goal("thread-1") == goal


def test_replace_unconditionally_resets_usage_and_generates_new_goal_id(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    first = store.replace_thread_goal(
        "thread-1",
        "first objective",
        ThreadGoalStatus.ACTIVE,
        token_budget=100,
    )
    accounted = store.account_thread_goal_usage("thread-1", time_delta=5, token_delta=60)
    second = store.replace_thread_goal(
        "thread-1",
        "second objective",
        ThreadGoalStatus.PAUSED,
        token_budget=None,
    )

    assert accounted is not None
    assert accounted.tokens_used == 60
    assert second.goal_id != first.goal_id
    assert second.objective == "second objective"
    assert second.status is ThreadGoalStatus.PAUSED
    assert second.token_budget is None
    assert second.tokens_used == 0
    assert second.time_used_seconds == 0


def test_update_expected_goal_id_prevents_stale_writes(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    original = store.replace_thread_goal(
        "thread-1",
        "old objective",
        ThreadGoalStatus.ACTIVE,
        token_budget=100,
    )
    current = store.replace_thread_goal(
        "thread-1",
        "new objective",
        ThreadGoalStatus.ACTIVE,
        token_budget=200,
    )

    stale = store.update_thread_goal(
        "thread-1",
        GoalUpdate(status=ThreadGoalStatus.COMPLETE, token_budget=50),
        expected_goal_id=original.goal_id,
    )
    fresh = store.get_thread_goal("thread-1")

    assert stale is None
    assert fresh == current


def test_account_usage_expected_goal_id_prevents_stale_usage_and_budget_status(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)

    original = store.replace_thread_goal(
        "thread-1",
        "old objective",
        ThreadGoalStatus.ACTIVE,
        token_budget=10,
    )
    current = store.replace_thread_goal(
        "thread-1",
        "new objective",
        ThreadGoalStatus.ACTIVE,
        token_budget=10,
    )

    stale = store.account_thread_goal_usage(
        "thread-1",
        time_delta=5,
        token_delta=10,
        expected_goal_id=original.goal_id,
    )

    assert stale is None
    assert store.get_thread_goal("thread-1") == current


def test_account_usage_accumulates_and_budget_limits_active_goal(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.replace_thread_goal(
        "thread-1",
        "stay within budget",
        ThreadGoalStatus.ACTIVE,
        token_budget=20,
    )

    first = store.account_thread_goal_usage("thread-1", time_delta=7, token_delta=5)
    second = store.account_thread_goal_usage("thread-1", time_delta=3, token_delta=15)

    assert first is not None
    assert first.status is ThreadGoalStatus.ACTIVE
    assert first.tokens_used == 5
    assert first.time_used_seconds == 7
    assert second is not None
    assert second.status is ThreadGoalStatus.BUDGET_LIMITED
    assert second.tokens_used == 20
    assert second.time_used_seconds == 10


def test_zero_budget_active_goal_is_budget_limited_on_create(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    goal = store.insert_thread_goal(
        "thread-1",
        "zero budget",
        ThreadGoalStatus.ACTIVE,
        token_budget=0,
    )

    assert goal.status is ThreadGoalStatus.BUDGET_LIMITED
    assert goal.tokens_used == 0
    assert goal.time_used_seconds == 0


def test_store_recovers_goal_from_new_instance(tmp_path: Path) -> None:
    db_path = tmp_path / goals_db_filename()
    first_store = GoalStore(db_path)
    goal = first_store.replace_thread_goal(
        "thread-1",
        "persist me",
        ThreadGoalStatus.ACTIVE,
        token_budget=100,
    )
    first_store.close()

    second_store = GoalStore(db_path)

    assert second_store.get_thread_goal("thread-1") == goal


def test_delete_thread_goal_removes_only_that_thread(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = store.replace_thread_goal(
        "thread-1",
        "delete me",
        ThreadGoalStatus.ACTIVE,
        token_budget=None,
    )
    second = store.replace_thread_goal(
        "thread-2",
        "keep me",
        ThreadGoalStatus.ACTIVE,
        token_budget=None,
    )

    deleted = store.delete_thread_goal("thread-1")

    assert deleted == first
    assert store.get_thread_goal("thread-1") is None
    assert store.get_thread_goal("thread-2") == second
