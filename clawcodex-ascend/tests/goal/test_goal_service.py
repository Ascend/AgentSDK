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

# pylint: disable=no-name-in-module,use-implicit-booleaness-not-comparison

"""GoalService parity tests for Spec 3."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from clawcodex_ext.feature_gate import get_registry, reset_registry
from clawcodex_ext.goal.gate import ensure_goals_feature_registered
from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
from clawcodex_ext.goal.files import (
    GOAL_FILE_NAME,
    MAX_THREAD_GOAL_OBJECTIVE_CHARS,
    objective_text_for_edit,
)
from clawcodex_ext.goal.service import (
    GoalService,
    GoalServiceError,
    clear_goal_for_context,
    goal_thread_id_from_context,
)
from clawcodex_ext.goal.store import GoalStore, goals_db_filename


def make_service(tmp_path: Path, *, codex_home: Path | None = None) -> GoalService:
    return GoalService(
        store=GoalStore(tmp_path / goals_db_filename()),
        codex_home=codex_home,
    )


@pytest.fixture(autouse=True)
def _fresh_feature_registry():
    reset_registry()
    ensure_goals_feature_registered()
    yield
    reset_registry()


def test_set_goal_creates_active_goal_and_get_reads_it(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    goal = service.set_goal("thread-1", "ship spec 3")

    assert goal.thread_id == "thread-1"
    assert goal.objective == "ship spec 3"
    assert goal.status is ThreadGoalStatus.ACTIVE
    assert goal.tokens_used == 0
    assert service.get_goal("thread-1") == goal


def test_set_goal_materializes_long_objective_before_persisting(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    service = make_service(tmp_path, codex_home=codex_home)
    objective = "ship spec 6\n" + ("x" * MAX_THREAD_GOAL_OBJECTIVE_CHARS)

    goal = service.set_goal("thread-1", objective)

    assert goal.objective.startswith("Read the Codex goal objective file at ")
    assert objective_text_for_edit(goal.objective, codex_home=codex_home) == objective
    goal_files = list((codex_home / "attachments").glob(f"*/{GOAL_FILE_NAME}"))
    assert len(goal_files) == 1


def test_replace_and_clear_remove_owned_goal_attachments(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    service = make_service(tmp_path, codex_home=codex_home)
    first = service.set_goal(
        "thread-1",
        "first\n" + ("x" * MAX_THREAD_GOAL_OBJECTIVE_CHARS),
    )
    first_file = next((codex_home / "attachments").glob(f"*/{GOAL_FILE_NAME}"))

    second = service.replace_goal(
        "thread-1",
        "second\n" + ("y" * MAX_THREAD_GOAL_OBJECTIVE_CHARS),
    )
    second_file = next((codex_home / "attachments").glob(f"*/{GOAL_FILE_NAME}"))

    assert first.objective != second.objective
    assert not first_file.exists()
    assert second_file.exists()
    assert service.clear_goal("thread-1")
    assert not second_file.exists()


def test_store_failure_rolls_back_new_goal_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_home = tmp_path / "codex-home"
    service = make_service(tmp_path, codex_home=codex_home)

    def fail_insert(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(service.store, "insert_thread_goal", fail_insert)

    with pytest.raises(RuntimeError, match="database unavailable"):
        service.create_goal(
            "thread-1",
            "new\n" + ("x" * MAX_THREAD_GOAL_OBJECTIVE_CHARS),
        )

    assert not list((codex_home / "attachments").glob(f"*/{GOAL_FILE_NAME}"))


@pytest.mark.parametrize(
    ("service_method", "store_method"),
    [
        ("set_goal", "update_thread_goal"),
        ("replace_goal", "replace_thread_goal"),
    ],
)
def test_existing_goal_mutation_failure_rolls_back_new_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service_method: str,
    store_method: str,
) -> None:
    codex_home = tmp_path / "codex-home"
    service = make_service(tmp_path, codex_home=codex_home)
    service.create_goal("thread-1", "existing")

    def fail_mutation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(service.store, store_method, fail_mutation)
    with pytest.raises(RuntimeError, match="database unavailable"):
        getattr(service, service_method)(
            "thread-1",
            "replacement\n" + ("x" * MAX_THREAD_GOAL_OBJECTIVE_CHARS),
        )

    assert not list((codex_home / "attachments").glob(f"*/{GOAL_FILE_NAME}"))


def test_set_goal_cas_conflict_rolls_back_new_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_home = tmp_path / "codex-home"
    service = make_service(tmp_path, codex_home=codex_home)
    service.create_goal("thread-1", "existing")
    monkeypatch.setattr(service.store, "update_thread_goal", lambda *_args, **_kwargs: None)

    with pytest.raises(GoalServiceError, match="cannot update goal"):
        service.set_goal(
            "thread-1",
            "replacement\n" + ("x" * MAX_THREAD_GOAL_OBJECTIVE_CHARS),
        )

    assert not list((codex_home / "attachments").glob(f"*/{GOAL_FILE_NAME}"))


def test_replace_goal_resets_usage_and_generates_new_goal_id(tmp_path: Path) -> None:
    from clawcodex_ext.goal.evaluator import GoalEvaluation

    service = make_service(tmp_path)
    first = service.set_goal(
        "thread-1",
        "old",
        token_budget=20,
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    accounted = service.account_usage(
        "thread-1",
        expected_goal_id=first.goal_id,
        token_delta=12,
        elapsed_seconds=5,
    )
    service.record_evaluation(
        "thread-1",
        GoalEvaluation(met=False, reason="not yet", usage={}),
        expected_goal_id=first.goal_id,
        expected_evaluation_count=0,
    )

    replacement = service.replace_goal("thread-1", "new")

    assert accounted is not None
    assert replacement.goal_id != first.goal_id
    assert replacement.objective == "new"
    assert replacement.tokens_used == 0
    assert replacement.time_used_seconds == 0
    assert replacement.status is ThreadGoalStatus.ACTIVE
    assert replacement.evaluation_count == 0
    assert replacement.last_evaluation_reason is None


def test_model_tool_goal_defaults_to_tool_completion_mode(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    created = service.create_goal("thread-1", "created by model tool")

    assert created.completion_mode is GoalCompletionMode.TOOL


def test_record_evaluation_publishes_snapshot_and_rejects_stale_goal(
    tmp_path: Path,
) -> None:
    from clawcodex_ext.goal.evaluator import GoalEvaluation

    service = make_service(tmp_path)
    stale = service.set_goal("thread-1", "old", completion_mode=GoalCompletionMode.EVALUATOR)
    current = service.replace_goal("thread-1", "new", completion_mode=GoalCompletionMode.EVALUATOR)
    snapshots = []
    service.subscribe("thread-1", snapshots.append)

    rejected = service.record_evaluation(
        "thread-1",
        GoalEvaluation(met=True, reason="stale result", usage={}),
        expected_goal_id=stale.goal_id,
        expected_evaluation_count=0,
    )
    recorded = service.record_evaluation(
        "thread-1",
        GoalEvaluation(met=False, reason="needs another turn", usage={}),
        expected_goal_id=current.goal_id,
        expected_evaluation_count=0,
    )
    rejected_version = service.record_evaluation(
        "thread-1",
        GoalEvaluation(met=False, reason="stale concurrent result", usage={}),
        expected_goal_id=current.goal_id,
        expected_evaluation_count=0,
    )

    assert rejected is None
    assert rejected_version is None
    assert recorded is not None
    assert recorded.status is ThreadGoalStatus.ACTIVE
    assert recorded.evaluation_count == 1
    assert recorded.last_evaluation_reason == "needs another turn"
    assert service.get_goal("thread-1") == recorded
    assert snapshots == [recorded]


def test_update_goal_uses_expected_goal_id_for_cas(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    stale = service.set_goal("thread-1", "old")
    current = service.replace_goal("thread-1", "new")

    result = service.update_goal(
        "thread-1",
        ThreadGoalStatus.COMPLETE,
        expected_goal_id=stale.goal_id,
    )

    assert result is None
    assert service.get_goal("thread-1") == current


def test_pause_and_resume_go_through_status_updates(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.set_goal("thread-1", "pause me")

    paused = service.pause_goal("thread-1")
    resumed = service.resume_goal("thread-1")

    assert paused is not None
    assert paused.status is ThreadGoalStatus.PAUSED
    assert resumed is not None
    assert resumed.status is ThreadGoalStatus.ACTIVE


def test_goal_thread_id_prefers_explicit_context_over_tool_session() -> None:
    context = SimpleNamespace(
        goal_thread_id="explicit-thread",
        tool_context=SimpleNamespace(session_id="tool-session"),
    )

    assert goal_thread_id_from_context(context) == "explicit-thread"


def test_goal_thread_id_falls_back_to_tool_context_session() -> None:
    context = SimpleNamespace(tool_context=SimpleNamespace(session_id="tool-session"))

    assert goal_thread_id_from_context(context) == "tool-session"


def test_resume_budget_limited_goal_stays_limited_while_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    goal = service.set_goal("thread-1", "budget", token_budget=10)
    limited = service.account_usage(
        "thread-1",
        expected_goal_id=goal.goal_id,
        token_delta=10,
        elapsed_seconds=1,
    )

    resumed = service.resume_goal("thread-1")

    assert limited is not None
    assert limited.status is ThreadGoalStatus.BUDGET_LIMITED
    assert resumed is not None
    assert resumed.status is ThreadGoalStatus.BUDGET_LIMITED


def test_clear_goal_returns_boolean_and_removes_goal(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.set_goal("thread-1", "clear me")

    assert service.clear_goal("thread-1") is True
    assert service.get_goal("thread-1") is None
    assert service.clear_goal("thread-1") is False


def test_service_rejects_all_operations_when_goal_feature_disabled(tmp_path: Path) -> None:
    ensure_goals_feature_registered()
    get_registry().set_override("goals", False)
    service = make_service(tmp_path)

    with pytest.raises(GoalServiceError, match="goals feature is disabled"):
        service.set_goal("thread-1", "nope")

    with pytest.raises(GoalServiceError, match="goals feature is disabled"):
        service.get_goal("thread-1")


def test_service_rejects_empty_objective_and_negative_budget(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    with pytest.raises(GoalServiceError, match="goal objective cannot be empty"):
        service.create_goal("thread-1", "  ")

    with pytest.raises(GoalServiceError, match="token budget must be non-negative"):
        service.create_goal("thread-1", "negative budget", token_budget=-1)


def test_subscribe_emits_current_and_all_committed_goal_snapshots(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    snapshots: list[tuple[str, int] | None] = []

    unsubscribe = service.subscribe(
        "thread-1",
        lambda goal: snapshots.append(None if goal is None else (goal.status.value, goal.tokens_used)),
        emit_current=True,
    )
    goal = service.set_goal("thread-1", "keep UI current")
    service.account_usage(
        "thread-1",
        expected_goal_id=goal.goal_id,
        token_delta=7,
        elapsed_seconds=2,
    )
    service.update_goal(
        "thread-1",
        ThreadGoalStatus.COMPLETE,
        expected_goal_id=goal.goal_id,
    )
    service.clear_goal("thread-1")

    assert snapshots == [
        None,
        ("active", 0),
        ("active", 7),
        ("complete", 7),
        None,
    ]

    unsubscribe()
    service.set_goal("thread-1", "not observed")
    assert snapshots[-1] is None


def test_subscriber_failures_do_not_rollback_goal_mutations(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    def fail(_goal) -> None:
        raise RuntimeError("UI listener failed")

    service.subscribe("thread-1", fail)

    goal = service.set_goal("thread-1", "mutation survives")

    assert service.get_goal("thread-1") == goal


def test_failed_cas_update_does_not_publish_snapshot(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    stale = service.set_goal("thread-1", "old")
    service.replace_goal("thread-1", "new")
    snapshots = []
    service.subscribe("thread-1", snapshots.append)

    result = service.update_goal(
        "thread-1",
        ThreadGoalStatus.COMPLETE,
        expected_goal_id=stale.goal_id,
    )

    assert result is None
    assert snapshots == []


def test_clear_goal_for_context_removes_session_goal(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.set_goal("thread-1", "clear with conversation")
    context = SimpleNamespace(session_id="thread-1", goal_service=service)

    assert clear_goal_for_context(context) is True
    assert service.get_goal("thread-1") is None


def test_clear_goal_for_context_propagates_store_failure() -> None:
    class FailingService:
        def clear_goal(self, thread_id: str) -> bool:
            raise RuntimeError(f"cannot clear {thread_id}")

    context = SimpleNamespace(
        session_id="thread-1",
        goal_service=FailingService(),
    )

    with pytest.raises(RuntimeError, match="cannot clear thread-1"):
        clear_goal_for_context(context)


def test_reset_progress_for_resume_preserves_active_condition_and_identity(
    tmp_path: Path,
) -> None:
    from clawcodex_ext.goal.evaluator import GoalEvaluation

    service = make_service(tmp_path)
    goal = service.set_goal(
        "thread-1",
        "resume condition",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    service.account_usage(
        "thread-1",
        expected_goal_id=goal.goal_id,
        token_delta=21,
        elapsed_seconds=8,
    )
    service.record_evaluation(
        "thread-1",
        GoalEvaluation(met=False, reason="continue", usage={}),
        expected_goal_id=goal.goal_id,
        expected_evaluation_count=0,
    )

    resumed = service.reset_progress_for_resume(
        "thread-1",
        expected_goal_id=goal.goal_id,
    )

    assert resumed is not None
    assert resumed.goal_id == goal.goal_id
    assert resumed.objective == "resume condition"
    assert resumed.status is ThreadGoalStatus.ACTIVE
    assert resumed.tokens_used == 0
    assert resumed.time_used_seconds == 0
    assert resumed.evaluation_count == 0
    assert resumed.last_evaluation_reason is None
