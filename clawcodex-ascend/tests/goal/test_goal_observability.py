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

# pylint: disable=no-name-in-module

"""Spec-6 goal observability tests."""

from __future__ import annotations

from pathlib import Path

from clawcodex_ext.goal.model import ThreadGoalStatus
from clawcodex_ext.goal.observability import GoalObservationRecorder
from clawcodex_ext.goal.runtime import GoalRuntime
from clawcodex_ext.goal.service import GoalService
from clawcodex_ext.goal.store import GoalStore, goals_db_filename


def _service(tmp_path: Path, recorder: GoalObservationRecorder) -> GoalService:
    return GoalService(
        store=GoalStore(tmp_path / goals_db_filename()),
        observer=recorder,
    )


def test_service_records_goal_lifecycle_and_accounting_events(tmp_path: Path) -> None:
    recorder = GoalObservationRecorder()
    service = _service(tmp_path, recorder)

    goal = service.set_goal("thread-1", "observe", token_budget=5)
    service.pause_goal("thread-1")
    service.resume_goal("thread-1")
    service.account_usage(
        "thread-1",
        expected_goal_id=goal.goal_id,
        token_delta=5,
        elapsed_seconds=7,
    )
    service.update_goal("thread-1", ThreadGoalStatus.COMPLETE, expected_goal_id=goal.goal_id)

    observations = recorder.observations
    assert [item.kind for item in observations] == [
        "created",
        "paused",
        "resumed",
        "usage_accounted",
        "budget_limited",
        "complete",
    ]
    usage = observations[3]
    assert usage.token_delta == 5
    assert usage.time_delta_seconds == 7
    assert usage.cumulative_tokens == 5


def test_runtime_records_continuation_skipped_reasons(tmp_path: Path) -> None:
    recorder = GoalObservationRecorder()
    service = _service(tmp_path, recorder)
    runtime = GoalRuntime(thread_id="thread-1", service=service, observer=recorder)
    service.register_runtime(runtime)

    assert runtime.continue_if_idle() is None
    service.set_goal("thread-1", "pause")
    service.pause_goal("thread-1")
    assert runtime.continue_if_idle() is None
    service.resume_goal("thread-1")
    runtime.on_turn_start("turn-1")
    assert runtime.continue_if_idle() is None

    skipped = [item.reason for item in recorder.observations if item.kind == "continuation_skipped"]
    assert skipped == ["no_active_goal", "paused", "not_idle"]


def test_runtime_records_feature_disabled_skip_reason(tmp_path: Path) -> None:
    recorder = GoalObservationRecorder()
    service = _service(tmp_path, recorder)
    runtime = GoalRuntime(
        thread_id="thread-1",
        service=service,
        is_enabled=lambda: False,
        observer=recorder,
    )

    assert runtime.continue_if_idle() is None

    assert recorder.observations[-1].kind == "continuation_skipped"
    assert recorder.observations[-1].reason == "feature_disabled"
