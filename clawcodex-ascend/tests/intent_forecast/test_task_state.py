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

from clawcodex_ext.intent_forecast.task_state import build_task_state, classify_intent_stage


def test_task_state_extracts_blocker_and_pending_tests() -> None:
    state = build_task_state(
        current_messages=[
            {"role": "user", "content": "继续实现 intent forecast"},
            {
                "role": "assistant",
                "content": "pytest failed: tests/intent_forecast/test_service.py",
            },
        ],
        sessions=[],
        workspace={
            "changed_files": ["clawcodex_ext/intent_forecast/service.py"],
            "changed_test_mapping": ["tests/intent_forecast"],
            "last_test_failures": ["tests/intent_forecast/test_service.py failed"],
        },
    )

    assert state["active_goal"] == "继续实现 intent forecast"
    assert state["blocked_reason"] == "tests/intent_forecast/test_service.py failed"
    assert state["pending_tests"] == ["tests/intent_forecast"]
    assert state["next_unfinished_step"].startswith("Fix the recent failure")


def test_intent_stage_prefers_recent_user_document_request_over_dirty_worktree() -> None:
    state = {"pending_tests": ["tests/intent_forecast"], "blocked_reason": ""}

    stage = classify_intent_stage(
        current_messages=[{"role": "user", "content": "基于 feature plan 继续补全文档说明"}],
        task_state=state,
        workspace={"git_status": " M clawcodex_ext/intent_forecast/service.py"},
    )

    assert stage == "document"


def test_intent_stage_enters_debug_when_recent_failure_exists() -> None:
    stage = classify_intent_stage(
        current_messages=[{"role": "assistant", "content": "AssertionError: expected A"}],
        task_state={"blocked_reason": "AssertionError: expected A"},
        workspace={},
    )

    assert stage == "debug"


def test_task_state_prefers_user_intent_over_verbose_assistant_output() -> None:
    messages = [
        {"role": "user", "content": "只写文档"},
        {"role": "assistant", "content": "I inspected code and could implement many files."},
    ]
    user_intent = {"initial_user_input": "只写文档", "latest_user_input": "只写文档"}
    state = build_task_state(
        current_messages=messages,
        sessions=[],
        workspace={"git_status": " M file.py"},
        user_intent=user_intent,
    )
    stage = classify_intent_stage(
        current_messages=messages,
        task_state=state,
        workspace={"git_status": " M file.py"},
        user_intent=user_intent,
    )

    assert state["active_goal"] == "只写文档"
    assert stage == "document"
