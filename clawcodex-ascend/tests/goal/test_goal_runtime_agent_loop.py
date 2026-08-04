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
"""Goal runtime continuation and lifecycle tests for F-122 Spec 5."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from clawcodex_ext.goal.evaluator import GoalEvaluationError
from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
from clawcodex_ext.goal.runtime import (
    CONTINUATION_STEERING_MARKER,
    GoalRuntime,
)
from clawcodex_ext.goal.steering import EVALUATOR_CONTINUATION_MARKER
from clawcodex_ext.goal.accounting import (
    GoalAccountingState,
)
from clawcodex_ext.goal.service import GoalService
from clawcodex_ext.goal.store import GoalStore, goals_db_filename
from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.defaults import build_default_registry
from clawcodex_ext.types.messages import UserMessage
from src.query.agent_loop_compat import run_query_as_agent_loop


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


def test_agent_loop_continues_active_goal_until_update_goal_complete(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    service.replace_goal("thread-1", "finish via continuation")
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )
    registry = build_default_registry()
    provider = MagicMock()
    provider.chat_stream_response.side_effect = NotImplementedError()
    provider.chat.side_effect = [
        ChatResponse(
            content="Initial turn done.",
            model="test",
            usage={"input_tokens": 2, "output_tokens": 2},
            finish_reason="end_turn",
            tool_uses=None,
        ),
        ChatResponse(
            content="Completing the goal.",
            model="test",
            usage={"input_tokens": 4, "output_tokens": 1},
            finish_reason="tool_use",
            tool_uses=[
                {
                    "id": "toolu_goal",
                    "name": "update_goal",
                    "input": {"status": "complete"},
                }
            ],
        ),
        ChatResponse(
            content="Goal complete.",
            model="test",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="end_turn",
            tool_uses=None,
        ),
    ]

    result = _run(
        run_query_as_agent_loop(
            initial_messages=[UserMessage(content="start")],
            provider=provider,
            tool_registry=registry,
            tool_context=context,
            system_prompt="You are helpful.",
            max_turns=5,
        )
    )

    goal = service.get_goal("thread-1")
    second_call_messages = provider.chat.call_args_list[1].args[0]
    assert goal is not None
    assert goal.status is ThreadGoalStatus.COMPLETE
    # The two responses through the completion tool are goal work.  The
    # final explanatory response happens after completion and is excluded.
    assert goal.tokens_used == 9
    assert result.response_text == "Goal complete."
    assert provider.chat.call_count == 3
    assert any(
        CONTINUATION_STEERING_MARKER in str(message.get("content", ""))
        for message in second_call_messages
        if message.get("role") == "user"
    )


def test_agent_loop_uses_independent_evaluator_until_condition_is_met(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    service.replace_goal(
        "thread-1",
        "produce verified evidence",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )
    provider = MagicMock()
    provider.chat_stream_response.side_effect = NotImplementedError()
    provider.chat.side_effect = [
        ChatResponse(
            content="I made partial progress.",
            model="test",
            usage={"input_tokens": 2, "output_tokens": 2},
            finish_reason="end_turn",
            tool_uses=None,
        ),
        ChatResponse(
            content="The verified evidence is now present.",
            model="test",
            usage={"input_tokens": 3, "output_tokens": 1},
            finish_reason="end_turn",
            tool_uses=None,
        ),
    ]
    provider.chat_async = AsyncMock(
        side_effect=[
            ChatResponse(
                content='{"met": false, "reason": "Verification evidence is missing."}',
                model="evaluator",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="end_turn",
            ),
            ChatResponse(
                content='{"met": true, "reason": "The transcript now contains verification."}',
                model="evaluator",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="end_turn",
            ),
        ]
    )
    persisted_messages: list[object] = []

    result = _run(
        run_query_as_agent_loop(
            initial_messages=[UserMessage(content="start")],
            provider=provider,
            tool_registry=build_default_registry(),
            tool_context=context,
            system_prompt="You are helpful.",
            max_turns=5,
            on_message=persisted_messages.append,
        )
    )

    goal = service.get_goal("thread-1")
    second_call_messages = provider.chat.call_args_list[1].args[0]
    assert goal is not None
    assert goal.status is ThreadGoalStatus.COMPLETE
    assert goal.evaluation_count == 2
    assert goal.last_evaluation_reason == "The transcript now contains verification."
    assert goal.tokens_used == 12
    assert result.response_text == "The verified evidence is now present."
    assert result.usage == {"input_tokens": 7, "output_tokens": 5}
    assert provider.chat.call_count == 2
    assert provider.chat_async.await_count == 2
    assert [
        getattr(message, "subtype", None)
        for message in persisted_messages
        if getattr(message, "role", None) == "system"
    ] == ["goal_evaluation", "goal_achieved"]
    achieved_notice = next(
        message for message in persisted_messages if getattr(message, "subtype", None) == "goal_achieved"
    )
    assert "2 turns" in str(achieved_notice.content)
    assert "12 tokens" in str(achieved_notice.content)
    assert achieved_notice.data["state"] == "achieved"
    assert achieved_notice.data["met"] is True
    assert achieved_notice.data["turns"] == 2
    assert any(
        EVALUATOR_CONTINUATION_MARKER in str(message.get("content", ""))
        for message in second_call_messages
        if message.get("role") == "user"
    )


def test_evaluator_only_continuation_respects_explicit_max_turns(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    service.replace_goal(
        "thread-1",
        "stop at the explicit safety cap",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )
    provider = MagicMock()
    provider.chat_stream_response.side_effect = NotImplementedError()
    provider.chat.side_effect = [
        ChatResponse(
            content="The condition is not met yet.",
            model="test",
            usage={"input_tokens": 2, "output_tokens": 1},
            finish_reason="end_turn",
            tool_uses=None,
        ),
        ChatResponse(
            content="The condition is still not met.",
            model="test",
            usage={"input_tokens": 3, "output_tokens": 1},
            finish_reason="end_turn",
            tool_uses=None,
        ),
    ]
    provider.chat_async = AsyncMock(
        side_effect=[
            ChatResponse(
                content='{"met": false, "reason": "More work is required."}',
                model="evaluator",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="end_turn",
            ),
            ChatResponse(
                content='{"met": false, "reason": "The condition remains unmet."}',
                model="evaluator",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="end_turn",
            ),
        ]
    )

    result = _run(
        run_query_as_agent_loop(
            initial_messages=[UserMessage(content="start")],
            provider=provider,
            tool_registry=build_default_registry(),
            tool_context=context,
            system_prompt="You are helpful.",
            max_turns=2,
        )
    )

    goal = service.get_goal("thread-1")
    assert result.response_text == "[Max tool turns reached]"
    assert result.num_turns == 2
    assert result.usage == {"input_tokens": 7, "output_tokens": 4}
    assert goal is not None
    assert goal.status is ThreadGoalStatus.ACTIVE
    assert goal.evaluation_count == 2
    assert provider.chat.call_count == 2
    assert provider.chat_async.await_count == 2


def test_goal_evaluator_failure_is_explicit_and_does_not_spin(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.replace_goal(
        "thread-1",
        "evaluate safely",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )
    provider = MagicMock()
    provider.chat_stream_response.side_effect = NotImplementedError()
    provider.chat.return_value = ChatResponse(
        content="Main turn finished.",
        model="test",
        usage={"input_tokens": 2, "output_tokens": 2},
        finish_reason="end_turn",
        tool_uses=None,
    )
    provider.chat_async = AsyncMock(side_effect=RuntimeError("evaluator unavailable"))

    with pytest.raises(GoalEvaluationError, match="evaluator unavailable"):
        _run(
            run_query_as_agent_loop(
                initial_messages=[UserMessage(content="start")],
                provider=provider,
                tool_registry=build_default_registry(),
                tool_context=context,
                system_prompt="You are helpful.",
                max_turns=5,
            )
        )

    goal = service.get_goal("thread-1")
    assert goal is not None
    assert goal.status is ThreadGoalStatus.ACTIVE
    assert goal.evaluation_count == 0
    assert provider.chat.call_count == 1
    assert provider.chat_async.await_count == 1


def test_goal_evaluator_invalid_response_usage_is_accounted(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.replace_goal(
        "thread-1",
        "evaluate safely",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )
    provider = MagicMock()
    provider.chat_stream_response.side_effect = NotImplementedError()
    provider.chat.return_value = ChatResponse(
        content="Main turn finished.",
        model="test",
        usage={"input_tokens": 2, "output_tokens": 2},
        finish_reason="end_turn",
        tool_uses=None,
    )
    provider.chat_async = AsyncMock(
        return_value=ChatResponse(
            content="not json",
            model="evaluator",
            usage={"input_tokens": 3, "output_tokens": 1},
            finish_reason="end_turn",
        )
    )
    persisted_messages: list[object] = []

    with pytest.raises(GoalEvaluationError, match="not valid JSON") as raised:
        _run(
            run_query_as_agent_loop(
                initial_messages=[UserMessage(content="start")],
                provider=provider,
                tool_registry=build_default_registry(),
                tool_context=context,
                system_prompt="You are helpful.",
                max_turns=5,
                on_message=persisted_messages.append,
            )
        )

    goal = service.get_goal("thread-1")
    assert goal is not None
    assert goal.status is ThreadGoalStatus.ACTIVE
    assert goal.tokens_used == 8
    error_notice = next(
        message for message in persisted_messages if getattr(message, "subtype", None) == "goal_evaluator_error"
    )
    assert getattr(error_notice, "usage", None) == {
        "input_tokens": 3,
        "output_tokens": 1,
    }
    assert error_notice.data["state"] == "active"
    assert error_notice.data["met"] is None
    assert error_notice.data["goalId"] == goal.goal_id
    assert raised.value.aggregate_usage == {
        "input_tokens": 5,
        "output_tokens": 3,
    }
    assert raised.value.num_turns == 1
