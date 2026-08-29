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

"""Goal runtime continuation and lifecycle tests for Spec 5."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
from clawcodex_ext.goal.runtime import (
    OBJECTIVE_UPDATED_STEERING_MARKER,
)
from clawcodex_ext.goal.service import GoalService
from clawcodex_ext.goal.store import GoalStore, goals_db_filename
from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.defaults import build_default_registry
from clawcodex_ext.types.messages import UserMessage
from src.query.agent_loop_compat import run_query_as_agent_loop


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


def test_goal_model_api_error_stops_run_and_keeps_goal_active(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.replace_goal(
        "thread-1",
        "retry after the provider recovers",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )
    provider = MagicMock()
    provider.chat_stream_response.side_effect = RuntimeError("Rate limit exceeded")
    provider.chat_async = AsyncMock()

    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
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
    assert provider.chat_stream_response.call_count == 1
    assert provider.chat_async.await_count == 0


def test_goal_unrecoverable_provider_error_propagates_original_error(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    service.replace_goal(
        "thread-1",
        "retry this goal manually",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )
    provider = MagicMock()
    provider.chat_stream_response.side_effect = ValueError("provider rejected request")
    provider.chat_async = AsyncMock()

    with pytest.raises(ValueError, match="provider rejected request") as raised:
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
    assert provider.chat_stream_response.call_count == 1
    assert provider.chat_async.await_count == 0
    assert raised.value.aggregate_usage == {  # type: ignore[attr-defined]
        "input_tokens": 0,
        "output_tokens": 0,
    }
    assert raised.value.num_turns == 0  # type: ignore[attr-defined]


def test_goal_max_output_recovery_exhaustion_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clawcodex_ext.query import query as query_module
    from clawcodex_ext.query.recovery_strategies import (
        RecoveryStrategy,
        _max_output_tokens_exhausted,
    )

    service = make_service(tmp_path)
    service.replace_goal(
        "thread-1",
        "produce the complete response",
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
        content="Partial output only.",
        model="test",
        usage={"input_tokens": 2, "output_tokens": 8},
        finish_reason="max_tokens",
        tool_uses=None,
    )
    provider.chat_async = AsyncMock()
    exhausted = RecoveryStrategy(
        name="test_max_output_tokens_exhausted",
        fn=_max_output_tokens_exhausted,
    )
    monkeypatch.setattr(
        query_module,
        "find_recovery_strategies",
        lambda _error_type, state: [
            exhausted
            if state.max_output_tokens_recovery_count >= 3
            else RecoveryStrategy(
                name="force_exhausted_state",
                fn=lambda ctx: (
                    type(ctx.state)(
                        messages=ctx.state.messages,
                        tool_use_context=ctx.state.tool_use_context,
                        max_output_tokens_recovery_count=3,
                        transition=ctx.state.transition,
                    ),
                    [],
                ),
            )
        ],
    )

    with pytest.raises(RuntimeError, match="output token recovery exhausted"):
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
    assert provider.chat.call_count == 2
    assert provider.chat_async.await_count == 0


def test_replacing_goal_mid_turn_defers_new_goal_evaluation(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    original = service.replace_goal(
        "thread-1",
        "old completion condition",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )
    provider = MagicMock()
    provider.chat_stream_response.side_effect = NotImplementedError()
    replacements = []

    def _main_turn(*_args, **_kwargs):
        if not replacements:
            replacements.append(
                service.replace_goal(
                    "thread-1",
                    "new completion condition",
                    completion_mode=GoalCompletionMode.EVALUATOR,
                )
            )
            return ChatResponse(
                content="OLD-TURN-EVIDENCE",
                model="test",
                usage={"input_tokens": 2, "output_tokens": 2},
                finish_reason="end_turn",
                tool_uses=None,
            )
        return ChatResponse(
            content="NEW-TURN-EVIDENCE",
            model="test",
            usage={"input_tokens": 3, "output_tokens": 1},
            finish_reason="end_turn",
            tool_uses=None,
        )

    provider.chat.side_effect = _main_turn
    provider.chat_async = AsyncMock(
        return_value=ChatResponse(
            content='{"met": true, "reason": "new turn completed the new condition"}',
            model="evaluator",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="end_turn",
        )
    )

    result = _run(
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
    assert goal.goal_id != original.goal_id
    assert goal.goal_id == replacements[0].goal_id
    assert goal.status is ThreadGoalStatus.COMPLETE
    assert goal.evaluation_count == 1
    assert goal.tokens_used == 6
    assert result.response_text == "NEW-TURN-EVIDENCE"
    assert provider.chat.call_count == 2
    assert provider.chat_async.await_count == 1
    second_main_request = provider.chat.call_args_list[1].args[0]
    assert any(
        OBJECTIVE_UPDATED_STEERING_MARKER in str(message.get("content", ""))
        for message in second_main_request
        if message.get("role") == "user"
    )
    evaluator_request = provider.chat_async.await_args.args[0]
    assert "NEW-TURN-EVIDENCE" in str(evaluator_request)
    assert "OLD-TURN-EVIDENCE" not in str(evaluator_request)


def test_goal_replacement_continuation_respects_explicit_max_turns(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    original = service.replace_goal(
        "thread-1",
        "old completion condition",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="thread-1",
        goal_service=service,
    )
    provider = MagicMock()
    provider.chat_stream_response.side_effect = NotImplementedError()

    def _replace_during_first_turn(*_args, **_kwargs):
        service.replace_goal(
            "thread-1",
            "new completion condition",
            completion_mode=GoalCompletionMode.EVALUATOR,
        )
        return ChatResponse(
            content="Output produced for the old condition.",
            model="test",
            usage={"input_tokens": 2, "output_tokens": 2},
            finish_reason="end_turn",
            tool_uses=None,
        )

    provider.chat.side_effect = _replace_during_first_turn
    provider.chat_async = AsyncMock()

    result = _run(
        run_query_as_agent_loop(
            initial_messages=[UserMessage(content="start")],
            provider=provider,
            tool_registry=build_default_registry(),
            tool_context=context,
            system_prompt="You are helpful.",
            max_turns=1,
        )
    )

    goal = service.get_goal("thread-1")
    assert result.response_text == "[Max tool turns reached]"
    assert result.num_turns == 1
    assert goal is not None
    assert goal.goal_id != original.goal_id
    assert goal.objective == "new completion condition"
    assert goal.status is ThreadGoalStatus.ACTIVE
    assert goal.evaluation_count == 0
    assert provider.chat.call_count == 1
    assert provider.chat_async.await_count == 0
