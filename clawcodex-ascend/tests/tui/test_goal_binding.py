# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Goal/session binding regressions for the Textual agent bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from clawcodex_ext.agent.conversation import Conversation
from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
from clawcodex_ext.goal.runtime import CONTINUATION_STEERING_MARKER
from clawcodex_ext.goal.service import GoalService
from clawcodex_ext.goal.store import GoalStore, goals_db_filename
from clawcodex_ext.tui.agent_bridge import AgentBridge
from clawcodex_ext.tui.state import AppState
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.registry import ToolRegistry


def _session(session_id: str):
    return SimpleNamespace(
        session_id=session_id,
        conversation=Conversation(),
        save_transcript=lambda: None,
    )


def _bridge(tmp_path, *, session_id: str = "session-1"):
    service = GoalService(store=GoalStore(tmp_path / goals_db_filename()))
    context = ToolContext(workspace_root=tmp_path, goal_service=service)
    state = AppState(model="test", provider="test")
    bridge = AgentBridge(
        post_message=lambda _message: None,
        session=_session(session_id),
        provider=SimpleNamespace(model="test"),
        tool_registry=ToolRegistry(),
        tool_context=context,
        app_state=state,
        run_worker=lambda *_args, **_kwargs: None,
        stream=False,
    )
    return bridge, service, context, state


def test_bridge_binds_session_and_hydrates_existing_goal(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("clawcodex_ext.tui.agent_bridge.time.time", lambda: 100.0)
    service = GoalService(store=GoalStore(tmp_path / goals_db_filename()))
    goal = service.set_goal("session-1", "hydrate this condition")
    service.account_usage(
        "session-1",
        expected_goal_id=goal.goal_id,
        token_delta=8,
        elapsed_seconds=4,
    )
    context = ToolContext(workspace_root=tmp_path, goal_service=service)
    state = AppState(model="test", provider="test")

    AgentBridge(
        post_message=lambda _message: None,
        session=_session("session-1"),
        provider=SimpleNamespace(model="test"),
        tool_registry=ToolRegistry(),
        tool_context=context,
        app_state=state,
        run_worker=lambda *_args, **_kwargs: None,
        stream=False,
    )

    assert context.session_id == "session-1"
    assert state.goal_status is not None
    assert state.goal_status["status"] == "active"
    assert state.goal_status["objective"] == "hydrate this condition"
    assert state.goal_status["tokensUsed"] == 8
    assert state.goal_status["activeSince"] == 96.0


def test_bridge_startup_resume_resets_persisted_progress(tmp_path) -> None:
    service = GoalService(store=GoalStore(tmp_path / goals_db_filename()))
    goal = service.set_goal("session-1", "resume this condition")
    service.account_usage(
        "session-1",
        expected_goal_id=goal.goal_id,
        token_delta=12,
        elapsed_seconds=5,
    )
    context = ToolContext(workspace_root=tmp_path, goal_service=service)
    state = AppState(model="test", provider="test")

    AgentBridge(
        post_message=lambda _message: None,
        session=_session("session-1"),
        provider=SimpleNamespace(model="test"),
        tool_registry=ToolRegistry(),
        tool_context=context,
        app_state=state,
        run_worker=lambda *_args, **_kwargs: None,
        stream=False,
        reset_goal_progress=True,
    )

    assert state.goal_status is not None
    assert state.goal_status["objective"] == "resume this condition"
    assert state.goal_status["tokensUsed"] == 0
    assert service.get_goal("session-1").time_used_seconds == 0


def test_bridge_tracks_tool_and_runtime_goal_mutations(tmp_path) -> None:
    _bridge_obj, service, _context, state = _bridge(tmp_path)
    goal = service.set_goal("session-1", "stay synchronized")

    service.account_usage(
        "session-1",
        expected_goal_id=goal.goal_id,
        token_delta=9,
        elapsed_seconds=3,
    )
    service.update_goal_from_runtime(
        "session-1",
        ThreadGoalStatus.COMPLETE,
        expected_goal_id=goal.goal_id,
    )

    assert state.goal_status is not None
    assert state.goal_status["status"] == "complete"
    assert state.goal_status["tokensUsed"] == 9

    service.clear_goal("session-1")
    assert state.goal_status is None


def test_replace_session_rebinds_goal_subscription(tmp_path) -> None:
    bridge, service, context, state = _bridge(tmp_path)
    old = service.set_goal("session-1", "old session")
    resumed = service.set_goal("session-2", "resumed session")
    service.account_usage(
        "session-2",
        expected_goal_id=resumed.goal_id,
        token_delta=17,
        elapsed_seconds=6,
    )

    assert bridge.replace_session(_session("session-2")) is True

    assert context.session_id == "session-2"
    assert state.goal_status is not None
    assert state.goal_status["objective"] == "resumed session"
    assert state.goal_status["tokensUsed"] == 0
    assert service.get_goal("session-2").time_used_seconds == 0

    service.update_goal_from_runtime(
        "session-1",
        ThreadGoalStatus.COMPLETE,
        expected_goal_id=old.goal_id,
    )
    assert state.goal_status["objective"] == "resumed session"


def test_replace_session_refuses_while_agent_is_busy(tmp_path) -> None:
    bridge, service, context, state = _bridge(tmp_path)
    service.set_goal("session-1", "keep current session")
    resumed = service.set_goal("session-2", "do not bind while busy")
    service.account_usage(
        "session-2",
        expected_goal_id=resumed.goal_id,
        token_delta=17,
        elapsed_seconds=6,
    )

    assert bridge.submit("in-flight turn") is True
    assert bridge.replace_session(_session("session-2")) is False

    assert context.session_id == "session-1"
    assert state.goal_status is not None
    assert state.goal_status["objective"] == "keep current session"
    resumed_after = service.get_goal("session-2")
    assert resumed_after is not None
    assert resumed_after.tokens_used == 17
    assert resumed_after.time_used_seconds == 6


def test_replace_session_refuses_missing_session(tmp_path) -> None:
    bridge, service, context, state = _bridge(tmp_path)
    service.set_goal("session-1", "keep current session")

    assert bridge.replace_session(None) is False

    assert context.session_id == "session-1"
    assert state.goal_status is not None
    assert state.goal_status["objective"] == "keep current session"


def test_replace_runtime_rebinds_goal_service_and_session(tmp_path) -> None:
    bridge, old_service, _old_context, state = _bridge(tmp_path)
    old_service.set_goal("session-1", "old runtime")

    new_service = GoalService(store=GoalStore(tmp_path / "new-runtime" / goals_db_filename()))
    new_service.set_goal("session-1", "new runtime")
    new_context = ToolContext(workspace_root=tmp_path, goal_service=new_service)

    bridge.replace_runtime(
        provider=SimpleNamespace(model="new"),
        tool_registry=ToolRegistry(),
        tool_context=new_context,
    )

    assert new_context.session_id == "session-1"
    assert state.goal_status is not None
    assert state.goal_status["objective"] == "new runtime"


def test_continue_goal_if_idle_claims_runtime_and_starts_worker(tmp_path) -> None:
    service = GoalService(store=GoalStore(tmp_path / goals_db_filename()))
    service.set_goal("session-1", "start immediately")
    context = ToolContext(workspace_root=tmp_path, goal_service=service)
    state = AppState(model="test", provider="test")
    worker_calls = []
    session = _session("session-1")
    bridge = AgentBridge(
        post_message=lambda _message: None,
        session=session,
        provider=SimpleNamespace(model="test"),
        tool_registry=ToolRegistry(),
        tool_context=context,
        app_state=state,
        run_worker=lambda *args, **kwargs: worker_calls.append((args, kwargs)),
        stream=False,
    )

    assert bridge.continue_goal_if_idle() is True
    assert worker_calls
    assert CONTINUATION_STEERING_MARKER in str(session.conversation.messages[-1].content)


def test_evaluator_goal_disables_default_turn_cap(tmp_path) -> None:
    bridge, service, _context, _state = _bridge(tmp_path)
    service.replace_goal(
        "session-1",
        "continue until evaluated",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )

    assert bridge._effective_max_turns() == 0

    service.clear_goal("session-1")
    assert bridge._effective_max_turns() == 20


def test_bridge_accounts_aggregate_usage_when_goal_evaluator_fails(
    tmp_path,
    monkeypatch,
) -> None:
    from clawcodex_ext.goal.evaluator import GoalEvaluationError
    from clawcodex_ext.tui.messages import AgentRunFinished
    from clawcodex_ext.utils.abort_controller import AbortController

    posted = []
    service = GoalService(store=GoalStore(tmp_path / goals_db_filename()))
    context = ToolContext(workspace_root=tmp_path, goal_service=service)
    state = AppState(model="test", provider="test")
    session = _session("session-1")
    bridge = AgentBridge(
        post_message=posted.append,
        session=session,
        provider=SimpleNamespace(model="test"),
        tool_registry=ToolRegistry(),
        tool_context=context,
        app_state=state,
        run_worker=lambda *_args, **_kwargs: None,
        stream=False,
    )
    failure = GoalEvaluationError(
        "goal evaluator response is not valid JSON",
        usage={"input_tokens": 3, "output_tokens": 1},
    )
    failure.aggregate_usage = {"input_tokens": 5, "output_tokens": 3}
    failure.num_turns = 1

    async def _fail_goal_run(**_kwargs):
        raise failure

    monkeypatch.setattr(
        "clawcodex_ext.tui.agent_bridge.build_effective_system_prompt",
        lambda *_args, **_kwargs: "system",
    )
    monkeypatch.setattr(
        "clawcodex_ext.tui.agent_bridge.run_query_as_agent_loop",
        _fail_goal_run,
    )
    bridge._busy = True
    bridge._abort_controller = AbortController()

    bridge._run_agent_in_thread()

    assert state.usage["input_tokens"] == 5
    assert state.usage["output_tokens"] == 3
    finished = next(message for message in posted if isinstance(message, AgentRunFinished))
    assert finished.error is None
    assert finished.num_turns == 1
    assert finished.usage == {"input_tokens": 5, "output_tokens": 3}


def test_bridge_persists_complete_goal_notice(tmp_path, monkeypatch) -> None:
    from clawcodex_ext.query.agent_loop_compat import AgentLoopRunResult
    from clawcodex_ext.types.messages import SystemMessage
    from clawcodex_ext.utils.abort_controller import AbortController

    bridge, _service, _context, _state = _bridge(tmp_path)
    notice = SystemMessage(
        content="Goal check: not achieved",
        subtype="goal_evaluation",
        level="info",
        data={"goalId": "goal-1", "state": "active", "met": False},
        usage={"input_tokens": 3, "output_tokens": 1},
    )

    async def _emit_notice(**kwargs):
        kwargs["on_message"](notice)
        return AgentLoopRunResult(
            response_text="main response",
            usage={"input_tokens": 5, "output_tokens": 2},
            num_turns=1,
        )

    monkeypatch.setattr(
        "clawcodex_ext.tui.agent_bridge.build_effective_system_prompt",
        lambda *_args, **_kwargs: "system",
    )
    monkeypatch.setattr(
        "clawcodex_ext.tui.agent_bridge.run_query_as_agent_loop",
        _emit_notice,
    )
    bridge._busy = True
    bridge._abort_controller = AbortController()

    bridge._run_agent_in_thread()

    persisted = next(
        message
        for message in bridge._session.conversation.messages
        if getattr(message, "subtype", None) == "goal_evaluation"
    )
    assert persisted is notice
    assert persisted.data == {"goalId": "goal-1", "state": "active", "met": False}
    assert persisted.usage == {"input_tokens": 3, "output_tokens": 1}


def test_tui_resume_replays_goal_evaluator_error(tmp_path) -> None:
    from unittest.mock import Mock

    from clawcodex_ext.tui.app import ClawCodexTUI
    from clawcodex_ext.types.messages import SystemMessage

    conversation = Conversation()
    conversation.add_existing_message(
        SystemMessage(
            content="Goal evaluator failed: invalid JSON",
            subtype="goal_evaluator_error",
            level="warning",
        )
    )
    transcript = SimpleNamespace(
        append_system=Mock(),
        append_user=Mock(),
        append_assistant=Mock(),
    )
    holder = SimpleNamespace(
        tool_context=SimpleNamespace(agent_type=""),
        session=SimpleNamespace(conversation=conversation),
        _repl_screen=SimpleNamespace(transcript=transcript),
        _replay_exit_snapshot_from_start=True,
    )

    ClawCodexTUI._replay_history_MARKER(holder)

    transcript.append_system.assert_called_once_with(
        "Goal evaluator failed: invalid JSON",
        style="error",
        render="markdown",
    )
