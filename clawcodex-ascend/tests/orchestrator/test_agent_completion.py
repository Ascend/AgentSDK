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
"""Focused tests for Agent Runtime SessionComplete decisions."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from extensions.orchestrator.agent_completion import AgentCompletionMixin


@pytest.fixture(autouse=True)
def query_events(monkeypatch):
    module = ModuleType("extensions.api.query")

    class _Event:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)

    module.PhaseComplete = type("PhaseComplete", (_Event,), {})
    module.TurnComplete = type("TurnComplete", (_Event,), {})
    module.SessionComplete = type("SessionComplete", (_Event,), {})
    api_module = ModuleType("extensions.api")
    api_module.query = module
    monkeypatch.setitem(sys.modules, "extensions.api", api_module)
    monkeypatch.setitem(sys.modules, "extensions.api.query", module)
    return module


class _Runner(AgentCompletionMixin):
    def __init__(self, *, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self.agent_config = SimpleNamespace(test_command=None)
        self.broadcasts: list[object] = []
        self.sink_events: list[tuple] = []
        self.rate_limited = False
        self.verification = False
        self.tracker_active = True
        self.workspace_dirty = False

    @staticmethod
    def _drain_control_commands(_session) -> bool:
        return False

    async def _broadcast_to_socket(self, _session, event) -> None:
        self.broadcasts.append(event)

    def _dispatch_sink(self, _sink, method, event, _session) -> None:
        self.sink_events.append((method, event))

    @staticmethod
    def _flush_turn_transcript(_session) -> None:
        return None

    def _is_429_response(self, _output) -> bool:
        return self.rate_limited

    @staticmethod
    async def _handle_rate_limit(_session, _output, _turn, _dashboard) -> str:
        return "running"

    async def _should_continue(self, issue, _tracker, _session):
        return self.tracker_active, issue

    async def _run_verification(self, _session) -> bool:
        return self.verification

    async def _workspace_is_dirty(self, _session) -> bool:
        return self.workspace_dirty

    @staticmethod
    async def _workspace_completion_state(_session):
        return True, True, True


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        issue=SimpleNamespace(id="I-137", state="open"),
        status="running",
        session_end_reason=None,
        session_end_summary="",
        consecutive_429_count=0,
        rate_limit_pending_turn=None,
        total_429_backoff_seconds=0,
        turn_count=0,
        has_made_progress=False,
        _transcript_storage=None,
        control_socket=None,
        run_id="run-1",
        workspace=SimpleNamespace(path="workspace"),
    )


def _run_state(**updates) -> SimpleNamespace:
    values = {
        "sink": None,
        "turn_number": 0,
        "tool_count": 0,
        "no_work_streak": 0,
        "read_only_streak": 0,
        "tool_signature_history": [],
        "max_no_op_turns": 3,
        "loop_window": 5,
        "loop_threshold": 3,
        "consecutive_clean_turns": 0,
        "updates": 0,
    }
    values.update(updates)
    state = SimpleNamespace(**values)
    state.update_diagnostics = lambda _session: setattr(state, "updates", state.updates + 1)
    return state


def _turn_state(**updates) -> SimpleNamespace:
    values = {
        "output": "done",
        "has_tool_calls": True,
        "has_modifying_tool": False,
        "tool_names": ["Read"],
    }
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_rate_limit_retries_same_turn() -> None:
    runner = _Runner()
    runner.rate_limited = True
    state = _run_state()
    action = await runner._handle_session_complete_event(
        _session(),
        SimpleNamespace(reason="exit_code=1"),
        state,
        _turn_state(output="Error code: 429"),
    )
    assert action == "retry"
    assert state.turn_number == 0


@pytest.mark.asyncio
async def test_operator_stop_skips_normal_turn_boundary() -> None:
    runner = _Runner()
    session = _session()
    session.session_end_reason = "operator_stop"
    action = await runner._handle_session_complete_event(
        session,
        SimpleNamespace(reason="success"),
        _run_state(),
        _turn_state(),
    )
    assert action == "return"
    assert runner.broadcasts[-1].reason == "operator_stop"


@pytest.mark.asyncio
async def test_active_tracker_continues_after_successful_turn() -> None:
    runner = _Runner()
    state = _run_state()
    action = await runner._handle_session_complete_event(
        _session(),
        SimpleNamespace(reason="success"),
        state,
        _turn_state(tool_names=["Read", "Grep"]),
        tracker=object(),
    )
    assert action == "continue"
    assert state.turn_number == 1
    assert [item[0] for item in runner.sink_events] == [
        "on_phase_complete",
        "on_turn_complete",
    ]


@pytest.mark.asyncio
async def test_read_only_guard_stops_fourth_turn() -> None:
    runner = _Runner()
    session = _session()
    state = _run_state(read_only_streak=3)
    action = await runner._handle_session_complete_event(
        session,
        SimpleNamespace(reason="success"),
        state,
        _turn_state(),
        tracker=object(),
    )
    assert action == "return"
    assert session.status == "read_only_loop"
    assert session.session_end_reason == "read_only_loop"


@pytest.mark.asyncio
async def test_bash_workspace_change_resets_read_only_streak() -> None:
    runner = _Runner()
    runner.workspace_dirty = True
    session = _session()
    state = _run_state(read_only_streak=3, turn_number=4)

    result = await runner._continuation_guard(
        session,
        state,
        _turn_state(tool_names=["Bash"]),
    )

    assert result is None
    assert state.read_only_streak == 0
    assert session.has_made_progress


@pytest.mark.asyncio
async def test_trackerless_stage_completes_after_analysis() -> None:
    runner = _Runner()
    session = _session()
    state = _run_state(tool_count=3)
    action = await runner._handle_session_complete_event(
        session,
        SimpleNamespace(reason="success"),
        state,
        _turn_state(),
    )
    assert action == "return"
    assert session.status == "completed"
    assert session.session_end_reason == "task_complete"


@pytest.mark.asyncio
async def test_non_success_reason_marks_failed() -> None:
    runner = _Runner()
    session = _session()
    action = await runner._handle_session_complete_event(
        session,
        SimpleNamespace(reason="exit_code=2"),
        _run_state(),
        _turn_state(),
    )
    assert action == "return"
    assert session.status == "failed"
    assert session.session_end_reason == "exit_code=2"


@pytest.mark.asyncio
async def test_bare_non_success_reason_gets_exit_code_prefix() -> None:
    runner = _Runner()
    session = _session()
    action = await runner._handle_session_complete_event(
        session,
        SimpleNamespace(reason="2"),
        _run_state(),
        _turn_state(),
    )

    assert action == "return"
    assert session.status == "failed"
    assert session.session_end_reason == "exit_code=2"


@pytest.mark.asyncio
async def test_max_turns_sets_budget_exhausted() -> None:
    runner = _Runner(max_turns=1)
    session = _session()
    action = await runner._handle_session_complete_event(
        session,
        SimpleNamespace(reason="success"),
        _run_state(),
        _turn_state(),
        tracker=object(),
    )
    assert action == "return"
    assert session.status == "max_turns_exceeded"
    assert session.session_end_reason == "budget_exhausted"
