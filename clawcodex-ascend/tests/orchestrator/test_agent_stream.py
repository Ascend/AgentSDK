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
"""Focused tests for Agent Runtime non-terminal stream events."""

from __future__ import annotations

import asyncio
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from extensions.orchestrator.agent_stream import AgentStreamMixin


class _Runner(AgentStreamMixin):
    max_tools_per_turn = 2

    def __init__(self) -> None:
        self.broadcasts: list[object] = []
        self.audit_rows: list[object] = []
        self.spawn_rows: list[object] = []
        self.stop_requested = False
        self.flushes = 0

    async def _broadcast_to_socket(self, _session, event) -> None:
        self.broadcasts.append(event)

    @staticmethod
    def _handle_tool_call(event, _context):
        event._approved = True
        event._deny_reason = None
        return event

    def _append_tool_event_log(self, event, _context) -> None:
        self.audit_rows.append(event)

    def _append_agent_spawn_result_log(self, event, _context) -> None:
        self.spawn_rows.append(event)

    def _drain_control_commands(self, _session) -> bool:
        return self.stop_requested

    def _flush_turn_transcript(self, _session) -> None:
        self.flushes += 1

    @staticmethod
    def _dispatch_sink(*_args) -> None:
        return None


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        issue=SimpleNamespace(id="I-137"),
        output_text="",
        _transcript_asst_text="",
        _transcript_storage=None,
        _transcript_tool_uses=[],
        _transcript_pending_results={},
        _transcript_result_order=[],
        event_queue=asyncio.Queue(),
        control_socket=None,
        has_made_progress=False,
        last_agent_event_at=None,
        last_agent_event=None,
        last_tool_name=None,
        debug_log_path="debug.ndjson",
        run_id="run-1",
        workspace=SimpleNamespace(path="workspace"),
    )


def _run_state() -> SimpleNamespace:
    state = SimpleNamespace(turn_number=0, tool_count=0, sink=None, updates=0)

    def _update(_session) -> None:
        state.updates += 1

    state.update_diagnostics = _update
    return state


def _turn_state(**updates) -> SimpleNamespace:
    values = {
        "output": "",
        "has_tool_calls": False,
        "has_modifying_tool": False,
        "tool_names": [],
        "tool_count": 0,
        "pending_tool_results": 0,
        "cap_reached": False,
        "megaturn_next_check_at": time.monotonic() + 9999,
        "megaturn_workspace_signature": None,
        "megaturn_workspace_changed_at": time.monotonic(),
        "megaturn_stop": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_text_delta_updates_output_and_all_live_consumers() -> None:
    runner = _Runner()
    session = _session()
    dashboard = SimpleNamespace(events=[])
    dashboard.on_event = lambda event, _session: dashboard.events.append(event)
    event = SimpleNamespace(content="hello")

    action = await runner._handle_text_delta_event(session, event, _turn_state(), _run_state(), dashboard)

    assert action == "continue"
    assert session.output_text == "hello"
    assert session._transcript_asst_text == "hello"
    assert dashboard.events == [event]
    assert session.event_queue.get_nowait() is event
    assert runner.broadcasts == [event]


@pytest.mark.asyncio
async def test_tool_call_applies_policy_audit_and_progress() -> None:
    runner = _Runner()
    session = _session()
    turn = _turn_state()
    run = _run_state()
    event = SimpleNamespace(
        tool_name="Write",
        tool_use_id="tool-1",
        params={"file_path": "a.py"},
    )
    context = {}

    await runner._handle_tool_call_event(session, event, turn, run, context)

    assert session.has_made_progress is True
    assert turn.has_modifying_tool is True
    assert turn.tool_names == ["Write"]
    assert turn.pending_tool_results == 1
    assert run.tool_count == 1
    assert context["turn"] == 0
    assert runner.audit_rows == [event]


@pytest.mark.asyncio
async def test_tool_cap_is_set_on_configured_call_count() -> None:
    runner = _Runner()
    session = _session()
    turn = _turn_state(tool_count=1, pending_tool_results=1)
    event = SimpleNamespace(tool_name="Read", tool_use_id="tool-2", params={})
    await runner._handle_tool_call_event(session, event, turn, _run_state(), {})
    assert turn.cap_reached is True
    assert turn.pending_tool_results == 2


@pytest.mark.asyncio
async def test_tool_result_breaks_after_cap_and_last_result() -> None:
    runner = _Runner()
    session = _session()
    turn = _turn_state(cap_reached=True, pending_tool_results=1)
    event = SimpleNamespace(
        tool_name="Read",
        tool_use_id="tool-1",
        result={"output": "ok", "is_error": False},
    )
    action = await runner._handle_tool_result_event(session, event, turn, _run_state(), {})
    assert action == "break"
    assert turn.pending_tool_results == 0


@pytest.mark.asyncio
async def test_tool_result_breaks_for_operator_stop() -> None:
    runner = _Runner()
    runner.stop_requested = True
    action = await runner._handle_tool_result_event(
        _session(),
        SimpleNamespace(tool_name="Read", tool_use_id=None, result={}),
        _turn_state(pending_tool_results=1),
        _run_state(),
        {},
    )
    assert action == "break"


@pytest.mark.asyncio
async def test_agent_result_writes_spawn_attribution() -> None:
    runner = _Runner()
    event = SimpleNamespace(
        tool_name="Agent",
        tool_use_id="spawn-1",
        result={"output": {"agent_id": "child-1"}},
    )
    await runner._handle_tool_result_event(
        _session(),
        event,
        _turn_state(pending_tool_results=2),
        _run_state(),
        {},
    )
    assert runner.spawn_rows == [event]


def test_transcript_tool_result_flushes_in_complete_batch(monkeypatch) -> None:
    messages = ModuleType("extensions.orchestrator_runtime.utils.messages_impl")
    messages.ToolResultBlock = lambda **values: SimpleNamespace(**values)
    monkeypatch.setitem(
        sys.modules,
        "extensions.orchestrator_runtime.utils.messages_impl",
        messages,
    )
    runner = _Runner()
    session = _session()
    session._transcript_storage = object()
    session._transcript_tool_uses = [SimpleNamespace(id="tool-1")]
    event = SimpleNamespace(
        tool_use_id="tool-1",
        result={"output": {"answer": 42}, "is_error": False},
    )
    runner._buffer_tool_result(session, event)
    assert session._transcript_pending_results["tool-1"].content == "{'answer': 42}"
    assert runner.flushes == 1
