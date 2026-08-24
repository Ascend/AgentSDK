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
"""Focused tests for Agent Runtime control and pause behavior."""

from __future__ import annotations

import asyncio
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from extensions.orchestrator.agent_control import AgentControlMixin


class _Socket:
    def __init__(self) -> None:
        self._command_queue: asyncio.Queue = asyncio.Queue()
        self.events: list[dict] = []

    async def send_event(self, event: dict) -> None:
        self.events.append(event)


class _Transcript:
    def __init__(self) -> None:
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1


def _session(tmp_path, *, socket: _Socket | None = None):
    resume_event = asyncio.Event()
    resume_event.set()
    pause_gate = asyncio.Event()
    pause_gate.set()
    return SimpleNamespace(
        control_socket=socket,
        paused=False,
        pause_reason="",
        pause_resume_event=resume_event,
        _pause_gate=pause_gate,
        prompt_override=None,
        status="running",
        session_end_reason=None,
        session_end_summary="",
        _on_pause_state_change=None,
        issue=SimpleNamespace(id="I-137"),
        _transcript_storage=None,
        _runtime_tasks=None,
        run_id="run-1",
        workspace=SimpleNamespace(path=tmp_path),
        turn_count=3,
        last_tool_name="Read",
    )


def test_pause_and_resume_toggle_both_gates(tmp_path) -> None:
    session = _session(tmp_path)

    AgentControlMixin._apply_pause_session(session, "review")
    assert session.paused is True
    assert session.pause_reason == "review"
    assert not session.pause_resume_event.is_set()
    assert not session._pause_gate.is_set()

    AgentControlMixin._apply_resume_session(session, "continue here")
    assert session.paused is False
    assert session.prompt_override == "continue here"
    assert session.pause_resume_event.is_set()
    assert session._pause_gate.is_set()


def test_stop_command_marks_session_failed(tmp_path) -> None:
    socket = _Socket()
    socket._command_queue.put_nowait(SimpleNamespace(cmd="stop", payload=""))
    session = _session(tmp_path, socket=socket)

    assert AgentControlMixin._drain_control_commands(session) is True
    assert session.status == "failed"
    assert session.session_end_reason == "operator_stop"


def test_pause_resume_commands_notify_orchestrator(tmp_path) -> None:
    socket = _Socket()
    socket._command_queue.put_nowait(SimpleNamespace(cmd="pause", payload=""))
    socket._command_queue.put_nowait(SimpleNamespace(cmd="resume", payload="new prompt"))
    session = _session(tmp_path, socket=socket)
    changes: list[tuple] = []
    session._on_pause_state_change = lambda *args: changes.append(args)

    assert AgentControlMixin._drain_control_commands(session) is False
    assert changes == [
        ("I-137", True, "operator_interrupt"),
        ("I-137", False, ""),
    ]
    assert session.prompt_override == "new prompt"


def test_control_drain_callback_translates_stop(tmp_path) -> None:
    socket = _Socket()
    socket._command_queue.put_nowait(SimpleNamespace(cmd="takeover", payload=""))
    session = _session(tmp_path, socket=socket)

    drain = AgentControlMixin._make_control_drain_fn(session)
    assert drain() == "stop"
    assert session.session_end_reason == "operator_takeover"


@pytest.mark.asyncio
async def test_pause_wait_emits_confirmation_events(tmp_path) -> None:
    socket = _Socket()
    session = _session(tmp_path, socket=socket)
    AgentControlMixin._apply_pause_session(session)
    socket._command_queue.put_nowait(SimpleNamespace(cmd="resume", payload=""))

    await AgentControlMixin._make_pause_wait_fn(session)()

    assert [event["type"] for event in socket.events] == ["Paused", "Resumed"]
    assert session.paused is False


def test_operator_hint_is_numbered_and_deduplicated(tmp_path) -> None:
    session = _session(tmp_path)

    AgentControlMixin._write_operator_hint(session, "please rerun the focused test")
    AgentControlMixin._write_operator_hint(session, "please rerun the focused test")
    AgentControlMixin._write_operator_hint(session, "also check the log")

    content = (tmp_path / ".operator_hints.md").read_text(encoding="utf-8")
    assert content.count("please rerun the focused test") == 1
    assert "Operator Hint #1" in content
    assert "Operator Hint #2" in content


def test_flush_transcript_command(tmp_path) -> None:
    socket = _Socket()
    socket._command_queue.put_nowait(SimpleNamespace(cmd="flush_transcript", payload=""))
    session = _session(tmp_path, socket=socket)
    session._transcript_storage = _Transcript()

    AgentControlMixin._drain_control_commands(session)
    assert session._transcript_storage.flush_count == 1


@pytest.mark.asyncio
async def test_broadcast_serializes_phase_event(monkeypatch, tmp_path) -> None:
    query_module = ModuleType("extensions.api.query")

    class _Event:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)

    for name in (
        "PhaseComplete",
        "SessionComplete",
        "TextDelta",
        "ToolCallEvent",
        "ToolResultEvent",
        "TurnComplete",
    ):
        setattr(query_module, name, type(name, (_Event,), {}))
    api_module = ModuleType("extensions.api")
    api_module.query = query_module
    monkeypatch.setitem(sys.modules, "extensions.api", api_module)
    monkeypatch.setitem(sys.modules, "extensions.api.query", query_module)

    socket = _Socket()
    session = _session(tmp_path, socket=socket)
    phase = query_module.PhaseComplete(phase=2, turn_count=7)
    await AgentControlMixin._broadcast_to_socket(session, phase)

    assert socket.events == [{"type": "PhaseComplete", "data": {"phase": 2, "turn_count": 7}}]


@pytest.mark.asyncio
async def test_broadcast_failure_is_visible_without_aborting(caplog, tmp_path) -> None:
    class _FailingSocket:
        @staticmethod
        async def send_event(_event: dict) -> None:
            raise RuntimeError("socket unavailable")

    session = _session(tmp_path)
    session.control_socket = _FailingSocket()

    with caplog.at_level(logging.WARNING):
        await AgentControlMixin._broadcast_to_socket(session, SimpleNamespace())

    assert "Control socket broadcast failed" in caplog.text
    assert "issue_id=I-137" in caplog.text
    assert "socket unavailable" in caplog.text


def test_tool_approval_decision_is_mirrored(monkeypatch) -> None:
    approval_module = ModuleType("extensions.orchestrator.approval_policy")

    class _PolicyEvent:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)
            self._approved = None
            self._deny_reason = None

    approval_module.ToolCallEvent = _PolicyEvent
    monkeypatch.setitem(sys.modules, "extensions.orchestrator.approval_policy", approval_module)

    class _Policy:
        @staticmethod
        def evaluate(event, _context) -> None:
            event._approved = False
            event._deny_reason = "blocked by policy"

    runner = AgentControlMixin()
    runner._approval_policy = _Policy()
    event = SimpleNamespace(tool_name="Bash", params={}, tool_use_id="tool-1")

    result = runner._handle_tool_call(event, {"cwd": "repo"})
    assert result is event
    assert event._approved is False
    assert event._deny_reason == "blocked by policy"
