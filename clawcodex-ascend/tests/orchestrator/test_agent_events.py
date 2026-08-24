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
"""Focused tests for Agent Runtime events, transcripts and rate limits."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from extensions.orchestrator.agent_events import AgentEventMixin


class _EventLog:
    def __init__(self, **values) -> None:
        self.values = values

    def to_json(self) -> str:
        return json.dumps(self.values)


class _Storage:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def write_message(self, message: dict) -> None:
        self.messages.append(message)


@pytest.fixture
def event_log_module(monkeypatch):
    module = ModuleType("extensions.orchestrator.tool_event_log")
    module.ToolEventLog = _EventLog
    monkeypatch.setitem(sys.modules, "extensions.orchestrator.tool_event_log", module)
    return module


def _runner() -> AgentEventMixin:
    runner = AgentEventMixin()
    runner.agent_config = SimpleNamespace(
        model="sonnet",
        rate_limit_base_delay_ms=1000,
        rate_limit_max_backoff_ms=5000,
        rate_limit_exponential_factor=2,
        rate_limit_max_retries=2,
    )
    return runner


def test_full_audit_log_writes_tool_decision(tmp_path, event_log_module) -> None:
    event = SimpleNamespace(
        tool_name="Read",
        params={"file_path": "README.md"},
        tool_use_id="tool-1",
        _approved=True,
        _deny_reason=None,
    )
    context = {
        "workspace_path": tmp_path,
        "run_id": "run-1",
        "permission_mode": "default",
        "turn": 2,
    }

    _runner()._append_tool_event_log(event, context)

    row = json.loads((tmp_path / ".reports/run-1.events.ndjson").read_text())
    assert row["tool"] == "Read"
    assert row["tool_use_id"] == "tool-1"
    assert row["approved"] is True


def test_none_and_minimal_audit_modes_filter_rows(tmp_path, event_log_module) -> None:
    approved = SimpleNamespace(
        tool_name="Read",
        params={},
        tool_use_id="tool-1",
        _approved=True,
        _deny_reason=None,
    )
    runner = _runner()
    runner._append_tool_event_log(approved, {"workspace_path": tmp_path, "run_id": "none", "audit_log": "none"})
    runner._append_tool_event_log(
        approved,
        {"workspace_path": tmp_path, "run_id": "minimal", "audit_log": "minimal"},
    )
    assert not (tmp_path / ".reports/none.events.ndjson").exists()
    assert not (tmp_path / ".reports/minimal.events.ndjson").exists()


def test_agent_spawn_result_records_child_id(tmp_path, event_log_module) -> None:
    event = SimpleNamespace(
        result={"output": {"agent_id": "child-1", "description": "review"}},
        tool_use_id="spawn-1",
    )
    _runner()._append_agent_spawn_result_log(event, {"workspace_path": tmp_path, "run_id": "run-2"})
    row = json.loads((tmp_path / ".reports/run-2.events.ndjson").read_text())
    assert row["kind"] == "agent_result"
    assert row["agent_id"] == "child-1"


def test_flush_transcript_preserves_tool_use_order(caplog, monkeypatch) -> None:
    messages = ModuleType("extensions.orchestrator_runtime.utils.messages_impl")
    messages.TextBlock = lambda **values: SimpleNamespace(**values)
    messages.ToolResultBlock = lambda **values: SimpleNamespace(**values)
    messages.create_assistant_message = lambda **values: {
        "role": "assistant",
        **values,
    }
    messages.create_user_message = lambda **values: {"role": "user", **values}
    monkeypatch.setitem(
        sys.modules,
        "extensions.orchestrator_runtime.utils.messages_impl",
        messages,
    )
    storage = _Storage()
    uses = [SimpleNamespace(id="a"), SimpleNamespace(id="b"), SimpleNamespace(id="missing")]
    result_a = SimpleNamespace(tool_use_id="a", content="A")
    result_b = SimpleNamespace(tool_use_id="b", content="B")
    session = SimpleNamespace(
        _transcript_storage=storage,
        _transcript_asst_text="working",
        _transcript_tool_uses=uses,
        _transcript_pending_results={"b": result_b, "a": result_a},
        _transcript_result_order=["b", "a"],
    )

    with caplog.at_level(logging.WARNING):
        _runner()._flush_turn_transcript(session)

    assert [block.tool_use_id for block in storage.messages[1]["content"]] == [
        "a",
        "b",
        "missing",
    ]
    assert storage.messages[1]["content"][-1].is_error is True
    assert "Transcript tool result missing tool_use_id=missing" in caplog.text
    assert session._transcript_tool_uses == []
    assert session._transcript_pending_results == {}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Error code: 429 rate_limit_error", True),
        ("RATE LIMIT reached", True),
        ("429 exceeded your current quota", False),
        ("Token Plan limit: 0 rate_limit_error", False),
        ("稍后重试 Error code: 429 quota", False),
        ("当前请求量较高，稍后重试；Error code: 429", True),
        ("ordinary provider error", False),
    ],
)
def test_rate_limit_detection(message: str, expected: bool) -> None:
    assert _runner()._is_429_response(message) is expected


def test_rotate_event_log_reports_cleanup_failure(caplog, monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "run.events.ndjson"
    rotated = tmp_path / "run.events.ndjson.1"
    log_path.write_text("current", encoding="utf-8")
    rotated.write_text("previous", encoding="utf-8")
    original_unlink = Path.unlink

    def _fail_rotated_unlink(path: Path, *args, **kwargs) -> None:
        if path == rotated:
            raise PermissionError("file is busy")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _fail_rotated_unlink)
    monkeypatch.setattr("extensions.orchestrator.agent_events._TOOL_EVENT_LOG_ROTATE_BYTES", 1)

    with caplog.at_level(logging.DEBUG):
        AgentEventMixin._rotate_event_log(log_path)

    assert "tool-event rotated log cleanup failed" in caplog.text
    assert str(rotated) in caplog.text
    assert "file is busy" in caplog.text


def test_backoff_is_exponential_and_capped(monkeypatch) -> None:
    monkeypatch.setattr("extensions.orchestrator.agent_events.random.uniform", lambda *_: 0)
    runner = _runner()
    session = SimpleNamespace(consecutive_429_count=1)
    assert runner._compute_rate_limit_backoff(session) == 1.0
    session.consecutive_429_count = 3
    assert runner._compute_rate_limit_backoff(session) == 4.0
    session.consecutive_429_count = 9
    assert runner._compute_rate_limit_backoff(session) == 5.0


@pytest.mark.asyncio
async def test_handle_rate_limit_sleeps_and_notifies(monkeypatch) -> None:
    query_module = ModuleType("extensions.api.query")
    query_module.TextDelta = lambda **values: SimpleNamespace(**values)
    api_module = ModuleType("extensions.api")
    api_module.query = query_module
    monkeypatch.setitem(sys.modules, "extensions.api", api_module)
    monkeypatch.setitem(sys.modules, "extensions.api.query", query_module)
    monkeypatch.setattr("extensions.orchestrator.agent_events.random.uniform", lambda *_: 0)
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    dashboard = SimpleNamespace(events=[])
    dashboard.on_event = lambda event, session: dashboard.events.append((event, session))
    runner = _runner()
    runner._sleep = _sleep
    session = SimpleNamespace(
        issue=SimpleNamespace(id="I-137"),
        consecutive_429_count=0,
        total_429_backoff_seconds=0.0,
        output_text="",
        rate_limit_pending_turn=None,
        status="running",
    )

    status = await runner._handle_rate_limit(session, "429", 4, dashboard)
    assert status == "running"
    assert sleeps == [1.0]
    assert session.rate_limit_pending_turn == 4
    assert "attempt 1/2" in session.output_text
    assert len(dashboard.events) == 1


@pytest.mark.asyncio
async def test_rate_limit_circuit_opens_without_sleeping() -> None:
    runner = _runner()
    runner._sleep = None
    session = SimpleNamespace(
        issue=SimpleNamespace(id="I-137"),
        consecutive_429_count=2,
        total_429_backoff_seconds=0.0,
        output_text="",
        rate_limit_pending_turn=None,
        status="running",
    )
    assert await runner._handle_rate_limit(session, "429", 4, None) == ("rate_limit_circuit_open")
