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

"""Session WebSocket bar updates, file notifications and encoded route IDs."""

from __future__ import annotations

import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from extensions.visualizer.ws import SessionLiveTail


def _ts(offset: float = 0.0) -> float:
    return 1_787_000_000.0 + offset


def _tool_entry(name: str, tool_use_id: str, *, timestamp=None, role="assistant", **extra):
    block = {"type": "tool_use", "name": name, "tool_use_id": tool_use_id, "input": {}}
    block.update(extra)
    return {"role": role, "timestamp": timestamp or _ts(), "content": [block]}


def _result_entry(tool_use_id: str, *, timestamp=None, is_error=False):
    return {
        "role": "user",
        "timestamp": timestamp or _ts(),
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": "error" if is_error else "ok",
                "is_error": is_error,
            }
        ],
    }


def _update(tail, entry, offset: float = 0.0):
    return tail._entry_to_bar_update(entry, emit_ts=_ts(offset))


class TestEntryToBarUpdate:
    def test_tool_use_emits_running_bar(self):
        tail = SessionLiveTail("s1")
        ev = _update(tail, _tool_entry("Bash", "toolu_abc", input={"command": "ls"}))
        assert ev is not None
        assert ev["type"] == "bar_update"
        assert ev["session_id"] == "s1"
        bar = ev["bar"]
        assert bar["id"] == "toolu_abc"
        assert bar["tool_name"] == "Bash"
        assert bar["label"] == "Bash"
        assert bar["start_time"] == _ts(0)
        assert bar["end_time"] == _ts(0)
        assert bar["status"] == "running"
        assert bar["category"] == "execute"
        assert bar["color"] == "#ee6666"

    def test_tool_result_updates_pending_bar_to_success(self):
        tail = SessionLiveTail("s1")
        _update(tail, _tool_entry("Read", "toolu_xyz"))
        ev = _update(tail, _result_entry("toolu_xyz", timestamp=_ts(2)), offset=2)
        assert ev is not None
        bar = ev["bar"]
        assert bar["id"] == "toolu_xyz"
        assert bar["end_time"] == _ts(2)
        assert bar["status"] == "success"

    def test_tool_result_error_sets_status_error(self):
        tail = SessionLiveTail("s1")
        _update(tail, _tool_entry("Bash", "toolu_err"))
        ev = _update(tail, _result_entry("toolu_err", timestamp=_ts(1), is_error=True), offset=1)
        assert ev["bar"]["status"] == "error"

    def test_orphan_tool_result_synthesizes_bar(self):
        tail = SessionLiveTail("s1")
        ev = _update(tail, _result_entry("toolu_orphan"))
        assert ev is not None
        assert ev["bar"]["id"] == "toolu_orphan"
        assert ev["bar"]["status"] == "success"
        assert ev["bar"]["label"] == "result"

    def test_text_block_returns_none(self):
        tail = SessionLiveTail("s1")
        ev = _update(
            tail,
            {"role": "assistant", "timestamp": _ts(), "content": [{"type": "text"}]},
        )
        assert ev is None

    def test_system_role_returns_none(self):
        tail = SessionLiveTail("s1")
        ev = _update(tail, _tool_entry("Bash", "t1", role="system"))
        assert ev is None

    def test_agent_tool_use_classified_as_orchestrate(self):
        tail = SessionLiveTail("s1")
        ev = _update(tail, _tool_entry("Agent", "toolu_orch", input={"subagent_type": "review"}))
        assert ev is not None
        assert ev["bar"]["category"] == "orchestrate"
        assert ev["bar"]["color"] == "#f778ba"

    def test_is_agent_invocation_flag_overrides_tool_name(self):
        tail = SessionLiveTail("s1")
        ev = _update(tail, _tool_entry("CustomDispatcher", "toolu_d", isAgentInvocation=True))
        assert ev["bar"]["category"] == "orchestrate"

    def test_unknown_tool_name_lands_in_other(self):
        tail = SessionLiveTail("s1")
        ev = _update(tail, _tool_entry("MyWeirdTool", "toolu_w"))
        assert ev["bar"]["category"] == "other"
        assert ev["bar"]["color"] == "#6e7681"

    def test_iso8601_timestamp_is_coerced_to_float(self):
        tail = SessionLiveTail("s1")
        ev = _update(tail, _tool_entry("Bash", "toolu_iso", timestamp="2024-06-04T12:00:00Z"))
        assert ev["bar"]["start_time"] > 0
        assert abs(ev["bar"]["start_time"] - 1717502400.0) < 1.0

    def test_pending_tools_cleared_after_match(self):
        tail = SessionLiveTail("s1")
        _update(tail, _tool_entry("Bash", "toolu_t"))
        assert "toolu_t" in tail._pending_tools
        _update(tail, _result_entry("toolu_t", timestamp=_ts(1)), offset=1)
        assert "toolu_t" not in tail._pending_tools

    def test_multiple_tool_blocks_emit_multiple_updates(self):
        tail = SessionLiveTail("s1")
        entry = _tool_entry("Read", "toolu_1")
        entry["content"].append(_tool_entry("Bash", "toolu_2")["content"][0])

        updates = tail._entry_to_bar_updates(entry, emit_ts=_ts())

        assert [update["bar"]["id"] for update in updates] == ["toolu_1", "toolu_2"]

    def test_explicit_zero_timestamp_is_preserved(self):
        tail = SessionLiveTail("s1")
        entry = _tool_entry("Read", "toolu_zero")
        entry["_timestamp"] = 0

        event = _update(tail, entry)

        assert event["bar"]["start_time"] == 0.0

    def test_pending_tools_are_bounded_and_cleared_on_stop(self):
        tail = SessionLiveTail("s1")
        tail._MAX_PENDING_TOOLS = 2
        for tool_id in ("toolu_1", "toolu_2", "toolu_3"):
            _update(tail, _tool_entry("Read", tool_id))

        assert list(tail._pending_tools) == ["toolu_2", "toolu_3"]
        tail.stop()
        assert tail._pending_tools == {}


@pytest.mark.asyncio
async def test_pathless_tail_task_can_restart_after_exit():
    tail = SessionLiveTail("late-file")

    first = tail.ensure_running()
    await first
    second = tail.ensure_running()
    await second

    assert first is not second
    assert tail._running is False


@pytest.mark.asyncio
async def test_session_json_change_emits_refetch_notification(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"conversation": {"messages": []}}), encoding="utf-8")
    tail = SessionLiveTail("single-file", path)
    events = []

    async def capture(event):
        events.append(event)

    tail.broadcast = capture
    task = asyncio.create_task(tail.tail_loop(interval=0.01))
    await asyncio.sleep(0.03)
    path.write_text(json.dumps({"conversation": {"messages": [{"role": "user"}]}}), encoding="utf-8")
    await asyncio.sleep(0.05)
    tail.stop()
    await task

    assert any(
        event.get("type") == "transcript_event"
        and event.get("source") == "session.json"
        and event.get("changed") is True
        for event in events
    )


def test_websocket_endpoint_accepts_urlencoded_session_id(tmp_path):
    from extensions.visualizer.server import create_app

    sessions = tmp_path / "sessions"
    session_dir = sessions / "session with #hash"
    session_dir.mkdir(parents=True)
    (session_dir / "transcript.jsonl").write_text(
        json.dumps({"role": "user", "content": "hello", "timestamp": _ts()}),
        encoding="utf-8",
    )
    app = create_app(sessions_dir=sessions, allow_import=False)
    client = TestClient(app)

    with client.websocket_connect("/api/viz/ws/sessions/session%20with%20%23hash") as ws:
        ws.send_text("ping")
        assert ws.receive_text() == "pong"
