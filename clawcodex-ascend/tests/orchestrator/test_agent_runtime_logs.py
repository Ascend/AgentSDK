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
"""Focused tests for the Agent Runtime debug and tool-event records."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from extensions.orchestrator.debug_log import append_debug_event
from extensions.orchestrator.tool_event_log import ToolEventLog


def test_tool_event_log_keeps_contract_order_and_optional_fields() -> None:
    event = ToolEventLog(
        tool="Agent",
        params={"prompt": "inspect"},
        approved=True,
        deny_reason=None,
        permission_mode="default",
        turn=3,
        session_run_id="run-1",
        ts=12.5,
        tool_use_id="tool-1",
        kind="agent_result",
        agent_id="child-1",
    )

    assert list(event.to_dict()) == [
        "ts",
        "tool",
        "params",
        "approved",
        "deny_reason",
        "permission_mode",
        "turn",
        "session_run_id",
        "tool_use_id",
        "kind",
        "agent_id",
    ]
    assert json.loads(event.to_json())["agent_id"] == "child-1"


def test_tool_event_log_omits_default_spawn_attribution() -> None:
    event = ToolEventLog("Read", {}, True, None, "plan", 1, "run-2", ts=1.0)

    assert set(event.to_dict()) == {
        "ts",
        "tool",
        "params",
        "approved",
        "deny_reason",
        "permission_mode",
        "turn",
        "session_run_id",
    }


def test_tool_event_log_serializes_non_json_params() -> None:
    event = ToolEventLog(
        "Read",
        {"path": Path("workspace/file.py")},
        True,
        None,
        "plan",
        1,
        "run-3",
        ts=1.0,
    )

    assert json.loads(event.to_json())["params"]["path"] == str(Path("workspace/file.py"))


def test_append_debug_event_writes_ndjson(tmp_path) -> None:
    path = tmp_path / "debug" / "events.ndjson"

    append_debug_event(path, "start", issue_id="F-1")
    append_debug_event(path, "finish", ok=True)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in rows] == ["start", "finish"]
    assert rows[0]["issue_id"] == "F-1"
    assert rows[1]["ok"] is True


def test_append_debug_event_is_best_effort(tmp_path) -> None:
    append_debug_event(None, "ignored")
    append_debug_event(tmp_path, "directory-is-not-a-file")


def test_append_debug_event_warns_when_write_fails(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="extensions.orchestrator.debug_log"):
        append_debug_event(tmp_path, "directory-is-not-a-file")

    assert "Failed to append orchestrator debug log" in caplog.text
