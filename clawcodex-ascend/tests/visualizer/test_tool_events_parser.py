#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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


"""Tool-event audit parser behavior."""

from __future__ import annotations

import json
from pathlib import Path

from extensions.visualizer.models.viz_models import BarStatus, BarType
from extensions.visualizer.parsers.tool_events_parser import ToolEventsParser


class TestToolEventsParser:
    def test_parse_empty(self, tmp_path):
        p = tmp_path / "events.ndjson"
        p.write_text("")
        bars = ToolEventsParser().parse_file(p)
        assert bars == []

    def test_parse_nonexistent(self):
        bars = ToolEventsParser().parse_file(Path("/nope"))
        assert bars == []

    def test_parse_approved_tool(self, tmp_path):
        p = tmp_path / "events.ndjson"
        p.write_text(
            json.dumps(
                {
                    "ts": 1717500000.0,
                    "tool": "Bash",
                    "approved": True,
                    "turn": 1,
                }
            )
            + "\n"
        )
        bars = ToolEventsParser().parse_file(p)
        assert len(bars) == 1
        assert bars[0].type == BarType.TOOL_CALL
        assert bars[0].label == "Bash"
        assert bars[0].status == BarStatus.SUCCESS

    def test_parse_denied_tool(self, tmp_path):
        p = tmp_path / "events.ndjson"
        p.write_text(
            json.dumps(
                {
                    "ts": 1717500000.0,
                    "tool": "Write",
                    "approved": False,
                    "deny_reason": "permission denied",
                }
            )
            + "\n"
        )
        bars = ToolEventsParser().parse_file(p)
        assert len(bars) == 1
        assert bars[0].status == BarStatus.ERROR

    def test_parse_pending_tool(self, tmp_path):
        p = tmp_path / "events.ndjson"
        p.write_text(
            json.dumps(
                {
                    "ts": 1717500000.0,
                    "tool": "Read",
                    "approved": None,
                }
            )
            + "\n"
        )
        bars = ToolEventsParser().parse_file(p)
        assert len(bars) == 1
        assert bars[0].status == BarStatus.WARNING

    def test_malformed_line_skipped(self, tmp_path):
        p = tmp_path / "events.ndjson"
        p.write_text("bad-json\n")
        bars = ToolEventsParser().parse_file(p)
        assert bars == []

    def test_multiple_events(self, tmp_path):
        p = tmp_path / "events.ndjson"
        lines = [
            json.dumps({"ts": 1.0, "tool": "Read", "approved": True}),
            json.dumps({"ts": 2.0, "tool": "Bash", "approved": False, "deny_reason": "no"}),
            json.dumps({"ts": 3.0, "tool": "Write", "approved": True}),
        ]
        p.write_text("\n".join(lines) + "\n")
        bars = ToolEventsParser().parse_file(p)
        assert len(bars) == 3
        assert bars[0].label == "Read"
        assert bars[1].status == BarStatus.ERROR
        assert bars[2].label == "Write"

    def test_agent_event_tolerates_non_object_params(self, tmp_path):
        p = tmp_path / "events.ndjson"
        p.write_text(
            json.dumps(
                {
                    "ts": 1717500000.0,
                    "tool": "Agent",
                    "approved": True,
                    "params": "task description",
                }
            )
            + "\n"
        )

        bars = ToolEventsParser().parse_file(p)

        assert len(bars) == 1
        assert bars[0].detail["is_agent_invocation"] is True
        assert "description" not in bars[0].detail

    def test_agent_result_is_joined_to_its_spawn(self, tmp_path):
        path = tmp_path / "events.ndjson"
        records = [
            {
                "ts": 1.0,
                "kind": "call",
                "tool": "Agent",
                "tool_use_id": "toolu-agent",
                "approved": True,
            },
            {
                "ts": 2.0,
                "kind": "agent_result",
                "tool": "Agent",
                "tool_use_id": "toolu-agent",
                "agent_id": "child-1",
            },
        ]
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

        bars = ToolEventsParser().parse_file(path)

        assert len(bars) == 1
        assert bars[0].detail["agent_id"] == "child-1"
