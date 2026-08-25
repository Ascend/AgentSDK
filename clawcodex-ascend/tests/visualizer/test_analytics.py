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

"""Focused tests for Visualizer stats files and agent-tree projection."""

from __future__ import annotations

import json

from extensions.visualizer.builders.agent_tree_builder import AgentTreeBuilder
from extensions.visualizer.models.viz_models import (
    AgentTreeNode,
    BarType,
    SessionVizData,
    TimelineBar,
)
from extensions.visualizer.parsers.stats_parser import StatsFileParser


def _session() -> SessionVizData:
    return SessionVizData(
        session_id="viz-1",
        title="<script>alert(1)</script>",
        start_time=10.0,
        end_time=12.0,
        duration_ms=2_000,
        timeline=[
            TimelineBar(
                id="read-1",
                type=BarType.TOOL_CALL,
                label="Read <unsafe>",
                start_time=10.0,
                end_time=11.0,
                duration_ms=1_000,
            )
        ],
        agent_tree=[
            AgentTreeNode(agent_id="root", name="root", session_ref="viz-1"),
            AgentTreeNode(
                agent_id="child",
                name="reviewer",
                parent_id="root",
                session_ref="viz-1",
            ),
        ],
    )


def test_stats_file_summary_skips_bad_rows_and_filters(tmp_path) -> None:
    path = tmp_path / "tool_stats.jsonl"
    rows = [
        {
            "ts": 1,
            "kind": "tool",
            "tool": "Read",
            "ok": True,
            "dur_ms": 10,
            "agent_id": "a",
        },
        {
            "ts": 2,
            "kind": "tool",
            "tool": "Read",
            "ok": False,
            "dur_ms": 30,
            "agent_id": "a",
        },
        {
            "ts": 3,
            "kind": "skill",
            "skill": "review",
            "ok": True,
            "dur_ms": 20,
            "agent_id": "b",
        },
    ]
    path.write_text(
        "\n".join([json.dumps(rows[0]), "not-json", *[json.dumps(row) for row in rows[1:]]]),
        encoding="utf-8",
    )

    parser = StatsFileParser(path)
    summary = parser.get_summary(kind="tool", agent_id="a")

    assert summary["total_calls"] == 2
    assert summary["by_name"] == {"Read": 2}
    assert summary["avg_duration_ms"] == 20.0
    assert summary["error_rate"] == 0.5
    assert parser.get_recent(limit=1)[0]["ts"] == 3


def test_agent_tree_builder_projects_edges_and_root() -> None:
    tree = AgentTreeBuilder().build(_session())

    assert tree["root"] == "root"
    assert tree["edges"] == [{"source": "root", "target": "child"}]
    assert {node["id"] for node in tree["nodes"]} == {"root", "child"}
