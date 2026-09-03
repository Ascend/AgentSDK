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

"""Tests for tool and skill invocation statistics.

Covers:
- clawcodex_ext/tool_stats.py: recording module
- extensions/visualizer/parsers/stats_parser.py: Visualizer parser
"""

from __future__ import annotations


# pylint: disable=E0611,W0621
import json
import os
import tempfile

import pytest

from clawcodex_ext.tool_stats import (
    configure,
    flush,
    get_stats,
    get_summary,
    record_skill,
    record_tool,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def stats_path():
    """Temporary stats file, cleaned up after test."""
    with tempfile.NamedTemporaryFile(encoding="utf-8", suffix=".jsonl", delete=False, mode="w") as f:
        tmp = f.name
    configure(tmp)
    yield tmp
    try:
        os.unlink(tmp)
    except OSError:
        pass  # Fixture cleanup tolerates a file removed by the test.


# ── Core recording tests ──────────────────────────────────────────────


def test_configure_keeps_buffered_rows_with_their_original_sink(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    configure(first)
    record_tool("First", dur_ms=1.0, ok=True)

    configure(second)
    record_tool("Second", dur_ms=2.0, ok=True)
    flush()

    assert json.loads(first.read_text(encoding="utf-8"))["tool"] == "First"
    assert json.loads(second.read_text(encoding="utf-8"))["tool"] == "Second"


class TestRecordTool:
    def test_record_tool_basic(self, stats_path: str):
        record_tool("Read", dur_ms=12.3, ok=True)
        flush()
        rows = get_stats()
        assert len(rows) == 1
        assert rows[0]["tool"] == "Read"
        assert rows[0]["dur_ms"] == 12.3
        assert rows[0]["ok"] is True
        assert rows[0]["kind"] == "tool"

    def test_record_tool_failure(self, stats_path: str):
        record_tool("Bash", dur_ms=2300.0, ok=False, error="timeout")
        flush()
        rows = get_stats()
        assert rows[0]["ok"] is False
        assert rows[0]["error"] == "timeout"

    def test_record_tool_custom_agent(self, stats_path: str):
        record_tool("Edit", dur_ms=450.0, ok=True, agent_id="orchestrator-001")
        flush()
        rows = get_stats(agent_id="orchestrator-001")
        assert len(rows) == 1
        assert rows[0]["agent_id"] == "orchestrator-001"


class TestRecordSkill:
    def test_record_skill_basic(self, stats_path: str):
        record_skill("code_review", dur_ms=3200.0, ok=True)
        flush()
        rows = get_stats(kind="skill")
        assert len(rows) == 1
        assert rows[0]["skill"] == "code_review"
        assert rows[0]["kind"] == "skill"

    def test_record_skill_with_params(self, stats_path: str):
        record_skill(
            "deploy",
            dur_ms=15000.0,
            ok=False,
            error="conn refused",
            params={"target": "prod"},
            skill_version="1.2",
        )
        flush()
        rows = get_stats(kind="skill")
        assert rows[0]["skill"] == "deploy"
        assert rows[0]["params"] == {"target": "prod"}
        assert rows[0]["skill_version"] == "1.2"


class TestGetStats:
    def test_get_stats_filter_kind(self, stats_path: str):
        record_tool("Read", dur_ms=10.0, ok=True)
        record_skill("review", dur_ms=100.0, ok=True)
        flush()
        assert len(get_stats(kind="tool")) == 1
        assert len(get_stats(kind="skill")) == 1
        assert len(get_stats()) == 2

    def test_get_stats_limit(self, stats_path: str):
        for i in range(10):
            record_tool(f"Tool{i}", dur_ms=float(i), ok=True)
        flush()
        rows = get_stats(limit=3)
        assert len(rows) == 3

    def test_get_stats_order(self, stats_path: str):
        record_tool("First", dur_ms=1.0, ok=True)
        record_tool("Second", dur_ms=2.0, ok=True)
        flush()
        rows = get_stats()
        assert rows[0]["tool"] == "Second"  # most recent first

    def test_get_stats_empty(self):
        # No records yet — returns empty list
        assert get_stats() == []


class TestGetSummary:
    def test_summary_basic(self, stats_path: str):
        record_tool("Read", dur_ms=10.0, ok=True)
        record_tool("Read", dur_ms=20.0, ok=True)
        record_tool("Bash", dur_ms=100.0, ok=False, error="timeout")
        record_skill("review", dur_ms=500.0, ok=True)
        flush()
        summary = get_summary()
        assert summary["total_calls"] == 4
        assert summary["by_name"]["Read"] == 2
        assert summary["by_name"]["Bash"] == 1
        assert summary["error_rate"] == 0.25
        assert summary["avg_duration_ms"] == pytest.approx((10 + 20 + 100 + 500) / 4, 0.1)

    def test_summary_empty(self):
        summary = get_summary()
        assert summary["total_calls"] == 0
        assert summary["by_name"] == {}

    def test_summary_filter_kind(self, stats_path: str):
        record_tool("Read", dur_ms=10.0, ok=True)
        record_skill("review", dur_ms=100.0, ok=True)
        flush()
        tool_summary = get_summary(kind="tool")
        assert tool_summary["total_calls"] == 1
        assert "Read" in tool_summary["by_name"]
        skill_summary = get_summary(kind="skill")
        assert skill_summary["total_calls"] == 1


class TestBufferedWrite:
    def test_batch_flush(self, stats_path: str):
        """Writing many records doesn't need explicit flush for each."""
        for i in range(25):  # > _BUFFER_FLUSH_SIZE=20
            record_tool(f"T{i}", dur_ms=1.0, ok=True)
        # At least the buffer should have flushed once automatically
        rows = get_stats()
        assert len(rows) == 25


# ── Visualizer parser integration tests ───────────────────────────────


class TestVisualizerStatsParser:
    @pytest.fixture
    def parser(self, stats_path: str):
        """Provide a parser pointing to the same temp stats file."""
        from extensions.visualizer.parsers.stats_parser import StatsFileParser

        return StatsFileParser(path=stats_path)

    def test_parser_summary(self, parser, stats_path: str):
        record_tool("Read", dur_ms=10.0, ok=True)
        record_tool("Bash", dur_ms=100.0, ok=False, error="timeout")
        record_skill("review", dur_ms=500.0, ok=True)
        flush()
        summary = parser.get_summary()
        assert summary["total_calls"] == 3
        assert summary["by_kind"]["tool"] == 2
        assert summary["by_kind"]["skill"] == 1
        assert summary["error_rate"] == pytest.approx(1 / 3, 0.01)

    def test_parser_recent(self, parser, stats_path: str):
        for i in range(5):
            record_tool(f"T{i}", dur_ms=float(i), ok=True)
        flush()
        recents = parser.get_recent(limit=2)
        assert len(recents) == 2
        assert recents[0]["tool"] == "T4"

    def test_parser_empty(self, parser):
        summary = parser.get_summary()
        assert summary["total_calls"] == 0
        assert parser.get_recent() == []

    def test_parser_filter_kind(self, parser, stats_path: str):
        record_tool("Read", dur_ms=10.0, ok=True)
        record_skill("review", dur_ms=50.0, ok=True)
        flush()
        s1 = parser.get_summary(kind="tool")
        assert s1["total_calls"] == 1
        assert s1["by_name"]["Read"] == 1
        s2 = parser.get_summary(kind="skill")
        assert s2["total_calls"] == 1
