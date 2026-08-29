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

"""Transcript parsing behavior, filtering and tool-call pairing."""

from __future__ import annotations

import json
from pathlib import Path
from extensions.visualizer.models.viz_models import BarStatus, BarType
from extensions.visualizer.parsers.transcript_parser import TranscriptParser
from ._parser_test_support import (
    _assistant_entry,
    _assistant_text,
    _assistant_tool_use,
    _iso,
    _tool_result_entry,
    _write_jsonl,
)


def _parse_entries(tmp_path, entries, **kwargs):
    path = _write_jsonl(tmp_path / "t.jsonl", entries)
    return TranscriptParser().parse_file(path, **kwargs)


class TestTranscriptParser:
    def test_parse_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert TranscriptParser().parse_file(p) == []

    def test_parse_nonexistent_file(self):
        assert TranscriptParser().parse_file(Path("/nonexistent/file.jsonl")) == []

    def test_parse_assistant_text_message(self, tmp_path):
        bars = _parse_entries(tmp_path, [_assistant_text(1717500000.0, "Hello, I will help you.")])
        assert [(bar.type, bar.label) for bar in bars] == [(BarType.LLM_CALL, "LLM text")]

    def test_parse_tool_use_block(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [_assistant_tool_use(1717500000.0, "Read", "tu-1", {"path": "a.py"})],
        )
        assert len(bars) == 2
        tool = next(bar for bar in bars if bar.type == BarType.TOOL_CALL)
        assert tool.label == "Read"
        assert tool.status == BarStatus.RUNNING

    def test_parse_tool_use_block_with_leading_text(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [_assistant_tool_use(1717500000.0, "Read", "tu-1", {"path": "a.py"}, text="Let me look")],
        )
        assert len(bars) == 2
        assert bars[0].type == BarType.LLM_CALL
        assert bars[1].type == BarType.TOOL_CALL
        assert bars[1].label == "Read"

    def test_parse_tool_result_block(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_tool_use(1717500000.0, "Bash", "tu-1", {}),
                _tool_result_entry(1717500001.0, "tu-1", "ok"),
            ],
        )
        assert len(bars) == 3
        tool = next(bar for bar in bars if bar.type == BarType.TOOL_CALL)
        result = next(bar for bar in bars if bar.type == BarType.TOOL_RESULT)
        assert result.status == BarStatus.SUCCESS
        assert tool.duration_ms == 1000

    def test_parse_tool_result_error(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_tool_use(1717500000.0, "Bash", "tu-1", {}),
                _tool_result_entry(1717500002.0, "tu-1", "error", is_error=True),
            ],
        )
        assert next(bar for bar in bars if bar.type == BarType.TOOL_RESULT).status == BarStatus.ERROR

    def test_tool_call_duration_backfilled_from_result(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_tool_use(1717500000.0, "Read", "tu-a", {}),
                _tool_result_entry(1717500001.5, "tu-a", "ok"),
            ],
        )
        assert len(bars) == 3
        tool = next(bar for bar in bars if bar.type == BarType.TOOL_CALL)
        assert tool.duration_ms == 1500
        assert tool.end_time == 1717500001.5

    def test_tool_call_duration_stays_zero_without_result(self, tmp_path):
        bars = _parse_entries(tmp_path, [_assistant_tool_use(1717500000.0, "Bash", "tu-x", {})])
        assert len(bars) == 2
        tool = next(bar for bar in bars if bar.type == BarType.TOOL_CALL)
        assert tool.duration_ms == 0
        assert tool.duration_unrecorded is True

    def test_tool_call_duration_estimated_from_next_bar(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_tool_use(100.0, "Agent", "call_a", {}),
                _assistant_tool_use(130.0, "Agent", "call_b", {}),
            ],
        )
        tools = [bar for bar in bars if bar.type == BarType.TOOL_CALL]
        assert len(tools) == 2
        assert all(tool.duration_ms == 0 for tool in tools)
        assert all(tool.duration_unrecorded for tool in tools)

    def test_tool_call_fallback_with_only_text_after(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [_assistant_tool_use(200.0, "Read", "call_x", {}), _assistant_text(200.5, "Done.")],
        )
        tool = next(bar for bar in bars if bar.type == BarType.TOOL_CALL)
        assert tool.duration_ms == 0
        assert tool.duration_unrecorded is True

    def test_implausible_duration_is_preserved_as_diagnostic(self, tmp_path):
        result = _tool_result_entry(1717500001.0, "tu-long", "ok")
        result["content"][0]["duration_ms"] = 60_001
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_tool_use(1717500000.0, "Read", "tu-long", {}),
                result,
            ],
        )

        tool = next(bar for bar in bars if bar.type == BarType.TOOL_CALL)
        assert tool.duration_ms == 0
        assert tool.duration_unrecorded is True
        assert tool.detail["duration_source"] == "unrecorded"
        assert tool.detail["reported_duration_ms"] == 60_001

    def test_entry_with_multiple_blocks_emits_one_bar_per_block(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "t.jsonl",
            [
                {
                    "role": "assistant",
                    "type": "message",
                    "timestamp": _iso(100.0),
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                    "content": [
                        {"type": "text", "text": "Let me look at a few files."},
                        {"type": "tool_use", "name": "Read", "id": "c1", "input": {}},
                        {"type": "tool_use", "name": "Read", "id": "c2", "input": {}},
                        {"type": "tool_use", "name": "Read", "id": "c3", "input": {}},
                    ],
                },
            ],
        )
        bars = TranscriptParser().parse_file(p)
        assert len(bars) == 4
        assert [bar.type for bar in bars] == [BarType.LLM_CALL, *([BarType.TOOL_CALL] * 3)]
        ids = [b.id for b in bars[1:]]
        assert len(set(ids)) == 3, f"expected 3 distinct bar ids, got {ids}"

    def test_single_text_only_entry_unchanged(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "t.jsonl",
            [
                {
                    "role": "assistant",
                    "type": "message",
                    "timestamp": _iso(50.0),
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                    "content": [{"type": "text", "text": "Just a single text block."}],
                },
            ],
        )
        bars = TranscriptParser().parse_file(p)
        assert len(bars) == 1
        assert bars[0].type == BarType.LLM_CALL

    def test_entry_with_empty_text_block_skipped(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "t.jsonl",
            [
                {
                    "role": "assistant",
                    "type": "message",
                    "timestamp": _iso(60.0),
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                    "content": [
                        {"type": "text", "text": ""},  # dropped
                        {"type": "tool_use", "name": "Read", "id": "keep", "input": {}},
                    ],
                },
            ],
        )
        bars = TranscriptParser().parse_file(p)
        assert len(bars) == 2
        tool = next(bar for bar in bars if bar.type == BarType.TOOL_CALL)
        assert tool.label == "Read"

    def test_orphan_tool_result_does_not_modify_other_calls(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "t.jsonl",
            [
                _assistant_tool_use(1717500000.0, "Read", "tu-a", {}),
                _tool_result_entry(1717500001.0, "tu-orphan", "ok"),
            ],
        )
        bars = TranscriptParser().parse_file(p)
        assert len(bars) == 3
        tool = next(bar for bar in bars if bar.type == BarType.TOOL_CALL)
        result = next(bar for bar in bars if bar.type == BarType.TOOL_RESULT)
        assert tool.detail.get("tool_use_id") == "tu-a"
        assert result.detail.get("tool_use_id") == "tu-orphan"
        assert bars[1].detail.get("parent_id") is None

    def test_skip_meta_entry(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_entry(1717500000.0, "this should be dropped", isMeta=True),
                _assistant_text(1717500001.0, "real text"),
            ],
        )
        assert len(bars) == 1
        assert bars[0].label == "LLM text"

    def test_skip_cost_block_entry(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "t.jsonl",
            [
                {"type": "cost_block", "cost": {"total_cost_usd": 0.42}},
                _assistant_text(1717500000.0, "ok"),
            ],
        )
        bars = TranscriptParser().parse_file(p)
        assert len(bars) == 1
        assert bars[0].label == "LLM text"

    def test_skip_progress_entry(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "t.jsonl",
            [
                {"type": "progress", "data": {"kind": "thinking"}, "toolUseID": "x"},
                _assistant_text(1717500000.0, "ok"),
            ],
        )
        bars = TranscriptParser().parse_file(p)
        assert len(bars) == 1

    def test_skip_compact_summary(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_entry(1717500000.0, "compaction summary", isCompactSummary=True),
                _assistant_text(1717500001.0, "post-compact"),
            ],
        )
        assert len(bars) == 1
        assert bars[0].label == "LLM text"

    def test_skip_api_error_message(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_entry(
                    1717500000.0,
                    "rate limit error",
                    isApiErrorMessage=True,
                    apiError="rate limit",
                ),
                _assistant_text(1717500001.0, "ok"),
            ],
        )
        assert len(bars) == 1

    def test_system_status_noise_is_dropped(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_entry(
                    1717500000.0,
                    "__background_complete__",
                    role="system",
                    subtype="background_complete",
                )
            ],
        )
        assert bars == []

    def test_thinking_block_emits_llm_text_bar(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_entry(
                    1717500000.0,
                    "",
                    content=[{"type": "thinking", "thinking": "hmm, let me think"}],
                )
            ],
        )
        assert [bar.type for bar in bars] == [BarType.LLM_CALL]

    def test_explicit_subagent_id_propagates_to_bars(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [
                _assistant_entry(
                    1717500000.0,
                    "",
                    parent_session_id="main-session-1",
                    content=[
                        {"type": "tool_use", "name": "Read", "id": "c1", "input": {}},
                    ],
                )
            ],
            agent_id="child-agent",
        )
        assert len(bars) == 2
        assert all(bar.agent_id == "child-agent" for bar in bars)

    def test_malformed_json_line_skipped(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("not-json\n")
        bars = TranscriptParser().parse_file(p)
        assert bars == []

    def test_parse_incremental(self, tmp_path):
        p = tmp_path / "inc.jsonl"
        p.write_text(json.dumps(_assistant_text(1.0, "hi")) + "\n")
        bars, offset = TranscriptParser().parse_incremental(p, 0)
        assert len(bars) == 1
        assert offset > 0

    def test_parse_incremental_reports_non_object_records(self, tmp_path):
        path = tmp_path / "inc.jsonl"
        path.write_text("[]\n", encoding="utf-8")
        parser = TranscriptParser()

        bars, _ = parser.parse_incremental(path)

        assert bars == []
        assert parser.warnings == ["inc.jsonl:1: record is not an object"]

    def test_lowercase_agent_tool_is_recognized(self, tmp_path):
        bars = _parse_entries(tmp_path, [_assistant_tool_use(1.0, "agent", "toolu-agent", {})])
        tool = next(bar for bar in bars if bar.type == BarType.TOOL_CALL)
        assert tool.detail["is_agent_invocation"] is True

    def test_parse_incremental_nonexistent(self):
        assert TranscriptParser().parse_incremental(Path("/nope"), 0) == ([], 0)

    def test_parse_resets_state_between_files(self, tmp_path):
        parser = TranscriptParser()
        p1 = _write_jsonl(tmp_path / "a.jsonl", [_assistant_text(1.0, "msg1")])
        p2 = _write_jsonl(tmp_path / "b.jsonl", [_assistant_text(2.0, "msg2")])
        bars1 = parser.parse_file(p1)
        bars2 = parser.parse_file(p2)
        assert bars1[0].id == "main-llm-0"
        assert bars2[0].id == "main-llm-0"

    def test_iso8601_timestamp_in_entry(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [_assistant_entry(0.0, "hi", timestamp="2024-06-04T12:00:00+00:00")],
        )
        assert len(bars) == 1
        assert bars[0].start_time > 0

    def test_numeric_timestamp_is_accepted(self, tmp_path):
        bars = _parse_entries(tmp_path, [_assistant_entry(0.0, "hi", timestamp=12345)])
        assert len(bars) == 1
        assert bars[0].ts_unrecorded is False

    def test_string_content_is_normalized(self, tmp_path):
        bars = _parse_entries(
            tmp_path,
            [_assistant_entry(1717500000.0, "", content="legacy bare string")],
        )
        assert len(bars) == 1
        assert bars[0].type == BarType.LLM_CALL
        assert bars[0].detail["text"] == "legacy bare string"
