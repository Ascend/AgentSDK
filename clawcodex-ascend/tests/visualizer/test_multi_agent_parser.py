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


"""Nested and flat sub-agent transcript discovery behavior."""

from __future__ import annotations

from extensions.visualizer.parsers.multi_agent_parser import MultiAgentParser
from ._parser_test_support import _assistant_tool_use, _iso, _write_jsonl


class TestMultiAgentParser:
    def test_parse_for_session_no_subagents(self, tmp_path):
        nodes = MultiAgentParser().parse_for_session(
            "main",
            sessions_dir=tmp_path,
            transcripts_dir=tmp_path / "transcripts",
        )
        assert nodes == []

    def test_parse_for_session_nested_subagent(self, tmp_path):
        session_dir = tmp_path / "main"
        sub_dir = session_dir / "subagents"
        sub_dir.mkdir(parents=True)
        _write_jsonl(
            sub_dir / "agent-abc123.jsonl",
            [
                {
                    "role": "assistant",
                    "type": "message",
                    "timestamp": _iso(1717500000.0),
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                    "parent_session_id": "main",
                    "content": [{"type": "text", "text": "I will search the repo"}],
                },
                _assistant_tool_use(1717500001.0, "Read", "tu-1", {}),
            ],
        )
        nodes = MultiAgentParser().parse_for_session(
            "main",
            sessions_dir=tmp_path,
            transcripts_dir=tmp_path / "transcripts",
        )
        assert len(nodes) == 1
        node = nodes[0]
        assert node.agent_id == "abc123"
        assert node.parent_id == "main"
        assert node.metadata["source"] == "nested"
        assert node.metadata["tool_count"] == 1

    def test_parse_for_session_flat_subagent_with_parent_marker(self, tmp_path):
        tx_dir = tmp_path / "transcripts"
        tx_dir.mkdir()
        _write_jsonl(
            tx_dir / "agent-xyz.jsonl",
            [
                {
                    "role": "assistant",
                    "type": "message",
                    "timestamp": _iso(1717500000.0),
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                    "parent_session_id": "main",
                    "content": [{"type": "text", "text": "hello from subagent"}],
                },
            ],
        )
        nodes = MultiAgentParser().parse_for_session(
            "main",
            sessions_dir=tmp_path,
            transcripts_dir=tx_dir,
        )
        assert len(nodes) == 1
        assert nodes[0].agent_id == "xyz"
        assert nodes[0].metadata["source"] == "flat"

    def test_parse_for_session_flat_subagent_with_wrong_parent_skipped(self, tmp_path):
        tx_dir = tmp_path / "transcripts"
        tx_dir.mkdir()
        _write_jsonl(
            tx_dir / "agent-other.jsonl",
            [
                {
                    "role": "assistant",
                    "type": "message",
                    "timestamp": _iso(1717500000.0),
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                    "parent_session_id": "different-session",
                    "content": [{"type": "text", "text": "x"}],
                },
            ],
        )
        nodes = MultiAgentParser().parse_for_session(
            "main",
            sessions_dir=tmp_path,
            transcripts_dir=tx_dir,
        )
        assert nodes == []

    def test_flat_discovery_skips_null_parent_before_valid_marker(self, tmp_path):
        tx_dir = tmp_path / "transcripts"
        tx_dir.mkdir()
        _write_jsonl(
            tx_dir / "agent-late-parent.jsonl",
            [
                {
                    "role": "assistant",
                    "timestamp": _iso(1717500000.0),
                    "parent_session_id": None,
                    "content": [],
                },
                {
                    "role": "assistant",
                    "timestamp": _iso(1717500001.0),
                    "parent_session_id": "main",
                    "content": [{"type": "text", "text": "late marker"}],
                },
            ],
        )

        nodes = MultiAgentParser().parse_for_session("main", sessions_dir=tmp_path, transcripts_dir=tx_dir)

        assert [node.agent_id for node in nodes] == ["late-parent"]
