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

from __future__ import annotations

import json
from pathlib import Path

from extensions.visualizer.builders.timeline_builder import TimelineBuilder
from extensions.visualizer.models.viz_models import BarType, TimelineBar


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")


def _session_dir(tmp_path: Path, session_id: str) -> tuple[Path, Path]:
    sessions = tmp_path / "sessions"
    session_dir = sessions / session_id
    session_dir.mkdir(parents=True)
    return sessions, session_dir


def test_spawn_id_join_skips_malformed_timestamps(tmp_path):
    spawns = tmp_path / "agent_spawns.ndjson"
    _write_jsonl(spawns, [{"agent_id": "bad", "description": "review", "ts": "not-a-number"}])
    bar = TimelineBar(
        id="spawn",
        type=BarType.TOOL_CALL,
        label="Agent",
        start_time=10.0,
        end_time=10.0,
        duration_ms=0,
        detail={"is_agent_invocation": True, "description": "review"},
    )

    TimelineBuilder._attach_spawn_ids([bar], spawns)

    assert "agent_id" not in bar.detail


def test_session_json_wins_and_metadata_fills_missing_fields(tmp_path):
    sessions, session_dir = _session_dir(tmp_path, "sid-json")
    (session_dir / "metadata.json").write_text(
        json.dumps(
            {
                "title": "Metadata title",
                "provider": "metadata-provider",
                "cwd": "C:/workspace",
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "session_id": "sid-json",
                "model": "session-model",
                "created_at": "2026-06-22T08:00:00Z",
                "conversation": {
                    "messages": [
                        {
                            "uuid": "u1",
                            "timestamp": "2026-06-22T08:00:00Z",
                            "message": {"role": "user", "content": "字符串用户消息"},
                        },
                        {
                            "uuid": "a1",
                            "parentUuid": "u1",
                            "timestamp": "2026-06-22T08:00:02Z",
                            "message": {
                                "role": "assistant",
                                "model": "record-model",
                                "usage": {
                                    "duration_ms": 800,
                                    "input_tokens": 12,
                                    "output_tokens": 5,
                                },
                                "content": [
                                    {"type": "text", "text": "开始读取"},
                                    {
                                        "type": "tool_use",
                                        "id": "tu-1",
                                        "name": "Read",
                                        "input": {"path": "a.py"},
                                    },
                                ],
                            },
                        },
                        {
                            "timestamp": "2026-06-22T08:00:03Z",
                            "message": {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "tu-1",
                                        "content": "ok",
                                        "duration_ms": 125,
                                    },
                                ],
                            },
                        },
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        session_dir / "transcript.jsonl",
        [
            {
                "timestamp": "2026-06-22T09:00:00Z",
                "role": "assistant",
                "model": "must-not-win",
                "content": [{"type": "text", "text": "wrong source"}],
            }
        ],
    )

    viz = TimelineBuilder(sessions_dir=sessions, transcripts_dir=tmp_path / "transcripts").build("sid-json")
    assert viz is not None
    assert viz.title == "Metadata title"
    assert viz.model == "session-model"
    assert viz.provider == "metadata-provider"
    assert viz.workspace == "C:/workspace"
    assert any(bar.type == BarType.USER and bar.user_text == "字符串用户消息" for bar in viz.timeline)
    assert not any("wrong source" in str(bar.detail) for bar in viz.timeline)
    tool = next(bar for bar in viz.timeline if bar.type == BarType.TOOL_CALL)
    assert tool.duration_ms == 125
    assert tool.duration_unrecorded is False
    llm = next(bar for bar in viz.timeline if bar.type == BarType.LLM_CALL)
    assert llm.duration_ms == 800
    assert llm.input_text == "字符串用户消息"


def test_nested_subagent_events_join_the_shared_timeline(tmp_path):
    sessions, session_dir = _session_dir(tmp_path, "parent")
    _write_jsonl(
        session_dir / "transcript.jsonl",
        [
            {"timestamp": "2026-06-22T08:00:00Z", "role": "user", "content": "派生子 agent"},
            {
                "timestamp": "2026-06-22T08:00:01Z",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "spawn",
                        "name": "Agent",
                        "input": {
                            "agent_id": "child-1",
                            "subagent_type": "review",
                            "description": "检查实现",
                        },
                    },
                ],
            },
        ],
    )
    _write_jsonl(
        session_dir / "subagents" / "agent-child-1.jsonl",
        [
            {
                "timestamp": "2026-06-22T08:00:02Z",
                "parent_session_id": "parent",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read-child",
                        "name": "Read",
                        "input": {"path": "child.py"},
                    },
                ],
            },
            {
                "timestamp": "2026-06-22T08:00:03Z",
                "parent_session_id": "parent",
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "read-child", "content": "child result"},
                ],
            },
        ],
    )

    viz = TimelineBuilder(sessions_dir=sessions, transcripts_dir=tmp_path / "transcripts").build("parent")
    assert viz is not None
    assert [node.agent_id for node in viz.agent_tree] == ["child-1"]
    assert viz.agent_tree[0].name == "review"
    child_events = [bar for bar in viz.timeline if bar.agent_id == "child-1"]
    assert {bar.type for bar in child_events} >= {
        BarType.LLM_CALL,
        BarType.TOOL_CALL,
        BarType.TOOL_RESULT,
    }
    assert viz.agent_tree[0].spawn_x is not None
    assert viz.agent_tree[0].join_x is not None


def test_flat_subagent_transcript_is_discovered_and_lane_stamped(tmp_path):
    sessions, session_dir = _session_dir(tmp_path, "flat-parent")
    transcripts = tmp_path / "transcripts"
    _write_jsonl(
        session_dir / "transcript.jsonl",
        [
            {
                "timestamp": "2026-06-22T08:00:00Z",
                "role": "assistant",
                "content": [{"type": "text", "text": "main"}],
            },
        ],
    )
    _write_jsonl(
        transcripts / "flat-child.jsonl",
        [
            {
                "timestamp": "2026-06-22T08:00:01Z",
                "parent_session_id": "flat-parent",
                "role": "assistant",
                "content": [{"type": "text", "text": "child"}],
            },
        ],
    )

    viz = TimelineBuilder(sessions_dir=sessions, transcripts_dir=transcripts).build("flat-parent")
    assert viz is not None
    assert any(node.agent_id == "flat-child" for node in viz.agent_tree)
    assert any(bar.agent_id == "flat-child" for bar in viz.timeline)


def test_corrupt_jsonl_is_reported_without_breaking_valid_records(tmp_path):
    sessions, session_dir = _session_dir(tmp_path, "damaged")
    (session_dir / "transcript.jsonl").write_text(
        "not-json\n"
        + json.dumps(
            {
                "timestamp": "2026-06-22T08:00:00Z",
                "role": "user",
                "content": "仍可读取",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    viz = TimelineBuilder(sessions_dir=sessions).build("damaged")
    assert viz is not None
    assert viz.parse_warnings
    assert len(viz.parse_warnings) == len(set(viz.parse_warnings))
    assert any(bar.type == BarType.USER for bar in viz.timeline)


def test_transcript_jsonl_fallback_when_session_json_is_absent(tmp_path):
    sessions, session_dir = _session_dir(tmp_path, "fallback")
    (session_dir / "metadata.json").write_text(
        json.dumps(
            {
                "title": "Fallback title",
                "provider": "metadata-provider",
                "cwd": "C:/fallback",
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        session_dir / "transcript.jsonl",
        [
            {"timestamp": "2026-06-22T08:00:00Z", "role": "user", "content": "fallback user"},
            {
                "timestamp": "2026-06-22T08:00:01Z",
                "role": "assistant",
                "model": "fallback-model",
                "content": [
                    {"type": "text", "text": "fallback assistant"},
                    {
                        "type": "tool_use",
                        "id": "tu-fallback",
                        "name": "Read",
                        "input": {"path": "fallback.py"},
                    },
                ],
            },
            {
                "timestamp": "2026-06-22T08:00:02Z",
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu-fallback", "content": "ok"},
                ],
            },
        ],
    )

    viz = TimelineBuilder(sessions_dir=sessions, transcripts_dir=tmp_path / "transcripts").build("fallback")
    assert viz is not None
    assert viz.title == "Fallback title"
    assert viz.provider == "metadata-provider"
    assert viz.workspace == "C:/fallback"
    assert viz.model == "fallback-model"
    assert viz.transcript_path and viz.transcript_path.endswith("transcript.jsonl")
    assert any(bar.type == BarType.USER and bar.user_text == "fallback user" for bar in viz.timeline)
    assert any(bar.type == BarType.TOOL_CALL and bar.label == "Read" for bar in viz.timeline)
