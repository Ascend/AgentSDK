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

"""Session metadata, status and artifact discovery behavior."""

from __future__ import annotations

import json
import logging

from extensions.visualizer.parsers.session_parser import SessionMetadataParser
from ._parser_test_support import (
    _assistant_text,
    _assistant_tool_use,
    _iso,
    _tool_result_entry,
    _user_text,
    _write_jsonl,
)


class TestSessionMetadataParser:
    def test_parse_nonexistent_session(self, tmp_path):
        parser = SessionMetadataParser(sessions_dir=tmp_path)
        result = parser.parse("nonexistent-session-id")
        assert result is None

    def test_parse_rejects_session_path_escape(self, tmp_path):
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "metadata.json").write_text('{"title":"secret"}', encoding="utf-8")

        parser = SessionMetadataParser(sessions_dir=sessions)

        assert parser.parse("../outside") is None
        (sessions / "linked").symlink_to(outside, target_is_directory=True)
        assert parser.parse("linked") is None

    def test_parse_session_with_metadata(self, tmp_path):
        session_dir = tmp_path / "test-session-001"
        session_dir.mkdir()
        meta = {
            "title": "Test Session",
            "start_time": 1717500000.0,
            "last_updated": 1717500030.0,
            "agent_name": "codex",
            "tags": ["test"],
            "cwd": "/tmp/proj",
        }
        (session_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        _write_jsonl(
            session_dir / "transcript.jsonl",
            [
                _user_text(1717500000.0, "hi"),
                _assistant_text(1717500030.0, "done", model="claude-opus-4-7"),
            ],
        )

        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("test-session-001")
        assert viz is not None
        assert viz.session_id == "test-session-001"
        assert viz.title == "Test Session"
        assert viz.model == "claude-opus-4-7"
        assert viz.duration_ms == 30000

    def test_parse_session_counts_from_transcript(self, tmp_path):
        session_dir = tmp_path / "ts-session"
        session_dir.mkdir()
        meta = {"title": "ts"}
        (session_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        _write_jsonl(
            session_dir / "transcript.jsonl",
            [
                _user_text(1717500000.0, "hi"),
                _assistant_tool_use(1717500001.0, "Read", "tu-1", {}),
                _tool_result_entry(1717500002.0, "tu-1", "ok"),
                _assistant_text(1717500003.0, "done"),
            ],
        )
        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("ts-session")
        assert viz is not None
        assert viz.turn_count == 4
        assert viz.tool_count == 1

    def test_parse_session_without_metadata(self, tmp_path):
        session_dir = tmp_path / "minimal-session"
        session_dir.mkdir()
        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("minimal-session")
        assert viz is not None
        assert viz.status == "unknown"

    def test_parse_session_discovers_report_artifacts(self, tmp_path):
        session_dir = tmp_path / "artifact-session"
        session_dir.mkdir()
        (session_dir / "report.md").write_text("# Report\n", encoding="utf-8")
        (session_dir / "events.ndjson").write_text('{"event":"ok"}\n', encoding="utf-8")
        (session_dir / "debug.ndjson").write_text('{"debug":"ok"}\n', encoding="utf-8")

        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("artifact-session")

        assert viz is not None
        assert viz.report_path and viz.report_path.endswith("report.md")
        assert viz.tool_events_path and viz.tool_events_path.endswith("events.ndjson")
        assert viz.debug_log_path and viz.debug_log_path.endswith("debug.ndjson")

    def test_state_journal_skips_only_malformed_line(self, tmp_path, caplog):
        sessions_dir = tmp_path / "sessions"
        session_dir = sessions_dir / "session-a"
        session_dir.mkdir(parents=True)
        reports_dir = tmp_path / "reports"
        run_dir = reports_dir / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "state_journal.ndjson").write_text(
            "not-json\n"
            + json.dumps(
                {
                    "type": "session_ref",
                    "session_id": "session-a",
                    "issue_id": "issue-1",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "verification",
                    "issue_id": "issue-1",
                    "verification_status": "passed",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with caplog.at_level(
            logging.WARNING,
            logger="extensions.visualizer.parsers.session_parser",
        ):
            viz = SessionMetadataParser(
                sessions_dir=sessions_dir,
                reports_dir=reports_dir,
            ).parse("session-a")

        assert viz is not None
        assert viz.issue_id == "issue-1"
        assert viz.verification_status == "passed"
        assert "Skipping malformed state journal line" in caplog.text

    def test_list_sessions(self, tmp_path):
        for sid in ["aaa", "bbb", "ccc"]:
            d = tmp_path / sid
            d.mkdir()
            (d / "metadata.json").write_text(json.dumps({"start_time": 1.0}), encoding="utf-8")
        parser = SessionMetadataParser(sessions_dir=tmp_path)
        sessions = parser.list_sessions()
        assert len(sessions) == 3

    def test_list_sessions_limit(self, tmp_path):
        for i in range(5):
            d = tmp_path / f"session-{i}"
            d.mkdir()
            (d / "metadata.json").write_text(json.dumps({"start_time": float(i)}), encoding="utf-8")
        parser = SessionMetadataParser(sessions_dir=tmp_path)
        sessions = parser.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_cost_block_entry_overrides_usage_sum(self, tmp_path):
        session_dir = tmp_path / "cb-session"
        session_dir.mkdir()
        (session_dir / "metadata.json").write_text(json.dumps({}), encoding="utf-8")
        _write_jsonl(
            session_dir / "transcript.jsonl",
            [
                _user_text(1717500000.0, "hi"),
                {
                    **{
                        "role": "assistant",
                        "type": "message",
                        "timestamp": _iso(1717500001.0),
                        "isMeta": False,
                        "isVirtual": False,
                        "isCompactSummary": False,
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                        "content": [{"type": "text", "text": "ok"}],
                    },
                },
                {
                    "type": "cost_block",
                    "cost": {
                        "total_cost_usd": 0.07,
                        "model_usage": {
                            "claude-opus-4-7": {
                                "input_tokens": 5000,
                                "output_tokens": 200,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                            },
                        },
                    },
                },
            ],
        )
        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("cb-session")
        assert viz is not None
        assert viz.stats.cost_usd == 0.07
        assert viz.stats.context_tokens == 5200

    def test_status_recent_transcript_is_running(self, tmp_path):
        import time as _time

        session_dir = tmp_path / "live-session"
        session_dir.mkdir()
        meta = {"start_time": _time.time() - 60, "last_updated": _time.time() - 60}
        (session_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        tp = session_dir / "transcript.jsonl"
        tp.write_text(
            json.dumps(_assistant_text(_time.time(), "x")) + "\n",
            encoding="utf-8",
        )

        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("live-session")
        assert viz is not None
        assert viz.status == "running"

    def test_status_recent_last_updated_is_running(self, tmp_path):
        import time as _time

        session_dir = tmp_path / "live-session"
        session_dir.mkdir()
        meta = {"start_time": _time.time() - 600, "last_updated": _time.time() - 30}
        (session_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        old = _time.time() - 600
        tp = session_dir / "transcript.jsonl"
        tp.write_text(
            json.dumps(_assistant_text(_time.time(), "x")) + "\n",
            encoding="utf-8",
        )
        import os

        os.utime(tp, (old, old))

        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("live-session")
        assert viz is not None
        assert viz.status == "running"

    def test_status_old_transcript_is_completed(self, tmp_path):
        import os
        import time as _time

        session_dir = tmp_path / "old-session"
        session_dir.mkdir()
        old = _time.time() - 3600
        meta = {"start_time": old - 60, "last_updated": old}
        (session_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        tp = session_dir / "transcript.jsonl"
        tp.write_text(
            json.dumps(_assistant_text(_time.time(), "x")) + "\n",
            encoding="utf-8",
        )
        os.utime(tp, (old, old))

        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("old-session")
        assert viz is not None
        assert viz.status == "completed"

    def test_status_explicit_wins_over_recency(self, tmp_path):
        import time as _time

        session_dir = tmp_path / "weird-session"
        session_dir.mkdir()
        meta = {
            "start_time": _time.time() - 60,
            "last_updated": _time.time() - 5,
            "status": "failed",
        }
        (session_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        tp = session_dir / "transcript.jsonl"
        tp.write_text(
            json.dumps(_assistant_text(_time.time(), "x")) + "\n",
            encoding="utf-8",
        )

        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("weird-session")
        assert viz is not None
        assert viz.status == "failed"

    def test_status_no_transcript_no_metadata_is_unknown(self, tmp_path):
        session_dir = tmp_path / "bare-session"
        session_dir.mkdir()
        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("bare-session")
        assert viz is not None
        assert viz.status == "unknown"

    def test_status_stale_short_session_is_completed(self, tmp_path):
        import os
        import time as _time

        session_dir = tmp_path / "stale-short"
        session_dir.mkdir()
        ancient = _time.time() - 47 * 3600
        meta = {"start_time": ancient, "last_updated": ancient + 0.027}
        (session_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        tp = session_dir / "transcript.jsonl"
        tp.write_text(
            json.dumps(_assistant_text(ancient, "x")) + "\n",
            encoding="utf-8",
        )
        os.utime(tp, (ancient, ancient))

        parser = SessionMetadataParser(sessions_dir=tmp_path)
        viz = parser.parse("stale-short")
        assert viz is not None
        assert viz.status == "completed", f"stale short session mis-classified as {viz.status!r} (expected 'completed')"

    def test_recent_end_time_does_not_mark_completed_session_running(self, tmp_path):
        import os
        import time as _time

        session_dir = tmp_path / "recently-completed"
        session_dir.mkdir()
        old = _time.time() - 3600
        meta = {"start_time": old, "end_time": _time.time() - 10}
        (session_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        transcript = session_dir / "transcript.jsonl"
        transcript.write_text(json.dumps(_assistant_text(old, "done")) + "\n", encoding="utf-8")
        os.utime(transcript, (old, old))

        viz = SessionMetadataParser(sessions_dir=tmp_path).parse("recently-completed")

        assert viz is not None
        assert viz.status == "completed"
