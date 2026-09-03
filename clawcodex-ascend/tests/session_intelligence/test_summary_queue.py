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

from clawcodex_ext.session_intelligence.index import load_summary
from clawcodex_ext.session_intelligence.queue import (
    enqueue_summary_job,
    process_pending_summary_jobs,
)
from clawcodex_ext.session_intelligence.summarizer import (
    summarize_session,
    update_summary_from_away_summary,
)


def test_enqueue_summary_job(tmp_path) -> None:
    path = enqueue_summary_job("s1", cwd=tmp_path, base_dir=tmp_path)
    rows = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["session_id"] == "s1"


def test_summary_pipeline_honors_runtime_sessions_override(tmp_path, monkeypatch) -> None:
    from clawcodex_ext.services import session_storage

    state_dir = tmp_path / "state"
    sessions_dir = state_dir / "sessions"
    session_dir = sessions_dir / "debug-session"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text('{"title":"Debug"}', encoding="utf-8")
    (session_dir / "transcript.jsonl").write_text(
        '{"role":"user","content":"isolated task"}\n',
        encoding="utf-8",
    )
    stale_sessions_dir = tmp_path / "stale-sessions"
    monkeypatch.setattr(session_storage, "SESSIONS_DIR", stale_sessions_dir)
    monkeypatch.setenv("CLAWCODEX_HOME", str(state_dir))
    monkeypatch.setenv("CLAWCODEX_SESSIONS_DIR", str(sessions_dir))

    enqueue_summary_job("debug-session", cwd=tmp_path)

    assert (session_dir / "summary.status.json").exists()
    assert not (stale_sessions_dir / "debug-session").exists()
    assert process_pending_summary_jobs() == {
        "processed": 1,
        "failed": 0,
        "remaining": 0,
    }
    summary = load_summary("debug-session")
    assert summary is not None
    assert summary["title"] == "Debug"


def test_summarize_session_atomic_write(tmp_path) -> None:
    session_dir = tmp_path / "s1"
    session_dir.mkdir()
    (session_dir / "metadata.json").write_text('{"title":"T"}', encoding="utf-8")
    (session_dir / "transcript.jsonl").write_text(
        '{"role":"user","content":"next task"}\n',
        encoding="utf-8",
    )
    result = summarize_session("s1", sessions_dir=tmp_path)
    assert result["generated"] is True
    summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["next_action_candidates"] == ["next task"]


def test_process_pending_summary_jobs(tmp_path) -> None:
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text('{"title":"T"}', encoding="utf-8")
    (session_dir / "transcript.jsonl").write_text(
        '{"role":"user","content":"queued task"}\n',
        encoding="utf-8",
    )
    enqueue_summary_job("s1", base_dir=tmp_path)
    result = process_pending_summary_jobs(base_dir=tmp_path, sessions_dir=tmp_path / "sessions")
    assert result["processed"] == 1
    assert (session_dir / "summary.json").exists()


def test_update_summary_from_away_summary(tmp_path) -> None:
    session_dir = tmp_path / "s1"
    session_dir.mkdir()
    (session_dir / "metadata.json").write_text('{"title":"T"}', encoding="utf-8")
    result = update_summary_from_away_summary(
        session_id="s1",
        recap="recap text",
        sessions_dir=tmp_path,
    )
    assert result["generated"] is True
    summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] == ["recap text"]
