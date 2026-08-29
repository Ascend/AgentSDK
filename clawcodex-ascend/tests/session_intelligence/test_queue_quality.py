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

import threading

from clawcodex_ext.session_intelligence.queue import (
    enqueue_summary_job,
    process_pending_summary_jobs,
    queue_path,
    read_pending_jobs,
)


def _write_session(sessions_dir, session_id: str) -> None:
    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text('{"title":"T"}', encoding="utf-8")
    (session_dir / "transcript.jsonl").write_text(
        f'{{"role":"user","content":"task {session_id}"}}\n',
        encoding="utf-8",
    )


def test_process_limit_preserves_jobs_not_selected_for_this_drain(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    for session_id in ("old-1", "old-2", "new-1"):
        _write_session(sessions_dir, session_id)
        enqueue_summary_job(session_id, base_dir=tmp_path)

    result = process_pending_summary_jobs(
        base_dir=tmp_path,
        sessions_dir=sessions_dir,
        limit=1,
    )

    assert result == {"processed": 1, "failed": 0, "remaining": 2}
    assert [job["session_id"] for job in read_pending_jobs(base_dir=tmp_path)] == [
        "old-2",
        "new-1",
    ]


def test_enqueue_during_drain_is_not_overwritten(tmp_path, monkeypatch) -> None:
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "draining")
    enqueue_summary_job("draining", base_dir=tmp_path)

    summarize_started = threading.Event()
    allow_summary_to_finish = threading.Event()

    def summarize_session(session_id: str, *, sessions_dir):
        assert session_id == "draining"
        summarize_started.set()
        assert allow_summary_to_finish.wait(timeout=2)
        return {"generated": True}

    monkeypatch.setattr(
        "clawcodex_ext.session_intelligence.summarizer.summarize_session",
        summarize_session,
    )

    drain_errors: list[BaseException] = []

    def drain() -> None:
        try:
            process_pending_summary_jobs(
                base_dir=tmp_path,
                sessions_dir=sessions_dir,
                limit=1,
            )
        except Exception as exc:  # pragma: no cover - assertion aid
            drain_errors.append(exc)

    drain_thread = threading.Thread(target=drain)
    drain_thread.start()
    assert summarize_started.wait(timeout=2)

    enqueue_started = threading.Event()
    enqueue_done = threading.Event()

    def enqueue_concurrently() -> None:
        enqueue_started.set()
        enqueue_summary_job("concurrent", base_dir=tmp_path)
        enqueue_done.set()

    enqueue_thread = threading.Thread(target=enqueue_concurrently)
    enqueue_thread.start()
    assert enqueue_started.wait(timeout=2)
    enqueue_done.wait(timeout=0.5)
    allow_summary_to_finish.set()

    drain_thread.join(timeout=2)
    enqueue_thread.join(timeout=2)
    assert not drain_thread.is_alive()
    assert not enqueue_thread.is_alive()
    assert not drain_errors
    assert [job["session_id"] for job in read_pending_jobs(base_dir=tmp_path)] == ["concurrent"]


def test_queue_path_respects_clawcodex_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))

    assert queue_path() == tmp_path / "session_summaries" / "queue.jsonl"


def test_empty_queue_drain_does_not_create_lock_file(tmp_path) -> None:
    result = process_pending_summary_jobs(base_dir=tmp_path)

    assert result == {"processed": 0, "failed": 0, "remaining": 0}
    assert not (tmp_path / "session_summaries" / "queue.jsonl.lock").exists()


def test_zero_limit_reports_existing_jobs_without_draining(tmp_path) -> None:
    enqueue_summary_job("first", base_dir=tmp_path)
    enqueue_summary_job("second", base_dir=tmp_path)

    result = process_pending_summary_jobs(base_dir=tmp_path, limit=0)

    assert result == {"processed": 0, "failed": 0, "remaining": 2}
    assert [job["session_id"] for job in read_pending_jobs(base_dir=tmp_path)] == [
        "first",
        "second",
    ]
