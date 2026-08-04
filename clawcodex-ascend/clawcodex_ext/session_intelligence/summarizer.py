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

# pylint: disable=cyclic-import

"""Lazy session summary sidecar generation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from clawcodex_ext.session_intelligence.queue import write_status
from clawcodex_ext.session_intelligence.summary_schema import SessionSummary


def summarize_session(session_id: str, *, sessions_dir: Path | None = None) -> dict[str, Any]:
    from clawcodex_ext.services.session_storage import SESSIONS_DIR

    base = sessions_dir or SESSIONS_DIR
    session_dir = Path(base) / session_id
    transcript = session_dir / "transcript.jsonl"
    metadata = _load_json(session_dir / "metadata.json")
    if not session_dir.is_dir():
        return {"generated": False, "reason": f"session not found: {session_id}"}

    entries = _read_tail(transcript, limit=40)
    user_texts = [str(e.get("content") or "") for e in entries if e.get("role") == "user"]
    title = str(metadata.get("title") or (user_texts[-1][:80] if user_texts else "Session summary"))
    next_action = user_texts[-1][:200] if user_texts else ""
    summary = SessionSummary(
        session_id=session_id,
        cwd=str(metadata.get("cwd") or ""),
        transcript_mtime=transcript.stat().st_mtime if transcript.exists() else 0.0,
        title=title,
        goals=user_texts[-3:],
        open_threads=[next_action] if next_action else [],
        next_action_candidates=[next_action] if next_action else [],
    )
    path = session_dir / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        write_status(
            session_dir,
            {
                "state": "complete",
                "transcript_mtime": summary.transcript_mtime,
                "attempts": 1,
                "last_error": "",
                "updated_at": time.time(),
            },
        )
    except Exception as exc:
        write_status(
            session_dir,
            {
                "state": "failed",
                "transcript_mtime": summary.transcript_mtime,
                "attempts": 1,
                "last_error": str(exc),
                "updated_at": time.time(),
            },
        )
        return {"generated": False, "reason": str(exc)}
    return {"generated": True, "summary_path": str(path), "summary": summary.to_dict()}


def update_summary_from_away_summary(
    *,
    session_id: str,
    recap: str,
    sessions_dir: Path | None = None,
) -> dict[str, Any]:
    """Bridge Away Summary text into the structured summary sidecar."""

    from clawcodex_ext.services.session_storage import SESSIONS_DIR

    base = sessions_dir or SESSIONS_DIR
    session_dir = Path(base) / session_id
    if not session_dir.is_dir():
        return {"generated": False, "reason": f"session not found: {session_id}"}
    transcript = session_dir / "transcript.jsonl"
    existing = _load_json(session_dir / "summary.json")
    metadata = _load_json(session_dir / "metadata.json")
    summary = SessionSummary(
        session_id=session_id,
        cwd=str(existing.get("cwd") or metadata.get("cwd") or ""),
        transcript_mtime=transcript.stat().st_mtime if transcript.exists() else 0.0,
        title=str(existing.get("title") or metadata.get("title") or "Session summary"),
        goals=list(existing.get("goals") or []),
        completed=list(existing.get("completed") or []),
        open_threads=list(existing.get("open_threads") or []),
        files_touched=list(existing.get("files_touched") or []),
        commands_seen=list(existing.get("commands_seen") or []),
        user_preferences=list(existing.get("user_preferences") or []),
        next_action_candidates=list(existing.get("next_action_candidates") or []),
    )
    if recap and recap not in summary.completed:
        summary.completed.append(recap[:500])
    if recap and not summary.next_action_candidates:
        summary.next_action_candidates.append("Continue from the latest Away Summary recap.")

    path = session_dir / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        write_status(
            session_dir,
            {
                "state": "complete",
                "transcript_mtime": summary.transcript_mtime,
                "attempts": 1,
                "last_error": "",
                "updated_at": time.time(),
            },
        )
    except Exception as exc:
        return {"generated": False, "reason": str(exc)}
    return {"generated": True, "summary_path": str(path)}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    except OSError:
        return []
    return rows
