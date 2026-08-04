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

"""Pending queue for session summary generation."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def queue_path(base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        root = base_dir
    else:
        env_home = os.environ.get("CLAWCODEX_HOME")
        root = Path(env_home) if env_home else Path.home() / ".clawcodex"
    return root / "session_summaries" / "queue.jsonl"


def _queue_lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


@contextlib.contextmanager
def _exclusive_queue_lock(lock_path: Path) -> Iterator[None]:
    """Lock the queue across threads and processes without external helpers."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        os.close(fd)


def _read_queue_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return []

    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed summary queue row in %s", path)
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _atomic_write_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    """Publish a complete queue snapshot without exposing partial writes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def enqueue_summary_job(
    session_id: str,
    *,
    cwd: str | Path | None = None,
    transcript_mtime: float = 0.0,
    base_dir: Path | None = None,
) -> Path:
    path = queue_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "session_id": session_id,
        "cwd": str(cwd or ""),
        "transcript_mtime": transcript_mtime,
        "state": "pending",
        "attempts": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with _exclusive_queue_lock(_queue_lock_path(path)):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    if session_id != "latest":
        try:
            from clawcodex_ext.services.session_storage import SESSIONS_DIR

            session_dir = (
                Path(base_dir) / "sessions" / session_id if base_dir is not None else Path(SESSIONS_DIR) / session_id
            )
            write_status(
                session_dir,
                {
                    "state": "pending",
                    "transcript_mtime": transcript_mtime,
                    "attempts": 0,
                    "last_error": "",
                    "updated_at": time.time(),
                },
            )
        except Exception:
            logger.warning("Queued summary job but failed to write status", exc_info=True)
    return path


def write_status(session_dir: Path, status: dict[str, Any]) -> Path:
    path = session_dir / "summary.status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(status)
    payload.setdefault("updated_at", time.time())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_pending_jobs(*, base_dir: Path | None = None, limit: int = 100) -> list[dict[str, Any]]:
    path = queue_path(base_dir)
    if not path.exists() or limit <= 0:
        return []
    with _exclusive_queue_lock(_queue_lock_path(path)):
        rows = _read_queue_rows(path)
    return [row for row in rows if row.get("state", "pending") == "pending"][:limit]


def process_pending_summary_jobs(
    *,
    base_dir: Path | None = None,
    sessions_dir: Path | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Best-effort queue drain for missing/stale ``summary.json`` files."""

    path = queue_path(base_dir)
    if not path.exists():
        return {"processed": 0, "failed": 0, "remaining": 0}
    if limit <= 0:
        with _exclusive_queue_lock(_queue_lock_path(path)):
            rows = _read_queue_rows(path)
        remaining = sum(row.get("state", "pending") == "pending" for row in rows)
        return {"processed": 0, "failed": 0, "remaining": remaining}

    from clawcodex_ext.session_intelligence.summarizer import summarize_session

    processed = 0
    failed = 0
    with _exclusive_queue_lock(_queue_lock_path(path)):
        rows = _read_queue_rows(path)
        selected_indexes = [index for index, row in enumerate(rows) if row.get("state", "pending") == "pending"][
            : max(0, limit)
        ]
        removed_indexes: set[int] = set()

        for index in selected_indexes:
            job = rows[index]
            session_id = str(job.get("session_id") or "")
            if not session_id or session_id == "latest":
                removed_indexes.add(index)
                continue
            result = summarize_session(session_id, sessions_dir=sessions_dir)
            if result.get("generated"):
                processed += 1
                removed_indexes.add(index)
                continue

            failed += 1
            retry = dict(job)
            retry["attempts"] = int(retry.get("attempts") or 0) + 1
            retry["last_error"] = str(result.get("reason") or "")
            retry["updated_at"] = time.time()
            if retry["attempts"] < 3:
                rows[index] = retry
            else:
                removed_indexes.add(index)

        remaining_rows = [row for index, row in enumerate(rows) if index not in removed_indexes]
        if selected_indexes or path.exists():
            _atomic_write_queue(path, remaining_rows)

    remaining = sum(row.get("state", "pending") == "pending" for row in remaining_rows)
    return {"processed": processed, "failed": failed, "remaining": remaining}


def start_summary_queue_worker(
    *,
    base_dir: Path | None = None,
    sessions_dir: Path | None = None,
    limit: int = 20,
) -> None:
    """Spawn a short-lived daemon worker to drain pending summary jobs."""

    import threading

    def _run() -> None:
        try:
            process_pending_summary_jobs(base_dir=base_dir, sessions_dir=sessions_dir, limit=limit)
        except Exception:
            logger.exception("Summary queue worker failed")

    thread = threading.Thread(target=_run, name="summary-queue-worker", daemon=True)
    thread.start()
