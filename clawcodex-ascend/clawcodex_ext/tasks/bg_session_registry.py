#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

"""Cross-process background-session registry.

The registry scans ``~/.clawcodex/sessions/*/.background-runner.json`` to
rebuild its in-memory view and caches that view in
``~/.clawcodex/bg_sessions/index.json``.

``scan`` is the source of truth; ``index.json`` is only a recoverable cache.
The registry does not modify runner markers, protects its in-memory mapping
with an ``RLock``, and avoids parsing full transcripts during scans.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .bg_session import (
    BgSession,
    BgSessionConfig,
    BgSessionStatus,
    marker_path_for,
    replace_session,
    transcript_path_for,
)
from . import bg_session_health

logger = logging.getLogger(__name__)


class BgSessionRegistry:
    """Thread-safe cross-process background-session index.

    Disk I/O is kept outside the lock where possible. ``scan`` rebuilds the
    source-of-truth view, while ``save`` writes the cache.
    """

    def __init__(self, *, config: BgSessionConfig | None = None) -> None:
        self._config = config if config is not None else BgSessionConfig.from_env()
        self._lock = threading.RLock()
        self._sessions: dict[str, BgSession] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> BgSessionConfig:
        return self._config

    @property
    def index_path(self) -> Path:
        return self._config.index_path

    @property
    def sessions_dir(self) -> Path:
        return self._config.sessions_dir

    # ------------------------------------------------------------------
    # Source-of-truth scan
    # ------------------------------------------------------------------

    def scan(self) -> list[BgSession]:
        """Rebuild the view from ``sessions_dir/*/.background-runner.json``.

        Each marker is reconciled by ``bg_session_health.reconcile``. The
        reconciled list replaces the in-memory view and is returned.
        """
        sessions: list[BgSession] = []
        sessions_dir = self.sessions_dir
        if not sessions_dir.exists():
            with self._lock:
                self._sessions = {}
            return sessions

        try:
            entries = list(sessions_dir.iterdir())
        except OSError:
            logger.exception("Cannot list sessions dir %s", sessions_dir)
            return sessions

        for entry in entries:
            if not entry.is_dir():
                continue
            session_id = entry.name
            marker = marker_path_for(session_id, sessions_dir)
            if not marker.exists():
                continue
            sess = self._session_from_marker(session_id, marker, sessions_dir)
            if sess is not None:
                sessions.append(sess)

        # Enforce max_sessions while retaining the most recently started sessions.
        if len(sessions) > self._config.max_sessions:
            sessions.sort(key=lambda s: s.started_at or "", reverse=True)
            sessions = sessions[: self._config.max_sessions]

        with self._lock:
            self._sessions = {s.id: s for s in sessions}
        return sessions

    def _session_from_marker(
        self,
        session_id: str,
        marker_path: Path,
        sessions_dir: Path,
    ) -> BgSession | None:
        """Build and health-reconcile a ``BgSession`` from marker JSON."""
        try:
            data = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Corrupt marker %s; skipping", marker_path, exc_info=True)
            return None

        transcript = transcript_path_for(session_id, sessions_dir)
        sess = BgSession(
            id=session_id,
            session_id=session_id,
            workspace_root=Path(data.get("workspace_root", ".")),
            status=data.get("status", "unknown"),  # type: ignore[arg-type]
            pid=data.get("pid"),
            task_id=data.get("task_id"),
            team_id=data.get("team_id"),
            agent_name=data.get("agent_name"),
            description=data.get("description", ""),
            transcript_path=transcript if transcript.exists() else None,
            marker_path=marker_path,
            output_file=Path(data["output_file"]) if data.get("output_file") else None,
            started_at=data.get("started_at"),
            updated_at=data.get("updated_at"),
            completed_at=data.get("completed_at"),
            last_activity_at=data.get("last_activity_at"),
            error=data.get("error"),
        )
        return bg_session_health.reconcile(sess, stale_after_seconds=self._config.stale_after_seconds)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list(self, *, workspace_root: Path | None = None) -> list[BgSession]:
        """List the in-memory sessions, optionally filtered by workspace.

        This method does not scan. Callers that need current disk state must
        call ``scan`` explicitly.
        """
        with self._lock:
            sessions = list(self._sessions.values())
        if workspace_root is not None:
            target = workspace_root.resolve()
            sessions = [s for s in sessions if _safe_resolve(s.workspace_root) == target]
        return sessions

    def get(self, bg_session_id: str) -> BgSession | None:
        with self._lock:
            return self._sessions.get(bg_session_id)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def upsert(self, session: BgSession) -> None:
        """Insert or update one session without persisting automatically."""
        with self._lock:
            self._sessions[session.id] = session

    def remove(self, bg_session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(bg_session_id, None) is not None

    def update_status(
        self,
        bg_session_id: str,
        status: BgSessionStatus,
        *,
        error: str | None = None,
    ) -> BgSession | None:
        """Update a session status in place without scanning."""
        with self._lock:
            sess = self._sessions.get(bg_session_id)
            if sess is None:
                return None
            changes: dict[str, Any] = {"status": status}
            if error is not None:
                changes["error"] = error
            if status in ("completed", "failed", "stopped"):
                changes["completed_at"] = _now_iso()
            changes["updated_at"] = _now_iso()
            updated = replace_session(sess, **changes)
            self._sessions[bg_session_id] = updated
            return updated

    # ------------------------------------------------------------------
    # index.json cache persistence
    # ------------------------------------------------------------------

    def save(self) -> Path | None:
        """Persist the in-memory view to ``index_path``.

        Return the written path, or ``None`` when persistence is disabled or
        the write fails.
        """
        if not self._config.enabled:
            return None
        with self._lock:
            snapshot = [s.to_dict() for s in self._sessions.values()]
        target = self.index_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(
                    {"version": 1, "updated_at": _now_iso(), "sessions": snapshot},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, target)
        except OSError:
            logger.exception("Failed to persist bg_sessions index %s", target)
            return None
        return target

    def load(self) -> list[BgSession]:
        """Load the cache from ``index_path`` without scanning.

        A corrupt cache is logged and treated as empty; ``scan`` performs the
        recovery when requested.
        """
        target = self.index_path
        if not target.exists():
            return []
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("bg_sessions index %s corrupt; will be rebuilt by scan()", target)
            return []
        raw_sessions = data.get("sessions", []) if isinstance(data, dict) else []
        sessions = [BgSession.from_dict(d) for d in raw_sessions if isinstance(d, dict)]
        with self._lock:
            self._sessions = {s.id: s for s in sessions}
        return sessions

    def rebuild_and_save(self) -> list[BgSession]:
        """Rebuild the registry from markers and persist the recovered view."""
        sessions = self.scan()
        self.save()
        return sessions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


def _safe_resolve(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return p


__all__ = [
    "BgSessionRegistry",
]
