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

"""P94-D Multi-signal background-session health assessment."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bg_session import (
    BgSession,
    BgSessionStatus,
    TRANSCRIPT_COMPLETION_SENTINEL,
    replace_session,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthAssessment:
    """Snapshot of a background-session health assessment."""

    status: BgSessionStatus
    is_stale: bool = False
    pid_alive: bool | None = None
    marker_status: str | None = None
    transcript_has_completion: bool | None = None
    transcript_mtime_age_s: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "is_stale": self.is_stale,
            "pid_alive": self.pid_alive,
            "marker_status": self.marker_status,
            "transcript_has_completion": self.transcript_has_completion,
            "transcript_mtime_age_s": self.transcript_mtime_age_s,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Signal primitives
# ---------------------------------------------------------------------------


def _pid_alive(pid: int | None) -> bool:
    """Return whether a process ID is alive."""
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        return False


def _read_marker(marker_path: Path | None) -> dict[str, Any] | None:
    """Read marker."""
    if marker_path is None or not marker_path.exists():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _transcript_has_completion(transcript_path: Path | None) -> bool | None:
    """Check whether a transcript contains its completion sentinel."""
    if transcript_path is None or not transcript_path.exists():
        return None
    try:
        # Read from the tail because the completion sentinel is near the end.
        with transcript_path.open("rb") as f:
            try:
                f.seek(-8192, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", errors="replace")
        return TRANSCRIPT_COMPLETION_SENTINEL in tail
    except Exception:
        return None


def _transcript_mtime_age_s(transcript_path: Path | None) -> float | None:
    """Return the age of a transcript's modification time."""
    if transcript_path is None or not transcript_path.exists():
        return None
    try:
        return time.time() - transcript_path.stat().st_mtime
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Health assessment
# ---------------------------------------------------------------------------


def assess(
    session: BgSession,
    *,
    stale_after_seconds: int = 600,
) -> HealthAssessment:
    """Assess a background session from its marker, PID, and transcript signals."""
    marker = _read_marker(session.marker_path)
    marker_status = marker.get("status") if marker else None
    pid = session.pid if session.pid is not None else (int(marker["pid"]) if marker and "pid" in marker else None)
    pid_ok = _pid_alive(pid)
    transcript_done = _transcript_has_completion(session.transcript_path)
    mtime_age = _transcript_mtime_age_s(session.transcript_path)

    # Priority 1: an explicit terminal marker.
    if marker_status in ("completed", "failed"):
        return HealthAssessment(
            status=marker_status,
            pid_alive=pid_ok,
            marker_status=marker_status,
            transcript_has_completion=transcript_done,
            transcript_mtime_age_s=mtime_age,
            reason="marker terminal status",
        )

    # Priority 2: verify the PID when the marker reports running.
    if session.status == "running" or marker_status == "running":
        if pid_ok:
            # Priority 4: a live PID with an unchanged transcript is stale.
            is_stale = mtime_age is not None and mtime_age > stale_after_seconds
            return HealthAssessment(
                status="running",
                is_stale=is_stale,
                pid_alive=True,
                marker_status=marker_status,
                transcript_has_completion=transcript_done,
                transcript_mtime_age_s=mtime_age,
                reason="stale" if is_stale else "pid alive",
            )
        # The PID is no longer live.
        # Priority 3: check the transcript completion sentinel.
        if transcript_done is True:
            return HealthAssessment(
                status="completed",
                pid_alive=False,
                marker_status=marker_status,
                transcript_has_completion=True,
                transcript_mtime_age_s=mtime_age,
                reason="pid gone, transcript completed",
            )
        if transcript_done is False:
            return HealthAssessment(
                status="orphaned",
                pid_alive=False,
                marker_status=marker_status,
                transcript_has_completion=False,
                transcript_mtime_age_s=mtime_age,
                reason="pid gone, no completion marker",
            )
        # Treat an unreadable transcript conservatively as orphaned.
        return HealthAssessment(
            status="orphaned",
            pid_alive=False,
            marker_status=marker_status,
            transcript_has_completion=None,
            transcript_mtime_age_s=mtime_age,
            reason="pid gone, transcript unreadable",
        )

    # Starting means the marker exists before a live PID is observable.
    if session.status == "starting":
        if pid_ok:
            return HealthAssessment(
                status="running",
                pid_alive=True,
                marker_status=marker_status,
                transcript_has_completion=transcript_done,
                transcript_mtime_age_s=mtime_age,
                reason="starting → running (pid alive)",
            )
        if marker is None:
            return HealthAssessment(
                status="unknown",
                pid_alive=False,
                marker_status=None,
                transcript_has_completion=transcript_done,
                transcript_mtime_age_s=mtime_age,
                reason="starting, no marker yet",
            )
        return HealthAssessment(
            status="failed",
            pid_alive=False,
            marker_status=marker_status,
            transcript_has_completion=transcript_done,
            transcript_mtime_age_s=mtime_age,
            reason="starting, marker present but pid dead",
        )

    # Preserve paused, orphaned, unknown, and terminal states; refresh signals.
    return HealthAssessment(
        status=session.status,
        pid_alive=pid_ok,
        marker_status=marker_status,
        transcript_has_completion=transcript_done,
        transcript_mtime_age_s=mtime_age,
        reason="preserve existing status",
    )


def reconcile(
    session: BgSession,
    *,
    stale_after_seconds: int = 600,
) -> BgSession:
    """Return a background session updated with its health assessment."""
    h = assess(session, stale_after_seconds=stale_after_seconds)
    return replace_session(
        session,
        status=h.status,
        error=h.reason if h.status in ("orphaned", "failed", "unknown") else session.error,
    )


__all__ = [
    "HealthAssessment",
    "assess",
    "reconcile",
]
