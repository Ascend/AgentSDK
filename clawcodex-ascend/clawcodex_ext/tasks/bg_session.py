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

"""Background-session data models and state transitions.

Defines ``BgSession``, ``BgSessionStatus``, ``BgSessionEvent``,
``BgSessionConfig``, and the subsystem's failure exceptions.

State transitions::

 starting
 ├─ marker written + pid alive ─▶ running
 ├─ launch failed ──────────────▶ failed
 └─ marker stale before pid ────▶ unknown

 running
 ├─ user attach/resume ─────────▶ paused
 ├─ completion marker ──────────▶ completed
 ├─ child exit non-zero ────────▶ failed
 ├─ pid gone, no completion ────▶ orphaned
 └─ user stop ──────────────────▶ stopped

 orphaned
 ├─ transcript completion ──────▶ completed
 ├─ cleanup removes marker ─────▶ stopped
 └─ user attach w/ transcript ──▶ paused

``health.py`` first honors an explicit completed or failed marker, then
checks process liveness and transcript completion. A live process with a
stale transcript produces a warning, while uncertain states become
``unknown`` or ``orphaned`` and are never deleted silently.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

#: Default global index path; ``~`` is expanded by ``BgSessionConfig``.
#: Uses the same ``~/.clawcodex`` root as ``background_runner._sessions_dir``.
DEFAULT_INDEX_PATH: Path = Path("~/.clawcodex/bg_sessions/index.json")

#: Default sessions directory shared with ``background_runner._sessions_dir``.
DEFAULT_SESSIONS_DIR: Path = Path("~/.clawcodex/sessions")

#: Marker filename shared with ``background_runner._runner_marker_path``.
RUNNER_MARKER_NAME: str = ".background-runner.json"

#: Completion sentinel written to the transcript by ``_run_agent_headless``.
TRANSCRIPT_COMPLETION_SENTINEL: str = "__background_complete__"


# ---------------------------------------------------------------------------
# Status literals
# ---------------------------------------------------------------------------

BgSessionStatus: TypeAlias = Literal[
    "starting",
    "running",
    "paused",
    "completed",
    "failed",
    "stopped",
    "orphaned",
    "unknown",
]

#: Terminal states that cannot transition again.
TERMINAL_BG_STATUSES: frozenset[BgSessionStatus] = frozenset({"completed", "failed", "stopped"})

#: Active states whose process should still be alive.
ACTIVE_BG_STATUSES: frozenset[BgSessionStatus] = frozenset({"starting", "running", "paused"})

BgSessionEventType: TypeAlias = Literal[
    "created",
    "backgrounded",
    "attached",
    "resumed",
    "stopped",
    "completed",
    "failed",
    "orphaned",
    "cleaned",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BgSession:
    """Snapshot of one background session.

    ``id`` and ``session_id`` match for Ctrl+B sessions. Agent-tool or team
    sessions may instead use a stable background id and distinguish the
    shared session-id namespace with ``source=bg_session``.
    """

    id: str
    session_id: str
    workspace_root: Path
    status: BgSessionStatus
    pid: int | None = None
    task_id: str | None = None
    team_id: str | None = None
    agent_name: str | None = None
    description: str = ""
    transcript_path: Path | None = None
    marker_path: Path | None = None
    output_file: Path | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    last_activity_at: str | None = None
    error: str | None = None

    def is_terminal(self) -> bool:
        """Return whether this session is in a terminal state."""
        return self.status in TERMINAL_BG_STATUSES

    def is_active(self) -> bool:
        """Return whether this session is active and should have a live PID."""
        return self.status in ACTIVE_BG_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary, converting paths to strings."""
        d = asdict(self)
        for key, val in list(d.items()):
            if isinstance(val, Path):
                d[key] = str(val)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BgSession":
        """Deserialize a dictionary while tolerating paths and missing fields."""
        path_keys = (
            "workspace_root",
            "transcript_path",
            "marker_path",
            "output_file",
        )
        kwargs: dict[str, Any] = {}
        for key in (
            "id",
            "session_id",
            "workspace_root",
            "status",
            "pid",
            "task_id",
            "team_id",
            "agent_name",
            "description",
            "transcript_path",
            "marker_path",
            "output_file",
            "started_at",
            "updated_at",
            "completed_at",
            "last_activity_at",
            "error",
        ):
            if key not in data:
                continue
            val = data[key]
            if key in path_keys and val is not None and not isinstance(val, Path):
                val = Path(str(val))
            kwargs[key] = val
        # Supply fallbacks for required fields.
        kwargs.setdefault("id", kwargs.get("session_id", ""))
        kwargs.setdefault("session_id", kwargs.get("id", ""))
        kwargs.setdefault("workspace_root", Path("."))
        kwargs.setdefault("status", "unknown")
        return cls(**kwargs)


@dataclass(frozen=True)
class BgSessionEvent:
    """One background-session event-log entry."""

    id: str
    bg_session_id: str
    event_type: BgSessionEventType
    actor: str
    message: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BgSessionEvent":
        return cls(
            id=str(data.get("id", "")),
            bg_session_id=str(data.get("bg_session_id", "")),
            event_type=data.get("event_type", "created"),  # type: ignore[arg-type]
            actor=str(data.get("actor", "")),
            message=str(data.get("message", "")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class BgSessionConfig:
    """Runtime configuration for background sessions.

    ``CLAWCODEX_BG_SESSIONS`` values ``off``, ``0``, and ``false`` disable
    the feature. Disabled mode keeps per-session markers but does not write
    the global index.
    """

    enabled: bool = False
    index_path: Path = DEFAULT_INDEX_PATH
    sessions_dir: Path = DEFAULT_SESSIONS_DIR
    stale_after_seconds: int = 600
    max_sessions: int = 200
    cleanup_completed_after_seconds: int = 86_400
    allow_agent_attach: bool = True
    allow_cross_workspace: bool = False

    @classmethod
    def from_env(cls) -> "BgSessionConfig":
        """Construct from the environment; ``CLAWCODEX_BG_SESSIONS=off`` disables it."""
        raw = os.environ.get("CLAWCODEX_BG_SESSIONS", "").strip().lower()
        enabled = raw not in ("", "off", "0", "false", "no", "disabled")
        return cls(
            enabled=enabled,
            index_path=DEFAULT_INDEX_PATH.expanduser(),
            sessions_dir=DEFAULT_SESSIONS_DIR.expanduser(),
        )


def is_bg_sessions_enabled(config: BgSessionConfig | None = None) -> bool:
    """Return whether background-session support is enabled."""
    cfg = config if config is not None else BgSessionConfig.from_env()
    return cfg.enabled


def replace_session(session: BgSession, **changes: Any) -> BgSession:
    """Return a new snapshot with selected fields replaced."""
    return replace(session, **changes)


def marker_path_for(session_id: str, sessions_dir: Path) -> Path:
    """Return the ``.background-runner.json`` path for ``session_id``."""
    return sessions_dir / session_id / RUNNER_MARKER_NAME


def transcript_path_for(session_id: str, sessions_dir: Path) -> Path:
    """Return the JSONL transcript path for ``session_id``."""
    return sessions_dir / session_id / f"{session_id}.jsonl"


# ---------------------------------------------------------------------------
# Failure exceptions
# ---------------------------------------------------------------------------


class BgSessionsDisabledError(RuntimeError):
    """Background sessions are disabled; callers should fall back to TaskList."""


class BgSessionNotFoundError(LookupError):
    """The requested background-session id does not exist."""


class BgSessionAlreadyRunningError(RuntimeError):
    """The same session was backgrounded more than once."""


class BgSessionAttachError(RuntimeError):
    """The transcript is missing or malformed."""


class BgSessionPermissionError(PermissionError):
    """The request crosses an unauthorized workspace or team boundary."""


class BgSessionOrphanedError(RuntimeError):
    """The process is gone without a completion marker."""


class BgSessionStopError(RuntimeError):
    """The background process could not be stopped."""


__all__ = [
    "ACTIVE_BG_STATUSES",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_SESSIONS_DIR",
    "RUNNER_MARKER_NAME",
    "TERMINAL_BG_STATUSES",
    "TRANSCRIPT_COMPLETION_SENTINEL",
    "BgSession",
    "BgSessionAlreadyRunningError",
    "BgSessionAttachError",
    "BgSessionConfig",
    "BgSessionEvent",
    "BgSessionEventType",
    "BgSessionNotFoundError",
    "BgSessionOrphanedError",
    "BgSessionPermissionError",
    "BgSessionStatus",
    "BgSessionStopError",
    "BgSessionsDisabledError",
    "is_bg_sessions_enabled",
    "marker_path_for",
    "replace_session",
    "transcript_path_for",
]
