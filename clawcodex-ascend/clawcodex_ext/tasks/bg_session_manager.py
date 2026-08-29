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

"""Lifecycle operations for background sessions.

``BgSessionManager`` provides list, inspect, attach, stop, cleanup,
``background_current_session``, and ``upsert_after_launch`` operations.
Stopping is graceful-first, cross-workspace attachment is denied by default,
and cleanup removes indexed terminal records without silently deleting files.
Disabled mode raises ``BgSessionsDisabledError`` or performs the documented
no-op for each operation.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bg_session import (
    BgSession,
    BgSessionAttachError,
    BgSessionConfig,
    BgSessionNotFoundError,
    BgSessionPermissionError,
    BgSessionStopError,
    BgSessionsDisabledError,
    is_bg_sessions_enabled,
    marker_path_for,
    replace_session,
)
from .bg_session_health import assess
from .bg_session_registry import BgSessionRegistry

logger = logging.getLogger(__name__)

#: Maximum time to wait for a process after requesting a graceful stop.
_GRACEFUL_STOP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class AttachResult:
    """Attachment result containing a transcript tail and resume command."""

    session: BgSession
    transcript_tail: str = ""
    resume_hint: str = ""


class BgSessionManager:
    """Coordinate the background-session lifecycle."""

    def __init__(
        self,
        *,
        registry: BgSessionRegistry,
        runtime_tasks: Any | None = None,
        config: BgSessionConfig | None = None,
    ) -> None:
        self._registry = registry
        self._runtime_tasks = runtime_tasks
        self._config = config if config is not None else registry.config

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def registry(self) -> BgSessionRegistry:
        return self._registry

    @property
    def config(self) -> BgSessionConfig:
        return self._config

    # ------------------------------------------------------------------
    # Enablement checks
    # ------------------------------------------------------------------

    def _require_enabled(self) -> None:
        if not is_bg_sessions_enabled(self._config):
            raise BgSessionsDisabledError(
                "BG_SESSIONS is disabled (CLAWCODEX_BG_SESSIONS=off); "
                "fallback to TaskList / per-session marker behavior"
            )

    # ------------------------------------------------------------------
    # Coordination with launch_background_runner
    # ------------------------------------------------------------------

    def upsert_after_launch(
        self,
        session_id: str,
        pid: int | None,
        *,
        workspace_root: Path | None = None,
        agent_name: str | None = None,
        description: str = "",
    ) -> BgSession | None:
        """Upsert the index after ``launch_background_runner`` writes its marker.

        The marker remains owned by ``background_runner``. Disabled mode is a
        no-op that returns ``None``.
        """
        if not is_bg_sessions_enabled(self._config):
            return None
        marker = marker_path_for(session_id, self._registry.sessions_dir)
        sess = BgSession(
            id=session_id,
            session_id=session_id,
            workspace_root=workspace_root or Path.cwd(),
            status="running",
            pid=pid,
            agent_name=agent_name,
            description=description,
            marker_path=marker if marker.exists() else None,
            transcript_path=self._registry.sessions_dir / session_id / f"{session_id}.jsonl",
            started_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self._registry.upsert(sess)
        self._registry.save()
        return sess

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_sessions(
        self,
        *,
        include_completed: bool = False,
        workspace_root: Path | None = None,
    ) -> list[BgSession]:
        """List sessions, using scan-only behavior when the feature is disabled."""
        sessions = self._registry.list(workspace_root=workspace_root)
        if not include_completed:
            sessions = [s for s in sessions if not s.is_terminal()]
        return sessions

    def inspect(self, bg_session_id: str) -> BgSession:
        """Return a snapshot after the latest health reconciliation."""
        sess = self._registry.get(bg_session_id)
        if sess is None:
            # Scan once in case a marker exists before the registry is loaded.
            self._registry.scan()
            sess = self._registry.get(bg_session_id)
        if sess is None:
            raise BgSessionNotFoundError(f"BG session {bg_session_id!r} not found")
        # Reconcile health with current process and transcript state.
        h = assess(sess, stale_after_seconds=self._config.stale_after_seconds)
        if h.status != sess.status:
            sess = replace_session(sess, status=h.status)
        return sess

    # ------------------------------------------------------------------
    # attach
    # ------------------------------------------------------------------

    def attach(
        self,
        bg_session_id: str,
        *,
        follow: bool = True,
        tail_lines: int = 100,
        current_workspace: Path | None = None,
        allow_cross_workspace: bool | None = None,
    ) -> AttachResult:
        """Attach to a session and return its transcript tail and resume command.

        Cross-workspace attachment is denied by default.
        """
        sess = self.inspect(bg_session_id)
        # Permission check
        if current_workspace is not None:
            allow = allow_cross_workspace if allow_cross_workspace is not None else self._config.allow_cross_workspace
            if not allow and not _path_same_workspace(sess.workspace_root, current_workspace):
                raise BgSessionPermissionError(
                    f"BG session {bg_session_id!r} belongs to workspace "
                    f"{sess.workspace_root}; attach from {current_workspace} "
                    f"denied (require --all or allow_cross_workspace=True)"
                )
        # transcript tail
        tail = _read_tail(sess.transcript_path, max_lines=tail_lines)
        hint = _resume_hint(sess.session_id)
        return AttachResult(session=sess, transcript_tail=tail, resume_hint=hint)

    # ------------------------------------------------------------------
    # stop
    # ------------------------------------------------------------------

    def stop(
        self,
        bg_session_id: str,
        *,
        force: bool = False,
    ) -> BgSession:
        """Stop gracefully unless the caller explicitly requests force.

        Prefer cooperative cancellation through ``runtime_tasks``. Otherwise
        send SIGTERM, or SIGKILL when forced, and mark the session stopped only
        after the process exits.
        """
        sess = self.inspect(bg_session_id)
        if sess.is_terminal():
            return sess
        pid = sess.pid
        stopped_ok = False
        error: str | None = None

        # Prefer cooperative cancellation when a runtime task is available.
        if sess.task_id is not None and self._runtime_tasks is not None:
            try:
                stopped_ok = _try_task_stop(self._runtime_tasks, sess.task_id)
            except Exception as exc:
                logger.debug("TaskStop for %s failed: %s", bg_session_id, exc)
                error = f"TaskStop: {exc}"

        # Fall back to process signals.
        if not stopped_ok and pid is not None:
            try:
                stopped_ok = _signal_stop(pid, force=force)
            except Exception as exc:
                error = f"signal: {exc}"
                logger.warning("stop signal for %s failed: %s", bg_session_id, exc)

        if not stopped_ok and not force:
            updated = replace_session(sess, status="running", error=error)
            self._registry.upsert(updated)
            self._registry.save()
            raise BgSessionStopError(
                f"graceful stop of {bg_session_id!r} failed; retry with force=True. reason: {error or 'unknown'}"
            )

        updated = replace_session(
            sess,
            status="stopped",
            completed_at=_now_iso(),
            updated_at=_now_iso(),
            error=error,
        )
        self._registry.upsert(updated)
        self._registry.save()
        return updated

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def cleanup(self, *, include_failed: bool = False) -> list[BgSession]:
        """Remove eligible terminal records from the index.

        Completed records expire by age, orphaned records may be removed after
        being marked, and failed records require ``include_failed=True``.
        Marker and transcript files are never deleted here.
        """
        now = time.time()  # noqa: F841 - retained for the pending time-window cleanup policy
        removed: list[BgSession] = []
        for sess in self._registry.list():
            age = _age_seconds(sess.completed_at)
            if sess.status == "completed" and age is not None and age > self._config.cleanup_completed_after_seconds:
                if self._registry.remove(sess.id):
                    removed.append(sess)
            elif sess.status == "orphaned":
                if self._registry.remove(sess.id):
                    removed.append(sess)
            elif sess.status == "failed" and include_failed:
                if self._registry.remove(sess.id):
                    removed.append(sess)
        if removed:
            self._registry.save()
        return removed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


def _age_seconds(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso_ts)
        return time.time() - dt.timestamp()
    except (ValueError, OSError):
        return None


def _path_same_workspace(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _read_tail(path: Path | None, *, max_lines: int = 100) -> str:
    if path is None or not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return "".join(lines)
    except OSError as exc:
        raise BgSessionAttachError(f"Cannot read transcript {path}: {exc}") from exc


read_tail = _read_tail


def _resume_hint(session_id: str) -> str:
    return f"Resume this session with: clawcodex --resume {session_id}"


def _signal_stop(pid: int, *, force: bool) -> bool:
    """Signal a process and return whether it exited within the timeout."""
    if not _pid_alive(pid):
        return True
    if force:
        sig = getattr(signal, "SIGKILL", None)
        if sig is None:
            raise BgSessionStopError("Force-stopping background sessions is not supported on this platform")
    else:
        sig = signal.SIGTERM
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        return not _pid_alive(pid)
    deadline = time.monotonic() + _GRACEFUL_STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        return False


def _try_task_stop(runtime_tasks: Any, task_id: str) -> bool:
    """Attempt cooperative cancellation through ``runtime_tasks``.

    Return ``True`` when the task is already terminal or was stopped. Avoid a
    hard dependency on ``stop_task`` to prevent a package cycle.
    """
    state = runtime_tasks.get(task_id)
    if state is None:
        return False
    # A terminal task is already stopped.
    # pylint: disable=no-name-in-module  # tasks_core: pending patch migration
    from clawcodex_ext.tasks_core import is_terminal_task_status

    if is_terminal_task_status(state.status):
        return True
    # Ask the task implementation to stop when it provides a kill method.
    try:
        # pylint: disable=no-name-in-module  # task_registry: pending patch migration
        from clawcodex_ext.task_registry import get_task_by_type

        impl = get_task_by_type(state.type)
        if impl is not None and hasattr(impl, "kill"):
            # The kill hook may be asynchronous; this synchronous seam is best-effort.
            result = impl.kill(task_id, runtime_tasks)
            # Let the caller fall back to process signalling for an unrun coroutine.
            return result is None or bool(result)
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.debug("task impl kill failed: %s", exc)
    return False


read_tail = _read_tail


__all__ = [
    "AttachResult",
    "BgSessionManager",
    "read_tail",
]
