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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""Daemon state file IO."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .constants import (
    DAEMON_STATE_DIRNAME,
    DAEMON_STATE_FILENAME_EXT,
    DAEMON_STATE_SUBDIR,
)

logger = logging.getLogger(__name__)


class DaemonStatus(str, Enum):
    """High-level daemon lifecycle status (stored in state files)."""

    RUNNING = "running"
    STOPPED = "stopped"
    STALE = "stale"  #: state file exists but PID is dead
    ERROR = "error"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DaemonState:
    """Persistent daemon state; ``pid`` is the *supervisor's* PID."""

    pid: int
    cwd: str
    started_at: str
    worker_kinds: list[str]
    name: str = "remote-control"
    last_status: DaemonStatus = DaemonStatus.RUNNING
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["last_status"] = self.last_status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DaemonState":
        kwargs = dict(data)
        status_str = kwargs.pop("last_status", DaemonStatus.RUNNING.value)
        try:
            kwargs["last_status"] = DaemonStatus(status_str)
        except ValueError:
            logger.warning("DaemonState: unknown status %r; defaulting to RUNNING", status_str)
            kwargs["last_status"] = DaemonStatus.RUNNING
        kwargs.setdefault("extra", {})
        _validate_state_fields(kwargs)
        return cls(**kwargs)


def _validate_state_fields(kwargs: dict[str, Any]) -> None:
    """Reject corrupt state payloads (wrong field types) before construction."""
    pid = kwargs.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        raise TypeError(f"DaemonState.pid must be an int, got {type(pid).__name__}")
    for field_name in ("cwd", "started_at", "name"):
        value = kwargs.get(field_name)
        if not isinstance(value, str):
            raise TypeError(f"DaemonState.{field_name} must be a str, got {type(value).__name__}")
    worker_kinds = kwargs.get("worker_kinds")
    if not isinstance(worker_kinds, list) or not all(isinstance(k, str) for k in worker_kinds):
        raise TypeError("DaemonState.worker_kinds must be a list[str]")
    if not isinstance(kwargs.get("extra"), dict):
        raise TypeError(f"DaemonState.extra must be a dict, got {type(kwargs.get('extra')).__name__}")


def get_state_dir(state_dir: Path | None = None) -> Path:
    """Return the daemon state directory (*state_dir* or ``~/.clawcodex/daemon``)."""
    if state_dir is not None:
        return state_dir
    return Path.home() / DAEMON_STATE_DIRNAME / DAEMON_STATE_SUBDIR


def get_state_path(name: str, *, state_dir: Path | None = None) -> Path:
    """Return the absolute path of a daemon state file."""
    _validate_state_name(name)
    return get_state_dir(state_dir) / f"{name}{DAEMON_STATE_FILENAME_EXT}"


def _validate_state_name(name: str) -> None:
    """Reject names that could escape the state directory."""
    if not isinstance(name, str) or not name:
        raise ValueError("daemon state name must be a non-empty string")
    if "/" in name or "\\" in name:
        raise ValueError(f"invalid daemon state name: {name!r}")


def write_daemon_state(state: DaemonState, *, state_dir: Path | None = None) -> Path:
    """Persist *state* atomically (unique tmp + ``os.replace``); returns the final path."""
    target = get_state_path(state.name, state_dir=state_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_dict(), indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_name, target)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return target


def read_daemon_state(
    name: str = "remote-control",
    *,
    state_dir: Path | None = None,
) -> DaemonState | None:
    """Load a daemon state from disk; ``None`` if missing, unreadable, or corrupt."""
    path = get_state_path(name, state_dir=state_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("DaemonState: failed to read %s", path, exc_info=True)
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("DaemonState: corrupt JSON at %s; ignoring", path)
        return None
    try:
        return DaemonState.from_dict(data)
    except (TypeError, KeyError):
        logger.warning("DaemonState: missing fields in %s; ignoring", path, exc_info=True)
        return None


def remove_daemon_state(
    name: str = "remote-control",
    *,
    state_dir: Path | None = None,
) -> None:
    """Remove a daemon state file. No-op if missing."""
    path = get_state_path(name, state_dir=state_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("DaemonState: failed to remove %s", path, exc_info=True)


def is_process_alive(pid: int) -> bool:
    """Return ``True`` if *pid* is alive."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _pid_is_alive_win32(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_is_alive_win32(pid: int) -> bool:
    """Check if a process is running on Windows using kernel32.OpenProcess."""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    PROCESS_QUERY_INFORMATION = 0x0400
    handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
    if not handle:
        err = ctypes.get_last_error()
        if err == 5:  # ERROR_ACCESS_DENIED
            return True
        return False
    kernel32.CloseHandle(handle)
    return True


def query_daemon_status(
    name: str = "remote-control",
    *,
    state_dir: Path | None = None,
) -> tuple[DaemonStatus, DaemonState | None]:
    """Resolve the high-level status of *name* as ``(status, state)``."""
    state = read_daemon_state(name, state_dir=state_dir)
    if state is None:
        return DaemonStatus.STOPPED, None
    if is_process_alive(state.pid):
        return DaemonStatus.RUNNING, state
    remove_daemon_state(name, state_dir=state_dir)
    return DaemonStatus.STALE, None


def make_state(
    *,
    pid: int,
    worker_kinds: list[str],
    name: str = "remote-control",
    cwd: Path | None = None,
) -> DaemonState:
    """Build a fresh :class:`DaemonState` with ``started_at`` pre-filled."""
    return DaemonState(
        pid=pid,
        cwd=str((cwd or Path.cwd()).resolve()),
        started_at=_utcnow_iso(),
        worker_kinds=list(worker_kinds),
        name=name,
        last_status=DaemonStatus.RUNNING,
    )
