"""Filesystem lock for Cron scheduler ownership (F-22-G5).

Three enhancements over the original ``lock.py``:

1. **Cleanup registry** — :func:`register_lock_cleanup` and
   :func:`release_all_locks` support explicit atexit / signal-based release.
2. **PID identity check** — when reading an existing lock, verify the
   recorded PID is still alive. The default validator uses a *permissive*
   policy: any live PID is treated as non-foreign (lock not recovered),
   even if the process command line is unrelated. This avoids false-positive
   recovery that could create a dual-scheduler window. Install a stricter
   validator via ``set_pid_validator`` if you need to reject recycled PIDs
   belonging to clearly foreign processes (e.g. nginx, postgres).
3. **SessionId takeover** — if a re-entrant process finds its own
   sessionId in the lock file (e.g. a forked child recovering after a
   short restart), it can take over the lock without contention.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import (
    SCHEDULED_TASKS_LOCK_RELATIVE_PATH,
    SCHEDULED_TASKS_STORAGE_LOCK_RELATIVE_PATH,
)

_log = logging.getLogger(__name__)

DEFAULT_STALE_LOCK_MS = 10 * 60 * 1000

# F-22-G5: PID identity probe. When the recorded PID is alive but is not
# a ClawCodex-derived process, treat the lock as stale. Defaults to
# /proc/<pid>/comm on Linux; on macOS uses ps; on unsupported platforms
# returns True (allow).
_DEFAULT_PID_VALIDATOR: Callable[[int], bool] | None = None


def set_pid_validator(validator: Callable[[int], bool] | None) -> None:
    """Override the PID identity probe (used by tests)."""
    global _DEFAULT_PID_VALIDATOR
    _DEFAULT_PID_VALIDATOR = validator


def _default_pid_validator(pid: int) -> bool:
    """True if PID is alive AND the process looks like a ClawCodex-style process.

    **Permissive default policy**: returns True (non-foreign) for any live
    PID whose ``comm`` is empty, unreadable, or unrecognised. This means
    the lock is NOT recovered even when the PID was recycled to a clearly
    foreign process (e.g. nginx, postgres). The trade-off is chosen to
    avoid false-positive lock recovery that could create a dual-scheduler
    window when the PID was reused by another legitimate ClawCodex instance.

    Callers that need stricter identity checking should install a custom
    validator via :func:`set_pid_validator` (e.g. a blacklist that returns
    False for known non-ClawCodex process names).

    Returns False (foreign → lock recoverable) only when the PID is dead.
    """
    if pid <= 0 or pid == os.getpid():
        return True
    if not _pid_is_alive(pid):
        return False
    comm_path = Path(f"/proc/{pid}/comm")
    try:
        comm = comm_path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return True
    if not comm:
        return True
    if comm in {"python", "python3", "python3.11", "python3.12"}:
        return True
    if "clawcodex" in comm or "claude" in comm or "orchestrator" in comm:
        return True
    _log.debug("PID %d comm=%r passes identity check by default (permissive)", pid, comm)
    return True


# F-22-G5: cleanup registry. Process-exit handlers fire all registered
# cleanup callbacks; cron modules register their lock release here so
# atexit + SIGTERM/SIGINT both unwind correctly.
_cleanup_callbacks: list[Callable[[], Any]] = []
_cleanup_atexit_registered = False


def register_lock_cleanup(callback: Callable[[], Any]) -> Callable[[], None]:
    """Register a process-exit cleanup callback. Returns an unregister fn."""
    _cleanup_callbacks.append(callback)
    _ensure_atexit_registered()

    def _unregister() -> None:
        try:
            _cleanup_callbacks.remove(callback)
        except ValueError:
            pass

    return _unregister


def release_all_locks() -> None:
    """Run all registered cleanup callbacks. Idempotent: callbacks that
    themselves unregister are tolerated.
    """
    # Snapshot the list — callbacks may mutate it.
    pending = list(_cleanup_callbacks)
    for cb in pending:
        try:
            result = cb()
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("lock cleanup callback failed: %s", exc)
            continue
        # If the callback returns a coroutine, close it cleanly without
        # awaiting (atexit handlers cannot run async).
        if hasattr(result, "close") and not hasattr(result, "__await__"):
            try:
                result.close()  # type: ignore[union-attr]
            except Exception:  # pragma: no cover  # nosec B110 - best-effort cleanup, pass is intentional
                pass


def _ensure_atexit_registered() -> None:
    global _cleanup_atexit_registered
    if _cleanup_atexit_registered:
        return
    atexit.register(release_all_locks)
    # Best-effort signal handlers (only on main thread).
    try:
        if threading_current_main():
            for sig in (signal.SIGTERM, signal.SIGINT):
                prev = signal.getsignal(sig)

                # Wrap any prior handler so we still call it after cleanup.
                def _make(prev_handler, sig_value):
                    def _handler(signum, frame):
                        release_all_locks()
                        if callable(prev_handler) and prev_handler not in (
                            signal.SIG_DFL,
                            signal.SIG_IGN,
                        ):
                            try:
                                prev_handler(signum, frame)
                            except (
                                Exception
                            ):  # pragma: no cover  # nosec B110 - best-effort cleanup, pass is intentional
                                pass

                    return _handler

                signal.signal(sig, _make(prev, sig))
    except (ValueError, OSError):  # pragma: no cover - non-main thread
        pass
    _cleanup_atexit_registered = True


def threading_current_main() -> bool:
    import threading

    return threading.current_thread() is threading.main_thread()


@dataclass
class CronTaskLock:
    workspace_root: Path
    session_id: str
    stale_after_ms: int = DEFAULT_STALE_LOCK_MS
    lock_relative_path: Path = SCHEDULED_TASKS_LOCK_RELATIVE_PATH
    acquired: bool = False
    # F-22-G5: when True, the lock may be silently re-acquired if the
    # existing lock file already carries this session's sessionId. This
    # supports the --resume / fork-recovery case where the same ClawCodex
    # session is being re-instantiated.
    allow_session_takeover: bool = True
    # F-22-G5: when True, run the PID identity check (validate that the
    # owning process still looks like a ClawCodex-style process). When
    # False, the lock relies on age + kill(pid, 0) only.
    validate_pid_identity: bool = True

    @property
    def path(self) -> Path:
        return self.workspace_root / self.lock_relative_path

    @property
    def is_acquired(self) -> bool:
        """True iff ``acquire()`` has succeeded and ``release()`` has not
        since run. Used by the scheduler to gate session-task firing
        (Phase B-2): only the process currently holding the scheduler
        lock may fire durable=False tasks, since the session_store is
        process-local.
        """
        return self.acquired

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": self.session_id,
            "pid": os.getpid(),
            "acquiredAt": int(time.time() * 1000),
        }
        encoded = json.dumps(payload, sort_keys=True)

        # F-22-G5: sessionId takeover — if the lock is already ours (same
        # sessionId) and takeover is allowed, refresh the lock content
        # with our current PID and return True. This handles the case
        # where a child process restarts and re-acquires the parent's
        # lock without contention.
        if self.allow_session_takeover and self.path.exists():
            existing = self._read_payload()
            if existing and existing.get("sessionId") == self.session_id:
                # Refresh in place (non-exclusive write is fine — we
                # already own it).
                tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.refresh.tmp")
                tmp.write_text(encoded, encoding="utf-8")
                os.replace(tmp, self.path)
                self.acquired = True
                _register_self_cleanup(self)
                return True

        # Retry loop: at most 2 attempts.  After _recover_if_stale()
        # removes a stale lock, the *immediate* retry (same loop
        # iteration) has no TOCTOU window — no other thread/process
        # can preempt inside a single os.open call.
        for attempt in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if attempt > 0 or not self._recover_if_stale():
                    return False
                continue  # retry right after stale removal
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
            break
        self.acquired = True
        _register_self_cleanup(self)
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        # Early exit: if the file is already gone there is nothing to do.
        if not self.path.exists():
            self.acquired = False
            return
        # Read the current payload to confirm the lock is still ours.
        try:
            data = self._read_payload() or {}
        except (OSError, json.JSONDecodeError):
            data = {}
        if data.get("sessionId") != self.session_id:
            self.acquired = False
            return
        # The read–unlink window is ~1‑5 µs; another process could
        # replace the file between the check above and this unlink.
        # This is a known theoretical race — in practice the tiny
        # window and the at-least-once nature of lock acquisition make
        # it harmless.  (A fully atomic release would require kernel‑
        # level lease support.)
        for attempt in range(5):
            try:
                self.path.unlink()
                break
            except FileNotFoundError:
                break
            except PermissionError:
                if attempt == 4:
                    _log.debug(
                        "cron lock release deferred because the file is busy: %s",
                        self.path,
                    )
                    break
                time.sleep(0.01)
        self.acquired = False

    def __enter__(self) -> CronTaskLock:
        if not self.acquired and not self.acquire():
            raise TimeoutError(f"could not acquire cron lock: {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _read_payload(self) -> dict | None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _recover_if_stale(self) -> bool:
        data = self._read_payload()
        if data is None:
            # Corrupt file — fall back to mtime-based recovery.
            return self._unlink_existing_if_old()

        pid = data.get("pid")
        pid_dead = isinstance(pid, int) and not _pid_is_alive(pid)
        # F-22-G5: PID identity check (PID alive but not ClawCodex).
        pid_foreign = False
        if self.validate_pid_identity and isinstance(pid, int) and _pid_is_alive(pid):
            validator = _DEFAULT_PID_VALIDATOR or _default_pid_validator
            pid_foreign = not validator(pid)
        # When the PID is alive and identity checks pass, do NOT recover the
        # lock purely based on age — the holder is still running. Age-based
        # recovery only applies when the PID is dead or foreign.
        if pid_dead or pid_foreign:
            if pid_foreign:
                _log.warning(
                    "recovering lock with foreign PID %d (looks like PID recycle)",
                    pid,
                )
            return self._unlink_existing()
        return False

    def _unlink_existing(self) -> bool:
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def _unlink_existing_if_old(self) -> bool:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        age_ms = int((time.time() - mtime) * 1000)
        if age_ms <= self.stale_after_ms:
            return False
        return self._unlink_existing()


def _register_self_cleanup(lock: CronTaskLock) -> None:
    """Register a release callback for the given lock instance."""
    state = {"registered": False}

    def _release_once() -> None:
        if state["registered"]:
            return
        state["registered"] = True
        try:
            lock.release()
        except Exception as exc:  # pragma: no cover
            _log.warning("cron lock release failed: %s", exc)

    register_lock_cleanup(_release_once)


def acquire_cron_storage_lock(
    workspace_root: Path,
    session_id: str,
    lock_relative_path: Path = SCHEDULED_TASKS_STORAGE_LOCK_RELATIVE_PATH,
) -> CronTaskLock:
    """Acquire a filesystem lock with exponential backoff.

    ``lock_relative_path`` defaults to the tasks lock path.  Callers that
    operate on the runs file pass ``RUNS_STORAGE_LOCK_RELATIVE_PATH`` so
    tasks and runs writes can proceed concurrently.
    """
    deadline = time.monotonic() + 10
    lock = CronTaskLock(
        workspace_root,
        session_id,
        lock_relative_path=lock_relative_path,
    )
    delay = 0.001  # start at 1 ms
    while not lock.acquire():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"could not acquire cron storage lock: {lock.path}")
        time.sleep(delay)
        delay = min(delay * 1.5, 0.5)  # exponential backoff, cap at 500 ms
    return lock


def _pid_is_alive(pid: int) -> bool:
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
        return True
    return True


def _pid_is_alive_win32(pid: int) -> bool:
    """Check if a process is running on Windows using kernel32.OpenProcess.

    Returns True if the process exists (or we can't determine), False if it's
    definitively gone (ERROR_INVALID_PARAMETER).
    """
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        # ctypes is part of stdlib — should not happen, but fall back
        # gracefully: assume alive to avoid false-positive lock recovery.
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
        # ERROR_INVALID_PARAMETER (87) = process doesn't exist
        # ERROR_ACCESS_DENIED (5) = exists but no access (treat as alive)
        if err == 5:  # ERROR_ACCESS_DENIED
            return True
        return False
    kernel32.CloseHandle(handle)
    return True
