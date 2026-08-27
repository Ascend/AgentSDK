#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=W0108

"""Board-level cross-process file lock (spec §7.4).

Implements an exclusive advisory lock on a board directory's ``.lock`` file
using real OS primitives — no ``O_EXCL`` sentinel-file pattern, no deletion
on release, no Windows no-op.

Platforms
---------
* POSIX: ``fcntl.flock(LOCK_EX)`` on the ``.lock`` file descriptor.  The FD
  is held open for the entire critical section; the kernel releases the lock
  on process exit (crash-safe).
* Windows: ``msvcrt.locking(fd, LK_LOCK, length)`` byte-range lock on the
  first byte of ``.lock``.  The FD is likewise held open; the OS releases
  the lock when the handle is closed (crash-safe).

Process-local guard
-------------------
A per-board ``threading.RLock`` is acquired *before* the OS lock and released
*after* it.  This guarantees that two threads in the same process cannot
bypass the OS lock semantics (e.g. on Windows, ``msvcrt.locking`` is
per-process and recursive within the same process, so threads in one process
would otherwise both think they hold the lock exclusively).

Nested acquisition of the *same* board lock from the same thread is forbidden
and raises ``BoardLockError`` — we track acquisition depth per thread.

Owner file
----------
After a successful acquisition we atomically write ``.lock.owner.json``
containing ``{pid, host, command, acquired_at}``.  The file is deleted on
normal release.  A stale owner file is **diagnostic only** — we never break
a live OS lock based on its contents (LKB-STORE-020).

Backoff & timeout
-----------------
Acquisition retries with exponential backoff + random jitter up to a total
timeout (default ~10 s).  On timeout, ``BoardStoreBusyError`` is raised.

Multi-lock ordering
-------------------
``acquire_board_locks`` acquires multiple board locks in a fixed canonical
order (sorted by resolved path).  A special ``.catalog.lock`` always comes
before any board ``.lock`` (spec §7.4).

Unsupported filesystems
-----------------------
On network / sync filesystems where ``flock`` / ``msvcrt.locking`` may not
be reliable, we do a best-effort detection and fail closed with
``BoardStoreUnsupportedFilesystemError`` rather than silently degenerating
to an unlocked write (LKB-FAIL-009).  The detection is conservative — we
prefer a false positive (refusing a safe FS) to a false negative
(corrupting a board on an unsafe FS).
"""

from __future__ import annotations

import errno
import getpass
import json
import os
import random
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .error_codes import LkbErrorCode

# ---------------------------------------------------------------------------
# Platform-specific low-level locking
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"
_WINDOWS_LOCK_CONFLICT_WINERRORS: frozenset[int]

if _IS_WINDOWS:
    import msvcrt  # type: ignore[import]
    import winerror

    _WINDOWS_LOCK_CONFLICT_WINERRORS = frozenset({winerror.ERROR_LOCK_VIOLATION, winerror.ERROR_SHARING_VIOLATION})
else:
    import fcntl  # type: ignore[import-not-found]

    _WINDOWS_LOCK_CONFLICT_WINERRORS = frozenset()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BoardLockError(Exception):
    """Base class for board lock errors."""


class BoardStoreBusyError(BoardLockError):
    """Raised when the board lock cannot be acquired within the timeout."""

    code = LkbErrorCode.BOARD_STORE_BUSY


class BoardStoreUnsupportedFilesystemError(BoardLockError):
    """Raised when the underlying filesystem cannot reliably provide locks.

    Fail-closed: we never silently fall back to an unlocked / in-memory mode
    (LKB-FAIL-009).
    """

    code = LkbErrorCode.BOARD_STORE_UNSUPPORTED_FILESYSTEM


# ---------------------------------------------------------------------------
# Thread-local tracking for nested-acquisition detection
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _thread_acquired_set() -> set[str]:
    s = getattr(_thread_local, "lkb_locks_held", None)
    if s is None:
        s = set()
        _thread_local.lkb_locks_held = s
    return s


# ---------------------------------------------------------------------------
# Per-board process-local RLock registry
# ---------------------------------------------------------------------------

_process_rlocks: dict[str, threading.RLock] = {}
_process_rlocks_lock = threading.Lock()


def _get_process_rlock(canonical_path: str) -> threading.RLock:
    with _process_rlocks_lock:
        rlock = _process_rlocks.get(canonical_path)
        if rlock is None:
            rlock = threading.RLock()
            _process_rlocks[canonical_path] = rlock
        return rlock


# ---------------------------------------------------------------------------
# Canonical path helpers
# ---------------------------------------------------------------------------


def _canonical_lock_path(lock_path: Path) -> str:
    """Return a canonical string key for a lock path.

    We use ``resolve()`` so that two different string representations of the
    same on-disk lock file map to the same process-local RLock.
    """
    try:
        return str(lock_path.resolve())
    except OSError:
        # If we can't resolve (e.g. parent doesn't exist yet), use absolute.
        return str(lock_path.absolute())


# ---------------------------------------------------------------------------
# OS lock primitives
# ---------------------------------------------------------------------------

# Number of bytes we lock in the file.  On POSIX flock is whole-file anyway;
# on Windows msvcrt.locking is a byte range.  We lock byte 0..0 (single byte)
# — enough to provide exclusivity, small enough that tiny files work.
_LOCK_BYTE_LENGTH = 1


def _acquire_os_lock(fd: int, non_blocking: bool = False) -> None:
    """Acquire an exclusive OS lock on *fd*.

    Raises ``BlockingIOError`` (subclass of ``OSError``) when *non_blocking*
    is True and the lock is held by another process.
    """
    if _IS_WINDOWS:
        # msvcrt.locking byte-range lock.
        # Seek to start, then lock _LOCK_BYTE_LENGTH bytes.
        os.lseek(fd, 0, os.SEEK_SET)
        mode = msvcrt.LK_NBLCK if non_blocking else msvcrt.LK_LOCK  # type: ignore[attr-defined]
        try:
            msvcrt.locking(fd, mode, _LOCK_BYTE_LENGTH)  # type: ignore[attr-defined]
        except OSError as exc:
            # msvcrt reports lock contention with a Windows lock/sharing
            # violation. Do not turn permission, descriptor, or I/O faults
            # into retryable contention.
            if non_blocking and getattr(exc, "winerror", None) in _WINDOWS_LOCK_CONFLICT_WINERRORS:
                raise BlockingIOError(
                    getattr(exc, "errno", 0),
                    getattr(exc, "strerror", "lock held"),
                ) from exc
            raise
    else:
        flags = fcntl.LOCK_EX
        if non_blocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(fd, flags)


def _release_os_lock(fd: int) -> None:
    """Release the OS lock on *fd* (best-effort — closing the FD also works)."""
    try:
        if _IS_WINDOWS:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, _LOCK_BYTE_LENGTH)  # type: ignore[attr-defined]
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        # If the FD is already closed or the lock was already released,
        # that's fine — the kernel will clean up when we close the FD.
        pass


# ---------------------------------------------------------------------------
# BoardFileLock
# ---------------------------------------------------------------------------


class BoardFileLock:
    """Exclusive cross-process lock on a board directory's ``.lock`` file.

    Usage::

        lock = BoardFileLock(board_dir)
        with lock:
            # critical section

    The lock is re-entrant at the process level (``threading.RLock``) but
    NOT at the OS-lock level from a single thread — intentionally, because
    the spec forbids nested acquisition of the same board lock (spec §7.4).
    We detect same-thread re-entry and raise ``BoardLockError``.

    Parameters
    ----------
    board_dir:
        Path to the board directory.  The ``.lock`` file lives directly
        inside it.
    timeout:
        Total seconds to wait for the lock before raising
        ``BoardStoreBusyError``.  Default 10 s.
    initial_delay:
        Initial backoff delay in seconds.  Default 1 ms.
    max_delay:
        Maximum backoff delay in seconds.  Default 500 ms.
    backoff_factor:
        Multiplicative backoff factor.  Default 1.5.
    jitter_fraction:
        Fraction of the current delay used as jitter amplitude.  Default 0.3.
    """

    def __init__(
        self,
        board_dir: Path | str,
        *,
        lock_name: str = ".lock",
        timeout: float = 10.0,
        initial_delay: float = 0.001,
        max_delay: float = 0.5,
        backoff_factor: float = 1.5,
        jitter_fraction: float = 0.3,
    ) -> None:
        self._board_dir = Path(board_dir)
        self._lock_path = self._board_dir / lock_name
        self._owner_path = self._board_dir / f"{lock_name}.owner.json"
        self._canonical = _canonical_lock_path(self._lock_path)
        self._timeout = timeout
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._backoff_factor = backoff_factor
        self._jitter_fraction = jitter_fraction
        # State set during acquisition:
        self._fd: int | None = None
        self._rlock: threading.RLock | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @property
    def owner_path(self) -> Path:
        return self._owner_path

    @property
    def is_held(self) -> bool:
        """True if this instance currently holds the OS lock."""
        return self._fd is not None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BoardFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Acquire the board lock.

        Raises ``BoardStoreBusyError`` on timeout, ``BoardLockError`` on
        invalid state (e.g. same-thread re-entry detected).
        """
        canonical = self._canonical
        acquired_set = _thread_acquired_set()

        # 1. Same-thread re-entry detection (forbid per spec §7.4).
        #    Note: threading.RLock is re-entrant, but we want to catch the
        #    case where code accidentally nests with-statements on the same
        #    board lock.  We track via a thread-local set keyed by canonical
        #    path.
        if canonical in acquired_set:
            raise BoardLockError(f"Nested acquisition of same board lock is forbidden: {self._lock_path}")

        # 2. Acquire process-local RLock + OS lock inside a single retry loop.
        #    Both locks must be acquired non-blockingly so we can enforce the
        #    total timeout and respect backoff.  The RLock prevents two threads
        #    in the same process from both "holding" the OS lock (which on
        #    Windows is per-process, and on POSIX flock is also per-process).
        rlock = _get_process_rlock(canonical)
        deadline = time.monotonic() + self._timeout
        delay = self._initial_delay

        while True:
            if time.monotonic() >= deadline:
                raise self._busy_error()
            # a) Try to grab the RLock (non-blocking).
            if not rlock.acquire(blocking=False):
                self._sleep_until_retry(delay, deadline)
                delay = min(delay * self._backoff_factor, self._max_delay)
                continue

            self._rlock = rlock

            try:
                # b) Ensure the .lock anchor file exists and we have an open FD.
                self._ensure_lock_file()
                assert self._fd is not None
                if time.monotonic() >= deadline:
                    raise self._busy_error()

                # c) Try the OS lock (non-blocking).
                try:
                    _acquire_os_lock(self._fd, non_blocking=True)
                except BlockingIOError:
                    # Lock held by another process — release RLock, back off, retry.
                    self._close_fd_only()
                    self._release_rlock_only()
                    self._sleep_until_retry(delay, deadline)
                    delay = min(delay * self._backoff_factor, self._max_delay)
                    continue
                except OSError as exc:
                    # Unsupported filesystem or other OS error — fail closed.
                    self._release_rlock_only()
                    if _is_unsupported_filesystem_error(exc):
                        raise BoardStoreUnsupportedFilesystemError(
                            f"Filesystem does not support reliable file locking ({exc.__class__.__name__}: {exc})"
                        ) from exc
                    raise

                # d) Success — both locks held.  Write owner file + record.
                if time.monotonic() >= deadline:
                    raise self._busy_error()
                self._write_owner_file()
                acquired_set.add(canonical)
                return

            except Exception:
                # Clean up RLock/FD state acquired in this iteration
                # (OS lock was either not acquired or already released above).
                self._cleanup_on_failed_acquire()
                raise

    def _busy_error(self) -> BoardStoreBusyError:
        return BoardStoreBusyError(f"Timed out waiting for board lock after {self._timeout:.1f}s: {self._lock_path}")

    def _sleep_until_retry(self, delay: float, deadline: float) -> None:
        """Sleep no longer than the remaining acquisition budget."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._busy_error()
        time.sleep(min(self._jitter_sleep(delay), remaining))

    def _jitter_sleep(self, base_delay: float) -> float:
        """Apply random jitter to a delay value."""
        # Retry jitter is deliberately non-cryptographic.
        jitter = random.uniform(  # nosec B311
            -base_delay * self._jitter_fraction,
            base_delay * self._jitter_fraction,
        )
        return max(0.0, base_delay + jitter)

    def _release_rlock_only(self) -> None:
        """Release the process-local RLock and clear our reference."""
        if self._rlock is not None:
            self._rlock.release()
            self._rlock = None

    def _close_fd_only(self) -> None:
        """Close a failed-attempt FD before the next retry."""
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None

    def _ensure_lock_file(self) -> None:
        """Create the board dir and .lock file if missing; open the FD."""
        self._board_dir.mkdir(parents=True, exist_ok=True)

        # Open (or create) the lock file.  We keep the FD open for the
        # entire critical section — that is what holds the OS lock.
        #
        # Using O_RDWR | O_CREAT — same pattern as session_index.py.
        # On POSIX, 0o600 mode.  On Windows, os.open doesn't take mode in
        # the same way — we use a try/except to pass mode only where it
        # matters.
        if _IS_WINDOWS:
            fd = os.open(
                str(self._lock_path),
                os.O_RDWR | os.O_CREAT,
            )
        else:
            fd = os.open(
                str(self._lock_path),
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
        self._fd = fd

    def _write_owner_file(self) -> None:
        """Atomically write ``.lock.owner.json`` with diagnostic metadata.

        This file is purely diagnostic — we never break a live OS lock based
        on its contents (LKB-STORE-020).
        """
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"

        payload: dict[str, Any] = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "user": user,
            "command": " ".join(sys.argv) if sys.argv else "",
            "acquired_at": time.time(),
            "platform": sys.platform,
        }
        data = json.dumps(payload, sort_keys=True)

        # Atomic write: write to a temp file in the same directory, then
        # os.replace.  We use a pid/thread-unique temp name.
        tmp_path = self._board_dir / f".lock.owner.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            tmp_path.write_text(data, encoding="utf-8")
            os.replace(tmp_path, self._owner_path)
        except OSError:
            # Owner file is diagnostic only — failure to write it should
            # never prevent lock acquisition.
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _cleanup_on_failed_acquire(self) -> None:
        """Clean up state after a failed acquire attempt."""
        if self._fd is not None:
            _release_os_lock(self._fd)
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if self._rlock is not None:
            self._rlock.release()
            self._rlock = None

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release the board lock.

        Idempotent: calling release() when the lock is not held is a no-op.
        """
        if self._fd is None:
            return

        # 1. Remove owner file (best-effort — diagnostic only).
        try:
            if self._owner_path.exists():
                self._owner_path.unlink()
        except OSError:
            pass

        # 2. Release OS lock.
        _release_os_lock(self._fd)

        # 3. Close the FD.
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None

        # 4. Remove from thread-local tracking.
        try:
            _thread_acquired_set().discard(self._canonical)
        # Thread-local cleanup is best effort during release.
        except Exception:  # nosec B110
            pass

        # 5. Release the process-local RLock.
        if self._rlock is not None:
            self._rlock.release()
            self._rlock = None

    # ------------------------------------------------------------------
    # Owner inspection (diagnostic)
    # ------------------------------------------------------------------

    def read_owner(self) -> dict[str, Any] | None:
        """Read the ``.lock.owner.json`` diagnostic file, if present.

        Returns ``None`` if the file is missing or unparseable.

        IMPORTANT: this is diagnostic only.  A stale owner file does NOT
        mean the OS lock is available — never use this to decide whether to
        break a lock (LKB-STORE-020).
        """
        try:
            raw = self._owner_path.read_text(encoding="utf-8")
            return json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return None


# ---------------------------------------------------------------------------
# Multi-lock acquisition helper
# ---------------------------------------------------------------------------

# Special sentinel file name that always comes first in lock ordering.
CATALOG_LOCK_NAME = ".catalog.lock"


def acquire_board_locks(
    board_dirs: list[Path],
    *,
    catalog_dir: Path | None = None,
    timeout: float = 10.0,
) -> list[BoardFileLock]:
    """Acquire multiple board locks in a fixed canonical order.

    This prevents deadlocks when two operations need multiple board locks
    and acquire them in different orders.  The canonical order is:

    1. ``.catalog.lock`` (if *catalog_dir* is provided)
    2. All board ``.lock`` files, sorted by their canonical resolved path

    Parameters
    ----------
    board_dirs:
        Board directories to lock.  Duplicates are removed.
    catalog_dir:
        If provided, acquire ``.catalog.lock`` inside this directory first.
    timeout:
        Total timeout for the entire acquisition sequence.  Because we
        acquire locks sequentially, a single slow lock can consume the
        whole budget — callers that need fairness should set a generous
        timeout.

    Returns
    -------
    list[BoardFileLock]
        All acquired locks, in acquisition order.  Callers are responsible
        for releasing them (or using a ``contextlib.ExitStack``).

    Raises
    ------
    BoardStoreBusyError
        If any lock cannot be acquired within the total timeout.  All
        already-acquired locks are released before raising.
    """
    # Deduplicate and compute canonical paths.
    seen: set[str] = set()
    unique_dirs: list[Path] = []
    for d in board_dirs:
        canon = _canonical_lock_path(d)
        if canon not in seen:
            seen.add(canon)
            unique_dirs.append(d)

    # Sort by canonical path for a global acquisition order.
    unique_dirs.sort(key=lambda p: _canonical_lock_path(p))

    acquired: list[BoardFileLock] = []
    deadline = time.monotonic() + timeout

    try:
        # 1. Catalog lock first, if requested.
        if catalog_dir is not None:
            lock = BoardFileLock(
                catalog_dir,
                lock_name=CATALOG_LOCK_NAME,
                timeout=max(0.0, deadline - time.monotonic()),
            )
            # The BoardFileLock's timeout is per-lock, but we have a global
            # deadline.  We approximate by passing the remaining budget.
            lock.acquire()
            acquired.append(lock)

        # 2. Board locks in sorted order.
        for board_dir in unique_dirs:
            remaining = max(0.0, deadline - time.monotonic())
            lock = BoardFileLock(board_dir, timeout=remaining)
            lock.acquire()
            acquired.append(lock)

    except Exception:
        # Release in reverse order.
        for lock in reversed(acquired):
            try:
                lock.release()
            # Preserve the original acquire failure raised below.
            except Exception:  # nosec B110
                pass
        raise

    return acquired


# ---------------------------------------------------------------------------
# Unsupported-FS detection helpers
# ---------------------------------------------------------------------------

_UNSUPPORTED_ERRNOS = (
    frozenset(
        value for name in ("EOPNOTSUPP", "ENOTSUP", "ENOSYS") if isinstance((value := getattr(errno, name, None)), int)
    )
    if not _IS_WINDOWS
    else frozenset()
)


def _is_unsupported_filesystem_error(exc: OSError) -> bool:
    """Best-effort detection of lock-unsupported filesystems.

    Conservative: if we're unsure, return False (don't treat as unsupported).
    A false negative is worse than a false positive here (LKB-FAIL-009), but
    we also don't want to spurious-block normal local filesystems.
    """
    errno_val = getattr(exc, "errno", None)
    if errno_val is None:
        return False
    return errno_val in _UNSUPPORTED_ERRNOS
