#!/usr/bin/env python3
# coding=utf-8

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

"""Cross-platform file-lock utilities.

Provides a safe ``fcntl.flock`` wrapper and a cross-platform context manager.
The context manager uses ``msvcrt.locking`` on Windows; the lower-level flock
helpers remain no-ops when ``fcntl`` is unavailable.

Usage::

    from clawcodex_ext.utils.file_lock import (
        HAS_FLOCK,
        exclusive_file_lock,
        flock_exclusive,
        flock_unlock,
    )

    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    if HAS_FLOCK:
        flock_exclusive(fd)          # may raise OSError
    …
    flock_unlock(fd)
    os.close(fd)

Or with the context manager::

    with exclusive_file_lock(lock_path) as fd:
        …
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

try:
    import fcntl as _fcntl  # type: ignore[import-not-found]

    HAS_FLOCK = True
except ImportError:
    _fcntl = None  # type: ignore[assignment]
    HAS_FLOCK = False

try:
    import msvcrt as _msvcrt  # type: ignore[import-not-found]
except ImportError:
    _msvcrt = None  # type: ignore[assignment]


def flock_exclusive(fd: int, non_blocking: bool = False) -> None:
    """Acquire an exclusive ``flock`` on *fd*.

    On Windows this is a no-op.  Raises ``BlockingIOError`` (subclass of
    ``OSError``) when *non_blocking* is ``True`` and the lock is held by
    another process.
    """
    if not HAS_FLOCK or _fcntl is None:
        return
    flags = _fcntl.LOCK_EX
    if non_blocking:
        flags |= _fcntl.LOCK_NB
    _fcntl.flock(fd, flags)


def flock_unlock(fd: int) -> None:
    """Release a ``flock`` on *fd* (no-op on Windows)."""
    if not HAS_FLOCK or _fcntl is None:
        return
    try:
        _fcntl.flock(fd, _fcntl.LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def exclusive_file_lock(lock_path: str | Path) -> Iterator[int]:
    """Acquire an exclusive advisory lock on *lock_path*.

    Uses ``fcntl.flock`` on POSIX and locks the first byte with
    ``msvcrt.locking`` on Windows. Yields the open file descriptor.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR)
    windows_locked = False
    try:
        if _msvcrt is not None:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
            windows_locked = True
        else:
            flock_exclusive(fd)
        yield fd
    finally:
        if windows_locked and _msvcrt is not None:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            flock_unlock(fd)
        os.close(fd)
