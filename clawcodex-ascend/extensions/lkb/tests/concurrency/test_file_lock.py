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
# pylint: disable=C0117
"""Concurrency tests for the LKB file_lock module.

Covers spec §7.4 and test matrix entries:

* LKB-STORE-018 — Windows replace sharing violation (the lock itself must
  survive sharing-violation-style contention; we test multi-process lock
  contention here, the replace-level test belongs with atomic_file.py).
* LKB-STORE-020 — live OS lock + stale owner metadata must not be broken.
* LKB-STORE-021 — process holding the lock exits forcefully; a new process
  can acquire the lock because the OS releases it on FD close.
* LKB-STORE-022 — lock-wait timeout raises board_store_busy and does NOT
  silently fall back to an unlocked mode.
* LKB-STORE-023 — 16 subprocesses contending for the same lock serialize
  correctly (no lost updates to a shared counter file).

All tests use ``tmp_lkb_root`` from the shared conftest so they never touch
the real HOME directory.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from lkb.file_lock import (
    BoardFileLock,
    BoardLockError,
    BoardStoreBusyError,
    acquire_board_locks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_counter(lock_dir: Path, value: int) -> None:
    """Write an integer to ``counter.json`` inside *lock_dir*."""
    path = lock_dir / "counter.json"
    tmp = lock_dir / f"counter.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps({"value": value}), encoding="utf-8")
    os.replace(tmp, path)


def _read_counter(lock_dir: Path) -> int:
    path = lock_dir / "counter.json"
    if not path.exists():
        return 0
    return json.loads(path.read_text(encoding="utf-8"))["value"]


# ---------------------------------------------------------------------------
# LKB-STORE-022 — lock-wait timeout
# ---------------------------------------------------------------------------


def test_lkb_store_022_lock_timeout_raises_busy(tmp_lkb_root: Path) -> None:
    """Timeout when another thread holds the lock -> BoardStoreBusyError."""
    board_dir = tmp_lkb_root / "boards" / "board-a"
    board_dir.mkdir(parents=True, exist_ok=True)

    barrier = threading.Barrier(2)
    error_holder: list[Exception | None] = [None]

    def holder() -> None:
        try:
            lock = BoardFileLock(board_dir, timeout=5.0)
            with lock:
                barrier.wait()  # signal "I have the lock"
                time.sleep(2.0)  # hold it long enough for the waiter to time out
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    barrier.wait()  # wait for holder to enter the lock

    # Try to acquire with a very short timeout — should fail.
    waiter = BoardFileLock(board_dir, timeout=0.1)
    with pytest.raises(BoardStoreBusyError):
        waiter.acquire()

    t.join(timeout=5.0)
    assert not t.is_alive(), "holder thread did not finish"
    assert error_holder[0] is None, f"holder thread raised: {error_holder[0]}"


def test_lkb_store_022_no_fallback_to_unlocked(tmp_lkb_root: Path) -> None:
    """On timeout we must NOT silently proceed with an unlocked state."""
    board_dir = tmp_lkb_root / "boards" / "board-b"
    board_dir.mkdir(parents=True, exist_ok=True)

    barrier = threading.Barrier(2)

    def holder() -> None:
        lock = BoardFileLock(board_dir, timeout=5.0)
        with lock:
            barrier.wait()
            time.sleep(2.0)

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    barrier.wait()

    waiter = BoardFileLock(board_dir, timeout=0.1)
    with pytest.raises(BoardStoreBusyError):
        waiter.acquire()
    # After the exception, the waiter instance must NOT consider itself held.
    assert not waiter.is_held

    t.join(timeout=5.0)


# ---------------------------------------------------------------------------
# LKB-STORE-020 — stale owner file must not break a live lock
# ---------------------------------------------------------------------------


def test_lkb_store_020_stale_owner_file_not_broken(tmp_lkb_root: Path) -> None:
    """A pre-existing .lock.owner.json must not prevent a new lock acquisition
    — and more importantly, must not cause us to "force-break" a live OS
    lock that another process actually holds.

    We test the case where the owner file is stale (written by a previous
    run) and the OS lock is *not* held — acquisition should succeed and the
    owner file should be overwritten with fresh metadata.
    """
    board_dir = tmp_lkb_root / "boards" / "board-c"
    board_dir.mkdir(parents=True, exist_ok=True)

    # Plant a stale owner file (with a fake PID from a "dead" process).
    stale_owner = board_dir / ".lock.owner.json"
    stale_owner.write_text(
        json.dumps(
            {
                "pid": 99999,
                "host": "old-host",
                "user": "old-user",
                "command": "old-command",
                "acquired_at": time.time() - 3600,
                "platform": sys.platform,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    lock = BoardFileLock(board_dir, timeout=1.0)
    with lock:
        # The owner file should have been overwritten with *our* metadata.
        owner = lock.read_owner()
        assert owner is not None
        assert owner["pid"] == os.getpid(), "owner file was not updated to current pid"
        # The stale pid must not be there.
        assert owner["pid"] != 99999

    # After release, owner file should be deleted.
    assert not lock.owner_path.exists(), "owner file was not cleaned up on release"


def test_lkb_store_020_owner_file_diagnostic_only(tmp_lkb_root: Path) -> None:
    """The presence of an owner file MUST NOT be used as the lock itself.

    We test that the OS lock, not the owner file, is the authority:
    1. Thread A holds the OS lock.
    2. We manually mess with the owner file (simulating a stale one).
    3. Thread B still cannot acquire the lock — because the OS lock is held,
       regardless of what owner.json says.
    """
    board_dir = tmp_lkb_root / "boards" / "board-d"
    board_dir.mkdir(parents=True, exist_ok=True)

    barrier = threading.Barrier(2)
    holder_done = threading.Event()

    def holder() -> None:
        lock = BoardFileLock(board_dir, timeout=5.0)
        with lock:
            barrier.wait()  # tell main thread we have the lock
            holder_done.wait()  # hold until released

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    barrier.wait()  # holder is now inside the lock

    # Corrupt / remove the owner file — this simulates a stale or missing
    # owner file while the OS lock is still held.
    owner_path = board_dir / ".lock.owner.json"
    try:
        owner_path.unlink()
    except FileNotFoundError:
        pass

    # Now try to acquire from this thread with a short timeout.
    # It MUST fail — the OS lock is held by the holder thread.
    waiter = BoardFileLock(board_dir, timeout=0.2)
    with pytest.raises(BoardStoreBusyError):
        waiter.acquire()

    # Release the holder.
    holder_done.set()
    t.join(timeout=5.0)
    assert not t.is_alive()


# ---------------------------------------------------------------------------
# Same-thread nested acquisition is forbidden
# ---------------------------------------------------------------------------


def test_nested_same_board_lock_forbidden(tmp_lkb_root: Path) -> None:
    """Same thread must not nest acquires on the same board lock."""
    board_dir = tmp_lkb_root / "boards" / "board-e"
    board_dir.mkdir(parents=True, exist_ok=True)

    lock = BoardFileLock(board_dir, timeout=1.0)
    with lock:
        with pytest.raises(BoardLockError, match="Nested acquisition"):
            lock2 = BoardFileLock(board_dir, timeout=0.1)
            lock2.acquire()


def test_different_boards_no_nesting_conflict(tmp_lkb_root: Path) -> None:
    """Two different board locks can be acquired by the same thread."""
    board_a = tmp_lkb_root / "boards" / "board-f-a"
    board_b = tmp_lkb_root / "boards" / "board-f-b"
    board_a.mkdir(parents=True, exist_ok=True)
    board_b.mkdir(parents=True, exist_ok=True)

    lock_a = BoardFileLock(board_a, timeout=1.0)
    lock_b = BoardFileLock(board_b, timeout=1.0)

    with lock_a:
        with lock_b:
            assert lock_a.is_held
            assert lock_b.is_held


# ---------------------------------------------------------------------------
# Thread-level mutual exclusion via process-local RLock
# ---------------------------------------------------------------------------


def test_thread_mutex_counter(tmp_lkb_root: Path) -> None:
    """Multiple threads incrementing a counter under the lock must not lose
    any increments (process-level mutex).
    """
    board_dir = tmp_lkb_root / "boards" / "board-g"
    board_dir.mkdir(parents=True, exist_ok=True)
    _write_counter(board_dir, 0)

    num_threads = 8
    increments_per_thread = 50
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(increments_per_thread):
                lock = BoardFileLock(board_dir, timeout=5.0)
                with lock:
                    val = _read_counter(board_dir)
                    # Small sleep to increase chance of interleaving.
                    time.sleep(0.0001)
                    _write_counter(board_dir, val + 1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive()

    assert not errors, f"worker errors: {errors}"
    assert _read_counter(board_dir) == num_threads * increments_per_thread, (
        f"expected {num_threads * increments_per_thread}, got {_read_counter(board_dir)}"
    )


# ---------------------------------------------------------------------------
# LKB-STORE-023 — 16 subprocesses contending for the same lock
# ---------------------------------------------------------------------------


def _subprocess_worker_main(board_dir_str: str, num_iterations: int) -> int:
    """Entry point run in a subprocess — increment a counter under the lock."""
    board_dir = Path(board_dir_str)
    for _ in range(num_iterations):
        with BoardFileLock(board_dir, timeout=30.0):
            path = board_dir / "counter.json"
            if path.exists():
                val = json.loads(path.read_text(encoding="utf-8"))["value"]
            else:
                val = 0
            # Simulate work that could race.
            time.sleep(0.001)
            tmp = board_dir / f"counter.{os.getpid()}.tmp"
            tmp.write_text(json.dumps({"value": val + 1}), encoding="utf-8")
            os.replace(tmp, path)
    return 0


def _holder_crasher_worker(board_dir_str: str, ready_event_path_str: str) -> None:
    """Spawn-safe worker that exits while holding a BoardFileLock."""
    board_dir = Path(board_dir_str)
    ready_event_path = Path(ready_event_path_str)
    lock = BoardFileLock(board_dir, timeout=5.0)
    lock.acquire()
    ready_event_path.write_text("1", encoding="utf-8")
    time.sleep(0.5)
    os._exit(0)


@pytest.mark.skipif(
    not sys.platform.startswith("linux") and not sys.platform == "darwin",
    reason="multiprocessing tests run on POSIX via WSL",
)
def test_lkb_store_023_sixteen_subprocesses_counter(tmp_lkb_root: Path) -> None:
    """16 subprocesses all increment a shared counter under the board lock.

    If the OS lock works correctly, the final counter equals
    16 * iterations_per_proc.  If locking is broken (e.g. Windows no-op),
    the counter will be lower.
    """
    import multiprocessing

    board_dir = tmp_lkb_root / "boards" / "board-h"
    board_dir.mkdir(parents=True, exist_ok=True)
    _write_counter(board_dir, 0)

    num_procs = 16
    iterations_per_proc = 20
    expected = num_procs * iterations_per_proc

    procs = []
    for _ in range(num_procs):
        p = multiprocessing.Process(
            target=_subprocess_worker_main,
            args=(str(board_dir), iterations_per_proc),
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join(timeout=60.0)
        assert p.exitcode == 0, f"subprocess exited with code {p.exitcode}"

    assert _read_counter(board_dir) == expected, (
        f"counter mismatch: expected {expected}, got {_read_counter(board_dir)} — OS lock may not be working correctly"
    )


# ---------------------------------------------------------------------------
# LKB-STORE-021 — process holding the lock exits; new process can acquire
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not sys.platform.startswith("linux") and not sys.platform == "darwin",
    reason="multiprocessing tests run on POSIX via WSL",
)
def test_lkb_store_021_lock_released_after_crash(tmp_lkb_root: Path) -> None:
    """When a process holding the OS lock exits (even abruptly), the OS
    releases the lock and a new process can acquire it.

    We simulate a "crash" by having a subprocess acquire the lock and then
    exit without explicitly releasing it.  Then we try to acquire from this
    process — it should succeed.
    """
    import multiprocessing

    board_dir = tmp_lkb_root / "boards" / "board-i"
    board_dir.mkdir(parents=True, exist_ok=True)

    ready_event_path = board_dir / "holder_ready.flag"

    spawn_context = multiprocessing.get_context("spawn")
    p = spawn_context.Process(
        target=_holder_crasher_worker,
        args=(str(board_dir), str(ready_event_path)),
    )
    p.start()

    # Wait for the holder to signal it has the lock.
    deadline = time.monotonic() + 5.0
    while not ready_event_path.exists():
        if time.monotonic() > deadline:
            p.kill()
            pytest.fail("holder process never signaled readiness")
        time.sleep(0.05)

    # Wait for the process to actually exit.
    p.join(timeout=10.0)
    assert not p.is_alive()

    # Now try to acquire from this process.  The OS should have released
    # the lock when the subprocess exited.
    lock = BoardFileLock(board_dir, timeout=5.0)
    acquired = False
    try:
        lock.acquire()
        acquired = True
    finally:
        if acquired:
            lock.release()

    assert acquired, (
        "Could not acquire lock after holder process exited — OS may not be releasing the lock on process exit"
    )


# ---------------------------------------------------------------------------
# Multi-lock ordering — acquire_board_locks
# ---------------------------------------------------------------------------


def test_multi_lock_acquire_sorted(tmp_lkb_root: Path) -> None:
    """acquire_board_locks acquires locks in canonical sorted order."""
    board_z = tmp_lkb_root / "boards" / "z-board"
    board_a = tmp_lkb_root / "boards" / "a-board"
    board_m = tmp_lkb_root / "boards" / "m-board"
    for d in (board_z, board_a, board_m):
        d.mkdir(parents=True, exist_ok=True)

    # Pass in reverse order — should be acquired in sorted order.
    locks = acquire_board_locks([board_z, board_a, board_m], timeout=5.0)
    try:
        assert len(locks) == 3
        # Check that all are held.
        for lock in locks:
            assert lock.is_held
    finally:
        for lock in reversed(locks):
            lock.release()


def test_multi_lock_catalog_first(tmp_lkb_root: Path) -> None:
    """Catalog lock is always acquired before board locks."""
    board_a = tmp_lkb_root / "boards" / "a-board"
    catalog_dir = tmp_lkb_root
    board_a.mkdir(parents=True, exist_ok=True)

    locks = acquire_board_locks([board_a], catalog_dir=catalog_dir, timeout=5.0)
    try:
        assert len(locks) == 2
        # First one should be the catalog lock (lives in catalog_dir).
        assert locks[0].lock_path == catalog_dir / ".catalog.lock"
        # Second should be the board lock.
        assert locks[1].lock_path == board_a / ".lock"
    finally:
        for lock in reversed(locks):
            lock.release()


def test_multi_lock_deadlock_prevention(tmp_lkb_root: Path) -> None:
    """Two threads acquiring two boards in opposite orders via
    acquire_board_locks should not deadlock — because the function sorts
    them into a canonical order.
    """
    board_x = tmp_lkb_root / "boards" / "x-board"
    board_y = tmp_lkb_root / "boards" / "y-board"
    for d in (board_x, board_y):
        d.mkdir(parents=True, exist_ok=True)

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker_a() -> None:
        try:
            barrier.wait()
            locks = acquire_board_locks([board_x, board_y], timeout=5.0)
            # Hold briefly then release.
            time.sleep(0.05)
            for lock in reversed(locks):
                lock.release()
        except Exception as e:
            errors.append(e)

    def worker_b() -> None:
        try:
            barrier.wait()
            # Pass in the opposite order.
            locks = acquire_board_locks([board_y, board_x], timeout=5.0)
            time.sleep(0.05)
            for lock in reversed(locks):
                lock.release()
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker_a, daemon=True)
    t2 = threading.Thread(target=worker_b, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not t1.is_alive(), "worker_a deadlocked"
    assert not t2.is_alive(), "worker_b deadlocked"
    assert not errors, f"worker errors: {errors}"


# ---------------------------------------------------------------------------
# .lock anchor file persists (spec §7.2 — not deleted on release)
# ---------------------------------------------------------------------------


def test_lock_anchor_file_persists_after_release(tmp_lkb_root: Path) -> None:
    """The .lock file itself must NOT be deleted on release (spec §7.2)."""
    board_dir = tmp_lkb_root / "boards" / "board-j"
    board_dir.mkdir(parents=True, exist_ok=True)

    lock = BoardFileLock(board_dir, timeout=1.0)
    with lock:
        assert lock.lock_path.exists()

    # After release, the .lock anchor file must still exist.
    assert lock.lock_path.exists(), (
        ".lock file must persist after release — it is a permanent anchor, not a sentinel to be deleted"
    )


# ---------------------------------------------------------------------------
# Error class hierarchy
# ---------------------------------------------------------------------------


def test_error_hierarchy() -> None:
    assert issubclass(BoardStoreBusyError, BoardLockError)


def test_failed_os_lock_retries_do_not_leak_fds(tmp_lkb_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        pytest.skip("FD accounting requires /proc")

    import lkb.file_lock as file_lock_module

    def always_busy(fd: int, non_blocking: bool = False) -> None:
        raise BlockingIOError("busy")

    monkeypatch.setattr(file_lock_module, "_acquire_os_lock", always_busy)
    before = len(list(fd_dir.iterdir()))
    lock = BoardFileLock(
        tmp_lkb_root / "fd-leak",
        timeout=0.03,
        initial_delay=0.001,
        max_delay=0.002,
        jitter_fraction=0,
    )
    with pytest.raises(BoardStoreBusyError):
        lock.acquire()
    after = len(list(fd_dir.iterdir()))
    assert after == before


def test_retry_sleep_is_bounded_by_remaining_deadline(tmp_lkb_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A jittered retry must not oversleep the total acquisition budget."""
    import lkb.file_lock as file_lock_module

    sleeps: list[float] = []
    monkeypatch.setattr(file_lock_module.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(file_lock_module.time, "sleep", sleeps.append)
    lock = BoardFileLock(
        tmp_lkb_root / "deadline-bound",
        jitter_fraction=0,
    )

    lock._sleep_until_retry(1.0, 10.25)

    assert sleeps == [0.25]


def test_windows_nonblocking_lock_only_maps_actual_contention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-contention Windows I/O errors must remain visible to callers."""
    import lkb.file_lock as file_lock_module

    class WindowsOSError(OSError):
        def __init__(self, winerror: int) -> None:
            super().__init__(5, "simulated Windows error")
            self.winerror = winerror

    class FakeMsvcrt:
        LK_NBLCK = 1
        LK_LOCK = 2
        error = WindowsOSError(33)

        @classmethod
        def locking(cls, _fd: int, _mode: int, _length: int) -> None:
            raise cls.error

    monkeypatch.setattr(file_lock_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(file_lock_module, "msvcrt", FakeMsvcrt, raising=False)
    monkeypatch.setattr(file_lock_module.os, "lseek", lambda *_args: 0)
    monkeypatch.setattr(file_lock_module, "_WINDOWS_LOCK_CONFLICT_WINERRORS", frozenset({33}))

    with pytest.raises(BlockingIOError):
        file_lock_module._acquire_os_lock(1, non_blocking=True)

    FakeMsvcrt.error = WindowsOSError(5)
    with pytest.raises(WindowsOSError):
        file_lock_module._acquire_os_lock(1, non_blocking=True)


# ---------------------------------------------------------------------------
# Windows-specific: msvcrt.locking byte-range lock
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_windows_msvcrt_lock_excludes_other_process(tmp_lkb_root: Path) -> None:
    """On Windows, msvcrt.locking must provide real cross-process exclusion.

    This mirrors LKB-STORE-023 but runs specifically on Windows.
    """
    import multiprocessing

    board_dir = tmp_lkb_root / "boards" / "board-win"
    board_dir.mkdir(parents=True, exist_ok=True)
    _write_counter(board_dir, 0)

    num_procs = 8
    iterations_per_proc = 15
    expected = num_procs * iterations_per_proc

    procs = []
    for _ in range(num_procs):
        p = multiprocessing.Process(
            target=_subprocess_worker_main,
            args=(str(board_dir), iterations_per_proc),
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join(timeout=60.0)
        assert p.exitcode == 0, f"subprocess exited with code {p.exitcode}"

    assert _read_counter(board_dir) == expected, (
        f"Windows msvcrt lock failed: expected {expected}, got {_read_counter(board_dir)}"
    )
