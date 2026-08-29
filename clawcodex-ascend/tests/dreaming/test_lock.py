#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for ``clawcodex_ext.dreaming.lock``.

Covers the file-based consolidation lock: readLastConsolidatedAt,
tryAcquireConsolidationLock, rollback, recordConsolidation, and the
session scan helper.

Tests use a tmp-path override of the auto-memory dir to keep
filesystem state hermetic.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from clawcodex_ext.dreaming import lock as lock_mod


@pytest.fixture
def memory_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the auto-memory helpers at a temp dir for the test."""
    monkeypatch.setenv("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# readLastConsolidatedAt
# ---------------------------------------------------------------------------


def test_read_last_consolidated_at_zero_when_no_lock(memory_dir: Path) -> None:
    assert lock_mod.read_last_consolidated_at() == 0


def test_read_last_consolidated_at_returns_mtime(memory_dir: Path) -> None:
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    lock_path.write_text("9999", encoding="utf-8")
    os.utime(lock_path, (1_000_000, 1_000_000))  # fixed mtime
    assert lock_mod.read_last_consolidated_at() == 1_000_000_000


# ---------------------------------------------------------------------------
# try_acquire_consolidation_lock
# ---------------------------------------------------------------------------


def test_acquire_lock_first_time_returns_zero_prior(memory_dir: Path) -> None:
    prior = lock_mod.try_acquire_consolidation_lock()
    assert prior == 0
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    assert lock_path.exists()
    assert int(lock_path.read_text()) == os.getpid()


def test_acquire_lock_blocked_by_foreign_live_pid(memory_dir: Path) -> None:
    """A foreign live PID (the parent process) holds the lock → blocked.

    We use ``os.getppid()`` (the test runner's parent) which is
    guaranteed alive during the test and never our own PID.
    """
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    foreign_pid = os.getppid()
    assert foreign_pid != os.getpid(), "parent PID must differ from ours"
    lock_path.write_text(str(foreign_pid), encoding="utf-8")
    now = time.time()
    os.utime(lock_path, (now, now))
    result = lock_mod.try_acquire_consolidation_lock()
    assert result is None  # blocked by foreign live PID


def test_acquire_lock_self_pid_returns_prior_mtime(memory_dir: Path) -> None:
    """If the lock is already held by our own PID, return the prior mtime
    (not ``None``) — self-acquisition is a no-op for the lock file.

    This is the contract: ``record_consolidation`` (called
    by the manual ``/dream`` path) pre-stamps the lock with our own
    PID; the subsequent ``try_acquire_consolidation_lock`` inside the
    dream service must NOT block on that self-stamp.
    """
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    now = time.time()
    os.utime(lock_path, (now, now))
    result = lock_mod.try_acquire_consolidation_lock()
    # We already hold the lock — return the prior mtime.
    assert result is not None
    # The body is still our PID (no re-write).
    assert int(lock_path.read_text()) == os.getpid()


def test_acquire_lock_reclaims_dead_pid(memory_dir: Path) -> None:
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    # PID 1 is almost always alive on Linux, so use a clearly dead pid
    # (max pid, very unlikely to be reused). Skip on platforms where
    # we can't predict.
    lock_path.write_text("999999", encoding="utf-8")
    now = time.time()
    os.utime(lock_path, (now, now))
    result = lock_mod.try_acquire_consolidation_lock()
    # Returns the prior mtime; we got the lock.
    assert result is not None
    assert int(lock_path.read_text()) == os.getpid()


def test_acquire_lock_reclaims_stale_mtime(memory_dir: Path) -> None:
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    # Write a live PID but a stale mtime (> HOLDER_STALE_MS ago).
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    stale = time.time() - lock_mod.HOLDER_STALE_MS / 1000 - 60
    os.utime(lock_path, (stale, stale))
    result = lock_mod.try_acquire_consolidation_lock()
    assert result is not None


# ---------------------------------------------------------------------------
# rollback_consolidation_lock
# ---------------------------------------------------------------------------


def test_rollback_unlinks_when_prior_is_zero(memory_dir: Path) -> None:
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    lock_path.write_text(str(os.getpid()))
    assert lock_path.exists()
    lock_mod.rollback_consolidation_lock(0)
    assert not lock_path.exists()


def test_rollback_rewinds_mtime(memory_dir: Path) -> None:
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    lock_path.write_text("")
    # Set current mtime to "now", then rewind to a known past point.
    now = time.time()
    os.utime(lock_path, (now, now))
    target_ms = 1_700_000_000_000  # arbitrary past
    lock_mod.rollback_consolidation_lock(target_ms)
    actual_ms = int(lock_path.stat().st_mtime * 1000)
    # Allow 1s slop for filesystem mtime resolution.
    assert abs(actual_ms - target_ms) < 1000
    # PID body cleared — our process should not look like the holder.
    assert lock_path.read_text() == ""


def test_rollback_swallows_oserror(monkeypatch, memory_dir: Path) -> None:
    """rollback must not raise even on filesystem failures."""

    def _raise(*_a, **_kw):
        raise OSError("boom")

    monkeypatch.setattr(lock_mod.Path, "write_text", _raise)
    lock_mod.rollback_consolidation_lock(0)  # must not raise


# ---------------------------------------------------------------------------
# record_consolidation
# ---------------------------------------------------------------------------


def test_record_consolidation_writes_pid(memory_dir: Path) -> None:
    lock_mod.record_consolidation()
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    assert lock_path.exists()
    assert int(lock_path.read_text()) == os.getpid()


# ---------------------------------------------------------------------------
# list_sessions_touched_since
# ---------------------------------------------------------------------------


def test_list_sessions_touched_since_empty_when_no_dir(monkeypatch: pytest.MonkeyPatch, memory_dir: Path) -> None:
    # Point the lock module's project_transcript_dir at a non-existent dir.
    monkeypatch.setattr(
        "clawcodex_ext.dreaming.lock.project_transcript_dir",
        lambda *_a, **_kw: "/nonexistent/__no_such_dir__",
    )
    assert lock_mod.list_sessions_touched_since(0) == []


def test_list_sessions_touched_since_filters_by_mtime(monkeypatch: pytest.MonkeyPatch, memory_dir: Path) -> None:
    proj_dir = memory_dir / "project_sessions"
    proj_dir.mkdir()
    # Two old + one recent. Names are arbitrary — the helper treats
    # each child dir / .jsonl file as a candidate.
    (proj_dir / "old_a").mkdir()
    (proj_dir / "old_b").mkdir()
    (proj_dir / "new").mkdir()

    old_time = time.time() - 7200
    new_time = time.time() - 60
    for p in (proj_dir / "old_a", proj_dir / "old_b"):
        os.utime(p, (old_time, old_time))
    os.utime(proj_dir / "new", (new_time, new_time))

    monkeypatch.setattr(
        "clawcodex_ext.dreaming.lock.project_transcript_dir",
        lambda *_a, **_kw: str(proj_dir),
    )
    # since_ms = 1h ago → only the "new" session qualifies.
    since_ms = int((time.time() - 3600) * 1000)
    result = lock_mod.list_sessions_touched_since(since_ms)
    assert result == ["new"]


def test_list_sessions_skips_dotfiles(monkeypatch: pytest.MonkeyPatch, memory_dir: Path) -> None:
    proj_dir = memory_dir / "project_sessions"
    proj_dir.mkdir()
    (proj_dir / ".hidden").mkdir()
    (proj_dir / "visible").mkdir()
    now = time.time()
    os.utime(proj_dir / ".hidden", (now, now))
    os.utime(proj_dir / "visible", (now, now))
    monkeypatch.setattr(
        "clawcodex_ext.dreaming.lock.project_transcript_dir",
        lambda *_a, **_kw: str(proj_dir),
    )
    result = lock_mod.list_sessions_touched_since(0)
    assert result == ["visible"]


# ---------------------------------------------------------------------------
# Phase B — TTL 30min diagnostics & active cleanup
# ---------------------------------------------------------------------------


def _write_lock(memory_dir: Path, *, pid: int, mtime_seconds: float) -> Path:
    """Drop a lock file with a specific PID body + mtime."""
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    lock_path.write_text(str(pid), encoding="utf-8")
    os.utime(lock_path, (mtime_seconds, mtime_seconds))
    return lock_path


def test_get_holder_pid_returns_int_when_present(memory_dir: Path) -> None:
    _write_lock(memory_dir, pid=4242, mtime_seconds=time.time())
    assert lock_mod.get_holder_pid() == 4242


def test_get_holder_pid_returns_none_when_missing(memory_dir: Path) -> None:
    assert lock_mod.get_holder_pid() is None


def test_get_holder_pid_returns_none_for_corrupt_body(memory_dir: Path) -> None:
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    lock_path.write_text("not-a-pid", encoding="utf-8")
    assert lock_mod.get_holder_pid() is None


def test_get_holder_pid_returns_none_for_zero_or_negative(memory_dir: Path) -> None:
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    lock_path.write_text("0", encoding="utf-8")
    assert lock_mod.get_holder_pid() is None
    lock_path.write_text("-7", encoding="utf-8")
    assert lock_mod.get_holder_pid() is None


def test_get_lock_age_seconds_zero_when_missing(memory_dir: Path) -> None:
    assert lock_mod.get_lock_age_seconds() == 0


def test_get_lock_age_seconds_uses_mtime(memory_dir: Path) -> None:
    past = time.time() - 1234
    _write_lock(memory_dir, pid=os.getpid(), mtime_seconds=past)
    age = lock_mod.get_lock_age_seconds()
    assert 1230 <= age <= 1240


def test_is_lock_stale_false_when_missing(memory_dir: Path) -> None:
    assert lock_mod.is_lock_stale() is False


def test_is_lock_stale_false_when_fresh(memory_dir: Path) -> None:
    _write_lock(memory_dir, pid=os.getpid(), mtime_seconds=time.time())
    assert lock_mod.is_lock_stale() is False


def test_is_lock_stale_true_when_past_ttl(memory_dir: Path) -> None:
    stale_seconds = lock_mod.HOLDER_STALE_MS / 1000 + 60
    _write_lock(memory_dir, pid=os.getpid(), mtime_seconds=time.time() - stale_seconds)
    assert lock_mod.is_lock_stale() is True


def test_is_lock_stale_true_when_corrupt_body(memory_dir: Path) -> None:
    """Unparseable body is treated as stale — nothing valid to preserve."""
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    lock_path.write_text("garbage", encoding="utf-8")
    assert lock_mod.is_lock_stale() is True


def test_force_release_if_stale_no_op_when_missing(memory_dir: Path) -> None:
    assert lock_mod.force_release_if_stale() is False


def test_force_release_if_stale_no_op_when_fresh(memory_dir: Path) -> None:
    lock_path = _write_lock(memory_dir, pid=os.getpid(), mtime_seconds=time.time())
    assert lock_mod.force_release_if_stale() is False
    assert lock_path.exists()


def test_force_release_if_stale_unlinks_when_stale(memory_dir: Path) -> None:
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    stale_seconds = lock_mod.HOLDER_STALE_MS / 1000 + 120
    _write_lock(memory_dir, pid=os.getpid(), mtime_seconds=time.time() - stale_seconds)
    assert lock_mod.is_lock_stale() is True
    assert lock_mod.force_release_if_stale() is True
    assert not lock_path.exists()


def test_force_release_if_stale_swallows_oserror(monkeypatch, memory_dir: Path) -> None:
    """Even on filesystem failure, force_release_if_stale must not raise."""
    _write_lock(memory_dir, pid=os.getpid(), mtime_seconds=time.time())
    monkeypatch.setattr(lock_mod, "is_lock_stale", lambda now_ms=None: True)

    def _raise(*_a, **_kw):
        raise OSError("boom")

    monkeypatch.setattr(lock_mod.Path, "unlink", _raise)
    assert lock_mod.force_release_if_stale() is False


def test_try_acquire_reclaims_live_pid_when_ttl_expired(memory_dir: Path) -> None:
    """Phase B core — PID is alive but mtime age > TTL.

    Without Phase B the freshness gate would refuse to reclaim a
    lock whose holder is still alive. With Phase B the TTL is
    authoritative: reclaim happens regardless of holder liveness.
    """
    stale_seconds = lock_mod.HOLDER_STALE_MS / 1000 + 60
    _write_lock(memory_dir, pid=os.getpid(), mtime_seconds=time.time() - stale_seconds)
    result = lock_mod.try_acquire_consolidation_lock()
    assert result is not None
    lock_path = memory_dir / lock_mod.LOCK_FILE_NAME
    assert int(lock_path.read_text()) == os.getpid()


def test_try_acquire_still_blocks_when_fresh(memory_dir: Path) -> None:
    """Phase A behavior preserved: fresh + live-PID → blocked."""
    foreign_pid = os.getppid()
    assert foreign_pid != os.getpid(), "parent PID must differ from ours"
    _write_lock(memory_dir, pid=foreign_pid, mtime_seconds=time.time())
    assert lock_mod.try_acquire_consolidation_lock() is None
