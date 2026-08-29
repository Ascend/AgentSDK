#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Consolidation lock (with Phase B TTL enhancement).

Mirrors ``typescript/src/services/autoDream/consolidationLock.ts``.
The lock file lives in the auto-memory dir and its mtime *is*
``lastConsolidatedAt``. Body is the holder's PID. Two reclaimers
both write the same PID — last write wins; loser bails on re-read.

**Phase B (TTL 30min) — 2026-07-07**

The original Phase A implementation only counted a holder as
"alive" when (a) the PID is still running *and* (b) the mtime is
within :data:`HOLDER_STALE_MS`. The Phase B enhancement goes
further — it treats the mtime TTL as authoritative even when
the PID is alive. This closes the PID-reuse race:

* A consolidator crashes without unlinking the lock. Linux may
  recycle the PID within minutes. A second consolidator then
  sees the same PID still alive, and the old Phase A code
  refuses to reclaim — so a stale 30min+ old mtime can keep
  blocking new consolidations forever.
* Phase B lets the second consolidator forcibly reclaim the
  stale lock once mtime age ≥ :data:`HOLDER_STALE_MS`,
  regardless of holder PID liveness. The act of reclaiming
  also rewrites the mtime to *now*, which doubles as the
  ``lastConsolidatedAt`` stamp for the time gate (rolling
  forward, since we did not run the consolidation).

New APIs (all O(1) stat reads, no extra I/O):

* :func:`get_holder_pid` — lock body PID (or ``None``).
* :func:`get_lock_age_seconds` — seconds since last stamp (``0``
  = no lock file).
* :func:`is_lock_stale` — age ≥ :data:`HOLDER_STALE_MS`
  (or lock file missing → ``False``; unparseable body → ``True``).
* :func:`force_release_if_stale` — unlink stale lock; return
  whether something was actually released.

The change is **conservative** — lock files younger than
TTL still gate on PID liveness exactly as before. Reclaim of
live-PID locks only happens past TTL.

Operations:

* :func:`read_last_consolidated_at` — one stat, returns mtime ms.
* :func:`try_acquire_consolidation_lock` — write PID, return prior
  mtime for rollback or ``None`` if blocked. **TTL-aware**: stale
  holders are reclaimed even if PID is still alive.
* :func:`rollback_consolidation_lock` — rewind mtime to prior
  (or unlink if prior was 0).
* :func:`record_consolidation` — optimistic stamp for manual /dream
  (no contention check).
* :func:`list_sessions_touched_since` — session ids with mtime
  after *since_ms* (used by the session gate).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

from clawcodex_ext.dreaming.paths import get_auto_mem_path, project_transcript_dir

_log = logging.getLogger(__name__)

LOCK_FILE_NAME = ".consolidate-lock"

# Stale past this even if PID is live (PID-reuse guard). Phase B
# promotes the TTL from "implicit reclaim-only" to the authoritative
# freshness gate in :func:`is_lock_stale`. 30 minutes matches the
# design doc; long enough to cover a slow LLM consolidation, short
# enough that PID recycling won't strand the lock for long.
HOLDER_STALE_MS = 30 * 60 * 1000


def _lock_path() -> Path:
    return Path(get_auto_mem_path()) / LOCK_FILE_NAME


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_last_consolidated_at() -> int:
    """mtime of the lock file = lastConsolidatedAt. 0 if absent.

    One stat per call. Used by the time gate.
    """
    try:
        return int(_lock_path().stat().st_mtime * 1000)
    except FileNotFoundError:
        return 0
    except OSError as e:  # pragma: no cover - defensive
        _log.debug("read_last_consolidated_at failed: %s", e)
        return 0


# ---------------------------------------------------------------------------
# Acquire
# ---------------------------------------------------------------------------


def try_acquire_consolidation_lock() -> int | None:
    """Acquire: write PID → mtime = now. Return the pre-acquire mtime
    (for rollback), or ``None`` if blocked / lost a race.

    * Success → mtime stays at now.
    * Failure → :func:`rollback_consolidation_lock` rewinds mtime.
    * Crash → mtime stuck, dead PID → next process reclaims.
    * **Self-acquisition** — if the lock is already held by our own
      PID (e.g. :func:`record_consolidation` was called optimistically
      by the manual ``/dream`` path), return the pre-acquire mtime
      without re-writing. The PID is the same, the intent is the
      same, and blocking here would make manual /dream a no-op.
    * **Phase B TTL reclaim** — if the lock is older than
      :data:`HOLDER_STALE_MS` (30min by default), reclaim it
      regardless of whether the holder PID is still alive. This
      closes the PID-reuse race: without this, a 30min+ stale
      lock held by a recycled PID would block forever.
    """
    path = _lock_path()
    mtime_ms: int | None = None
    holder_pid: int | None = None

    if path.exists():
        try:
            st = path.stat()
            mtime_ms = int(st.st_mtime * 1000)
        except OSError:
            mtime_ms = None
        try:
            raw = path.read_text(encoding="utf-8")
            parsed = int(raw.strip())
            holder_pid = parsed if parsed > 0 else None
        except (OSError, ValueError):
            holder_pid = None

    # Self-acquisition short-circuit — see docstring.
    if holder_pid == os.getpid():
        return mtime_ms or 0

    # Fresh + live holder → blocked (Phase A behavior, preserved).
    if mtime_ms is not None and _now_ms() - mtime_ms < HOLDER_STALE_MS:
        if holder_pid is not None and _pid_is_alive(holder_pid):
            _log.debug(
                "consolidation lock held by live PID %d (mtime %ds ago)",
                holder_pid,
                (_now_ms() - mtime_ms) // 1000,
            )
            return None
        # Fresh-but-dead-PID — reclaim below (dead PID would still
        # alive-check as False here, so we naturally fall through).
    # else: Phase B — stale TTL is authoritative; reclaim even if
    # the holder PID is still alive (PID-reuse race close).

    # Ensure the memory dir exists before writing.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:  # pragma: no cover - defensive
        _log.debug("mkdir for consolidation lock failed: %s", e)
        return None

    try:
        path.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as e:  # pragma: no cover - defensive
        _log.debug("write consolidation lock failed: %s", e)
        return None

    # Two reclaimers both write → last wins the PID. Loser bails on re-read.
    try:
        verify = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if int(verify) != os.getpid():
        return None

    return mtime_ms or 0


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def rollback_consolidation_lock(prior_mtime: int) -> None:
    """Rewind mtime to *prior_mtime* after a failed fork.

    prior_mtime == 0 → unlink (restore "no-file" state). Else write
    empty body + utimes to set mtime. Best-effort — never raises.
    """
    path = _lock_path()
    try:
        if prior_mtime <= 0:
            path.unlink(missing_ok=True)
            return
        path.write_text("", encoding="utf-8")
        seconds = prior_mtime / 1000.0
        os.utime(path, (seconds, seconds))
    except OSError as e:
        _log.debug(
            "consolidation lock rollback failed: %s — next trigger delayed to minHours",
            e,
        )


# ---------------------------------------------------------------------------
# Record (manual /dream optimistic stamp)
# ---------------------------------------------------------------------------


def record_consolidation() -> None:
    """Stamp from manual ``/dream``. Optimistic — fires at
    prompt-build time, no post-skill completion hook. Best-effort.
    """
    path = _lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as e:  # pragma: no cover - defensive
        _log.debug("record_consolidation write failed: %s", e)


# ---------------------------------------------------------------------------
# Phase B — TTL 30min diagnostics & active cleanup
# ---------------------------------------------------------------------------


def get_holder_pid() -> int | None:
    """Return the PID recorded in the lock body, or ``None``.

    No mtime / liveness check — purely a stat + parse. ``0`` and
    negative values (legacy / corrupted body) normalize to
    ``None``. Missing file → ``None``.
    """
    path = _lock_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as e:  # pragma: no cover - defensive
        _log.debug("get_holder_pid read failed: %s", e)
        return None
    try:
        pid = int(raw.strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def get_lock_age_seconds(now_ms: int | None = None) -> int:
    """Seconds since the lock was last stamped. ``0`` when absent.

    A write to the lock file (acquire / record / rollback) updates
    mtime; this reads back the same mtime used by the time gate.
    Treats missing-file as ``0`` rather than "infinitely old" —
    callers that need fresh-only behavior should check
    :func:`is_lock_stale` directly.
    """
    try:
        mtime_ms = read_last_consolidated_at()
    except Exception:  # pragma: no cover - defensive
        return 0
    if mtime_ms <= 0:
        return 0
    if now_ms is None:
        now_ms = _now_ms()
    return max(0, (now_ms - mtime_ms) // 1000)


def is_lock_stale(now_ms: int | None = None) -> bool:
    """Whether the current lock has aged past :data:`HOLDER_STALE_MS`.

    Returns ``False`` if no lock file exists, ``True`` if the lock
    is missing *or* the body is unparseable *or* the age window
    has elapsed. Used by :func:`force_release_if_stale` to decide
    whether unlinking is safe.

    This is the Phase B enhancement — the underlying TTL value
    itself did not change, but it now drives an authoritative
    freshness signal (instead of only being one of two gates
    inside :func:`try_acquire_consolidation_lock`).
    """
    if now_ms is None:
        now_ms = _now_ms()
    path = _lock_path()
    if not path.exists():
        return False
    # Treat empty / unparseable body as stale — there's nothing
    # valid to preserve.
    if get_holder_pid() is None:
        return True
    last_ms = read_last_consolidated_at()
    if last_ms <= 0:
        return True
    return (now_ms - last_ms) >= HOLDER_STALE_MS


def force_release_if_stale(now_ms: int | None = None) -> bool:
    """Best-effort unlink if :func:`is_lock_stale` reports stale.

    Returns ``True`` only when a file was actually removed (or
    found missing-but-not-stale). Caller-safe — never raises.
    Designed to be invoked once per service tick before the Lock
    gate so a process that crashed last week never strands the
    lock file forever.

    * Missing file → ``False`` (nothing to release).
    * Stale lock → unlink, return ``True``.
    * Fresh lock → leave it alone, return ``False``.

    This complements — does not replace — the in-band reclaim
    inside :func:`try_acquire_consolidation_lock`. The two
    checks together mean: a fresh lock is blocked by PID, a
    stale lock is reclaimed regardless of PID, and explicitly
    calling :func:`force_release_if_stale` lets the next
    consolidator run even if it never reaches the acquire
    branch (e.g. gates above blocked it).
    """
    path = _lock_path()
    try:
        if not path.exists():
            return False
        if not is_lock_stale(now_ms=now_ms):
            return False
        path.unlink(missing_ok=True)
        _log.info(
            "force_release_if_stale: unlinked stale consolidation lock (age %ds, TTL %ds)",
            get_lock_age_seconds(now_ms=now_ms),
            HOLDER_STALE_MS // 1000,
        )
        return True
    except OSError as e:  # pragma: no cover - defensive
        _log.debug("force_release_if_stale failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Session scan
# ---------------------------------------------------------------------------


def list_sessions_touched_since(since_ms: int) -> list[str]:
    """Session ids in the project transcript dir with mtime after
    *since_ms*.

    The upstream implementation filters by UUID shape and skips
    ``agent-*.jsonl`` files. clawcodex's session storage has a
    different on-disk layout; we filter by suffix and by parseable
    filename instead. Returns ids (not paths) so the caller can
    exclude the current session by id.

    Scans **only** the per-cwd :func:`project_transcript_dir` — never
    falls back to the global session dir. Falling back to the global
    dir would let unrelated projects inflate the session count for
    this dream run.
    """
    try:
        base = Path(project_transcript_dir())
    except Exception as e:  # pragma: no cover - defensive
        _log.debug("project_transcript_dir failed: %s", e)
        return []
    if not base.is_dir():
        return []

    results: list[str] = []
    for child in _iter_candidates(base):
        try:
            if int(child.stat().st_mtime * 1000) > since_ms:
                results.append(child.name)
        except OSError:
            continue
    return results


def _iter_candidates(base: Path) -> Iterable[Path]:
    """Yield candidate session files under *base*.

    Upstream: ``listCandidates(dir, true)`` filters to UUID-shaped
    filenames and excludes ``agent-*.jsonl``. clawcodex session files
    live as ``<session_id>/...`` directories; we treat each
    subdirectory (or ``.jsonl`` file) as a candidate and let the
    caller filter by mtime.
    """
    try:
        for child in sorted(base.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir() or child.suffix == ".jsonl":
                yield child
    except OSError:
        pass  # Best-effort operation failed; keep the surrounding fallback.


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time_ms())


def time_ms() -> int:
    """Return current wall time in milliseconds. Indirection for tests."""
    import time as _t

    return int(_t.time() * 1000)
