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
# pylint: disable=E0402

"""Read-only integrity checks used by the LKB doctor."""

from __future__ import annotations

import time
from pathlib import Path

from .audit_compaction import validate_history_segment
from .json_store import validate_envelope_schema, verify_payload_hash
from .doctor_models import DoctorReport, FindingArea, FindingSeverity
from .doctor_support import (
    _backup_explains_invalid_primary,
    _primary_store_revision,
    _read_json_safe,
)


_LIFECYCLE_IN_PROGRESS_STATES = frozenset({"archiving", "purging"})
_LIFECYCLE_STABLE_STATES = frozenset({"active", "closed", "archived"})
_LIFECYCLE_TRASHED_STATE = "trashed"


def _inspect_primary(report: DoctorReport, paths: dict[str, Path]) -> None:
    """Check board.json: existence, JSON, schema, board ID, payload hash."""
    p = paths["board_json"]
    if not p.is_file():
        report.add(
            FindingSeverity.CRITICAL,
            FindingArea.PRIMARY,
            f"board.json is missing: {p}",
            auto_fixable=False,
        )
        return

    data = _read_json_safe(p)
    if data is None:
        report.add(
            FindingSeverity.CRITICAL,
            FindingArea.PRIMARY,
            f"board.json is not valid JSON: {p}",
            auto_fixable=False,
        )
        return

    # Schema / structure check.
    try:
        validate_envelope_schema(data, board_id=None)
    except Exception as exc:  # noqa: BLE001
        report.add(
            FindingSeverity.CRITICAL,
            FindingArea.SCHEMA,
            f"board.json fails schema validation: {exc}",
            auto_fixable=False,
        )
        return

    # board_id sanity check.
    envelope_bid = data.get("board", {}).get("board_id", "")
    if report.board_id and envelope_bid and envelope_bid != report.board_id:
        report.add(
            FindingSeverity.CRITICAL,
            FindingArea.PRIMARY,
            f"board_id mismatch: directory expects {report.board_id!r}, envelope has {envelope_bid!r}",
            auto_fixable=False,
        )
        return

    # If we don't know the board_id yet, take it from the envelope.
    if not report.board_id and envelope_bid:
        report.board_id = envelope_bid

    # Payload hash verification.
    if not verify_payload_hash(data):
        report.add(
            FindingSeverity.CRITICAL,
            FindingArea.PRIMARY,
            "board.json payload hash does not match content",
            auto_fixable=False,
        )
        return

    # Revision chain sanity: previousPayloadHash must match previous
    # revision if there is one — we can only check chain consistency if
    # we also have the .bak to compare against (done in _inspect_backup).
    report.add(
        FindingSeverity.INFO,
        FindingArea.PRIMARY,
        f"board.json is valid (store_revision={data.get('storeRevision', '?')})",
    )


def _inspect_backup(report: DoctorReport, paths: dict[str, Path]) -> None:
    """Check board.json.bak and its relationship to the primary."""
    bak = paths["board_json_bak"]
    if not bak.is_file():
        report.add(
            FindingSeverity.INFO,
            FindingArea.BACKUP,
            "board.json.bak does not exist (board has never been updated)",
        )
        return

    data = _read_json_safe(bak)
    if data is None:
        report.add(
            FindingSeverity.WARNING,
            FindingArea.BACKUP,
            "board.json.bak is not valid JSON",
            auto_fixable=False,
        )
        return

    try:
        validate_envelope_schema(data, board_id=report.board_id or None)
    except Exception as exc:  # noqa: BLE001
        report.add(
            FindingSeverity.WARNING,
            FindingArea.BACKUP,
            f"board.json.bak fails schema validation: {exc}",
            auto_fixable=False,
        )
        return

    if not verify_payload_hash(data):
        report.add(
            FindingSeverity.WARNING,
            FindingArea.BACKUP,
            "board.json.bak payload hash does not match content",
            auto_fixable=False,
        )
        return

    bak_bid = data.get("board", {}).get("board_id", "")
    if report.board_id and bak_bid and bak_bid != report.board_id:
        report.add(
            FindingSeverity.WARNING,
            FindingArea.BACKUP,
            f"board.json.bak belongs to different board {bak_bid!r} (expected {report.board_id!r})",
            auto_fixable=False,
        )
        return

    # Revision relationship: backup revision must be <= primary revision
    # (if primary is valid).  If primary is corrupt we note that backup
    # is usable for recovery.
    bak_rev = data.get("storeRevision", 0)
    primary_rev = _primary_store_revision(paths)

    if primary_rev is not None:
        if bak_rev <= primary_rev:
            report.add(
                FindingSeverity.INFO,
                FindingArea.BACKUP,
                f"board.json.bak is valid (store_revision={bak_rev}, primary={primary_rev}, explainable)",
            )
        else:
            report.add(
                FindingSeverity.WARNING,
                FindingArea.BACKUP,
                f"board.json.bak has store_revision={bak_rev} which is "
                f"newer than primary {primary_rev} — unexplainable",
                auto_fixable=False,
            )
    else:
        # Primary is missing / corrupt — backup looks recoverable.
        report.add(
            FindingSeverity.INFO,
            FindingArea.BACKUP,
            (
                f"board.json.bak is valid and its chain relationship permits recovery (store_revision={bak_rev})"
                if _backup_explains_invalid_primary(paths, data)
                else "board.json.bak is valid but cannot be proven to immediately "
                f"precede the invalid primary (store_revision={bak_rev})"
            ),
            auto_fixable=_backup_explains_invalid_primary(paths, data),
        )


def _inspect_lifecycle(report: DoctorReport, paths: dict[str, Path]) -> None:
    """Inspect lifecycle state for stuck mid-states (archiving, purging)."""
    p = paths["board_json"]
    data = _read_json_safe(p)
    if data is None:
        return  # primary already reported as bad

    lifecycle = data.get("lifecycle", {})
    if not isinstance(lifecycle, dict):
        report.add(
            FindingSeverity.ERROR,
            FindingArea.LIFECYCLE,
            "lifecycle field is not a dict",
            auto_fixable=False,
        )
        return

    state = lifecycle.get("state", "active")
    if state in _LIFECYCLE_IN_PROGRESS_STATES:
        report.add(
            FindingSeverity.WARNING,
            FindingArea.LIFECYCLE,
            f"Board is in mid-state '{state}' — a previous operation may have been interrupted",
            auto_fixable=True,
        )
    elif state == _LIFECYCLE_TRASHED_STATE:
        report.add(
            FindingSeverity.INFO,
            FindingArea.LIFECYCLE,
            "Board is in 'trashed' state (grace period before purge)",
        )
    elif state in _LIFECYCLE_STABLE_STATES:
        report.add(
            FindingSeverity.INFO,
            FindingArea.LIFECYCLE,
            f"Lifecycle state is '{state}' (stable)",
        )
    else:
        report.add(
            FindingSeverity.WARNING,
            FindingArea.LIFECYCLE,
            f"Unknown lifecycle state: {state!r}",
            auto_fixable=False,
        )


def _inspect_tmp_files(
    report: DoctorReport,
    paths: dict[str, Path],
    threshold_seconds: float,
) -> None:
    """Check .tmp/ directory for orphaned temp files."""
    tmp_dir = paths["tmp_dir"]
    if not tmp_dir.is_dir():
        return

    now = time.time()
    orphans: list[Path] = []
    recent: list[Path] = []
    try:
        entries = list(tmp_dir.iterdir())
    except OSError:
        return

    for entry in entries:
        if not entry.is_file():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        age = now - mtime
        if age > threshold_seconds:
            orphans.append(entry)
        else:
            recent.append(entry)

    if orphans:
        report.add(
            FindingSeverity.WARNING,
            FindingArea.TMP,
            f"Found {len(orphans)} orphaned .tmp/ file(s) older than {threshold_seconds / 3600:.1f}h",
            auto_fixable=True,
        )
    if recent:
        report.add(
            FindingSeverity.INFO,
            FindingArea.TMP,
            f"Found {len(recent)} recent .tmp/ file(s) (under age threshold)",
        )


def _inspect_history_segments(report: DoctorReport, paths: dict[str, Path]) -> None:
    """Check that history/ files referenced in historySegments actually exist,
    and that no orphan history files exist on disk.
    """
    hist_dir = paths["history_dir"]
    p = paths["board_json"]
    data = _read_json_safe(p)
    if data is None:
        return  # primary already reported as bad

    segments = data.get("historySegments", [])
    if not isinstance(segments, list):
        return

    referenced: set[str] = set()
    previous_hash = ""
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        fname = seg.get("file")
        if isinstance(fname, str) and fname and Path(fname).name == fname:
            referenced.add(fname)
        try:
            if seg.get("previousSegmentHash", "") != previous_hash:
                raise ValueError("history segment hash chain mismatch")
            validate_history_segment(hist_dir, seg)
            previous_hash = str(seg.get("sha256") or "")
        except (OSError, ValueError) as exc:
            report.add(
                FindingSeverity.ERROR,
                FindingArea.HISTORY,
                f"Invalid history segment {fname or '<unknown>'}: {exc}",
                auto_fixable=False,
            )

    # Check referenced files exist (kept separate for a concise summary).
    missing_refs: list[str] = []
    for fname in referenced:
        fpath = hist_dir / fname
        if not fpath.is_file():
            missing_refs.append(fname)

    if missing_refs:
        report.add(
            FindingSeverity.ERROR,
            FindingArea.HISTORY,
            f"{len(missing_refs)} referenced history segment(s) missing from disk: "
            f"{', '.join(missing_refs[:3])}{'...' if len(missing_refs) > 3 else ''}",
            auto_fixable=False,
        )

    # Check for orphan files on disk.
    if hist_dir.is_dir():
        try:
            disk_files = {e.name for e in hist_dir.iterdir() if e.is_file()}
        except OSError:
            disk_files = set()
        orphans = disk_files - referenced
        if orphans:
            report.add(
                FindingSeverity.WARNING,
                FindingArea.HISTORY,
                f"{len(orphans)} orphan history file(s) on disk (not referenced by historySegments)",
                auto_fixable=False,
            )


def _inspect_quarantine(
    report: DoctorReport,
    paths: dict[str, Path],
    threshold_seconds: float,
) -> None:
    """Report on quarantine directory contents and age."""
    qdir = paths["quarantine_dir"]
    if not qdir.is_dir():
        return

    try:
        entries = [e for e in qdir.iterdir() if e.is_file()]
    except OSError:
        return

    if not entries:
        return

    now = time.time()
    old_count = 0
    for e in entries:
        try:
            age = now - e.stat().st_mtime
            if age > threshold_seconds:
                old_count += 1
        except OSError:
            continue

    report.add(
        FindingSeverity.INFO,
        FindingArea.QUARANTINE,
        f"quarantine/ contains {len(entries)} file(s); {old_count} older than {threshold_seconds / 86400:.0f} days",
    )


def _inspect_lock_anchor(
    report: DoctorReport,
    paths: dict[str, Path],
    lock_acquired: bool,
) -> None:
    """Check .lock anchor file and .lock.owner.json diagnostics.

    Per LKB-LIFE-018: a .lock file with no OS lock is fine — it's a
    permanent anchor.  We never delete it.  A stale .lock.owner.json
    when no OS lock is held is purely informational.
    """
    lock_file = paths["lock_file"]
    owner_file = paths["lock_owner_json"]

    if not lock_file.is_file():
        # .lock anchor missing — informational, will be created on next write.
        report.add(
            FindingSeverity.INFO,
            FindingArea.LOCK,
            ".lock anchor file is missing (will be created on next write)",
        )

    if owner_file.is_file():
        owner_data = _read_json_safe(owner_file)
        if owner_data is None:
            report.add(
                FindingSeverity.INFO,
                FindingArea.LOCK,
                ".lock.owner.json exists but is unparseable (diagnostic only)",
            )
            return

        pid = owner_data.get("pid", "?")
        acquired_at = owner_data.get("acquired_at", 0)
        try:
            age_s = time.time() - float(acquired_at)
        except (TypeError, ValueError):
            age_s = 0.0

        if lock_acquired:
            # We hold the lock — owner file should reflect us.
            report.add(
                FindingSeverity.INFO,
                FindingArea.LOCK,
                f".lock.owner.json present (pid={pid}, age={age_s:.0f}s) — lock is held by this process",
            )
        else:
            # We could NOT acquire the lock, OR lock is available but
            # owner file is stale from a crashed process.
            report.add(
                FindingSeverity.INFO,
                FindingArea.LOCK,
                f".lock.owner.json is present (pid={pid}, age={age_s:.0f}s) — diagnostic only; lock anchor not removed",
            )
    else:
        report.add(
            FindingSeverity.INFO,
            FindingArea.LOCK,
            ".lock.owner.json is not present (no active lock holder)",
        )


def _inspect_tombstone(report: DoctorReport, paths: dict[str, Path]) -> None:
    """Look for tombstone files in the parent directory.

    Tombstones live outside the board directory (in a sibling
    ``tombstones/`` directory at the boards level).  We just check if any
    tombstone exists for this board and report its age.
    """
    # Board dir is .../lkb/boards/<safe-id>/
    # Tombs are at .../lkb/tombstones/<safe-id>.json
    boards_parent = report.board_dir.parent
    tomb_dir = boards_parent.parent / "tombstones"
    if not tomb_dir.is_dir():
        return

    safe_id = report.board_dir.name
    tomb_file = tomb_dir / f"{safe_id}.json"
    if not tomb_file.is_file():
        return

    try:
        age = time.time() - tomb_file.stat().st_mtime
    except OSError:
        age = 0.0

    report.add(
        FindingSeverity.INFO,
        FindingArea.TOMBSTONE,
        f"Tombstone exists for this board (age={age / 86400:.1f} days)",
    )
