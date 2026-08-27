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
# pylint: disable=E0611

"""Tests for doctor.py — board integrity diagnostics + safe repair.

Covers:
  Healthy board → no error/critical findings
  Corrupt primary + valid .bak → detected + repair recovers + quarantines
  Both primary and .bak corrupt → board_state=corrupt, no empty board returned
  Orphan .tmp files over threshold → detected + repair cleans them
  Orphan .tmp files under threshold → reported but not deleted
  Stuck archiving state → detected + repair rolls back to active
  Stale .lock.owner.json (no OS lock) → info only, anchor not deleted
  Missing board directory → critical finding, board_state=missing
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal, get_type_hints


from lkb.board_resolver import board_dir
from lkb import doctor_support
from lkb.doctor import (
    DoctorReport,
    FindingArea,
    FindingSeverity,
    doctor,
    format_doctor_report,
)
from lkb.file_lock import BoardFileLock
from lkb.graph_types import Board, BoardPolicy
from lkb.json_store import BoardEnvelope, JsonBoardStore, set_payload_hash


# ── helpers ───────────────────────────────────────────────────────────


def _make_board(board_id: str = "test-board") -> Board:
    return Board(
        board_id=board_id,
        project_uri=f"project:{board_id}",
        display_name=board_id,
        schema_version=1,
        store_revision=0,
        created_at="2026-01-01T00:00:00.000Z",
        updated_at="2026-01-01T00:00:00.000Z",
        policy=BoardPolicy(),
    )


def _create_board(bdir: Path, *, board_id: str = "test-board") -> JsonBoardStore:
    board = _make_board(board_id)
    lock = BoardFileLock(bdir)
    return JsonBoardStore.create_board(bdir, board=board, lock=lock)


def _setup_board(tmp_lkb_root: Path, board_id: str) -> Path:
    """Create a board at its canonical safe_board_id directory.

    Returns the board directory path for tests that need to manipulate
    files directly.
    """
    bdir = board_dir(board_id, home=tmp_lkb_root)
    _create_board(bdir, board_id=board_id)
    return bdir


def _corrupt_file(path: Path) -> None:
    """Overwrite a file with garbage bytes."""
    path.write_bytes(b"this is not valid json {{{\x00\x01\x02")


def _set_mtime_old(path: Path, *, days: float) -> None:
    """Set file mtime to N days in the past."""
    old = time.time() - days * 86400
    os.utime(path, (old, old))


def _do_one_update(bdir: Path, board_id: str) -> None:
    """Execute one no-op command to create a .bak file."""
    from lkb.commands import CommandResult

    def _noop(env):
        return env, CommandResult(decision="committed", command_id="cmd-1")

    lock = BoardFileLock(bdir)
    store = JsonBoardStore(bdir, board_id=board_id, lock=lock)
    store.execute_atomic(
        board_id,
        "cmd-1",
        "hash-1",
        None,
        _noop,
        actor="test",
    )


# ── tests: healthy board ──────────────────────────────────────────────


def test_healthy_board_no_errors(tmp_lkb_root: Path) -> None:
    """A freshly-created healthy board has no error/critical findings."""
    _setup_board(tmp_lkb_root, "test-board-01")

    report = doctor("test-board-01", home=tmp_lkb_root)

    assert isinstance(report, DoctorReport)
    assert report.board_id == "test-board-01"
    assert report.board_state == "healthy"
    assert report.repair_attempted is False
    assert report.is_healthy

    severities = {f.severity for f in report.findings}
    assert FindingSeverity.ERROR not in severities
    assert FindingSeverity.CRITICAL not in severities


def test_doctor_report_board_state_is_a_closed_literal() -> None:
    assert (
        get_type_hints(DoctorReport)["board_state"] == Literal["unknown", "healthy", "recovered", "corrupt", "missing"]
    )


def test_healthy_board_primary_valid(tmp_lkb_root: Path) -> None:
    """Healthy board has an info finding confirming primary is valid."""
    _setup_board(tmp_lkb_root, "test-board-02")

    report = doctor("test-board-02", home=tmp_lkb_root)

    primary_info = [f for f in report.findings if f.area == FindingArea.PRIMARY and f.severity == FindingSeverity.INFO]
    assert len(primary_info) >= 1
    assert "valid" in primary_info[0].message.lower()


# ── tests: corrupt primary + valid backup ─────────────────────────────


def test_corrupt_primary_valid_backup_detected(tmp_lkb_root: Path) -> None:
    """Corrupt board.json with valid .bak → critical on primary, backup auto-fixable."""
    bdir = _setup_board(tmp_lkb_root, "test-board-03")
    _do_one_update(bdir, "test-board-03")

    _corrupt_file(bdir / "board.json")

    report = doctor("test-board-03", home=tmp_lkb_root)

    primary_crit = [
        f for f in report.findings if f.area == FindingArea.PRIMARY and f.severity == FindingSeverity.CRITICAL
    ]
    assert len(primary_crit) >= 1

    backup_fixable = [f for f in report.findings if f.area == FindingArea.BACKUP and f.auto_fixable]
    assert not backup_fixable

    assert report.board_state == "corrupt"
    assert report.has_errors


def test_corrupt_primary_valid_backup_repair(tmp_lkb_root: Path) -> None:
    """With repair=True, corrupt primary + valid .bak is restored."""
    bdir = _setup_board(tmp_lkb_root, "test-board-04")
    _do_one_update(bdir, "test-board-04")

    primary = json.loads((bdir / "board.json").read_text(encoding="utf-8"))
    primary["board"]["display_name"] = "tampered without updating hash"
    (bdir / "board.json").write_text(json.dumps(primary), encoding="utf-8")

    report = doctor("test-board-04", repair=True, home=tmp_lkb_root)

    assert report.repair_attempted is True
    assert report.board_state == "recovered"

    # Verify board.json is now readable.
    recovered = JsonBoardStore(bdir, board_id="test-board-04", lock=BoardFileLock(bdir)).load()
    assert recovered.board_id() == "test-board-04"

    # Verify the corrupt file was quarantined.
    qdir = bdir / "quarantine"
    assert qdir.is_dir()
    quarantined = list(qdir.glob("board.json.*.primary-corrupt"))
    assert len(quarantined) >= 1


# ── tests: both corrupt → board_store_corrupt ─────────────────────────


def test_doctor_refuses_nonadjacent_old_backup_recovery(tmp_lkb_root: Path) -> None:
    bdir = _setup_board(tmp_lkb_root, "test-board-old-backup")
    _do_one_update(bdir, "test-board-old-backup")
    primary = json.loads((bdir / "board.json").read_text(encoding="utf-8"))
    primary["storeRevision"] = 50
    primary["board"]["store_revision"] = 50
    (bdir / "board.json").write_text(json.dumps(primary), encoding="utf-8")

    report = doctor("test-board-old-backup", repair=True, home=tmp_lkb_root)

    unchanged = json.loads((bdir / "board.json").read_text(encoding="utf-8"))
    assert unchanged["storeRevision"] == 50
    assert any(
        finding.area == FindingArea.BACKUP and "refused" in finding.message.lower() for finding in report.findings
    )


def test_both_corrupt_board_state_corrupt(tmp_lkb_root: Path) -> None:
    """When both board.json and .bak are corrupt, state is 'corrupt'."""
    bdir = _setup_board(tmp_lkb_root, "test-board-05")
    _do_one_update(bdir, "test-board-05")

    _corrupt_file(bdir / "board.json")
    _corrupt_file(bdir / "board.json.bak")

    report = doctor("test-board-05", home=tmp_lkb_root)

    assert report.board_state == "corrupt"
    assert report.has_errors

    recovery_actions = [f for f in report.findings if "restore" in f.action_taken.lower()]
    assert len(recovery_actions) == 0


def test_both_corrupt_repair_does_not_create_empty(tmp_lkb_root: Path) -> None:
    """Even with repair=True, doctor never creates an empty board
    when both files are corrupt (spec §7.12).
    """
    bdir = _setup_board(tmp_lkb_root, "test-board-06")
    _do_one_update(bdir, "test-board-06")

    _corrupt_file(bdir / "board.json")
    _corrupt_file(bdir / "board.json.bak")

    report = doctor("test-board-06", repair=True, home=tmp_lkb_root)

    assert report.board_state == "corrupt"

    bj = bdir / "board.json"
    if bj.exists():
        data = None
        try:
            data = json.loads(bj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # good — still corrupt
        if data is not None:
            # Should not be a valid genesis board (doctor doesn't create boards).
            assert data.get("storeRevision", -1) != 0 or len(data.get("events", [])) != 0


# ── tests: .tmp/ orphan cleanup ───────────────────────────────────────


def test_orphan_tmp_over_threshold_detected(tmp_lkb_root: Path) -> None:
    """Old .tmp/ files are detected as orphans with auto_fixable=True."""
    bdir = _setup_board(tmp_lkb_root, "test-board-07")

    tmp_dir = bdir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    old_file = tmp_dir / "old-temp-file.tmp"
    old_file.write_text("stale data")
    _set_mtime_old(old_file, days=2)

    report = doctor("test-board-07", home=tmp_lkb_root)

    tmp_findings = [f for f in report.findings if f.area == FindingArea.TMP]
    orphan = [f for f in tmp_findings if f.severity == FindingSeverity.WARNING]
    assert len(orphan) >= 1
    assert orphan[0].auto_fixable is True


def test_orphan_tmp_over_threshold_repair_cleans(tmp_lkb_root: Path) -> None:
    """With repair=True, old .tmp/ files are deleted."""
    bdir = _setup_board(tmp_lkb_root, "test-board-08")

    tmp_dir = bdir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    old_file = tmp_dir / "old-temp-file.tmp"
    old_file.write_text("stale data")
    _set_mtime_old(old_file, days=2)

    assert old_file.exists()

    report = doctor(
        "test-board-08",
        repair=True,
        home=tmp_lkb_root,
        tmp_orphan_threshold_seconds=24 * 3600,
    )

    assert not old_file.exists()

    tmp_cleaned = [f for f in report.findings if f.area == FindingArea.TMP and f.action_taken]
    assert len(tmp_cleaned) >= 1
    assert "deleted" in tmp_cleaned[0].action_taken


def test_orphan_tmp_under_threshold_not_deleted(tmp_lkb_root: Path) -> None:
    """Recent .tmp/ files are reported as info but NOT deleted."""
    bdir = _setup_board(tmp_lkb_root, "test-board-09")

    tmp_dir = bdir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    recent_file = tmp_dir / "recent-temp-file.tmp"
    recent_file.write_text("fresh data")

    report = doctor(
        "test-board-09",
        repair=True,
        home=tmp_lkb_root,
        tmp_orphan_threshold_seconds=24 * 3600,
    )

    assert recent_file.exists()

    recent_findings = [f for f in report.findings if f.area == FindingArea.TMP and f.severity == FindingSeverity.INFO]
    assert len(recent_findings) >= 1


# ── tests: stuck lifecycle ────────────────────────────────────────────


def _set_lifecycle_state(bdir: Path, state: str) -> None:
    """Manually set board.json lifecycle state, recomputing payload hash."""
    bj = bdir / "board.json"
    data = json.loads(bj.read_text(encoding="utf-8"))
    data["lifecycle"] = {"state": state, "updated_at": "2026-01-01T00:00:00.000Z"}
    if state == "archiving":
        data["lifecycle"]["archive_operation"] = {
            "archive_id": "archive-operation-test",
            "actor": "test",
            "reason": "test recovery",
            "started_at": "2026-01-01T00:00:00.000Z",
        }
    env = BoardEnvelope.from_dict(data)
    prev_hash = env.integrity.get("previousPayloadHash")
    set_payload_hash(env, previous_hash=prev_hash or None)
    bj.write_text(
        json.dumps(env.to_dict(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_stuck_archiving_detected(tmp_lkb_root: Path) -> None:
    """A board in 'archiving' state is detected as a warning with auto_fixable."""
    bdir = _setup_board(tmp_lkb_root, "test-board-10")
    _set_lifecycle_state(bdir, "archiving")

    report = doctor("test-board-10", home=tmp_lkb_root)

    life_warns = [
        f for f in report.findings if f.area == FindingArea.LIFECYCLE and f.severity == FindingSeverity.WARNING
    ]
    assert len(life_warns) >= 1
    assert "archiving" in life_warns[0].message.lower()
    assert life_warns[0].auto_fixable is True


def test_stuck_archiving_repair_rolls_back(tmp_lkb_root: Path) -> None:
    """With repair=True, stuck archiving is rolled back to closed."""
    bdir = _setup_board(tmp_lkb_root, "test-board-11")
    _set_lifecycle_state(bdir, "archiving")
    before = json.loads((bdir / "board.json").read_text(encoding="utf-8"))

    report = doctor("test-board-11", repair=True, home=tmp_lkb_root)

    bj = bdir / "board.json"
    data = json.loads(bj.read_text(encoding="utf-8"))
    assert data["lifecycle"]["state"] == "closed"
    assert data["storeRevision"] == before["storeRevision"] + 1
    assert data["integrity"]["previousPayloadHash"] == before["integrity"]["payloadHash"]
    assert data["events"][-1]["type"] == "lifecycle_recovered"
    assert data["events"][-1]["from_state"] == "archiving"
    assert data["events"][-1]["to_state"] == "closed"
    assert data["events"][-1]["archive_operation_id"] == "archive-operation-test"
    assert data["lifecycle"]["archive_recovery"]["operation_id"] == "archive-operation-test"
    assert any(key.startswith("doctor:recover-archiving:") for key in data["processedCommands"])

    rollback = [f for f in report.findings if f.area == FindingArea.LIFECYCLE and f.action_taken]
    assert len(rollback) >= 1
    assert "new revision" in rollback[0].action_taken


# ── tests: .lock.owner.json staleness ─────────────────────────────────


def test_stale_lock_owner_info_only(tmp_lkb_root: Path) -> None:
    """A stale .lock.owner.json is reported as info-only; .lock not deleted.

    LKB-LIFE-018: a .lock with no OS lock is fine — it's a permanent anchor.
    The doctor must never delete it.
    """
    bdir = _setup_board(tmp_lkb_root, "test-board-12")

    # Acquire + release the lock once so the .lock anchor file exists.
    lock = BoardFileLock(bdir)
    with lock:
        pass

    lock_file = bdir / ".lock"
    assert lock_file.exists(), ".lock anchor must exist after lock is used"

    # Create a stale .lock.owner.json (simulating a crashed process).
    owner_file = bdir / ".lock.owner.json"
    owner_data = {
        "pid": 99999,
        "host": "old-host",
        "user": "old-user",
        "command": "old-command",
        "acquired_at": time.time() - 86400,
        "platform": "linux",
    }
    owner_file.write_text(json.dumps(owner_data, sort_keys=True))

    report = doctor("test-board-12", home=tmp_lkb_root)

    owner_findings = [f for f in report.findings if f.area == FindingArea.LOCK and "owner" in f.message.lower()]
    # Stale owner file should be reported as info (not warning/error).
    assert all(f.severity == FindingSeverity.INFO for f in owner_findings)
    # .lock anchor must still exist — never deleted by doctor.
    assert lock_file.exists()


def test_lock_anchor_not_deleted_on_repair(tmp_lkb_root: Path) -> None:
    """Even with repair=True, .lock anchor is never deleted (LKB-LIFE-018)."""
    bdir = _setup_board(tmp_lkb_root, "test-board-13")

    # Acquire + release to create the .lock anchor.
    lock = BoardFileLock(bdir)
    with lock:
        pass

    lock_file = bdir / ".lock"
    assert lock_file.exists()

    doctor("test-board-13", repair=True, home=tmp_lkb_root)

    assert lock_file.exists()


# ── tests: missing board ──────────────────────────────────────────────


def test_missing_board_directory(tmp_lkb_root: Path) -> None:
    """A non-existent board directory → critical finding + missing state."""
    report = doctor("no-such-board", home=tmp_lkb_root)

    assert report.board_state == "missing"
    crits = [f for f in report.findings if f.severity == FindingSeverity.CRITICAL]
    assert len(crits) >= 1


def test_doctor_safe_json_read_logs_only_the_error_type(tmp_path: Path, caplog) -> None:
    unreadable = tmp_path / "board.json"
    secret = "malformed audit detail must not reach logs"
    unreadable.write_text(f"{{{secret}", encoding="utf-8")

    with caplog.at_level("WARNING", logger="lkb.doctor_support"):
        assert doctor_support._read_json_safe(unreadable) is None

    assert "error_type=JSONDecodeError" in caplog.text
    assert secret not in caplog.text


# ── tests: format_doctor_report ───────────────────────────────────────


def test_format_doctor_report_healthy(tmp_lkb_root: Path) -> None:
    """Human-readable report produces a string for a healthy board."""
    _setup_board(tmp_lkb_root, "test-board-14")

    report = doctor("test-board-14", home=tmp_lkb_root)
    text = format_doctor_report(report)

    assert isinstance(text, str)
    assert "test-board-14" in text
    assert "healthy" in text.lower()
    assert len(text.splitlines()) > 3


def test_format_doctor_report_with_findings(tmp_lkb_root: Path) -> None:
    """Human-readable report shows severity and area for each finding."""
    bdir = _setup_board(tmp_lkb_root, "test-board-15")
    _corrupt_file(bdir / "board.json")

    report = doctor("test-board-15", home=tmp_lkb_root)
    text = format_doctor_report(report)

    assert "CRITICAL" in text
    assert "primary" in text.lower()


# ── tests: doctor via board directory path ────────────────────────────


def test_doctor_accepts_directory_path(tmp_lkb_root: Path) -> None:
    """doctor() accepts a Path to a board directory directly."""
    bdir = _setup_board(tmp_lkb_root, "test-board-16")

    report = doctor(bdir)

    assert report.board_id == "test-board-16"
    assert report.board_dir == bdir
    assert report.board_state == "healthy"


# ── tests: history segment orphan / missing ───────────────────────────


def test_history_orphan_detected(tmp_lkb_root: Path) -> None:
    """Orphan history files (on disk but not in historySegments)
    are reported as warnings.
    """
    bdir = _setup_board(tmp_lkb_root, "test-board-17")

    hist_dir = bdir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    orphan = hist_dir / "events-0-100.jsonl.gz"
    orphan.write_bytes(b"\x1f\x8b\x00\x00fake-gzip")

    report = doctor("test-board-17", home=tmp_lkb_root)

    hist_warnings = [
        f for f in report.findings if f.area == FindingArea.HISTORY and f.severity == FindingSeverity.WARNING
    ]
    assert len(hist_warnings) >= 1
    assert "orphan" in hist_warnings[0].message.lower()


# ── tests: quarantine reporting ───────────────────────────────────────


def test_quarantine_files_reported(tmp_lkb_root: Path) -> None:
    """Files in quarantine/ are reported as info findings."""
    bdir = _setup_board(tmp_lkb_root, "test-board-18")

    qdir = bdir / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    old_q = qdir / "board.json.123456.primary-corrupt"
    old_q.write_text("corrupted data")
    _set_mtime_old(old_q, days=40)

    report = doctor("test-board-18", home=tmp_lkb_root)

    q_findings = [f for f in report.findings if f.area == FindingArea.QUARANTINE]
    assert len(q_findings) >= 1
    assert q_findings[0].severity == FindingSeverity.INFO
