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

"""Board integrity diagnostics and safe repair facade."""

from __future__ import annotations

from pathlib import Path

from .board_resolver import board_file_paths
from .file_lock import BoardFileLock, BoardStoreBusyError
from .doctor_checks import (
    _inspect_backup,
    _inspect_history_segments,
    _inspect_lifecycle,
    _inspect_lock_anchor,
    _inspect_primary,
    _inspect_quarantine,
    _inspect_tmp_files,
    _inspect_tombstone,
)
from .doctor_models import (
    DEFAULT_QUARANTINE_THRESHOLD_SECONDS,
    DEFAULT_TMP_ORPHAN_THRESHOLD_SECONDS,
    DoctorFinding,
    DoctorReport,
    FindingArea,
    FindingSeverity,
)
from .doctor_repairs import _do_safe_repairs
from .doctor_support import (
    _path_map,
    _primary_is_healthy,
    _primary_was_recovered,
    _resolve_target,
)

__all__ = [
    "DoctorFinding",
    "DoctorReport",
    "FindingArea",
    "FindingSeverity",
    "doctor",
    "format_doctor_report",
    "DEFAULT_TMP_ORPHAN_THRESHOLD_SECONDS",
    "DEFAULT_QUARANTINE_THRESHOLD_SECONDS",
]


def doctor(
    board_id_or_dir: str | Path,
    *,
    repair: bool = False,
    home: Path | None = None,
    tmp_orphan_threshold_seconds: float = DEFAULT_TMP_ORPHAN_THRESHOLD_SECONDS,
    quarantine_threshold_seconds: float = DEFAULT_QUARANTINE_THRESHOLD_SECONDS,
) -> DoctorReport:
    """Inspect a board directory for integrity / lifecycle problems.

    Parameters
    ----------
    board_id_or_dir:
        Either a ``board_id`` string (resolved via ``board_dir``) or a
        ``Path`` to a board directory.  When a path is given, the
        ``board_id`` is read from the on-disk envelope (when possible).
    repair:
        If True, perform *safe* automatic repairs.  Default False
        (diagnostics only).
    home:
        Optional home directory override (only used when *board_id_or_dir*
        is a board_id string rather than a path).
    tmp_orphan_threshold_seconds:
        How old a ``.tmp/`` file must be before it is considered a
        confirmed orphan eligible for cleanup.  Default 24 hours.
    quarantine_threshold_seconds:
        How old files in ``quarantine/`` must be before the doctor
        reports them as aging.  Default 30 days.

    Returns
    -------
    DoctorReport
        Structured findings.  Does not raise for common error conditions —
        they are reported as findings instead.  Only raises for truly
        unexpected programming errors.
    """
    # Resolve input to a directory + tentative board_id.
    bdir, expected_board_id, from_path = _resolve_target(board_id_or_dir, home=home)
    report = DoctorReport(board_id=expected_board_id, board_dir=bdir)

    if not bdir.is_dir():
        report.board_state = "missing"
        report.add(
            FindingSeverity.CRITICAL,
            FindingArea.PRIMARY,
            f"Board directory does not exist: {bdir}",
        )
        return report

    # When input was a directory path, always derive paths from that
    # directory directly — don't re-resolve via board_id (which would
    # use the default home and might point to a different location).
    if from_path:
        paths = _path_map(bdir)
    else:
        paths = board_file_paths(expected_board_id, home=home)

    # Try to acquire the board lock.  If we can't (busy), we still do a
    # read-only inspection but we won't perform any repairs.
    lock = BoardFileLock(bdir, timeout=2.0)
    lock_acquired = False
    try:
        lock.acquire()
        lock_acquired = True
    except BoardStoreBusyError:
        report.add(
            FindingSeverity.WARNING,
            FindingArea.LOCK,
            "Board lock is currently held by another process; skipping locked-only checks and all repairs",
        )
    except Exception as exc:  # noqa: BLE001 — defensive
        report.add(
            FindingSeverity.ERROR,
            FindingArea.LOCK,
            f"Failed to acquire board lock: {exc}",
        )

    try:
        _inspect_primary(report, paths)
        _inspect_backup(report, paths)
        _inspect_lifecycle(report, paths)
        _inspect_tmp_files(report, paths, tmp_orphan_threshold_seconds)
        _inspect_history_segments(report, paths)
        _inspect_quarantine(report, paths, quarantine_threshold_seconds)
        _inspect_lock_anchor(report, paths, lock_acquired)
        _inspect_tombstone(report, paths)

        # Determine overall state from findings so far.
        if _primary_is_healthy(report):
            report.board_state = "healthy"
        elif _primary_was_recovered(report):
            report.board_state = "recovered"
        else:
            report.board_state = "corrupt"

        # ── repair phase (only if lock is held and repair=True) ─────
        if repair and lock_acquired:
            report.repair_attempted = True
            _do_safe_repairs(report, paths, tmp_orphan_threshold_seconds)

    finally:
        if lock_acquired:
            try:
                lock.release()
            except Exception:  # nosec B110 - Diagnostic cleanup must preserve the primary finding.
                pass

    return report


# ── human-readable report ─────────────────────────────────────────────


def format_doctor_report(report: DoctorReport) -> str:
    """Format a DoctorReport as a human-readable multi-line string."""
    lines: list[str] = []
    lines.append(f"Board:   {report.board_id}")
    lines.append(f"Path:    {report.board_dir}")
    lines.append(f"State:   {report.board_state}")
    lines.append(f"Repair:  {'attempted' if report.repair_attempted else 'not attempted'}")
    lines.append(f"Findings: {len(report.findings)}")
    lines.append("")
    if not report.findings:
        lines.append("  (no findings — board is clean)")
        return "\n".join(lines)
    for i, f in enumerate(report.findings, 1):
        sev = f.severity.value.upper()
        tag = f"[{sev}] {f.area.value}"
        lines.append(f"  {i:2d}. {tag}: {f.message}")
        if f.auto_fixable and not f.action_taken:
            lines.append("      → auto-fixable")
        if f.action_taken:
            lines.append(f"      → action: {f.action_taken}")
    return "\n".join(lines)
