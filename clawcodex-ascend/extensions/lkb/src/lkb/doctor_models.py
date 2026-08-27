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
# pylint: disable=R1714
"""Board integrity doctor — diagnostics + safe repair (spec §7.12, §7.10).

The doctor inspects a board directory for problems and reports them as a
structured ``DoctorReport``.  When ``repair=True`` it performs only *safe*
fixes — things that cannot destroy user data:

  * Quarantine a corrupt ``board.json`` and restore a valid ``board.json.bak``
    (only when the backup is for the same board, has a valid payload hash,
    and has an explainable revision relationship).
  * Clean up confirmed-orphan ``.tmp/`` files older than the configured
    threshold (only when the board lock is held).
  * Resume stuck ``archiving`` / ``purging`` mid-states by rolling back to
    the last stable state (only when it can be done safely, without losing
    committed data).

It **never** auto-deletes:
  * Project boards
  * History segments
  * Archives
  * Exports

It also **never** breaks a live OS lock — ``.lock.owner.json`` staleness is
purely informational (LKB-STORE-020 / LKB-LIFE-018).

Spec §7.12 — corruption detection, backup, and recovery
Spec §7.10 — close / archive / restore / purge
Spec §7.9 — file lifecycle table

This module imports nothing from ToolContext or Task-v2 (spec §11.4 inv 12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "DoctorFinding",
    "DoctorReport",
    "FindingArea",
    "FindingSeverity",
    "DEFAULT_TMP_ORPHAN_THRESHOLD_SECONDS",
    "DEFAULT_QUARANTINE_THRESHOLD_SECONDS",
]

# ── default thresholds (spec §7.9) ────────────────────────────────────

DEFAULT_TMP_ORPHAN_THRESHOLD_SECONDS = 24 * 3600  # 24 hours
DEFAULT_QUARANTINE_THRESHOLD_SECONDS = 30 * 24 * 3600  # 30 days


# ── severity / area enums ─────────────────────────────────────────────


class FindingSeverity(str, Enum):
    """Severity of a doctor finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class FindingArea(str, Enum):
    """Broad area of the board storage that a finding relates to."""

    PRIMARY = "primary"  # board.json
    BACKUP = "backup"  # board.json.bak
    LIFECYCLE = "lifecycle"  # archiving / purging / trashed / etc.
    TMP = "tmp"  # .tmp/ orphan files
    HISTORY = "history"  # history segments
    QUARANTINE = "quarantine"  # quarantine directory / age
    LOCK = "lock"  # .lock / .lock.owner.json
    TOMBSTONE = "tombstone"  # tombstone files / age
    SCHEMA = "schema"  # schema version / migration


DoctorBoardState = Literal["unknown", "healthy", "recovered", "corrupt", "missing"]


# ── finding / report data classes ────────────────────────────────────


@dataclass
class DoctorFinding:
    """A single finding from a doctor inspection."""

    severity: FindingSeverity
    area: FindingArea
    message: str
    auto_fixable: bool = False
    action_taken: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "area": self.area.value,
            "message": self.message,
            "auto_fixable": self.auto_fixable,
            "action_taken": self.action_taken,
        }


@dataclass
class DoctorReport:
    """Result of a doctor inspection (and optional repair) of a board."""

    board_id: str
    board_dir: Path
    findings: list[DoctorFinding] = field(default_factory=list)
    repair_attempted: bool = False
    board_state: DoctorBoardState = "unknown"

    # ── convenience helpers ──────────────────────────────────────────

    @property
    def is_healthy(self) -> bool:
        return all(
            f.severity != FindingSeverity.CRITICAL and f.severity != FindingSeverity.ERROR for f in self.findings
        )

    @property
    def has_errors(self) -> bool:
        return any(f.severity in (FindingSeverity.ERROR, FindingSeverity.CRITICAL) for f in self.findings)

    def add(
        self,
        severity: FindingSeverity,
        area: FindingArea,
        message: str,
        *,
        auto_fixable: bool = False,
        action_taken: str = "",
    ) -> DoctorFinding:
        f = DoctorFinding(
            severity=severity,
            area=area,
            message=message,
            auto_fixable=auto_fixable,
            action_taken=action_taken,
        )
        self.findings.append(f)
        return f

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_id": self.board_id,
            "board_dir": str(self.board_dir),
            "board_state": self.board_state,
            "repair_attempted": self.repair_attempted,
            "findings": [f.to_dict() for f in self.findings],
        }


# ── public API ────────────────────────────────────────────────────────
