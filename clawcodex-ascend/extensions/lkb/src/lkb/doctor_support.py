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
"""Shared target resolution and integrity helpers for the LKB doctor."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .board_resolver import board_dir
from .doctor_models import DoctorReport, FindingArea, FindingSeverity
from .json_store_models import validate_envelope_schema, verify_payload_hash

logger = logging.getLogger(__name__)


def _resolve_target(
    board_id_or_dir: str | Path,
    *,
    home: Path | None = None,
) -> tuple[Path, str, bool]:
    """Resolve the input into (board_dir, expected_board_id, from_path).

    *from_path* is True when the input was a filesystem path (rather than
    a board_id string).  The caller uses this to decide whether to
    derive sub-paths from the directory directly or via
    ``board_file_paths``.

    If the input looks like a path that exists, treat it as a board
    directory and derive the board_id by attempting to read board.json.
    Otherwise treat it as a board_id and resolve via board_dir().
    """
    p = Path(board_id_or_dir) if isinstance(board_id_or_dir, str) else board_id_or_dir

    # If it's an existing directory, treat as board dir.
    if p.is_dir():
        board_json = p / "board.json"
        bid = ""
        data = _read_json_safe(board_json)
        if data is not None:
            bid = str(data.get("board", {}).get("board_id", ""))
        return p, bid, True

    # If it's a Path object or a string with path separators, treat as
    # board directory (but report missing).
    if isinstance(board_id_or_dir, Path) or (
        isinstance(board_id_or_dir, str) and ("/" in board_id_or_dir or "\\" in board_id_or_dir)
    ):
        return p, "", True

    # Otherwise treat as board_id.
    bid = str(board_id_or_dir)
    return board_dir(bid, home=home), bid, False


def _path_map(bdir: Path) -> dict[str, Path]:
    """Build a paths dict for a board directory (without knowing board_id)."""
    return {
        "board_json": bdir / "board.json",
        "board_json_bak": bdir / "board.json.bak",
        "lock_file": bdir / ".lock",
        "lock_owner_json": bdir / ".lock.owner.json",
        "tmp_dir": bdir / ".tmp",
        "history_dir": bdir / "history",
        "quarantine_dir": bdir / "quarantine",
    }


def _read_json_safe(path: Path) -> dict[str, Any] | None:
    """Read and parse a JSON file, warning safely when it is unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("LKB doctor ignored unreadable JSON input (error_type=%s)", type(exc).__name__)
        return None
    if not isinstance(data, dict):
        return None
    return data


def _primary_store_revision(paths: dict[str, Path]) -> int | None:
    """Return storeRevision of primary, or None if unreadable."""
    data = _read_json_safe(paths["board_json"])
    if data is None:
        return None
    rev = data.get("storeRevision")
    if isinstance(rev, int):
        return rev
    return None


def _backup_explains_invalid_primary(
    paths: dict[str, Path],
    backup: dict[str, Any],
) -> bool:
    """Return whether raw primary metadata proves an immediate hash-chain link."""
    primary = _read_json_safe(paths["board_json"])
    if primary is None:
        return False
    primary_revision = primary.get("storeRevision")
    backup_revision = backup.get("storeRevision")
    primary_integrity = primary.get("integrity")
    backup_integrity = backup.get("integrity")
    primary_board = primary.get("board")
    backup_board = backup.get("board")
    if not all(isinstance(value, dict) for value in (primary_integrity, backup_integrity, primary_board, backup_board)):
        return False
    return (
        isinstance(primary_revision, int)
        and isinstance(backup_revision, int)
        and primary_revision == backup_revision + 1
        and primary_integrity.get("previousPayloadHash") == backup_integrity.get("payloadHash")
        and primary_board.get("board_id") == backup_board.get("board_id")
    )


def _primary_is_healthy(report: DoctorReport) -> bool:
    """True if no critical/error findings relate to primary/schema integrity."""
    for f in report.findings:
        if f.area in (FindingArea.PRIMARY, FindingArea.SCHEMA):
            if f.severity in (FindingSeverity.ERROR, FindingSeverity.CRITICAL):
                return False
    # Also must have at least one INFO finding for primary (meaning we
    # actually inspected it successfully).
    return any(f.area == FindingArea.PRIMARY and f.severity == FindingSeverity.INFO for f in report.findings)


def _primary_was_recovered(report: DoctorReport) -> bool:
    """True if a recovery action was taken."""
    return any(
        f.area == FindingArea.PRIMARY and f.action_taken and "restore" in f.action_taken.lower()
        for f in report.findings
    )


def _get_backup_validity(
    paths: dict[str, Path],
) -> tuple[dict[str, Any], int, str] | None:
    """If backup is valid, return (data, store_revision, board_id).
    Otherwise None.
    """
    bak = paths["board_json_bak"]
    if not bak.is_file():
        return None
    data = _read_json_safe(bak)
    if data is None:
        return None
    try:
        validate_envelope_schema(data, board_id=None)
    except Exception:
        return None
    if not verify_payload_hash(data):
        return None
    rev = data.get("storeRevision", 0)
    bid = data.get("board", {}).get("board_id", "")
    if not isinstance(rev, int):
        return None
    return data, rev, str(bid)


def _atomic_write_simple(target: Path, data: dict[str, Any]) -> None:
    """Minimal atomic JSON write for doctor repair operations.

    Uses a temp file + os.replace in the same directory.  No fsync on
    the directory (doctor is best-effort repair, not the hot path).
    """
    target = Path(target)
    tmp_path = target.with_suffix(target.suffix + ".doctor.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target)


def _iso_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
