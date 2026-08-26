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

# AgentSDK publishes this standalone src-layout package across independent Parts; the complete
# ClawCodex source and focused tests validate imports and dynamic patterns during migration.
# The target hook also enables legacy default diagnostics beyond its declared high-value set.
# pylint: disable=E0402

"""Safe repair operations used by the LKB doctor."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from .atomic_file import atomic_write_json
from .ir_hash import canonical_hash
from .json_store import (
    BoardEnvelope,
    _validate_envelope_schema,
    _verify_payload_hash,
    set_payload_hash,
)
from .doctor_models import DoctorReport, FindingArea, FindingSeverity
from .doctor_support import (
    _backup_explains_invalid_primary,
    _get_backup_validity,
    _iso_now,
    _read_json_safe,
)


def _do_safe_repairs(
    report: DoctorReport,
    paths: dict[str, Path],
    tmp_threshold_seconds: float,
) -> None:
    """Perform safe, data-preserving repairs.

    Order matters:
      1. Restore from backup (if primary is corrupt and backup is valid)
      2. Clean confirmed-orphan .tmp/ files
      3. Roll back stuck lifecycle mid-states (archiving → active)

    All repairs use proper locking (caller holds the lock) and follow
    atomic-write semantics where appropriate.
    """
    # 1. Backup recovery (primary corrupt + backup valid)
    _repair_restore_from_backup(report, paths)

    # 2. Orphan .tmp cleanup
    _repair_clean_tmp_orphans(report, paths, tmp_threshold_seconds)

    # 3. Stuck lifecycle mid-state rollback
    _repair_stuck_lifecycle(report, paths)


def _repair_restore_from_backup(report: DoctorReport, paths: dict[str, Path]) -> None:
    """If primary is corrupt and backup is valid, quarantine primary and
    install backup as primary.

    Only runs when:
      - primary is missing / corrupt (critical finding in PRIMARY area)
      - backup is valid and same board
      - backup revision is "explainable" (we accept any valid backup —
        if primary is corrupt we can't compare revisions, but spec §7.12
        says "explainable revision"; we at least verify the backup has
        a valid payload hash and same board_id)
    """
    # Only act if primary has a critical finding and backup is usable.
    primary_critical = any(
        f.area == FindingArea.PRIMARY and f.severity == FindingSeverity.CRITICAL for f in report.findings
    )
    backup_info = _get_backup_validity(paths)

    if not primary_critical or backup_info is None:
        return

    bak_data, bak_rev, bak_bid = backup_info

    # Same board check.
    if report.board_id and bak_bid and bak_bid != report.board_id:
        return
    if not _backup_explains_invalid_primary(paths, bak_data):
        report.add(
            FindingSeverity.WARNING,
            FindingArea.BACKUP,
            "Automatic recovery refused: invalid primary does not prove that "
            "board.json.bak is its immediate predecessor",
            auto_fixable=False,
        )
        return

    # Perform the recovery.
    board_json = paths["board_json"]
    qdir = paths["quarantine_dir"]

    try:
        qdir.mkdir(parents=True, exist_ok=True)
        # Quarantine a byte-for-byte specimen without creating a gap in the
        # authoritative path.
        if board_json.exists():
            stamp = time.time_ns()
            target = qdir / f"board.json.{stamp}.primary-corrupt"
            shutil.copy2(board_json, target)

        recovered = BoardEnvelope.from_dict(bak_data)
        previous_hash = str(recovered.integrity.get("payloadHash", ""))
        recovered.store_revision = bak_rev + 1
        recovered.board["store_revision"] = recovered.store_revision
        recovered.events.append(
            {
                "type": "store_recovered",
                "actor": "doctor",
                "reason": "primary invalid; restored from board.json.bak",
                "recovered_from_store_revision": bak_rev,
                "store_revision": recovered.store_revision,
            }
        )
        set_payload_hash(recovered, previous_hash=previous_hash)
        recovered_data = recovered.to_dict()
        _validate_envelope_schema(recovered_data, board_id=bak_bid)
        atomic_write_json(
            board_json,
            recovered_data,
            backup_path=None,
            payload_hash_key="payloadHash",
        )

        # Verify the restored file.
        restored = _read_json_safe(board_json)
        if restored is None or not _verify_payload_hash(restored):
            # Recovery didn't take — revert by re-copying from backup
            # (shouldn't happen, but be defensive).
            raise RuntimeError("restored board.json failed hash verification")

        report.add(
            FindingSeverity.WARNING,
            FindingArea.PRIMARY,
            f"Restored board.json from backup and recorded Recovery Event (store_revision={recovered.store_revision})",
            action_taken=f"restored from board.json.bak (rev {bak_rev}); user warning emitted",
        )
        report.board_state = "recovered"

    except Exception as exc:  # noqa: BLE001
        report.add(
            FindingSeverity.ERROR,
            FindingArea.PRIMARY,
            f"Failed to restore from backup: {exc}",
            action_taken="recovery attempt failed",
        )


def _repair_clean_tmp_orphans(
    report: DoctorReport,
    paths: dict[str, Path],
    threshold_seconds: float,
) -> None:
    """Delete .tmp/ files older than the threshold.

    Only safe because:
      - Temp files are never authoritative (spec §7.5).
      - We hold the board lock, so no in-progress write exists.
      - We only delete files older than the threshold, so we don't
        interfere with concurrent operations (though the lock already
        prevents those).
    """
    tmp_dir = paths["tmp_dir"]
    if not tmp_dir.is_dir():
        return

    now = time.time()
    deleted = 0
    failed = 0
    try:
        entries = list(tmp_dir.iterdir())
    except OSError:
        return

    for entry in entries:
        if not entry.is_file():
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if age > threshold_seconds:
            try:
                entry.unlink()
                deleted += 1
            except OSError:
                failed += 1

    if deleted:
        report.add(
            FindingSeverity.INFO,
            FindingArea.TMP,
            f"Cleaned {deleted} orphaned .tmp/ file(s) "
            f"(older than {threshold_seconds / 3600:.1f}h)" + (f", {failed} failed" if failed else ""),
            action_taken=f"deleted {deleted} orphan .tmp file(s)",
        )


def _repair_stuck_lifecycle(report: DoctorReport, paths: dict[str, Path]) -> None:
    """Roll back stuck ``archiving`` / ``purging`` mid-states.

    For ``archiving``: the archive file may or may not have been written
    to archives/.  The safe rollback is to return the board to its
    previous stable state (``closed``) — if an archive
    exists, it's an immutable snapshot and can be left there.

    For ``purging``: we do NOT auto-recover — purging is destructive and
    we don't know how far it got.  We just report it as stuck.
    """
    p = paths["board_json"]
    data = _read_json_safe(p)
    if data is None:
        return

    lifecycle = data.get("lifecycle", {})
    if not isinstance(lifecycle, dict):
        return

    state = lifecycle.get("state")
    if state == "archiving":
        # Safe rollback: return to closed, the legal predecessor of archiving.
        # We don't delete any partial archive file — it's harmless.
        try:
            _validate_envelope_schema(data, board_id=report.board_id or None)
            if not _verify_payload_hash(data):
                raise ValueError("stuck archiving primary has invalid payload hash")
            env = BoardEnvelope.from_dict(data)
            previous_hash = str(env.integrity.get("payloadHash", ""))
            previous_revision = env.store_revision
            operation = env.lifecycle.get("archive_operation")
            operation_id = str(operation.get("archive_id", "")) if isinstance(operation, dict) else ""
            env.lifecycle = dict(env.lifecycle)
            env.lifecycle["state"] = "closed"
            env.lifecycle["updated_at"] = _iso_now()
            env.lifecycle["archiving_interrupted"] = True
            env.lifecycle["archive_recovery"] = {
                "action": "rolled_back_to_closed",
                "operation_id": operation_id,
                "recovered_at": env.lifecycle["updated_at"],
            }
            env.store_revision = previous_revision + 1
            env.board["store_revision"] = env.store_revision
            env.events.append(
                {
                    "type": "lifecycle_recovered",
                    "actor": "doctor",
                    "from_state": "archiving",
                    "to_state": "closed",
                    "reason": "stuck archive operation rolled back to legal predecessor",
                    "archive_operation_id": operation_id,
                    "store_revision": env.store_revision,
                }
            )
            command_id = f"doctor:recover-archiving:{previous_revision}"
            env.processed_commands[command_id] = {
                "command_id": command_id,
                "request_hash": canonical_hash(
                    {
                        "kind": "doctor_recover_archiving",
                        "board_id": env.board_id(),
                        "source_store_revision": previous_revision,
                        "archive_operation_id": operation_id,
                    }
                ),
                "decision": "committed",
                "actor": "doctor",
                "store_revision": env.store_revision,
                "reason": "rolled back stuck archiving to closed",
                "derived_facts": [],
                "revision_vector": env.current_revision_vector().to_dict(),
            }
            set_payload_hash(env, previous_hash=previous_hash)
            final_data = env.to_dict()
            _validate_envelope_schema(final_data, board_id=env.board_id())

            # Atomic write.
            atomic_write_json(
                p,
                final_data,
                backup_path=paths["board_json_bak"],
                payload_hash_key="payloadHash",
            )

            report.add(
                FindingSeverity.INFO,
                FindingArea.LIFECYCLE,
                "Committed recovery of stuck 'archiving' state to 'closed'",
                action_taken=(
                    "created a new revision with Recovery/Lifecycle event and "
                    "reverted lifecycle.state from archiving to closed"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            report.add(
                FindingSeverity.ERROR,
                FindingArea.LIFECYCLE,
                f"Failed to roll back stuck archiving state: {exc}",
                action_taken="rollback attempt failed",
            )

    elif state == "purging":
        # We don't auto-recover purging — it's destructive and
        # we can't know how far it got.  Just report.
        report.add(
            FindingSeverity.WARNING,
            FindingArea.LIFECYCLE,
            "Board is stuck in 'purging' state — manual intervention required "
            "(auto-repair skipped: purge is destructive)",
            auto_fixable=False,
        )


# ── utility helpers ───────────────────────────────────────────────────
