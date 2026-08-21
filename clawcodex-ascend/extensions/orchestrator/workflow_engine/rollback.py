#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=relative-beyond-top-level

"""Stage rollback manager.

Stage snapshots + versioned rollback for GATE rejection / DECISION loop exhaustion.
"""

from __future__ import annotations

import logging
import shutil
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

from .errors import RollbackError
from .workflow_state import StageNode, StageStatus, WorkflowState

logger = logging.getLogger(__name__)


# -- Stage snapshots --───────────────────────────────────────────────────────


@dataclass
class StageSnapshot:
    """Snapshot of a single stage."""

    stage_id: int
    stage_name: str
    snapshot_dir: str
    timestamp: str
    files: list[str] = field(default_factory=list)
    size_bytes: int = 0


@dataclass
class RollbackTarget:
    """Rollback target."""

    stage_id: int
    stage_name: str
    reason: str
    snapshot: StageSnapshot | None = None


# -- Rollback manager --─────────────────────────────────────────────────────


class RollbackManager:
    """Stage rollback manager: workspace snapshots + versioned rollback."""

    def __init__(
        self,
        workspace_dir: str | Path,
        max_snapshots: int = 10,
    ) -> None:
        self._workspace_dir = Path(workspace_dir)
        self._snapshots_dir = self._workspace_dir / ".workflow_snapshots"
        self._max_snapshots = max_snapshots
        self._snapshots: dict[int, StageSnapshot] = {}

    def snapshot_dir(self, stage_id: int) -> Path:
        """Get the snapshot directory for a stage."""
        return self._snapshots_dir / f"stage_{stage_id:03d}"

    def save_snapshot(self, stage: StageNode) -> StageSnapshot | None:
        """Save a snapshot before the stage executes.

        Copy workspace files into the snapshot directory.
        Only save files that are not under .workflow_snapshots.
        """
        if not self._workspace_dir.exists():
            logger.debug("Workspace dir %s does not exist, skipping snapshot", self._workspace_dir)
            return None

        dest_dir = self.snapshot_dir(stage.id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        files_copied: list[str] = []
        total_size = 0

        try:
            for item in self._workspace_dir.iterdir():
                if item.name == ".workflow_snapshots":
                    continue
                if item.name.startswith("."):
                    continue

                dest = dest_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest, ignore_errors=True)
                    try:
                        shutil.copytree(
                            item,
                            dest,
                            symlinks=True,
                            dirs_exist_ok=True,
                            ignore=_snapshot_ignore,
                        )
                    except shutil.Error as copy_err:
                        # Some files (sockets, FIFOs) can't be copied;
                        # log and continue with whatever was copied
                        logger.debug("Partial copy for %s: %s", item.name, copy_err)
                    files_copied.append(item.name)
                elif item.is_file():
                    # Skip non-regular files (sockets, FIFOs, device files)
                    if not _is_regular_file(item):
                        logger.debug("Skipping non-regular file: %s", item)
                        continue
                    shutil.copy2(item, dest)
                    total_size += item.stat().st_size
                    files_copied.append(item.name)
                else:
                    logger.debug("Skipping special file: %s", item)
        except Exception as exc:
            logger.warning("Failed to save snapshot for stage %s: %s", stage.id, exc)
            return None

        snapshot = StageSnapshot(
            stage_id=stage.id,
            stage_name=stage.name,
            snapshot_dir=str(dest_dir),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            files=files_copied,
            size_bytes=total_size,
        )

        self._snapshots[stage.id] = snapshot
        self._prune_old_snapshots()
        logger.info(
            "Snapshot saved for stage %s (%s files, %s bytes)",
            stage.id,
            len(files_copied),
            total_size,
        )

        return snapshot

    def restore_snapshot(self, stage_id: int) -> bool:
        """Restore the snapshot of a target stage.

        Clear current workspace files and restore from the snapshot.
        """
        snapshot = self._snapshots.get(stage_id)
        if snapshot is None:
            # Try loading from disk
            snapshot = self._load_snapshot_from_disk(stage_id)

        if snapshot is None:
            logger.warning("No snapshot found for stage %s", stage_id)
            return False

        src_dir = Path(snapshot.snapshot_dir)
        if not src_dir.exists():
            logger.warning("Snapshot dir %s does not exist", src_dir)
            return False

        try:
            # Clear current workspace (keep the snapshot dir)
            for item in self._workspace_dir.iterdir():
                if item.name == ".workflow_snapshots":
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)

            # Restore from snapshot
            for item in src_dir.iterdir():
                dest = self._workspace_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, symlinks=True, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            logger.info("Restored snapshot for stage %s", stage_id)
            return True
        except Exception as exc:
            logger.error("Failed to restore snapshot for stage %s: %s", stage_id, exc)
            raise RollbackError(
                f"Failed to restore snapshot for stage {stage_id}: {exc}",
                stage_id=stage_id,
            ) from exc

    def get_snapshot(self, stage_id: int) -> StageSnapshot | None:
        """Get snapshot info for a stage."""
        return self._snapshots.get(stage_id) or self._load_snapshot_from_disk(stage_id)

    def has_snapshot(self, stage_id: int) -> bool:
        """Check whether a stage snapshot exists."""
        return self.get_snapshot(stage_id) is not None

    def resolve_rollback_target(
        self,
        stage: StageNode,
        gate_rollback_to: int | None = None,
        decision_rollback_to: int | None = None,
    ) -> RollbackTarget:
        """Resolve the Rollback target.

        Priority:
        1. Explicitly specified rollback_to
        2. First dependency of the stage
        3. Most recent completed stage that has a snapshot
        """
        target_id = gate_rollback_to or decision_rollback_to

        if target_id is not None:
            return RollbackTarget(
                stage_id=target_id,
                stage_name=f"stage-{target_id}",
                reason="Explicit rollback target",
                snapshot=self.get_snapshot(target_id),
            )

        # First dependency stage
        if stage.depends_on:
            target_id = stage.depends_on[0]
            return RollbackTarget(
                stage_id=target_id,
                stage_name=f"stage-{target_id}",
                reason="Rollback to first dependency",
                snapshot=self.get_snapshot(target_id),
            )

        raise RollbackError(
            f"No rollback target found for stage {stage.id}",
            stage_id=stage.id,
        )

    def update_state_on_rollback(
        self,
        state: WorkflowState,
        target_id: int,
        failed_stage_id: int,
    ) -> None:
        """Update WorkflowState to reflect a rollback.

        Mark all stages after target_id as pending re-execution.
        """
        # Remove all completed stages after target_id
        removed = []
        for sid in list(state.completed_stages):
            if sid > target_id:
                state.completed_stages.remove(sid)
                removed.append(sid)

        # Reset these stages' status
        for sid in removed:
            state.stage_statuses[sid] = StageStatus.PENDING
            if sid in state.stage_results:
                del state.stage_results[sid]

        # Mark rollback
        state.add_rollback_event(
            from_stage=failed_stage_id,
            to_stage=target_id,
        )

        logger.info("State rolled back: removed stages %s, target=%s", removed, target_id)

    def cleanup(self) -> None:
        """Clean up all snapshots."""
        if self._snapshots_dir.exists():
            shutil.rmtree(self._snapshots_dir, ignore_errors=True)
        self._snapshots.clear()

    # -- Internal methods --─────────────────────────────────────────────────

    def _load_snapshot_from_disk(self, stage_id: int) -> StageSnapshot | None:
        """Load a snapshot from disk (for restore scenarios)."""
        snap_dir = self.snapshot_dir(stage_id)
        if not snap_dir.exists():
            return None

        try:
            files = []
            total_size = 0
            for item in snap_dir.iterdir():
                files.append(item.name)
                if item.is_file():
                    total_size += item.stat().st_size

            return StageSnapshot(
                stage_id=stage_id,
                stage_name=f"stage-{stage_id}",
                snapshot_dir=str(snap_dir),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(snap_dir.stat().st_mtime)),
                files=files,
                size_bytes=total_size,
            )
        except Exception:
            return None

    def _prune_old_snapshots(self) -> None:
        """Prune old snapshots, keeping the most recent max_snapshots."""
        if len(self._snapshots) <= self._max_snapshots:
            return

        sorted_ids = sorted(self._snapshots.keys())
        to_remove = sorted_ids[: len(sorted_ids) - self._max_snapshots]

        for sid in to_remove:
            snap_dir = self.snapshot_dir(sid)
            if snap_dir.exists():
                shutil.rmtree(snap_dir, ignore_errors=True)
            del self._snapshots[sid]
            logger.debug("Pruned old snapshot for stage %s", sid)


# -- Module-level utilities --─────────────────────────────────────────────────


def _snapshot_ignore(directory: str, files: list[str]) -> set[str]:
    """shutil.copytree ignore callback: skip files/dirs that do not need copying."""
    ignored: set[str] = set()
    for f in files:
        full = Path(directory) / f
        if full.name.startswith("."):
            ignored.add(f)
        elif not _is_regular_file(full):
            ignored.add(f)
    return ignored


def _is_regular_file(path: Path) -> bool:
    """Check whether the path is a regular file (excludes sockets, FIFOs, etc.)."""
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except (OSError, FileNotFoundError):
        return False
