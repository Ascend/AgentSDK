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

# Pytest loads this namespace package in the complete ordered migration state.
# pylint: disable=relative-beyond-top-level
"""Tests for lifecycle.py + migrations package.

Lifecycle (spec §7.8 – §7.11):
  LKB-LIFE-001 — lifecycle state defaults to "active" on genesis boards
  LKB-LIFE-002 — valid forward transitions succeed (active -> closed -> archived -> trashed -> purged)
  LKB-LIFE-005 — invalid transitions are rejected (e.g. active -> archived)
  LKB-LIFE-006 — reopen restores a closed board to active
  LKB-LIFE-007 — archive creates an archive copy with hash + revision
  LKB-LIFE-008 — restore returns board to active with source archive reference
  LKB-LIFE-009 — purge removes payload data but leaves tombstone
  LKB-LIFE-010 — gc_scan dry-run never modifies the filesystem
  LKB-LIFE-011 — gc_scan finds old temp files as candidates
  LKB-LIFE-016 — close with active claims rejected unless override+reason
  LKB-LIFE-017 — lifecycle transitions are persisted via execute_atomic

Migrations (spec §7.13):
  LKB-STORE-008 — v0 -> v1 migration upgrades schemaVersion and recomputes hash
  LKB-STORE-009 — migration is idempotent (already-v1 envelope is unchanged)
  LKB-STORE-025 — forward-compat: schema newer than code raises BoardSchemaTooNewError
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lkb.commands import CommandResult
from .._support import Failpoint
from lkb.board_resolver import safe_board_id
from lkb.file_lock import BoardFileLock
from lkb.graph_types import Board, BoardPolicy
from lkb.json_store import (
    BoardEnvelope,
    BoardTombstonedError,
    CURRENT_SCHEMA_VERSION,
    JsonBoardStore,
    STORE_FORMAT,
    set_payload_hash,
)
from lkb.lifecycle import (
    GC_SESSION_ORPHAN_AGE_SECONDS,
    GC_TEMP_AGE_SECONDS,
    LifecycleError,
    LifecycleTransitionDenied,
    archive_board,
    board_lifecycle_state,
    close_board,
    gc_apply,
    gc_scan,
    genesis_lifecycle,
    ordinary_write_allowed,
    ordinary_write_denial_reason,
    purge_board,
    read_archive,
    read_tombstone,
    restore_board,
    trash_board,
    transition,
)
from lkb.repository import ArchiveRef
from lkb.migrations import (
    MigrationError,
    migrate_board_file,
)
from lkb.refs import NodeRef


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


def _create_store(
    board_dir: Path,
    *,
    board_id: str = "test-board",
    home: Path | None = None,
) -> JsonBoardStore:
    board = _make_board(board_id)
    lock = BoardFileLock(board_dir)
    return JsonBoardStore.create_board(
        board_dir,
        board=board,
        lock=lock,
        lkb_root=home,
    )


def _create_session_store(root: Path, board_id: str) -> JsonBoardStore:
    board_dir = root / "boards" / safe_board_id(board_id)
    board = Board(
        board_id=board_id,
        project_uri=f"session:{board_id}",
        display_name=board_id,
        schema_version=1,
        store_revision=0,
        created_at="2026-01-01T00:00:00.000Z",
        updated_at="2026-01-01T00:00:00.000Z",
        policy=BoardPolicy(),
    )
    return JsonBoardStore.create_board(
        board_dir,
        board=board,
        lock=BoardFileLock(board_dir),
        lkb_root=root,
    )


def _make_envelope(
    board_id: str = "test-board",
    *,
    state: str = "active",
) -> BoardEnvelope:
    """Build a minimal BoardEnvelope for lifecycle testing."""
    env = BoardEnvelope(
        store_format=STORE_FORMAT,
        schema_version=CURRENT_SCHEMA_VERSION,
        store_revision=0,
        board={
            "board_id": board_id,
            "project_uri": f"project:{board_id}",
            "display_name": board_id,
            "schema_version": 1,
            "store_revision": 0,
            "policy": {},
        },
        lifecycle={
            "state": state,
            "scope": "project",
            "created_at": "2026-01-01T00:00:00.000Z",
            "updated_at": "2026-01-01T00:00:00.000Z",
        },
    )
    set_payload_hash(env)
    return env


def _add_active_claim(env: BoardEnvelope, *, claim_id: str = "c1") -> None:
    """Add an active claim to *env* (mutates in place)."""
    task_ref = NodeRef("plan", "task", "T-001")
    owner_ref = NodeRef("plan", "agent", "agent-1")
    env.claims[claim_id] = {
        "claim_id": claim_id,
        "task_ref": task_ref.to_str(),
        "owner_ref": owner_ref.to_str(),
        "status": "active",
        "claimed_at": "2026-01-01T00:00:00.000Z",
        "claim_revision": 1,
    }


def _cmd_id(prefix: str) -> str:
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _archive_ref(store: JsonBoardStore) -> ArchiveRef:
    envelope = store.load()
    info = envelope.lifecycle["archive_info"]
    document = read_archive(Path(info["archive_path"]), expected_board_id=envelope.board_id())
    return ArchiveRef(
        board_id=envelope.board_id(),
        archive_path=Path(info["archive_path"]),
        store_revision=int(document["sourceStoreRevision"]),
        payload_hash=str(document["payloadHash"]),
    )


# ── LKB-LIFE-001 ──────────────────────────────────────────────────────


class TestPhase2LifecycleBoundaries:
    def test_genesis_lifecycle_has_every_required_field(self) -> None:
        lifecycle = genesis_lifecycle(
            scope="project",
            created_at="2026-01-01T00:00:00Z",
            origin_project_uri="project:/repo",
        )
        assert set(lifecycle) == {
            "state",
            "scope",
            "created_at",
            "updated_at",
            "closed_at",
            "archived_at",
            "retention_policy",
            "origin_project_uri",
        }

    @pytest.mark.parametrize(
        ("state", "allowed"),
        [
            ("active", True),
            ("closed", False),
            ("archiving", False),
            ("archived", False),
            ("trashed", False),
            ("purging", False),
        ],
    )
    def test_ordinary_write_gate_is_pure(self, state: str, allowed: bool) -> None:
        env = _make_envelope(state=state)
        before = env.to_dict()
        assert ordinary_write_allowed(env) is allowed
        assert (ordinary_write_denial_reason(env) is None) is allowed
        assert env.to_dict() == before

    def test_archive_document_links_hash_revision_and_source(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)
        close_board(store, "test-board", actor="u", command_id="close", request_hash="hc")
        archive_board(store, "test-board", actor="u", command_id="archive", request_hash="ha")
        archived = store.load()
        info = archived.lifecycle["archive_info"]
        archive_path = Path(info["archive_path"])
        document = json.loads(archive_path.read_text(encoding="utf-8"))
        assert document["boardId"] == "test-board"
        assert document["sourceStoreRevision"] == info["source_store_revision"]
        assert document["payloadHash"] == info["archive_hash"]
        assert archive_path.is_file()

    def test_corrupt_archive_refuses_restore_and_archive_is_retained(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)
        close_board(store, "test-board", actor="u", command_id="close", request_hash="hc")
        archive_board(store, "test-board", actor="u", command_id="archive", request_hash="ha")
        archive_ref = _archive_ref(store)
        archive_path = Path(store.load().lifecycle["archive_info"]["archive_path"])
        document = json.loads(archive_path.read_text(encoding="utf-8"))
        document["sourceStoreRevision"] += 1
        archive_path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(LifecycleError, match="archive payload hash mismatch"):
            restore_board(
                store,
                "test-board",
                archive_ref=archive_ref,
                actor="u",
                command_id="restore",
                request_hash="hr",
            )
        assert archive_path.is_file()
        assert board_lifecycle_state(store.load()) == "archived"

    def test_purge_requires_permission(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)
        close_board(store, "test-board", actor="u", command_id="close", request_hash="hc")
        trash_board(store, "test-board", actor="u", command_id="trash", request_hash="ht")
        with pytest.raises(PermissionError):
            purge_board(
                store,
                "test-board",
                actor="u",
                command_id="purge",
                request_hash="hp",
                reason="cleanup",
                confirm="test-board",
                authorized=False,
            )

    def test_active_claim_blocks_archive_and_purge(self) -> None:
        closed = _make_envelope(state="closed")
        _add_active_claim(closed)
        with pytest.raises(LifecycleTransitionDenied):
            transition(closed, "archiving", actor="u")
        trashed = _make_envelope(state="trashed")
        _add_active_claim(trashed)
        with pytest.raises(LifecycleTransitionDenied):
            transition(trashed, "purging", actor="u")


class TestPhase2GcBoundaries:
    def test_expired_session_orphan_is_discovered_and_deleted_when_requested(self, tmp_lkb_root: Path) -> None:
        store = _create_session_store(tmp_lkb_root, "session-board")
        board_dir = Path(store._board_dir)
        board_json = board_dir / "board.json"
        now = 4_000_000.0
        old = now - GC_SESSION_ORPHAN_AGE_SECONDS - 1
        os.utime(board_json, (old, old))
        dry_run = gc_scan(tmp_lkb_root, dry_run=True, now=now)
        assert any(item.kind == "session_orphan" for item in dry_run)
        assert board_json.is_file()
        gc_scan(tmp_lkb_root, dry_run=False, now=now)
        assert not board_json.exists()
        assert (board_dir / ".lock").is_file()

    def test_non_dry_run_deletes_only_named_atomic_temp(self, tmp_lkb_root: Path) -> None:
        tmp_dir = tmp_lkb_root / "boards" / "board-a" / ".tmp"
        tmp_dir.mkdir(parents=True)
        orphan = tmp_dir / ".board.json.abc.tmp"
        suspicious = tmp_dir / "notes.txt"
        orphan.write_text("{}", encoding="utf-8")
        suspicious.write_text("keep", encoding="utf-8")
        now = 2_000_000.0
        old = now - GC_TEMP_AGE_SECONDS - 1
        os.utime(orphan, (old, old))
        os.utime(suspicious, (old, old))
        gc_scan(tmp_lkb_root, dry_run=False, now=now)
        assert not orphan.exists()
        assert not suspicious.exists()
        assert (tmp_dir.parent / "quarantine" / "notes.txt").is_file()
        assert (tmp_dir.parent / ".lock").is_file()

    def test_symlink_is_reported_and_never_followed(self, tmp_lkb_root: Path) -> None:
        target = tmp_lkb_root / "outside"
        target.mkdir()
        marker = target / "marker"
        marker.write_text("keep", encoding="utf-8")
        boards = tmp_lkb_root / "boards"
        boards.mkdir()
        link = boards / "linked-board"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("symlink unavailable")
        candidates = gc_scan(tmp_lkb_root, dry_run=False, now=2_000_000.0)
        assert any(item.kind == "unsafe_path" and item.path == link for item in candidates)
        assert marker.is_file()

    def test_expired_quarantine_is_report_only(self, tmp_lkb_root: Path) -> None:
        quarantine = tmp_lkb_root / "boards" / "b" / "quarantine"
        quarantine.mkdir(parents=True)
        item = quarantine / "candidate"
        item.write_text("keep", encoding="utf-8")
        now = 3_000_000.0
        old = now - 31 * 24 * 3600
        os.utime(item, (old, old))
        candidates = gc_scan(tmp_lkb_root, dry_run=False, now=now)
        candidate = next(value for value in candidates if value.path == item)
        assert candidate.action == "report"
        assert item.is_file()


class TestPhase2MigrationOrchestrator:
    def test_v0_file_migrates_atomically_and_preserves_unknown_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "board.json"
        original = {
            "board": {
                "board_id": "b",
                "project_uri": "project:/repo",
                "custom": {"retained": True},
            },
            "events": [{"legacy": 1}],
            "unknownTopLevel": {"x": 2},
        }
        path.write_text(json.dumps(original), encoding="utf-8")
        outcome = migrate_board_file(path, expected_board_id="b")
        migrated = json.loads(path.read_text(encoding="utf-8"))
        assert migrated["schemaVersion"] == CURRENT_SCHEMA_VERSION
        assert migrated["board"]["custom"] == {"retained": True}
        assert migrated["board"]["compatibility_metadata"]["legacy_top_level"]["unknownTopLevel"] == {"x": 2}
        assert outcome.backup_path.is_file()
        assert json.loads(outcome.backup_path.read_text(encoding="utf-8")) == original

    def test_migration_failpoint_preserves_backup_and_diagnostic_candidate(self, tmp_path: Path) -> None:
        path = tmp_path / "board.json"
        original = {"board": {"board_id": "b"}}
        path.write_text(json.dumps(original), encoding="utf-8")
        failpoint = Failpoint()
        failpoint.register("after_fsync_before_replace", OSError("injected"))
        with pytest.raises(MigrationError, match="candidate"):
            migrate_board_file(path, expected_board_id="b", failpoint=failpoint)
        assert json.loads(path.read_text(encoding="utf-8")) == original
        backups = list((tmp_path / "migration-backups").glob("*.json"))
        candidates = list((tmp_path / "quarantine").glob("*.migration-*.json"))
        diagnostics = list((tmp_path / "quarantine").glob("*.error.json"))
        assert backups and candidates and diagnostics

    @pytest.mark.parametrize(
        "stage",
        [
            "migration_before_backup",
            "migration_after_backup",
            "migration_after_transform",
            "migration_after_validate",
            "migration_before_publish",
            "migration_after_publish",
        ],
    )
    def test_each_migration_stage_leaves_one_diagnostic(self, tmp_path: Path, stage: str) -> None:
        path = tmp_path / "board.json"
        original = {"board": {"board_id": "b"}}
        path.write_text(json.dumps(original), encoding="utf-8")
        failpoint = Failpoint()
        failpoint.register(stage, RuntimeError(stage))
        with pytest.raises(MigrationError, match=stage):
            migrate_board_file(path, expected_board_id="b", failpoint=failpoint)
        diagnostics = list((tmp_path / "quarantine").glob("*.error.json"))
        candidates = list((tmp_path / "quarantine").glob("*.candidate.json"))
        assert len(diagnostics) == 1
        assert len(candidates) == 1
        diagnostic = json.loads(diagnostics[0].read_text(encoding="utf-8"))
        assert diagnostic["error"] == stage
        assert json.loads(path.read_text(encoding="utf-8")).get("schemaVersion", 0) in {
            0,
            CURRENT_SCHEMA_VERSION,
        }


class TestArchiveCrashRecoveryAndCas:
    def test_failure_after_prepare_resumes_same_operation(self, tmp_lkb_root: Path) -> None:
        store = _create_store(
            tmp_lkb_root / "boards" / "archive-prepare",
            board_id="archive-prepare",
            home=tmp_lkb_root,
        )
        close_board(store, "archive-prepare", actor="u", command_id="close", request_hash="hc")
        failpoint = Failpoint()
        failpoint.register("archive_after_prepare", RuntimeError("stop-after-prepare"))
        store._failpoint = failpoint
        with pytest.raises(RuntimeError, match="stop-after-prepare"):
            archive_board(
                store,
                "archive-prepare",
                actor="original-actor",
                command_id="archive",
                request_hash="ha",
                reason="original reason",
            )
        prepared = store.load()
        assert board_lifecycle_state(prepared) == "archiving"
        operation_id = prepared.lifecycle["archive_operation"]["archive_id"]
        failpoint.unregister("archive_after_prepare")
        result = archive_board(
            store,
            "archive-prepare",
            actor="recovery-actor",
            command_id="archive-recovery",
            request_hash="ha-recovery",
            reason="replacement reason",
        )
        assert result.committed
        archive_info = store.load().lifecycle["archive_info"]
        assert archive_info["archive_id"] == operation_id
        assert archive_info["archived_by"] == "original-actor"
        assert archive_info["reason"] == "original reason"
        wrapper = read_archive(Path(archive_info["archive_path"]), expected_board_id="archive-prepare")
        assert wrapper["createdBy"] == "original-actor"
        assert wrapper["reason"] == "original reason"

    def test_failure_after_publish_keeps_immutable_archive_and_resumes(self, tmp_lkb_root: Path) -> None:
        store = _create_store(
            tmp_lkb_root / "boards" / "archive-publish",
            board_id="archive-publish",
            home=tmp_lkb_root,
        )
        close_board(store, "archive-publish", actor="u", command_id="close", request_hash="hc")
        failpoint = Failpoint()
        failpoint.register("archive_after_publish", RuntimeError("stop-after-publish"))
        store._failpoint = failpoint
        with pytest.raises(RuntimeError, match="stop-after-publish"):
            archive_board(
                store,
                "archive-publish",
                actor="u",
                command_id="archive",
                request_hash="ha",
            )
        assert board_lifecycle_state(store.load()) == "archiving"
        archive_files = list((tmp_lkb_root / "archives").rglob("*.json"))
        assert len(archive_files) == 1
        original_bytes = archive_files[0].read_bytes()
        failpoint.unregister("archive_after_publish")
        archive_board(
            store,
            "archive-publish",
            actor="u",
            command_id="archive",
            request_hash="ha",
        )
        assert archive_files[0].read_bytes() == original_bytes
        assert board_lifecycle_state(store.load()) == "archived"

    def test_archive_cas_rejects_concurrent_lifecycle_change(self, tmp_lkb_root: Path) -> None:
        store = _create_store(
            tmp_lkb_root / "boards" / "archive-race",
            board_id="archive-race",
            home=tmp_lkb_root,
        )
        close_board(store, "archive-race", actor="u", command_id="close", request_hash="hc")
        failpoint = Failpoint()

        def change_state(_name: str) -> None:
            def mutate(env: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
                candidate = transition(env, "closed", actor="racer", reason="race")
                candidate.lifecycle.pop("archive_operation", None)
                return candidate, CommandResult(decision="committed", command_id="race")

            store.execute_atomic(
                "archive-race",
                "race",
                "race-hash",
                None,
                mutate,
                actor="racer",
                lifecycle_operation=True,
            )

        failpoint.register("archive_after_publish", change_state)
        store._failpoint = failpoint
        with pytest.raises(LifecycleError, match="CAS"):
            archive_board(
                store,
                "archive-race",
                actor="u",
                command_id="archive",
                request_hash="ha",
            )
        assert board_lifecycle_state(store.load()) == "closed"
        assert list((tmp_lkb_root / "archives").rglob("*.json"))

    def test_repeated_archive_does_not_overwrite_or_increment(self, tmp_lkb_root: Path) -> None:
        store = _create_store(
            tmp_lkb_root / "boards" / "archive-repeat",
            board_id="archive-repeat",
            home=tmp_lkb_root,
        )
        close_board(store, "archive-repeat", actor="u", command_id="close", request_hash="hc")
        archive_board(
            store,
            "archive-repeat",
            actor="u",
            command_id="archive-1",
            request_hash="ha1",
        )
        revision = store.load().store_revision
        files = list((tmp_lkb_root / "archives").rglob("*.json"))
        archive_board(
            store,
            "archive-repeat",
            actor="u",
            command_id="archive-2",
            request_hash="ha2",
        )
        assert store.load().store_revision == revision
        assert list((tmp_lkb_root / "archives").rglob("*.json")) == files


class TestPurgeTwoPhaseRecovery:
    def _trashed_store(self, root: Path, board_id: str) -> JsonBoardStore:
        store = _create_store(root / "boards" / board_id, board_id=board_id, home=root)
        close_board(store, board_id, actor="u", command_id="close", request_hash="hc")
        trash_board(store, board_id, actor="u", command_id="trash", request_hash="ht")
        return store

    def test_failure_after_purging_resumes_without_reporting_success(self, tmp_lkb_root: Path) -> None:
        store = self._trashed_store(tmp_lkb_root, "purge-prepare")
        failpoint = Failpoint()
        failpoint.register("purge_after_prepare", RuntimeError("stop-after-purging"))
        store._failpoint = failpoint
        with pytest.raises(RuntimeError, match="stop-after-purging"):
            purge_board(
                store,
                "purge-prepare",
                actor="original-admin",
                command_id="purge",
                request_hash="hp",
                reason="original cleanup",
                confirm="purge-prepare",
                authorized=True,
            )
        assert board_lifecycle_state(store.load()) == "purging"
        assert not list((tmp_lkb_root / "tombstones").glob("*.json"))
        failpoint.unregister("purge_after_prepare")
        assert purge_board(
            store,
            "purge-prepare",
            actor="recovery-admin",
            command_id="purge-recovery",
            request_hash="hp-recovery",
            reason="replacement cleanup",
            confirm="purge-prepare",
            authorized=True,
        ).committed
        assert not (tmp_lkb_root / "boards" / "purge-prepare" / "board.json").exists()
        marker = next((tmp_lkb_root / "tombstones").glob("*.json"))
        tombstone = read_tombstone(marker, expected_board_id="purge-prepare")
        assert tombstone["purgedBy"] == "original-admin"
        assert tombstone["reason"] == "original cleanup"

    @pytest.mark.parametrize("stage", ["purge_after_pending", "purge_before_delete"])
    def test_failure_before_managed_deletion_is_reentrant(self, tmp_lkb_root: Path, stage: str) -> None:
        store = self._trashed_store(tmp_lkb_root, f"purge-{stage}")
        board_id = f"purge-{stage}"
        failpoint = Failpoint()
        failpoint.register(stage, RuntimeError(stage))
        store._failpoint = failpoint
        with pytest.raises(RuntimeError, match=stage):
            purge_board(
                store,
                board_id,
                actor="admin",
                command_id="purge",
                request_hash="hp",
                reason="cleanup",
                confirm=board_id,
                authorized=True,
            )
        assert board_lifecycle_state(store.load()) == "purging"
        assert not list((tmp_lkb_root / "tombstones").glob("*.json"))

        failpoint.unregister(stage)
        assert purge_board(
            store,
            board_id,
            actor="recovery-admin",
            command_id="purge-recovery",
            request_hash="hp-recovery",
            reason="replacement cleanup",
            confirm=board_id,
            authorized=True,
        ).committed

    @pytest.mark.parametrize("stage", ["purge_after_delete", "purge_before_tombstone"])
    def test_failure_after_managed_deletion_publishes_tombstone_on_retry(self, tmp_lkb_root: Path, stage: str) -> None:
        store = self._trashed_store(tmp_lkb_root, "purge-delete")
        failpoint = Failpoint()
        failpoint.register(stage, RuntimeError(stage))
        store._failpoint = failpoint
        with pytest.raises(RuntimeError, match=stage):
            purge_board(
                store,
                "purge-delete",
                actor="admin",
                command_id="purge",
                request_hash="hp",
                reason="cleanup",
                confirm="purge-delete",
                authorized=True,
            )
        assert not (tmp_lkb_root / "boards" / "purge-delete" / "board.json").exists()
        assert not list((tmp_lkb_root / "tombstones").glob("*.json"))
        assert list((tmp_lkb_root / "tombstones").glob("*.purge-pending"))

        failpoint.unregister(stage)
        result = purge_board(
            store,
            "purge-delete",
            actor="recovery-admin",
            command_id="purge-recovery",
            request_hash="hp-recovery",
            reason="replacement cleanup",
            confirm="purge-delete",
            authorized=True,
        )
        assert result.committed
        marker = next((tmp_lkb_root / "tombstones").glob("*.json"))
        tombstone = read_tombstone(marker, expected_board_id="purge-delete")
        assert tombstone["purgedBy"] == "admin"
        assert tombstone["reason"] == "cleanup"
        assert not list((tmp_lkb_root / "tombstones").glob("*.purge-pending"))

    def test_failure_after_tombstone_resumes_completed_purge(self, tmp_lkb_root: Path) -> None:
        store = self._trashed_store(tmp_lkb_root, "purge-tombstone")
        failpoint = Failpoint()
        failpoint.register("purge_after_tombstone", RuntimeError("stop-after-tombstone"))
        store._failpoint = failpoint
        with pytest.raises(RuntimeError, match="stop-after-tombstone"):
            purge_board(
                store,
                "purge-tombstone",
                actor="admin",
                command_id="purge",
                request_hash="hp",
                reason="cleanup",
                confirm="purge-tombstone",
                authorized=True,
            )
        assert not (tmp_lkb_root / "boards" / "purge-tombstone" / "board.json").exists()
        assert list((tmp_lkb_root / "tombstones").glob("*.json"))
        failpoint.unregister("purge_after_tombstone")
        result = purge_board(
            store,
            "purge-tombstone",
            actor="admin",
            command_id="purge",
            request_hash="hp",
            reason="cleanup",
            confirm="purge-tombstone",
            authorized=True,
        )
        assert result.committed
        assert not (tmp_lkb_root / "boards" / "purge-tombstone" / "board.json").exists()

    def test_active_watcher_blocks_purge(self, tmp_lkb_root: Path) -> None:
        store = self._trashed_store(tmp_lkb_root, "purge-watcher")
        store._active_watchers = 1
        with pytest.raises(LifecycleTransitionDenied, match="watcher"):
            purge_board(
                store,
                "purge-watcher",
                actor="admin",
                command_id="purge",
                request_hash="hp",
                reason="cleanup",
                confirm="purge-watcher",
                authorized=True,
            )
        assert board_lifecycle_state(store.load()) == "trashed"

    def test_actor_name_alone_does_not_grant_purge_permission(self, tmp_lkb_root: Path) -> None:
        store = self._trashed_store(tmp_lkb_root, "purge-auth")
        with pytest.raises(PermissionError):
            purge_board(
                store,
                "purge-auth",
                actor="admin",
                command_id="purge",
                request_hash="hp",
                reason="cleanup",
                confirm="purge-auth",
            )

    def test_unverified_archive_entry_blocks_before_tombstone(self, tmp_lkb_root: Path) -> None:
        store = _create_store(
            tmp_lkb_root / "boards" / "purge-archive",
            board_id="purge-archive",
            home=tmp_lkb_root,
        )
        close_board(store, "purge-archive", actor="u", command_id="close", request_hash="hc")
        archive_board(
            store,
            "purge-archive",
            actor="u",
            command_id="archive",
            request_hash="ha",
        )
        trash_board(store, "purge-archive", actor="u", command_id="trash", request_hash="ht")
        archive_dir = Path(store.load().lifecycle["archive_info"]["archive_path"]).parent
        (archive_dir / "unverified.bin").write_bytes(b"?")
        with pytest.raises(LifecycleTransitionDenied, match="unverified archive"):
            purge_board(
                store,
                "purge-archive",
                actor="admin",
                command_id="purge",
                request_hash="hp",
                reason="cleanup",
                confirm="purge-archive",
                authorized=True,
            )
        assert board_lifecycle_state(store.load()) == "trashed"
        assert not list((tmp_lkb_root / "tombstones").glob("*.json"))

    def test_old_store_and_direct_create_cannot_resurrect(self, tmp_lkb_root: Path) -> None:
        store = self._trashed_store(tmp_lkb_root, "purge-resurrection")
        purge_board(
            store,
            "purge-resurrection",
            actor="admin",
            command_id="purge",
            request_hash="hp",
            reason="cleanup",
            confirm="purge-resurrection",
            authorized=True,
        )
        with pytest.raises(BoardTombstonedError):
            store.execute_atomic(
                "purge-resurrection",
                "old-session",
                "old-hash",
                None,
                lambda env: (
                    env,
                    CommandResult(decision="committed", command_id="old-session"),
                ),
                actor="old-session",
            )
        with pytest.raises(BoardTombstonedError):
            store.read_snapshot()
        with pytest.raises(BoardTombstonedError):
            _create_store(
                tmp_lkb_root / "boards" / "purge-resurrection",
                board_id="purge-resurrection",
                home=tmp_lkb_root,
            )


class TestGcObservationRevalidation:
    def test_candidate_records_observation_and_changed_temp_is_retained(self, tmp_lkb_root: Path) -> None:
        tmp_dir = tmp_lkb_root / "boards" / "observed" / ".tmp"
        tmp_dir.mkdir(parents=True)
        item = tmp_dir / ".board.json.observed.tmp"
        item.write_text("old", encoding="utf-8")
        now = 8_000_000.0
        old = now - GC_TEMP_AGE_SECONDS - 1
        os.utime(item, (old, old))
        candidate = next(value for value in gc_scan(tmp_lkb_root, dry_run=True, now=now) if value.path == item)
        assert candidate.root == tmp_lkb_root
        assert candidate.observed_mtime_ns is not None
        assert candidate.observed_hash.startswith("sha256:")
        item.write_text("new owner", encoding="utf-8")
        gc_apply([candidate], now=now)
        assert item.is_file()

    def test_open_session_board_is_not_collected(self, tmp_lkb_root: Path) -> None:
        store = _create_session_store(tmp_lkb_root, "open-session")
        board_dir = Path(store._board_dir)
        board_json = board_dir / "board.json"
        now = 9_000_000.0
        old = now - GC_SESSION_ORPHAN_AGE_SECONDS - 1
        os.utime(board_json, (old, old))
        assert not any(
            value.kind == "session_orphan"
            for value in gc_scan(
                tmp_lkb_root,
                dry_run=False,
                now=now,
                open_board_ids={"open-session"},
            )
        )
        assert board_json.is_file()

    def test_checksum_damaged_board_is_reported_and_retained(self, tmp_lkb_root: Path) -> None:
        board_id = "damaged-project"
        board_dir = tmp_lkb_root / "boards" / safe_board_id(board_id)
        store = _create_store(board_dir, board_id=board_id, home=tmp_lkb_root)
        board_json = Path(store._board_json)
        data = json.loads(board_json.read_text(encoding="utf-8"))
        data["lifecycle"]["scope"] = "session"
        board_json.write_text(json.dumps(data), encoding="utf-8")
        now = 11_000_000.0
        old = now - GC_SESSION_ORPHAN_AGE_SECONDS - 1
        os.utime(board_json, (old, old))

        candidates = gc_scan(tmp_lkb_root, dry_run=False, now=now)
        assert any(candidate.kind == "invalid_board" and candidate.path == board_json for candidate in candidates)
        assert board_json.is_file()

    def test_hash_valid_forged_session_scope_on_project_is_retained(self, tmp_lkb_root: Path) -> None:
        board_id = "forged-project"
        board_dir = tmp_lkb_root / "boards" / safe_board_id(board_id)
        store = _create_store(board_dir, board_id=board_id, home=tmp_lkb_root)
        envelope = store.load()
        envelope.lifecycle["scope"] = "session"
        set_payload_hash(envelope, previous_hash=None)
        board_json = Path(store._board_json)
        board_json.write_text(json.dumps(envelope.to_dict()), encoding="utf-8")
        now = 12_000_000.0
        old = now - GC_SESSION_ORPHAN_AGE_SECONDS - 1
        os.utime(board_json, (old, old))

        candidates = gc_scan(tmp_lkb_root, dry_run=False, now=now)
        assert any(candidate.kind == "invalid_board" for candidate in candidates)
        assert board_json.is_file()

    def test_descendant_symlink_prevents_directory_action(self, tmp_lkb_root: Path) -> None:
        tmp_dir = tmp_lkb_root / "boards" / "descendant" / ".tmp"
        tmp_dir.mkdir(parents=True)
        suspicious = tmp_dir / "old-tree"
        suspicious.mkdir()
        outside = tmp_lkb_root / "outside"
        outside.mkdir()
        marker = outside / "marker"
        marker.write_text("keep", encoding="utf-8")
        try:
            (suspicious / "link").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlink unavailable")
        now = 10_000_000.0
        old = now - GC_TEMP_AGE_SECONDS - 1
        os.utime(suspicious, (old, old))
        candidates = gc_scan(tmp_lkb_root, dry_run=False, now=now)
        assert any(value.path == suspicious for value in candidates)
        assert suspicious.is_dir()
        assert marker.is_file()
