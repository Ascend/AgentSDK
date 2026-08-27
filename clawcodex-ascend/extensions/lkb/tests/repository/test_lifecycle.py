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

# pylint: disable=C1803

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

import os
import time
from pathlib import Path

import pytest

from lkb.commands import CommandResult
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
    GC_TEMP_AGE_SECONDS,
    LifecycleData,
    LifecycleTransitionDenied,
    archive_board,
    board_lifecycle_state,
    close_board,
    gc_scan,
    purge_board,
    read_archive,
    read_tombstone,
    reopen_board,
    restore_board,
    trash_board,
    transition,
)
from lkb.repository import ArchiveRef
from lkb.migrations import (
    CURRENT_SCHEMA_VERSION as MIG_CURRENT_SCHEMA,
    BoardSchemaTooNewError,
    MigrationError,
    migrate,
    v0_to_v1,
    v1_to_v2,
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
        home=home,
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
        home=root,
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


class TestLkbLife001DefaultState:
    """Genesis boards default to lifecycle state 'active'."""

    def test_genesis_envelope_state_active(self) -> None:
        env = _make_envelope()
        assert board_lifecycle_state(env) == "active"

    def test_store_create_board_state_active(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board")
        env = store.load()
        assert board_lifecycle_state(env) == "active"

    def test_lifecycle_data_defaults(self) -> None:
        lc = LifecycleData()
        assert lc.state == "active"
        assert lc.scope == "project"
        assert lc.retention_policy == "default"


# ── LKB-LIFE-002 ──────────────────────────────────────────────────────


class TestLkbLife002ValidTransitions:
    """Valid transitions include explicit crash-recovery intermediate states."""

    def test_active_to_closed(self) -> None:
        env = _make_envelope()
        new_env = transition(env, "closed", actor="test-user")
        assert board_lifecycle_state(new_env) == "closed"
        # Original is unchanged.
        assert board_lifecycle_state(env) == "active"

    def test_closed_to_archived(self) -> None:
        env = _make_envelope(state="closed")
        archiving = transition(env, "archiving", actor="test-user")
        new_env = transition(archiving, "archived", actor="test-user")
        assert board_lifecycle_state(new_env) == "archived"

    def test_closed_to_trashed(self) -> None:
        env = _make_envelope(state="closed")
        new_env = transition(env, "trashed", actor="test-user")
        assert board_lifecycle_state(new_env) == "trashed"

    def test_archived_to_trashed(self) -> None:
        env = _make_envelope(state="archived")
        new_env = transition(env, "trashed", actor="test-user")
        assert board_lifecycle_state(new_env) == "trashed"

    def test_trashed_to_purging(self) -> None:
        env = _make_envelope(state="trashed")
        new_env = transition(env, "purging", actor="test-user")
        assert board_lifecycle_state(new_env) == "purging"

    def test_transition_records_event(self) -> None:
        env = _make_envelope()
        new_env = transition(env, "closed", actor="alice", reason="cleanup")
        events = [e for e in new_env.events if e.get("type") == "lifecycle_transition"]
        assert len(events) == 1
        ev = events[0]
        assert ev["from_state"] == "active"
        assert ev["to_state"] == "closed"
        assert ev["actor"] == "alice"
        assert ev["reason"] == "cleanup"

    def test_idempotent_same_state(self) -> None:
        env = _make_envelope(state="closed")
        new_env = transition(env, "closed", actor="test-user")
        # Already in target state — returns the same object (no-op).
        assert new_env is env

    def test_timestamp_updated(self) -> None:
        env = _make_envelope()
        new_env = transition(env, "closed", actor="test-user")
        lc = LifecycleData.from_dict(new_env.lifecycle)
        assert lc.closed_at != ""
        assert lc.updated_at != ""


# ── LKB-LIFE-005 ──────────────────────────────────────────────────────


class TestLkbLife005InvalidTransitions:
    """Invalid state transitions are rejected."""

    def test_active_to_archived_direct_rejected(self) -> None:
        env = _make_envelope()
        with pytest.raises(LifecycleTransitionDenied) as exc_info:
            transition(env, "archived", actor="test-user")
        assert exc_info.value.from_state == "active"
        assert exc_info.value.to_state == "archived"

    def test_active_to_trashed_direct_rejected(self) -> None:
        env = _make_envelope()
        with pytest.raises(LifecycleTransitionDenied):
            transition(env, "trashed", actor="test-user")

    def test_active_to_purged_direct_rejected(self) -> None:
        env = _make_envelope()
        with pytest.raises(LifecycleTransitionDenied):
            transition(env, "purged", actor="test-user")

    def test_purged_no_outgoing_transitions(self) -> None:
        env = _make_envelope(state="purged")
        with pytest.raises(LifecycleTransitionDenied):
            transition(env, "active", actor="test-user")
        with pytest.raises(LifecycleTransitionDenied):
            transition(env, "closed", actor="test-user")

    def test_invalid_target_state_rejected(self) -> None:
        env = _make_envelope()
        with pytest.raises(LifecycleTransitionDenied, match="invalid target state"):
            transition(env, "bogus", actor="test-user")

    def test_board_id_in_error(self) -> None:
        env = _make_envelope("board-X")
        with pytest.raises(LifecycleTransitionDenied) as exc_info:
            transition(env, "archived", actor="test-user")
        assert exc_info.value.board_id == "board-X"


# ── LKB-LIFE-006 ──────────────────────────────────────────────────────


class TestLkbLife006Reopen:
    """Reopen restores a closed board to active."""

    def test_closed_to_active(self) -> None:
        env = _make_envelope(state="closed")
        new_env = transition(env, "active", actor="test-user")
        assert board_lifecycle_state(new_env) == "active"
        lc = LifecycleData.from_dict(new_env.lifecycle)
        assert lc.closed_at == ""

    def test_trashed_to_active(self) -> None:
        env = _make_envelope(state="trashed")
        new_env = transition(env, "active", actor="test-user")
        assert board_lifecycle_state(new_env) == "active"

    def test_archived_to_active_restore(self) -> None:
        env = _make_envelope(state="archived")
        new_env = transition(env, "active", actor="test-user")
        assert board_lifecycle_state(new_env) == "active"
        lc = LifecycleData.from_dict(new_env.lifecycle)
        assert lc.archived_at == ""


# ── LKB-LIFE-007 / LKB-LIFE-008 ──────────────────────────────────────


class TestLkbLife007ArchiveRestore:
    """Archive creates archive copy; restore returns to active."""

    def test_archive_board_persists_transition(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        # First close the board (archive requires closed state).
        close_board(
            store,
            "test-board",
            actor="test-user",
            command_id=_cmd_id("close"),
            request_hash="hash-close",
        )

        # Then archive.
        result = archive_board(
            store,
            "test-board",
            actor="test-user",
            command_id=_cmd_id("archive"),
            request_hash="hash-archive",
            reason="project completed",
        )
        assert result.committed

        env = store.load()
        assert board_lifecycle_state(env) == "archived"

    def test_archive_creates_archive_file(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        close_board(
            store,
            "test-board",
            actor="test-user",
            command_id=_cmd_id("close"),
            request_hash="hash-close",
        )
        archive_board(
            store,
            "test-board",
            actor="test-user",
            command_id=_cmd_id("archive"),
            request_hash="hash-archive",
        )

        # Archive directory should exist.
        archive_dir = tmp_lkb_root / "archives"
        assert archive_dir.is_dir()
        # There should be one archive subdirectory.
        archives = list(archive_dir.iterdir())
        assert len(archives) >= 1

    def test_restore_board_returns_to_active(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        # Close -> archive -> restore.
        close_board(store, "test-board", actor="u", command_id=_cmd_id("c"), request_hash="h1")
        archive_board(store, "test-board", actor="u", command_id=_cmd_id("a"), request_hash="h2")
        archive_ref = _archive_ref(store)
        result = restore_board(
            store,
            "test-board",
            archive_ref=archive_ref,
            actor="u",
            command_id=_cmd_id("r"),
            request_hash="h3",
            reason="need it back",
        )
        assert result.committed

        env = store.load()
        assert board_lifecycle_state(env) == "active"

    def test_restore_records_source_archive(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        close_board(store, "test-board", actor="u", command_id=_cmd_id("c"), request_hash="h1")
        archive_board(store, "test-board", actor="u", command_id=_cmd_id("a"), request_hash="h2")
        restore_board(
            store,
            "test-board",
            archive_ref=_archive_ref(store),
            actor="u",
            command_id=_cmd_id("r"),
            request_hash="h3",
        )

        env = store.load()
        restore_info = env.lifecycle.get("restore_info")
        assert restore_info is not None
        assert restore_info.get("restored_by") == "u"


# ── LKB-LIFE-009 ──────────────────────────────────────────────────────


class TestLkbLife009Purge:
    """Purge removes payload data but leaves tombstone."""

    def test_purge_requires_confirm(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        # Close -> trash -> purge.
        close_board(store, "test-board", actor="u", command_id=_cmd_id("c"), request_hash="h1")
        trash_board(store, "test-board", actor="u", command_id=_cmd_id("t"), request_hash="h2")

        with pytest.raises(ValueError, match="confirm"):
            purge_board(
                store,
                "test-board",
                actor="u",
                command_id=_cmd_id("p"),
                request_hash="h3",
                reason="cleanup",
                confirm="wrong-id",
            )

    def test_purge_requires_reason(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        close_board(store, "test-board", actor="u", command_id=_cmd_id("c"), request_hash="h1")
        trash_board(store, "test-board", actor="u", command_id=_cmd_id("t"), request_hash="h2")

        with pytest.raises(ValueError, match="reason"):
            purge_board(
                store,
                "test-board",
                actor="u",
                command_id=_cmd_id("p"),
                request_hash="h3",
                reason="",
                confirm="test-board",
            )

    def test_purge_clears_payload(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        # Add a node so there's payload to clear.
        def add_node(env: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
            ref = NodeRef("plan", "task", "T-001")
            env.nodes["T-001"] = {
                "ref": ref.to_str(),
                "title": "test task",
                "state": "pending",
                "revision": 1,
                "payload": {},
            }
            env.graphs["plan"] = {
                "graph_id": "plan",
                "board_id": "test-board",
                "graph_kind": "plan",
                "revision": 1,
            }
            result = CommandResult(decision="committed", command_id="add")
            return env, result

        store.execute_atomic(
            "test-board",
            "cmd-add",
            "hash-add",
            None,
            add_node,
            actor="u",
        )

        close_board(store, "test-board", actor="u", command_id=_cmd_id("c"), request_hash="h1")
        trash_board(store, "test-board", actor="u", command_id=_cmd_id("t"), request_hash="h2")
        result = purge_board(
            store,
            "test-board",
            actor="u",
            command_id=_cmd_id("p"),
            request_hash="h3",
            reason="gdpr request",
            confirm="test-board",
            authorized=True,
        )
        assert result.committed

        with pytest.raises(BoardTombstonedError):
            store.load()
        assert not (board_dir / "board.json").exists()
        marker = next((tmp_lkb_root / "tombstones").glob("*.json"))
        tombstone = read_tombstone(marker, expected_board_id="test-board")
        assert tombstone["purgedBy"] == "u"
        assert tombstone["reason"] == "gdpr request"
        assert (board_dir / ".lock").is_file()


# ── LKB-LIFE-010 / LKB-LIFE-011 ──────────────────────────────────────


class TestLkbLife010GcScan:
    """gc_scan dry-run never modifies; finds old temp files."""

    def test_dry_run_does_not_delete(self, tmp_lkb_root: Path) -> None:
        # Create a temp file that's old enough to be a candidate.
        tmp_dir = tmp_lkb_root / "boards" / "test-board" / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        old_tmp = tmp_dir / ".board.json.old-file.tmp"
        old_tmp.write_text("stale", encoding="utf-8")

        # Set mtime to 48h ago.
        old_time = time.time() - (GC_TEMP_AGE_SECONDS + 3600)
        os.utime(old_tmp, (old_time, old_time))

        before_size = old_tmp.stat().st_size
        candidates = gc_scan(tmp_lkb_root, dry_run=True)

        # File still exists after dry-run.
        assert old_tmp.is_file()
        assert old_tmp.stat().st_size == before_size
        # But it was reported as a candidate.
        assert any(c.path == old_tmp for c in candidates)

    def test_finds_old_temp_files(self, tmp_lkb_root: Path) -> None:
        tmp_dir = tmp_lkb_root / "boards" / "board-a" / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Old temp file (should be found).
        old_tmp = tmp_dir / ".board.json.old.tmp"
        old_tmp.write_text("old", encoding="utf-8")
        old_time = time.time() - (GC_TEMP_AGE_SECONDS + 100)
        os.utime(old_tmp, (old_time, old_time))

        # Recent temp file (should NOT be found).
        new_tmp = tmp_dir / ".board.json.new.tmp"
        new_tmp.write_text("new", encoding="utf-8")
        new_time = time.time() - 60  # 1 minute ago
        os.utime(new_tmp, (new_time, new_time))

        candidates = gc_scan(tmp_lkb_root, dry_run=True)
        temp_candidates = [c for c in candidates if c.kind == "temp"]
        assert any(c.path == old_tmp for c in temp_candidates)
        assert not any(c.path == new_tmp for c in temp_candidates)

    def test_empty_root_returns_empty(self, tmp_path: Path) -> None:
        empty_root = tmp_path / "lkb-empty"
        empty_root.mkdir()
        candidates = gc_scan(empty_root, dry_run=True)
        assert candidates == []

    def test_candidates_sorted_by_age_desc(self, tmp_lkb_root: Path) -> None:
        tmp_dir = tmp_lkb_root / "boards" / "board-b" / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Create two temp files with different ages.
        for i, hours in enumerate([48, 72]):
            f = tmp_dir / f".board.json.file-{i}.tmp"
            f.write_text("x", encoding="utf-8")
            t = time.time() - hours * 3600
            os.utime(f, (t, t))

        candidates = gc_scan(tmp_lkb_root, dry_run=True)
        temp_cands = [c for c in candidates if c.kind == "temp"]
        assert len(temp_cands) >= 2
        for i in range(len(temp_cands) - 1):
            assert temp_cands[i].age_seconds >= temp_cands[i + 1].age_seconds

    def test_quarantine_candidates(self, tmp_lkb_root: Path) -> None:
        q_dir = tmp_lkb_root / "boards" / "board-c" / "quarantine"
        q_dir.mkdir(parents=True, exist_ok=True)

        q_file = q_dir / "corrupt.1234.primary-corrupt"
        q_file.write_text("broken", encoding="utf-8")
        # 35 days old (past the 30-day threshold).
        old_time = time.time() - 35 * 24 * 3600
        os.utime(q_file, (old_time, old_time))

        candidates = gc_scan(tmp_lkb_root, dry_run=True)
        q_cands = [c for c in candidates if c.kind == "quarantine"]
        assert any(c.path == q_file for c in q_cands)

    def test_gc_candidate_attributes(self, tmp_lkb_root: Path) -> None:
        tmp_dir = tmp_lkb_root / "boards" / "board-d" / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        f = tmp_dir / ".board.json.stale.tmp"
        f.write_bytes(b"x" * 100)
        old_time = time.time() - (GC_TEMP_AGE_SECONDS + 1000)
        os.utime(f, (old_time, old_time))

        candidates = gc_scan(tmp_lkb_root, dry_run=True)
        temp_cands = [c for c in candidates if c.path == f]
        assert len(temp_cands) == 1
        c = temp_cands[0]
        assert c.kind == "temp"
        assert c.age_seconds > 0
        assert c.size_bytes == 100
        assert "temp" in c.reason


# ── LKB-LIFE-016 ──────────────────────────────────────────────────────


class TestLkbLife016ActiveClaimsGuard:
    """Close with active claims rejected unless override + reason (LKB-LIFE-016)."""

    def test_close_with_active_claims_rejected(self) -> None:
        env = _make_envelope()
        _add_active_claim(env)
        with pytest.raises(LifecycleTransitionDenied, match="active claims"):
            transition(env, "closed", actor="test-user")

    def test_close_with_override_no_reason_rejected(self) -> None:
        env = _make_envelope()
        _add_active_claim(env)
        with pytest.raises(LifecycleTransitionDenied, match="active claims"):
            transition(env, "closed", actor="test-user", override_active_claims=True)

    def test_close_with_override_and_reason_allowed(self) -> None:
        env = _make_envelope()
        _add_active_claim(env)
        new_env = transition(
            env,
            "closed",
            actor="test-user",
            reason="emergency shutdown",
            override_active_claims=True,
        )
        assert board_lifecycle_state(new_env) == "closed"
        # Verify the override is recorded in the event.
        events = [e for e in new_env.events if e.get("type") == "lifecycle_transition"]
        assert events
        assert events[-1].get("override_active_claims") is True

    def test_close_no_claims_succeeds(self) -> None:
        env = _make_envelope()
        new_env = transition(env, "closed", actor="test-user")
        assert board_lifecycle_state(new_env) == "closed"

    def test_trash_with_active_claims_rejected(self) -> None:
        env = _make_envelope(state="closed")
        _add_active_claim(env)
        with pytest.raises(LifecycleTransitionDenied, match="active claims"):
            transition(env, "trashed", actor="test-user")


# ── LKB-LIFE-017 ──────────────────────────────────────────────────────


class TestLkbLife017PersistedTransitions:
    """Lifecycle transitions are persisted via execute_atomic."""

    def test_close_board_persists(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        result = close_board(
            store,
            "test-board",
            actor="test-user",
            command_id="cmd-close-1",
            request_hash="hash-close-1",
            reason="cleanup",
        )
        assert result.committed

        # Re-load and verify.
        env = store.load()
        assert board_lifecycle_state(env) == "closed"
        assert env.store_revision == 1  # genesis (0) + 1 transition

    def test_close_is_idempotent_via_command_id(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        cmd_id = _cmd_id("close")
        close_board(
            store,
            "test-board",
            actor="test-user",
            command_id=cmd_id,
            request_hash="hash-close",
        )
        # Same command_id + same hash — should return cached result.
        result = close_board(
            store,
            "test-board",
            actor="test-user",
            command_id=cmd_id,
            request_hash="hash-close",
        )
        assert result.committed
        # store_revision should still be 1 (no new revision).
        env = store.load()
        assert env.store_revision == 1

    def test_full_lifecycle_chain_persisted(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        # active -> closed
        close_board(store, "test-board", actor="u", command_id=_cmd_id("c1"), request_hash="h1")
        # closed -> archived
        archive_board(store, "test-board", actor="u", command_id=_cmd_id("a1"), request_hash="h2")
        # archived -> trashed (via trash_board)
        trash_board(store, "test-board", actor="u", command_id=_cmd_id("t1"), request_hash="h3")
        # trashed -> purged
        purge_board(
            store,
            "test-board",
            actor="u",
            command_id=_cmd_id("p1"),
            request_hash="h4",
            reason="end of life",
            confirm="test-board",
            authorized=True,
        )

        with pytest.raises(BoardTombstonedError):
            store.load()
        assert not (board_dir / "board.json").exists()
        assert list((tmp_lkb_root / "tombstones").glob("*.json"))

    def test_reopen_after_close(self, tmp_lkb_root: Path) -> None:
        board_dir = tmp_lkb_root / "boards" / "test-board"
        store = _create_store(board_dir, board_id="test-board", home=tmp_lkb_root)

        close_board(store, "test-board", actor="u", command_id=_cmd_id("c"), request_hash="h1")
        result = reopen_board(
            store,
            "test-board",
            actor="u",
            command_id=_cmd_id("r"),
            request_hash="h2",
            reason="reconsidered",
        )
        assert result.committed
        env = store.load()
        assert board_lifecycle_state(env) == "active"


# ── LKB-STORE-008 ────────────────────────────────────────────────────


class TestLkbStore008V0ToV1Migration:
    """v0 -> v1 migration upgrades schemaVersion and recomputes hash."""

    def test_migration_adds_schema_version(self) -> None:
        v0_env = {
            "storeFormat": "lkb-json-v1",
            "storeRevision": 0,
            "board": {"board_id": "test-board"},
        }
        result, applied = migrate(v0_env, target_schema=1)
        assert result["schemaVersion"] == 1
        assert applied == [1]

    def test_migration_sets_integrity_hash(self) -> None:
        v0_env = {
            "storeFormat": "lkb-json-v1",
            "storeRevision": 0,
            "board": {"board_id": "test-board"},
        }
        result, _ = migrate(v0_env, target_schema=1)
        integrity = result.get("integrity")
        assert isinstance(integrity, dict)
        assert integrity.get("algorithm") == "sha256"
        assert integrity.get("payloadHash", "").startswith("sha256:")

    def test_migration_sets_lifecycle_default(self) -> None:
        v0_env = {
            "board": {"board_id": "test-board"},
        }
        result, _ = migrate(v0_env, target_schema=1)
        lifecycle = result.get("lifecycle")
        assert isinstance(lifecycle, dict)
        assert lifecycle.get("state") == "active"

    def test_migration_preserves_board_data(self) -> None:
        v0_env = {
            "board": {
                "board_id": "my-board",
                "display_name": "My Board",
                "project_uri": "project:my-proj",
            },
            "graphs": {"plan": {"graph_id": "plan", "graph_kind": "plan"}},
        }
        result, _ = migrate(v0_env, target_schema=1)
        assert result["board"]["board_id"] == "my-board"
        assert result["board"]["display_name"] == "My Board"
        assert "plan" in result["graphs"]


# ── LKB-STORE-009 ────────────────────────────────────────────────────


class TestLkbStore009MigrationIdempotent:
    """Migration is idempotent — already-v1 envelope is unchanged."""

    def test_v1_envelope_no_op(self) -> None:
        v1_env = {
            "storeFormat": "lkb-json-v1",
            "schemaVersion": 1,
            "storeRevision": 3,
            "board": {"board_id": "test-board"},
            "graphs": {},
            "nodes": {},
            "edges": {},
            "claims": {},
            "validationRuns": {},
            "processedCommands": {},
            "events": [],
            "historySegments": [],
            "lifecycle": {"state": "active"},
            "integrity": {"algorithm": "sha256", "payloadHash": "sha256:abc123"},
        }
        # Pass the same dict object — migrate should return it unchanged.
        result, applied = migrate(v1_env, target_schema=1)
        assert applied == []
        # The envelope should be returned as-is (same dict reference)
        # when no migration is needed.
        assert result is v1_env
        assert result["schemaVersion"] == 1

    def test_v0_to_v1_function_direct_idempotent(self) -> None:
        v1_env = {
            "schemaVersion": 1,
            "board": {"board_id": "test"},
        }
        result = v0_to_v1(dict(v1_env))
        # Already v1 — returns unchanged.
        assert result["schemaVersion"] == 1

    def test_v1_to_v2_drops_retired_collections(self) -> None:
        v1_env = {
            "storeFormat": "lkb-json-v1",
            "schemaVersion": 1,
            "storeRevision": 3,
            "board": {"board_id": "test-board"},
            "graphs": {},
            "nodes": {},
            "edges": {},
            "claims": {},
            "assertions": {"A-1": {"value": "retired"}},
            "evidence": {"E-1": {"value": "retired"}},
            "validationRuns": {},
            "processedCommands": {},
            "events": [],
            "historySegments": [],
            "lifecycle": {"state": "active"},
            "integrity": {"algorithm": "sha256", "payloadHash": "sha256:old"},
        }

        result = v1_to_v2(v1_env)

        assert result["schemaVersion"] == 2
        assert result["storeFormat"] == "lkb-json-v2"
        assert "assertions" not in result
        assert "evidence" not in result


# ── LKB-STORE-025 ────────────────────────────────────────────────────


class TestLkbStore025ForwardCompat:
    """Forward-compat: schema newer than code raises BoardSchemaTooNewError."""

    def test_newer_schema_raises(self) -> None:
        v5_env = {
            "schemaVersion": 5,
            "board": {"board_id": "future-board"},
        }
        with pytest.raises(BoardSchemaTooNewError) as exc_info:
            migrate(v5_env, target_schema=1)
        assert exc_info.value.board_id == "future-board"
        assert exc_info.value.on_disk_version == 5
        assert exc_info.value.supported_version == 1

    def test_current_schema_constant_consistent(self) -> None:
        """The migrations CURRENT_SCHEMA_VERSION matches json_store's."""
        assert MIG_CURRENT_SCHEMA == CURRENT_SCHEMA_VERSION

    def test_exact_match_no_error(self) -> None:
        v1_env = {
            "schemaVersion": 1,
            "board": {"board_id": "ok-board"},
        }
        result, applied = migrate(v1_env, target_schema=1)
        assert applied == []
        assert result["schemaVersion"] == 1

    def test_migration_error_on_missing_chain(self) -> None:
        """If the migration chain has a gap, MigrationError is raised."""
        # Trying to migrate to a version beyond what's registered.
        # With only v0->v1 registered, target=3 should fail.
        v0_env = {
            "board": {"board_id": "test"},
        }
        with pytest.raises(MigrationError, match="No migration registered"):
            migrate(v0_env, target_schema=3)
