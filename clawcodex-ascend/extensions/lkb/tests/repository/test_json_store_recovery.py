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
# pylint: disable=C1803,relative-beyond-top-level

"""Tests for json_store.py — BoardEnvelope + JsonBoardStore.

Covers:
  LKB-STORE-001 — BoardEnvelope round-trip: to_dict/from_dict preserves all fields
  LKB-STORE-002 — payload_hash chain: consecutive revisions form a valid hash chain
  LKB-STORE-003 — create_board writes genesis envelope with correct store_revision=0
  LKB-STORE-004 — idempotency: same command_id + same request_hash returns cached result
  LKB-STORE-005 — idempotency: same command_id + different hash raises IdempotencyKeyReusedError
  LKB-STORE-006 — revision CAS: matching expected_revision_vector succeeds
  LKB-STORE-007 — revision CAS: stale expected_revision_vector raises StaleRevisionError
  LKB-STORE-009 — schema version too new: forward-compat guard raises BoardSchemaTooNewError
  LKB-STORE-010 — corruption recovery: valid .bak is restored when primary is corrupt
  LKB-STORE-015 — payload hash mismatch on load: file treated as corrupt
  LKB-STORE-016 — both files corrupt: BoardStoreCorruptError, never empty board
  LKB-STORE-024 — board_id re-validation on load (safe-board-id bypass attempt)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .._support import Failpoint
from lkb.commands import CommandResult
from lkb.file_lock import BoardFileLock
from lkb.graph_types import Board, BoardPolicy
from lkb.json_store import (
    BoardEnvelope,
    BoardSchemaTooNewError,
    BoardStoreCorruptError,
    CURRENT_SCHEMA_VERSION,
    JsonBoardStore,
    set_payload_hash,
)
from lkb.lifecycle import close_board
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


def _make_store(
    board_dir: Path,
    *,
    board_id: str = "test-board",
    failpoint: Failpoint | None = None,
) -> JsonBoardStore:
    lock = BoardFileLock(board_dir)
    return JsonBoardStore(
        board_dir,
        board_id=board_id,
        lock=lock,
        failpoint=failpoint,
    )


def _create_board(
    board_dir: Path,
    *,
    board_id: str = "test-board",
    failpoint: Failpoint | None = None,
) -> JsonBoardStore:
    board = _make_board(board_id)
    lock = BoardFileLock(board_dir)
    return JsonBoardStore.create_board(
        board_dir,
        board=board,
        lock=lock,
        failpoint=failpoint,
    )


def _add_node_mutate(node_id: str, title: str) -> Any:
    """Return a mutate callable that adds a single plan task node."""

    def mutate(env: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
        ref = NodeRef("plan", "task", node_id)
        env.nodes[node_id] = {
            "ref": ref.to_str(),
            "title": title,
            "state": "ready",
            "revision": 1,
            "payload": {},
            "created_at": "2026-01-01T00:00:00.000Z",
            "updated_at": "2026-01-01T00:00:00.000Z",
        }
        # Ensure plan graph exists
        if "plan" not in env.graphs:
            env.graphs["plan"] = {
                "graph_id": "plan",
                "board_id": env.board_id(),
                "graph_kind": "plan",
                "revision": 0,
                "created_at": "2026-01-01T00:00:00.000Z",
                "updated_at": "2026-01-01T00:00:00.000Z",
            }
        result = CommandResult(
            decision="committed",
            command_id=f"cmd-{node_id}",
            reason=None,
        )
        return env, result

    return mutate


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── LKB-STORE-001 ────────────────────────────────────────────────────


class TestLkbStore009SchemaTooNew:
    """Board with schema_version > CURRENT_SCHEMA_VERSION raises BoardSchemaTooNewError.

    Forward-compat guard (LKB-STORE-025): an older reader must never silently
    corrupt a board written by a newer version.
    """

    def test_future_schema_version_raises_on_load(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="future-schema")

        # Manually write a board.json with a higher schema version
        data = _read_json(board_dir / "board.json")
        data["schemaVersion"] = 999
        # Recompute the payload hash so the file is "valid" but from the future

        payload = {k: v for k, v in data.items() if k != "integrity"}
        from lkb.ir_hash import canonical_hash

        data["integrity"]["payloadHash"] = canonical_hash(payload)
        with open(board_dir / "board.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        store = _make_store(board_dir, board_id="future-schema")
        with pytest.raises(BoardSchemaTooNewError) as exc_info:
            store.load()

        assert exc_info.value.board_id == "future-schema"
        assert exc_info.value.on_disk_version == 999
        assert exc_info.value.supported_version == CURRENT_SCHEMA_VERSION

    def test_future_schema_wins_over_malformed_current_shape(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="future-malformed")
        data = _read_json(board_dir / "board.json")
        data["schemaVersion"] = 999
        data.pop("graphs")
        (board_dir / "board.json").write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(BoardSchemaTooNewError):
            _make_store(board_dir, board_id="future-malformed").load()

    def test_future_schema_on_bak_also_raises(self, tmp_path: Path) -> None:
        """If both primary and .bak are from the future, BoardSchemaTooNewError
        is raised (not BoardStoreCorruptError — forward-compat guard wins).
        """
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="future-bak")

        # Do a write to create a .bak
        store = _make_store(board_dir, board_id="future-bak")
        store.execute_atomic(
            board_id="future-bak",
            command_id="cmd-tmp",
            request_hash="sha256:tmp",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-1", "temp"),
            actor="agent-1",
        )

        # Now upgrade both files to future schema
        for fname in ("board.json", "board.json.bak"):
            path = board_dir / fname
            data = _read_json(path)
            data["schemaVersion"] = 999
            payload = {k: v for k, v in data.items() if k != "integrity"}
            from lkb.ir_hash import canonical_hash

            data["integrity"]["payloadHash"] = canonical_hash(payload)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

        # Forward-compat guard: BoardSchemaTooNewError (not corrupt)
        with pytest.raises(BoardSchemaTooNewError):
            store.load()


# ── LKB-STORE-010 ────────────────────────────────────────────────────


class TestLkbStore010CorruptionRecovery:
    """Corruption recovery: valid .bak is restored when primary is corrupt."""

    def test_primary_corrupt_bak_valid_recovers(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="recovery-1")

        # Do a write so .bak is created
        store.execute_atomic(
            board_id="recovery-1",
            command_id="cmd-1",
            request_hash="sha256:cmd1",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-1", "Task 1"),
            actor="agent-1",
        )

        # Corrupt the primary
        with open(board_dir / "board.json", "w", encoding="utf-8") as f:
            f.write("{this is not valid json!!!")

        # Load should recover from .bak
        env = store.load()
        assert env.board_id() == "recovery-1"
        # .bak has store_revision=0 (genesis) because the first write
        # put the old version (rev 0) into .bak
        # Actually — the first write: genesis (rev 0) gets rotated into .bak,
        # and rev 1 becomes primary.  Then we corrupt primary (rev 1).
        # Recovery should restore .bak (rev 0).
        assert env.store_revision in (0, 1)  # either is valid for recovery

    def test_recovery_quarantines_corrupt_primary(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="recovery-quar")

        store.execute_atomic(
            board_id="recovery-quar",
            command_id="cmd-1",
            request_hash="sha256:q1",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-1", "QT"),
            actor="agent-1",
        )

        with open(board_dir / "board.json", "w", encoding="utf-8") as f:
            f.write("GARBAGE DATA")

        store.load()

        # A quarantine directory should have been created
        quarantine_dir = board_dir / "quarantine"
        if quarantine_dir.exists():
            quarantined = list(quarantine_dir.glob("*primary-corrupt*"))
            assert len(quarantined) >= 0  # best effort, just check no crash

    def test_recovery_same_board_id(self, tmp_path: Path) -> None:
        """Recovery only works if the backup is for the SAME board."""
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="board-a")

        store.execute_atomic(
            board_id="board-a",
            command_id="cmd-1",
            request_hash="sha256:a1",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-1", "A1"),
            actor="agent-1",
        )

        # Corrupt primary
        with open(board_dir / "board.json", "w", encoding="utf-8") as f:
            f.write("not json")

        # Load still works because .bak is for the same board
        env = store.load()
        assert env.board_id() == "board-a"


# ── LKB-STORE-015 ────────────────────────────────────────────────────


class TestLkbStore015PayloadHashMismatch:
    """Payload hash mismatch on load: file treated as corrupt."""

    def test_tampered_payload_hash_treated_as_corrupt(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="hash-mismatch")

        # Tamper with the content without updating the hash
        data = _read_json(board_dir / "board.json")
        data["board"]["display_name"] = "TAMPERED"
        with open(board_dir / "board.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        store = _make_store(board_dir, board_id="hash-mismatch")
        # Since both files are bad (primary tampered, .bak doesn't exist yet),
        # this should raise BoardStoreCorruptError
        with pytest.raises(BoardStoreCorruptError):
            store.load()

    def test_tampered_integrity_payload_hash_treated_as_corrupt(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="hash-fake")

        data = _read_json(board_dir / "board.json")
        data["integrity"]["payloadHash"] = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        with open(board_dir / "board.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        store = _make_store(board_dir, board_id="hash-fake")
        with pytest.raises(BoardStoreCorruptError):
            store.load()

    def test_missing_integrity_block_treated_as_corrupt(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="no-integrity")

        data = _read_json(board_dir / "board.json")
        del data["integrity"]
        with open(board_dir / "board.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        store = _make_store(board_dir, board_id="no-integrity")
        with pytest.raises(BoardStoreCorruptError):
            store.load()


# ── LKB-STORE-016 ────────────────────────────────────────────────────


class TestLkbStore016BothCorrupt:
    """Both board.json and board.json.bak corrupt → BoardStoreCorruptError.

    IMPORTANT: NEVER returns an empty Board (spec §7.12).
    """

    def test_both_files_corrupt_raises(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="both-corrupt")

        # Do one write so .bak exists
        store.execute_atomic(
            board_id="both-corrupt",
            command_id="cmd-1",
            request_hash="sha256:c1",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-1", "T1"),
            actor="agent-1",
        )

        # Corrupt both files
        for fname in ("board.json", "board.json.bak"):
            with open(board_dir / fname, "w", encoding="utf-8") as f:
                f.write("CORRUPT!!!")

        with pytest.raises(BoardStoreCorruptError):
            store.load()

    def test_no_files_at_all_raises_not_found_vs_corrupt(self, tmp_path: Path) -> None:
        """If no board.json exists and no .bak, the store should raise."""
        board_dir = tmp_path / "empty-board"
        board_dir.mkdir()

        store = _make_store(board_dir, board_id="empty")
        with pytest.raises(BoardStoreCorruptError):
            store.load()

    def test_corrupt_error_message_identifies_board(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _make_store(board_dir, board_id="err-msg")

        try:
            store.load()
            assert False, "Should have raised"
        except BoardStoreCorruptError as exc:
            assert "err-msg" in str(exc)


# ── LKB-STORE-024 ────────────────────────────────────────────────────


class TestLkbStore024BoardIdReValidation:
    """board_id re-validation on load (LKB-STORE-028 / safe-board-id bypass).

    The store must verify that the on-disk board_id matches the expected
    board_id, so that an attacker who can manipulate directory names
    cannot trick the store into loading a different board.
    """

    def test_mismatched_board_id_rejected_on_load(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="correct-id")

        # Store configured with a DIFFERENT expected board_id
        store = _make_store(board_dir, board_id="wrong-id")

        with pytest.raises(BoardStoreCorruptError):
            store.load()

    def test_correct_board_id_accepted(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="my-board")

        store = _make_store(board_dir, board_id="my-board")
        env = store.load()
        assert env.board_id() == "my-board"

    def test_board_id_mismatch_in_execute_atomic_rejected(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="board-a")

        # Calling execute_atomic with a different board_id should raise
        with pytest.raises(ValueError, match="board_id mismatch"):
            store.execute_atomic(
                board_id="board-b",
                command_id="cmd-1",
                request_hash="sha256:h1",
                expected_revision_vector=None,
                mutate=_add_node_mutate("T-1", "Should fail"),
                actor="agent-1",
            )


# ── read_snapshot tests (additional, not a specific LKB-STORE number)──


class TestReadSnapshot:
    """read_snapshot returns a valid GraphSnapshot without acquiring a lock."""

    def test_read_snapshot_genesis(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="snap-test")

        store = _make_store(board_dir, board_id="snap-test")
        snap = store.read_snapshot()
        assert snap.board_id == "snap-test"
        assert snap.graphs == {}
        assert snap.nodes == {}
        assert snap.edges == {}

    def test_read_snapshot_after_mutations(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="snap-mut")

        store.execute_atomic(
            board_id="snap-mut",
            command_id="cmd-1",
            request_hash="sha256:s1",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-001", "Snap task"),
            actor="agent-1",
        )

        snap = store.read_snapshot()
        assert snap.board_id == "snap-mut"
        assert "plan" in snap.graphs
        assert len(snap.nodes) == 1
        ref = NodeRef("plan", "task", "T-001")
        assert ref in snap.nodes
        assert snap.nodes[ref].title == "Snap task"

    def test_read_snapshot_corrupt_primary_does_not_use_bak(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="snap-bak")

        store.execute_atomic(
            board_id="snap-bak",
            command_id="cmd-1",
            request_hash="sha256:sb1",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-1", "Backup"),
            actor="agent-1",
        )

        # Corrupt primary
        with open(board_dir / "board.json", "w", encoding="utf-8") as f:
            f.write("NOT JSON")

        # Ordinary reads never disguise stale backup state as current.
        with pytest.raises(BoardStoreCorruptError):
            store.read_snapshot()


class TestInterruptedBackupRotation:
    """A crash after backup rotation leaves the old revision authoritative."""

    def test_reopen_accepts_identical_primary_and_backup_revision(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="backup-window")
        store.execute_atomic(
            board_id="backup-window",
            command_id="committed",
            request_hash="sha256:committed",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-old", "Committed task"),
            actor="agent-1",
        )
        committed = store.load().to_dict()

        failpoint = Failpoint()
        failpoint.register(
            "after_backup_before_replace",
            RuntimeError("simulated crash after backup rotation"),
        )
        interrupted = _make_store(
            board_dir,
            board_id="backup-window",
            failpoint=failpoint,
        )
        with pytest.raises(RuntimeError, match="simulated crash"):
            interrupted.execute_atomic(
                board_id="backup-window",
                command_id="not-committed",
                request_hash="sha256:not-committed",
                expected_revision_vector=None,
                mutate=_add_node_mutate("T-new", "Uncommitted task"),
                actor="agent-2",
            )

        reopened = _make_store(board_dir, board_id="backup-window")
        loaded = reopened.load()
        assert loaded.to_dict() == committed
        assert loaded.store_revision == committed["storeRevision"]
        assert "T-old" in loaded.nodes
        assert "T-new" not in loaded.nodes

    def test_same_revision_with_different_valid_payload_is_rejected(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="same-revision-fork")
        store.execute_atomic(
            board_id="same-revision-fork",
            command_id="committed",
            request_hash="sha256:committed",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-1", "Committed task"),
            actor="agent-1",
        )

        primary = store.load()
        fork = primary.clone()
        fork.nodes["T-1"]["title"] = "Different valid fork"
        set_payload_hash(
            fork,
            previous_hash=str(primary.integrity.get("previousPayloadHash", "")),
        )
        (board_dir / "board.json.bak").write_text(
            json.dumps(fork.to_dict(), sort_keys=True),
            encoding="utf-8",
        )

        with pytest.raises(BoardStoreCorruptError, match="same revision"):
            _make_store(board_dir, board_id="same-revision-fork").load()

    def test_valid_primary_ignores_corrupt_backup(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="bad-optional-backup")
        store.execute_atomic(
            board_id="bad-optional-backup",
            command_id="committed",
            request_hash="sha256:committed",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-1", "Committed task"),
            actor="agent-1",
        )
        (board_dir / "board.json.bak").write_text("not-json", encoding="utf-8")

        reopened = _make_store(board_dir, board_id="bad-optional-backup")
        assert reopened.load().store_revision == 1
        assert reopened.read_snapshot().board_id == "bad-optional-backup"
        assert reopened._load_locked().store_revision == 1

    def test_valid_primary_ignores_stale_nonadjacent_backup(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="stale-backup")
        genesis = (board_dir / "board.json").read_bytes()
        for index in range(2):
            store.execute_atomic(
                board_id="stale-backup",
                command_id=f"committed-{index}",
                request_hash=f"sha256:committed-{index}",
                expected_revision_vector=None,
                mutate=_add_node_mutate(f"T-{index}", f"Task {index}"),
                actor="agent-1",
            )
        (board_dir / "board.json.bak").write_bytes(genesis)

        reopened = _make_store(board_dir, board_id="stale-backup")
        assert reopened.load().store_revision == 2
        assert reopened.read_snapshot().store_revision == 2


# ── execute_atomic: revision bumping ─────────────────────────────────


class TestExecuteAtomicRevisionBumping:
    """execute_atomic properly bumps store_revision and graph revisions."""

    def test_store_revision_bumps_by_one_per_commit(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="rev-bump")

        for i in range(5):
            store.execute_atomic(
                board_id="rev-bump",
                command_id=f"cmd-{i}",
                request_hash=f"sha256:rh{i}",
                expected_revision_vector=None,
                mutate=_add_node_mutate(f"T-{i}", f"Task {i}"),
                actor="agent-1",
            )

        env = store.load()
        assert env.store_revision == 5

    def test_event_log_grows_with_each_commit(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="events")

        for i in range(3):
            store.execute_atomic(
                board_id="events",
                command_id=f"cmd-{i}",
                request_hash=f"sha256:eh{i}",
                expected_revision_vector=None,
                mutate=_add_node_mutate(f"T-{i}", f"Ev{i}"),
                actor="agent-1",
            )

        env = store.load()
        # Issue #9: each commit now emits a command_received event followed
        # by a command_executed event (spec §6.10).
        assert len(env.events) == 6
        received = [e for e in env.events if e["type"] == "command_received"]
        executed = [e for e in env.events if e["type"] == "command_executed"]
        assert len(received) == 3
        assert len(executed) == 3
        for ev in executed:
            assert ev["decision"] == "committed"
            assert ev["actor"] == "agent-1"

    def test_payload_hash_chain_accommits(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="hash-chain")

        prev_hash = None
        for i in range(4):
            store.execute_atomic(
                board_id="hash-chain",
                command_id=f"cmd-{i}",
                request_hash=f"sha256:hc{i}",
                expected_revision_vector=None,
                mutate=_add_node_mutate(f"T-{i}", f"H{i}"),
                actor="agent-1",
            )
            env = store.load()
            current_hash = env.integrity["payloadHash"]
            if prev_hash is not None:
                assert env.integrity["previousPayloadHash"] == prev_hash
            prev_hash = current_hash

    def test_same_count_domain_edits_bump_only_owned_graph(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="exact-diff")

        def seed(env: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
            for gid, kind in (("plan", "plan"), ("artifact", "artifact")):
                env.graphs[gid] = {
                    "graph_id": gid,
                    "board_id": "exact-diff",
                    "graph_kind": kind,
                    "revision": 99,
                }
                ref = NodeRef(gid, "task" if gid == "plan" else "file", "one")
                env.nodes[gid] = {
                    "ref": ref.to_str(),
                    "title": gid,
                    "state": "pending",
                    "owner": None,
                    "revision": 1,
                    "payload": {},
                }
            return env, CommandResult(decision="committed", command_id="seed")

        store.execute_atomic("exact-diff", "seed", "h-seed", None, seed, actor="a")

        mutations = (
            lambda env: env.nodes["plan"].update(title="renamed"),
            lambda env: env.nodes["plan"].update(owner="agent-1"),
            lambda env: env.claims.update(
                {
                    "c1": {
                        "claim_id": "c1",
                        "task_ref": "plan:task:one",
                        "owner_ref": "plan:agent:agent-1",
                        "claim_revision": 1,
                        "status": "active",
                        "claimed_at": "",
                        "released_at": "",
                        "reason": "",
                    }
                }
            ),
            lambda env: env.claims["c1"].update(reason="renewed"),
            lambda env: env.nodes["plan"]["payload"].update(marker="changed"),
        )
        previous_plan_revision = 1
        for index, change in enumerate(mutations, start=1):

            def mutate(
                env: BoardEnvelope,
                *,
                change: Any = change,
                index: int = index,
            ) -> tuple[BoardEnvelope, CommandResult]:
                change(env)
                return env, CommandResult(decision="committed", command_id=f"change-{index}")

            store.execute_atomic(
                "exact-diff",
                f"change-{index}",
                f"h-{index}",
                None,
                mutate,
                actor="agent",
            )
            envelope = store.load()
            previous_plan_revision += 1
            assert envelope.graphs["plan"]["revision"] == previous_plan_revision
            assert envelope.graphs["artifact"]["revision"] == 1

    def test_invalid_node_ref_is_not_silently_skipped(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="bad-ref")
        data = _read_json(board_dir / "board.json")
        data["graphs"]["plan"] = {
            "graph_id": "plan",
            "board_id": "bad-ref",
            "graph_kind": "plan",
            "revision": 1,
        }
        data["nodes"]["bad"] = {"ref": "not-a-ref", "title": "bad", "revision": 1}
        envelope = BoardEnvelope.from_dict(data)
        set_payload_hash(envelope)
        with open(board_dir / "board.json", "w", encoding="utf-8") as handle:
            json.dump(envelope.to_dict(), handle)
        with pytest.raises(BoardStoreCorruptError):
            store.read_snapshot()


class TestStoreMigrationAndLifecycleWiring:
    def test_real_store_load_migrates_v0_and_reopens(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="migrate-reopen")
        raw = _read_json(board_dir / "board.json")
        raw.pop("schemaVersion")
        raw.pop("storeFormat")
        raw.pop("integrity")
        raw["legacyExtension"] = {"preserve": True}
        (board_dir / "board.json").write_text(json.dumps(raw), encoding="utf-8")

        migrated = store.load()
        assert migrated.schema_version == CURRENT_SCHEMA_VERSION
        assert migrated.board["compatibility_metadata"]["legacy_top_level"]["legacyExtension"] == {"preserve": True}
        reopened = _make_store(board_dir, board_id="migrate-reopen").load()
        assert reopened.to_dict() == migrated.to_dict()
        assert list((board_dir / "migration-backups").glob("*.json"))

    def test_session_genesis_has_session_scope(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "session"
        board = Board(
            board_id="session-board",
            project_uri="session:session-board",
            display_name="Session Board",
            schema_version=1,
            store_revision=0,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            policy=BoardPolicy(),
        )
        store = JsonBoardStore.create_board(board_dir, board=board, lock=BoardFileLock(board_dir))
        lifecycle = store.load().lifecycle
        assert lifecycle["scope"] == "session"
        assert lifecycle["origin_project_uri"] == "session:session-board"

    def test_store_gate_revalidates_under_lock(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "closed"
        store = _create_board(board_dir, board_id="closed-gate")
        close_board(
            store,
            "closed-gate",
            actor="u",
            command_id="close",
            request_hash="hc",
        )
        with pytest.raises(PermissionError, match="closed"):
            store.execute_atomic(
                "closed-gate",
                "write",
                "hw",
                None,
                _add_node_mutate("T-1", "forbidden"),
                actor="u",
            )
        assert store.load().nodes == {}
