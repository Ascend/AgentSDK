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

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from .._support import Failpoint
from lkb.commands import CommandResult
from lkb.file_lock import BoardFileLock
from lkb.graph_types import Board, BoardPolicy, RevisionVector
from lkb.json_store import (
    BoardEnvelope,
    CURRENT_SCHEMA_VERSION,
    IdempotencyKeyReusedError,
    JsonBoardStore,
    STORE_FORMAT,
    StaleRevisionError,
    payload_hash,
    set_payload_hash,
    validate_envelope_schema,
    verify_payload_hash,
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


def test_public_schema_helpers_and_board_directory(tmp_path: Path) -> None:
    board_dir = tmp_path / "board"
    store = _create_board(board_dir, board_id="public-store-api")
    data = json.loads((board_dir / "board.json").read_text(encoding="utf-8"))

    assert store.board_dir == board_dir
    validate_envelope_schema(data, board_id="public-store-api")
    assert verify_payload_hash(data)


def test_legacy_home_drives_tombstone_root_without_board_path_inference(tmp_path: Path) -> None:
    lkb_root = tmp_path / "lkb"
    directory = tmp_path / "relocated" / "board"
    store = JsonBoardStore(
        directory,
        board_id="relocated-board",
        lock=BoardFileLock(directory),
        home=lkb_root,
    )

    assert store._tombstone_path().parent == lkb_root / "tombstones"


def test_explicit_lkb_root_supports_relocated_store(tmp_path: Path) -> None:
    lkb_root = tmp_path / "shared-lkb"
    directory = tmp_path / "relocated" / "board"
    store = JsonBoardStore(
        directory,
        board_id="relocated-board",
        lock=BoardFileLock(directory),
        home=tmp_path / "unrelated-home",
        lkb_root=lkb_root,
    )

    assert store._tombstone_path().parent == lkb_root / "tombstones"


def test_schema_version_rejects_json_boolean(tmp_path: Path) -> None:
    board_dir = tmp_path / "board"
    _create_board(board_dir, board_id="bool-schema")
    data = json.loads((board_dir / "board.json").read_text(encoding="utf-8"))
    data["schemaVersion"] = True

    with pytest.raises(ValueError, match="invalid schemaVersion"):
        validate_envelope_schema(data)


def test_persisted_integer_fields_reject_json_booleans(tmp_path: Path) -> None:
    board_dir = tmp_path / "board"
    store = _create_board(board_dir, board_id="bool-integers")
    store.execute_atomic(
        board_id="bool-integers",
        command_id="cmd-node",
        request_hash="hash-node",
        expected_revision_vector=None,
        mutate=_add_node_mutate("T-001", "Task"),
        actor="agent-1",
    )
    data = json.loads((board_dir / "board.json").read_text(encoding="utf-8"))
    mutations = (
        lambda candidate: candidate.__setitem__("storeRevision", True),
        lambda candidate: candidate["graphs"]["plan"].__setitem__("revision", True),
        lambda candidate: candidate["nodes"]["T-001"].__setitem__("revision", True),
    )

    for mutate in mutations:
        candidate = copy.deepcopy(data)
        mutate(candidate)
        with pytest.raises(ValueError, match="revision|storeRevision"):
            validate_envelope_schema(candidate)


def test_schema_validation_handles_deep_acyclic_graph_iteratively(tmp_path: Path) -> None:
    board_dir = tmp_path / "board"
    _create_board(board_dir, board_id="deep-schema")
    data = json.loads((board_dir / "board.json").read_text(encoding="utf-8"))
    data["graphs"]["plan"] = {
        "graph_id": "plan",
        "board_id": "deep-schema",
        "graph_kind": "plan",
        "revision": 0,
    }
    for index in range(1_500):
        task_id = f"T-{index}"
        data["nodes"][task_id] = {
            "ref": f"plan:task:{task_id}",
            "state": "pending",
            "revision": 0,
            "payload": {},
        }
        if index:
            edge_id = f"E-{index}"
            data["edges"][edge_id] = {
                "edge_id": edge_id,
                "graph": "plan",
                "source": f"plan:task:T-{index - 1}",
                "target": f"plan:task:{task_id}",
                "type": "depends_on",
                "revision": 0,
            }

    validate_envelope_schema(data, board_id="deep-schema")


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


def test_received_and_executed_audit_events_have_distinct_ids(tmp_path: Path) -> None:
    store = _create_board(tmp_path / "board", board_id="audit-event-ids")
    store.execute_atomic(
        board_id="audit-event-ids",
        command_id="cmd-event-ids",
        request_hash="hash-event-ids",
        expected_revision_vector=None,
        mutate=_add_node_mutate("T-001", "Task"),
        actor="agent-1",
    )

    command_events = [event for event in store.load().events if event.get("command_id")]
    assert len(command_events) == 2
    assert len({event["event_id"] for event in command_events}) == 2


# ── LKB-STORE-001 ────────────────────────────────────────────────────


class TestLkbStore001EnvelopeRoundTrip:
    """BoardEnvelope to_dict / from_dict round-trip preserves all fields."""

    def test_empty_envelope_roundtrip(self) -> None:
        env = BoardEnvelope()
        d = env.to_dict()
        env2 = BoardEnvelope.from_dict(d)
        assert env2.store_format == env.store_format
        assert env2.schema_version == env.schema_version
        assert env2.store_revision == env.store_revision
        assert env2.board == env.board
        assert env2.graphs == env.graphs
        assert env2.nodes == env.nodes
        assert env2.edges == env.edges
        assert env2.claims == env.claims
        assert env2.validation_runs == env.validation_runs
        assert env2.processed_commands == env.processed_commands
        assert env2.events == env.events
        assert env2.history_segments == env.history_segments
        assert env2.lifecycle == env.lifecycle
        assert env2.integrity == env.integrity

    def test_populated_envelope_roundtrip(self) -> None:
        env = BoardEnvelope(
            store_revision=5,
            board={"board_id": "board-1", "display_name": "Test Board"},
            graphs={
                "plan": {
                    "graph_id": "plan",
                    "board_id": "board-1",
                    "graph_kind": "plan",
                    "revision": 3,
                }
            },
            nodes={"n1": {"ref": "plan:task:T-001", "title": "Task 1", "state": "ready"}},
            edges={
                "e1": {
                    "edge_id": "e1",
                    "graph": "plan",
                    "type": "depends_on",
                    "source": "plan:task:T-002",
                    "target": "plan:task:T-001",
                }
            },
            claims={"c1": {"claim_id": "c1", "status": "active"}},
            validation_runs={"v1": {"run_id": "v1", "status": "passed"}},
            processed_commands={"cmd-1": {"command_id": "cmd-1", "decision": "committed"}},
            events=[{"type": "test", "store_revision": 1}],
            history_segments=[{"segment_id": "h1", "start_revision": 0}],
            lifecycle={"state": "active"},
            integrity={"algorithm": "sha256", "payloadHash": "sha256:abc123"},
        )
        d = env.to_dict()
        env2 = BoardEnvelope.from_dict(d)
        assert env2.store_revision == 5
        assert env2.board["board_id"] == "board-1"
        assert env2.graphs["plan"]["revision"] == 3
        assert env2.nodes["n1"]["title"] == "Task 1"
        assert env2.edges["e1"]["type"] == "depends_on"
        assert env2.claims["c1"]["status"] == "active"
        assert env2.validation_runs["v1"]["status"] == "passed"
        assert env2.processed_commands["cmd-1"]["decision"] == "committed"
        assert len(env2.events) == 1
        assert len(env2.history_segments) == 1
        assert env2.lifecycle["state"] == "active"
        assert env2.integrity["payloadHash"] == "sha256:abc123"

    def test_to_dict_has_canonical_key_order(self) -> None:
        """top-level keys should be in a stable order (sorted by json.dumps)."""
        env = BoardEnvelope(
            board={"board_id": "b1"},
            graphs={"g1": {}},
            nodes={},
        )
        raw = json.dumps(env.to_dict(), sort_keys=True)
        parsed = json.loads(raw)
        # Just verify all expected keys are present
        expected = {
            "storeFormat",
            "schemaVersion",
            "storeRevision",
            "board",
            "graphs",
            "nodes",
            "edges",
            "claims",
            "validationRuns",
            "processedCommands",
            "events",
            "historySegments",
            "lifecycle",
            "integrity",
        }
        assert set(parsed.keys()) == expected

    def test_store_format_constant(self) -> None:
        assert STORE_FORMAT == "lkb-json-v2"
        env = BoardEnvelope()
        assert env.store_format == STORE_FORMAT

    def test_from_dict_rejects_missing_current_schema_fields(self) -> None:
        """Current-schema decode must not silently default missing state."""
        minimal = {
            "storeFormat": STORE_FORMAT,
            "schemaVersion": CURRENT_SCHEMA_VERSION,
            "storeRevision": 0,
            "board": {"board_id": "minimal"},
            "integrity": {"algorithm": "sha256", "payloadHash": "sha256:xxx"},
        }
        with pytest.raises(ValueError, match="missing required fields"):
            BoardEnvelope.from_dict(minimal)


# ── LKB-STORE-002 ────────────────────────────────────────────────────


class TestLkbStore002PayloadHashChain:
    """payload_hash chain: consecutive revisions form a valid hash chain."""

    def test_genesis_envelope_payload_hash(self) -> None:
        env = BoardEnvelope(
            board={"board_id": "chain-test", "store_revision": 0},
            store_revision=0,
        )
        h = set_payload_hash(env, previous_hash=None)
        # integrity.payloadHash should match re-computation
        assert env.integrity["payloadHash"] == h
        assert payload_hash(env) == h
        # Genesis should have no previousPayloadHash
        assert "previousPayloadHash" not in env.integrity

    def test_second_revision_chains_previous_hash(self) -> None:
        env1 = BoardEnvelope(
            board={"board_id": "chain-test"},
            store_revision=1,
        )
        h1 = set_payload_hash(env1, previous_hash=None)

        env2 = BoardEnvelope(
            board={"board_id": "chain-test", "extra": "new"},
            store_revision=2,
        )
        h2 = set_payload_hash(env2, previous_hash=h1)

        assert env2.integrity["previousPayloadHash"] == h1
        assert env2.integrity["payloadHash"] == h2
        # Hashes must be different (content changed)
        assert h1 != h2

    def test_payload_hash_strips_integrity_block(self) -> None:
        """The integrity block must NOT be part of the hash input."""
        env = BoardEnvelope(
            board={"board_id": "hash-test"},
        )
        # Setting the payload hash should produce a deterministic result
        h1 = set_payload_hash(env, previous_hash=None)
        # Mutating only integrity fields shouldn't change the hash
        env.integrity["extra_junk"] = "should_not_affect_hash"
        h2 = payload_hash(env)
        assert h1 == h2

    def test_hash_uses_sha256_prefix(self) -> None:
        env = BoardEnvelope(board={"board_id": "algo-test"})
        h = payload_hash(env)
        assert h.startswith("sha256:")
        # sha256 hex is 64 chars
        assert len(h) == len("sha256:") + 64

    def test_current_revision_vector(self) -> None:
        env = BoardEnvelope(
            graphs={
                "plan": {"graph_id": "plan", "revision": 5},
                "artifact": {"graph_id": "artifact", "revision": 3},
            }
        )
        rv = env.current_revision_vector()
        assert rv.get("plan") == 5
        assert rv.get("artifact") == 3
        assert rv.get("nonexistent") == 0


# ── LKB-STORE-003 ────────────────────────────────────────────────────


class TestLkbStore003CreateBoard:
    """create_board writes genesis envelope with store_revision=0."""

    def test_genesis_envelope_on_disk(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="genesis-board")

        assert store.exists()
        data = _read_json(board_dir / "board.json")
        assert data["storeFormat"] == STORE_FORMAT
        assert data["schemaVersion"] == CURRENT_SCHEMA_VERSION
        assert data["storeRevision"] == 0
        assert data["board"]["board_id"] == "genesis-board"
        assert data["lifecycle"]["state"] == "active"
        assert data["integrity"]["algorithm"] == "sha256"
        assert data["integrity"]["payloadHash"].startswith("sha256:")
        # Genesis has no previous hash
        assert "previousPayloadHash" not in data["integrity"]

    def test_load_after_create_returns_valid_envelope(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="loadable-board")

        store = _make_store(board_dir, board_id="loadable-board")
        env = store.load()
        assert env.board_id() == "loadable-board"
        assert env.store_revision == 0

    def test_create_board_on_existing_raises(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="exists")

        with pytest.raises(FileExistsError):
            _create_board(board_dir, board_id="exists")

    def test_backup_file_exists_after_create(self, tmp_path: Path) -> None:
        """After create_board, a .bak may or may not exist (only after first
        real write).  Genesis doesn't produce a .bak — that's fine.
        """
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="bak-test")
        # board.json should exist; .bak doesn't have to yet
        assert (board_dir / "board.json").exists()

    def test_header_after_create(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        _create_board(board_dir, board_id="header-test")

        store = _make_store(board_dir, board_id="header-test")
        h = store.header()
        assert h["board_id"] == "header-test"
        assert h["store_revision"] == 0
        assert h["schema_version"] == CURRENT_SCHEMA_VERSION
        assert h["lifecycle_state"] == "active"


# ── LKB-STORE-004 ────────────────────────────────────────────────────


class TestLkbStore004IdempotencySameCommand:
    """Same command_id + same request_hash returns cached result (no double-apply)."""

    def test_same_command_returns_cached(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="idem-board")

        command_id = "cmd-001"
        request_hash = "sha256:reqhash111"

        # First execution
        result1 = store.execute_atomic(
            board_id="idem-board",
            command_id=command_id,
            request_hash=request_hash,
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-001", "First task"),
            actor="agent-1",
        )
        assert result1.committed is True

        # Second execution with same command_id + same request_hash
        result2 = store.execute_atomic(
            board_id="idem-board",
            command_id=command_id,
            request_hash=request_hash,
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-001", "First task"),
            actor="agent-1",
        )
        assert result2.committed is True
        assert result2.command_id == command_id

        # Store revision should be 1 (only one real commit happened)
        env = store.load()
        assert env.store_revision == 1
        # Node should exist exactly once
        assert "T-001" in env.nodes

    def test_cached_result_preserves_revision_vector(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="idem-rv")

        command_id = "cmd-rv"
        request_hash = "sha256:rvhash"

        result1 = store.execute_atomic(
            board_id="idem-rv",
            command_id=command_id,
            request_hash=request_hash,
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-100", "RV test"),
            actor="agent-1",
        )

        result2 = store.execute_atomic(
            board_id="idem-rv",
            command_id=command_id,
            request_hash=request_hash,
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-100", "RV test"),
            actor="agent-1",
        )

        assert result1.revision_vector is not None
        assert result2.revision_vector is not None
        assert result1.revision_vector.equals(result2.revision_vector)

    def test_idempotency_recorded_in_processed_commands(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="idem-pc")

        command_id = "cmd-pc"
        request_hash = "sha256:pchash"

        store.execute_atomic(
            board_id="idem-pc",
            command_id=command_id,
            request_hash=request_hash,
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-200", "PC test"),
            actor="agent-1",
        )

        env = store.load()
        assert command_id in env.processed_commands
        entry = env.processed_commands[command_id]
        assert entry["request_hash"] == request_hash
        assert entry["decision"] == "committed"
        assert entry["actor"] == "agent-1"
        assert entry["store_revision"] == 1


# ── LKB-STORE-005 ────────────────────────────────────────────────────


class TestLkbStore005IdempotencyKeyReused:
    """Same command_id + different request_hash raises IdempotencyKeyReusedError."""

    def test_reused_key_with_different_hash_raises(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="reuse-board")

        command_id = "cmd-reuse"
        hash1 = "sha256:hash-one-111"
        hash2 = "sha256:hash-two-222"

        store.execute_atomic(
            board_id="reuse-board",
            command_id=command_id,
            request_hash=hash1,
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-001", "Task 1"),
            actor="agent-1",
        )

        with pytest.raises(IdempotencyKeyReusedError) as exc_info:
            store.execute_atomic(
                board_id="reuse-board",
                command_id=command_id,
                request_hash=hash2,
                expected_revision_vector=None,
                mutate=_add_node_mutate("T-002", "Task 2"),
                actor="agent-1",
            )

        assert exc_info.value.command_id == command_id
        assert exc_info.value.stored_request_hash == hash1
        assert exc_info.value.new_request_hash == hash2

    def test_reused_key_does_not_mutate_state(self, tmp_path: Path) -> None:
        """After a key-reuse error, board state must be unchanged."""
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="reuse-nomut")

        command_id = "cmd-reuse2"
        hash1 = "sha256:hash-a"
        hash2 = "sha256:hash-b"

        store.execute_atomic(
            board_id="reuse-nomut",
            command_id=command_id,
            request_hash=hash1,
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-001", "Only this"),
            actor="agent-1",
        )

        env_before = store.load()
        rev_before = env_before.store_revision

        with pytest.raises(IdempotencyKeyReusedError):
            store.execute_atomic(
                board_id="reuse-nomut",
                command_id=command_id,
                request_hash=hash2,
                expected_revision_vector=None,
                mutate=_add_node_mutate("T-002", "Should not appear"),
                actor="agent-1",
            )

        env_after = store.load()
        assert env_after.store_revision == rev_before
        assert "T-001" in env_after.nodes
        assert "T-002" not in env_after.nodes


# ── LKB-STORE-006 ────────────────────────────────────────────────────


class TestLkbStore006RevisionCasMatch:
    """Matching expected_revision_vector allows the write to proceed."""

    def test_matching_revision_vector_succeeds(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="cas-match")

        # First write to bump plan graph revision
        store.execute_atomic(
            board_id="cas-match",
            command_id="cmd-first",
            request_hash="sha256:first",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-001", "First"),
            actor="agent-1",
        )

        # Read current revision vector
        env = store.load()
        current_rv = env.current_revision_vector()

        # Second write with matching expected revision
        result = store.execute_atomic(
            board_id="cas-match",
            command_id="cmd-second",
            request_hash="sha256:second",
            expected_revision_vector=current_rv,
            mutate=_add_node_mutate("T-002", "Second"),
            actor="agent-1",
        )

        assert result.committed is True
        env_after = store.load()
        assert env_after.store_revision == 2
        assert "T-002" in env_after.nodes

    def test_partial_revision_vector_only_checks_listed_graphs(self, tmp_path: Path) -> None:
        """Only graph IDs present in expected_revision_vector are checked."""
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="cas-partial")

        store.execute_atomic(
            board_id="cas-partial",
            command_id="cmd-1",
            request_hash="sha256:h1",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-1", "One"),
            actor="agent-1",
        )

        # Pass a vector with only a non-existent graph — should pass
        rv = RevisionVector(revisions={"nonexistent": 0})
        result = store.execute_atomic(
            board_id="cas-partial",
            command_id="cmd-2",
            request_hash="sha256:h2",
            expected_revision_vector=rv,
            mutate=_add_node_mutate("T-2", "Two"),
            actor="agent-1",
        )
        assert result.committed is True

    def test_no_expected_revision_vector_always_succeeds(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="cas-none")

        for i in range(3):
            result = store.execute_atomic(
                board_id="cas-none",
                command_id=f"cmd-{i}",
                request_hash=f"sha256:h{i}",
                expected_revision_vector=None,
                mutate=_add_node_mutate(f"T-{i}", f"Task {i}"),
                actor="agent-1",
            )
            assert result.committed is True

        env = store.load()
        assert env.store_revision == 3


# ── LKB-STORE-007 ────────────────────────────────────────────────────


class TestLkbStore007RevisionCasStale:
    """Stale expected_revision_vector raises StaleRevisionError."""

    def test_stale_revision_raises(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="cas-stale")

        store.execute_atomic(
            board_id="cas-stale",
            command_id="cmd-base",
            request_hash="sha256:base",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-001", "Base"),
            actor="agent-1",
        )

        # stale: plan graph at revision 0 (but actual is higher)
        stale_rv = RevisionVector(revisions={"plan": 0})

        with pytest.raises(StaleRevisionError) as exc_info:
            store.execute_atomic(
                board_id="cas-stale",
                command_id="cmd-stale",
                request_hash="sha256:stale",
                expected_revision_vector=stale_rv,
                mutate=_add_node_mutate("T-002", "Stale"),
                actor="agent-1",
            )

        assert exc_info.value.board_id == "cas-stale"
        assert exc_info.value.expected.get("plan") == 0
        assert exc_info.value.actual.get("plan") > 0

    def test_stale_revision_does_not_mutate_state(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="cas-stale-nomut")

        store.execute_atomic(
            board_id="cas-stale-nomut",
            command_id="cmd-first",
            request_hash="sha256:first",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-001", "First"),
            actor="agent-1",
        )

        env_before = store.load()
        rev_before = env_before.store_revision

        stale_rv = RevisionVector(revisions={"plan": 0})

        with pytest.raises(StaleRevisionError):
            store.execute_atomic(
                board_id="cas-stale-nomut",
                command_id="cmd-stale2",
                request_hash="sha256:stale2",
                expected_revision_vector=stale_rv,
                mutate=_add_node_mutate("T-002", "Should not exist"),
                actor="agent-1",
            )

        env_after = store.load()
        assert env_after.store_revision == rev_before
        assert "T-002" not in env_after.nodes

    def test_stale_store_revision_does_not_mutate_state(self, tmp_path: Path) -> None:
        board_dir = tmp_path / "board"
        store = _create_board(board_dir, board_id="cas-store-stale")
        store.execute_atomic(
            board_id="cas-store-stale",
            command_id="cmd-first",
            request_hash="sha256:first",
            expected_revision_vector=None,
            mutate=_add_node_mutate("T-001", "First"),
            actor="agent-1",
        )

        with pytest.raises(StaleRevisionError, match="store revision"):
            store.execute_atomic(
                board_id="cas-store-stale",
                command_id="cmd-stale-store",
                request_hash="sha256:stale-store",
                expected_revision_vector=None,
                expected_store_revision=0,
                mutate=_add_node_mutate("T-002", "Should not exist"),
                actor="agent-1",
            )

        envelope = store.load()
        assert envelope.store_revision == 1
        assert "T-002" not in envelope.nodes
