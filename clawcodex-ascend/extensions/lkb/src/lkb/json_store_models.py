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

"""JSON-based board store — BoardEnvelope + payload hash chain + idempotency +
revision CAS + corruption recovery.

Spec §7.3 — BoardEnvelope on-disk format
Spec §5.1.1 — RevisionVector
Spec §5.10 — command idempotency + revision CAS
Spec §7.5 — atomic-write protocol (delegates to atomic_file)
Spec §7.6 — execute_atomic two-phase (lock → re-read → validate → mutate → write)
Spec §7.12 — corruption recovery

This module imports nothing from ToolContext or Task-v2 (spec §11.4 inv 12).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .error_codes import LkbErrorCode
from .graph_types import (
    Graph,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
    RevisionVector,
)
from .ir_hash import canonical_hash
from .refs import NodeRef
from .store_schema import CURRENT_SCHEMA_VERSION, HASH_ALGORITHM, STORE_FORMAT

# ── error types ───────────────────────────────────────────────────────


class BoardStoreCorruptError(Exception):
    """Raised when both board.json and board.json.bak are unreadable/invalid.

    Per spec §7.12, the store NEVER returns an empty Board when both files
    are corrupt — callers must get a clear error and handle recovery
    explicitly (quarantine, doctor, restore from history, etc.).
    """

    code = LkbErrorCode.BOARD_STORE_CORRUPT


class BoardRecoveryWarning(UserWarning):
    """Visible warning emitted after an automatic backup recovery."""


class StaleRevisionError(Exception):
    """Raised when expected_revision_vector CAS check fails (LKB-STORE-006/007).

    The caller's snapshot was taken at a revision that no longer matches
    the current state for at least one graph.  Callers should re-read and
    retry, or surface the conflict to the user.
    """

    code = LkbErrorCode.STALE_REVISION

    def __init__(
        self,
        board_id: str,
        expected: RevisionVector,
        actual: RevisionVector,
        *,
        reason: str = "",
    ) -> None:
        self.board_id = board_id
        self.expected = expected
        self.actual = actual
        msg = f"Stale revision for board {board_id!r}: expected {expected.to_dict()}, actual {actual.to_dict()}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class IdempotencyKeyReusedError(Exception):
    """Raised when a command_id is reused with a different request_hash
    (LKB-STORE-005).

    The same command_id was previously committed with a different command
    payload.  Callers must pick a new command_id if they intend a new
    command — reusing the id with different content is forbidden.
    """

    code = LkbErrorCode.IDEMPOTENCY_KEY_REUSED

    def __init__(
        self,
        command_id: str,
        stored_request_hash: str,
        new_request_hash: str,
    ) -> None:
        self.command_id = command_id
        self.stored_request_hash = stored_request_hash
        self.new_request_hash = new_request_hash
        super().__init__(
            f"Idempotency key {command_id!r} reused with different request hash: "
            f"stored={stored_request_hash}, new={new_request_hash}"
        )


class BoardSchemaTooNewError(Exception):
    """Raised when the on-disk schema_version is newer than this code knows.

    This prevents an older reader from silently corrupting a board written
    by a newer version (LKB-STORE-025 / forward-compatibility guard).
    """

    code = LkbErrorCode.BOARD_SCHEMA_TOO_NEW

    def __init__(self, board_id: str, on_disk: int, supported: int) -> None:
        self.board_id = board_id
        self.on_disk_version = on_disk
        self.supported_version = supported
        super().__init__(
            f"Board {board_id!r} has schema_version={on_disk}, but this build only supports up to {supported}"
        )


class BoardNotFoundError(Exception):
    """Raised when a board directory or board.json does not exist."""

    code = LkbErrorCode.BOARD_NOT_FOUND

    def __init__(self, board_id: str, path: Path) -> None:
        self.board_id = board_id
        self.path = path
        super().__init__(f"Board {board_id!r} not found at {path}")


class BoardTombstonedError(BoardNotFoundError):
    """Raised when a Tombstone forbids loading or recreating a purged Board."""

    code = LkbErrorCode.BOARD_TOMBSTONED

    def __init__(self, board_id: str, path: Path) -> None:
        self.tombstone_path = path
        super().__init__(board_id, path)


# ── constants ─────────────────────────────────────────────────────────


def now_iso() -> str:
    """Current UTC time in ISO-8601 for audit event timestamps (spec §6.10)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Keys in the top-level envelope, in canonical (sorted) order.
_ENVELOPE_TOP_KEYS = (
    "board",
    "claims",
    "edges",
    "events",
    "graphs",
    "historySegments",
    "integrity",
    "lifecycle",
    "nodes",
    "processedCommands",
    "schemaVersion",
    "storeFormat",
    "storeRevision",
    "validationRuns",
)


# ── BoardEnvelope ─────────────────────────────────────────────────────


@dataclass
class BoardEnvelope:
    """On-disk JSON envelope for a board (spec §7.3).

    The envelope holds *every* piece of board state — board metadata,
    graphs, nodes, edges, claims, validation runs,
    processed-command log, events, history segments, lifecycle state, and
    the integrity block (payload hash chain).

    All collection fields are plain dicts keyed by stable identifiers so
    that canonical JSON (sorted keys) produces a deterministic hash.
    """

    store_format: str = STORE_FORMAT
    schema_version: int = CURRENT_SCHEMA_VERSION
    store_revision: int = 0

    # Core board metadata (dict form — Board is reconstructed on load).
    board: dict[str, Any] = field(default_factory=dict)

    # Collections: {id -> record_dict}
    graphs: dict[str, dict[str, Any]] = field(default_factory=dict)
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    claims: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation_runs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Processed commands: {command_id -> {request_hash, decision, ...}}
    processed_commands: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Event log (append-only list).
    events: list[dict[str, Any]] = field(default_factory=list)

    # History segment references.
    history_segments: list[dict[str, Any]] = field(default_factory=list)

    # Lifecycle state (active | closed | archived | trashed).
    lifecycle: dict[str, Any] = field(default_factory=dict)

    # Integrity block.  payload_hash and previous_payload_hash form the
    # revision chain; algorithm names the hash function used.
    integrity: dict[str, Any] = field(default_factory=dict)

    # ── serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a stable-key-ordered dict.

        Keys match the on-disk camelCase convention (spec §7.3).
        """
        return {
            "storeFormat": self.store_format,
            "schemaVersion": self.schema_version,
            "storeRevision": self.store_revision,
            "board": self.board,
            "graphs": self.graphs,
            "nodes": self.nodes,
            "edges": self.edges,
            "claims": self.claims,
            "validationRuns": self.validation_runs,
            "processedCommands": self.processed_commands,
            "events": list(self.events),
            "historySegments": list(self.history_segments),
            "lifecycle": self.lifecycle,
            "integrity": self.integrity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoardEnvelope":
        """Deserialize from a dict (e.g. parsed JSON).

        This decoder is deliberately strict.  Migration code must first
        transform older envelopes into the current complete shape; silently
        defaulting missing fields here would make corruption look like valid
        empty state.
        """
        if not isinstance(data, dict):
            raise ValueError("envelope is not a JSON object")
        missing = set(_ENVELOPE_TOP_KEYS) - set(data)
        extra = set(data) - set(_ENVELOPE_TOP_KEYS)
        if missing:
            raise ValueError(f"envelope is missing required fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"envelope has unknown fields: {sorted(extra)}")
        return cls(
            store_format=data["storeFormat"],
            schema_version=data["schemaVersion"],
            store_revision=data["storeRevision"],
            board=copy.deepcopy(data["board"]),
            graphs=copy.deepcopy(data["graphs"]),
            nodes=copy.deepcopy(data["nodes"]),
            edges=copy.deepcopy(data["edges"]),
            claims=copy.deepcopy(data["claims"]),
            validation_runs=copy.deepcopy(data["validationRuns"]),
            processed_commands=copy.deepcopy(data["processedCommands"]),
            events=copy.deepcopy(data["events"]),
            history_segments=copy.deepcopy(data["historySegments"]),
            lifecycle=copy.deepcopy(data["lifecycle"]),
            integrity=copy.deepcopy(data["integrity"]),
        )

    # ── derived views ─────────────────────────────────────────────────

    def board_id(self) -> str:
        return str(self.board.get("board_id", ""))

    def current_revision_vector(self) -> RevisionVector:
        """Build a RevisionVector from graph revisions."""
        revs: dict[str, int] = {}
        for gid, g in self.graphs.items():
            revs[gid] = int(g.get("revision", 0))
        return RevisionVector(revisions=revs)

    def build_graph_snapshot(self) -> GraphSnapshot:
        """Reconstruct a GraphSnapshot from the envelope contents."""
        graphs: dict[str, Graph] = {}
        for gid, g in self.graphs.items():
            graphs[gid] = Graph(
                graph_id=str(g.get("graph_id", gid)),
                board_id=str(g.get("board_id", self.board_id())),
                graph_kind=str(g.get("graph_kind", "")),
                revision=int(g.get("revision", 0)),
                created_at=str(g.get("created_at", "")),
                updated_at=str(g.get("updated_at", "")),
                metadata=copy.deepcopy(g.get("plan", {})) if isinstance(g.get("plan"), dict) else {},
            )

        nodes: dict[NodeRef, GraphNode] = {}
        for _nid, n in self.nodes.items():
            ref_str = str(n.get("ref", ""))
            ref = NodeRef.from_str(ref_str)
            nodes[ref] = GraphNode(
                ref=ref,
                title=str(n.get("title", "")),
                state=n.get("state"),
                owner=n.get("owner"),
                revision=int(n.get("revision", 0)),
                payload=dict(n.get("payload", {})),
                created_at=str(n.get("created_at", "")),
                updated_at=str(n.get("updated_at", "")),
            )

        edges: dict[str, GraphEdge] = {}
        for eid, e in self.edges.items():
            src = NodeRef.from_str(str(e.get("source", "")))
            tgt = NodeRef.from_str(str(e.get("target", "")))
            edges[eid] = GraphEdge(
                edge_id=str(e.get("edge_id", eid)),
                graph=str(e.get("graph", "")),
                type=str(e.get("type", "")),
                source=src,
                target=tgt,
                revision=int(e.get("revision", 0)),
                payload=dict(e.get("payload", {})),
            )

        rv = self.current_revision_vector()
        board_dict = self.board if isinstance(self.board, dict) else {}
        policy_dict = board_dict.get("policy") if isinstance(board_dict.get("policy"), dict) else {}
        snap = GraphSnapshot(
            board_id=self.board_id(),
            store_revision=self.store_revision,
            graphs=graphs,
            nodes=nodes,
            edges=edges,
            revision_vector=rv,
            policy=copy.deepcopy(policy_dict),
        )
        return snap

    def clone(self) -> "BoardEnvelope":
        """Return a deep copy (mutations on clone don't touch original)."""
        return BoardEnvelope.from_dict(copy.deepcopy(self.to_dict()))


# ── payload hash chain ────────────────────────────────────────────────


def payload_hash(envelope: BoardEnvelope, *, algorithm: str = HASH_ALGORITHM) -> str:
    """Compute the payload hash of *envelope* (spec §7.3).

    The ``integrity`` block is stripped before hashing — the hash cannot
    include itself.  Returns ``"sha256:<hex>"`` (or the requested
    algorithm prefix).
    """
    data = envelope.to_dict()
    data.pop("integrity", None)
    return canonical_hash(data, algorithm=algorithm)


def set_payload_hash(
    envelope: BoardEnvelope,
    *,
    previous_hash: str | None = None,
    algorithm: str = HASH_ALGORITHM,
) -> str:
    """Compute and set ``integrity.payloadHash`` on *envelope* in place.

    If *previous_hash* is given, sets ``integrity.previousPayloadHash``
    to form the chain (spec §7.3).  Returns the computed payload hash.
    """
    # Strip integrity first, compute the hash, then set both fields.
    envelope.integrity = {}
    h = payload_hash(envelope, algorithm=algorithm)
    envelope.integrity = {
        "algorithm": algorithm,
        "payloadHash": h,
    }
    if previous_hash is not None:
        envelope.integrity["previousPayloadHash"] = previous_hash
    return h


# ── schema validation ─────────────────────────────────────────────────


def _is_nonnegative_int(value: Any) -> bool:
    """Return whether *value* is a JSON integer, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_envelope_schema(
    data: dict[str, Any],
    *,
    board_id: str | None = None,
) -> None:
    """Lightweight schema validation of a raw envelope dict.

    Checks store format, schema version range, required board fields,
    and board_id match (if *board_id* is provided).  Raises
    ``ValueError`` on any problem.
    """
    if not isinstance(data, dict):
        raise ValueError("envelope is not a JSON object")
    schema_ver = data.get("schemaVersion")
    # A real future schema is always reported before shape diagnostics, so
    # callers never mistake an upgrade requirement for malformed corruption.
    if isinstance(schema_ver, int) and not isinstance(schema_ver, bool) and schema_ver > CURRENT_SCHEMA_VERSION:
        board = data.get("board")
        board_name = board.get("board_id", "?") if isinstance(board, dict) else "?"
        raise BoardSchemaTooNewError(board_name, schema_ver, CURRENT_SCHEMA_VERSION)

    missing = set(_ENVELOPE_TOP_KEYS) - set(data)
    extra = set(data) - set(_ENVELOPE_TOP_KEYS)
    if missing:
        raise ValueError(f"envelope is missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"envelope has unknown fields: {sorted(extra)}")

    store_fmt = data.get("storeFormat")
    if store_fmt != STORE_FORMAT:
        raise ValueError(f"unexpected storeFormat: {store_fmt!r}")

    if not isinstance(schema_ver, int) or isinstance(schema_ver, bool) or schema_ver < 1:
        raise ValueError(f"invalid schemaVersion: {schema_ver!r}")

    board = data.get("board")
    if not isinstance(board, dict):
        raise ValueError("board field is missing or not an object")
    if not board.get("board_id"):
        raise ValueError("board.board_id is missing or empty")

    store_rev = data.get("storeRevision")
    if not _is_nonnegative_int(store_rev):
        raise ValueError(f"invalid storeRevision: {store_rev!r}")

    if board_id is not None and board.get("board_id") != board_id:
        raise ValueError(f"board_id mismatch: envelope has {board.get('board_id')!r}, expected {board_id!r}")

    # Collections are required and must be dictionaries.
    for key in (
        "graphs",
        "nodes",
        "edges",
        "claims",
        "validationRuns",
        "processedCommands",
    ):
        if not isinstance(data[key], dict):
            raise ValueError(f"{key} is not a dict")

    for key in ("events", "historySegments"):
        if not isinstance(data[key], list):
            raise ValueError(f"{key} is not a list")
        if not all(isinstance(item, dict) for item in data[key]):
            raise ValueError(f"{key} contains a non-object record")
    if not isinstance(data["lifecycle"], dict):
        raise ValueError("lifecycle is not an object")

    # Integrity block must be present and have payloadHash.
    integrity = data.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("integrity block is missing or not an object")
    if not integrity.get("payloadHash"):
        raise ValueError("integrity.payloadHash is missing or empty")
    if integrity.get("algorithm", HASH_ALGORITHM) != HASH_ALGORITHM:
        raise ValueError(f"unsupported integrity algorithm: {integrity.get('algorithm')!r}")
    if not isinstance(integrity["payloadHash"], str):
        raise ValueError("integrity.payloadHash is not a string")

    _assert_envelope_invariants(data)


def parse_ref(value: Any, *, location: str) -> NodeRef:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a NodeRef string")
    try:
        return NodeRef.from_str(value)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{location} is not a valid NodeRef: {value!r}") from exc


def _assert_envelope_invariants(data: dict[str, Any]) -> None:
    """Validate all graph-store invariants available in the v1 envelope."""
    board_id = data["board"]["board_id"]
    graphs = data["graphs"]
    nodes = data["nodes"]
    edges = data["edges"]

    for gid, graph in graphs.items():
        if not isinstance(gid, str) or not gid or not isinstance(graph, dict):
            raise ValueError(f"invalid graph record {gid!r}")
        if graph.get("graph_id") != gid:
            raise ValueError(f"graph key/id mismatch for {gid!r}")
        if graph.get("board_id") != board_id:
            raise ValueError(f"graph {gid!r} belongs to another board")
        if not isinstance(graph.get("graph_kind"), str) or not graph["graph_kind"]:
            raise ValueError(f"graph {gid!r} has invalid graph_kind")
        revision = graph.get("revision")
        if not _is_nonnegative_int(revision):
            raise ValueError(f"graph {gid!r} has invalid revision")

    refs: dict[NodeRef, str] = {}
    for node_id, node in nodes.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            raise ValueError(f"invalid node record {node_id!r}")
        ref = parse_ref(node.get("ref"), location=f"nodes[{node_id!r}].ref")
        if ref in refs:
            raise ValueError(f"duplicate NodeRef {ref.to_str()!r} in {refs[ref]!r} and {node_id!r}")
        if ref.graph not in graphs:
            raise ValueError(f"node {ref.to_str()!r} refers to missing graph")
        refs[ref] = node_id
        revision = node.get("revision", 0)
        if not _is_nonnegative_int(revision):
            raise ValueError(f"node {ref.to_str()!r} has invalid revision")
        if "payload" in node and not isinstance(node["payload"], dict):
            raise ValueError(f"node {ref.to_str()!r} payload is not an object")

    adjacency: dict[NodeRef, set[NodeRef]] = {ref: set() for ref in refs}
    for edge_id, edge in edges.items():
        if not isinstance(edge_id, str) or not isinstance(edge, dict):
            raise ValueError(f"invalid edge record {edge_id!r}")
        if edge.get("edge_id") != edge_id:
            raise ValueError(f"edge key/id mismatch for {edge_id!r}")
        graph_id = edge.get("graph")
        if graph_id not in graphs:
            raise ValueError(f"edge {edge_id!r} refers to missing graph")
        source = parse_ref(edge.get("source"), location=f"edges[{edge_id!r}].source")
        target = parse_ref(edge.get("target"), location=f"edges[{edge_id!r}].target")
        if source == target:
            raise ValueError(f"edge {edge_id!r} is a self-dependency")
        if source not in refs or target not in refs:
            raise ValueError(f"edge {edge_id!r} has a dangling endpoint")
        if source.graph != graph_id or target.graph != graph_id:
            raise ValueError(f"edge {edge_id!r} crosses its declared graph")
        revision = edge.get("revision", 0)
        if not _is_nonnegative_int(revision):
            raise ValueError(f"edge {edge_id!r} has invalid revision")
        if edge.get("type") == "depends_on":
            adjacency[source].add(target)

    unvisited, visiting, visited = 0, 1, 2
    colors: dict[NodeRef, int] = {}
    for start in adjacency:
        if colors.get(start, unvisited) != unvisited:
            continue
        colors[start] = visiting
        stack: list[tuple[NodeRef, Any]] = [(start, iter(adjacency[start]))]
        while stack:
            node, children = stack[-1]
            try:
                target = next(children)
            except StopIteration:
                colors[node] = visited
                stack.pop()
                continue
            target_color = colors.get(target, unvisited)
            if target_color == visiting:
                raise ValueError(f"dependency cycle includes {target.to_str()!r}")
            if target_color == unvisited:
                colors[target] = visiting
                stack.append((target, iter(adjacency[target])))

    active_claims: set[NodeRef] = set()
    for claim_id, claim in data["claims"].items():
        if not isinstance(claim_id, str) or not isinstance(claim, dict):
            raise ValueError(f"invalid claim record {claim_id!r}")
        if claim.get("claim_id", claim_id) != claim_id:
            raise ValueError(f"claim key/id mismatch for {claim_id!r}")
        task_ref = parse_ref(claim.get("task_ref"), location=f"claims[{claim_id!r}].task_ref")
        if task_ref not in refs:
            raise ValueError(f"claim {claim_id!r} refers to a missing task")
        # Spec §5.6 / issue #10: owner_ref is a NodeRef (plan:agent:<actor>),
        # never a bare actor string.  Validate the format and, for active
        # claims, that the owner_ref.id matches the task node's owner field
        # (Claim/Projection bidirectional consistency).
        owner_ref = parse_ref(claim.get("owner_ref"), location=f"claims[{claim_id!r}].owner_ref")
        if owner_ref.kind != "agent":
            raise ValueError(f"claim {claim_id!r} owner_ref {owner_ref.to_str()!r} is not an agent NodeRef")
        claim_revision = claim.get("claim_revision")
        if not _is_nonnegative_int(claim_revision):
            raise ValueError(f"claim {claim_id!r} has no real claim_revision")
        if claim.get("status", "active") == "active":
            if task_ref in active_claims:
                raise ValueError(f"multiple active claims for {task_ref.to_str()!r}")
            task_node = nodes[refs[task_ref]]
            if task_node.get("state") in {"blocked", "completed", "needs_recheck"}:
                raise ValueError(f"active claim {claim_id!r} targets non-claimable state {task_node.get('state')!r}")
            node_owner = task_node.get("owner")
            if not isinstance(node_owner, str) or not node_owner:
                raise ValueError(f"active claim {claim_id!r} targets task with no owner field")
            if owner_ref.id != node_owner:
                raise ValueError(
                    f"active claim {claim_id!r} owner_ref {owner_ref.to_str()!r} "
                    f"disagrees with task node owner {node_owner!r}"
                )
            active_claims.add(task_ref)

    for name in ("validationRuns", "processedCommands"):
        for record_id, record in data[name].items():
            if not isinstance(record_id, str) or not isinstance(record, dict):
                raise ValueError(f"invalid {name} record {record_id!r}")

    previous_segment_hash = ""
    segment_ids: set[str] = set()
    segment_files: set[str] = set()
    for index, segment in enumerate(data["historySegments"]):
        segment_id = segment.get("segmentId")
        filename = segment.get("file")
        digest = segment.get("sha256")
        if not isinstance(segment_id, str) or not segment_id:
            raise ValueError(f"historySegments[{index}] has no segmentId")
        if segment_id in segment_ids:
            raise ValueError(f"duplicate history segment ID {segment_id!r}")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"historySegments[{index}] has an unsafe file name")
        if filename in segment_files:
            raise ValueError(f"duplicate history segment file {filename!r}")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError(f"historySegments[{index}] has an invalid sha256")
        if segment.get("previousSegmentHash", "") != previous_segment_hash:
            raise ValueError(f"historySegments[{index}] breaks the segment hash chain")
        start = segment.get("startStoreRevision")
        end = segment.get("endStoreRevision")
        if not _is_nonnegative_int(start) or not _is_nonnegative_int(end) or end < start:
            raise ValueError(f"historySegments[{index}] has invalid revision bounds")
        for count_key in ("eventCount", "processedCommandCount", "validationRunCount"):
            count = segment.get(count_key)
            if not _is_nonnegative_int(count):
                raise ValueError(f"historySegments[{index}].{count_key} is invalid")
        for size_key in ("uncompressedBytes", "compressedBytes"):
            size = segment.get(size_key)
            if not _is_nonnegative_int(size):
                raise ValueError(f"historySegments[{index}].{size_key} is invalid")
        if not isinstance(segment.get("createdAt"), str) or not segment["createdAt"]:
            raise ValueError(f"historySegments[{index}].createdAt is invalid")
        segment_ids.add(segment_id)
        segment_files.add(filename)
        previous_segment_hash = digest

    for command_id, record in data["processedCommands"].items():
        segment_id = record.get("history_segment_id")
        if segment_id is not None and segment_id not in segment_ids:
            raise ValueError(f"processed command {command_id!r} refers to unknown history segment {segment_id!r}")


def verify_payload_hash(data: dict[str, Any]) -> bool:
    """Return True if the payload hash in *data* matches the content.

    Strips the integrity block, hashes the rest, and compares.
    """
    integrity = data.get("integrity", {})
    expected = integrity.get("payloadHash", "")
    if not expected:
        return False

    payload = {k: v for k, v in data.items() if k != "integrity"}
    actual = canonical_hash(payload)
    return actual == expected


# Compatibility aliases for code that imported the former internal helpers.
_now_iso = now_iso
_parse_ref = parse_ref
_validate_envelope_schema = validate_envelope_schema
_verify_payload_hash = verify_payload_hash
