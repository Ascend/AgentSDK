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
# ruff: noqa: F401

"""Core graph data structures for the LKB Plan Graph.

Defines Board, Graph, GraphNode, GraphEdge, Claim, RevisionVector,
BoardPolicy, GraphSnapshot, and PlanSnapshot.  These are the
kernel-pure domain types for Phase 1 of the Plan Graph MVP — they do
not import ToolContext or any Task-v2 machinery.

Spec §5.1 — Board + BoardPolicy
Spec §5.1.1 — RevisionVector
Spec §5.4 — GraphNode
Spec §5.5 — GraphEdge
Spec §5.6 — Claim
Spec §5.9 — GraphSnapshot + PlanSnapshot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .ir_hash import canonical_hash
from .refs import NodeRef, plan_task_ref


class _FrozenDict(dict[Any, Any]):
    """JSON-compatible dict that cannot be changed after construction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        dict.__init__(self, *args, **kwargs)

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("snapshot mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> "_FrozenDict":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenDict":
        del memo
        return self


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_thaw_json(item) for item in value), key=str)
    return value


# ── revision vector ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RevisionVector:
    """A logical clock vector mapping graph_id → revision.

    Used for causal ordering across multiple graphs on the same board.
    Spec §5.1.1.
    """

    revisions: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        revisions: dict[str, int] = {}
        for graph_id, revision in self.revisions.items():
            if not isinstance(graph_id, str) or not graph_id:
                raise ValueError("revision graph ids must be non-empty strings")
            if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
                raise ValueError("graph revisions must be non-negative integers")
            revisions[graph_id] = revision
        object.__setattr__(self, "revisions", _FrozenDict(revisions))

    def get(self, graph_id: str) -> int:
        """Return the revision for *graph_id* (0 if not present)."""
        return self.revisions.get(graph_id, 0)

    def with_update(self, graph_id: str, revision: int) -> "RevisionVector":
        """Return a new vector with *graph_id* advanced to *revision*."""
        new = dict(self.revisions)
        new[graph_id] = revision
        return RevisionVector(revisions=new)

    def equals(self, other: object) -> bool:
        """Element-wise equality check."""
        if not isinstance(other, RevisionVector):
            return False
        return self.revisions == other.revisions

    def to_dict(self) -> dict[str, int]:
        return dict(self.revisions)


# ── board policy ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class BoardPolicy:
    """Policy flags governing how a board evolves.

    Spec §5.1 — these are the Phase 1 knobs; more fields will be added
    in later phases.
    """

    single_active_task_per_agent: bool = False
    force_override_roles: tuple[str, ...] | bool = ()
    invalidation_mode: str = "cascade"  # cascade | direct | off

    def __post_init__(self) -> None:
        roles = self.force_override_roles
        if isinstance(roles, bool):
            # The old boolean meant "override enabled without role scoping".
            normalized = ("*",) if roles else ()
        else:
            normalized = tuple(dict.fromkeys(str(role) for role in roles if str(role)))
        object.__setattr__(self, "force_override_roles", normalized)

    def allows_force_override(self, role: str) -> bool:
        roles = self.force_override_roles
        return "*" in roles or role in roles

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON shape used by every persistence adapter."""
        return {
            "single_active_task_per_agent": self.single_active_task_per_agent,
            # A JSON array is deliberately different from the legacy boolean.
            "force_override_roles": list(self.force_override_roles),
            "invalidation_mode": self.invalidation_mode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "BoardPolicy":
        """Load both the role-scoped policy and its historical boolean form."""
        raw = dict(value or {})
        roles = raw.get("force_override_roles", ())
        if not isinstance(roles, (bool, list, tuple)):
            raise ValueError("force_override_roles must be a boolean or a JSON array")
        return cls(
            single_active_task_per_agent=bool(raw.get("single_active_task_per_agent", False)),
            force_override_roles=roles,
            invalidation_mode=str(raw.get("invalidation_mode", "cascade")),
        )


# ── board / graph record types ───────────────────────────────────────


@dataclass(frozen=True)
class Board:
    """Top-level board metadata.

    A board owns multiple Plan graphs and a shared policy.  Spec §5.1.
    """

    board_id: str
    project_uri: str
    display_name: str = ""
    schema_version: int = 1
    store_revision: int = 0
    created_at: str = ""
    updated_at: str = ""
    policy: BoardPolicy = field(default_factory=BoardPolicy)


@dataclass(frozen=True)
class Graph:
    """Metadata for a single graph on a board.

    Spec §5.1 — current runtime graphs use kind ``"plan"`` and each has
    its own revision counter.  The string remains open only so existing
    Board files can still be loaded and migrated.
    """

    graph_id: str
    board_id: str
    graph_kind: str
    revision: int = 0
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Reuse NodeRef validation without conflating graph identity with kind.
        NodeRef(self.graph_id, "_graph", "_id")
        NodeRef(self.graph_kind, "_kind", "_id")
        object.__setattr__(self, "metadata", _freeze_json(dict(self.metadata or {})))


# ── nodes / edges ────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphNode:
    """A node in a plan graph.

    Spec §5.4.  ``state`` and ``owner`` remain optional at the storage
    boundary so older Board files can be loaded before migration.
    """

    ref: NodeRef
    title: str
    state: str | None = None
    owner: str | None = None
    revision: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class GraphEdge:
    """A directed edge between two graph nodes.

    The current runtime emits ``depends_on`` edges.  The type remains a
    string at the storage boundary for backward-compatible loading.

    Canonical direction convention (spec §5.5):
      * Store ``depends_on`` edges pointing from dependent → prerequisite.
      * ``A blocks B`` is the inverse of ``B depends_on A`` — use
        :meth:`project` to flip the direction on demand.
    """

    edge_id: str
    graph: str
    type: str
    source: NodeRef
    target: NodeRef
    revision: int = 0
    payload: dict[str, Any] = field(default_factory=dict)

    def project(self, new_type: str) -> "GraphEdge":
        """Return a new edge with source/target swapped and *new_type*.

        This lets callers project ``depends_on`` into ``blocks`` (or
        vice versa) without mutating the canonical store.
        """
        return GraphEdge(
            edge_id=self.edge_id,
            graph=self.graph,
            type=new_type,
            source=self.target,
            target=self.source,
            revision=self.revision,
            payload=dict(self.payload),
        )


# ── claim ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Claim:
    """An agent's claim on a task.

    Spec §5.6 — claims track which agent owns which task and for how
    long.  The status cycle is: active → released | completed |
    overridden.
    """

    task_ref: NodeRef
    owner_ref: NodeRef
    claim_id: str
    claimed_at: str = ""
    claim_revision: int = 0
    status: str = "active"  # active | released | completed | overridden
    released_at: str = ""
    reason: str = ""


# ── snapshots ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphSnapshot:
    """A generic, immutable snapshot of a board's graphs.

    Contains every graph, node, and edge on the board together with a
    revision vector and a content hash.  Readiness, cycle detection,
    and other plan-specific computations are NOT part of this generic
    snapshot — they live on :class:`PlanSnapshot`.

    Spec §5.9, §7.6.
    """

    board_id: str
    store_revision: int = 0
    graphs: dict[str, Graph] = field(default_factory=dict)
    nodes: dict[NodeRef, GraphNode] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)
    revision_vector: RevisionVector = field(default_factory=RevisionVector)
    hash: str = ""
    # Board-level policy (single_active_task_per_agent,
    # invalidation_mode, ...). Carried on the
    # snapshot so lock-free validators (spec §7.6) can read policy without
    # the Board File Lock - the envelope is the only other place policy
    # lives. Policy is board-level metadata and is deliberately
    # *excluded* from :meth:`payload_dict` so policy-only
    # changes do not invalidate the graph-content hash (they still bump
    # ``store_revision`` via the envelope).
    policy: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        graphs = dict(self.graphs)
        nodes: dict[NodeRef, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        for ref, node in self.nodes.items():
            if not isinstance(ref, NodeRef) or not isinstance(node, GraphNode):
                raise TypeError("GraphSnapshot.nodes must map NodeRef to GraphNode")
            nodes[ref] = GraphNode(
                ref=node.ref,
                title=node.title,
                state=node.state,
                owner=node.owner,
                revision=node.revision,
                payload=_freeze_json(dict(node.payload)),
                created_at=node.created_at,
                updated_at=node.updated_at,
            )
        for edge_id, edge in self.edges.items():
            if not isinstance(edge_id, str) or not isinstance(edge, GraphEdge):
                raise TypeError("GraphSnapshot.edges must map strings to GraphEdge")
            edges[edge_id] = GraphEdge(
                edge_id=edge.edge_id,
                graph=edge.graph,
                type=edge.type,
                source=edge.source,
                target=edge.target,
                revision=edge.revision,
                payload=_freeze_json(dict(edge.payload)),
            )
        object.__setattr__(self, "graphs", _FrozenDict(graphs))
        object.__setattr__(self, "nodes", _FrozenDict(nodes))
        object.__setattr__(self, "edges", _FrozenDict(edges))
        object.__setattr__(self, "policy", _FrozenDict(dict(self.policy or {})))

        computed = self.payload_hash()
        if self.hash and self.hash != computed:
            raise ValueError(f"GraphSnapshot hash mismatch: supplied {self.hash!r}, computed {computed!r}")
        object.__setattr__(self, "hash", computed)

    def graph_kind_for_ref(self, ref: NodeRef) -> str | None:
        """Resolve a node's graph kind through its graph *identity*."""
        graph = self.graphs.get(ref.graph)
        return graph.graph_kind if graph is not None else None

    def payload_dict(self) -> dict[str, Any]:
        """Return a canonical dict suitable for hashing (hash field stripped)."""
        return {
            "board_id": self.board_id,
            "graphs": {
                gid: {
                    "graph_id": g.graph_id,
                    "board_id": g.board_id,
                    "graph_kind": g.graph_kind,
                    "revision": g.revision,
                    "created_at": g.created_at,
                    "updated_at": g.updated_at,
                    "metadata": _thaw_json(g.metadata),
                }
                for gid, g in sorted(self.graphs.items())
            },
            "nodes": {
                ref.to_str(): {
                    "ref": ref.to_str(),
                    "title": n.title,
                    "state": n.state,
                    "owner": n.owner,
                    "revision": n.revision,
                    "payload": _thaw_json(n.payload),
                    "created_at": n.created_at,
                    "updated_at": n.updated_at,
                }
                for ref, n in sorted(self.nodes.items(), key=lambda kv: kv[0].to_str())
            },
            "edges": {
                eid: {
                    "edge_id": e.edge_id,
                    "graph": e.graph,
                    "type": e.type,
                    "source": e.source.to_str(),
                    "target": e.target.to_str(),
                    "revision": e.revision,
                    "payload": _thaw_json(e.payload),
                }
                for eid, e in sorted(self.edges.items())
            },
            "revision_vector": self.revision_vector.to_dict(),
        }

    def payload_hash(self, *, algorithm: str = "sha256") -> str:
        """Compute a deterministic hash of the snapshot content.

        The ``hash`` attribute on the snapshot itself is *not* included
        in the computation (hash can't hash itself).
        """
        return canonical_hash(self.payload_dict(), algorithm=algorithm)


@dataclass(frozen=True)
class PlanSnapshot:
    """Plan-specific projection of a graph snapshot.

    Filters the generic :class:`GraphSnapshot` down to
    ``graph_kind == "plan"`` and adds plan-specific derived state:
    ready IDs, blocked IDs, cycle node refs, and active blockers.

    Use :meth:`from_graph` to project deterministically from a
    :class:`GraphSnapshot`.
    """

    graph_snapshot: GraphSnapshot
    ready_ids: frozenset[NodeRef] = field(default_factory=frozenset)
    blocked_ids: frozenset[NodeRef] = field(default_factory=frozenset)
    cycle_node_refs: list[NodeRef] = field(default_factory=list)
    active_blockers: dict[NodeRef, tuple[NodeRef, ...]] = field(default_factory=dict)

    # ── projection from GraphSnapshot ───────────────────────────────

    @classmethod
    def from_graph(cls, snapshot: GraphSnapshot) -> "PlanSnapshot":
        """Project a plan-graph snapshot from a generic GraphSnapshot.

        Computes readiness, blockers, and cycle detection over all
        ``depends_on`` edges in plan graphs.  The algorithm uses an iterative three-color DFS.
        """
        # collect plan nodes + depends_on edges
        plan_nodes: dict[NodeRef, GraphNode] = {}
        plan_graph_ids: set[str] = set()
        for gid, g in snapshot.graphs.items():
            if gid != g.graph_id:
                raise ValueError(f"graph map key {gid!r} does not match graph_id {g.graph_id!r}")
            if g.board_id != snapshot.board_id:
                raise ValueError(f"graph {gid!r} belongs to a different board {g.board_id!r}")
            if g.graph_kind == "plan":
                plan_graph_ids.add(gid)
        for ref, node in snapshot.nodes.items():
            if ref != node.ref:
                raise ValueError(f"node map key {ref} does not match node.ref {node.ref}")
            if ref.graph in plan_graph_ids and ref.kind == "task":
                plan_nodes[ref] = node

        # build outgoing adjacency from depends_on edges
        # canonical direction: source depends_on target  →  source → target
        outgoing: dict[NodeRef, set[NodeRef]] = {ref: set() for ref in plan_nodes}
        incoming: dict[NodeRef, set[NodeRef]] = {ref: set() for ref in plan_nodes}
        for edge in snapshot.edges.values():
            if edge.type != "depends_on":
                continue
            if edge.source in plan_nodes or edge.target in plan_nodes:
                if (
                    edge.source not in plan_nodes
                    or edge.target not in plan_nodes
                    or edge.source.graph != edge.target.graph
                    or edge.graph != edge.source.graph
                ):
                    raise ValueError(f"invalid plan dependency graph identity for {edge.edge_id!r}")
            if edge.source not in plan_nodes or edge.target not in plan_nodes:
                continue
            outgoing[edge.source].add(edge.target)
            incoming[edge.target].add(edge.source)

        # cycle detection (iterative three-color DFS)
        cycle_set = _detect_cycles(outgoing)

        # readiness computation
        # Spec §5.9: a task with base_status=completed but derived_status in
        # {needs_recheck, needs_review} is NOT a satisfied dependency — it must
        # be treated as still incomplete for downstream readiness checks.
        _UNSATISFIED_DERIVED = frozenset({"needs_recheck", "needs_review"})
        completed_refs: set[NodeRef] = set()
        for ref, node in plan_nodes.items():
            if node.state != "completed":
                continue
            derived = ""
            payload = node.payload if isinstance(node.payload, dict) else {}
            derived_val = payload.get("derived_status")
            if isinstance(derived_val, str):
                derived = derived_val
            if derived in _UNSATISFIED_DERIVED:
                continue
            completed_refs.add(ref)
        ready: set[NodeRef] = set()
        blocked: set[NodeRef] = set()
        blockers_map: dict[NodeRef, tuple[NodeRef, ...]] = {}

        for ref, node in plan_nodes.items():
            if node.state == "completed":
                continue
            if ref in cycle_set:
                blocked.add(ref)
                blockers_map[ref] = tuple(sorted(outgoing[ref], key=lambda r: r.to_str()))
                continue
            active = sorted(
                (p for p in outgoing[ref] if p not in completed_refs),
                key=lambda r: r.to_str(),
            )
            if active:
                blocked.add(ref)
                blockers_map[ref] = tuple(active)
            else:
                ready.add(ref)

        return cls(
            graph_snapshot=snapshot,
            ready_ids=frozenset(ready),
            blocked_ids=frozenset(blocked),
            cycle_node_refs=sorted(cycle_set, key=lambda r: r.to_str()),
            active_blockers=blockers_map,
        )


# ── helpers ──────────────────────────────────────────────────────────

_UNVISITED = 0
_VISITING = 1
_VISITED = 2


def _detect_cycles(graph: dict[NodeRef, set[NodeRef]]) -> set[NodeRef]:
    """Iterative 3-color DFS cycle detector.

    Returns every node that participates in at least one cycle.
    """
    color: dict[NodeRef, int] = {node: _UNVISITED for node in graph}
    cycles: set[NodeRef] = set()

    for start in sorted(graph.keys(), key=lambda r: r.to_str()):
        if color[start] != _UNVISITED:
            continue
        stack: list[tuple[NodeRef, list[NodeRef]]] = [
            (start, sorted(graph.get(start, set()), key=lambda r: r.to_str()))
        ]
        color[start] = _VISITING
        path: list[NodeRef] = [start]
        active_path_index: dict[NodeRef, int] = {start: 0}
        # use an index-based cursor into each child list
        cursors: list[int] = [0]

        while stack:
            node, children = stack[-1]
            cursor = cursors[-1]
            if cursor < len(children):
                cursors[-1] = cursor + 1
                child = children[cursor]
                child_color = color.get(child, _UNVISITED)
                if child_color == _UNVISITED:
                    color[child] = _VISITING
                    active_path_index[child] = len(path)
                    path.append(child)
                    stack.append(
                        (
                            child,
                            sorted(graph.get(child, set()), key=lambda r: r.to_str()),
                        )
                    )
                    cursors.append(0)
                elif child_color == _VISITING:
                    idx = active_path_index.get(child)
                    if idx is not None:
                        cycles.update(path[idx:])
                    else:
                        cycles.add(child)
            else:
                color[node] = _VISITED
                stack.pop()
                cursors.pop()
                if path and path[-1] == node:
                    active_path_index.pop(node, None)
                    path.pop()

    return cycles


__all__ = [
    "Board",
    "BoardPolicy",
    "Claim",
    "Graph",
    "GraphEdge",
    "GraphNode",
    "GraphSnapshot",
    "PlanSnapshot",
    "RevisionVector",
]
