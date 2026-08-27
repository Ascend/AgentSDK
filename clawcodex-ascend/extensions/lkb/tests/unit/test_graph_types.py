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

"""Unit tests for lkb.graph_types.

Covers Board / Graph / GraphNode / GraphEdge / Claim / RevisionVector
/ GraphSnapshot / PlanSnapshot round-trips and computations.
"""

from __future__ import annotations

import pytest

from lkb.graph_types import (
    Board,
    BoardPolicy,
    Claim,
    Graph,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
    PlanSnapshot,
    RevisionVector,
    _detect_cycles,
)
from lkb.refs import NodeRef
# ── RevisionVector ──────────────────────────────────────────────────


class TestRevisionVector:
    def test_default_empty(self) -> None:
        rv = RevisionVector()
        assert rv.get("plan") == 0

    def test_with_update_immutable(self) -> None:
        rv = RevisionVector()
        rv2 = rv.with_update("plan", 3)
        assert rv.get("plan") == 0
        assert rv2.get("plan") == 3
        assert rv2.get("artifact") == 0

    def test_equals(self) -> None:
        a = RevisionVector(revisions={"plan": 1, "artifact": 2})
        b = RevisionVector(revisions={"plan": 1, "artifact": 2})
        c = RevisionVector(revisions={"plan": 1})
        assert a.equals(b)
        assert not a.equals(c)
        assert not a.equals("not a vector")


def test_board_policy_normalizes_legacy_bool_and_role_collections() -> None:
    assert BoardPolicy(force_override_roles=False).force_override_roles == ()
    assert BoardPolicy(force_override_roles=True).force_override_roles == ("*",)
    policy = BoardPolicy(force_override_roles=("operator", "admin", "operator"))
    assert policy.force_override_roles == ("operator", "admin")
    assert policy.allows_force_override("admin")
    assert not policy.allows_force_override("viewer")


# ── Board + BoardPolicy ─────────────────────────────────────────────


class TestBoard:
    def test_default_policy(self) -> None:
        board = Board(board_id="b1", project_uri="file:///tmp")
        assert board.board_id == "b1"
        assert board.schema_version == 1
        assert isinstance(board.policy, BoardPolicy)
        assert board.policy.invalidation_mode == "cascade"

    def test_custom_policy(self) -> None:
        policy = BoardPolicy(
            single_active_task_per_agent=True,
            invalidation_mode="direct",
        )
        board = Board(
            board_id="b2",
            project_uri="https://example.com/repo",
            display_name="My Board",
            policy=policy,
        )
        assert board.display_name == "My Board"
        assert board.policy.single_active_task_per_agent is True
        assert board.policy.invalidation_mode == "direct"


# ── Graph ───────────────────────────────────────────────────────────


class TestGraph:
    def test_basic(self) -> None:
        g = Graph(
            graph_id="plan-main",
            board_id="b1",
            graph_kind="plan",
            revision=5,
        )
        assert g.graph_id == "plan-main"
        assert g.graph_kind == "plan"
        assert g.revision == 5


# ── GraphNode ───────────────────────────────────────────────────────


class TestGraphNode:
    def test_basic(self) -> None:
        ref = NodeRef("plan", "task", "T-001")
        node = GraphNode(
            ref=ref,
            title="Implement auth",
            state="ready",
            owner="agent-a",
            revision=2,
        )
        assert node.ref == ref
        assert node.title == "Implement auth"
        assert node.state == "ready"
        assert node.owner == "agent-a"
        assert node.revision == 2
        assert node.payload == {}


# ── GraphEdge ───────────────────────────────────────────────────────


class TestGraphEdge:
    def test_depends_on_edge(self) -> None:
        src = NodeRef("plan", "task", "T-002")
        tgt = NodeRef("plan", "task", "T-001")
        edge = GraphEdge(
            edge_id="e1",
            graph="plan",
            type="depends_on",
            source=src,
            target=tgt,
        )
        # canonical: T-002 depends_on T-001  →  T-001 blocks T-002
        assert edge.source == src
        assert edge.target == tgt
        assert edge.type == "depends_on"

    def test_project_blocks(self) -> None:
        """Projecting depends_on → blocks must flip source/target."""
        src = NodeRef("plan", "task", "T-002")
        tgt = NodeRef("plan", "task", "T-001")
        edge = GraphEdge(
            edge_id="e1",
            graph="plan",
            type="depends_on",
            source=src,
            target=tgt,
        )
        blocks = edge.project("blocks")
        # T-001 blocks T-002
        assert blocks.source == tgt
        assert blocks.target == src
        assert blocks.type == "blocks"
        # same edge_id, graph, revision, payload
        assert blocks.edge_id == edge.edge_id
        assert blocks.graph == edge.graph
        assert blocks.revision == edge.revision
        # original edge is untouched (frozen)
        assert edge.source == src
        assert edge.type == "depends_on"

    def test_open_edge_types(self) -> None:
        """Edge type is an open string — custom types work."""
        edge = GraphEdge(
            edge_id="e2",
            graph="plan",
            type="custom_relates_to",
            source=NodeRef("plan", "task", "T-1"),
            target=NodeRef("plan", "risk", "R-1"),
        )
        assert edge.type == "custom_relates_to"


# ── Claim ───────────────────────────────────────────────────────────


class TestClaim:
    def test_active_claim(self) -> None:
        claim = Claim(
            task_ref=NodeRef("plan", "task", "T-001"),
            owner_ref=NodeRef("plan", "agent", "agent-a"),
            claim_id="c1",
            status="active",
        )
        assert claim.status == "active"
        assert claim.task_ref.task_id == "T-001"
        assert claim.owner_ref.kind == "agent"

    def test_released_claim(self) -> None:
        claim = Claim(
            task_ref=NodeRef("plan", "task", "T-001"),
            owner_ref=NodeRef("plan", "agent", "agent-a"),
            claim_id="c1",
            status="released",
            reason="timeout",
        )
        assert claim.status == "released"
        assert claim.reason == "timeout"


# ── GraphSnapshot.payload_hash ──────────────────────────────────────


class TestGraphSnapshotHash:
    def test_deterministic_hash(self) -> None:
        """Same content → same hash every time."""
        ref = NodeRef("plan", "task", "T-1")
        node = GraphNode(ref=ref, title="t1")
        gs = GraphSnapshot(
            board_id="b1",
            nodes={ref: node},
        )
        h1 = gs.payload_hash()
        h2 = gs.payload_hash()
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_different_content_different_hash(self) -> None:
        """Different content → different hash."""
        ref = NodeRef("plan", "task", "T-1")
        gs1 = GraphSnapshot(
            board_id="b1",
            nodes={ref: GraphNode(ref=ref, title="t1")},
        )
        gs2 = GraphSnapshot(
            board_id="b1",
            nodes={ref: GraphNode(ref=ref, title="t2")},
        )
        assert gs1.payload_hash() != gs2.payload_hash()

    def test_forged_hash_is_rejected(self) -> None:
        """A snapshot cannot advertise a hash for different content."""
        ref = NodeRef("plan", "task", "T-1")
        with pytest.raises(ValueError, match="hash mismatch"):
            GraphSnapshot(
                board_id="b1",
                nodes={ref: GraphNode(ref=ref, title="t1")},
                hash="sha256:" + ("0" * 64),
            )


# ── PlanSnapshot.from_graph ─────────────────────────────────────────


class TestPlanSnapshotFromGraph:
    def _make_diamond(self) -> GraphSnapshot:
        """Build a simple diamond DAG:  A → {B, C} → D."""
        nodes = {
            "A": GraphNode(ref=NodeRef("plan", "task", "A"), title="A", state="completed"),
            "B": GraphNode(ref=NodeRef("plan", "task", "B"), title="B", state="todo"),
            "C": GraphNode(ref=NodeRef("plan", "task", "C"), title="C", state="todo"),
            "D": GraphNode(ref=NodeRef("plan", "task", "D"), title="D", state="todo"),
        }
        edges = {
            "e1": GraphEdge(
                edge_id="e1",
                graph="plan",
                type="depends_on",
                source=nodes["B"].ref,
                target=nodes["A"].ref,
            ),
            "e2": GraphEdge(
                edge_id="e2",
                graph="plan",
                type="depends_on",
                source=nodes["C"].ref,
                target=nodes["A"].ref,
            ),
            "e3": GraphEdge(
                edge_id="e3",
                graph="plan",
                type="depends_on",
                source=nodes["D"].ref,
                target=nodes["B"].ref,
            ),
            "e4": GraphEdge(
                edge_id="e4",
                graph="plan",
                type="depends_on",
                source=nodes["D"].ref,
                target=nodes["C"].ref,
            ),
        }
        graph = Graph(graph_id="plan", board_id="b1", graph_kind="plan")
        return GraphSnapshot(
            board_id="b1",
            graphs={"plan": graph},
            nodes={n.ref: n for n in nodes.values()},
            edges=edges,
        )

    def test_ready_and_blocked_in_diamond(self) -> None:
        gs = self._make_diamond()
        ps = PlanSnapshot.from_graph(gs)

        # A is completed → not in ready or blocked
        # B depends on completed A → ready
        # C depends on completed A → ready
        # D depends on B+C (both todo) → blocked
        assert ps.ready_ids == {
            NodeRef("plan", "task", "B"),
            NodeRef("plan", "task", "C"),
        }
        assert NodeRef("plan", "task", "D") in ps.blocked_ids
        # D's active blockers should be B and C
        d_ref = NodeRef("plan", "task", "D")
        assert set(ps.active_blockers[d_ref]) == {
            NodeRef("plan", "task", "B"),
            NodeRef("plan", "task", "C"),
        }

    def test_no_cycles_in_diamond(self) -> None:
        gs = self._make_diamond()
        ps = PlanSnapshot.from_graph(gs)
        assert ps.cycle_node_refs == []

    def test_cycle_detection(self) -> None:
        """A → B → A cycle must be detected."""
        a = NodeRef("plan", "task", "A")
        b = NodeRef("plan", "task", "B")
        nodes = {
            a: GraphNode(ref=a, title="A", state="todo"),
            b: GraphNode(ref=b, title="B", state="todo"),
        }
        edges = {
            "e1": GraphEdge(
                edge_id="e1",
                graph="plan",
                type="depends_on",
                source=a,
                target=b,
            ),
            "e2": GraphEdge(
                edge_id="e2",
                graph="plan",
                type="depends_on",
                source=b,
                target=a,
            ),
        }
        graph = Graph(graph_id="plan", board_id="b1", graph_kind="plan")
        gs = GraphSnapshot(
            board_id="b1",
            graphs={"plan": graph},
            nodes=nodes,
            edges=edges,
        )
        ps = PlanSnapshot.from_graph(gs)
        assert set(ps.cycle_node_refs) == {a, b}
        # Both A and B are blocked (in a cycle)
        assert a in ps.blocked_ids
        assert b in ps.blocked_ids

    def test_cycle_detection_handles_repeated_back_edges(self) -> None:
        refs = [NodeRef("plan", "task", f"T-{index:03d}") for index in range(160)]
        graph = {
            ref: {refs[index + 1], refs[0]} if index < len(refs) - 1 else {refs[0]} for index, ref in enumerate(refs)
        }

        assert _detect_cycles(graph) == set(refs)

    def test_non_plan_graph_ignored(self) -> None:
        """Nodes from a legacy foreign graph must not affect Plan readiness."""
        task_ref = NodeRef("plan", "task", "T-1")
        file_ref = NodeRef("legacy", "node", "legacy-record")
        plan_graph = Graph(graph_id="plan", board_id="b1", graph_kind="plan")
        legacy_graph = Graph(graph_id="legacy", board_id="b1", graph_kind="legacy")
        gs = GraphSnapshot(
            board_id="b1",
            graphs={"plan": plan_graph, "legacy": legacy_graph},
            nodes={
                task_ref: GraphNode(ref=task_ref, title="T1", state="todo"),
                file_ref: GraphNode(ref=file_ref, title="legacy-record"),
            },
            edges={},
        )
        ps = PlanSnapshot.from_graph(gs)
        assert task_ref in ps.ready_ids
        assert file_ref not in ps.ready_ids
        assert file_ref not in ps.blocked_ids
