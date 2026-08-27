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
# pylint: disable=W0621
"""Unit tests for the Plan Graph Layer1 solver (lkb.plan_graph_rules).

Each R-PG-xxx rule gets at least one pass and one denial case, exercised
directly against ``PlanGraphLayer1Solver.evaluate`` with hand-built
GraphSnapshots (no repository / application-service stack).  A final
section verifies that the command handlers in ``lkb.plan_graph`` are thin
delegates whose ValidationRun shape is unchanged (engine="plan-graph").
"""

from __future__ import annotations

import pytest

from lkb.commands import GraphCommand
from lkb.graph_types import Graph, GraphEdge, GraphNode, GraphSnapshot, plan_task_ref
from lkb.plan_graph import plan_command_dispatcher
from lkb.plan_graph_rules import PlanGraphLayer1Solver, plan_graph_layer1

# ── harness ──────────────────────────────────────────────────────────

_GRAPH_ID = "plan"
_BOARD_ID = "b-1"


def _ref(task_id: str):
    return plan_task_ref(task_id, graph_id=_GRAPH_ID)


def _node(
    task_id: str,
    *,
    state: str = "pending",
    owner: str | None = None,
    revision: int = 1,
    payload: dict | None = None,
) -> GraphNode:
    ref = _ref(task_id)
    return GraphNode(
        ref=ref,
        title=task_id,
        state=state,
        owner=owner,
        revision=revision,
        payload=dict(payload or {}),
    )


def _snapshot(
    nodes: list[GraphNode] | None = None,
    edges: list[tuple[str, str]] | None = None,
    *,
    policy: dict | None = None,
    plan_state: str = "active",
) -> GraphSnapshot:
    edge_map = {}
    for index, (source, target) in enumerate(edges or []):
        edge_map[f"e-{index}"] = GraphEdge(
            edge_id=f"e-{index}",
            graph=_GRAPH_ID,
            type="depends_on",
            source=_ref(source),
            target=_ref(target),
        )
    return GraphSnapshot(
        board_id=_BOARD_ID,
        graphs={
            _GRAPH_ID: Graph(
                graph_id=_GRAPH_ID,
                board_id=_BOARD_ID,
                graph_kind="plan",
                metadata={"state": plan_state},
            )
        },
        nodes={node.ref: node for node in (nodes or [])},
        edges=edge_map,
        policy=dict(policy or {}),
    )


def _cmd(
    kind: str,
    *,
    actor: str = "agent-a",
    payload: dict | None = None,
    expected_node_revision: int | None = None,
    reason: str | None = None,
) -> GraphCommand:
    return GraphCommand(
        command_id=f"cmd-{kind}-1",
        board_id=_BOARD_ID,
        actor=actor,
        kind=kind,
        payload=dict(payload or {}),
        expected_node_revision=expected_node_revision,
        reason=reason,
    )


@pytest.fixture()
def solver() -> PlanGraphLayer1Solver:
    return plan_graph_layer1()


def _denied(outcome):
    assert outcome.issues, "expected a denial"
    assert outcome.derived_facts == ()
    return outcome.issues[0]


def _accepted(outcome):
    assert outcome.issues == (), f"expected acceptance, got {outcome.issues!r}"


# ── solver identity / registry ───────────────────────────────────────


def test_solver_version_and_rule_registry(solver: PlanGraphLayer1Solver) -> None:
    assert solver.solver_version == "lkb-layer1-plan-graph-v1"
    assert solver.known_kinds() == frozenset(
        {
            "create_task",
            "update_task_fields",
            "add_dependency",
            "remove_dependency",
            "claim_task",
            "release_task",
            "transfer_task",
            "start_task",
            "complete_task",
            "reopen_task",
            "delete_task",
            "revalidate",
            "patch_task",
        }
    )
    # Universal pre-gates run first for every kind.
    for kind in solver.known_kinds():
        rule_ids = [rule_id for rule_id, _check in solver.rules_for(kind)]
        assert rule_ids[:2] == ["R-PG-001", "R-PG-002"]
        assert all(rule_id.startswith("R-PG-") for rule_id in rule_ids)


# ── R-PG-001 plan_not_active (universal pre-gate) ────────────────────


def test_r_pg_001_denies_write_on_inactive_plan(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot(plan_state="archived")
    outcome = solver.evaluate(_cmd("create_task", payload={"task_id": "T-1", "subject": "S"}), snap)
    issue = _denied(outcome)
    assert issue.code == "plan_not_active"
    assert issue.rule == "R-PG-001"
    assert issue.message == "Plan 'plan' is archived; reopen it before writing"


def test_r_pg_001_precedes_task_rules(solver: PlanGraphLayer1Solver) -> None:
    # plan_not_active wins over task_not_found (pre-gate ordering).
    snap = _snapshot(plan_state="completed")
    outcome = solver.evaluate(_cmd("claim_task", payload={"task_id": "T-9"}), snap)
    assert _denied(outcome).code == "plan_not_active"


def test_r_pg_001_passes_on_active_plan(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot(plan_state="active")
    outcome = solver.evaluate(_cmd("create_task", payload={"task_id": "T-1", "subject": "S"}), snap)
    _accepted(outcome)


# ── R-PG-002 stale_revision (universal pre-gate) ─────────────────────


def test_r_pg_002_denies_stale_expected_revision(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", revision=2)])
    outcome = solver.evaluate(
        _cmd("update_task_fields", payload={"task_id": "T-1"}, expected_node_revision=1),
        snap,
    )
    issue = _denied(outcome)
    assert issue.code == "stale_revision"
    assert issue.rule == "R-PG-002"
    assert issue.message == "Task revision mismatch: expected 1, got 2"


def test_r_pg_002_passes_on_matching_revision(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", revision=2)])
    outcome = solver.evaluate(
        _cmd("update_task_fields", payload={"task_id": "T-1"}, expected_node_revision=2),
        snap,
    )
    _accepted(outcome)


# ── R-PG-003 task_not_found ──────────────────────────────────────────


@pytest.mark.parametrize(
    "kind",
    [
        "update_task_fields",
        "claim_task",
        "release_task",
        "transfer_task",
        "start_task",
        "complete_task",
        "reopen_task",
        "delete_task",
        "revalidate",
        "patch_task",
    ],
)
def test_r_pg_003_denies_missing_task(solver: PlanGraphLayer1Solver, kind: str) -> None:
    payload: dict = {"task_id": "T-9"}
    context = {"sub_kinds": ("start_task",)} if kind == "patch_task" else None
    outcome = solver.evaluate(_cmd(kind, payload=payload), _snapshot(), context=context)
    issue = _denied(outcome)
    assert issue.code == "task_not_found"
    assert issue.rule == "R-PG-003"
    assert issue.message == "Task 'T-9' not found"
    assert issue.subject_ref == _ref("T-9")


def test_r_pg_003_passes_for_existing_task(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("update_task_fields", payload={"task_id": "T-1"}), snap)
    _accepted(outcome)
    assert outcome.subject_ref == _ref("T-1")


# ── R-PG-004 needs_recheck ───────────────────────────────────────────


@pytest.mark.parametrize("kind", ["claim_task", "start_task"])
@pytest.mark.parametrize("derived", ["needs_recheck", "needs_review"])
def test_r_pg_004_denies_stale_derived_status(solver: PlanGraphLayer1Solver, kind: str, derived: str) -> None:
    snap = _snapshot([_node("T-1", payload={"derived_status": derived})])
    outcome = solver.evaluate(_cmd(kind, payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "needs_recheck"
    assert issue.rule == "R-PG-004"
    assert issue.message == (f"Task is in derived_status={derived}; revalidate before proceeding")


def test_r_pg_004_passes_for_clean_task(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("claim_task", payload={"task_id": "T-1"}), snap)
    _accepted(outcome)


# ── R-PG-010 / R-PG-011 create_task ──────────────────────────────────


def test_r_pg_010_denies_empty_subject(solver: PlanGraphLayer1Solver) -> None:
    outcome = solver.evaluate(_cmd("create_task", payload={"task_id": "T-1", "subject": ""}), _snapshot())
    issue = _denied(outcome)
    assert issue.code == "invalid_task"
    assert issue.rule == "R-PG-010"
    assert issue.message == "subject must not be empty"


def test_r_pg_011_denies_duplicate_task(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("create_task", payload={"task_id": "T-1", "subject": "B"}), snap)
    issue = _denied(outcome)
    assert issue.code == "duplicate_task"
    assert issue.rule == "R-PG-011"
    assert issue.message == "Task 'T-1' already exists"


def test_r_pg_010_011_pass_for_new_task(solver: PlanGraphLayer1Solver) -> None:
    outcome = solver.evaluate(
        _cmd("create_task", payload={"task_id": "T-2", "subject": "B"}),
        _snapshot([_node("T-1")]),
    )
    _accepted(outcome)
    assert outcome.subject_ref == _ref("T-2")


# ── R-PG-020 / R-PG-021 / R-PG-022 add_dependency ────────────────────


def test_r_pg_020_denies_unknown_dependent_first(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("add_dependency", payload={"task_id": "T-x", "depends_on": "T-y"}), snap)
    issue = _denied(outcome)
    assert issue.code == "unknown_task_reference"
    assert issue.rule == "R-PG-020"
    assert issue.message == "Task 'T-x' not found"


def test_r_pg_020_denies_unknown_prerequisite(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("add_dependency", payload={"task_id": "T-1", "depends_on": "T-y"}), snap)
    issue = _denied(outcome)
    assert issue.code == "unknown_task_reference"
    assert issue.message == "Task 'T-y' not found"


def test_r_pg_021_denies_self_dependency(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("add_dependency", payload={"task_id": "T-1", "depends_on": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "self_dependency"
    assert issue.rule == "R-PG-021"
    assert issue.message == "A task cannot depend on itself"


def test_r_pg_022_denies_cycle_on_candidate_edge(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1"), _node("T-2")], edges=[("T-1", "T-2")])
    outcome = solver.evaluate(_cmd("add_dependency", payload={"task_id": "T-2", "depends_on": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "dependency_cycle"
    assert issue.rule == "R-PG-022"
    assert issue.message == (
        "Adding dependency source_ref=plan:task:T-2 would create a cycle: "
        "plan:task:T-2 -> plan:task:T-1 -> plan:task:T-2"
    )


def test_r_pg_022_passes_for_acyclic_edge(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1"), _node("T-2"), _node("T-3")], edges=[("T-1", "T-2")])
    outcome = solver.evaluate(_cmd("add_dependency", payload={"task_id": "T-3", "depends_on": "T-1"}), snap)
    _accepted(outcome)
    assert outcome.subject_ref == _ref("T-3")


def test_r_pg_022_passes_idempotently_for_existing_edge(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1"), _node("T-2")], edges=[("T-2", "T-1")])
    outcome = solver.evaluate(_cmd("add_dependency", payload={"task_id": "T-2", "depends_on": "T-1"}), snap)
    _accepted(outcome)


def test_remove_dependency_always_accepted(solver: PlanGraphLayer1Solver) -> None:
    outcome = solver.evaluate(
        _cmd("remove_dependency", payload={"task_id": "T-2", "depends_on": "T-1"}),
        _snapshot(),
    )
    _accepted(outcome)
    assert outcome.subject_ref == _ref("T-2")


# ── R-PG-030..R-PG-033 claim_task ────────────────────────────────────


def test_r_pg_031_denies_claim_of_completed_task(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", state="completed")])
    outcome = solver.evaluate(_cmd("claim_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "already_resolved"
    assert issue.rule == "R-PG-031"
    assert issue.message == "Task is already completed"


def test_r_pg_030_denies_claim_of_blocked_task(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1"), _node("T-2")], edges=[("T-2", "T-1")])
    outcome = solver.evaluate(_cmd("claim_task", payload={"task_id": "T-2"}), snap)
    issue = _denied(outcome)
    assert issue.code == "blocked"
    assert issue.rule == "R-PG-030"
    assert issue.message == "Task is blocked"
    assert issue.blockers == ("T-1",)


def test_r_pg_032_denies_claim_by_other_owner(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", owner="agent-b")])
    outcome = solver.evaluate(_cmd("claim_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "already_claimed"
    assert issue.rule == "R-PG-032"
    assert issue.message == "Task already claimed by 'agent-b'"


def test_r_pg_032_passes_for_idempotent_reclaim(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", owner="agent-a")])
    outcome = solver.evaluate(_cmd("claim_task", payload={"task_id": "T-1"}), snap)
    _accepted(outcome)


def test_r_pg_033_denies_agent_busy_under_single_active_policy(
    solver: PlanGraphLayer1Solver,
) -> None:
    snap = _snapshot(
        [_node("T-1", owner="agent-a"), _node("T-2")],
        policy={"single_active_task_per_agent": True},
    )
    outcome = solver.evaluate(_cmd("claim_task", payload={"task_id": "T-2"}), snap)
    issue = _denied(outcome)
    assert issue.code == "agent_busy"
    assert issue.rule == "R-PG-033"
    assert "Agent 'agent-a' already has an active task: " in issue.message
    assert "T-1" in issue.message


def test_r_pg_033_passes_when_policy_disabled(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", owner="agent-a"), _node("T-2")])
    outcome = solver.evaluate(_cmd("claim_task", payload={"task_id": "T-2"}), snap)
    _accepted(outcome)


def test_claim_passes_for_unowned_ready_task(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("claim_task", payload={"task_id": "T-1"}), snap)
    _accepted(outcome)
    assert outcome.subject_ref == _ref("T-1")


# ── R-PG-034 transfer_task ───────────────────────────────────────────


def test_r_pg_034_denies_missing_new_owner(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("transfer_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "invalid_transfer"
    assert issue.rule == "R-PG-034"
    assert issue.message == "transfer_task requires a non-empty 'new_owner'"


def test_r_pg_034_denies_new_owner_equal_actor(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("transfer_task", payload={"task_id": "T-1", "new_owner": "agent-a"}), snap)
    issue = _denied(outcome)
    assert issue.code == "invalid_transfer"
    assert issue.message == ("transfer_task new_owner must differ from actor (use claim_task instead)")


def test_r_pg_034_passes_for_valid_transfer(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("transfer_task", payload={"task_id": "T-1", "new_owner": "agent-b"}), snap)
    _accepted(outcome)


# ── R-PG-040..R-PG-042 start_task ────────────────────────────────────


def test_r_pg_041_denies_start_of_completed_task(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", state="completed", owner="agent-a")])
    outcome = solver.evaluate(_cmd("start_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "already_resolved"
    assert issue.rule == "R-PG-041"


def test_r_pg_040_denies_start_of_blocked_task(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1"), _node("T-2", owner="agent-a")], edges=[("T-2", "T-1")])
    outcome = solver.evaluate(_cmd("start_task", payload={"task_id": "T-2"}), snap)
    issue = _denied(outcome)
    assert issue.code == "blocked"
    assert issue.rule == "R-PG-040"
    assert issue.message == "Task is blocked"


def test_r_pg_042_denies_start_without_claim(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("start_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "owner_required"
    assert issue.rule == "R-PG-042"
    assert issue.message == "Task must be claimed before start"


def test_r_pg_042_denies_start_by_non_owner(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", owner="agent-b")])
    outcome = solver.evaluate(_cmd("start_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "not_owner"
    assert issue.message == "Task is owned by 'agent-b'"


def test_start_passes_for_owner(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", owner="agent-a")])
    outcome = solver.evaluate(_cmd("start_task", payload={"task_id": "T-1"}), snap)
    _accepted(outcome)


# ── R-PG-050..R-PG-053 complete_task ─────────────────────────────────


def test_r_pg_050_denies_complete_from_pending(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("complete_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "invalid_transition"
    assert issue.rule == "R-PG-050"
    assert issue.message == "Task must be in_progress to complete"


def test_r_pg_052_denies_complete_with_active_blockers(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot(
        [_node("T-1"), _node("T-2", state="in_progress", owner="agent-a")],
        edges=[("T-2", "T-1")],
    )
    outcome = solver.evaluate(_cmd("complete_task", payload={"task_id": "T-2"}), snap)
    issue = _denied(outcome)
    assert issue.code == "blocked"
    assert issue.rule == "R-PG-052"
    assert issue.message == "Task has active blockers"


def test_r_pg_053_denies_complete_without_claim(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", state="in_progress")])
    outcome = solver.evaluate(_cmd("complete_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "owner_required"
    assert issue.rule == "R-PG-053"
    assert issue.message == "Task must be claimed before completion"


def test_r_pg_053_denies_complete_by_non_owner_without_reason(
    solver: PlanGraphLayer1Solver,
) -> None:
    snap = _snapshot([_node("T-1", state="in_progress", owner="agent-b")])
    outcome = solver.evaluate(_cmd("complete_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "not_owner"
    assert issue.message == "Task is owned by 'agent-b'"


def test_r_pg_053_passes_non_owner_with_reason(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1", state="in_progress", owner="agent-b")])
    outcome = solver.evaluate(_cmd("complete_task", payload={"task_id": "T-1"}, reason="handoff approved"), snap)
    _accepted(outcome)


# ── R-PG-060 delete_task ─────────────────────────────────────────────


def test_r_pg_060_denies_dangling_dependency(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1"), _node("T-2")], edges=[("T-2", "T-1")])
    outcome = solver.evaluate(_cmd("delete_task", payload={"task_id": "T-1"}), snap)
    issue = _denied(outcome)
    assert issue.code == "dangling_dependency"
    assert issue.rule == "R-PG-060"
    assert issue.message == "Task is referenced by dependencies; use cascade=true to remove"


def test_r_pg_060_passes_with_cascade(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1"), _node("T-2")], edges=[("T-2", "T-1")])
    outcome = solver.evaluate(_cmd("delete_task", payload={"task_id": "T-1", "cascade": True}), snap)
    _accepted(outcome)


def test_r_pg_060_passes_without_references(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("delete_task", payload={"task_id": "T-1"}), snap)
    _accepted(outcome)


# ── R-PG-070 / R-PG-071 revalidate ───────────────────────────────────


def test_r_pg_070_denies_revalidate_with_unverified_upstream(
    solver: PlanGraphLayer1Solver,
) -> None:
    snap = _snapshot(
        [_node("T-1"), _node("T-2", payload={"derived_status": "needs_recheck"})],
        edges=[("T-2", "T-1")],
    )
    outcome = solver.evaluate(_cmd("revalidate", payload={"task_id": "T-2"}), snap)
    issue = _denied(outcome)
    assert issue.code == "needs_recheck"
    assert issue.rule == "R-PG-070"
    assert issue.message == "Cannot revalidate: upstream not verified: T-1"


def test_r_pg_070_passes_with_verified_upstream(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot(
        [
            _node("T-1", state="completed"),
            _node("T-2", payload={"derived_status": "needs_recheck"}),
        ],
        edges=[("T-2", "T-1")],
    )
    outcome = solver.evaluate(_cmd("revalidate", payload={"task_id": "T-2"}), snap)
    _accepted(outcome)


# ── R-PG-090 patch_task ──────────────────────────────────────────────


def test_r_pg_090_denies_unsupported_status(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(
        _cmd("patch_task", payload={"task_id": "T-1", "status": "bogus"}),
        snap,
        context={"sub_kinds": ("start_task",)},
    )
    issue = _denied(outcome)
    assert issue.code == "invalid_status"
    assert issue.rule == "R-PG-090"
    assert issue.message == "Unsupported task status 'bogus'"


def test_r_pg_090_denies_empty_patch(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(_cmd("patch_task", payload={"task_id": "T-1"}), snap, context={"sub_kinds": ()})
    issue = _denied(outcome)
    assert issue.code == "empty_patch"
    assert issue.message == "patch_task requires at least one sub-intent"


def test_r_pg_090_denies_unknown_sub_intent_kind(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(
        _cmd("patch_task", payload={"task_id": "T-1"}),
        snap,
        context={"sub_kinds": ("start_task", "bogus_kind")},
    )
    issue = _denied(outcome)
    assert issue.code == "unknown_command"
    assert issue.message == "Patch sub-intent maps to unknown kind 'bogus_kind'"


def test_r_pg_090_passes_for_valid_patch(solver: PlanGraphLayer1Solver) -> None:
    snap = _snapshot([_node("T-1")])
    outcome = solver.evaluate(
        _cmd("patch_task", payload={"task_id": "T-1", "status": "in_progress"}),
        snap,
        context={"sub_kinds": ("start_task",)},
    )
    _accepted(outcome)


# ── handler delegation (ValidationRun shape unchanged) ───────────────


def test_dispatcher_delegates_and_keeps_run_shape() -> None:
    dispatcher = plan_command_dispatcher()
    snap = _snapshot([_node("T-1")])

    accepted = dispatcher.validate(_cmd("claim_task", payload={"task_id": "T-1"}), snap)
    assert accepted.result == "pass"
    assert accepted.engine == "plan-graph"
    assert accepted.subject_ref == _ref("T-1")
    assert accepted.issues == ()

    denied = dispatcher.validate(_cmd("claim_task", payload={"task_id": "T-9"}), snap)
    assert denied.result == "denied"
    assert denied.engine == "plan-graph"
    assert denied.subject_ref == _ref("T-9")
    assert len(denied.issues) == 1
    assert denied.issues[0].code == "task_not_found"
    assert denied.issues[0].rule == "R-PG-003"


def test_dispatcher_pre_gates_run_through_solver() -> None:
    dispatcher = plan_command_dispatcher()
    snap = _snapshot([_node("T-1")], plan_state="archived")
    denied = dispatcher.validate(_cmd("claim_task", payload={"task_id": "T-1"}), snap)
    assert denied.result == "denied"
    assert denied.issues[0].code == "plan_not_active"
    assert denied.issues[0].rule == "R-PG-001"


def test_dispatcher_unknown_command_unchanged() -> None:
    dispatcher = plan_command_dispatcher()
    denied = dispatcher.validate(_cmd("bogus_kind"), _snapshot())
    assert denied.result == "denied"
    assert denied.issues[0].code == "unknown_command"
    assert denied.issues[0].message == "No handler for kind 'bogus_kind'"
