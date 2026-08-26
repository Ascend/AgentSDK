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
# pylint: disable=E0402,E0611,R1714

"""Plan Graph command validation rules — the Layer1 solver for the Plan Graph domain.

Every constraint check that used to be hardcoded inside the command
handlers' ``validate()`` methods in :mod:`lkb.plan_graph` lives here as a
numbered, ordered, individually-testable rule (``R-PG-xxx``).  The
:class:`PlanGraphLayer1Solver` picks the rule set by ``command.kind`` and
short-circuits in a fixed order that preserves each handler's original
early-exit semantics — issue codes and message strings are verbatim.

The solver is a synchronous, in-process evaluator over a command and an
immutable Board snapshot. It is the only validation path for Plan Graph
commands.

Apply-time owner and override authorization re-checks stay in
:mod:`lkb.plan_graph` because they must run under the Board File Lock.
"""

from __future__ import annotations

from typing import Any, Callable

from . import plan_graph as _pg
from .commands import GraphCommand
from .error_codes import LkbErrorCode
from .graph_types import (
    BoardPolicy,
    GraphNode,
    GraphSnapshot,
    NodeRef,
    PlanSnapshot,
    plan_task_ref,
)
from .validation import ValidationIssue

__all__ = [
    "PlanGraphLayer1Solver",
    "PlanRuleOutcome",
    "Rule",
    "RuleContext",
    "RuleResult",
    "plan_graph_layer1",
]

SOLVER_VERSION = "lkb-layer1-plan-graph-v1"

_PASS: "RuleResult"  # forward declaration; assigned after RuleResult


# ── result types ─────────────────────────────────────────────────────


class RuleResult:
    """Outcome of a single rule check.

    ``issue`` set means the rule denies the command (short-circuit).
    ``derived_facts`` carries system-produced facts for the accept run.
    """

    __slots__ = ("issue", "derived_facts")

    def __init__(
        self,
        issue: ValidationIssue | None = None,
        derived_facts: tuple[str, ...] = (),
    ) -> None:
        self.issue = issue
        self.derived_facts = derived_facts


_PASS = RuleResult()


class PlanRuleOutcome:
    """Aggregate result of :meth:`PlanGraphLayer1Solver.evaluate`.

    ``issues`` empty means the command is accepted; ``subject_ref`` is the
    subject the accept run should carry.  On denial the run's subject is
    the first issue's ``subject_ref`` (identical to the historical handler
    behaviour, where run and issue subjects always matched).
    """

    __slots__ = ("issues", "derived_facts", "subject_ref", "solver_version")

    def __init__(
        self,
        *,
        issues: tuple[ValidationIssue, ...] = (),
        derived_facts: tuple[str, ...] = (),
        subject_ref: NodeRef | None = None,
        solver_version: str = SOLVER_VERSION,
    ) -> None:
        self.issues = issues
        self.derived_facts = derived_facts
        self.subject_ref = subject_ref
        self.solver_version = solver_version


class RuleContext:
    """Per-evaluation shared context.

    The :class:`PlanSnapshot` projection is computed lazily and cached so
    rules that need readiness/blocker state share a single projection
    (handlers previously projected once per ``validate`` call).
    """

    __slots__ = ("snapshot", "extras", "known_kinds", "_plan")

    def __init__(
        self,
        snapshot: GraphSnapshot,
        *,
        extras: dict[str, Any],
        known_kinds: frozenset[str],
    ) -> None:
        self.snapshot = snapshot
        self.extras = extras
        self.known_kinds = known_kinds
        self._plan: PlanSnapshot | None = None

    @property
    def plan(self) -> PlanSnapshot:
        if self._plan is None:
            self._plan = PlanSnapshot.from_graph(self.snapshot)
        return self._plan


Rule = tuple[str, Callable[[GraphCommand, GraphSnapshot, RuleContext], RuleResult]]


# ── helpers moved from lkb.plan_graph (validation-only) ──────────────


def _board_policy_from_snapshot(snapshot: GraphSnapshot) -> BoardPolicy:
    """Load the board policy from the snapshot (spec §7.6 lock-free read)."""
    return BoardPolicy.from_dict(dict(snapshot.policy or {}))


def _snapshot_prerequisites(snapshot: GraphSnapshot, ref: NodeRef) -> list[NodeRef]:
    """Direct prerequisites of *ref* (targets of its depends_on edges)."""
    target = ref.to_str()
    out: list[NodeRef] = []
    for edge in snapshot.edges.values():
        if edge.type != "depends_on":
            continue
        if edge.source.to_str() == target:
            out.append(edge.target)
    return out


def _node_is_verified_completed(node: GraphNode) -> bool:
    """True if *node* is completed and its completion is still valid (spec §5.9)."""
    if node.state != "completed":
        return False
    payload = node.payload if isinstance(node.payload, dict) else {}
    derived = str(payload.get("derived_status", "") or "")
    return derived not in ("needs_recheck", "needs_review")


def _edges_from_snapshot(snapshot: GraphSnapshot) -> list[tuple[str, str, str]]:
    return [
        (e.edge_id, e.source.to_str(), e.target.to_str()) for e in snapshot.edges.values() if e.type == "depends_on"
    ]


def _would_create_cycle_in_snapshot(snapshot: GraphSnapshot, source_ref: str, target_ref: str) -> list[str] | None:
    """Return the cycle path if adding ``source -> target`` creates one.

    ``depends_on`` runs dependent -> prerequisite.  Adding ``source ->
    target`` closes a cycle iff ``target`` can already reach ``source``
    along existing depends_on edges.  Returns the path (including the
    closing edge) or None.
    """
    adjacency: dict[str, list[str]] = {}
    for _eid, src, tgt in _edges_from_snapshot(snapshot):
        adjacency.setdefault(src, []).append(tgt)
    stack: list[tuple[str, list[str]]] = [(target_ref, [target_ref])]
    visited: set[str] = set()
    while stack:
        node, path = stack.pop()
        if node == source_ref:
            return [source_ref, *path]
        if node in visited:
            continue
        visited.add(node)
        for nxt in adjacency.get(node, []):
            stack.append((nxt, [*path, nxt]))
    return None


def _snapshot_active_claims(snapshot: GraphSnapshot, graph_id: str) -> dict[str, str]:
    """Map ``task_ref_str -> owner`` for active claims (from snapshot).

    Snapshot edges/nodes are available but claims live on the envelope;
    for lock-free validation we approximate via node.owner when a claim
    record is absent.  The authoritative check happens in ``apply``.
    """
    out: dict[str, str] = {}
    for ref, node in snapshot.nodes.items():
        if ref.graph == graph_id and node.owner:
            out[ref.to_str()] = node.owner
    return out


def _issue(
    rule_id: str,
    code: LkbErrorCode,
    message: str,
    *,
    subject_ref: NodeRef | None = None,
    blockers: tuple[str, ...] = (),
) -> RuleResult:
    return RuleResult(issue=_pg._denied_issue(code, message, rule=rule_id, subject_ref=subject_ref, blockers=blockers))


def _node_for(snapshot: GraphSnapshot, ref: NodeRef) -> GraphNode | None:
    return next((n for n in snapshot.nodes.values() if n.ref == ref), None)


def _dependency_refs(command: GraphCommand) -> tuple[str, str, NodeRef, NodeRef]:
    """Resolve ``(dependent_id, prerequisite_id, dependent, prerequisite)``."""
    graph_id = _pg._plan_graph_id(command)
    dependent_id = str(command.payload.get("task_id") or command.payload.get("dependent") or "")
    prerequisite_id = str(command.payload.get("depends_on") or command.payload.get("prerequisite") or "")
    dependent = plan_task_ref(dependent_id, graph_id=graph_id)
    prerequisite = plan_task_ref(prerequisite_id, graph_id=graph_id)
    return dependent_id, prerequisite_id, dependent, prerequisite


def _already_resolved_issue(command: GraphCommand, snapshot: GraphSnapshot, rule_id: str) -> RuleResult:
    ref = _pg._task_ref(command)
    node = _node_for(snapshot, ref)
    if node is not None and node.state == "completed":
        return _issue(
            rule_id,
            LkbErrorCode.ALREADY_RESOLVED,
            "Task is already completed",
            subject_ref=ref,
        )
    return _PASS


def _blocked_issue(
    command: GraphCommand,
    snapshot: GraphSnapshot,
    ctx: RuleContext,
    rule_id: str,
    message: str,
) -> RuleResult:
    ref = _pg._task_ref(command)
    plan = ctx.plan
    if ref in plan.blocked_ids:
        blockers = plan.active_blockers.get(ref, ())
        return _issue(
            rule_id,
            LkbErrorCode.BLOCKED,
            message,
            subject_ref=ref,
            blockers=tuple(b.id for b in blockers),
        )
    return _PASS


# ── universal pre-gates (formerly PlanCommandDispatcher.validate) ────


def _r_pg_001_plan_not_active(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-001: writes to a non-active plan are denied (dispatcher pre-gate)."""
    del ctx
    graph_id = _pg._plan_graph_id(command)
    graph = snapshot.graphs.get(graph_id)
    if graph is not None:
        state = str(graph.metadata.get("state") or "active")
        if state != "active":
            return _issue(
                "R-PG-001",
                LkbErrorCode.PLAN_NOT_ACTIVE,
                f"Plan {graph_id!r} is {state}; reopen it before writing",
                subject_ref=command.primary_subject_ref,
            )
    return _PASS


def _r_pg_002_stale_revision(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-002: ``expected_node_revision`` must match the current node revision.

    Spec §6.4 #7 / LKB-STATE-007.  Universal pre-gate (formerly both the
    dispatcher pre-gate and the ``_check_expected_node_revision`` base
    helper used by Claim/Start — the two had identical semantics).
    """
    del ctx
    if command.expected_node_revision is None:
        return _PASS
    ref = _pg._task_ref(command)
    node = _node_for(snapshot, ref)
    if node is None:
        return _PASS  # handled by R-PG-003 task_not_found
    if node.revision != command.expected_node_revision:
        return _issue(
            "R-PG-002",
            LkbErrorCode.STALE_REVISION,
            f"Task revision mismatch: expected {command.expected_node_revision}, got {node.revision}",
            subject_ref=ref,
        )
    return _PASS


# ── shared task gates ────────────────────────────────────────────────


def _r_pg_003_task_not_found(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-003: the target task must exist (formerly base ``_require_task``)."""
    del ctx
    ref = _pg._task_ref(command)
    if ref.to_str() not in {n.to_str() for n in snapshot.nodes}:
        return _issue(
            "R-PG-003",
            LkbErrorCode.TASK_NOT_FOUND,
            f"Task {ref.id!r} not found",
            subject_ref=ref,
        )
    return _PASS


def _r_pg_004_needs_recheck(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-004: a task in needs_recheck cannot be claimed/started (spec §6.4/§6.5).

    Formerly base ``_check_not_needs_recheck``; only Claim and Start ran it.
    """
    del ctx
    ref = _pg._task_ref(command)
    node = _node_for(snapshot, ref)
    if node is None:
        return _PASS  # handled by R-PG-003 task_not_found
    payload = node.payload if isinstance(node.payload, dict) else {}
    derived = str(payload.get("derived_status", "") or "")
    if derived in ("needs_recheck", "needs_review"):
        return _issue(
            "R-PG-004",
            LkbErrorCode.NEEDS_RECHECK,
            f"Task is in derived_status={derived}; revalidate before proceeding",
            subject_ref=ref,
        )
    return _PASS


# ── create_task (spec §6.2) ──────────────────────────────────────────


def _r_pg_010_task_shape(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-010: subject must not be empty."""
    del snapshot, ctx
    subject = str(command.payload.get("subject", ""))
    if not subject:
        return _issue("R-PG-010", LkbErrorCode.INVALID_TASK, "subject must not be empty")
    return _PASS


def _r_pg_011_duplicate_task(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-011: an explicit task_id must not already exist."""
    del ctx
    task_id = str(command.payload.get("task_id", ""))
    graph_id = _pg._plan_graph_id(command)
    ref = plan_task_ref(task_id, graph_id=graph_id)
    if task_id and ref.to_str() in {n.to_str() for n in snapshot.nodes}:
        return _issue(
            "R-PG-011",
            LkbErrorCode.DUPLICATE_TASK,
            f"Task {task_id!r} already exists",
            subject_ref=ref,
        )
    return _PASS


# ── add_dependency (spec §6.3) ───────────────────────────────────────


def _r_pg_020_unknown_task_reference(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-020: both endpoints of a new dependency must exist (dependent first)."""
    del ctx
    dependent_id, prerequisite_id, dependent, prerequisite = _dependency_refs(command)
    known = {n.to_str() for n in snapshot.nodes}
    if dependent.to_str() not in known:
        return _issue(
            "R-PG-020",
            LkbErrorCode.UNKNOWN_TASK_REFERENCE,
            f"Task {dependent_id!r} not found",
            subject_ref=dependent,
        )
    if prerequisite.to_str() not in known:
        return _issue(
            "R-PG-020",
            LkbErrorCode.UNKNOWN_TASK_REFERENCE,
            f"Task {prerequisite_id!r} not found",
            subject_ref=prerequisite,
        )
    return _PASS


def _r_pg_021_self_dependency(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-021: a task cannot depend on itself."""
    del snapshot, ctx
    _dependent_id, _prerequisite_id, dependent, prerequisite = _dependency_refs(command)
    if dependent.to_str() == prerequisite.to_str():
        return _issue(
            "R-PG-021",
            LkbErrorCode.SELF_DEPENDENCY,
            "A task cannot depend on itself",
            subject_ref=dependent,
        )
    return _PASS


def _r_pg_022_dependency_cycle(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-022: the candidate edge must not close a cycle.

    An already-present edge passes idempotently (no-op apply).
    """
    del ctx
    _dependent_id, _prerequisite_id, dependent, prerequisite = _dependency_refs(command)
    # Idempotent: edge already present -> pass (no-op apply).
    for _eid, src, tgt in _edges_from_snapshot(snapshot):
        if src == dependent.to_str() and tgt == prerequisite.to_str():
            return _PASS
    # Cycle check on the candidate graph.
    cycle = _would_create_cycle_in_snapshot(snapshot, dependent.to_str(), prerequisite.to_str())
    if cycle is not None:
        return _issue(
            "R-PG-022",
            LkbErrorCode.DEPENDENCY_CYCLE,
            "Adding dependency source_ref=" + dependent.to_str() + " would create a cycle: " + " -> ".join(cycle),
            subject_ref=dependent,
        )
    return _PASS


# ── claim_task (spec §6.4) ───────────────────────────────────────────


def _r_pg_030_claim_blocked(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-030: a blocked task cannot be claimed."""
    return _blocked_issue(command, snapshot, ctx, "R-PG-030", "Task is blocked")


def _r_pg_031_claim_already_resolved(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-031: a completed task cannot be claimed."""
    del ctx
    return _already_resolved_issue(command, snapshot, "R-PG-031")


def _r_pg_032_already_claimed(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-032: an active claim by another owner denies; same owner passes."""
    del ctx
    ref = _pg._task_ref(command)
    actor = command.actor
    env_claims = _snapshot_active_claims(snapshot, ref.graph)
    existing = env_claims.get(ref.to_str())
    if existing is not None and existing != actor:
        return _issue(
            "R-PG-032",
            LkbErrorCode.ALREADY_CLAIMED,
            f"Task already claimed by {existing!r}",
            subject_ref=ref,
        )
    return _PASS


def _r_pg_033_agent_busy(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-033: single_active_task_per_agent policy (spec §6.4 #6 / LKB-CLAIM-006).

    When enabled, an agent that already holds an active claim on a
    *different* task is denied with ``agent_busy``.  Idempotent re-claim of
    the same task is handled by R-PG-032 (same owner -> pass).
    """
    del ctx
    ref = _pg._task_ref(command)
    actor = command.actor
    env_claims = _snapshot_active_claims(snapshot, ref.graph)
    if _board_policy_from_snapshot(snapshot).single_active_task_per_agent:
        owner_tasks = {tref for tref, owner in env_claims.items() if owner == actor and tref != ref.to_str()}
        if owner_tasks:
            return _issue(
                "R-PG-033",
                LkbErrorCode.AGENT_BUSY,
                f"Agent {actor!r} already has an active task: {sorted(owner_tasks)[0]}",
                subject_ref=ref,
            )
    return _PASS


# ── transfer_task (spec §5.6, §6.4, LKB-CLAIM-008/009) ───────────────


def _r_pg_034_invalid_transfer(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-034: transfer requires a new_owner that differs from the actor."""
    del snapshot, ctx
    ref = _pg._task_ref(command)
    new_owner = str(command.payload.get("new_owner", "") or "").strip()
    if not new_owner:
        return _issue(
            "R-PG-034",
            LkbErrorCode.INVALID_TRANSFER,
            "transfer_task requires a non-empty 'new_owner'",
            subject_ref=ref,
        )
    if new_owner == command.actor:
        return _issue(
            "R-PG-034",
            LkbErrorCode.INVALID_TRANSFER,
            "transfer_task new_owner must differ from actor (use claim_task instead)",
            subject_ref=ref,
        )
    return _PASS


# ── start_task (spec §6.5) ───────────────────────────────────────────


def _r_pg_040_start_blocked(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-040: a blocked task cannot be started."""
    return _blocked_issue(command, snapshot, ctx, "R-PG-040", "Task is blocked")


def _r_pg_041_start_already_resolved(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-041: a completed task cannot be started."""
    del ctx
    return _already_resolved_issue(command, snapshot, "R-PG-041")


def _r_pg_042_start_owner(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-042: start requires an active claim held by the caller."""
    del ctx
    ref = _pg._task_ref(command)
    node = _node_for(snapshot, ref)
    owner = node.owner if node is not None else None
    if owner != command.actor:
        code = LkbErrorCode.OWNER_REQUIRED if owner is None else LkbErrorCode.NOT_OWNER
        message = "Task must be claimed before start" if owner is None else (f"Task is owned by {owner!r}")
        return _issue("R-PG-042", code, message, subject_ref=ref)
    return _PASS


# ── complete_task (spec §6.6) ────────────────────────────────────────


def _r_pg_050_invalid_transition(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-050: only an in_progress task can complete."""
    del ctx
    ref = _pg._task_ref(command)
    node = _node_for(snapshot, ref)
    if node is None:
        return _issue("R-PG-050", LkbErrorCode.TASK_NOT_FOUND, "Task not found", subject_ref=ref)
    if node.state != "in_progress":
        return _issue(
            "R-PG-050",
            LkbErrorCode.INVALID_TRANSITION,
            "Task must be in_progress to complete",
            subject_ref=ref,
        )
    return _PASS


def _r_pg_052_complete_blocked(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-052: a task with active blockers cannot complete."""
    return _blocked_issue(command, snapshot, ctx, "R-PG-052", "Task has active blockers")


def _r_pg_053_complete_owner(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-053: completion requires the owner, or a reason for override."""
    del ctx
    ref = _pg._task_ref(command)
    node = _node_for(snapshot, ref)
    owner = node.owner if node is not None else None
    if owner != command.actor and not (command.reason and str(command.reason).strip()):
        code = LkbErrorCode.OWNER_REQUIRED if owner is None else LkbErrorCode.NOT_OWNER
        message = "Task must be claimed before completion" if owner is None else f"Task is owned by {owner!r}"
        return _issue("R-PG-053", code, message, subject_ref=ref)
    return _PASS


# ── delete_task (spec §6.8) ──────────────────────────────────────────


def _r_pg_060_dangling_dependency(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-060: deleting a referenced task requires cascade=true."""
    del ctx
    ref = _pg._task_ref(command)
    cascade = bool(command.payload.get("cascade", False))
    if not cascade:
        referencing = [
            e.edge_id
            for e in snapshot.edges.values()
            if e.type == "depends_on" and (e.source == ref or e.target == ref)
        ]
        if referencing:
            return _issue(
                "R-PG-060",
                LkbErrorCode.DANGLING_DEPENDENCY,
                "Task is referenced by dependencies; use cascade=true to remove",
                subject_ref=ref,
            )
    return _PASS


# ── revalidate (spec §6.7, Phase 6) ──────────────────────────────────


def _r_pg_070_revalidate_upstream(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-070: no revalidate while any prerequisite is unverified.

    Spec §6.7 / §13.5 Step 11: a task may not be revalidated (cleared out
    of ``needs_recheck``) while any prerequisite is still pending,
    in_progress, or itself ``needs_recheck``.  Each node recovers first,
    then progressively unblocks its downstream.
    """
    del ctx
    ref = _pg._task_ref(command)
    node = _node_for(snapshot, ref)
    if node is None:
        return _issue("R-PG-070", LkbErrorCode.TASK_NOT_FOUND, "Task not found", subject_ref=ref)
    unsatisfied: list[str] = []
    for prereq in _snapshot_prerequisites(snapshot, ref):
        prereq_node = snapshot.nodes.get(prereq)
        if prereq_node is None or not _node_is_verified_completed(prereq_node):
            unsatisfied.append(prereq.id)
    if unsatisfied:
        return _issue(
            "R-PG-070",
            LkbErrorCode.NEEDS_RECHECK,
            "Cannot revalidate: upstream not verified: " + ",".join(unsatisfied),
            subject_ref=ref,
        )
    return _PASS


# ── patch_task (spec §6.1, T2-GAP-09, LKB-ADAPT-011/012) ─────────────


def _r_pg_090_patch_structure(command: GraphCommand, snapshot: GraphSnapshot, ctx: RuleContext) -> RuleResult:
    """R-PG-090: patch structure — supported status, non-empty, known sub-kinds.

    The sub-intent decomposition itself stays in
    ``PatchTaskHandler._decompose``; the handler passes the decomposed
    sub-command kinds via ``context={"sub_kinds": (...)}`` so this rule can
    check them without re-running the decomposition.
    """
    del snapshot
    ref = _pg._task_ref(command)
    sub_kinds = tuple(ctx.extras.get("sub_kinds", ()))
    status = command.payload.get("status")
    if status is not None and str(status) not in _pg._STATUS_KIND:
        return _issue(
            "R-PG-090",
            LkbErrorCode.INVALID_STATUS,
            f"Unsupported task status {status!r}",
            subject_ref=ref,
        )
    if not sub_kinds:
        return _issue(
            "R-PG-090",
            LkbErrorCode.EMPTY_PATCH,
            "patch_task requires at least one sub-intent",
            subject_ref=ref,
        )
    for kind in sub_kinds:
        if kind not in ctx.known_kinds:
            return _issue(
                "R-PG-090",
                LkbErrorCode.UNKNOWN_COMMAND,
                f"Patch sub-intent maps to unknown kind {kind!r}",
                subject_ref=ref,
            )
    return _PASS


# ── rule registry ────────────────────────────────────────────────────
#
# Order IS the semantics: the solver short-circuits on the first denial,
# preserving each handler's original early-exit sequence.  Universal
# pre-gates (R-PG-001/002) run first for every command kind.

_PRE_GATE_RULES: tuple[Rule, ...] = (
    ("R-PG-001", _r_pg_001_plan_not_active),
    ("R-PG-002", _r_pg_002_stale_revision),
)

_KIND_RULES: dict[str, tuple[Rule, ...]] = {
    "create_task": (
        ("R-PG-010", _r_pg_010_task_shape),
        ("R-PG-011", _r_pg_011_duplicate_task),
    ),
    "update_task_fields": (("R-PG-003", _r_pg_003_task_not_found),),
    "add_dependency": (
        ("R-PG-020", _r_pg_020_unknown_task_reference),
        ("R-PG-021", _r_pg_021_self_dependency),
        ("R-PG-022", _r_pg_022_dependency_cycle),
    ),
    "remove_dependency": (),
    "claim_task": (
        ("R-PG-003", _r_pg_003_task_not_found),
        ("R-PG-004", _r_pg_004_needs_recheck),
        ("R-PG-031", _r_pg_031_claim_already_resolved),
        ("R-PG-030", _r_pg_030_claim_blocked),
        ("R-PG-032", _r_pg_032_already_claimed),
        ("R-PG-033", _r_pg_033_agent_busy),
    ),
    "release_task": (("R-PG-003", _r_pg_003_task_not_found),),
    "transfer_task": (
        ("R-PG-003", _r_pg_003_task_not_found),
        ("R-PG-034", _r_pg_034_invalid_transfer),
    ),
    "start_task": (
        ("R-PG-003", _r_pg_003_task_not_found),
        ("R-PG-004", _r_pg_004_needs_recheck),
        ("R-PG-041", _r_pg_041_start_already_resolved),
        ("R-PG-040", _r_pg_040_start_blocked),
        ("R-PG-042", _r_pg_042_start_owner),
    ),
    "complete_task": (
        ("R-PG-003", _r_pg_003_task_not_found),
        ("R-PG-050", _r_pg_050_invalid_transition),
        ("R-PG-052", _r_pg_052_complete_blocked),
        ("R-PG-053", _r_pg_053_complete_owner),
    ),
    "reopen_task": (("R-PG-003", _r_pg_003_task_not_found),),
    "delete_task": (
        ("R-PG-003", _r_pg_003_task_not_found),
        ("R-PG-060", _r_pg_060_dangling_dependency),
    ),
    "revalidate": (
        ("R-PG-003", _r_pg_003_task_not_found),
        ("R-PG-070", _r_pg_070_revalidate_upstream),
    ),
    "patch_task": (
        ("R-PG-003", _r_pg_003_task_not_found),
        ("R-PG-090", _r_pg_090_patch_structure),
    ),
}


def _accept_subject_ref(command: GraphCommand) -> NodeRef:
    """Subject ref carried by the accept run (matches historical handlers)."""
    if command.kind == "create_task":
        task_id = str(command.payload.get("task_id", ""))
        return plan_task_ref(task_id, graph_id=_pg._plan_graph_id(command))
    if command.kind == "add_dependency":
        dependent_id = str(command.payload.get("task_id") or command.payload.get("dependent") or "")
        return plan_task_ref(dependent_id, graph_id=_pg._plan_graph_id(command))
    if command.kind == "remove_dependency":
        dependent_id = str(command.payload.get("task_id", ""))
        return plan_task_ref(dependent_id, graph_id=_pg._plan_graph_id(command))
    return _pg._task_ref(command)


# ── solver ───────────────────────────────────────────────────────────


class PlanGraphLayer1Solver:
    """Layer1 solver for the Plan Graph command domain.

    ``evaluate`` runs the universal pre-gates followed by the kind-specific
    rules in registry order and returns the first denial, or an accept
    outcome carrying any derived facts.  Synchronous and in-process — no
    guard threads and no external solver calls.
    """

    solver_version = SOLVER_VERSION

    def known_kinds(self) -> frozenset[str]:
        """Command kinds with a registered rule set (== dispatcher handler kinds)."""
        return frozenset(_KIND_RULES)

    def rules_for(self, kind: str) -> tuple[Rule, ...]:
        """Ordered rules evaluated for *kind* (universal pre-gates first)."""
        return _PRE_GATE_RULES + _KIND_RULES.get(kind, ())

    def evaluate(
        self,
        command: GraphCommand,
        snapshot: GraphSnapshot,
        *,
        context: dict[str, Any] | None = None,
    ) -> PlanRuleOutcome:
        """Evaluate *command* against *snapshot*; empty ``issues`` == accepted."""
        ctx = RuleContext(
            snapshot,
            extras=dict(context or {}),
            known_kinds=self.known_kinds(),
        )
        derived: list[str] = []
        for _rule_id, check in self.rules_for(command.kind):
            result = check(command, snapshot, ctx)
            if result.issue is not None:
                return PlanRuleOutcome(issues=(result.issue,))
            if result.derived_facts:
                derived.extend(result.derived_facts)
        return PlanRuleOutcome(
            issues=(),
            derived_facts=tuple(derived),
            subject_ref=_accept_subject_ref(command),
        )


_solver: PlanGraphLayer1Solver | None = None


def plan_graph_layer1() -> PlanGraphLayer1Solver:
    """Return the shared Plan Graph Layer1 solver instance."""
    global _solver
    if _solver is None:
        _solver = PlanGraphLayer1Solver()
    return _solver
