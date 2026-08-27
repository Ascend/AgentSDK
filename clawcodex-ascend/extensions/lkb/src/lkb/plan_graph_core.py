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
# pylint: disable=E0402,W0621

"""Plan Graph domain command handlers (spec §6, Phase 4).

Each handler implements the two callbacks the
:class:`lkb.application.LkbApplicationService` needs:

* ``validate(command, snapshot) -> ValidationRun``  (lock-free).
* ``apply(command, envelope, validation) -> (envelope, CommandResult)``
  (runs under the Board File Lock and applies the domain mutation).

The :class:`PlanCommandDispatcher` maps ``command.kind`` to the matching
handler and exposes ``validate`` / ``apply`` callables suitable for
``LkbApplicationService.execute``.

Validation rules follow spec §6.2-§6.8; Claim concurrency, cycle
rejection and the single-active-task policy are centralized in
:mod:`lkb.plan_graph_rules` (the Plan Graph Layer1 solver) — every
handler's ``validate`` is a thin delegate to it. Invalidation propagation
is applied when completed work is reopened or its contract changes.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .plan_graph_rules import PlanRuleOutcome

from .commands import CommandResult, GraphCommand
from .error_codes import LkbErrorCode
from .graph_types import (
    BoardPolicy,
    GraphSnapshot,
    NodeRef,
    plan_task_ref,
)
from .json_store import BoardEnvelope
from .refs import DEFAULT_PLAN_GRAPH_ID
from .validation import ValidationIssue, ValidationRun

logger = logging.getLogger(__name__)

__all__ = [
    "PlanCommandHandler",
    "CreateTaskHandler",
    "UpdateTaskFieldsHandler",
    "AddDependencyHandler",
    "RemoveDependencyHandler",
    "ClaimTaskHandler",
    "committed",
    "plan_graph_id",
    "plan_graph_layer1",
    "run_from_outcome",
    "task_node",
    "task_ref",
]

_PLAN_GRAPH_ID = DEFAULT_PLAN_GRAPH_ID


# ── helpers ──────────────────────────────────────────────────────────


def _new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _plan_graph_id(command: GraphCommand) -> str:
    """Resolve the concrete Plan graph targeted by a command."""
    if command.primary_subject_ref is not None:
        return command.primary_subject_ref.graph
    raw = command.payload.get("plan_id") or command.payload.get("plan_graph_id")
    if isinstance(raw, str) and raw:
        NodeRef(raw, "_plan", "_id")
        return raw
    return _PLAN_GRAPH_ID


# Public integration helpers.  Private aliases remain below so existing
# internal call sites stay stable during the migration.
plan_graph_id = _plan_graph_id


def _ensure_plan_graph(env: BoardEnvelope, board_id: str, graph_id: str) -> None:
    if graph_id not in env.graphs:
        now = _now()
        env.graphs[graph_id] = {
            "graph_id": graph_id,
            "board_id": board_id,
            "graph_kind": "plan",
            "revision": 0,
            "created_at": now,
            "updated_at": now,
            "plan": {
                "plan_id": graph_id,
                "title": "Legacy plan" if graph_id == _PLAN_GRAPH_ID else graph_id,
                "state": "active",
                "session_ids": [],
            },
        }


def _agent_ref(actor: str, graph_id: str) -> NodeRef:
    """Canonical ``plan:agent:<actor>`` NodeRef for a claim owner.

    Spec §5.6 / §5.3 - ``Claim.owner_ref`` is a :class:`NodeRef`, never a
    bare actor string.  The agent id is the actor verbatim so the
    Store invariant ``active_claim.owner_ref.id == task_node.owner`` holds.
    """
    return NodeRef(graph_id, "agent", str(actor))


def _agent_ref_str(actor: str, graph_id: str) -> str:
    return _agent_ref(actor, graph_id).to_str()


def _task_ref(command: GraphCommand) -> NodeRef:
    """Resolve the task NodeRef targeted by *command*."""
    if command.primary_subject_ref is not None:
        return command.primary_subject_ref
    task_id = str(command.payload.get("task_id", ""))
    return plan_task_ref(task_id, graph_id=_plan_graph_id(command))


task_ref = _task_ref


def _task_node(env: BoardEnvelope, ref: NodeRef) -> dict[str, Any] | None:
    """Return the raw node dict for *ref*, or None."""
    for nid, node in env.nodes.items():
        if str(node.get("ref", "")) == ref.to_str():
            return node
    return None


task_node = _task_node


def _active_claim_for(env: BoardEnvelope, task_ref: NodeRef) -> dict[str, Any] | None:
    target = task_ref.to_str()
    for claim in env.claims.values():
        if str(claim.get("task_ref", "")) == target and claim.get("status") == "active":
            return claim
    return None


def _depends_on_edges(env: BoardEnvelope) -> list[tuple[str, str, str]]:
    """Return ``(edge_id, source_ref_str, target_ref_str)`` for depends_on."""
    out: list[tuple[str, str, str]] = []
    for eid, edge in env.edges.items():
        if edge.get("type") == "depends_on":
            out.append((eid, str(edge.get("source", "")), str(edge.get("target", ""))))
    return out


def _would_create_cycle(env: BoardEnvelope, source_ref: str, target_ref: str) -> list[str] | None:
    """Return the cycle path if adding ``source -> target`` creates one.

    ``depends_on`` runs dependent -> prerequisite.  Adding ``source ->
    target`` closes a cycle iff ``target`` can already reach ``source``
    along existing depends_on edges.  Returns the path (including the
    closing edge) or None.
    """
    adjacency: dict[str, list[str]] = {}
    for _eid, src, tgt in _depends_on_edges(env):
        adjacency.setdefault(src, []).append(tgt)
    # DFS from target; if we reach source, there is a cycle.
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


def _run(
    command: GraphCommand,
    *,
    accepted: bool,
    subject_ref: NodeRef | None = None,
    issues: tuple[ValidationIssue, ...] = (),
    derived_facts: tuple[str, ...] = (),
) -> ValidationRun:
    return ValidationRun(
        validation_run_id=_new_id("V-"),
        proposal_id=command.command_id,
        subject_ref=subject_ref,
        result="pass" if accepted else "denied",
        issues=issues,
        derived_facts=derived_facts,
        engine="plan-graph",
    )


def _committed(
    command: GraphCommand,
    *,
    validation_run_id: str | None = None,
    claim_id: str | None = None,
    affected_refs: tuple[str, ...] = (),
) -> CommandResult:
    return CommandResult(
        decision="committed",
        command_id=command.command_id,
        validation_run_id=validation_run_id,
        claim_id=claim_id,
        affected_refs=affected_refs,
    )


committed = _committed


def _denied_issue(
    code: LkbErrorCode,
    message: str,
    *,
    rule: str = "plan",
    subject_ref: NodeRef | None = None,
    blockers: tuple[str, ...] = (),
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        rule=rule,
        subject_ref=subject_ref,
        blockers=blockers,
    )


# ── invalidation propagation ─────────────────────────────────────────


def _downstream_closure(env: BoardEnvelope, task_ref: NodeRef, *, mode: str = "cascade") -> list[NodeRef]:
    """Tasks that depend on *task_ref*, scoped by *mode* (spec §5.1).

    ``depends_on`` runs dependent -> prerequisite, so the downstream of
    *task_ref* (those that depend on it) are the sources of edges whose
    target is *task_ref*.

    ``mode`` (BoardPolicy.invalidation_mode):
      * ``off``    - no propagation; return ``[]``.
      * ``direct`` - only direct dependents (one hop).
      * ``cascade``- transitive closure (default, spec §6.7).
    """
    if mode == "off":
        return []
    target_to_sources: dict[str, list[str]] = {}
    for _eid, src, tgt in _depends_on_edges(env):
        target_to_sources.setdefault(tgt, []).append(src)
    if mode == "direct":
        return [NodeRef.from_str(s) for s in target_to_sources.get(task_ref.to_str(), [])]
    result: list[NodeRef] = []
    seen: set[str] = set()
    stack = [task_ref.to_str()]
    while stack:
        current = stack.pop()
        for src in target_to_sources.get(current, []):
            if src in seen:
                continue
            seen.add(src)
            stack.append(src)
            result.append(NodeRef.from_str(src))
    return result


def _mark_needs_recheck(env: BoardEnvelope, ref: NodeRef, cause: NodeRef, reason: str) -> bool:
    """Mark a completed task ``derived_status=needs_recheck``.

    Used when a completed task's own contract changes or upstream work is
    reopened. The task keeps ``base_status=completed``. Returns True if the
    task was changed.
    """
    node = _task_node(env, ref)
    if node is None or node.get("state") != "completed":
        return False
    payload = node.get("payload") if isinstance(node.get("payload"), dict) else {}
    payload["derived_status"] = "needs_recheck"
    payload["invalidation_cause"] = cause.to_str()
    payload["invalidation_reason"] = reason
    node["payload"] = payload
    node["updated_at"] = _now()
    return True


def _invalidate_downstream(
    env: BoardEnvelope,
    root_ref: NodeRef,
    reason: str,
    *,
    mode: str = "cascade",
) -> list[NodeRef]:
    """Mark completed downstream tasks ``needs_recheck``.

    Pending/in_progress downstream tasks become blocked naturally (the
    prerequisite is pending again).  Completed downstream tasks keep
    ``base_status=completed`` but gain ``derived_status=needs_recheck``.
    Returns the affected refs (propagation path). ``mode`` scopes the
    closure (``off`` / ``direct`` / ``cascade``).
    """
    affected: list[NodeRef] = []
    now = _now()
    for ref in _downstream_closure(env, root_ref, mode=mode):
        node = _task_node(env, ref)
        if node is None:
            continue
        if node.get("state") != "completed":
            continue
        if _mark_needs_recheck(env, ref, root_ref, reason):
            # _mark_needs_recheck already set updated_at; keep the single
            # timestamp consistent across the propagation path.
            node["updated_at"] = now
            affected.append(ref)
    return affected


def _invalidation_event(
    envelope: BoardEnvelope,
    command: GraphCommand,
    validation: ValidationRun,
    cause_ref: NodeRef,
    reason: str,
    affected: list[NodeRef],
) -> dict[str, Any]:
    """Build an ``invalidation_propagation`` audit event (spec §6.10).

    ``store_revision`` is filled from the candidate envelope; the store
    layer patches every command-scoped event to the post-bump revision
    after ``execute_atomic`` advances it (issue #9: override / invalidation
    events must not record the pre-increment store_revision).
    """
    return {
        "type": "invalidation_propagation",
        "event_id": f"E-{uuid.uuid4().hex[:16]}",
        "board_id": envelope.board_id(),
        "command_id": command.command_id,
        "decision": "committed",
        "actor": command.actor,
        "subject_ref": cause_ref.to_str(),
        "cause": cause_ref.to_str(),
        "reason": reason,
        "affected_refs": [r.to_str() for r in affected],
        "store_revision": envelope.store_revision,
        "validation_run_id": validation.validation_run_id,
        "timestamp": _now(),
    }


# ── handler base ─────────────────────────────────────────────────────


class PlanCommandHandler:
    """Base class for Plan Graph command handlers.

    Validation logic lives in :mod:`lkb.plan_graph_rules` (the Plan Graph
    Layer1 solver); each handler's ``validate`` is a thin delegate via
    :func:`_run_from_outcome`.  ``apply`` keeps its lock-critical-section
    re-checks local (spec §6.4) and never invokes the solver.
    """

    kind: str = ""

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        return run_from_outcome(command, plan_graph_layer1().evaluate(command, snapshot))

    def apply(
        self, command: GraphCommand, envelope: BoardEnvelope, validation: ValidationRun
    ) -> tuple[BoardEnvelope, CommandResult]:
        raise NotImplementedError


def _run_from_outcome(command: GraphCommand, outcome: "PlanRuleOutcome") -> ValidationRun:
    """Build the ValidationRun for a :class:`PlanRuleOutcome` (engine unchanged)."""
    if outcome.issues:
        return _run(
            command,
            accepted=False,
            subject_ref=outcome.issues[0].subject_ref,
            issues=outcome.issues,
        )
    return _run(
        command,
        accepted=True,
        subject_ref=outcome.subject_ref,
        derived_facts=outcome.derived_facts,
    )


run_from_outcome = _run_from_outcome


def plan_graph_layer1():
    """Return the shared Plan Graph Layer1 solver (lazy import avoids a cycle)."""
    from .plan_graph_rules import plan_graph_layer1 as _get_solver

    return _get_solver()


# ── CreateTask (spec §6.2) ───────────────────────────────────────────


class CreateTaskHandler(PlanCommandHandler):
    kind = "create_task"

    def apply(
        self, command: GraphCommand, envelope: BoardEnvelope, validation: ValidationRun
    ) -> tuple[BoardEnvelope, CommandResult]:
        task_id = str(command.payload.get("task_id", "")) or _new_id("T-")
        graph_id = _plan_graph_id(command)
        ref = plan_task_ref(task_id, graph_id=graph_id)
        _ensure_plan_graph(envelope, command.board_id, graph_id)
        now = _now()
        subject = str(command.payload.get("subject", ""))
        node_key = task_id if graph_id == _PLAN_GRAPH_ID else ref.to_str()
        envelope.nodes[node_key] = {
            "ref": ref.to_str(),
            "title": subject,
            "state": "pending",
            "owner": None,
            "revision": 1,
            "payload": {
                "subject": subject,
                "description": str(command.payload.get("description", "")),
                "activeForm": str(command.payload.get("activeForm", subject)),
                "base_status": "pending",
                "metadata": dict(command.payload.get("metadata", {}) or {}),
                "output": "",
            },
            "created_at": now,
            "updated_at": now,
        }
        return envelope, _committed(command, validation_run_id=validation.validation_run_id)


# ── UpdateTaskFields (spec §6.1) ─────────────────────────────────────


class UpdateTaskFieldsHandler(PlanCommandHandler):
    kind = "update_task_fields"

    def apply(
        self, command: GraphCommand, envelope: BoardEnvelope, validation: ValidationRun
    ) -> tuple[BoardEnvelope, CommandResult]:
        ref = _task_ref(command)
        node = _task_node(envelope, ref)
        if node is None:
            return envelope, CommandResult(
                decision="denied",
                command_id=command.command_id,
                error_code=LkbErrorCode.TASK_NOT_FOUND,
                reason=str(LkbErrorCode.TASK_NOT_FOUND),
            )
        raw_payload = node.get("payload")
        if isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            logger.warning(
                "Replacing non-object payload while updating task %s (got %s)",
                ref.to_str(),
                type(raw_payload).__name__,
            )
            payload = {}
        contract_changed = False
        for field in ("subject", "description", "activeForm"):
            if field in command.payload:
                payload[field] = command.payload[field]
        if "subject" in command.payload:
            node["title"] = str(command.payload["subject"])
            contract_changed = True
        if "description" in command.payload:
            contract_changed = True
        if "metadata" in command.payload:
            merged_metadata = dict(payload.get("metadata", {}) or {})
            for key, value in dict(command.payload["metadata"] or {}).items():
                if value is None:
                    merged_metadata.pop(key, None)
                else:
                    merged_metadata[key] = value
            payload["metadata"] = merged_metadata
        node["payload"] = payload
        node["revision"] = int(node.get("revision", 0)) + 1
        node["updated_at"] = _now()
        # Spec §6.7 / T2-GAP-06: a completed task whose contract
        # (subject / description) changed must be marked
        # ``needs_recheck`` and invalidation propagated to completed
        # downstream tasks. Independent branches are untouched.
        if contract_changed and node.get("state") == "completed":
            mode = _board_policy(envelope).invalidation_mode
            reason = str(command.reason or "contract changed")
            _mark_needs_recheck(envelope, ref, ref, reason)
            affected = _invalidate_downstream(envelope, ref, reason, mode=mode)
            if affected:
                envelope.events.append(_invalidation_event(envelope, command, validation, ref, reason, affected))
        return envelope, _committed(command, validation_run_id=validation.validation_run_id)


# ── AddDependency (spec §6.3) ────────────────────────────────────────


class AddDependencyHandler(PlanCommandHandler):
    kind = "add_dependency"

    def apply(
        self, command: GraphCommand, envelope: BoardEnvelope, validation: ValidationRun
    ) -> tuple[BoardEnvelope, CommandResult]:
        graph_id = _plan_graph_id(command)
        dependent_id = str(command.payload.get("task_id") or command.payload.get("dependent") or "")
        prerequisite_id = str(command.payload.get("depends_on") or command.payload.get("prerequisite") or "")
        dependent = plan_task_ref(dependent_id, graph_id=graph_id)
        prerequisite = plan_task_ref(prerequisite_id, graph_id=graph_id)
        _ensure_plan_graph(envelope, command.board_id, graph_id)
        # Idempotent: skip if the edge already exists.
        for eid, edge in envelope.edges.items():
            if (
                edge.get("type") == "depends_on"
                and str(edge.get("source", "")) == dependent.to_str()
                and str(edge.get("target", "")) == prerequisite.to_str()
            ):
                return envelope, _committed(command, validation_run_id=validation.validation_run_id)
        edge_id = (
            f"dep-{dependent_id}-{prerequisite_id}"
            if graph_id == _PLAN_GRAPH_ID
            else f"{graph_id}-dep-{dependent_id}-{prerequisite_id}"
        )
        envelope.edges[edge_id] = {
            "edge_id": edge_id,
            "graph": graph_id,
            "type": "depends_on",
            "source": dependent.to_str(),
            "target": prerequisite.to_str(),
            "revision": 1,
            "payload": {},
        }
        # Spec §6.7 / T2-GAP-06: adding a new prerequisite to a COMPLETED
        # task invalidates its completion - it now depends on work that may
        # not be done.  Mark it ``needs_recheck`` and propagate to
        # completed downstream.  (Adding a dependency to a pending/in_progress
        # task just makes it blocked naturally - no invalidation needed.)
        dep_node = _task_node(envelope, dependent)
        if dep_node is not None and dep_node.get("state") == "completed":
            mode = _board_policy(envelope).invalidation_mode
            reason = str(command.reason or "new dependency added")
            _mark_needs_recheck(envelope, dependent, dependent, reason)
            affected = _invalidate_downstream(envelope, dependent, reason, mode=mode)
            if affected:
                envelope.events.append(_invalidation_event(envelope, command, validation, dependent, reason, affected))
        return envelope, _committed(command, validation_run_id=validation.validation_run_id)


# ── RemoveDependency ─────────────────────────────────────────────────


class RemoveDependencyHandler(PlanCommandHandler):
    kind = "remove_dependency"

    def apply(
        self, command: GraphCommand, envelope: BoardEnvelope, validation: ValidationRun
    ) -> tuple[BoardEnvelope, CommandResult]:
        graph_id = _plan_graph_id(command)
        dependent_id = str(command.payload.get("task_id", ""))
        prerequisite_id = str(command.payload.get("depends_on", ""))
        dependent = plan_task_ref(dependent_id, graph_id=graph_id)
        prerequisite = plan_task_ref(prerequisite_id, graph_id=graph_id)
        to_remove = [
            eid
            for eid, edge in envelope.edges.items()
            if edge.get("type") == "depends_on"
            and str(edge.get("source", "")) == dependent.to_str()
            and str(edge.get("target", "")) == prerequisite.to_str()
        ]
        for eid in to_remove:
            envelope.edges.pop(eid, None)
        return envelope, _committed(command, validation_run_id=validation.validation_run_id)


# ── ClaimTask (spec §6.4) ────────────────────────────────────────────


class ClaimTaskHandler(PlanCommandHandler):
    kind = "claim_task"

    def apply(
        self, command: GraphCommand, envelope: BoardEnvelope, validation: ValidationRun
    ) -> tuple[BoardEnvelope, CommandResult]:
        ref = _task_ref(command)
        node = _task_node(envelope, ref)
        if node is None:
            return envelope, CommandResult(
                decision="denied",
                command_id=command.command_id,
                error_code=LkbErrorCode.TASK_NOT_FOUND,
                reason=str(LkbErrorCode.TASK_NOT_FOUND),
            )
        # Idempotent: same actor already holds an active claim.
        existing = _active_claim_for(envelope, ref)
        if existing is not None and existing.get("owner_ref") == _agent_ref_str(command.actor, ref.graph):
            return envelope, _committed(
                command,
                validation_run_id=validation.validation_run_id,
                claim_id=str(existing.get("claim_id") or "") or None,
            )
        claim_id = _new_id("C-")
        now = _now()
        envelope.claims[claim_id] = {
            "task_ref": ref.to_str(),
            "owner_ref": _agent_ref_str(command.actor, ref.graph),
            "claim_id": claim_id,
            "claimed_at": now,
            "claim_revision": int(node.get("revision", 0)),
            "status": "active",
            "released_at": "",
            "reason": str(command.reason or ""),
        }
        node["owner"] = command.actor
        node["revision"] = int(node.get("revision", 0)) + 1
        node["updated_at"] = now
        return envelope, _committed(
            command,
            validation_run_id=validation.validation_run_id,
            claim_id=claim_id,
        )


# ── ReleaseTask (spec §5.6) ──────────────────────────────────────────


def _board_policy(envelope: BoardEnvelope) -> BoardPolicy:
    """Load the board policy from the envelope (defence-in-depth).

    Validator runs lock-free without policy access, so authorization for
    force-override (Release / Transfer) is enforced under the Board lock
    in ``apply`` (spec §6.4: authorization must be checked while
    holding the Board lock).
    """
    board = envelope.board if isinstance(envelope.board, dict) else {}
    policy_dict = board.get("policy") if isinstance(board.get("policy"), dict) else {}
    return BoardPolicy.from_dict(policy_dict)


def _authorize_override(
    command: GraphCommand,
    envelope: BoardEnvelope,
    current_owner: str | None,
) -> LkbErrorCode | None:
    """Return ``None`` if *command* may override *current_owner*, else a
    denial code.

    Rules (spec §5.6, LKB-CLAIM-008/009):
      - actor is the current owner → allowed (no override needed);
      - a host-asserted actor role is in ``force_override_roles`` AND provides a reason → allowed;
      - an authorized role is present but no reason →
        ``override_reason_required``;
      - no authorized role is present → ``override_not_authorized``.
    """
    if current_owner is not None and current_owner == command.actor:
        return None
    policy = _board_policy(envelope)
    policy_roles = tuple(policy.force_override_roles)
    authorized = "*" in policy_roles or any(policy.allows_force_override(role) for role in command.roles)
    if not authorized:
        return LkbErrorCode.OVERRIDE_NOT_AUTHORIZED
    if not (command.reason and str(command.reason).strip()):
        return LkbErrorCode.OVERRIDE_REASON_REQUIRED
    return None
