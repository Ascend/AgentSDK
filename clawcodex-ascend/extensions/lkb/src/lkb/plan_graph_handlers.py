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

"""Single-intent Plan Graph lifecycle command handlers."""

from __future__ import annotations

import uuid

from .commands import CommandResult, GraphCommand
from .error_codes import LkbErrorCode
from .graph_types import GraphSnapshot
from .json_store import BoardEnvelope
from .validation import ValidationRun
from .plan_graph_core import (
    PlanCommandHandler,
    _agent_ref_str,
    _authorize_override,
    _board_policy,
    _committed,
    _invalidate_downstream,
    _invalidation_event,
    _new_id,
    _now,
    _run_from_outcome,
    _task_node,
    _task_ref,
    plan_graph_layer1,
)


class ReleaseTaskHandler(PlanCommandHandler):
    kind = "release_task"

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        return _run_from_outcome(command, plan_graph_layer1().evaluate(command, snapshot))

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
        current_owner = node.get("owner") or None
        if current_owner is None:
            return envelope, _committed(
                command,
                validation_run_id=validation.validation_run_id,
            )
        denial = _authorize_override(command, envelope, current_owner)
        if denial is not None:
            return envelope, CommandResult(
                decision="denied",
                command_id=command.command_id,
                error_code=denial,
                reason=str(denial),
            )
        now = _now()
        released_any = False
        for claim in envelope.claims.values():
            if str(claim.get("task_ref", "")) == ref.to_str() and claim.get("status") == "active":
                claim["status"] = "released"
                claim["released_at"] = now
                claim["reason"] = str(command.reason or claim.get("reason", ""))
                released_any = True
        if released_any:
            node["owner"] = None
            node["revision"] = int(node.get("revision", 0)) + 1
            node["updated_at"] = now
            # Override audit (only when actor != previous owner).
            if current_owner is not None and current_owner != command.actor:
                envelope.events.append(
                    {
                        "type": "claim_override",
                        "event_id": f"E-{uuid.uuid4().hex[:16]}",
                        "board_id": envelope.board_id(),
                        "command_id": command.command_id,
                        "decision": "committed",
                        "action": "release",
                        "actor": command.actor,
                        "reason": str(command.reason or ""),
                        "subject_ref": ref.to_str(),
                        "previous_owner": current_owner,
                        "task_ref": ref.to_str(),
                        "store_revision": envelope.store_revision,
                        "validation_run_id": validation.validation_run_id,
                        "timestamp": now,
                    }
                )
        return envelope, _committed(command, validation_run_id=validation.validation_run_id)


# ── TransferTask (spec §5.6, §6.4, LKB-CLAIM-008/009) ────────────────


class TransferTaskHandler(PlanCommandHandler):
    """Force-transfer ownership of a task to another agent.

    Spec §6.4: assignment to another agent requires Assign/Transfer
    permission and a dedicated audit record.
    Unlike Claim (which only assigns to the current actor), Transfer
    assigns to an arbitrary ``new_owner`` and therefore always requires
    ``force_override_roles`` authorization + ``reason`` + an override
    audit event — even when the actor happens to be the current owner
    (re-assigning to a third party is never a self-claim).
    """

    kind = "transfer_task"

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        return _run_from_outcome(command, plan_graph_layer1().evaluate(command, snapshot))

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
        new_owner = str(command.payload.get("new_owner", "") or "").strip()
        if not new_owner or new_owner == command.actor:
            return envelope, CommandResult(
                decision="denied",
                command_id=command.command_id,
                error_code=LkbErrorCode.INVALID_TRANSFER,
                reason=str(LkbErrorCode.INVALID_TRANSFER),
            )
        current_owner = node.get("owner") or None
        denial = _authorize_override(command, envelope, current_owner)
        if denial is not None:
            return envelope, CommandResult(
                decision="denied",
                command_id=command.command_id,
                error_code=denial,
                reason=str(denial),
            )
        now = _now()
        # Release the existing active claim (if any) in the same atomic snapshot.
        for claim in envelope.claims.values():
            if str(claim.get("task_ref", "")) == ref.to_str() and claim.get("status") == "active":
                claim["status"] = "overridden"
                claim["released_at"] = now
                claim["reason"] = str(command.reason or "transferred")
        # Create a new active claim for new_owner.
        claim_id = _new_id("C-")
        envelope.claims[claim_id] = {
            "task_ref": ref.to_str(),
            "owner_ref": _agent_ref_str(new_owner, ref.graph),
            "claim_id": claim_id,
            "claimed_at": now,
            "claim_revision": int(node.get("revision", 0)),
            "status": "active",
            "released_at": "",
            "reason": str(command.reason or "transferred"),
        }
        node["owner"] = new_owner
        node["revision"] = int(node.get("revision", 0)) + 1
        node["updated_at"] = now
        # Override audit (always recorded for Transfer — spec §6.4).
        envelope.events.append(
            {
                "type": "claim_override",
                "event_id": f"E-{uuid.uuid4().hex[:16]}",
                "board_id": envelope.board_id(),
                "command_id": command.command_id,
                "decision": "committed",
                "action": "transfer",
                "actor": command.actor,
                "reason": str(command.reason or ""),
                "subject_ref": ref.to_str(),
                "previous_owner": current_owner or "",
                "new_owner": new_owner,
                "task_ref": ref.to_str(),
                "store_revision": envelope.store_revision,
                "validation_run_id": validation.validation_run_id,
                "timestamp": now,
            }
        )
        return envelope, _committed(
            command,
            validation_run_id=validation.validation_run_id,
            claim_id=claim_id,
        )


# ── StartTask (spec §6.5) ────────────────────────────────────────────


class StartTaskHandler(PlanCommandHandler):
    kind = "start_task"

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        return _run_from_outcome(command, plan_graph_layer1().evaluate(command, snapshot))

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
        owner = node.get("owner") or None
        if owner != command.actor:
            error_code = LkbErrorCode.OWNER_REQUIRED if owner is None else LkbErrorCode.NOT_OWNER
            return envelope, CommandResult(
                decision="denied",
                command_id=command.command_id,
                error_code=error_code,
                reason=str(error_code),
            )
        if node.get("state") == "in_progress":
            return envelope, _committed(command, validation_run_id=validation.validation_run_id)
        node["state"] = "in_progress"
        payload = node.get("payload") if isinstance(node.get("payload"), dict) else {}
        payload["base_status"] = "in_progress"
        node["payload"] = payload
        node["revision"] = int(node.get("revision", 0)) + 1
        node["updated_at"] = _now()
        return envelope, _committed(command, validation_run_id=validation.validation_run_id)


# ── CompleteTask (spec §6.6) ─────────────────────────────────────────


class CompleteTaskHandler(PlanCommandHandler):
    kind = "complete_task"

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        return _run_from_outcome(command, plan_graph_layer1().evaluate(command, snapshot))

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
        current_owner = node.get("owner") or None
        if current_owner != command.actor:
            denial = _authorize_override(command, envelope, current_owner)
            if denial is not None:
                return envelope, CommandResult(
                    decision="denied",
                    command_id=command.command_id,
                    error_code=denial,
                    reason=str(denial),
                )
        now = _now()
        node["state"] = "completed"
        payload = node.get("payload") if isinstance(node.get("payload"), dict) else {}
        payload["base_status"] = "completed"
        node["payload"] = payload
        node["revision"] = int(node.get("revision", 0)) + 1
        node["updated_at"] = now
        # Complete the active claim in the same atomic snapshot.
        for claim in envelope.claims.values():
            if str(claim.get("task_ref", "")) == ref.to_str() and claim.get("status") == "active":
                claim["status"] = "completed"
                claim["released_at"] = now
        return envelope, _committed(command, validation_run_id=validation.validation_run_id)


# ── ReopenTask (spec §6.7) ───────────────────────────────────────────


class ReopenTaskHandler(PlanCommandHandler):
    kind = "reopen_task"

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        return _run_from_outcome(command, plan_graph_layer1().evaluate(command, snapshot))

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
        now = _now()
        node["state"] = "pending"
        payload = node.get("payload") if isinstance(node.get("payload"), dict) else {}
        payload["base_status"] = "pending"
        node["payload"] = payload
        node["revision"] = int(node.get("revision", 0)) + 1
        node["updated_at"] = now
        # Release any active claim on the reopened task and clear the
        # owner: ownership must not survive the claim release (Store
        # invariant ``active_claim.owner_ref.id == task_node.owner``).
        # The previous owner must re-claim before restarting - otherwise a
        # reopen would let them bypass the claim protocol and start
        # directly.
        for claim in envelope.claims.values():
            if str(claim.get("task_ref", "")) == ref.to_str() and claim.get("status") == "active":
                claim["status"] = "released"
                claim["released_at"] = now
                claim["reason"] = str(command.reason or "reopened")
        node["owner"] = None
        # Propagate invalidation to completed downstream tasks (spec §6.7):
        # they keep base=completed but become derived=needs_recheck.
        # Independent branches are untouched.
        # The propagation scope honours BoardPolicy.invalidation_mode
        # (off / direct / cascade) - spec §5.1, §6.7.
        mode = _board_policy(envelope).invalidation_mode
        reason = str(command.reason or "upstream reopened")
        affected = _invalidate_downstream(envelope, ref, reason, mode=mode)
        if affected:
            envelope.events.append(_invalidation_event(envelope, command, validation, ref, reason, affected))
        return envelope, _committed(command, validation_run_id=validation.validation_run_id)


# ── DeleteTask (spec §6.8) ───────────────────────────────────────────


class DeleteTaskHandler(PlanCommandHandler):
    kind = "delete_task"

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        return _run_from_outcome(command, plan_graph_layer1().evaluate(command, snapshot))

    def apply(
        self, command: GraphCommand, envelope: BoardEnvelope, validation: ValidationRun
    ) -> tuple[BoardEnvelope, CommandResult]:
        ref = _task_ref(command)
        cascade = bool(command.payload.get("cascade", False))
        affected_refs = [ref.to_str()]
        # Remove the node.
        removed = [nid for nid, node in envelope.nodes.items() if str(node.get("ref", "")) == ref.to_str()]
        for nid in removed:
            envelope.nodes.pop(nid, None)
        # Remove or cascade referencing edges.
        if cascade:
            to_remove = [
                eid
                for eid, edge in envelope.edges.items()
                if edge.get("type") == "depends_on"
                and (str(edge.get("source", "")) == ref.to_str() or str(edge.get("target", "")) == ref.to_str())
            ]
            for edge_id in to_remove:
                edge = envelope.edges[edge_id]
                affected_refs.extend(
                    str(endpoint) for endpoint in (edge.get("source", ""), edge.get("target", "")) if endpoint
                )
            for eid in to_remove:
                envelope.edges.pop(eid, None)
        # Drop every claim record referencing the deleted task — active
        # ones included.  Keeping terminal (completed/released/overridden)
        # claims would leave dangling ``task_ref`` pointers to a node that
        # no longer exists, and they would wrongly attach to a re-created
        # task reusing the same id.  The delete command itself stays in the
        # event log, so no audit history is lost.
        dangling = [cid for cid, claim in envelope.claims.items() if str(claim.get("task_ref", "")) == ref.to_str()]
        for cid in dangling:
            envelope.claims.pop(cid, None)
        return envelope, _committed(
            command,
            validation_run_id=validation.validation_run_id,
            affected_refs=tuple(dict.fromkeys(affected_refs)),
        )


# ── Revalidate (spec §6.7, Phase 6) ──────────────────────────────────


class RevalidateHandler(PlanCommandHandler):
    kind = "revalidate"

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        # R-PG-070 requires every direct prerequisite to be current.
        return _run_from_outcome(command, plan_graph_layer1().evaluate(command, snapshot))

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
        payload = node.get("payload") if isinstance(node.get("payload"), dict) else {}
        had_stale_derived = payload.get("derived_status") in ("needs_recheck", "needs_review")
        payload.pop("derived_status", None)
        payload.pop("invalidation_cause", None)
        payload.pop("invalidation_reason", None)
        node["payload"] = payload
        # Only bump revision if we actually cleared a stale derived flag -
        # a revalidate of an already-clean task is a no-op (spec §11.4
        # inv 5: no content change -> no revision bump).
        if had_stale_derived:
            node["revision"] = int(node.get("revision", 0)) + 1
            node["updated_at"] = _now()
        return envelope, _committed(command, validation_run_id=validation.validation_run_id)
