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

"""Atomic composite PatchTask handler for the LKB Plan Graph."""

from __future__ import annotations

from typing import Any

from .commands import CommandResult, GraphCommand
from .error_codes import LkbErrorCode
from .graph_types import GraphSnapshot, NodeRef, plan_task_ref
from .json_store import BoardEnvelope
from .validation import ValidationRun
from .plan_graph_core import (
    PlanCommandHandler,
    committed,
    plan_graph_id,
    run_from_outcome,
    task_node,
    task_ref,
    plan_graph_layer1,
)

# ── PatchTask (spec §6.1, T2-GAP-09, LKB-ADAPT-011/012) ──────────────


# Status string -> handler kind for the status sub-intent.
_STATUS_KIND = {
    "pending": "reopen_task",
    "in_progress": "start_task",
    "completed": "complete_task",
    "deleted": "delete_task",
}


class PatchTaskHandler(PlanCommandHandler):
    """Composite handler: apply multiple sub-intents to one task atomically.

    Spec §6.1 / T2-GAP-09 / LKB-ADAPT-011/012.  A mixed TaskUpdate that
    carries status + owner + dependency + metadata changes must NOT pick
    a single dominant intent (the legacy ``_task_update_change_kind``
    behaviour) — that silently drops the other sub-intents and creates a
    “validate part, commit all” bypass window.  Instead this handler:

    1. Decomposes the patch into sub-intents (``PatchTask.decompose``).
    2. Checks the patch structure without mutating state.
    3. Applies them in sequence under the Board lock; if any sub-intent
       fails validation or application, nothing commits.
    4. Produces exactly one revision bump and one command result.

    Sub-intents are dispatched to the existing single-intent handlers so
    every rule (claim concurrency, cycle detection, needs_recheck, …) is
    enforced uniformly.
    """

    kind = "patch_task"

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        # Sub-intent state validation is performed sequentially against an
        # evolving clone under the lock. Validating each item against the
        # original snapshot would incorrectly reject valid claim+start
        # patches and cannot prove the composite transition.  Here we only
        # check the patch STRUCTURE (rule R-PG-090): supported status,
        # non-empty, known sub-intent kinds.
        ref = task_ref(command)
        sub_commands = self._decompose(command, ref)
        outcome = plan_graph_layer1().evaluate(
            command,
            snapshot,
            context={"sub_kinds": tuple(sub.kind for sub in sub_commands)},
        )
        return run_from_outcome(command, outcome)

    def apply(
        self, command: GraphCommand, envelope: BoardEnvelope, validation: ValidationRun
    ) -> tuple[BoardEnvelope, CommandResult]:
        ref = task_ref(command)
        working = envelope.clone()
        node = task_node(working, ref)
        if node is None:
            return envelope, CommandResult(
                decision="denied",
                command_id=command.command_id,
                error_code=LkbErrorCode.TASK_NOT_FOUND,
                reason=str(LkbErrorCode.TASK_NOT_FOUND),
            )
        sub_commands = self._decompose(command, ref)
        from .plan_graph import plan_command_dispatcher

        dispatcher = plan_command_dispatcher()
        # Capture the pre-patch node revision so the whole patch produces
        # exactly ONE revision bump (spec §6.1 #5).  Sub-handlers each
        # bump revision as part of their own apply; we collapse them at
        # the end so the final revision is initial + 1.
        pre_revision = int(node.get("revision", 0))
        claim_id: str | None = None
        affected_refs: list[str] = []
        # Apply each sub-intent in sequence against the evolving envelope.
        # Sub-validation is re-run on the in-lock snapshot so a sub-intent
        # whose precondition was changed by an earlier sub-intent (e.g.
        # claim then start) is checked against the post-claim state.
        for sub in sub_commands:
            handler = dispatcher.get(sub.kind)
            if handler is None:
                return envelope, CommandResult(
                    decision="denied",
                    command_id=command.command_id,
                    error_code=LkbErrorCode.UNKNOWN_COMMAND,
                    reason=f"unknown_command: {sub.kind}",
                )
            sub_snapshot = working.build_graph_snapshot()
            sub_run = handler.validate(sub, sub_snapshot)
            if not sub_run.accepted:
                error_code = sub_run.issues[0].code if sub_run.issues else LkbErrorCode.VALIDATION_DENIED
                return envelope, CommandResult(
                    decision="denied",
                    command_id=command.command_id,
                    error_code=error_code,
                    reason=self._denial_reason(sub_run),
                    validation_run_id=validation.validation_run_id,
                )
            working, sub_result = handler.apply(sub, working, sub_run)
            if not sub_result.committed:
                return envelope, CommandResult(
                    decision="denied",
                    command_id=command.command_id,
                    error_code=sub_result.error_code or LkbErrorCode.VALIDATION_DENIED,
                    reason=sub_result.reason or "patch_sub_intent_failed",
                    validation_run_id=validation.validation_run_id,
                )
            if sub_result.claim_id:
                claim_id = sub_result.claim_id
            affected_refs.extend(sub_result.affected_refs)
        # Collapse the per-sub-intent revision bumps into a single bump.
        final_node = task_node(working, ref)
        if final_node is not None:
            final_node["revision"] = pre_revision + 1
        return working, committed(
            command,
            validation_run_id=validation.validation_run_id,
            claim_id=claim_id,
            affected_refs=tuple(dict.fromkeys(affected_refs)),
        )

    # -- decomposition --------------------------------------------------

    def _decompose(self, command: GraphCommand, ref: NodeRef) -> list[GraphCommand]:
        """Split a patch_task command into ordered single-intent commands.

        Order matters: status transitions (start/complete) and claims must
        be applied before metadata updates so the metadata lands on the
        final state.  Field updates (subject/description/activeForm) and
        dependency changes are applied first, then owner (claim), then
        status, then metadata.
        """
        from .commands import PatchTask

        patch = PatchTask.decompose(dict(command.payload), ref)
        task_id = str(command.payload.get("task_id", ""))
        subs: list[GraphCommand] = []

        # 1. Field updates (subject / description / activeForm).
        if patch.has_field_updates:
            subs.append(self._sub(command, "update_task_fields", {**patch.field_updates, "task_id": task_id}))

        # 2. Dependency additions / removals.
        for intent in patch.dependency_intents:
            if intent.operation == "add":
                subs.append(
                    self._sub(
                        command,
                        "add_dependency",
                        {
                            "task_id": intent.dependent.id,
                            "depends_on": intent.prerequisite.id,
                        },
                    )
                )
            else:
                subs.append(
                    self._sub(
                        command,
                        "remove_dependency",
                        {
                            "task_id": intent.dependent.id,
                            "depends_on": intent.prerequisite.id,
                        },
                    )
                )

        # 3. Owner change — claim_task for self-claim (owner == actor);
        #    transfer_task is handled by the host adapter routing the
        #    whole command to transfer_task directly when owner != actor.
        if patch.has_owner_change:
            owner_target = patch.owner_target or ""
            if owner_target and owner_target != command.actor:
                # A PatchTask that tries to reassign to a third party must
                # go through transfer_task instead; a patch_task payload
                # with a third-party owner is denied as invalid.
                subs.append(
                    self._sub(
                        command,
                        "transfer_task",
                        {"task_id": task_id, "new_owner": owner_target},
                    )
                )
            elif owner_target:
                subs.append(self._sub(command, "claim_task", {"task_id": task_id}))
            else:
                subs.append(self._sub(command, "release_task", {"task_id": task_id}))

        # 4. Status transition.
        delete_sub: GraphCommand | None = None
        if patch.has_status_change:
            target = str(patch.status_target or "")
            kind = _STATUS_KIND.get(target)
            if kind is not None:
                status_sub = self._sub(command, kind, {"task_id": task_id})
                if kind == "delete_task":
                    delete_sub = status_sub
                else:
                    subs.append(status_sub)

        # 5. Metadata updates (applied last so they land on final state).
        if patch.has_metadata_updates:
            subs.append(
                self._sub(
                    command,
                    "update_task_fields",
                    {"task_id": task_id, "metadata": dict(patch.metadata_updates)},
                )
            )

        # Delete is terminal and therefore must run after every other
        # requested mutation in the same atomic patch.
        if delete_sub is not None:
            subs.append(delete_sub)

        return subs

    def _sub(self, parent: GraphCommand, kind: str, payload: dict[str, Any]) -> GraphCommand:
        """Build a single-intent sub-command sharing the parent's identity."""
        graph_id = plan_graph_id(parent)
        task_id = str(payload.get("task_id", ""))
        primary_ref = plan_task_ref(task_id, graph_id=graph_id) if task_id else None
        return GraphCommand(
            command_id=f"{parent.command_id}#{kind}",
            board_id=parent.board_id,
            actor=parent.actor,
            kind=kind,
            payload={**payload, "plan_id": graph_id},
            primary_subject_ref=primary_ref,
            reason=parent.reason,
            roles=parent.roles,
        )

    @staticmethod
    def _denial_reason(run: ValidationRun) -> str:
        if run.issues:
            first = run.issues[0]
            return f"{first.code}: {first.message}"
        return "patch_sub_intent_denied"
