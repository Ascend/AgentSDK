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
# pylint: disable=E0402,E1111

"""LkbApplicationService: two-phase optimistic commit (spec §7.6, Phase 3).

Single entry point that turns a :class:`GraphCommand` into an atomically
committed :class:`CommandResult`.  The flow is:

1. **Lock-free read** - ``repository.load_snapshot`` returns a
   :class:`GraphSnapshot` together with its ``revision_vector`` (``R``)
   and content hash.
2. **Lock-free validation** - deterministic Plan Graph rules run here,
   *never* while the Board File Lock is held. The resulting
   :class:`ValidationRun` is bound to ``R`` and the snapshot hash it
   actually read (spec §5.10, §7.6).
3. **Denied** - persist a denial audit (CAS-free; only ``store_revision``
   advances) and return ``denied``.
4. **Accepted** - ``repository.execute_atomic`` with ``R`` as the CAS
   guard.  The ``apply`` callback runs *under* the Board lock, applies
   the domain mutation, records the ``ValidationRun``, and returns the
   candidate envelope + committed result.  The Store performs the
   ``os.replace``; only on success is ``committed`` returned.
5. **Revision conflict** - ``StaleRevisionError`` (another writer
   committed first) triggers a bounded retry from step 1.

This eliminates the "LKB records ``committed`` then the native write
fails" window (T2-GAP-09): ``committed`` is only ever produced after the
Store's atomic replace succeeds, and every state change is traceable to
command + validation + commit (LKB-AUDIT-001..003).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol, runtime_checkable

from .commands import CommandResult, GraphCommand, compute_request_hash, validate_request_hash
from .error_codes import LkbErrorCode, coerce_error_code
from .graph_types import GraphSnapshot, RevisionVector
from .json_store import BoardEnvelope, StaleRevisionError
from .repository import LkbRepository
from .validation import ValidationIssue, ValidationRun

__all__ = [
    "CommandValidator",
    "CommandApplier",
    "LkbApplicationService",
]


@runtime_checkable
class CommandValidator(Protocol):
    """Lock-free validation callable (runs WITHOUT the Board lock).

    Receives the command and the lock-free snapshot.  Must return a
    :class:`ValidationRun`. The service binds the run to the revision vector + snapshot
    hash, so the validator does not need to set them.
    """

    def __call__(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun: ...


@runtime_checkable
class CommandApplier(Protocol):
    """In-lock mutation callable (runs WHILE the Board lock is held).

    Receives the command, a *cloned* :class:`BoardEnvelope`, and the
    accepted :class:`ValidationRun`.  Must apply the domain mutation,
    record the validation run on the candidate (``validation_runs``),
    and return ``(candidate, CommandResult(decision="committed", ...))``.

    Must not perform slow or external work while the Board File Lock is held.
    """

    def __call__(
        self,
        command: GraphCommand,
        envelope: BoardEnvelope,
        validation: ValidationRun,
    ) -> tuple[BoardEnvelope, CommandResult]: ...


@dataclass
class LkbApplicationService:
    """Orchestrates the consistency kernel's atomic commit pipeline."""

    repository: LkbRepository
    max_retries: int = 4

    # ── public API ───────────────────────────────────────────────────

    def execute(
        self,
        command: GraphCommand,
        *,
        validate: CommandValidator,
        apply: CommandApplier,
    ) -> CommandResult:
        """Execute *command* via two-phase optimistic commit.

        Returns the final :class:`CommandResult` (``committed`` or
        ``denied``).  ``committed`` is only returned after the Store's
        atomic replace succeeds.
        """
        if not validate_request_hash(command):
            # Do not use a caller-declared digest as an idempotency or audit
            # authority.  Recompute the trusted value and persist the denial
            # exactly like every other command rejection.
            trusted_command = replace(command, request_hash=compute_request_hash(command))
            snapshot = self.repository.load_snapshot(command.board_id)
            mismatch = ValidationRun(
                validation_run_id=f"V-{uuid.uuid4().hex[:12]}",
                proposal_id=trusted_command.command_id,
                subject_ref=trusted_command.primary_subject_ref,
                result="fail",
                issues=(
                    ValidationIssue(
                        code=LkbErrorCode.REQUEST_HASH_MISMATCH,
                        message="request_hash does not match the canonical command payload",
                        rule="request_hash",
                        subject_ref=trusted_command.primary_subject_ref,
                    ),
                ),
                engine="application-security-audit",
                requested_by=trusted_command.actor,
            )
            return self._persist_denial(trusted_command, mismatch, snapshot.hash)

        last_conflict: StaleRevisionError | None = None
        last_snapshot_hash = ""
        for attempt in range(self._retry_budget()):
            snapshot = self.repository.load_snapshot(command.board_id)
            revision_vector = snapshot.revision_vector
            snapshot_hash = snapshot.hash
            last_snapshot_hash = snapshot_hash

            revision_denial = self._revision_precondition(command, snapshot)
            if revision_denial is not None:
                revision_denial = replace(
                    revision_denial,
                    revision_vector=revision_vector,
                    snapshot_hash=snapshot_hash,
                )
                return self._persist_denial(command, revision_denial, snapshot_hash)

            validation = self._validate_lock_free(command, validate, snapshot, revision_vector, snapshot_hash)
            if not validation.accepted:
                return self._persist_denial(command, validation, snapshot_hash)

            try:
                return self.repository.execute_atomic(
                    command.board_id,
                    command.command_id,
                    command.request_hash,
                    self._cas_vector_for(command, snapshot, revision_vector),
                    self._make_apply(command, apply, validation),
                    expected_store_revision=command.expected_store_revision,
                    actor=command.actor,
                    reason=command.reason,
                    audit_context=self._audit_context(command, validation, snapshot_hash),
                )
            except StaleRevisionError as exc:
                last_conflict = exc
                continue

        # Exhausted retries - persist a denial audit with the conflict
        # detail (issue #9: CAS retry exhaustion must not return a bare
        # denial without an audit record).
        exhausted = ValidationRun(
            validation_run_id=f"V-{uuid.uuid4().hex[:12]}",
            proposal_id=command.command_id,
            subject_ref=command.primary_subject_ref,
            result="fail",
            issues=(
                ValidationIssue(
                    code=LkbErrorCode.REVISION_CONFLICT,
                    message=(f"exhausted {self._retry_budget()} retries ({last_conflict})"),
                    rule="revision_cas",
                    subject_ref=command.primary_subject_ref,
                ),
            ),
            engine="application-cas",
            requested_by=command.actor,
        )
        return self._persist_denial(command, exhausted, last_snapshot_hash)

    # ── internals ────────────────────────────────────────────────────

    def _retry_budget(self) -> int:
        return max(1, self.max_retries + 1)

    @staticmethod
    def _revision_precondition(
        command: GraphCommand,
        snapshot: GraphSnapshot,
    ) -> ValidationRun | None:
        """Fail closed when caller-supplied optimistic revisions are stale."""
        message: str | None = None
        if command.expected_store_revision is not None and command.expected_store_revision != snapshot.store_revision:
            message = (
                f"Store revision mismatch: expected {command.expected_store_revision}, got {snapshot.store_revision}"
            )
        elif command.expected_revision_vector is not None:
            for graph_id, expected in command.expected_revision_vector.revisions.items():
                actual = snapshot.revision_vector.get(graph_id)
                if actual != expected:
                    message = f"Graph {graph_id!r} revision mismatch: expected {expected}, got {actual}"
                    break

        if message is None:
            return None
        return ValidationRun(
            validation_run_id=f"V-{uuid.uuid4().hex[:12]}",
            proposal_id=command.command_id,
            subject_ref=command.primary_subject_ref,
            result="fail",
            issues=(
                ValidationIssue(
                    code=LkbErrorCode.STALE_REVISION,
                    message=message,
                    rule="revision_cas",
                    subject_ref=command.primary_subject_ref,
                ),
            ),
            engine="application-cas",
            requested_by=command.actor,
        )

    def _cas_vector_for(
        self,
        command: GraphCommand,
        snapshot: GraphSnapshot,
        revision_vector: RevisionVector,
    ) -> RevisionVector | None:
        """Filter the full revision vector down to graphs this command
        may touch (spec §5.1.1).

        Plan commands contend only with the selected Plan. A Board may hold
        multiple independent Plans, so unrelated Plan revisions must not
        cause a false conflict.
        """
        # A command that names a concrete Plan graph must only contend with
        # that Plan.  A workspace Board may contain many independent Plans.
        explicit_graph_id: str | None = None
        if command.primary_subject_ref is not None:
            explicit_graph_id = command.primary_subject_ref.graph
        else:
            raw_graph_id = command.payload.get("plan_id") or command.payload.get("plan_graph_id")
            if isinstance(raw_graph_id, str) and raw_graph_id:
                explicit_graph_id = raw_graph_id
        if explicit_graph_id is not None:
            graph = snapshot.graphs.get(explicit_graph_id)
            if graph is None:
                return None
            if graph.graph_kind != "plan":
                return revision_vector
            return RevisionVector(revisions={explicit_graph_id: revision_vector.get(explicit_graph_id)})

        # Commands without an explicit Plan contend with every Plan on the Board.
        relevant_graph_ids = {gid for gid, graph in snapshot.graphs.items() if graph.graph_kind == "plan"}
        if not relevant_graph_ids:
            return revision_vector
        filtered = {gid: rev for gid, rev in revision_vector.revisions.items() if gid in relevant_graph_ids}
        if not filtered:
            return None
        return RevisionVector(revisions=filtered)

    def _audit_context(
        self,
        command: GraphCommand,
        validation: ValidationRun,
        snapshot_hash: str,
    ) -> dict[str, Any]:
        """Build the audit context passed to ``execute_atomic``.

        The store layer does not know the command's subject or the
        snapshot hash the validator read; this helper extracts them from
        the command + validation run so the ``command_executed`` audit
        event can carry the full field set required by spec §6.10.
        """
        ctx: dict[str, Any] = {"input_snapshot_hash": snapshot_hash}
        subject = command.primary_subject_ref or validation.subject_ref
        if subject is not None:
            ctx["subject_ref"] = str(subject)
        affected = tuple(str(ref) for ref in command.subject_refs) if command.subject_refs else ()
        if affected:
            ctx["affected_refs"] = list(affected)
        return ctx

    def _validate_lock_free(
        self,
        command: GraphCommand,
        validate: CommandValidator,
        snapshot: GraphSnapshot,
        revision_vector: RevisionVector,
        snapshot_hash: str,
    ) -> ValidationRun:
        run = validate(command, snapshot)
        # Bind the run to the revision vector + snapshot hash it read,
        # regardless of what the validator set (spec §5.10).
        return replace(
            run,
            revision_vector=revision_vector,
            snapshot_hash=snapshot_hash or run.snapshot_hash,
        )

    def _persist_denial(
        self,
        command: GraphCommand,
        validation: ValidationRun,
        snapshot_hash: str,
    ) -> CommandResult:
        """Record a denial audit without any domain mutation.

        CAS-free (``expected_revision_vector=None``): a denial is an
        audit record, not a state change, so it must not be blocked by a
        concurrent unrelated write.  ``execute_atomic`` still advances
        ``store_revision`` and records ``processed_commands`` for
        idempotency.
        """

        def mutate(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
            envelope.validation_runs[validation.validation_run_id] = validation.to_dict()
            return envelope, CommandResult(
                decision="denied",
                command_id=command.command_id,
                validation_run_id=validation.validation_run_id,
                error_code=(coerce_error_code(validation.issues[0].code) if validation.issues else None)
                or LkbErrorCode.VALIDATION_DENIED,
                reason=self._denial_reason(validation),
            )

        return self.repository.execute_atomic(
            command.board_id,
            command.command_id,
            command.request_hash,
            None,
            mutate,
            actor=command.actor,
            reason=command.reason,
            audit_context=self._audit_context(command, validation, snapshot_hash),
        )

    def _make_apply(
        self,
        command: GraphCommand,
        apply: CommandApplier,
        validation: ValidationRun,
    ) -> Callable[[BoardEnvelope], tuple[BoardEnvelope, CommandResult]]:
        def mutate(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
            candidate, result = apply(command, envelope, validation)
            # Record the validation run on the committed candidate so the
            # audit trace (command -> validation -> commit) is durable.
            candidate.validation_runs[validation.validation_run_id] = validation.to_dict()
            return candidate, result

        return mutate

    @staticmethod
    def _denial_reason(validation: ValidationRun) -> str:
        if validation.issues:
            first = validation.issues[0]
            return f"{first.code}: {first.message}"
        return "validation_denied"
