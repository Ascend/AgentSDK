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
# pylint: disable=C1803
"""Unit tests for :class:`LkbApplicationService` (Phase 3, spec §7.6).

These tests use an in-memory fake repository so the two-phase optimistic
commit orchestration can be exercised without disk I/O.  Real Store
integration is covered by the repository/integration suites.
"""

from __future__ import annotations

from lkb.application import LkbApplicationService
from lkb.commands import CommandResult, GraphCommand
from lkb.graph_types import GraphSnapshot, RevisionVector
from lkb.json_store import BoardEnvelope, StaleRevisionError
from lkb.validation import ValidationRun


# ── fixtures / helpers ────────────────────────────────────────────────


def _snapshot(board_id: str = "b1", plan_rev: int = 0) -> GraphSnapshot:
    return GraphSnapshot(
        board_id=board_id,
        store_revision=plan_rev,
        revision_vector=RevisionVector(revisions={"plan": plan_rev}),
    )


def _command(
    kind: str = "create_task",
    board_id: str = "b1",
    command_id: str = "cmd-1",
    actor: str = "agent-a",
) -> GraphCommand:
    return GraphCommand(
        command_id=command_id,
        board_id=board_id,
        actor=actor,
        kind=kind,
    )


def _validation(accepted: bool = True, run_id: str = "V-1") -> ValidationRun:
    return ValidationRun(
        validation_run_id=run_id,
        proposal_id="P-1",
        result="pass" if accepted else "denied",
    )


class _FakeRepo:
    """In-memory LkbRepository stub with controllable conflicts."""

    def __init__(self, snapshot: GraphSnapshot, *, conflicts: int = 0) -> None:
        self._snapshot = snapshot
        self._conflicts = conflicts
        self.execute_calls: list[dict] = []
        self.last_candidate: BoardEnvelope | None = None
        self.last_result: CommandResult | None = None

    def load_snapshot(self, board_id: str) -> GraphSnapshot:
        return self._snapshot

    def execute_atomic(
        self,
        board_id: str,
        command_id: str,
        request_hash: str,
        expected_revision_vector: RevisionVector | None,
        mutate,
        *,
        expected_store_revision: int | None = None,
        actor: str = "x",
        reason: str | None = None,
        audit_context: dict | None = None,
    ) -> CommandResult:
        self.execute_calls.append(
            {
                "command_id": command_id,
                "request_hash": request_hash,
                "expected_revision_vector": expected_revision_vector,
                "expected_store_revision": expected_store_revision,
                "actor": actor,
                "reason": reason,
                "audit_context": audit_context,
            }
        )
        # Denials are persisted CAS-free (expected_revision_vector=None);
        # they must not be blocked by a simulated CAS conflict.  Only
        # guarded writes (those carrying a revision vector) conflict.
        if expected_revision_vector is not None and self._conflicts > 0:
            self._conflicts -= 1
            raise StaleRevisionError(
                board_id,
                expected_revision_vector,
                self._snapshot.revision_vector,
                reason="simulated conflict",
            )
        envelope = BoardEnvelope()
        candidate, result = mutate(envelope.clone())
        self.last_candidate = candidate
        self.last_result = result
        return result


# ── accepted path ────────────────────────────────────────────────────


def test_accepted_command_commits_via_execute_atomic() -> None:
    repo = _FakeRepo(_snapshot())
    svc = LkbApplicationService(repository=repo)

    applied: list[bool] = []

    def apply(command, envelope, validation):
        envelope.nodes["T-1"] = {"ref": "plan:task:T-1", "title": "Test"}
        applied.append(True)
        return envelope, CommandResult(
            decision="committed",
            command_id=command.command_id,
            validation_run_id="V-1",
        )

    result = svc.execute(_command(), validate=lambda c, s: _validation(True), apply=apply)

    assert result.decision == "committed"
    assert result.command_id == "cmd-1"
    assert len(repo.execute_calls) == 1
    assert applied == [True]
    # The ValidationRun is recorded on the committed candidate (audit trace).
    assert repo.last_candidate is not None
    assert "V-1" in repo.last_candidate.validation_runs


def test_forged_request_hash_is_denied_with_trusted_audit() -> None:
    repo = _FakeRepo(_snapshot())
    service = LkbApplicationService(repository=repo)
    command = _command()
    object.__setattr__(command, "actor", "attacker")

    result = service.execute(
        command,
        validate=lambda c, s: _validation(True),
        apply=lambda c, e, v: (
            e,
            CommandResult(decision="committed", command_id=c.command_id),
        ),
    )

    assert result.decision == "denied"
    assert "request_hash_mismatch" in (result.reason or "")
    assert len(repo.execute_calls) == 1
    assert repo.execute_calls[0]["request_hash"] != command.request_hash
    assert repo.execute_calls[0]["expected_revision_vector"] is None
    assert repo.last_candidate is not None
    assert repo.last_candidate.validation_runs


def test_validation_is_bound_to_revision_vector_and_snapshot_hash() -> None:
    snap = _snapshot(plan_rev=7)
    repo = _FakeRepo(snap)
    svc = LkbApplicationService(repository=repo)

    seen: dict = {}

    def validate(command, snapshot):
        seen["snapshot_hash"] = snapshot.hash
        return ValidationRun(
            validation_run_id="V-bind",
            proposal_id="P-1",
            result="pass",
            # Validator intentionally leaves these unset; service must bind.
            snapshot_hash="",
            revision_vector=None,  # type: ignore[arg-type]
        )

    def apply(command, envelope, validation):
        # The validation passed to apply must carry the read revision vector.
        seen["bound_rv"] = validation.revision_vector
        seen["bound_hash"] = validation.snapshot_hash
        return envelope, CommandResult(decision="committed", command_id=command.command_id, validation_run_id="V-bind")

    svc.execute(_command(), validate=validate, apply=apply)

    assert seen["bound_rv"] is not None
    assert seen["bound_rv"].get("plan") == 7
    assert seen["bound_hash"] == seen["snapshot_hash"]
    # And the recorded validation run carries the binding.
    recorded = repo.last_candidate.validation_runs["V-bind"]
    assert recorded["revisionVector"] == {"plan": 7}


# ── denial path ──────────────────────────────────────────────────────


def test_denial_persists_audit_without_domain_mutation() -> None:
    repo = _FakeRepo(_snapshot())
    svc = LkbApplicationService(repository=repo)

    applied: list[bool] = []

    def apply(command, envelope, validation):
        applied.append(True)
        return envelope, CommandResult(decision="committed", command_id=command.command_id)

    result = svc.execute(_command(), validate=lambda c, s: _validation(False), apply=apply)

    assert result.decision == "denied"
    assert applied == []  # apply never runs on a denial
    assert len(repo.execute_calls) == 1
    # Denials are CAS-free (audit-only, must not block on unrelated writes).
    assert repo.execute_calls[0]["expected_revision_vector"] is None
    assert "V-1" in repo.last_candidate.validation_runs


def test_stale_caller_revision_vector_is_denied_before_validation() -> None:
    repo = _FakeRepo(_snapshot(plan_rev=4))
    svc = LkbApplicationService(repository=repo)
    command = GraphCommand(
        command_id="cmd-stale-vector",
        board_id="b1",
        actor="agent-a",
        kind="create_task",
        expected_revision_vector=RevisionVector(revisions={"plan": 3}),
    )
    validated: list[bool] = []

    result = svc.execute(
        command,
        validate=lambda c, s: validated.append(True) or _validation(True),
        apply=lambda c, e, v: (e, CommandResult(decision="committed", command_id=c.command_id)),
    )

    assert result.decision == "denied"
    assert "stale_revision" in (result.reason or "")
    assert validated == []


def test_stale_caller_store_revision_is_denied_before_validation() -> None:
    repo = _FakeRepo(_snapshot(plan_rev=5))
    svc = LkbApplicationService(repository=repo)
    command = GraphCommand(
        command_id="cmd-stale-store",
        board_id="b1",
        actor="agent-a",
        kind="create_task",
        expected_store_revision=4,
    )

    result = svc.execute(
        command,
        validate=lambda c, s: _validation(True),
        apply=lambda c, e, v: (e, CommandResult(decision="committed", command_id=c.command_id)),
    )

    assert result.decision == "denied"
    assert "Store revision mismatch" in (result.reason or "")


# ── revision conflict / retry ────────────────────────────────────────


def test_revision_conflict_retries_then_succeeds() -> None:
    repo = _FakeRepo(_snapshot(plan_rev=3), conflicts=2)
    svc = LkbApplicationService(repository=repo, max_retries=4)

    validate_calls: list[int] = []

    def validate(command, snapshot):
        validate_calls.append(snapshot.revision_vector.get("plan"))
        return _validation(True)

    def apply(command, envelope, validation):
        return envelope, CommandResult(decision="committed", command_id=command.command_id, validation_run_id="V-1")

    result = svc.execute(_command(), validate=validate, apply=apply)

    assert result.decision == "committed"
    assert len(validate_calls) == 3  # initial attempt + 2 retries
    assert len(repo.execute_calls) == 3
    # The CAS guard passed each time is the read revision vector.
    for call in repo.execute_calls:
        assert call["expected_revision_vector"] is not None
        assert call["expected_revision_vector"].get("plan") == 3


def test_revision_conflict_exhausted_returns_denial() -> None:
    repo = _FakeRepo(_snapshot(), conflicts=99)  # always conflicts
    svc = LkbApplicationService(repository=repo, max_retries=2)

    result = svc.execute(
        _command(),
        validate=lambda c, s: _validation(True),
        apply=lambda c, e, v: (e, CommandResult(decision="committed", command_id=c.command_id)),
    )

    assert result.decision == "denied"
    assert result.reason is not None
    assert "revision_conflict" in result.reason
    # max_retries=2 => budget = 3 conflict attempts, then a 4th CAS-free
    # call persists the denial audit (issue #9).
    assert len(repo.execute_calls) == 4
    # The denial persist is CAS-free.
    assert repo.execute_calls[-1]["expected_revision_vector"] is None
    assert repo.execute_calls[-1]["audit_context"]["input_snapshot_hash"] == repo._snapshot.hash


# ── idempotency (Store returns cached result without re-applying) ────


class _IdempotentRepo(_FakeRepo):
    """Simulates a Store that already processed the command: returns the
    cached committed result WITHOUT invoking ``mutate``.
    """

    def __init__(self, snapshot: GraphSnapshot, cached: CommandResult) -> None:
        super().__init__(snapshot)
        self._cached = cached

    def execute_atomic(self, *args, **kwargs) -> CommandResult:  # type: ignore[override]
        self.execute_calls.append({"command_id": args[1]})
        return self._cached  # mutate never called


def test_idempotent_retry_does_not_re_apply() -> None:
    cached = CommandResult(decision="committed", command_id="cmd-1", validation_run_id="V-old")
    repo = _IdempotentRepo(_snapshot(), cached)
    svc = LkbApplicationService(repository=repo)

    applied: list[bool] = []

    def apply(command, envelope, validation):
        applied.append(True)
        return envelope, CommandResult(decision="committed", command_id=command.command_id)

    result = svc.execute(_command(), validate=lambda c, s: _validation(True), apply=apply)

    assert result is cached  # cached committed result returned as-is
    assert applied == []  # mutate/apply never re-run
