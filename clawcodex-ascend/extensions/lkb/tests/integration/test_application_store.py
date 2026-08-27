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

"""Integration tests for LkbApplicationService against the real JSON Store.

Exercises the two-phase optimistic commit (spec §7.6) through the real
:class:`JsonFileLkbRepository` / :class:`JsonBoardStore` stack:

* accepted command atomically commits and bumps revisions
* a concurrent write between lock-free read and commit triggers a real
  ``StaleRevisionError`` CAS failure and the service retries
* command_id + request_hash idempotency is honoured by the Store
* a denial is persisted as audit without a domain mutation
"""

from __future__ import annotations

from pathlib import Path


from lkb.application import LkbApplicationService
from lkb.commands import CommandResult, GraphCommand
from lkb.graph_types import NodeRef
from lkb.json_store import BoardEnvelope
from lkb.repository import JsonFileLkbRepository
from lkb.validation import ValidationRun


# ── helpers ───────────────────────────────────────────────────────────


def _make_repo(tmp_home: Path) -> JsonFileLkbRepository:
    return JsonFileLkbRepository(home=tmp_home)


def _ensure_plan_graph(env: BoardEnvelope, board_id: str) -> None:
    if "plan" not in env.graphs:
        env.graphs["plan"] = {
            "graph_id": "plan",
            "board_id": board_id,
            "graph_kind": "plan",
            "revision": 0,
            "created_at": "2026-01-01T00:00:00.000Z",
            "updated_at": "2026-01-01T00:00:00.000Z",
        }


def _add_task_node(env: BoardEnvelope, node_id: str, title: str) -> None:
    ref = NodeRef("plan", "task", node_id)
    env.nodes[node_id] = {
        "ref": ref.to_str(),
        "title": title,
        "state": "pending",
        "owner": None,
        "revision": 1,
        "payload": {},
        "created_at": "2026-01-01T00:00:00.000Z",
        "updated_at": "2026-01-01T00:00:00.000Z",
    }


def _add_node_direct(
    repo: JsonFileLkbRepository, board_id: str, command_id: str, node_id: str, title: str
) -> CommandResult:
    """Bypass the ApplicationService to perform a direct Store write."""

    def mutate(env: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
        _ensure_plan_graph(env, board_id)
        _add_task_node(env, node_id, title)
        return env, CommandResult(decision="committed", command_id=command_id)

    return repo.execute_atomic(board_id, command_id, f"sha256:{command_id}", None, mutate, actor="concurrent")


# ── tests ─────────────────────────────────────────────────────────────


def test_accepted_command_commits_to_real_store(tmp_home: Path) -> None:
    repo = _make_repo(tmp_home)
    board_id = repo.resolve_board(explicit_id="app-commit").board_id
    svc = LkbApplicationService(repository=repo)

    applied: list[int] = []

    def validate(command, snapshot):
        return ValidationRun(validation_run_id="V-1", proposal_id="P-1", result="pass")

    def apply(command, envelope, validation):
        _ensure_plan_graph(envelope, board_id)
        _add_task_node(envelope, "T-1", "First task")
        applied.append(1)
        return envelope, CommandResult(decision="committed", command_id=command.command_id, validation_run_id="V-1")

    command = GraphCommand(command_id="cmd-1", board_id=board_id, actor="agent-a", kind="create_task")
    result = svc.execute(command, validate=validate, apply=apply)

    assert result.decision == "committed"
    assert applied == [1]

    # The node and validation run are durable in the Store.
    snapshot = repo.load_snapshot(board_id)
    assert NodeRef("plan", "task", "T-1") in snapshot.nodes
    env = repo._get_store(board_id).load()
    assert "V-1" in env.validation_runs
    # Plan graph revision advanced.
    assert snapshot.revision_vector.get("plan") >= 1


def test_concurrent_write_triggers_cas_retry(tmp_home: Path) -> None:
    repo = _make_repo(tmp_home)
    board_id = repo.resolve_board(explicit_id="app-conflict").board_id
    # Pre-create the plan graph so the first load_snapshot returns a
    # non-empty revision vector; otherwise the CAS guard has nothing to
    # compare against and a concurrent write would not be detected.
    _add_node_direct(repo, board_id, "seed-1", "T-0", "Seed")
    svc = LkbApplicationService(repository=repo, max_retries=4)

    validate_calls: list[int] = []

    def validate(command, snapshot):
        validate_calls.append(1)
        if len(validate_calls) == 1:
            # First (lock-free) validation: a concurrent writer commits
            # before our execute_atomic acquires the Board lock, bumping
            # the plan graph revision the service read as R.
            _add_node_direct(repo, board_id, "concurrent-1", "T-c", "Concurrent")
        return ValidationRun(validation_run_id="V-2", proposal_id="P-2", result="pass")

    def apply(command, envelope, validation):
        _ensure_plan_graph(envelope, board_id)
        _add_task_node(envelope, "T-2", "Second task")
        return envelope, CommandResult(decision="committed", command_id=command.command_id, validation_run_id="V-2")

    command = GraphCommand(command_id="cmd-2", board_id=board_id, actor="agent-a", kind="create_task")
    result = svc.execute(command, validate=validate, apply=apply)

    assert result.decision == "committed"
    # Retried: validate ran twice (first hit the conflict).
    assert len(validate_calls) == 2
    # The concurrent write and the retried command's node both survived.
    snapshot = repo.load_snapshot(board_id)
    assert NodeRef("plan", "task", "T-c") in snapshot.nodes
    assert NodeRef("plan", "task", "T-2") in snapshot.nodes


def test_command_id_idempotency_against_real_store(tmp_home: Path) -> None:
    repo = _make_repo(tmp_home)
    board_id = repo.resolve_board(explicit_id="app-idem").board_id
    svc = LkbApplicationService(repository=repo)

    applied: list[int] = []

    def validate(command, snapshot):
        return ValidationRun(validation_run_id="V-3", proposal_id="P-3", result="pass")

    def apply(command, envelope, validation):
        _ensure_plan_graph(envelope, board_id)
        _add_task_node(envelope, "T-3", "Idempotent task")
        applied.append(1)
        return envelope, CommandResult(decision="committed", command_id=command.command_id, validation_run_id="V-3")

    command = GraphCommand(command_id="cmd-3", board_id=board_id, actor="agent-a", kind="create_task")
    first = svc.execute(command, validate=validate, apply=apply)
    assert first.decision == "committed"
    assert applied == [1]

    # Replay the exact same command: Store returns the cached result and
    # the apply callback must NOT run again.
    second = svc.execute(command, validate=validate, apply=apply)
    assert second.decision == "committed"
    assert applied == [1]  # apply still only ran once

    # Only one node exists (no duplicate create from the replay).
    snapshot = repo.load_snapshot(board_id)
    assert sum(1 for n in snapshot.nodes.values() if n.title == "Idempotent task") == 1


def test_denial_persists_audit_without_domain_mutation(tmp_home: Path) -> None:
    repo = _make_repo(tmp_home)
    board_id = repo.resolve_board(explicit_id="app-deny").board_id
    svc = LkbApplicationService(repository=repo)

    applied: list[int] = []

    def validate(command, snapshot):
        return ValidationRun(validation_run_id="V-4", proposal_id="P-4", result="denied")

    def apply(command, envelope, validation):
        applied.append(1)
        return envelope, CommandResult(decision="committed", command_id=command.command_id)

    command = GraphCommand(command_id="cmd-4", board_id=board_id, actor="agent-a", kind="create_task")
    result = svc.execute(command, validate=validate, apply=apply)

    assert result.decision == "denied"
    assert applied == []  # no domain mutation

    # The denial audit + validation run are durable.
    env = repo._get_store(board_id).load()
    assert "V-4" in env.validation_runs
    entry = env.processed_commands.get("cmd-4")
    assert entry is not None
    assert entry["decision"] == "denied"
    # No task nodes were created.
    snapshot = repo.load_snapshot(board_id)
    assert all(n.ref.kind != "task" for n in snapshot.nodes.values())


def test_command_executed_event_carries_full_audit_fields(tmp_home: Path) -> None:
    """B4: the ``command_executed`` audit event must include every field
    required by spec §6.10 (event_id, board_id, store_revision, command_id,
    actor, timestamp, subject_ref, decision, reason, input_snapshot_hash,
    validation_run_id, affected_refs).
    """
    repo = _make_repo(tmp_home)
    board_id = repo.resolve_board(explicit_id="app-audit").board_id
    svc = LkbApplicationService(repository=repo)

    def validate(command, snapshot):
        return ValidationRun(validation_run_id="V-A", proposal_id="P-A", result="pass")

    def apply(command, envelope, validation):
        _ensure_plan_graph(envelope, board_id)
        _add_task_node(envelope, "T-A", "Audited task")
        return envelope, CommandResult(
            decision="committed",
            command_id=command.command_id,
            validation_run_id="V-A",
        )

    command = GraphCommand(
        command_id="cmd-audit",
        board_id=board_id,
        actor="agent-a",
        kind="create_task",
        primary_subject_ref=NodeRef("plan", "task", "T-A"),
    )
    input_snapshot_hash = repo.load_snapshot(board_id).hash
    result = svc.execute(command, validate=validate, apply=apply)
    assert result.decision == "committed"

    env = repo._get_store(board_id).load()
    events = [e for e in env.events if e.get("type") == "command_executed"]
    assert events, "expected at least one command_executed event"
    event = events[-1]
    # §6.10 MUST fields:
    assert event["event_id"], "event_id missing"
    assert event["board_id"] == board_id, "board_id missing/wrong"
    assert event["command_id"] == "cmd-audit", "command_id missing/wrong"
    assert event["actor"] == "agent-a", "actor missing/wrong"
    assert event["timestamp"], "timestamp missing"
    assert event["decision"] == "committed", "decision missing/wrong"
    assert event["store_revision"] >= 1, "store_revision missing/wrong"
    assert event["input_snapshot_hash"] == input_snapshot_hash
    assert event["validation_run_id"] == "V-A", "validation_run_id missing/wrong"
    # subject_ref comes from the command's primary_subject_ref via audit_context.
    assert event["subject_ref"] == "plan:task:T-A", "subject_ref missing/wrong"
    received = [e for e in env.events if e.get("type") == "command_received"][-1]
    for field in (
        "event_id",
        "board_id",
        "store_revision",
        "command_id",
        "actor",
        "timestamp",
        "subject_ref",
        "decision",
        "rule",
        "input_snapshot_hash",
        "validation_run_id",
        "affected_refs",
    ):
        assert field in received
    assert received["input_snapshot_hash"] == input_snapshot_hash
