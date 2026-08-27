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

"""Regression coverage for approved LKB review fixes (#666, #669, #672)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from lkb import lifecycle_gc
from lkb import plan_scope
from lkb.application import LkbApplicationService
from lkb.commands import CommandResult, GraphCommand
from lkb.file_lock import BoardStoreBusyError
from lkb.json_store import BoardEnvelope
from lkb.lifecycle_core import LifecycleError, archive_board
from lkb.lifecycle_gc import GcCandidate, gc_apply
from lkb.migrations import BoardSchemaTooNewError, MigrationError, migrate
from lkb.error_codes import ERROR_CODES, LkbErrorCode
from lkb.plan_graph_core import UpdateTaskFieldsHandler
from lkb.plan_graph import plan_command_dispatcher
from lkb.refs import NodeRef
from lkb.repository import JsonFileLkbRepository
from lkb.validation import ValidationRun


def test_error_code_registry_keeps_only_the_emitted_transition_code() -> None:
    assert LkbErrorCode.INVALID_TRANSITION.value in ERROR_CODES
    assert "INVALID_STATE_TRANSITION" not in LkbErrorCode.__members__


def test_migration_preserves_existing_lifecycle_values() -> None:
    result, _ = migrate(
        {
            "board": {"board_id": "test-board", "project_uri": "session:abc"},
            "lifecycle": {"state": "closed", "scope": "custom"},
        },
        target_schema=1,
    )
    assert result["lifecycle"]["state"] == "closed"
    assert result["lifecycle"]["scope"] == "custom"
    assert result["lifecycle"]["retention_policy"] == "default"


def test_future_schema_precedes_malformed_board_shape_during_migration() -> None:
    with pytest.raises(BoardSchemaTooNewError):
        migrate({"schemaVersion": 99, "board": "not-an-object"}, target_schema=1)


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_non_integer_schema_version_is_migration_error(version: object) -> None:
    with pytest.raises(MigrationError, match="invalid schemaVersion"):
        migrate({"schemaVersion": version, "board": {"board_id": "bad"}})


def test_field_update_warns_when_replacing_a_non_object_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ref = NodeRef("plan", "task", "T-1")
    envelope = BoardEnvelope(
        board={"board_id": "approved-payload", "policy": {}},
        nodes={
            "node-1": {
                "ref": ref.to_str(),
                "title": "Before",
                "state": "pending",
                "revision": 1,
                "payload": ["invalid"],
            }
        },
    )
    command = GraphCommand(
        command_id="repair-payload",
        board_id="approved-payload",
        actor="reviewer",
        kind="update_task_fields",
        payload={"task_id": "T-1", "subject": "After"},
    )
    validation = ValidationRun(validation_run_id="V-1", proposal_id=command.command_id)

    with caplog.at_level("WARNING"):
        candidate, result = UpdateTaskFieldsHandler().apply(command, envelope, validation)

    assert result.committed
    assert candidate.nodes["node-1"]["payload"]["subject"] == "After"
    assert "Replacing non-object payload" in caplog.text


def test_repeated_forged_request_hash_is_deduplicated_under_trusted_hash(tmp_home) -> None:
    repository = JsonFileLkbRepository(home=tmp_home)
    board_id = repository.resolve_board(explicit_id="forged-request-audit").board_id
    service = LkbApplicationService(repository=repository)
    dispatcher = plan_command_dispatcher()
    command = GraphCommand(
        command_id="forged-request",
        board_id=board_id,
        actor="reviewer",
        kind="create_task",
        payload={"task_id": "T-forged", "subject": "Never committed"},
    )

    first = service.execute(
        replace(command, request_hash="attacker-declared-one"),
        validate=dispatcher.validate,
        apply=dispatcher.apply,
    )
    revision_after_first = repository.load_envelope(board_id).store_revision
    second = service.execute(
        replace(command, request_hash="attacker-declared-two"),
        validate=dispatcher.validate,
        apply=dispatcher.apply,
    )
    envelope = repository.load_envelope(board_id)

    assert first.decision == second.decision == "denied"
    assert first.error_code == second.error_code
    assert envelope.store_revision == revision_after_first
    assert list(envelope.processed_commands) == [command.command_id]
    assert len([event for event in envelope.events if event.get("command_id") == command.command_id]) == 2


def test_archived_board_with_non_string_archive_path_has_stable_error() -> None:
    envelope = BoardEnvelope(
        board={"board_id": "approved-review"},
        lifecycle={"state": "archived", "archive_info": {"archive_path": None}},
    )
    store = SimpleNamespace(load=lambda: envelope)

    with pytest.raises(LifecycleError, match="invalid archive provenance path"):
        archive_board(
            store,
            "approved-review",
            actor="reviewer",
            command_id="approved-path",
            request_hash="approved-path-hash",
        )


def test_gc_skips_only_busy_candidates_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    busy = GcCandidate(tmp_path / "busy", "busy", 1, "review", action="delete")
    available = GcCandidate(tmp_path / "available", "available", 1, "review", action="delete")
    executed: list[str] = []

    def execute(candidate: GcCandidate, **_kwargs: object) -> None:
        executed.append(candidate.kind)
        if candidate is busy:
            raise BoardStoreBusyError("lock held")

    monkeypatch.setattr(lifecycle_gc, "_execute_gc_candidate", execute)

    gc_apply([busy, available], now=1.0)

    assert executed == ["busy", "available"]


def test_gc_propagates_non_busy_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    candidate = GcCandidate(tmp_path / "broken", "broken", 1, "review", action="delete")

    def fail(_candidate: GcCandidate, **_kwargs: object) -> None:
        raise PermissionError("readonly")

    monkeypatch.setattr(lifecycle_gc, "_execute_gc_candidate", fail)

    with pytest.raises(PermissionError, match="readonly"):
        gc_apply([candidate], now=1.0)


def test_bootstrap_failure_uses_stable_context_local_session_id(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import src.bootstrap.state as bootstrap_state

    def broken_session_id() -> str:
        raise RuntimeError("bootstrap unavailable")

    monkeypatch.setattr(bootstrap_state, "get_session_id", broken_session_id)
    first_context = SimpleNamespace()
    second_context = SimpleNamespace()

    first = plan_scope.current_session_id(first_context)

    assert first == plan_scope.current_session_id(first_context)
    assert first != plan_scope.current_session_id(second_context)
    assert first.startswith("context-")
    assert "context-local fallback" in caplog.text


def test_set_plan_state_repairs_corrupt_session_bindings(tmp_home) -> None:
    repository = JsonFileLkbRepository(home=tmp_home)
    board_id = repository.resolve_board(explicit_id="approved-bindings").board_id
    plan = plan_scope.create_plan(repository, board_id, "session-approved", title="Approved")

    def corrupt_bindings(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
        envelope.board["session_plan_bindings"] = []
        return envelope, CommandResult(decision="committed", command_id="corrupt-bindings")

    repository.execute_atomic(
        board_id,
        "corrupt-bindings",
        "corrupt-bindings-hash",
        None,
        corrupt_bindings,
        actor="reviewer",
    )

    plan_scope.set_plan_state(
        repository,
        board_id,
        "session-approved",
        plan.plan_id,
        plan_scope.PLAN_SUSPENDED,
    )

    assert plan_scope.bound_plan_id(repository, board_id, "session-approved") == plan.plan_id
