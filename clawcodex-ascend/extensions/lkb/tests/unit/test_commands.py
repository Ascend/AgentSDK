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

# pylint: disable=C1803

"""Unit tests for GraphCommand hashing and atomic PatchTask intent parsing."""

from __future__ import annotations

import pytest
from lkb.commands import (
    CommandResult,
    GraphCommand,
    PatchTask,
    compute_request_hash,
    validate_request_hash,
)
from lkb.error_codes import ERROR_CODES, LkbErrorCode
from lkb.graph_types import RevisionVector
from lkb.refs import NodeRef


def _make_task_ref(task_id: str = "T-001") -> NodeRef:
    return NodeRef(graph="plan", kind="task", id=task_id)


def test_error_code_registry_is_unique_and_complete() -> None:
    assert len(ERROR_CODES) == len(LkbErrorCode.__members__)
    assert ERROR_CODES == {code.value for code in LkbErrorCode}


# ── GraphCommand & request_hash (LKB-STORE-004 / 005) ────────────────


class TestRequestHash:
    def test_roles_are_normalized_and_part_of_hash(self) -> None:
        base = {
            "command_id": "cmd-role",
            "board_id": "board-X",
            "actor": "agent-A",
            "kind": "release_task",
        }
        unprivileged = GraphCommand(**base)
        privileged = GraphCommand(**base, roles=(" admin ", "admin"))

        assert privileged.roles == ("admin",)
        assert privileged.request_hash != unprivileged.request_hash
        with pytest.raises(TypeError, match="roles must be"):
            GraphCommand(**base, roles="admin")  # type: ignore[arg-type]

    def test_same_command_same_hash(self) -> None:
        """Identical canonical fields produce identical hashes."""
        ref = _make_task_ref()
        cmd_a = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="agent-A",
            kind="update_task_fields",
            primary_subject_ref=ref,
            payload={"subject": "Hello"},
        )
        cmd_b = GraphCommand(
            command_id="cmd-2",
            board_id="board-X",
            actor="agent-A",
            kind="update_task_fields",
            primary_subject_ref=ref,
            payload={"subject": "Hello"},
        )
        # Different command_id, same canonical fields → same hash.
        assert cmd_a.request_hash == cmd_b.request_hash
        assert compute_request_hash(cmd_a) == compute_request_hash(cmd_b)

    def test_different_payload_different_hash(self) -> None:
        """LKB-STORE-005: different payload → different request_hash."""
        ref = _make_task_ref()
        cmd_a = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="agent-A",
            kind="update_task_fields",
            primary_subject_ref=ref,
            payload={"subject": "Hello"},
        )
        cmd_b = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="agent-A",
            kind="update_task_fields",
            primary_subject_ref=ref,
            payload={"subject": "World"},
        )
        assert cmd_a.request_hash != cmd_b.request_hash

    def test_hash_excludes_command_id(self) -> None:
        """request_hash must NOT depend on command_id (spec §5.10)."""
        ref = _make_task_ref()
        cmd_a = GraphCommand(
            command_id="cmd-AAA",
            board_id="board-X",
            actor="agent-A",
            kind="claim_task",
            primary_subject_ref=ref,
        )
        cmd_b = GraphCommand(
            command_id="cmd-BBB",
            board_id="board-X",
            actor="agent-A",
            kind="claim_task",
            primary_subject_ref=ref,
        )
        assert compute_request_hash(cmd_a) == compute_request_hash(cmd_b)

    def test_hash_excludes_created_at(self) -> None:
        """request_hash must NOT depend on created_at timing."""
        ref = _make_task_ref()
        cmd_a = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="agent-A",
            kind="claim_task",
            primary_subject_ref=ref,
            created_at="2025-01-01T00:00:00+00:00",
        )
        cmd_b = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="agent-A",
            kind="claim_task",
            primary_subject_ref=ref,
            created_at="2099-12-31T23:59:59+00:00",
        )
        assert compute_request_hash(cmd_a) == compute_request_hash(cmd_b)

    def test_hash_includes_board_id(self) -> None:
        """Different board → different hash."""
        ref = _make_task_ref()
        cmd_a = GraphCommand(
            command_id="cmd-1",
            board_id="board-A",
            actor="agent-A",
            kind="claim_task",
            primary_subject_ref=ref,
        )
        cmd_b = GraphCommand(
            command_id="cmd-1",
            board_id="board-B",
            actor="agent-A",
            kind="claim_task",
            primary_subject_ref=ref,
        )
        assert compute_request_hash(cmd_a) != compute_request_hash(cmd_b)

    def test_hash_includes_actor(self) -> None:
        """Different actor → different hash (trusted actor is part of identity)."""
        ref = _make_task_ref()
        cmd_a = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="alice",
            kind="claim_task",
            primary_subject_ref=ref,
        )
        cmd_b = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="bob",
            kind="claim_task",
            primary_subject_ref=ref,
        )
        assert compute_request_hash(cmd_a) != compute_request_hash(cmd_b)

    def test_hash_includes_expected_revisions(self) -> None:
        """Different expected revision vector → different hash."""
        ref = _make_task_ref()
        rv_a = RevisionVector(revisions={"plan": 1})
        rv_b = RevisionVector(revisions={"plan": 2})
        cmd_a = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="agent-A",
            kind="claim_task",
            primary_subject_ref=ref,
            expected_revision_vector=rv_a,
        )
        cmd_b = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="agent-A",
            kind="claim_task",
            primary_subject_ref=ref,
            expected_revision_vector=rv_b,
        )
        assert compute_request_hash(cmd_a) != compute_request_hash(cmd_b)

    def test_hash_is_sha256_hex(self) -> None:
        """request_hash is a 64-char lowercase hex string (sha256)."""
        cmd = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="agent-A",
            kind="claim_task",
            primary_subject_ref=_make_task_ref(),
        )
        h = cmd.request_hash
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_request_hash_matches_instance_field(self) -> None:
        """Auto-populated request_hash field matches compute_request_hash()."""
        cmd = GraphCommand(
            command_id="cmd-1",
            board_id="board-X",
            actor="agent-A",
            kind="patch_task",
            primary_subject_ref=_make_task_ref(),
            payload={"field_updates": {"subject": "X"}},
        )
        assert cmd.request_hash == compute_request_hash(cmd)

    def test_subject_refs_have_stable_set_semantics(self) -> None:
        a = _make_task_ref("A")
        b = _make_task_ref("B")
        left = GraphCommand("c1", "b", "actor", "patch_task", subject_refs=(b, a, a))
        right = GraphCommand("c2", "b", "actor", "patch_task", subject_refs=(a, b))
        assert left.subject_refs == (a, b)
        assert left.request_hash == right.request_hash

    def test_external_payload_mutation_does_not_change_command(self) -> None:
        payload = {"nested": {"values": [1, 2]}}
        command = GraphCommand("c1", "b", "actor", "patch_task", payload=payload)
        payload["nested"]["values"].append(3)
        assert command.payload["nested"]["values"] == (1, 2)
        assert validate_request_hash(command)
        with pytest.raises(TypeError, match="immutable"):
            command.payload["new"] = "value"

    @pytest.mark.parametrize(
        "payload",
        [
            {"value": object()},
            {"value": float("nan")},
            {1: "not-a-string-key"},
        ],
    )
    def test_non_json_payload_is_rejected(self, payload: dict[object, object]) -> None:
        with pytest.raises((TypeError, ValueError)):
            GraphCommand("c1", "b", "actor", "patch_task", payload=payload)  # type: ignore[arg-type]

    def test_execution_boundary_rejects_post_init_tampering(self) -> None:
        command = GraphCommand("c1", "b", "actor", "patch_task")
        object.__setattr__(command, "actor", "attacker")
        assert not validate_request_hash(command)


# ── GraphCommand basic shape ──────────────────────────────────────────


class TestGraphCommand:
    def test_post_init_sets_created_at(self) -> None:
        cmd = GraphCommand(
            command_id="cmd-1",
            board_id="b",
            actor="a",
            kind="create_task",
        )
        assert cmd.created_at != ""

    def test_post_init_preserves_explicit_created_at(self) -> None:
        cmd = GraphCommand(
            command_id="cmd-1",
            board_id="b",
            actor="a",
            kind="create_task",
            created_at="2025-01-01T00:00:00+00:00",
        )
        assert cmd.created_at == "2025-01-01T00:00:00+00:00"

    def test_subject_refs_default_empty_tuple(self) -> None:
        cmd = GraphCommand(
            command_id="cmd-1",
            board_id="b",
            actor="a",
            kind="create_task",
        )
        assert cmd.subject_refs == ()

    def test_reason_none_by_default(self) -> None:
        cmd = GraphCommand(
            command_id="cmd-1",
            board_id="b",
            actor="a",
            kind="create_task",
        )
        assert cmd.reason is None


# ── CommandResult ─────────────────────────────────────────────────────


class TestCommandResult:
    def test_committed_decision(self) -> None:
        r = CommandResult(decision="committed", command_id="cmd-1")
        assert r.committed is True

    def test_denied_decision(self) -> None:
        r = CommandResult(decision="denied", command_id="cmd-1")
        assert r.committed is False


# ── PatchTask.decompose (T2-GAP-09 fix contract) ──────────────────────


class TestPatchTaskIntents:
    """The T2-GAP-09 fix contract.

    Legacy ``_task_update_change_kind`` picks a single dominant kind from
    a TaskUpdate payload — if a payload has both status and addBlockedBy,
    only the status is "seen" and the dependency change is dropped on the
    floor.  ``PatchTask.decompose`` must capture ALL sub-intents.
    """

    def test_mixed_payload_captures_all_sub_intents(self) -> None:
        """Mixed {status, addBlockedBy, owner, metadata, subject} → all 5 sub-intent categories present."""
        task_ref = _make_task_ref("T-100")
        tool_input = {
            "taskId": "T-100",
            "subject": "New subject",
            "status": "in_progress",
            "owner": "agent-X",
            "addBlockedBy": ["T-001", "T-002"],
            "metadata": {"priority": "high"},
        }
        patch = PatchTask.decompose(tool_input, task_ref)

        # Core T2-GAP-09 assertion: ALL sub-intents captured, not just one.
        assert patch.has_field_updates is True
        assert patch.has_status_change is True
        assert patch.has_owner_change is True
        assert patch.has_add_dependencies is True
        assert patch.has_metadata_updates is True
        assert patch.sub_intent_count == 5

        # Field detail
        assert patch.field_updates == {"subject": "New subject"}
        assert patch.status_target == "in_progress"
        assert patch.owner_target == "agent-X"
        assert len(patch.add_dependencies) == 2
        assert patch.metadata_updates == {"priority": "high"}

    def test_status_only_captures_status(self) -> None:
        task_ref = _make_task_ref()
        tool_input = {"taskId": "T-001", "status": "completed"}
        patch = PatchTask.decompose(tool_input, task_ref)
        assert patch.has_status_change is True
        assert patch.status_target == "completed"
        assert patch.has_field_updates is False
        assert patch.has_owner_change is False
        assert patch.has_add_dependencies is False
        assert patch.has_metadata_updates is False
        assert patch.sub_intent_count == 1

    def test_owner_only_captures_owner(self) -> None:
        task_ref = _make_task_ref()
        tool_input = {"taskId": "T-001", "owner": "alice"}
        patch = PatchTask.decompose(tool_input, task_ref)
        assert patch.has_owner_change is True
        assert patch.owner_target == "alice"
        assert patch.sub_intent_count == 1

    def test_add_blocks_captures_dependency(self) -> None:
        """addBlocks is the downstream direction — still captured as a dependency change."""
        task_ref = _make_task_ref("T-A")
        tool_input = {"taskId": "T-A", "addBlocks": ["T-B"]}
        patch = PatchTask.decompose(tool_input, task_ref)
        assert patch.has_add_dependencies is True
        assert len(patch.add_dependencies) == 1
        # The other end of the edge is T-B, same graph/kind as task_ref
        other = patch.add_dependencies[0]
        assert other.id == "T-B"
        assert other.graph == task_ref.graph
        assert other.kind == task_ref.kind
        intent = patch.dependency_intents[0]
        assert intent.dependent == other
        assert intent.prerequisite == task_ref
        assert intent.source_field == "blocks"

    def test_blocked_by_uses_canonical_dependent_to_prerequisite_direction(self) -> None:
        task_ref = _make_task_ref("dependent")
        prerequisite = _make_task_ref("prerequisite")
        patch = PatchTask.decompose({"addBlockedBy": [prerequisite.id]}, task_ref)

        intent = patch.dependency_intents[0]
        assert intent.operation == "add"
        assert intent.dependent == task_ref
        assert intent.prerequisite == prerequisite
        assert patch.add_dependencies == (prerequisite,)

    def test_add_blocked_by_list(self) -> None:
        """addBlockedBy with a list → multiple add_dependencies captured."""
        task_ref = _make_task_ref("T-Z")
        tool_input = {"taskId": "T-Z", "addBlockedBy": ["T-1", "T-2", "T-3"]}
        patch = PatchTask.decompose(tool_input, task_ref)
        assert len(patch.add_dependencies) == 3
        ids = [r.id for r in patch.add_dependencies]
        assert ids == ["T-1", "T-2", "T-3"]

    def test_add_blocks_single_string(self) -> None:
        """Single-string addBlocks also works (not just lists)."""
        task_ref = _make_task_ref("T-A")
        tool_input = {"taskId": "T-A", "addBlocks": "T-B"}
        patch = PatchTask.decompose(tool_input, task_ref)
        assert len(patch.add_dependencies) == 1
        assert patch.add_dependencies[0].id == "T-B"

    def test_remove_dependencies_captured(self) -> None:
        task_ref = _make_task_ref("T-A")
        tool_input = {
            "taskId": "T-A",
            "removeBlocks": ["T-B"],
            "removeBlockedBy": ["T-C"],
        }
        patch = PatchTask.decompose(tool_input, task_ref)
        assert patch.has_remove_dependencies is True
        assert len(patch.remove_dependencies) == 2
        ids = [r.id for r in patch.remove_dependencies]
        assert set(ids) == {"T-B", "T-C"}

    def test_field_updates_subject_description_activeform(self) -> None:
        task_ref = _make_task_ref()
        tool_input = {
            "taskId": "T-001",
            "subject": "New subject",
            "description": "New desc",
            "activeForm": "Doing thing",
        }
        patch = PatchTask.decompose(tool_input, task_ref)
        assert patch.has_field_updates is True
        assert patch.field_updates == {
            "subject": "New subject",
            "description": "New desc",
            "activeForm": "Doing thing",
        }

    def test_empty_payload_is_empty_patch(self) -> None:
        task_ref = _make_task_ref()
        tool_input: dict = {"taskId": "T-001"}
        patch = PatchTask.decompose(tool_input, task_ref)
        assert patch.is_empty is True
        assert patch.sub_intent_count == 0

    def test_metadata_empty_dict_not_counted(self) -> None:
        """Empty metadata dict → no metadata sub-intent."""
        task_ref = _make_task_ref()
        tool_input = {"taskId": "T-001", "metadata": {}}
        patch = PatchTask.decompose(tool_input, task_ref)
        assert patch.has_metadata_updates is False

    def test_task_ref_preserved(self) -> None:
        task_ref = _make_task_ref("T-999")
        tool_input = {"taskId": "T-999", "status": "completed"}
        patch = PatchTask.decompose(tool_input, task_ref)
        assert patch.task_ref is task_ref
        assert patch.task_ref.id == "T-999"

    def test_dependency_refs_are_plan_task_kind(self) -> None:
        """Decomposed dependency refs inherit graph/kind from the task_ref."""
        task_ref = NodeRef(graph="plan", kind="task", id="T-main")
        tool_input = {"taskId": "T-main", "addBlockedBy": ["T-dep"]}
        patch = PatchTask.decompose(tool_input, task_ref)
        dep_ref = patch.add_dependencies[0]
        assert dep_ref.graph == "plan"
        assert dep_ref.kind == "task"
        assert dep_ref.id == "T-dep"

    def test_fully_mixed_payload_every_category(self) -> None:
        """All 6 sub-intent categories at once — T2-GAP-09 stress test."""
        task_ref = _make_task_ref("T-1")
        tool_input = {
            "taskId": "T-1",
            "subject": "S",
            "description": "D",
            "activeForm": "A",
            "status": "in_progress",
            "owner": "agent-A",
            "addBlockedBy": ["T-2"],
            "removeBlocks": ["T-3"],
            "metadata": {"k": "v"},
        }
        patch = PatchTask.decompose(tool_input, task_ref)
        # 6 categories: fields + status + owner + add_deps + remove_deps + metadata
        assert patch.sub_intent_count == 6
        assert patch.is_empty is False
