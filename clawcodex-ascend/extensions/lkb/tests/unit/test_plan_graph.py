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

# ruff: noqa: F841

"""Unit tests for Plan Graph domain command handlers (Phase 4, spec §6).

Exercises each handler through the real LkbApplicationService +
JsonFileLkbRepository stack via the PlanCommandDispatcher, covering the
validation rules from spec §6.2-§6.8.
"""

from __future__ import annotations

from pathlib import Path


from lkb.application import LkbApplicationService
from lkb.commands import CommandResult, GraphCommand
from lkb.error_codes import LkbErrorCode
from lkb.graph_types import NodeRef
from lkb.plan_graph import PlanCommandDispatcher, plan_command_dispatcher
from lkb.repository import JsonFileLkbRepository


# ── harness ──────────────────────────────────────────────────────────


def _svc(tmp_home: Path) -> tuple[LkbApplicationService, JsonFileLkbRepository, str]:
    repo = JsonFileLkbRepository(home=tmp_home)
    board_id = repo.resolve_board(explicit_id="plan-test").board_id
    dispatcher = plan_command_dispatcher()
    svc = LkbApplicationService(repository=repo)
    return svc, repo, board_id


def _cmd(
    kind: str,
    board_id: str,
    *,
    command_id: str | None = None,
    actor: str = "agent-a",
    payload: dict | None = None,
    reason: str | None = None,
    roles: tuple[str, ...] = (),
) -> GraphCommand:
    return GraphCommand(
        command_id=command_id or f"cmd-{kind}-{_counter()}",
        board_id=board_id,
        actor=actor,
        kind=kind,
        payload=payload or {},
        reason=reason,
        roles=roles,
    )


_counter_n = 0


def _counter() -> int:
    global _counter_n
    _counter_n += 1
    return _counter_n


def _execute(
    svc: LkbApplicationService,
    dispatcher: PlanCommandDispatcher,
    command: GraphCommand,
):
    return svc.execute(command, validate=dispatcher.validate, apply=dispatcher.apply)


def _task_ref(task_id: str) -> NodeRef:
    return NodeRef("plan", "task", task_id)


# ── CreateTask ───────────────────────────────────────────────────────


def test_create_task_commits_and_persists(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()

    result = _execute(
        svc,
        dispatcher,
        _cmd("create_task", board_id, payload={"task_id": "T-1", "subject": "First"}),
    )
    assert result.decision == "committed"

    snap = repo.load_snapshot(board_id)
    assert _task_ref("T-1") in snap.nodes
    assert snap.nodes[_task_ref("T-1")].title == "First"
    assert snap.nodes[_task_ref("T-1")].state == "pending"


def test_create_task_rejects_empty_subject(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    result = _execute(
        svc,
        plan_command_dispatcher(),
        _cmd("create_task", board_id, payload={"task_id": "T-x", "subject": ""}),
    )
    assert result.decision == "denied"


def test_create_task_rejects_duplicate(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _execute(svc, dispatcher, _cmd("create_task", board_id, payload={"task_id": "T-1", "subject": "A"}))
    result = _execute(
        svc,
        dispatcher,
        _cmd("create_task", board_id, command_id="dup", payload={"task_id": "T-1", "subject": "B"}),
    )
    assert result.decision == "denied"


# ── AddDependency ────────────────────────────────────────────────────


def _seed_tasks(svc, dispatcher, board_id, *ids: str) -> None:
    for tid in ids:
        _execute(svc, dispatcher, _cmd("create_task", board_id, payload={"task_id": tid, "subject": tid}))


def test_add_dependency_commits(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2")
    result = _execute(
        svc,
        dispatcher,
        _cmd("add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"}),
    )
    assert result.decision == "committed"
    snap = repo.load_snapshot(board_id)
    assert any(e.type == "depends_on" for e in snap.edges.values())


def test_add_dependency_rejects_self_dependency(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    result = _execute(
        svc,
        dispatcher,
        _cmd("add_dependency", board_id, payload={"task_id": "T-1", "depends_on": "T-1"}),
    )
    assert result.decision == "denied"


def test_add_dependency_rejects_cycle(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2", "T-3")
    # T-2 -> T-1, T-3 -> T-2, then T-1 -> T-3 would close the cycle.
    _execute(
        svc,
        dispatcher,
        _cmd("add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"}),
    )
    _execute(
        svc,
        dispatcher,
        _cmd("add_dependency", board_id, payload={"task_id": "T-3", "depends_on": "T-2"}),
    )
    result = _execute(
        svc,
        dispatcher,
        _cmd("add_dependency", board_id, payload={"task_id": "T-1", "depends_on": "T-3"}),
    )
    assert result.decision == "denied"


def test_add_dependency_idempotent(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2")
    payload = {"task_id": "T-2", "depends_on": "T-1"}
    r1 = _execute(svc, dispatcher, _cmd("add_dependency", board_id, payload=payload))
    r2 = _execute(svc, dispatcher, _cmd("add_dependency", board_id, command_id="dup-edge", payload=payload))
    assert r1.decision == "committed"
    assert r2.decision == "committed"
    snap = repo.load_snapshot(board_id)
    assert sum(1 for e in snap.edges.values() if e.type == "depends_on") == 1


# ── ClaimTask ────────────────────────────────────────────────────────


def test_claim_task_succeeds_for_ready_unowned(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    command = _cmd("claim_task", board_id, payload={"task_id": "T-1"})
    result = _execute(svc, dispatcher, command)
    assert result.decision == "committed"
    assert result.claim_id
    retried = _execute(svc, dispatcher, command)
    assert retried.claim_id == result.claim_id
    snap = repo.load_snapshot(board_id)
    assert snap.nodes[_task_ref("T-1")].owner == "agent-a"


def test_claim_task_idempotent_same_owner(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    first = _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}))
    result = _execute(
        svc,
        dispatcher,
        _cmd("claim_task", board_id, command_id="re-claim", payload={"task_id": "T-1"}),
    )
    assert result.decision == "committed"
    assert result.claim_id == first.claim_id


def test_claim_task_denied_for_different_owner(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))
    result = _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-b"))
    assert result.decision == "denied"


def test_claim_task_denied_for_blocked(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2")
    _execute(
        svc,
        dispatcher,
        _cmd("add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"}),
    )
    result = _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-2"}))
    assert result.decision == "denied"


# ── StartTask / CompleteTask / ReopenTask ────────────────────────────


def test_start_then_complete_lifecycle(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}))
    assert _execute(svc, dispatcher, _cmd("start_task", board_id, payload={"task_id": "T-1"})).decision == "committed"
    assert (
        _execute(svc, dispatcher, _cmd("complete_task", board_id, payload={"task_id": "T-1"})).decision == "committed"
    )
    snap = repo.load_snapshot(board_id)
    assert snap.nodes[_task_ref("T-1")].state == "completed"


def test_complete_rejects_when_not_in_progress(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    result = _execute(svc, dispatcher, _cmd("complete_task", board_id, payload={"task_id": "T-1"}))
    assert result.decision == "denied"


def test_start_rejects_non_owner(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))
    result = _execute(svc, dispatcher, _cmd("start_task", board_id, payload={"task_id": "T-1"}, actor="agent-b"))
    assert result.decision == "denied"


def test_reopen_releases_claim(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}))
    _execute(svc, dispatcher, _cmd("start_task", board_id, payload={"task_id": "T-1"}))
    _execute(svc, dispatcher, _cmd("complete_task", board_id, payload={"task_id": "T-1"}))
    result = _execute(svc, dispatcher, _cmd("reopen_task", board_id, payload={"task_id": "T-1"}))
    assert result.decision == "committed"
    snap = repo.load_snapshot(board_id)
    assert snap.nodes[_task_ref("T-1")].state == "pending"


def test_reopen_clears_owner_and_requires_reclaim(tmp_home: Path) -> None:
    """Reopen releases the claim AND clears ``node.owner`` - the previous
    owner must re-claim before starting again.  Keeping the owner would
    let them bypass the claim protocol and re-start directly (Store
    invariant: ``active_claim.owner_ref.id == task_node.owner``).
    """
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}))
    _execute(svc, dispatcher, _cmd("start_task", board_id, payload={"task_id": "T-1"}))
    _execute(svc, dispatcher, _cmd("complete_task", board_id, payload={"task_id": "T-1"}))
    _execute(svc, dispatcher, _cmd("reopen_task", board_id, payload={"task_id": "T-1"}))

    snap = repo.load_snapshot(board_id)
    assert snap.nodes[_task_ref("T-1")].owner is None

    # The previous owner cannot re-start without a fresh claim.
    denied = _execute(svc, dispatcher, _cmd("start_task", board_id, payload={"task_id": "T-1"}))
    assert denied.decision == "denied"

    # Re-claim restores the normal lifecycle.
    claimed = _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}))
    assert claimed.decision == "committed"
    started = _execute(svc, dispatcher, _cmd("start_task", board_id, payload={"task_id": "T-1"}))
    assert started.decision == "committed"


# ── DeleteTask ───────────────────────────────────────────────────────


def test_delete_rejects_referenced_without_cascade(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2")
    _execute(
        svc,
        dispatcher,
        _cmd("add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"}),
    )
    result = _execute(svc, dispatcher, _cmd("delete_task", board_id, payload={"task_id": "T-1"}))
    assert result.decision == "denied"


def test_delete_cascade_removes_edges(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2")
    _execute(
        svc,
        dispatcher,
        _cmd("add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"}),
    )
    command = _cmd("delete_task", board_id, payload={"task_id": "T-1", "cascade": True})
    result = _execute(svc, dispatcher, command)
    assert result.decision == "committed"
    assert result.affected_refs == ("plan:task:T-1", "plan:task:T-2")
    assert _execute(svc, dispatcher, command).affected_refs == result.affected_refs
    snap = repo.load_snapshot(board_id)
    assert _task_ref("T-1") not in snap.nodes
    assert all(e.type != "depends_on" for e in snap.edges.values())


def test_delete_removes_dangling_claims(tmp_home: Path) -> None:
    """Deleting a task must not leave claim records pointing at a node that
    no longer exists — including terminal (completed) claims, which would
    otherwise dangle and wrongly attach to a re-created task with the same
    id.
    """
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2")
    # T-1: full lifecycle -> completed claim.  T-2: active claim only.
    for tid in ("T-1", "T-2"):
        _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": tid}))
    _execute(svc, dispatcher, _cmd("start_task", board_id, payload={"task_id": "T-1"}))
    _execute(svc, dispatcher, _cmd("complete_task", board_id, payload={"task_id": "T-1"}))
    env = repo._get_store(board_id).load()
    assert any(
        str(c.get("task_ref", "")) == "plan:task:T-1" and c.get("status") == "completed" for c in env.claims.values()
    )

    for tid in ("T-1", "T-2"):
        result = _execute(svc, dispatcher, _cmd("delete_task", board_id, payload={"task_id": tid}))
        assert result.decision == "committed"
    env = repo._get_store(board_id).load()
    assert all(str(c.get("task_ref", "")) not in ("plan:task:T-1", "plan:task:T-2") for c in env.claims.values())


# ── UpdateTaskFields ─────────────────────────────────────────────────


def test_update_task_fields(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "update_task_fields",
            board_id,
            payload={"task_id": "T-1", "subject": "Renamed", "description": "desc"},
        ),
    )
    assert result.decision == "committed"
    snap = repo.load_snapshot(board_id)
    node = snap.nodes[_task_ref("T-1")]
    assert node.title == "Renamed"
    assert node.payload.get("description") == "desc"


def test_update_task_metadata_merges_and_none_deletes(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(
        svc,
        dispatcher,
        _cmd(
            "update_task_fields",
            board_id,
            payload={"task_id": "T-1", "metadata": {"keep": 1, "remove": 2}},
        ),
    )
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "update_task_fields",
            board_id,
            payload={"task_id": "T-1", "metadata": {"remove": None, "added": 3}},
        ),
    )
    assert result.decision == "committed"
    metadata = dict(repo.load_snapshot(board_id).nodes[_task_ref("T-1")].payload["metadata"])
    assert metadata == {"keep": 1, "added": 3}


def test_expected_node_revision_applies_to_field_updates(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    command = GraphCommand(
        command_id="stale-field-update",
        board_id=board_id,
        actor="agent-a",
        kind="update_task_fields",
        expected_node_revision=0,
        payload={"task_id": "T-1", "subject": "must not land"},
    )
    result = _execute(svc, dispatcher, command)
    assert result.decision == "denied"
    assert "stale_revision" in (result.reason or "")
    assert repo.load_snapshot(board_id).nodes[_task_ref("T-1")].title == "T-1"


def test_start_requires_an_active_claim(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    result = _execute(
        svc,
        dispatcher,
        _cmd("start_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"),
    )
    assert result.decision == "denied"
    assert "owner_required" in (result.reason or "")
    assert repo.load_snapshot(board_id).nodes[_task_ref("T-1")].state == "pending"


def test_complete_by_non_owner_is_denied(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(
        svc,
        dispatcher,
        _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"),
    )
    _execute(
        svc,
        dispatcher,
        _cmd("start_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"),
    )
    result = _execute(
        svc,
        dispatcher,
        _cmd("complete_task", board_id, payload={"task_id": "T-1"}, actor="agent-b"),
    )
    assert result.decision == "denied"
    assert "not_owner" in (result.reason or "")
    assert repo.load_snapshot(board_id).nodes[_task_ref("T-1")].state == "in_progress"


# ── ReleaseTask authorization (spec §5.6, LKB-CLAIM-008/009) ────────


def test_release_by_owner_succeeds(tmp_home: Path) -> None:
    """Owner releasing their own claim must succeed without override."""
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "release_task",
            board_id,
            payload={"task_id": "T-1"},
            actor="agent-a",
            reason="done with my part",
        ),
    )
    assert result.decision == "committed"
    snap = repo.load_snapshot(board_id)
    assert snap.nodes[_task_ref("T-1")].owner is None


def test_release_unowned_task_is_idempotent(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    result = _execute(
        svc,
        dispatcher,
        _cmd("release_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"),
    )
    assert result.decision == "committed"
    assert repo.load_snapshot(board_id).nodes[_task_ref("T-1")].owner is None


def test_release_by_non_owner_denied_without_force(tmp_home: Path) -> None:
    """Spec §5.6: non-owner release must be denied unless actor has
    force_override_roles AND provides a reason.
    """
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))
    # agent-b has no override role and no reason -> denied.
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "release_task",
            board_id,
            payload={"task_id": "T-1"},
            actor="agent-b",
        ),
    )
    assert result.decision == "denied"
    assert result.error_code is LkbErrorCode.OVERRIDE_NOT_AUTHORIZED
    assert "override_not_authorized" in (result.reason or "")
    # Owner unchanged.
    snap = repo.load_snapshot(board_id)
    assert snap.nodes[_task_ref("T-1")].owner == "agent-a"


def _svc_with_policy(
    tmp_home: Path, *, force_override_roles: tuple[str, ...]
) -> tuple[LkbApplicationService, JsonFileLkbRepository, str]:
    """Build a service whose board policy grants *force_override_roles*.

    Policy is written via ``execute_atomic`` so the store revision chain
    stays consistent (writing directly to the store would desync
    primary/backup revisions and trip the corruption guard).
    """
    from lkb.graph_types import BoardPolicy

    repo = JsonFileLkbRepository(home=tmp_home)
    board_id = repo.resolve_board(explicit_id="plan-test").board_id
    policy = BoardPolicy(force_override_roles=force_override_roles)

    def mutate(env):
        env.board["policy"] = policy.to_dict()
        return env, CommandResult(
            decision="committed",
            command_id="policy-setup",
        )

    repo.execute_atomic(
        board_id,
        "policy-setup",
        "policy-setup-hash",
        None,
        mutate,
        actor="system",
        reason="test policy setup",
    )
    svc = LkbApplicationService(repository=repo)
    return svc, repo, board_id


def test_release_by_non_owner_with_role_and_reason_succeeds(tmp_home: Path) -> None:
    """Spec §5.6 / LKB-CLAIM-009: force_override_roles + reason -> success + audit."""
    svc, repo, board_id = _svc_with_policy(tmp_home, force_override_roles=("admin",))
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "release_task",
            board_id,
            payload={"task_id": "T-1"},
            actor="agent-b",
            reason="force release: agent-a unresponsive",
            roles=("admin",),
        ),
    )
    assert result.decision == "committed"
    snap = repo.load_snapshot(board_id)
    assert snap.nodes[_task_ref("T-1")].owner is None
    # Override audit event must be recorded.
    store = repo._get_store(board_id)
    env = store.load()
    override_events = [e for e in env.events if e.get("type") == "claim_override"]
    assert override_events
    ev = override_events[-1]
    assert ev.get("actor") == "agent-b"
    assert ev.get("reason") == "force release: agent-a unresponsive"
    assert ev.get("previous_owner") == "agent-a"


def test_release_by_non_owner_with_role_but_no_reason_denied(tmp_home: Path) -> None:
    """Spec §5.6 / LKB-CLAIM-008: force_override_roles without reason -> denied."""
    svc, repo, board_id = _svc_with_policy(tmp_home, force_override_roles=("admin",))
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "release_task",
            board_id,
            payload={"task_id": "T-1"},
            actor="agent-b",
            roles=("admin",),
            # no reason provided
        ),
    )
    assert result.decision == "denied"
    assert result.error_code is LkbErrorCode.OVERRIDE_REASON_REQUIRED
    assert "override_reason_required" in (result.reason or "")


# ── TransferTask (spec §5.6, §6.4, LKB-CLAIM-008/009) ───────────────


def test_actor_name_does_not_grant_force_override_role(tmp_home: Path) -> None:
    """An actor called ``admin`` has no admin role unless the host asserts it."""
    svc, repo, board_id = _svc_with_policy(tmp_home, force_override_roles=("admin",))
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))

    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "release_task",
            board_id,
            payload={"task_id": "T-1"},
            actor="admin",
            reason="actor id must not imply a role",
        ),
    )

    assert result.decision == "denied"
    assert "override_not_authorized" in (result.reason or "")
    assert repo.load_snapshot(board_id).nodes[_task_ref("T-1")].owner == "agent-a"


def test_transfer_task_succeeds_with_role_and_reason(tmp_home: Path) -> None:
    """Spec §6.4: transferring ownership to another agent requires
    force_override_roles authorization + reason + audit event.
    """
    svc, repo, board_id = _svc_with_policy(tmp_home, force_override_roles=("admin",))
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "transfer_task",
            board_id,
            payload={"task_id": "T-1", "new_owner": "agent-c"},
            actor="agent-b",
            reason="reassign to agent-c for expertise",
            roles=("admin",),
        ),
    )
    assert result.decision == "committed"
    assert result.claim_id
    snap = repo.load_snapshot(board_id)
    assert snap.nodes[_task_ref("T-1")].owner == "agent-c"
    # Old claim released, new claim created.
    store = repo._get_store(board_id)
    env = store.load()
    t1_claims = [c for c in env.claims.values() if c.get("task_ref") == "plan:task:T-1"]
    active = [c for c in t1_claims if c.get("status") == "active"]
    assert len(active) == 1
    # Issue #10: owner_ref is a plan:agent:<actor> NodeRef, not a bare string.
    assert active[0].get("owner_ref") == "plan:agent:agent-c"
    # Override audit event recorded.
    override_events = [e for e in env.events if e.get("type") == "claim_override"]
    assert override_events
    ev = override_events[-1]
    assert ev.get("actor") == "agent-b"
    assert ev.get("reason") == "reassign to agent-c for expertise"
    assert ev.get("previous_owner") == "agent-a"


def test_transfer_task_denied_without_role(tmp_home: Path) -> None:
    """Spec §6.4: unauthorized actor cannot transfer."""
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "transfer_task",
            board_id,
            payload={"task_id": "T-1", "new_owner": "agent-c"},
            actor="agent-b",
            reason="try to transfer",
        ),
    )
    assert result.decision == "denied"
    assert "override_not_authorized" in (result.reason or "")


def test_transfer_task_denied_without_reason(tmp_home: Path) -> None:
    """Spec §5.6 / LKB-CLAIM-008: force transfer requires reason."""
    svc, repo, board_id = _svc_with_policy(tmp_home, force_override_roles=("admin",))
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "transfer_task",
            board_id,
            payload={"task_id": "T-1", "new_owner": "agent-c"},
            actor="agent-b",
            roles=("admin",),
            # no reason
        ),
    )
    assert result.decision == "denied"
    assert "override_reason_required" in (result.reason or "")


# ── PatchTask (spec §6.1, T2-GAP-09, LKB-ADAPT-011/012) ─────────────


def test_patch_transfer_preserves_host_asserted_roles(tmp_home: Path) -> None:
    svc, repo, board_id = _svc_with_policy(tmp_home, force_override_roles=("admin",))
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    _execute(svc, dispatcher, _cmd("claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a"))

    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "patch_task",
            board_id,
            payload={
                "task_id": "T-1",
                "owner": "agent-c",
                "metadata": {"transfer": "approved"},
            },
            actor="agent-b",
            reason="approved patch transfer",
            roles=("admin",),
        ),
    )

    assert result.decision == "committed"
    assert result.claim_id
    node = repo.load_snapshot(board_id).nodes[_task_ref("T-1")]
    assert node.owner == "agent-c"
    assert dict(node.payload["metadata"])["transfer"] == "approved"


def test_patch_task_atomic_status_owner_and_metadata(tmp_home: Path) -> None:
    """Spec §6.1 / T2-GAP-09: mixed TaskUpdate must apply all sub-intents
    atomically — one revision bump, one command result, all-or-nothing.
    """
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1")
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "patch_task",
            board_id,
            payload={
                "task_id": "T-1",
                "status": "in_progress",
                "owner": "agent-a",  # claim by actor
                "metadata": {"priority": "high"},
                "subject": "Patched subject",
            },
            actor="agent-a",
        ),
    )
    assert result.decision == "committed"
    snap = repo.load_snapshot(board_id)
    node = snap.nodes[_task_ref("T-1")]
    # All three sub-intents applied.
    assert node.state == "in_progress"
    assert node.owner == "agent-a"
    assert node.title == "Patched subject"
    assert dict(node.payload.get("metadata", {})).get("priority") == "high"
    # Exactly one revision bump (from 1 to 2, not 1 -> 2 -> 3 -> 4).
    assert node.revision == 2


def test_patch_task_denied_leaves_envelope_unchanged(tmp_home: Path) -> None:
    """Spec §6.1 / LKB-ADAPT-011: if any sub-intent fails, nothing commits."""
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2")
    # T-1 is pending; completing it via patch requires in_progress first,
    # so a status=completed sub-intent must be denied by CompleteTask's
    # validate.  Nothing should commit.
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "patch_task",
            board_id,
            payload={
                "task_id": "T-1",
                "status": "completed",  # invalid: T-1 is pending, not in_progress
                "metadata": {"note": "should not land"},
            },
            actor="agent-a",
        ),
    )
    assert result.decision == "denied"
    snap = repo.load_snapshot(board_id)
    node = snap.nodes[_task_ref("T-1")]
    # Nothing changed.
    assert node.state == "pending"
    assert dict(node.payload.get("metadata", {})).get("note") is None


def test_patch_rolls_back_dependency_when_later_start_is_denied(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2")
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "patch_task",
            board_id,
            payload={
                "task_id": "T-2",
                "addBlockedBy": ["T-1"],
                "status": "in_progress",
                "owner": "agent-a",
            },
            actor="agent-a",
        ),
    )
    assert result.decision == "denied"
    snapshot = repo.load_snapshot(board_id)
    assert snapshot.nodes[_task_ref("T-2")].state == "pending"
    assert not any(
        edge.source == _task_ref("T-2") and edge.target == _task_ref("T-1") for edge in snapshot.edges.values()
    )


def test_patch_task_dependency_and_field_update(tmp_home: Path) -> None:
    """Spec §6.1: dependency sub-intents are part of the atomic patch."""
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _seed_tasks(svc, dispatcher, board_id, "T-1", "T-2")
    result = _execute(
        svc,
        dispatcher,
        _cmd(
            "patch_task",
            board_id,
            payload={
                "task_id": "T-2",
                "addBlockedBy": ["T-1"],
                "subject": "T-2 with prereq",
            },
            actor="agent-a",
        ),
    )
    assert result.decision == "committed"
    snap = repo.load_snapshot(board_id)
    node = snap.nodes[_task_ref("T-2")]
    assert node.title == "T-2 with prereq"
    # Dependency edge created in the same atomic snapshot.
    assert any(
        e.type == "depends_on" and e.source == _task_ref("T-2") and e.target == _task_ref("T-1")
        for e in snap.edges.values()
    )
