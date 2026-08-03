"""Integration tests for the LKB host adapter against the real Graph Store.

Issue #13: the existing host-adapter tests monkeypatch the board to a fixed
stub, so they never exercise real workspace resolution, projection, or the
Store idempotency cache.  These tests route Task-v2 calls through the real
JsonFileLkbRepository so the projection (issue #5), divergence protocol
(issue #7), and idempotent retry (issue #12) are covered end-to-end.
"""

from __future__ import annotations


# pylint: disable=E0611,W0621
from pathlib import Path
from types import SimpleNamespace

import pytest

from lkb.clawcodex_task_adapter import try_handle


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CLAWCODEX_HOME", str(home))
    return home


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: True)


def _install(monkeypatch: pytest.MonkeyPatch, tmp_home: Path, key: str):
    from lkb.repository import JsonFileLkbRepository

    repo = JsonFileLkbRepository(home=tmp_home)
    board_id = repo.resolve_board(explicit_id=key).board_id
    monkeypatch.setattr("lkb.repository.get_repository", lambda *, home=None: repo)
    monkeypatch.setattr(repo, "resolve_board", lambda *a, **kw: SimpleNamespace(board_id=board_id))
    return repo, board_id


def _ctx(
    tasks=None,
    *,
    tool_use_id=None,
    agent_id="agent-a",
    actor_roles: tuple[str, ...] = (),
    session_id: str = "session-a",
) -> SimpleNamespace:
    return SimpleNamespace(
        tasks=tasks or {},
        agent_id=agent_id,
        actor_roles=actor_roles,
        tool_use_id=tool_use_id,
        workspace_root=None,
        session_id=session_id,
        lkb_plan_id=None,
    )


def test_tasklist_projection_has_blocks_lkb_and_board_summary(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #5: TaskList must carry blocks/blockedBy computed from the
    Store edges, a nested ``lkb`` summary per task, and a top-level
    ``lkbBoard`` aggregate.
    """
    _enable(monkeypatch)
    _install(monkeypatch, tmp_home, "proj")
    ctx = _ctx()
    try_handle("TaskCreate", {"subject": "A", "description": "d"}, ctx)
    try_handle("TaskCreate", {"subject": "B", "description": "d"}, ctx)
    a_id = next(t for t in ctx.tasks if ctx.tasks[t]["subject"] == "A")
    b_id = next(t for t in ctx.tasks if ctx.tasks[t]["subject"] == "B")
    # B depends on A.
    try_handle("TaskUpdate", {"taskId": b_id, "addBlockedBy": [a_id]}, ctx)
    _, result = try_handle("TaskList", {}, ctx)
    tasks = {t["id"]: t for t in result.output["tasks"]}
    # blocks/blockedBy are computed from the canonical edges.
    assert tasks[a_id]["blocks"] == [b_id]
    assert tasks[b_id]["blockedBy"] == [a_id]
    # Each task has a nested lkb summary.
    assert "lkb" in tasks[a_id]
    assert tasks[a_id]["lkb"]["derivedStatus"] == "ready"
    assert tasks[b_id]["lkb"]["derivedStatus"] == "blocked"
    assert tasks[b_id]["lkb"]["activeBlockers"] == [a_id]
    assert tasks[a_id]["lkb"]["nextActionCommands"] == [
        {
            "action": "claim_task",
            "tool": "TaskUpdate",
            "input": {"taskId": a_id, "owner": "$self"},
            "description": "Claim this task as the current Agent.",
        }
    ]
    # Top-level lkbBoard aggregate.
    assert "lkbBoard" in result.output
    counts = result.output["lkbBoard"]["counts"]
    assert counts["ready"] == 1
    assert counts["blocked"] == 1


def test_denial_returns_executable_self_claim_recovery(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    _install(monkeypatch, tmp_home, "self-claim-recovery")
    ctx = _ctx(agent_id="opaque-runtime-agent-id")
    _, created = try_handle("TaskCreate", {"subject": "Claim me"}, ctx)
    task_id = created.output["task"]["id"]

    _, denied = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "owner": "agent-a"},
        ctx,
    )

    assert denied.is_error is True
    assert denied.output["reason"]["code"] == "override_not_authorized"
    recovery = denied.output["nextActions"]
    assert recovery == denied.output["lkb"]["nextActions"]
    assert recovery[0]["tool"] == "TaskUpdate"
    assert recovery[0]["input"] == {"taskId": task_id, "owner": "$self"}

    _, claimed = try_handle("TaskUpdate", recovery[0]["input"], ctx)
    assert claimed.output["success"] is True
    assert claimed.output["task"]["owner"] == "opaque-runtime-agent-id"
    assert claimed.output["task"]["lkb"]["derivedStatus"] == "ready"


def test_completed_task_denial_returns_executable_reopen_recovery(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch)
    _install(monkeypatch, tmp_home, "reopen-recovery")
    ctx = _ctx(agent_id="agent-a")
    _, created = try_handle("TaskCreate", {"subject": "Revise me"}, ctx)
    task_id = created.output["task"]["id"]
    _, started = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "owner": "$self", "status": "in_progress"},
        ctx,
    )
    assert started.is_error is False
    _, completed = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "status": "completed"},
        ctx,
    )
    assert completed.is_error is False

    _, denied = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "status": "in_progress"},
        ctx,
    )

    assert denied.is_error is True
    assert denied.output["reason"]["code"] == "already_resolved"
    recovery = denied.output["nextActions"]
    assert recovery == denied.output["lkb"]["nextActions"]
    assert recovery == [
        {
            "action": "reopen_task",
            "tool": "TaskUpdate",
            "input": {
                "taskId": task_id,
                "status": "pending",
                "reason": "Reopen completed task for additional work.",
            },
            "description": (
                "Explicitly reopen the completed task; LKB will release its old claim "
                "and propagate downstream rechecks."
            ),
        }
    ]

    _, reopened = try_handle("TaskUpdate", recovery[0]["input"], ctx)
    assert reopened.is_error is False
    assert reopened.output["task"]["status"] == "pending"
    assert reopened.output["task"]["owner"] is None


def test_task_update_prompt_explains_explicit_reopen() -> None:
    from clawcodex_ext.tool_system.tools.tasks_v2 import TaskUpdateTool

    prompt = TaskUpdateTool.prompt()
    normalized_prompt = " ".join(prompt.split())

    assert "explicitly reopen a completed task" in prompt
    assert '"status": "pending"' in prompt
    assert "Do not try to move a completed task directly to `in_progress`" in normalized_prompt
    assert "marks completed downstream tasks for recheck" in normalized_prompt


def test_metadata_owner_is_rejected_instead_of_masquerading_as_claim(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch)
    _install(monkeypatch, tmp_home, "reserved-metadata-owner")
    ctx = _ctx(agent_id="opaque-runtime-agent-id")
    _, created = try_handle("TaskCreate", {"subject": "Claim me"}, ctx)
    task_id = created.output["task"]["id"]

    _, denied = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "metadata": {"owner": "agent-a"}},
        ctx,
    )

    assert denied.is_error is True
    assert denied.output["reason"]["code"] == "invalid_metadata"
    assert "never changes task ownership" in denied.output["reason"]["message"]
    assert denied.output["nextActions"][0]["input"]["owner"] == "$self"
    assert ctx.tasks[task_id]["owner"] is None


def test_host_asserted_role_authorizes_transfer_but_actor_name_does_not(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lkb.commands import CommandResult
    from lkb.graph_types import BoardPolicy

    _enable(monkeypatch)
    repo, board_id = _install(monkeypatch, tmp_home, "role-override")

    def set_policy(envelope):
        envelope.board["policy"] = BoardPolicy(force_override_roles=("admin",)).to_dict()
        return envelope, CommandResult(decision="committed", command_id="set-role-policy")

    repo.execute_atomic(
        board_id,
        "set-role-policy",
        "set-role-policy-hash",
        None,
        set_policy,
        actor="system",
        reason="configure trusted override role",
    )

    owner_ctx = _ctx(agent_id="agent-a")
    _, created = try_handle("TaskCreate", {"subject": "Transfer me"}, owner_ctx)
    task_id = created.output["task"]["id"]
    _, claimed = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "owner": "agent-a"},
        owner_ctx,
    )
    assert claimed.is_error is False
    assert claimed.output["claimId"]

    spoofed_ctx = _ctx(agent_id="admin")
    _, denied = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "owner": "agent-c", "reason": "spoof actor name"},
        spoofed_ctx,
    )
    assert denied.is_error is True
    assert "override_not_authorized" in denied.output["reason"]["message"]

    admin_ctx = _ctx(agent_id="operator", actor_roles=("admin",))
    _, transferred = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "owner": "agent-c", "reason": "approved reassignment"},
        admin_ctx,
    )
    assert transferred.is_error is False
    assert transferred.output["claimId"]
    assert admin_ctx.tasks[task_id]["owner"] == "agent-c"


def test_self_owner_sentinel_claims_and_starts_as_trusted_actor(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model-safe sentinel must never become a persisted owner string."""

    _enable(monkeypatch)
    repo, board_id = _install(monkeypatch, tmp_home, "self-owner")
    ctx = _ctx(agent_id="opaque-agent-id")
    _, created = try_handle("TaskCreate", {"subject": "Claim safely"}, ctx)
    task_id = created.output["task"]["id"]

    _, started = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "owner": "$self", "status": "in_progress"},
        ctx,
    )

    assert started.is_error is False
    assert started.output["updatedFields"] == ["owner", "status"]
    assert started.output["claimId"]
    assert ctx.tasks[task_id]["owner"] == "opaque-agent-id"
    assert ctx.tasks[task_id]["status"] == "in_progress"
    envelope = repo._get_store(board_id).load()
    node = next(n for n in envelope.nodes.values() if n.get("ref") == f"{ctx.lkb_plan_id}:task:{task_id}")
    assert node["owner"] == "opaque-agent-id"
    assert node["state"] == "in_progress"
    claim = envelope.claims[started.output["claimId"]]
    assert claim["owner_ref"] == f"{ctx.lkb_plan_id}:agent:opaque-agent-id"


def test_divergence_detected_when_context_has_unimported_tasks(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #7: when the context already holds tasks the Store does not
    know about, TaskList must NOT silently drop them; it reports
    divergence and preserves the in-memory tasks.
    """
    _enable(monkeypatch)
    _install(monkeypatch, tmp_home, "divergence")
    ctx = _ctx(tasks={"T-legacy": {"id": "T-legacy", "subject": "old", "status": "pending"}})
    _, result = try_handle("TaskList", {}, ctx)
    # The legacy task is preserved (not wiped).
    assert "T-legacy" in ctx.tasks
    assert result.output.get("divergence")
    assert "T-legacy" in result.output["divergence"]["unimported"]


def test_idempotent_retry_does_not_duplicate_create(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #12: a transport retry with the same tool_use_id returns the
    first result and does NOT create a second Task.
    """
    _enable(monkeypatch)
    repo, board_id = _install(monkeypatch, tmp_home, "idem")
    ctx1 = _ctx(tool_use_id="call-42")
    _, first = try_handle("TaskCreate", {"subject": "Once", "description": "d"}, ctx1)
    first_id = first.output["task"]["id"]
    # Retry with the same tool_use_id (simulating a transport retry).
    ctx2 = _ctx(tool_use_id="call-42")
    _, second = try_handle("TaskCreate", {"subject": "Once", "description": "d"}, ctx2)
    # Same task id returned (cached result), no duplicate created.
    assert second.output["task"]["id"] == first_id
    env = repo._get_store(board_id).load()
    plan_tasks = [n for n in env.nodes.values() if str(n.get("ref", "")).startswith(f"{ctx1.lkb_plan_id}:task:")]
    assert len(plan_tasks) == 1


def test_sessions_share_board_but_not_default_plan(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new session gets an independent Plan inside the same Board."""
    _enable(monkeypatch)
    _repo, _board_id = _install(monkeypatch, tmp_home, "plan-isolation")
    first = _ctx(session_id="session-one")
    second = _ctx(session_id="session-two")

    _, created = try_handle("TaskCreate", {"subject": "Only in first"}, first)
    task_id = created.output["task"]["id"]
    _, listed = try_handle("TaskList", {}, second)

    assert first.lkb_plan_id != second.lkb_plan_id
    assert task_id in first.tasks
    assert listed.output["tasks"] == []
    assert listed.output["lkbBoard"]["planId"] == second.lkb_plan_id


def test_equal_task_and_idempotency_ids_do_not_collide_across_plans(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch)
    repo, board_id = _install(monkeypatch, tmp_home, "plan-id-collision")
    first = _ctx(session_id="session-one", tool_use_id="same-tool-call")
    second = _ctx(session_id="session-two", tool_use_id="same-tool-call")

    _, first_result = try_handle("TaskCreate", {"subject": "First"}, first)
    _, second_result = try_handle("TaskCreate", {"subject": "Second"}, second)

    assert first_result.is_error is False
    assert second_result.is_error is False
    assert first_result.output["task"]["id"] == second_result.output["task"]["id"]
    envelope = repo._get_store(board_id).load()
    refs = {
        str(node.get("ref", ""))
        for node in envelope.nodes.values()
        if str(node.get("ref", "")).endswith(f":task:{first_result.output['task']['id']}")
    }
    assert refs == {
        f"{first.lkb_plan_id}:task:{first_result.output['task']['id']}",
        f"{second.lkb_plan_id}:task:{second_result.output['task']['id']}",
    }


def test_explicit_plan_binding_shares_an_existing_plan(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second session can opt in to an existing Plan explicitly."""
    _enable(monkeypatch)
    repo, board_id = _install(monkeypatch, tmp_home, "plan-binding")
    first = _ctx(session_id="session-one")
    second = _ctx(session_id="session-two")

    _, created = try_handle("TaskCreate", {"subject": "Shared deliberately"}, first)
    task_id = created.output["task"]["id"]

    from lkb.plan_scope import bind_plan

    bind_plan(repo, board_id, second.session_id, first.lkb_plan_id)
    second.lkb_plan_id = first.lkb_plan_id
    _, listed = try_handle("TaskList", {}, second)

    assert [task["id"] for task in listed.output["tasks"]] == [task_id]
    assert listed.output["lkbBoard"]["planId"] == first.lkb_plan_id


def test_suspended_plan_releases_claims_and_rejects_writes(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    repo, board_id = _install(monkeypatch, tmp_home, "plan-lifecycle")
    ctx = _ctx(session_id="session-owner")
    _, created = try_handle("TaskCreate", {"subject": "Pause safely"}, ctx)
    task_id = created.output["task"]["id"]
    _, claimed = try_handle(
        "TaskUpdate",
        {"taskId": task_id, "owner": ctx.agent_id},
        ctx,
    )
    assert claimed.is_error is False

    from lkb.plan_scope import set_plan_state

    set_plan_state(repo, board_id, ctx.session_id, ctx.lkb_plan_id, "suspended")
    _, denied = try_handle("TaskCreate", {"subject": "Must not be written"}, ctx)

    assert denied.is_error is True
    assert "plan_not_active" in denied.output["reason"]["message"]
    envelope = repo._get_store(board_id).load()
    node = next(n for n in envelope.nodes.values() if n.get("ref") == f"{ctx.lkb_plan_id}:task:{task_id}")
    claim = next(c for c in envelope.claims.values() if c.get("task_ref") == node["ref"])
    assert node["owner"] is None
    assert claim["status"] == "released"


def test_status_explain_audit_read_from_store(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #6: /lkb explain|audit read the Graph Store authority (not
    the old Context Sidecar) when the Plan Graph is enabled.
    """
    _enable(monkeypatch)
    _install(monkeypatch, tmp_home, "cmds")
    ctx = _ctx()
    try_handle("TaskCreate", {"subject": "Probe", "description": "d"}, ctx)
    tid = next(iter(ctx.tasks))

    from clawcodex_ext.command_system.lkb_command import (
        _audit_from_store,
        _explain_from_store,
        _lkb_is_on,
    )

    assert _lkb_is_on() is True
    tool_ctx = SimpleNamespace(
        tasks=ctx.tasks,
        agent_id="agent-a",
        workspace_root=None,
        session_id=ctx.session_id,
        lkb_plan_id=ctx.lkb_plan_id,
    )
    explain = _explain_from_store(tool_ctx, tid)
    assert "Probe" in explain
    # Audit may report "no events" for a fresh task, but it must not raise
    # and must read from the Store envelope.
    audit = _audit_from_store(tool_ctx, tid)
    assert isinstance(audit, str)
