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
# pylint: disable=W0404,W0621
"""Tests for Plan Graph invalidation propagation.

Reopening an upstream task marks completed downstream tasks
``needs_recheck`` (base_status stays completed); independent branches
are untouched. ``revalidate`` clears the derived flag after upstream
tasks are completed again.
"""

from __future__ import annotations

from pathlib import Path

from lkb.application import LkbApplicationService
from lkb.commands import GraphCommand
from lkb.plan_graph import plan_command_dispatcher
from lkb.repository import JsonFileLkbRepository


def _svc(tmp_home: Path):
    repo = JsonFileLkbRepository(home=tmp_home)
    board_id = repo.resolve_board(explicit_id="tms-test").board_id
    return LkbApplicationService(repository=repo), repo, board_id


_n = 0


def _cid() -> str:
    global _n
    _n += 1
    return f"t-{_n}"


def _exec(svc, dispatcher, kind, board_id, *, payload=None, actor="agent-a", reason=None):
    return svc.execute(
        GraphCommand(
            command_id=_cid(),
            board_id=board_id,
            actor=actor,
            kind=kind,
            payload=payload or {},
            reason=reason,
        ),
        validate=dispatcher.validate,
        apply=dispatcher.apply,
    )


def _complete(svc, dispatcher, board_id, task_id, actor="agent-a"):
    _exec(svc, dispatcher, "claim_task", board_id, payload={"task_id": task_id}, actor=actor)
    _exec(svc, dispatcher, "start_task", board_id, payload={"task_id": task_id}, actor=actor)
    _exec(svc, dispatcher, "complete_task", board_id, payload={"task_id": task_id}, actor=actor)


def test_reopen_marks_downstream_completed_needs_recheck(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-1", "subject": "upstream"},
    )
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-2", "subject": "downstream"},
    )
    _exec(svc, dispatcher, "add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"})
    _complete(svc, dispatcher, board_id, "T-1")
    _complete(svc, dispatcher, board_id, "T-2")

    # Reopen T-1.
    _exec(
        svc,
        dispatcher,
        "reopen_task",
        board_id,
        payload={"task_id": "T-1"},
        reason="contract changed",
    )

    env = repo._get_store(board_id).load()
    t2 = next(n for n in env.nodes.values() if n.get("ref") == "plan:task:T-2")
    # base_status stays completed; derived becomes needs_recheck.
    assert t2["state"] == "completed"
    assert t2["payload"].get("derived_status") == "needs_recheck"
    assert t2["payload"].get("invalidation_cause") == "plan:task:T-1"
    # An invalidation_propagation event was recorded.
    assert any(e.get("type") == "invalidation_propagation" for e in env.events)


def test_reopen_does_not_affect_independent_branch(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-1", "subject": "upstream"},
    )
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-0", "subject": "independent"},
    )
    _complete(svc, dispatcher, board_id, "T-1")
    _complete(svc, dispatcher, board_id, "T-0")

    _exec(svc, dispatcher, "reopen_task", board_id, payload={"task_id": "T-1"}, reason="x")

    env = repo._get_store(board_id).load()
    t0 = next(n for n in env.nodes.values() if n.get("ref") == "plan:task:T-0")
    assert t0["payload"].get("derived_status") is None  # unaffected


def test_revalidate_clears_needs_recheck(tmp_home: Path) -> None:
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-1", "subject": "upstream"},
    )
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-2", "subject": "downstream"},
    )
    _exec(svc, dispatcher, "add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"})
    _complete(svc, dispatcher, board_id, "T-1")
    _complete(svc, dispatcher, board_id, "T-2")
    _exec(svc, dispatcher, "reopen_task", board_id, payload={"task_id": "T-1"}, reason="x")

    # Issue #1 fix: revalidate may NOT bypass an un-revalidated upstream.
    # While T-1 is still pending, revalidating T-2 must be denied rather
    # than washing away its needs_recheck status.
    r_pre = _exec(svc, dispatcher, "revalidate", board_id, payload={"task_id": "T-2"})
    assert r_pre.decision == "denied"
    assert "needs_recheck" in (r_pre.reason or "")

    # Re-complete upstream T-1, then revalidate T-2.
    # Reopen cleared the owner (claim released), so T-1 must be re-claimed
    # before it can be started again.
    _exec(svc, dispatcher, "claim_task", board_id, payload={"task_id": "T-1"}, actor="agent-a")
    _exec(svc, dispatcher, "start_task", board_id, payload={"task_id": "T-1"}, actor="agent-a")
    _exec(svc, dispatcher, "complete_task", board_id, payload={"task_id": "T-1"}, actor="agent-a")

    _exec(svc, dispatcher, "revalidate", board_id, payload={"task_id": "T-2"})

    env = repo._get_store(board_id).load()
    t2 = next(n for n in env.nodes.values() if n.get("ref") == "plan:task:T-2")
    assert t2["payload"].get("derived_status") is None


def test_needs_recheck_dependency_blocks_downstream_claim(tmp_home: Path) -> None:
    """Spec §5.9: needs_recheck must be treated as an unsatisfied dependency.

    After T-1 Reopen, T-2 (base=completed, derived=needs_recheck) must NOT
    be treated as a completed prerequisite for T-3.  T-3 should be blocked
    and its Claim/Start should be denied.
    """
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    for tid in ("T-1", "T-2", "T-3"):
        _exec(
            svc,
            dispatcher,
            "create_task",
            board_id,
            payload={"task_id": tid, "subject": tid},
        )
    # T-2 depends on T-1, T-3 depends on T-2.
    _exec(svc, dispatcher, "add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"})
    _exec(svc, dispatcher, "add_dependency", board_id, payload={"task_id": "T-3", "depends_on": "T-2"})
    _complete(svc, dispatcher, board_id, "T-1")
    _complete(svc, dispatcher, board_id, "T-2")
    # Reopen T-1 -> T-2 becomes needs_recheck.
    _exec(svc, dispatcher, "reopen_task", board_id, payload={"task_id": "T-1"}, reason="x")

    # T-3 must now be blocked because T-2 is needs_recheck (not a satisfied dep).
    # ClaimTask on T-3 must be denied (blocked).
    claim_result = _exec(svc, dispatcher, "claim_task", board_id, payload={"task_id": "T-3"}, actor="agent-c")
    assert claim_result.decision == "denied"
    assert "blocked" in (claim_result.reason or "")


def test_start_task_denied_for_needs_recheck(tmp_home: Path) -> None:
    """Spec §6.5: Task must not be in needs_recheck to start.

    A task that has been reopened (base=pending, derived=none) but later had
    its derived_status set to needs_recheck must not be startable until
    revalidated.  We simulate by directly reopening a completed task whose
    downstream was completed, then trying to start the downstream.
    """
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    for tid in ("T-1", "T-2"):
        _exec(
            svc,
            dispatcher,
            "create_task",
            board_id,
            payload={"task_id": tid, "subject": tid},
        )
    _exec(svc, dispatcher, "add_dependency", board_id, payload={"task_id": "T-2", "depends_on": "T-1"})
    _complete(svc, dispatcher, board_id, "T-1")
    _complete(svc, dispatcher, board_id, "T-2")
    # Reopen T-1 -> T-2 becomes needs_recheck.
    _exec(svc, dispatcher, "reopen_task", board_id, payload={"task_id": "T-1"}, reason="x")

    # T-2 is base=completed, derived=needs_recheck.  StartTask must be denied.
    # T-2 needs to be pending first to test StartTask; but StartTask checks
    # derived_status regardless of base.  Reopen T-2 to make it pending.
    _exec(svc, dispatcher, "reopen_task", board_id, payload={"task_id": "T-2"}, reason="recheck")
    # Now T-2 is pending but still has needs_recheck derived flag from the
    # earlier invalidation (Reopen only clears base, not derived).
    start_result = _exec(svc, dispatcher, "start_task", board_id, payload={"task_id": "T-2"}, actor="agent-a")
    assert start_result.decision == "denied"
    assert "needs_recheck" in (start_result.reason or "")


def test_claim_task_denied_for_stale_node_revision(tmp_home: Path) -> None:
    """Spec §6.4 #7 / LKB-STATE-007: expected_node_revision mismatch -> denied.

    A ClaimTask command carrying expected_node_revision that doesn't match
    the current node revision must be denied with stale_revision, not
    silently overwrite.
    """
    svc, repo, board_id = _svc(tmp_home)
    dispatcher = plan_command_dispatcher()
    _exec(
        svc,
        dispatcher,
        "create_task",
        board_id,
        payload={"task_id": "T-1", "subject": "T1"},
    )
    # Bump node revision by updating fields.
    _exec(
        svc,
        dispatcher,
        "update_task_fields",
        board_id,
        payload={"task_id": "T-1", "subject": "T1-v2"},
    )

    # Claim with stale expected_node_revision=1 (current is 2).
    from lkb.commands import GraphCommand

    svc_obj = svc
    cmd = GraphCommand(
        command_id="stale-rev-claim",
        board_id=board_id,
        actor="agent-a",
        kind="claim_task",
        payload={"task_id": "T-1"},
        expected_node_revision=1,  # stale
    )
    result = svc_obj.execute(cmd, validate=dispatcher.validate, apply=dispatcher.apply)
    assert result.decision == "denied"
    assert "stale_revision" in (result.reason or "")
