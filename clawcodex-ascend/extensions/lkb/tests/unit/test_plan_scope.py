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
# pylint: disable=R0402
"""Session Plan resolution must converge under first-use races."""

from __future__ import annotations

import pytest

import lkb.plan_scope as plan_scope
from lkb.repository import JsonFileLkbRepository


def test_resolve_plan_recovers_when_a_concurrent_creator_wins(tmp_home, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = JsonFileLkbRepository(home=tmp_home)
    board_id = repo.resolve_board(explicit_id="scope-race").board_id
    session_id = "shared-session"
    selected = plan_scope.default_plan_id(session_id)
    winner = plan_scope.create_plan(
        repo,
        board_id,
        session_id,
        plan_id=selected,
        title="Session plan",
    )
    real_get_plan = plan_scope.get_plan
    get_calls = 0

    def stale_first_read(repository, current_board_id, plan_id):
        nonlocal get_calls
        get_calls += 1
        if get_calls == 1:
            raise plan_scope.PlanScopeError(f"Plan {plan_id!r} does not exist in this Board")
        return real_get_plan(repository, current_board_id, plan_id)

    def concurrent_collision(*_args, **_kwargs):
        raise plan_scope.PlanScopeError(f"Plan {selected!r} already exists")

    monkeypatch.setattr(plan_scope, "get_plan", stale_first_read)
    monkeypatch.setattr(plan_scope, "create_plan", concurrent_collision)

    resolved = plan_scope.resolve_plan(repo, board_id, session_id)

    assert resolved.plan_id == winner.plan_id
    assert plan_scope.bound_plan_id(repo, board_id, session_id) == winner.plan_id
