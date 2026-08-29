#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""WI-6.4 tests — ``is_team_lead`` predicate (4 truth cases)."""

from __future__ import annotations

from pathlib import Path


from src.services.swarm.team_membership import is_team_lead
from src.tool_system.context import ToolContext


def test_is_team_lead_true_when_agent_id_matches(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    ctx.team = {"team_name": "t", "lead_agent_id": "lead-123"}
    ctx.agent_id = "lead-123"
    assert is_team_lead(ctx) is True


def test_is_team_lead_false_when_agent_id_is_member_not_lead(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    ctx.team = {"team_name": "t", "lead_agent_id": "lead-123"}
    ctx.agent_id = "member-456"
    assert is_team_lead(ctx) is False


def test_is_team_lead_false_when_no_team_active(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    ctx.agent_id = "lead-123"
    # ctx.team is None
    assert is_team_lead(ctx) is False


def test_is_team_lead_false_when_agent_id_unset(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    ctx.team = {"team_name": "t", "lead_agent_id": "lead-123"}
    # ctx.agent_id is None
    assert is_team_lead(ctx) is False


def test_is_team_lead_false_when_team_lacks_lead_agent_id(tmp_path: Path) -> None:
    """Defensive — a malformed team dict missing ``lead_agent_id``
    must NOT spuriously authorize anyone.
    """
    ctx = ToolContext(workspace_root=tmp_path)
    ctx.team = {"team_name": "t"}  # no lead_agent_id
    ctx.agent_id = "lead-123"
    assert is_team_lead(ctx) is False


def test_is_team_lead_false_when_team_is_not_a_dict(tmp_path: Path) -> None:
    """Defensive — a malformed team field must not authorize.

    Per the docstring: the predicate collapses 'team unavailable' and
    'not authorized' into the same False return so authorization
    branches handle both safely. Granting permission to either would
    be catastrophic.
    """
    ctx = ToolContext(workspace_root=tmp_path)
    ctx.team = "not-a-dict"  # type: ignore[assignment]
    ctx.agent_id = "lead-123"
    assert is_team_lead(ctx) is False
