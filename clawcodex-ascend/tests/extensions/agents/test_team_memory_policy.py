#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Unit tests for TeamMemoryPolicy.

Covers membership gate, scope visibility (team / lead_only /
agent_pair), write authorization, delete rules, and compact=lead-only.
"""

from __future__ import annotations

import pytest

from clawcodex_ext.services.swarm.team_file import TeamFile, TeamMember

from extensions.agents.team_memory import (
    TeamMemoryEntry,
    TeamMemoryPermissionError,
    make_iso_timestamp,
)
from extensions.agents.team_memory_policy import TeamMemoryPolicy


def _team(lead: str = "lead-1", members: tuple[str, ...] = ("lead-1", "agent-2")) -> TeamFile:
    return TeamFile(
        team_name="t1",
        lead_agent_id=lead,
        members=tuple(TeamMember(agent_id=aid, name=aid) for aid in members),
    )


def _entry(*, author: str = "lead-1", scope: str = "team", related: tuple[str, ...] = ()) -> TeamMemoryEntry:
    return TeamMemoryEntry(
        id="e1",
        team_id="t1",
        content="c",
        summary="s",
        author_agent_id=author,
        created_at=make_iso_timestamp(),
        scope=scope,  # type: ignore[arg-type]
        related_agents=related,
    )


def test_non_member_denied_read() -> None:
    pol = TeamMemoryPolicy(team_file=_team())
    with pytest.raises(TeamMemoryPermissionError):
        pol.authorize_read(requester_agent_id="outsider")


def test_member_can_read_team_scope() -> None:
    pol = TeamMemoryPolicy(team_file=_team())
    pol.authorize_read(requester_agent_id="agent-2")  # no raise
    assert pol.can_see(requester_agent_id="agent-2", entry=_entry(scope="team"))


def test_lead_only_hidden_from_regular_member() -> None:
    pol = TeamMemoryPolicy(team_file=_team())
    assert pol.can_see(requester_agent_id="lead-1", entry=_entry(scope="lead_only"))
    assert not pol.can_see(requester_agent_id="agent-2", entry=_entry(scope="lead_only"))


def test_agent_pair_visible_to_author_and_pair_only() -> None:
    pol = TeamMemoryPolicy(team_file=_team())
    e = _entry(author="agent-2", scope="agent_pair", related=("lead-1",))
    assert pol.can_see(requester_agent_id="agent-2", entry=e)
    assert pol.can_see(requester_agent_id="lead-1", entry=e)  # related + audit
    # Add a third member not in the pair.
    team = _team(members=("lead-1", "agent-2", "agent-3"))
    pol3 = TeamMemoryPolicy(team_file=team)
    assert not pol3.can_see(requester_agent_id="agent-3", entry=e)


def test_lead_only_write_by_member_requires_approval() -> None:
    pol = TeamMemoryPolicy(team_file=_team())
    with pytest.raises(TeamMemoryPermissionError):
        pol.authorize_write(author_agent_id="agent-2", scope="lead_only", require_lead_approval=True)
    # With approval disabled (config escape hatch), member write allowed.
    pol.authorize_write(author_agent_id="agent-2", scope="lead_only", require_lead_approval=False)


def test_lead_can_write_lead_only() -> None:
    pol = TeamMemoryPolicy(team_file=_team())
    pol.authorize_write(author_agent_id="lead-1", scope="lead_only")  # no raise


def test_non_member_denied_write() -> None:
    pol = TeamMemoryPolicy(team_file=_team())
    with pytest.raises(TeamMemoryPermissionError):
        pol.authorize_write(author_agent_id="outsider", scope="team")


def test_delete_author_or_lead_only() -> None:
    pol = TeamMemoryPolicy(team_file=_team())
    e = _entry(author="agent-2")
    pol.authorize_delete(actor="lead-1", entry=e)
    pol.authorize_delete(actor="agent-2", entry=e)
    with pytest.raises(TeamMemoryPermissionError):
        pol.authorize_delete(actor="agent-3", entry=e)


def test_compact_lead_only() -> None:
    pol = TeamMemoryPolicy(team_file=_team())
    pol.authorize_compact("lead-1")  # no raise
    with pytest.raises(TeamMemoryPermissionError):
        pol.authorize_compact("agent-2")
