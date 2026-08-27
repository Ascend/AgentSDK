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

"""TeamMem — permission & scope policy.

Single authority on team-memory visibility. The :class:`TeamMemoryService`
calls into :class:`TeamMemoryPolicy` on every read/write/delete so the
scope rules from §1.7 are enforced in exactly one place.

Rules (§1.7)::

    | scenario                      | rule                                            |
    |-------------------------------|-------------------------------------------------|
    | requester not in team.json    | deny read/write                                 |
    | scope=team                    | all members read; members write                 |
    | scope=lead_only               | lead read/write; member write needs lead approval
    | scope=agent_pair              | author + related_agents read; lead can audit    |
    | delete                        | lead any; author own; others denied             |
    | compact                       | lead only                                       |

The policy is stateless and side-effect free (the audit log lives in
the store, not here). It raises :class:`TeamMemoryPermissionError` on
denial; callers are expected to catch and surface it.
"""

from __future__ import annotations

from typing import Iterable

from clawcodex_ext.services.swarm.team_file import TeamFile  # pylint: disable=import-error

from .team_memory import EntryScope, TeamMemoryEntry, TeamMemoryPermissionError

__all__ = ["TeamMemoryPolicy"]


class TeamMemoryPolicy:
    """Team-memory permission + scope checker.

    Constructed with a :class:`TeamFile` roster. All methods are pure
    functions of ``(roster, agent_id, entry)`` — no I/O, no audit.
    """

    def __init__(self, *, team_file: TeamFile) -> None:
        self._team = team_file

    # -- roster helpers --------------------------------------------------

    @property
    def team_file(self) -> TeamFile:
        return self._team

    def _member_ids(self) -> set[str]:
        ids = {m.agent_id for m in self._team.members}
        # The lead may or may not be in the members list (TeamCreate
        # initializes ``members: []`` and teammates are appended later).
        ids.add(self._team.lead_agent_id)
        return ids

    def is_member(self, agent_id: str) -> bool:
        return agent_id in self._member_ids()

    def is_lead(self, agent_id: str) -> bool:
        return agent_id == self._team.lead_agent_id

    # -- read ------------------------------------------------------------

    def authorize_read(self, *, requester_agent_id: str) -> None:
        """Membership gate for any read operation (recall / list)."""
        if not self.is_member(requester_agent_id):
            raise TeamMemoryPermissionError(
                f"agent {requester_agent_id!r} is not a member of team {self._team.team_name!r}"
            )

    def can_see(self, *, requester_agent_id: str, entry: TeamMemoryEntry) -> bool:
        """Scope-level visibility filter applied after ``authorize_read``.

        - ``team``: every member sees it.
        - ``lead_only``: only the lead.
        - ``agent_pair``: author + named ``related_agents`` (lead can audit).
        """
        if entry.scope == "team":
            return True
        if entry.scope == "lead_only":
            return self.is_lead(requester_agent_id)
        if entry.scope == "agent_pair":
            if self.is_lead(requester_agent_id):
                return True  # audit visibility
            return requester_agent_id in {entry.author_agent_id, *entry.related_agents}
        # Unknown scope → fail closed.
        return False

    # -- write ------------------------------------------------------------

    def authorize_write(
        self,
        *,
        author_agent_id: str,
        scope: EntryScope,
        related_agents: Iterable[str] = (),
        require_lead_approval: bool = True,
    ) -> None:
        """Per-scope write authorization.

        - Non-members cannot write at all.
        - ``team``: any member writes.
        - ``lead_only``: lead writes freely; a non-lead member may only
          write ``lead_only`` when ``require_lead_approval`` is False
          (config-driven escape hatch — §1.5
          ``require_lead_approval_for_lead_only``). Otherwise the
          caller must downgrade to ``team`` scope.
        - ``agent_pair``: any member may write (the related_agents
          tuple names the readers, not the writers).
        """
        if not self.is_member(author_agent_id):
            raise TeamMemoryPermissionError(
                f"agent {author_agent_id!r} is not a member of team {self._team.team_name!r}"
            )
        if scope == "lead_only" and not self.is_lead(author_agent_id):
            if require_lead_approval:
                raise TeamMemoryPermissionError(
                    "scope=lead_only writes require lead approval; "
                    "either downgrade to scope=team or have the lead write."
                )

    # -- delete / compact ------------------------------------------------

    def authorize_delete(self, *, actor: str, entry: TeamMemoryEntry) -> None:
        """Lead can delete anything; author can delete their own; others denied."""
        if not self.is_member(actor):
            raise TeamMemoryPermissionError(f"agent {actor!r} is not a member of team {self._team.team_name!r}")
        if self.is_lead(actor):
            return
        if actor == entry.author_agent_id:
            return
        raise TeamMemoryPermissionError(
            f"agent {actor!r} may not delete entry {entry.id!r} (authored by {entry.author_agent_id!r})"
        )

    def authorize_compact(self, actor: str) -> None:
        """Compaction collapses team history — lead only."""
        if not self.is_lead(actor):
            raise TeamMemoryPermissionError(f"only the team lead may compact team memory (actor={actor!r})")
