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
#

"""TeamMem — Protocol contract for the team-memory service.

Layer 2 → Layer 1 boundary: three-party extensions depend on this
Protocol instead of the concrete
:class:`~extensions.agents.team_memory.TeamMemoryService`; structural
only, with ``runtime_checkable`` for defensive isinstance casts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

__all__ = [
    "TeamMemoryEntryProtocol",
    "TeamMemoryResultProtocol",
    "TeamMemoryServiceProtocol",
    "TeamMemoryStoreProtocol",
]


@runtime_checkable
class TeamMemoryEntryProtocol(Protocol):
    """Read-only view of a team-memory entry."""

    id: str
    team_id: str
    content: str
    summary: str
    author_agent_id: str
    source: str
    scope: str
    tags: tuple[str, ...]
    related_agents: tuple[str, ...]
    created_at: str
    confidence: float


@runtime_checkable
class TeamMemoryResultProtocol(Protocol):
    """One ranked recall result."""

    entry: TeamMemoryEntryProtocol
    score: float
    matched_terms: tuple[str, ...]
    reason: str


@runtime_checkable
class TeamMemoryStoreProtocol(Protocol):
    """Append-only persistence layer."""

    def append(self, entry: Any) -> Any: ...

    def list_entries(self, *, include_expired: bool = False) -> list[Any]: ...

    def get(self, entry_id: str) -> Any | None: ...

    def delete(self, entry_id: str, *, actor: str, reason: str) -> bool: ...

    def compact(self, *, actor: str) -> Any: ...

    def archive(self, *, reason: str) -> Path: ...


@runtime_checkable
class TeamMemoryServiceProtocol(Protocol):
    """Facade for Team / Agent / Tool callers.

    Concrete implementation lives in
    :mod:`extensions.agents.team_memory`, swappable via this Protocol.
    """

    @property
    def team_id(self) -> str: ...

    def remember(
        self,
        content: str,
        *,
        author_agent_id: str,
        tags: Iterable[str] = (),
        source: str = "manual",
        scope: str = "team",
    ) -> Any: ...

    def recall(self, query: Any) -> list[Any]: ...

    def list_entries(
        self,
        *,
        requester_agent_id: str,
        limit: int = 50,
        tags: Iterable[str] = (),
        sources: Iterable[str] = (),
        include_expired: bool = False,
    ) -> list[Any]: ...

    def delete(self, entry_id: str, *, actor: str, reason: str) -> bool: ...

    def compact(self, *, actor: str) -> Any: ...

    def build_prompt_section(self, *, requester_agent_id: str, task: str) -> str: ...

    def record_message_summary(
        self,
        *,
        sender: str,
        recipients: Iterable[str],
        summary: str,
        message: str,
    ) -> Any | None: ...
