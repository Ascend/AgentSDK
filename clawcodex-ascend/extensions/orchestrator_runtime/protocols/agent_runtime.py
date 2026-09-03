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

"""AgentRuntime protocol: one multi-step agent stream.

Each ``stream()`` yields events until ``SessionComplete``.
``AgentRunner`` talks to a backend through this interface
(a Clawcodex ``QueryRunner`` wrapper, or a third-party adapter).
"""

from __future__ import annotations
# pylint: disable=W2301

from pathlib import Path
from typing import AsyncIterator, Protocol, runtime_checkable

from .messages import AgentEvent


@runtime_checkable
class SessionContext(Protocol):
    """Per-stream session handle that ``AgentRuntime`` writes to.

    Implementations may back this with ``clawcodex_ext.agent.session.Session``
    (Phase 4 ClawcodexBackend) or a stub (test-only).
    """

    session_id: str
    workspace: Path

    def persist(self) -> None:
        """Flush in-memory state to disk / backend."""
        ...


@runtime_checkable
class AgentRuntime(Protocol):
    """One multi-turn agent execution; emits events until ``SessionComplete``.

    The orchestrator's :class:`AgentRunner` calls :meth:`stream` once per
    AgentSession; the runtime drives the conversation loop, tool execution,
    and emits events until :class:`SessionComplete`.

    Event sequence (semantic):

      - zero or more :class:`TextDelta`
      - zero or more interleave of :class:`ToolCallEvent` / :class:`ToolResultEvent`
      - zero or more :class:`PhaseComplete`
      - exactly one terminal :class:`SessionComplete` (success or failure)

    Notes:
      - Implementations MUST yield ``SessionComplete`` exactly once.
      - Errors during execution MUST yield ``SessionComplete(reason="error",
        final_text=str(exc))`` rather than raising.
    """

    async def stream(
        self,
        *,
        prompt: str,
        workspace: Path,
        provider_name: str | None = None,
        model: str | None = None,
        tools: list[str] | None = None,
        session_id: str | None = None,
        on_session: SessionContext | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    async def resume(
        self,
        session_id: str,
        prompt: str,
        workspace: Path,
    ) -> AsyncIterator[AgentEvent]:
        """Resume a previously persisted session.

        ``SessionComplete`` carries ``reason="resumed"`` on success.
        """
        ...


__all__ = ["AgentRuntime", "SessionContext"]
