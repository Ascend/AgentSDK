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

"""ClawcodexAgentRuntime — concrete ``AgentRuntime`` Protocol adapter.

Wraps ``extensions.api.query.QueryRunner`` and maps its ``QueryEvent``
stream (TextDelta / ToolCallEvent / ToolResultEvent / PhaseComplete /
TurnComplete / SessionComplete) onto the sum-type ``AgentEvent`` in
``extensions.orchestrator_runtime.protocols.messages``.

``AgentRunner`` still owns ``QueryRunner`` today. This adapter is the
reference mapping for a future backend swap (Aider / Continue / etc.).
``stream()`` and ``resume()`` both delegate to ``QueryRunner``; each
``stream()`` call builds a new runner from the constructor
``QueryConfig``, matching upstream semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

from extensions.orchestrator_runtime.protocols.agent_runtime import (
    AgentRuntime,
    SessionContext,
)
from extensions.orchestrator_runtime.protocols.messages import (
    AgentEvent,
    PhaseComplete as ProtocolPhaseComplete,
    SessionComplete as ProtocolSessionComplete,
    TextDelta as ProtocolTextDelta,
    ToolCallEvent as ProtocolToolCallEvent,
    ToolResultEvent as ProtocolToolResultEvent,
)


def _map_query_event(event: Any) -> AgentEvent:
    """Map one ``extensions.api.query.QueryEvent`` to ``AgentEvent``.

    Type-discriminated dispatch (avoids attribute-name drift between the two
    dataclass hierarchies).
    """
    cls_name = type(event).__name__
    if cls_name == "TextDelta":
        # upstream: TextDelta(content: str)
        return AgentEvent(
            type="text_delta",
            payload=ProtocolTextDelta(text=getattr(event, "content", "")),
        )
    if cls_name == "ToolCallEvent":
        # upstream: ToolCallEvent(tool_name, params, tool_use_id=None, ...)
        return AgentEvent(
            type="tool_call",
            payload=ProtocolToolCallEvent(
                tool_name=getattr(event, "tool_name", ""),
                tool_input=dict(getattr(event, "params", {}) or {}),
                call_id=getattr(event, "tool_use_id", "") or "",
            ),
        )
    if cls_name == "ToolResultEvent":
        # upstream: ToolResultEvent(tool_name, result, tool_use_id=None)
        return AgentEvent(
            type="tool_result",
            payload=ProtocolToolResultEvent(
                call_id=getattr(event, "tool_use_id", "") or "",
                output=getattr(event, "result", None),
                is_error=False,
            ),
        )
    if cls_name == "PhaseComplete":
        # upstream: PhaseComplete(phase: int, turn_count: int)
        # protocol: PhaseComplete(phase: str, cost=0.0, turn_count=0)
        return AgentEvent(
            type="phase_complete",
            payload=ProtocolPhaseComplete(
                phase=str(getattr(event, "phase", "")),
                turn_count=int(getattr(event, "turn_count", 0)),
            ),
        )
    if cls_name == "TurnComplete":
        # TurnComplete(turn: int) — fold into a synthetic PhaseComplete.
        return AgentEvent(
            type="phase_complete",
            payload=ProtocolPhaseComplete(
                phase="turn",
                turn_count=int(getattr(event, "turn", 0)),
            ),
        )
    if cls_name == "SessionComplete":
        # SessionComplete(reason: str)
        return AgentEvent(
            type="session_complete",
            payload=ProtocolSessionComplete(reason=getattr(event, "reason", "")),
        )
    # Unknown event — drop with a generic phase_complete so callers see it.
    return AgentEvent(
        type="phase_complete",
        payload=ProtocolPhaseComplete(phase=cls_name or "unknown"),
    )


class ClawcodexAgentRuntime(AgentRuntime):
    """Adapter that wraps ``extensions.api.query.QueryRunner``."""

    def __init__(self, config_factory: Any | None = None) -> None:
        """``config_factory(prompt, workspace, provider_name, model, tools,
        session_id) -> QueryConfig`` lets callers customise QueryConfig
        construction; default factory builds a vanilla QueryConfig.
        """
        self._config_factory = config_factory or _default_query_config_factory

    def _build_runner(
        self,
        *,
        prompt: str,
        workspace: Path,
        provider_name: str | None,
        model: str | None,
        tools: list[str] | None,
        session_id: str | None,
    ) -> Any:
        from extensions.api.query import QueryRunner

        config = self._config_factory(
            prompt=prompt,
            workspace=workspace,
            provider_name=provider_name,
            model=model,
            tools=tools,
            session_id=session_id,
        )
        return QueryRunner(config)

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
    ) -> AsyncIterator[AgentEvent]:
        runner = self._build_runner(
            prompt=prompt,
            workspace=workspace,
            provider_name=provider_name,
            model=model,
            tools=tools,
            session_id=session_id,
        )
        async for event in runner.stream():
            yield _map_query_event(event)

    async def resume(
        self,
        session_id: str,
        prompt: str,
        workspace: Path,
    ) -> AsyncIterator[AgentEvent]:
        # QueryRunner does not have a separate ``resume``; callers resume
        # by passing ``session_id`` into ``QueryConfig`` and re-``stream()``.
        # Default to that behaviour.
        async for event in self.stream(
            prompt=prompt,
            workspace=workspace,
            session_id=session_id,
        ):
            yield event


def _default_query_config_factory(
    *,
    prompt: str,
    workspace: Path,
    provider_name: str | None,
    model: str | None,
    tools: list[str] | None,
    session_id: str | None,
) -> Any:
    """Build a default ``QueryConfig`` mirroring agent_runner's typical call."""
    from extensions.api.query import QueryConfig

    # source QueryConfig has tools field, local does not yet
    # pylint: disable=E1123
    return QueryConfig(
        prompt=prompt,
        workspace=str(workspace),
        provider=provider_name,
        model=model,
        tools=tools,
        run_id=session_id,
    )


__all__ = ["ClawcodexAgentRuntime"]
