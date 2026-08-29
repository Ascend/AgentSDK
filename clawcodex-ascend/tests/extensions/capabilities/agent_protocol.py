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

"""AgentLoop Protocol — interface for multi-turn agent execution.

Concrete implementation: src/tool_system/agent_loop.py.
"""

from __future__ import annotations

from typing import Any, Protocol

from .provider_protocol import LLMProviderProtocol
from .tool_protocol import ToolContextProtocol, ToolRegistryProtocol

__all__ = ["AgentLoopProtocol", "AgentLoopResultProtocol"]


class AgentLoopResultProtocol(Protocol):
    """Protocol for the result returned by an agent loop.

    Concrete: clawcodex_ext.tool_system.renderers.AgentLoopResult.
    """

    response_text: str
    usage: dict[str, Any] | None
    num_turns: int


class AgentLoopProtocol(Protocol):
    """Protocol for multi-turn tool-calling agent loops.

    Provides: run_agent_loop, summarize_tool_result, summarize_tool_use,
    is_anthropic_provider.
    """

    def run_agent_loop(
        self,
        conversation: "Conversation",  # noqa: F821
        provider: LLMProviderProtocol,
        tool_registry: ToolRegistryProtocol,
        tool_context: ToolContextProtocol,
        max_turns: int = 20,
        stream: bool = False,
        verbose: bool = False,
        on_event: "ToolEventHandler | None" = None,  # noqa: F821
        on_text_chunk: "TextChunkHandler | None" = None,  # noqa: F821
        cancel_signal: "AbortSignal | None" = None,  # noqa: F821
    ) -> AgentLoopResultProtocol: ...  # pragma: no cover

    def summarize_tool_result(self, name: str, output: object) -> str: ...  # pragma: no cover

    def summarize_tool_use(self, name: str, tool_input: dict[str, object]) -> str: ...  # pragma: no cover

    def is_anthropic_provider(self, provider: LLMProviderProtocol) -> bool: ...  # pragma: no cover
