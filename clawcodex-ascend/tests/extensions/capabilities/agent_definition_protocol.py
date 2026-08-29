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

"""AgentDefinition Protocol — interface for SOP-convertible agent definitions.

Contract consumed by ``extensions/sop_converter/``; the default
implementation is ``clawcodex_ext.agent.agent_definitions.AgentDefinition``.
Mirrors the field subset the SOP converter consumes (see
``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.3). The
``AgentToolConstants`` namespace exposes the tool-name constants that
``bundle_context`` / ``agent_builder`` borrow from
``clawcodex_ext.agent.constants``.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

__all__ = [
    "AgentDefinitionProtocol",
    "AgentSourceLiteral",
    "AgentToolConstants",
]


# Matches ``clawcodex_ext.agent.agent_definitions.AgentSource``; re-exported
# verbatim to preserve IDE auto-completion parity.
AgentSourceLiteral = Literal[
    "built-in",
    "user",
    "project",
    "managed",
    "plugin",
    "dynamic",
    "clawcodex_ext",
    "extensions",
]


@runtime_checkable
class AgentDefinitionProtocol(Protocol):
    """Protocol for an agent definition the SOP converter can serialize.

    Field names mirror ``AgentDefinition`` for ``@runtime_checkable``
    semantics; Plan-listed aliases (``name``, ``memory_scope``,
    ``persistent``) are read-only properties adapted by the default
    adapter (Phase 3+).
    """

    agent_type: str
    when_to_use: str
    tools: Optional[list[str]]
    source: AgentSourceLiteral
    base_dir: str
    model: Optional[str]
    provider: Optional[str]
    permission_mode: Optional[str]
    max_turns: Optional[int]
    background: bool
    color: Optional[str]
    memory: Optional[str]
    omit_claude_md: bool
    disallowed_tools: Optional[list[str]]
    hooks: Optional[dict[str, Any]]
    skills: Optional[list[str]]
    isolation: Optional[Literal["worktree", "remote"]]
    required_mcp_servers: Optional[list[str]]
    mcp_servers: Optional[list[Any]]
    effort: Optional[str]
    get_system_prompt: Optional[Callable[..., str]]
    callback: Optional[Callable[[], None]]
    critical_system_reminder: Optional[str]


class AgentToolConstants:
    """SOP tool-name and toolset constants (Layer-2 surface).

    Aligned with ``clawcodex_ext.agent.constants.POS_PROXY_BASE_TOOLS``.
    """

    MAX_INLINE_TOOL_DISPLAY: int = 20

    # Names to strip from SOP allowlists when absent from ALL_STATIC_TOOLS.
    UNREGISTERED_SPECIAL_TOOLS: frozenset[str] = frozenset()

    POS_PROXY_BASE_TOOLS: frozenset[str] = frozenset(
        (
            "Skill",
            "ToolSearch",
            "Agent",
            "Read",
            "TodoWrite",
            "StructuredOutput",
            "resource-catalog",
            "register-macro-workflow",
            "register-macro-from-trace",
            "promote-macro-workflow",
        )
    )

    POS_SOP_DOMAIN_AGENT_TOOLS: frozenset[str] = frozenset(
        (
            "Skill",
            "ToolSearch",
            "Bash",
            "Read",
            "TodoWrite",
            "StructuredOutput",
            "resource-catalog",
            "register-macro-workflow",
            "register-macro-from-trace",
            "promote-macro-workflow",
        )
    )

    @classmethod
    def registered_proxy_base_tools(cls) -> list[str]:
        """Sorted proxy allowlist excluding unregistered special tools."""
        return sorted(cls.POS_PROXY_BASE_TOOLS - cls.UNREGISTERED_SPECIAL_TOOLS)

    @classmethod
    def registered_domain_agent_tools(cls) -> list[str]:
        """Sorted domain-agent allowlist excluding unregistered special tools."""
        return sorted(cls.POS_SOP_DOMAIN_AGENT_TOOLS - cls.UNREGISTERED_SPECIAL_TOOLS)
