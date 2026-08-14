# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
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

from typing import Any, Callable, Literal, Protocol, runtime_checkable

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
    tools: list[str] | None
    source: AgentSourceLiteral
    base_dir: str
    model: str | None
    provider: str | None
    permission_mode: str | None
    max_turns: int | None
    background: bool
    color: str | None
    memory: str | None
    omit_claude_md: bool
    disallowed_tools: list[str] | None
    hooks: dict[str, Any] | None
    skills: list[str] | None
    isolation: Literal["worktree", "remote"] | None
    required_mcp_servers: list[str] | None
    mcp_servers: list[Any] | None
    effort: str | None
    get_system_prompt: Callable[..., str] | None
    callback: Callable[[], None] | None
    critical_system_reminder: str | None


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
