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
"""ToolSystem Protocol — interface for the tool registry and execution.

Concrete implementation: src/tool_system/registry.py / build_tool.py
(consumed via clawcodex_ext.tool_system.protocol).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from clawcodex_ext.tool_system.protocol import ToolCall, ToolResult

__all__ = [
    "ToolContextProtocol",
    "ToolPermissionContextProtocol",
    "ToolProtocol",
    "ToolRegistryProtocol",
    "ToolSystemProtocol",
]


class ToolPermissionContextProtocol(Protocol):
    """Protocol for permission context passed to tool assembly/dispatch."""

    mode: str
    is_bypass_permissions_mode_available: bool
    should_avoid_permission_prompts: bool

    def blocks(self, tool_name: str) -> bool: ...  # pragma: no cover


class ToolProtocol(Protocol):
    """Protocol for a single tool definition."""

    name: str
    aliases: tuple[str, ...]

    def matches_name(self, name: str) -> bool: ...  # pragma: no cover


class ToolContextProtocol(Protocol):
    """Protocol for the execution context passed to tool dispatch."""

    workspace_root: Path | None
    cwd: Path | None
    plan_mode: bool
    permission_context: ToolPermissionContextProtocol | None


class ToolRegistryProtocol(Protocol):
    """Protocol for a registry of available tools."""

    def register(self, tool: ToolProtocol) -> None: ...  # pragma: no cover

    def unregister(self, name: str) -> bool: ...  # pragma: no cover

    def get(self, name: str) -> ToolProtocol | None: ...  # pragma: no cover

    def list_tools(self) -> list[ToolProtocol]: ...  # pragma: no cover

    def dispatch(self, call: ToolCall, context: ToolContextProtocol) -> ToolResult: ...  # pragma: no cover


class ToolSystemProtocol(Protocol):
    """Protocol for tool registry and tool execution.

    Provides: get_tools, find_tool_by_name, build_tool,
    assemble_tool_pool, dispatch.
    """

    def get_tools(self) -> list[ToolProtocol]: ...  # pragma: no cover

    def find_tool_by_name(self, name: str) -> ToolProtocol | None: ...  # pragma: no cover

    def build_tool(self, tool_def: dict[str, object]) -> ToolProtocol: ...  # pragma: no cover

    def assemble_tool_pool(
        self,
        registry: ToolRegistryProtocol,
        permission_context: ToolPermissionContextProtocol,
        mcp_tools: list[ToolProtocol] | None = None,
    ) -> list[ToolProtocol]: ...  # pragma: no cover

    def dispatch(self, call: ToolCall, context: ToolContextProtocol) -> ToolResult: ...  # pragma: no cover
