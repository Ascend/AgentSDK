#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Registry extension — tracks agent-created tools at runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool_system.build_tool import Tool

# In-memory store of agent-created tools, keyed by name.
# Persisted to disk via ``persistence.py`` and restored on startup.
_AGENT_CREATED_TOOLS: dict[str, Tool] = {}


def add_tool(tool: Tool) -> None:
    """Register an agent-created tool in the runtime registry."""
    _AGENT_CREATED_TOOLS[tool.name] = tool


def get_tool(name: str) -> Tool | None:
    """Look up an agent-created tool by name."""
    return _AGENT_CREATED_TOOLS.get(name)


def list_tools() -> list[Tool]:
    """Return all agent-created tools."""
    return list(_AGENT_CREATED_TOOLS.values())


def remove_tool(name: str) -> bool:
    """Remove an agent-created tool. Returns True if it existed."""
    return _AGENT_CREATED_TOOLS.pop(name, None) is not None


def clear() -> None:
    """Remove all agent-created tools (used in tests / resets)."""
    _AGENT_CREATED_TOOLS.clear()
