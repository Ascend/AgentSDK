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
"""
Tool System Extension Layer

Provides optional tool bundle loading and per-agent tool configuration
without modifying upstream tool_system code.

Architecture:
    - bundles.py: Tool bundle definitions
    - registry_ext.py: Extended ToolRegistry with bundle support
    - agent_config.py: Agent tool configuration dataclass

Upstream patches are stored in patches/tool_system/ for quick adaptation.
"""

from __future__ import annotations


from .bundles import (
    TOOL_BUNDLES,
    MODE_BUNDLES,
    ALL_BUNDLE_NAMES,
    get_bundle_tools,
    get_all_bundle_tools,
)

from .registry_ext import ToolRegistryExt

from .agent_config import AgentToolConfig, ToolMode, load_tool_config

from .team_filter import (
    TEAM_ONLY_TOOL_NAMES,
    filter_team_only_tools,
    has_team_context,
)

__all__ = [
    "TOOL_BUNDLES",
    "MODE_BUNDLES",
    "ALL_BUNDLE_NAMES",
    "get_bundle_tools",
    "get_all_bundle_tools",
    "TEAM_ONLY_TOOL_NAMES",
    "ToolRegistryExt",
    "AgentToolConfig",
    "ToolMode",
    "filter_team_only_tools",
    "has_team_context",
    "load_tool_config",
]
