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
Tool Bundle Definitions

Defines 4 tool loading modes for agents:
- bare: Zero tools (pure reasoning agent)
- default: Default bundle (bash, edit, read, search)
- clawcodex: All native built-in tools
- all: All available tools
"""

from __future__ import annotations


# Bundle definitions: bundle_name -> list of tool names
TOOL_BUNDLES: dict[str, list[str]] = {
    "default": [
        "Bash",
        "Edit",
        "Write",
        "Read",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
    ],
    "clawcodex": [
        # All native built-in tools
        "AskUserQuestion",
        "Bash",
        "Brief",
        "ClipboardRead",
        "ClipboardWrite",
        "Config",
        "CronCreate",
        "CronDelete",
        "CronList",
        "Edit",
        "EnterPlanMode",
        "EnterWorktree",
        "ExitPlanMode",
        "ExitWorktree",
        "Glob",
        "Grep",
        "LSP",
        "ListMcpResources",
        "MCPTool",
        "NotebookEdit",
        "ReadMcpResource",
        "Read",
        "SendMessage",
        "SendUserMessage",
        "Skill",
        "Sleep",
        "Status",
        "StructuredOutput",
        "TaskCreate",
        "TaskGet",
        "TaskList",
        "TaskOutput",
        "TaskStop",
        "TaskUpdate",
        "TeamCreate",
        "TeamDelete",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
        "Write",
        # Internal tools
        "Agent",
        "ToolSearch",
    ],
}

# Mode to bundle names mapping
MODE_BUNDLES: dict[str, list[str]] = {
    "bare": [],
    "default": ["default"],
    "clawcodex": ["clawcodex"],
    "all": list(TOOL_BUNDLES.keys()),
}

# All available bundle names
ALL_BUNDLE_NAMES: list[str] = list(TOOL_BUNDLES.keys())


def get_bundle_tools(bundle_name: str) -> list[str]:
    """Get tool names for a bundle, returns empty list if bundle not found."""
    return list(TOOL_BUNDLES.get(bundle_name, []))


def get_all_bundle_tools() -> list[str]:
    """Get all tool names across all bundles (deduped)."""
    seen: set[str] = set()
    result: list[str] = []
    for tools in TOOL_BUNDLES.values():
        for t in tools:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result
