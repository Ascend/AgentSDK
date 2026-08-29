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

"""Filter agents by their declared ``required_mcp_servers``.

Port of ``hasRequiredMcpServers`` / ``filterAgentsByMcpRequirements`` in
typescript/src/tools/AgentTool/loadAgentsDir.ts:228-254.

Built-in agents are never dropped — they're trusted regardless of MCP
availability.
"""

from __future__ import annotations

from collections.abc import Iterable

from clawcodex_ext.agent.agent_definitions import AgentDefinition, is_built_in_agent


def has_required_mcp_servers(
    agent: AgentDefinition,
    available_servers: Iterable[str],
) -> bool:
    """Return True iff every required pattern matches an available server.

    Matching is case-insensitive substring (same as the TS reference): the
    pattern ``slack`` matches the server name ``MySlackServer``. Empty
    requirements pass through.
    """
    if not agent.required_mcp_servers:
        return True
    available_lower = [s.lower() for s in available_servers]
    return all(any(pattern.lower() in server for server in available_lower) for pattern in agent.required_mcp_servers)


def filter_agents_by_mcp_requirements(
    agents: Iterable[AgentDefinition],
    available_servers: Iterable[str],
) -> list[AgentDefinition]:
    """Drop agents whose required MCP servers aren't available.

    Built-ins are exempt: they're not allowed to declare requirements and
    must always be reachable.
    """
    available_list = list(available_servers)
    return [agent for agent in agents if is_built_in_agent(agent) or has_required_mcp_servers(agent, available_list)]
