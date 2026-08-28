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

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.


"""Build frontmatter tool lists from capability profiles."""

from __future__ import annotations

from extensions.capabilities.agent_definition_protocol import AgentToolConstants

from ..capability.models import ExecutionMode, StageCapabilityProfile


def stage_agent_tool_names(sdk_tools: list[str]) -> list[str]:
    """SOP stage agents need Skill/ToolSearch plus deferred SDK tools."""
    return sorted(set(AgentToolConstants.POS_SOP_DOMAIN_AGENT_TOOLS) | set(sdk_tools))


def tools_for_profile(
    profile: StageCapabilityProfile,
    *,
    bridge_tool: str | None = None,
) -> list[str]:
    tools = list(profile.recommended_tools)
    if profile.execution_mode in (ExecutionMode.WRAPPER, ExecutionMode.HYBRID) and bridge_tool:
        if bridge_tool not in tools:
            tools.append(bridge_tool)
    return tools
