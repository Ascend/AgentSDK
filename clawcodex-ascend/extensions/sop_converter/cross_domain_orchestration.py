#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

# pylint: disable=undefined-loop-variable
# Backward-compatibility stub — re-exports from runtime/cross_domain_orchestration.py
from extensions.sop_converter.runtime.cross_domain_orchestration import (
    ORCHESTRATION_ROUTES_NAME,
    ORCHESTRATION_ROUTES_MAX,
    OrchestrationStep,
    OrchestrationRoute,
    skill_name_to_agent,
    build_tool_to_agent_map,
    discover_orchestration_routes,
    generate_orchestration_routes_markdown,
    write_orchestration_routes,
    format_orchestration_routes_block,
)

__all__ = [
    "ORCHESTRATION_ROUTES_NAME",
    "ORCHESTRATION_ROUTES_MAX",
    "OrchestrationStep",
    "OrchestrationRoute",
    "skill_name_to_agent",
    "build_tool_to_agent_map",
    "discover_orchestration_routes",
    "generate_orchestration_routes_markdown",
    "write_orchestration_routes",
    "format_orchestration_routes_block",
]

from extensions.sop_converter.runtime import cross_domain_orchestration as _impl

for _name, _value in vars(_impl).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value, _impl
