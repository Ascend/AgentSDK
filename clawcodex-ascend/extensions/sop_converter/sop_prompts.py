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
# Backward-compatibility stub — re-exports from core/sop_prompts.py
from extensions.sop_converter.core.sop_prompts import (
    SOP_INTERACTIVE_TERMINAL_STOP_LOSS,
    SOP_SOURCE_EXPLORATION_POLICY,
    SOP_NO_SOURCE_EXPLORATION,
    SOP_TOOL_FAILURE_RECOVERY,
    SOP_TOOLSEARCH_GUIDANCE,
    SOP_OVERVIEW_ROUTING,
    format_sdk_source_dir_block,
    agent_type_to_skill_name,
    pick_pipeline_execute_tool,
    infer_stage_label_from_skill,
    stage_agent_sop_body,
    domain_agent_sop_body,
    format_overview_stage_pipeline_block,
    append_sop_overview_routing,
)

__all__ = [
    "SOP_INTERACTIVE_TERMINAL_STOP_LOSS",
    "SOP_SOURCE_EXPLORATION_POLICY",
    "SOP_NO_SOURCE_EXPLORATION",
    "SOP_TOOL_FAILURE_RECOVERY",
    "SOP_TOOLSEARCH_GUIDANCE",
    "SOP_OVERVIEW_ROUTING",
    "format_sdk_source_dir_block",
    "agent_type_to_skill_name",
    "pick_pipeline_execute_tool",
    "infer_stage_label_from_skill",
    "stage_agent_sop_body",
    "domain_agent_sop_body",
    "format_overview_stage_pipeline_block",
    "append_sop_overview_routing",
]

from extensions.sop_converter.core import sop_prompts as _impl

for _name, _value in vars(_impl).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value, _impl
