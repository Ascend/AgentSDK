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

"""Small in-process catalog for executable workflow definitions and routes."""

from .catalog import (
    DEFAULT_MACRO_CATALOG,
    MacroCatalog,
    ensure_builtin_macros,
    register_macro,
    resolve_macro,
)
from .convert import MacroConvertResult, convert_handwritten_macros
from .errors import MacroConvertError
from .loader import discover_macro_sources, load_macro_definitions, load_macro_yaml
from .models import MacroDefinition, MacroRoute
from .overview_intent import (
    assign_macros_to_owner_skills,
    format_overview_macro_intent_block,
    pick_owner_skill,
    resolve_macro_delegate_agent,
)
from .routing import (
    DEFAULT_MACRO_ROUTE_CATALOG,
    MacroRouteCatalog,
    ensure_builtin_routes,
    match_macro_routes,
    register_macro_route,
    resolve_macro_route,
    resolve_macro_route_details,
)
from .compiler import MacroDraft, compile_macro_definition
from .promote import promote_session_macro_to_bundle
from .register_tool import (
    PROMOTE_MACRO_WORKFLOW_TOOL_NAME,
    REGISTER_MACRO_FROM_TRACE_TOOL_NAME,
    REGISTER_MACRO_WORKFLOW_TOOL_NAME,
    PromoteMacroWorkflowTool,
    RegisterMacroFromTraceTool,
    RegisterMacroWorkflowTool,
    build_session_macro_tool_index,
    collect_protected_builtin_exclusive_targets,
    collect_workflow_tool_names,
    format_session_macro_plan_for_ui,
)
from .resolve_tool import resolve_tool_for_context
from .session import (
    SessionMacroOverlay,
    SessionMacroPlan,
    SessionMacroPlanStep,
    SessionMacroSnapshot,
    clear_session_macros_for_context,
    is_session_macro_tool,
    iter_effective_tools,
    mark_session_macro_tool,
    register_session_macro,
    sync_effective_tools,
)
from .session_parse import parse_session_macro_definition, parse_session_macro_route
from extensions.sop_converter.runtime.macros.templates import HANDWRITTEN_MACRO_TEMPLATE, TEMPLATES_DIR
from .trace import (
    TraceToolStep,
    extract_successful_tool_steps,
    trace_steps_to_definition_dict,
)
from ...core.tool_retrieval import (
    MacroCoverage,
    ToolRetrievalIndex,
    ToolRetrievalProfile,
    load_tool_retrieval_index,
    write_tool_retrieval_index,
)

__all__ = [
    "DEFAULT_MACRO_CATALOG",
    "MacroCatalog",
    "ensure_builtin_macros",
    "register_macro",
    "resolve_macro",
    "MacroDefinition",
    "MacroRoute",
    "DEFAULT_MACRO_ROUTE_CATALOG",
    "MacroRouteCatalog",
    "ensure_builtin_routes",
    "match_macro_routes",
    "register_macro_route",
    "resolve_macro_route",
    "resolve_macro_route_details",
    "MacroConvertError",
    "MacroConvertResult",
    "convert_handwritten_macros",
    "discover_macro_sources",
    "load_macro_definitions",
    "load_macro_yaml",
    "format_overview_macro_intent_block",
    "resolve_macro_delegate_agent",
    "pick_owner_skill",
    "assign_macros_to_owner_skills",
    "MacroCoverage",
    "ToolRetrievalIndex",
    "ToolRetrievalProfile",
    "load_tool_retrieval_index",
    "write_tool_retrieval_index",
    "HANDWRITTEN_MACRO_TEMPLATE",
    "TEMPLATES_DIR",
    "MacroDraft",
    "compile_macro_definition",
    "promote_session_macro_to_bundle",
    "REGISTER_MACRO_WORKFLOW_TOOL_NAME",
    "REGISTER_MACRO_FROM_TRACE_TOOL_NAME",
    "PROMOTE_MACRO_WORKFLOW_TOOL_NAME",
    "RegisterMacroWorkflowTool",
    "RegisterMacroFromTraceTool",
    "PromoteMacroWorkflowTool",
    "build_session_macro_tool_index",
    "collect_protected_builtin_exclusive_targets",
    "collect_workflow_tool_names",
    "format_session_macro_plan_for_ui",
    "resolve_tool_for_context",
    "SessionMacroOverlay",
    "SessionMacroPlan",
    "SessionMacroPlanStep",
    "SessionMacroSnapshot",
    "clear_session_macros_for_context",
    "is_session_macro_tool",
    "iter_effective_tools",
    "mark_session_macro_tool",
    "register_session_macro",
    "sync_effective_tools",
    "parse_session_macro_definition",
    "parse_session_macro_route",
    "TraceToolStep",
    "extract_successful_tool_steps",
    "trace_steps_to_definition_dict",
]
