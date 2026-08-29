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

"""register-macro-workflow — interactive session-macro registration tool.

Calls :func:`register_session_macro` with a capability-gated confirm path.
``tool_index`` is the active bundle allowlist ∪ base tools ∪ non-session
names already on ``options.tools`` — never the full global registry.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from extensions.capabilities.agent_definition_protocol import AgentToolConstants
from clawcodex_ext.agent.tool_authoring.factory import create_and_validate
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.build_tool import Tool, build_tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.protocol import ToolResult

from .errors import MacroConvertError
from .session import (
    SessionMacroPlan,
    is_session_macro_tool,
    mark_session_macro_tool,
    register_session_macro,
)
from .validation import _FORBIDDEN_STEP_TOOLS

REGISTER_MACRO_WORKFLOW_TOOL_NAME = "register-macro-workflow"

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "definition": {
            "type": "object",
            "description": (
                "Session macro definition (scope=session). Same shape as a "
                "handwritten macro: name, description, workflow, routing, provenance."
            ),
        },
        "replace": {
            "type": "boolean",
            "description": (
                "When true, replace an existing session macro of the same name. "
                "Default false — colliding names are rejected."
            ),
            "default": False,
        },
    },
    "required": ["definition"],
}


def format_session_macro_plan_for_ui(plan: SessionMacroPlan) -> str:
    """Render a registration plan for TUI modal / REPL confirm prompts.

    Shows action/name plus each step's id, tool, and args template.
    """
    lines = [
        f"Action: {plan.action}",
        f"Name: {plan.name}",
        f"Description: {plan.description}",
        f"Catalog: {plan.catalog_id}",
    ]
    if getattr(plan, "target_path", ""):
        lines.append(f"Target path: {plan.target_path}")
    lines.append("Steps:")
    for step in plan.steps:
        args = dict(step.args_template)
        try:
            args_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            args_text = str(args)
        lines.append(f"  - id={step.step_id} tool={step.tool} args={args_text}")
    route = dict(plan.route_summary)
    if route:
        lines.append(f"Route: {json.dumps(route, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines)


def build_session_macro_tool_index(context: ToolContext) -> set[str]:
    """Bundle allowlist ∪ base tools ∪ non-session names on options.tools."""
    names: set[str] = set(AgentToolConstants.registered_proxy_base_tools())

    bundle = getattr(context, "bundle_context", None)
    if bundle is None:
        try:
            from extensions.sop_converter.bundle_context import get_active_bundle

            bundle = get_active_bundle()
        except Exception:
            bundle = None
    if bundle is not None:
        tool_names = getattr(bundle, "tool_names", None) or ()
        names.update(str(n) for n in tool_names)

    options = getattr(context, "options", None)
    for tool in list(getattr(options, "tools", None) or []):
        if is_session_macro_tool(tool):
            continue
        name = getattr(tool, "name", None)
        if name:
            names.add(str(name))
    return names


def collect_workflow_tool_names(context: ToolContext) -> set[str]:
    """Names forbidden as nested workflow/session-macro steps."""
    names: set[str] = set(_FORBIDDEN_STEP_TOOLS)

    overlay = getattr(context, "session_macro_overlay", None)
    if overlay is not None:
        snapshot = overlay.read()
        if snapshot is not None:
            names.update(str(k) for k in snapshot.tools.keys())
            names.update(str(k) for k in snapshot.definitions.keys())

    options = getattr(context, "options", None)
    for tool in list(getattr(options, "tools", None) or []):
        if is_session_macro_tool(tool):
            name = getattr(tool, "name", None)
            if name:
                names.add(str(name))
            continue
        # Deferred / authored workflow tools expose call_type via source tags.
        source = getattr(tool, "source", None) or ""
        tags = getattr(tool, "tags", None) or ()
        if source in {"sop-converter-macro", "composite-tool"} or "workflow" in tags:
            name = getattr(tool, "name", None)
            if name:
                names.add(str(name))
    return names


def collect_protected_builtin_exclusive_targets() -> set[str]:
    """Verified builtin exclusive route targets (e.g. invoke-existing-agent)."""
    from .routing import DEFAULT_MACRO_ROUTE_CATALOG, ensure_builtin_routes

    ensure_builtin_routes(DEFAULT_MACRO_ROUTE_CATALOG)
    return {
        route.target_tool
        for route in DEFAULT_MACRO_ROUTE_CATALOG.get_routes()
        if route.scope == "builtin" and route.selection == "exclusive" and route.verified
    }


def _create_session_macro_tool(spec: AgentToolSpec) -> Tool:
    return mark_session_macro_tool(create_and_validate(spec))


def _register_macro_workflow_call(
    input_data: dict[str, Any],
    context: ToolContext,
) -> ToolResult:
    definition = input_data.get("definition")
    if not isinstance(definition, Mapping):
        return ToolResult(
            name=REGISTER_MACRO_WORKFLOW_TOOL_NAME,
            output={
                "error_code": "macro_schema_invalid",
                "message": "definition must be an object",
            },
            is_error=True,
        )
    replace = bool(input_data.get("replace", False))
    try:
        payload = register_session_macro(
            context,
            definition,
            replace=replace,
            tool_index=build_session_macro_tool_index(context),
            workflow_tool_names=collect_workflow_tool_names(context),
            protected_builtin_exclusive_targets=collect_protected_builtin_exclusive_targets(),
            create_tool=_create_session_macro_tool,
        )
    except MacroConvertError as exc:
        return ToolResult(
            name=REGISTER_MACRO_WORKFLOW_TOOL_NAME,
            output=exc.to_dict(),
            is_error=True,
        )
    return ToolResult(
        name=REGISTER_MACRO_WORKFLOW_TOOL_NAME,
        output=payload,
        is_error=False,
    )


RegisterMacroWorkflowTool: Tool = build_tool(
    name=REGISTER_MACRO_WORKFLOW_TOOL_NAME,
    input_schema=_INPUT_SCHEMA,
    call=_register_macro_workflow_call,
    prompt=(
        "Register a session-scoped composite workflow macro for the current "
        "conversation. Requires interactive user confirmation. Does not persist "
        "across sessions. Pass replace=true only when intentionally overwriting "
        "an existing session macro of the same name."
    ),
    description=lambda _input: "Register a session-scoped macro workflow",
    aliases=("RegisterMacroWorkflow",),
    search_hint="register session macro workflow composite",
    max_result_size_chars=20_000,
    is_read_only=lambda _input: False,
    is_destructive=lambda _input: False,
    is_concurrency_safe=lambda _input: False,
)


REGISTER_MACRO_FROM_TRACE_TOOL_NAME = "register-macro-from-trace"

_FROM_TRACE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {
            "type": "string",
            "description": "Kebab-case session macro name to register.",
        },
        "description": {
            "type": "string",
            "description": "Optional human description for the macro.",
        },
        "phrases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional ToolSearch recall phrases.",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional ToolSearch keywords.",
        },
        "max_steps": {
            "type": "integer",
            "description": "Max successful trailing tool steps to include (default 16).",
            "default": 16,
            "minimum": 1,
            "maximum": 16,
        },
        "replace": {
            "type": "boolean",
            "description": "Replace an existing session macro of the same name.",
            "default": False,
        },
    },
    "required": ["name"],
}


def _register_macro_from_trace_call(
    input_data: dict[str, Any],
    context: ToolContext,
) -> ToolResult:
    from .trace import extract_successful_tool_steps, trace_steps_to_definition_dict

    if not getattr(context, "allow_session_macro_registration", False):
        return ToolResult(
            name=REGISTER_MACRO_FROM_TRACE_TOOL_NAME,
            output={
                "error_code": "macro_capability_denied",
                "message": "session macro registration is not allowed",
            },
            is_error=True,
        )

    name = str(input_data.get("name") or "").strip()
    description = str(input_data.get("description") or "")
    phrases = input_data.get("phrases")
    keywords = input_data.get("keywords")
    try:
        max_steps = int(input_data.get("max_steps") or 16)
    except (TypeError, ValueError):
        max_steps = 16
    replace = bool(input_data.get("replace", False))

    try:
        steps = extract_successful_tool_steps(
            getattr(context, "messages", None) or [],
            max_steps=max_steps,
        )
        definition = trace_steps_to_definition_dict(
            steps,
            name=name,
            description=description,
            phrases=phrases if isinstance(phrases, list) else None,
            keywords=keywords if isinstance(keywords, list) else None,
        )
        payload = register_session_macro(
            context,
            definition,
            replace=replace,
            tool_index=build_session_macro_tool_index(context),
            workflow_tool_names=collect_workflow_tool_names(context),
            protected_builtin_exclusive_targets=collect_protected_builtin_exclusive_targets(),
            create_tool=_create_session_macro_tool,
        )
        payload["source"] = "session_trace"
        payload["trace_step_count"] = len(steps)
    except MacroConvertError as exc:
        return ToolResult(
            name=REGISTER_MACRO_FROM_TRACE_TOOL_NAME,
            output=exc.to_dict(),
            is_error=True,
        )
    except Exception as exc:
        return ToolResult(
            name=REGISTER_MACRO_FROM_TRACE_TOOL_NAME,
            output={
                "error_code": "macro_trace_failed",
                "message": str(exc),
            },
            is_error=True,
        )
    return ToolResult(
        name=REGISTER_MACRO_FROM_TRACE_TOOL_NAME,
        output=payload,
        is_error=False,
    )


RegisterMacroFromTraceTool: Tool = build_tool(
    name=REGISTER_MACRO_FROM_TRACE_TOOL_NAME,
    input_schema=_FROM_TRACE_SCHEMA,
    call=_register_macro_from_trace_call,
    prompt=(
        "Register a session macro from the latest successful tool-call trace "
        "in this conversation ('remember those steps as a macro'). Requires "
        "interactive confirmation. Does not write to the bundle."
    ),
    description=lambda _input: "Register a session macro from recent tool trace",
    aliases=("RegisterMacroFromTrace",),
    search_hint="register macro from trace session steps",
    max_result_size_chars=20_000,
    is_read_only=lambda _input: False,
    is_destructive=lambda _input: False,
    is_concurrency_safe=lambda _input: False,
)


PROMOTE_MACRO_WORKFLOW_TOOL_NAME = "promote-macro-workflow"

_PROMOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of an already-registered session macro to promote.",
        },
        "replace": {
            "type": "boolean",
            "description": "Overwrite an existing bundle macro of the same name.",
            "default": False,
        },
    },
    "required": ["name"],
}


def _promote_macro_workflow_call(
    input_data: dict[str, Any],
    context: ToolContext,
) -> ToolResult:
    from .promote import promote_session_macro_to_bundle

    if not getattr(context, "allow_session_macro_registration", False):
        return ToolResult(
            name=PROMOTE_MACRO_WORKFLOW_TOOL_NAME,
            output={
                "error_code": "macro_capability_denied",
                "message": "session macro promote is not allowed",
            },
            is_error=True,
        )

    name = str(input_data.get("name") or "").strip()
    replace = bool(input_data.get("replace", False))
    try:
        payload = promote_session_macro_to_bundle(
            context,
            name,
            replace=replace,
            tool_index=build_session_macro_tool_index(context),
        )
    except MacroConvertError as exc:
        return ToolResult(
            name=PROMOTE_MACRO_WORKFLOW_TOOL_NAME,
            output=exc.to_dict(),
            is_error=True,
        )
    return ToolResult(
        name=PROMOTE_MACRO_WORKFLOW_TOOL_NAME,
        output=payload,
        is_error=False,
    )


PromoteMacroWorkflowTool: Tool = build_tool(
    name=PROMOTE_MACRO_WORKFLOW_TOOL_NAME,
    input_schema=_PROMOTE_SCHEMA,
    call=_promote_macro_workflow_call,
    prompt=(
        "Promote a registered session macro into the active bundle "
        "(.clawcodex/macros/). Requires a second interactive confirmation. "
        "Keeps the session overlay entry for the current conversation."
    ),
    description=lambda _input: "Promote a session macro into the bundle",
    aliases=("PromoteMacroWorkflow",),
    search_hint="promote session macro to bundle persist",
    max_result_size_chars=20_000,
    is_read_only=lambda _input: False,
    is_destructive=lambda _input: False,
    is_concurrency_safe=lambda _input: False,
)


__all__ = [
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
]
