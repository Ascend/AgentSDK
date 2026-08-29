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

# pylint: disable=redefined-builtin
"""Factory — builds a Tool from an AgentToolSpec, then registers it."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from src.tool_system.build_tool import Tool, build_tool
from src.tool_system.context import ToolContext
from clawcodex_ext.tool_system.protocol import ToolResult
from clawcodex_ext.tool_system.protocol import ToolCall
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.agent.tool_authoring.validators import validate_spec
from clawcodex_ext.agent.tool_authoring.call_handlers import (
    BashCallError,
    execute_bash,
    parse_sop_wrapper_stdout,
    HttpCallError,
    execute_http,
    PythonCallError,
    execute_python,
    SdkWrapperCallError,
    execute_sdk_wrapper_in_process,
    parse_sdk_wrapper_cli_options,
    parse_sdk_wrapper_call_impl,
    should_use_in_process_sdk_wrapper,
)

logger = logging.getLogger(__name__)


def _workflow_trace_payload(trace: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "step_id": step.step_id,
            "kind": step.kind,
            "status": step.status,
            "error_code": step.error_code,
            "error": step.error,
        }
        for step in trace
    ]


def _catalog_execution_context(spec: AgentToolSpec, context: ToolContext) -> Any:
    from extensions.sop_converter.resource_catalog import context_from_env

    bundle = context.bundle_context
    if bundle is None:
        try:
            from extensions.sop_converter.bundle_context import get_active_bundle

            bundle = get_active_bundle()
        except ImportError:
            bundle = None
    bundle_path = getattr(bundle, "bundle_path", None)
    bundle_id = str(getattr(bundle, "bundle_name", "") or getattr(bundle, "bundle_id", "") or spec.bundle_id or "")
    if bundle_path is not None:
        from extensions.sop_converter.bundle_venv import activate_bundle_venv_imports

        activate_bundle_venv_imports(bundle_path)
    session_id = getattr(context, "session_id", None) or None
    return context_from_env(
        bundle_path=bundle_path,
        bundle_id=bundle_id,
        session_id=session_id,
    )


def _run_workflow_tool(
    spec: AgentToolSpec,
    call_input: dict[str, Any],
    context: ToolContext,
) -> ToolResult:
    from extensions.sop_converter.composite_runtime import (
        CompositeWorkflowError,
        CompositeWorkflowRunner,
    )
    from extensions.sop_converter.runtime.macros import resolve_macro

    if context.tool_registry is None:
        return ToolResult(
            name=spec.name,
            output={
                "error": "workflow tool has no active ToolRegistry",
                "error_code": "workflow_tool_missing",
            },
            is_error=True,
        )
    stack = context.workflow_stack
    if spec.name in stack:
        return ToolResult(
            name=spec.name,
            output={
                "error": f"workflow cycle detected: {' -> '.join([*stack, spec.name])}",
                "error_code": "workflow_cycle_detected",
            },
            is_error=True,
        )
    if len(stack) >= 8:
        return ToolResult(
            name=spec.name,
            output={
                "error": "workflow nesting depth exceeds 8",
                "error_code": "workflow_cycle_detected",
            },
            is_error=True,
        )
    try:
        catalog_context = _catalog_execution_context(spec, context)
        workflow = resolve_macro(
            dict(spec.call_impl),
            bundle_path=catalog_context.bundle_path,
            session_overlay=getattr(context, "session_macro_overlay", None),
            owner_session_id=getattr(context, "session_id", None),
        )
    except Exception as exc:
        return ToolResult(
            name=spec.name,
            output={
                "error": f"workflow definition not found: {exc}",
                "error_code": "workflow_tool_missing",
            },
            is_error=True,
        )
    tool_names = [step.callable_ref for step in workflow.steps if step.kind == "tool"]
    if tool_names:
        from extensions.sop_converter.bundle_context import (
            ensure_bundle_tools_registered,
        )

        ensure_bundle_tools_registered(
            context.tool_registry,
            tool_names,
            bundle_path=catalog_context.bundle_path,
        )

    def tool_runner(name: str, args: dict[str, Any]) -> Any:
        result = context.tool_registry.dispatch(
            ToolCall(name=name, input=args),
            context,
        )
        if result.is_error:
            error_code = "workflow_step_failed"
            error = str(result.output)
            if isinstance(result.output, dict):
                error_code = str(result.output.get("error_code") or error_code)
                error = str(result.output.get("error") or result.output.get("message") or result.output)
            raise CompositeWorkflowError(error_code, error)
        return result.output

    stack.append(spec.name)
    try:
        result = CompositeWorkflowRunner(tool_runner=tool_runner).run(
            workflow,
            call_input,
            resources={"catalog": catalog_context},
        )
    finally:
        stack.pop()
    trace = _workflow_trace_payload(result.trace)
    if result.is_error:
        failed_step = trace[-1]["step_id"] if trace else ""
        return ToolResult(
            name=spec.name,
            output={
                "error": result.error,
                "error_code": result.error_code or "workflow_step_failed",
                "step_id": failed_step,
                "trace": trace,
            },
            is_error=True,
        )
    output = dict(result.output)
    output.setdefault("trace", trace)
    if spec.output_schema is not None:
        try:
            from clawcodex_ext.tool_system.schema_validation import validate_json_schema

            validate_json_schema(output, spec.output_schema, root_name=spec.name)
        except Exception as exc:
            return ToolResult(
                name=spec.name,
                output={
                    "error": str(exc),
                    "error_code": "workflow_output_schema_mismatch",
                    "trace": trace,
                },
                is_error=True,
            )
    return ToolResult(name=spec.name, output=output)


def _validate_tool_output(spec: AgentToolSpec, output: Any) -> ToolResult | None:
    if spec.output_schema is None:
        return None
    try:
        from clawcodex_ext.tool_system.schema_validation import validate_json_schema

        validate_json_schema(output, spec.output_schema, root_name=f"{spec.name}.output")
    except Exception as exc:
        return ToolResult(
            name=spec.name,
            output={
                "error": str(exc),
                "error_code": "tool_output_schema_mismatch",
            },
            is_error=True,
        )
    return None


# ---------------------------------------------------------------------------
# Tool builder
# ---------------------------------------------------------------------------


def build_tool_from_spec(spec: AgentToolSpec) -> Tool:
    """Construct a ``Tool`` instance from an ``AgentToolSpec``.

    The tool is fully operational: ``tool.call`` dispatches to the appropriate
    ``call_handlers`` entry based on ``spec.call_type``, and all security
    constraints have already been validated by ``validate_spec``.

    Args:
        spec: A validated ``AgentToolSpec``.

    Returns:
        A ``Tool`` object that can be registered with a ``ToolRegistry``.
    """

    def _call_impl(input: dict[str, Any], _context: ToolContext) -> ToolResult:
        call_input = input
        try:
            if should_use_in_process_sdk_wrapper(spec):
                parsed = parse_sdk_wrapper_call_impl(str(spec.call_impl))
                if parsed is None:
                    raise SdkWrapperCallError(f"Cannot parse stateful wrapper call_impl for {spec.name}")
                script_path, method_name = parsed
                output = execute_sdk_wrapper_in_process(
                    script_path=script_path,
                    method_name=method_name,
                    kwargs=call_input,
                    session_id=_context.session_id,
                    agent_id=_context.agent_id,
                    catalog_fallback=parse_sdk_wrapper_cli_options(str(spec.call_impl)).get("--catalog-fallback"),
                )
            elif spec.call_type == "bash":
                # Auto-inject {json_args} for sop-converter bridge tools
                # (and any other tool whose template uses this placeholder).
                # Harmless no-op when the template doesn't contain {json_args}.
                enriched = call_input
                if "{json_args}" in spec.call_impl and "json_args" not in enriched:
                    enriched = {**call_input, "json_args": json.dumps(call_input)}
                # Persisted specs use bare ``python``; on many Linux systems that
                # resolves to Python 2. Always run bridge scripts with the
                # current interpreter (the venv that loaded clawcodex).
                call_impl = spec.call_impl
                if "{json_args}" not in call_impl and (
                    call_impl.startswith("python ") or call_impl.startswith("python3 ")
                ):
                    call_impl = f'"{sys.executable}" {call_impl.split(" ", 1)[1]}'
                output = execute_bash(
                    call_impl,
                    enriched,
                    context=_context,
                    abort_signal=_context.abort_controller.signal,
                )
                if "{json_args}" in spec.call_impl:
                    output = parse_sop_wrapper_stdout(output)
            elif spec.call_type == "http":
                output = execute_http(spec.call_impl, call_input)
            elif spec.call_type == "python":
                output = execute_python(spec.call_impl, call_input)
            elif spec.call_type == "workflow":
                return _run_workflow_tool(spec, call_input, _context)
            else:
                output = f"Unknown call_type: {spec.call_type}"
        except BashCallError as exc:
            parsed = None
            if "{json_args}" in spec.call_impl:
                for raw in (exc.stderr, exc.stdout):
                    if not raw:
                        continue
                    candidate = parse_sop_wrapper_stdout(raw)
                    if isinstance(candidate, dict):
                        parsed = candidate
                        break
            if parsed is not None:
                return ToolResult(
                    name=spec.name,
                    output=parsed,
                    is_error=True,
                )
            if str(exc).startswith("bundle_venv_not_ready:"):
                return ToolResult(
                    name=spec.name,
                    output={
                        "error": str(exc),
                        "error_code": "bundle_venv_not_ready",
                        "recovery": "rerun_sop_convert",
                    },
                    is_error=True,
                )
            return ToolResult(
                name=spec.name,
                output={"error": str(exc)},
                is_error=True,
            )
        except (HttpCallError, PythonCallError, SdkWrapperCallError) as exc:
            return ToolResult(
                name=spec.name,
                output={"error": str(exc)},
                is_error=True,
            )

        if isinstance(output, dict):
            if output.get("error"):
                return ToolResult(
                    name=spec.name,
                    output=output,
                    is_error=True,
                )
            returncode = output.get("returncode")
            if returncode is not None and returncode != 0:
                return ToolResult(
                    name=spec.name,
                    output=output,
                    is_error=True,
                )

        output_error = _validate_tool_output(spec, output)
        if output_error is not None:
            return output_error

        return ToolResult(
            name=spec.name,
            output=output,
        )

    # SOP-converter tools are workflow-specific and numerous;
    # defer them so they load via ToolSearch on demand.
    should_defer = spec.source in {
        "sop-converter",
        "pos-converter",
        "composite-tool",
        "sop-converter-composite",
        "sop-converter-macro",
    }

    return build_tool(
        name=spec.name,
        input_schema=spec.input_schema,
        call=_call_impl,
        prompt=f"Tool: {spec.name}\n\n{spec.description}",
        description=spec.description,
        aliases=spec.aliases,
        search_hint=" ".join(spec.tags) if spec.tags else None,
        should_defer=should_defer,
    )


def create_and_validate(spec: AgentToolSpec) -> Tool:
    """Validate a spec and build a ``Tool`` from it.

    This is the main entry point used by ``CreateAgentTool``.

    Raises:
        ValidationError: If the spec fails validation.
    """
    validate_spec(spec)
    return build_tool_from_spec(spec)
