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

"""Executable P0 runtime for SOP composite workflows."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping


StepKind = Literal["python", "tool", "catalog"]
StepVisibility = Literal["public", "private"]


class CompositeWorkflowError(RuntimeError):
    """A workflow failure with a stable machine-readable error code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class CompositeWorkflowStep:
    """One linear P0 workflow step."""

    id: str
    kind: StepKind
    callable_ref: str
    args: dict[str, Any] = field(default_factory=dict)
    visibility: StepVisibility = "public"
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompositeWorkflowSpec:
    """Declarative composite workflow with simple reference bindings."""

    name: str
    description: str
    inputs: dict[str, dict[str, Any]]
    steps: tuple[CompositeWorkflowStep, ...]
    outputs: dict[str, Any]
    trusted: bool = False


@dataclass(frozen=True)
class StepTrace:
    """Execution status for a single workflow step."""

    step_id: str
    kind: StepKind
    status: Literal["success", "error"]
    error_code: str = ""
    error: str = ""


@dataclass
class CompositeResult:
    """Result and trace returned by :class:`CompositeWorkflowRunner`."""

    output: dict[str, Any] = field(default_factory=dict)
    trace: list[StepTrace] = field(default_factory=list)
    is_error: bool = False
    error_code: str | None = None
    error: str = ""


class CompositeWorkflowRunner:
    """Run linear P0 composite workflows with an explicit callable allowlist."""

    _DEFAULT_PYTHON_CALLABLES = frozenset(
        {
            "extensions.sop_converter.resource_catalog:get_agent_record",
            "extensions.sop_converter.resource_catalog:get_resource_record",
            "extensions.sop_converter.resource_catalog:resolve_agent_record",
            "extensions.sop_converter.resource_catalog:resolve_record",
            "extensions.sop_converter.core.agent_runtime:materialize_agent",
            "extensions.sop_converter.core.agent_runtime:invoke_agent",
            "extensions.sop_converter.resource_runtime:materialize_resource",
            "extensions.sop_converter.resource_runtime:invoke_resource",
        }
    )

    def __init__(
        self,
        *,
        python_callables: Mapping[str, Callable[..., Any]] | None = None,
        tool_runner: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self._python_callables = dict(python_callables or {})
        self._tool_runner = tool_runner

    def run(
        self,
        spec: CompositeWorkflowSpec,
        inputs: Mapping[str, Any],
        *,
        resources: Mapping[str, Any] | None = None,
    ) -> CompositeResult:
        input_values = {name: inputs.get(name) for name in spec.inputs}
        input_values.update(dict(inputs))
        context: dict[str, Any] = {
            "input": input_values,
            "steps": {},
            "private": {},
            "resources": dict(resources or {}),
        }
        missing = [name for name, schema in spec.inputs.items() if schema.get("required") and inputs.get(name) is None]
        if missing:
            return CompositeResult(
                is_error=True,
                error_code="workflow_binding_missing",
                error=f"missing required workflow input: {missing[0]}",
            )

        trace: list[StepTrace] = []
        for step in spec.steps:
            try:
                if step.kind != "tool" and not spec.trusted:
                    raise CompositeWorkflowError(
                        "workflow_step_failed",
                        f"untrusted workflow cannot execute {step.kind} step {step.id!r}",
                    )
                if step.visibility == "private" and not spec.trusted:
                    raise CompositeWorkflowError(
                        "workflow_step_failed",
                        f"untrusted workflow cannot use private step {step.id!r}",
                    )
                resolved_args = self._resolve_value(
                    step.args,
                    context,
                    allow_private=spec.trusted,
                )
                result = self._run_step(step, resolved_args)
                if step.visibility == "private":
                    context["private"][step.id] = {"output": result}
                else:
                    normalized = normalize_workflow_output(result)
                    self._validate_output_schema(step, normalized)
                    context["steps"][step.id] = {"output": normalized}
            except Exception as exc:
                code = getattr(exc, "error_code", "workflow_step_failed")
                trace.append(
                    StepTrace(
                        step_id=step.id,
                        kind=step.kind,
                        status="error",
                        error_code=str(code),
                        error=str(exc),
                    )
                )
                return CompositeResult(
                    trace=trace,
                    is_error=True,
                    error_code=str(code),
                    error=str(exc),
                )
            trace.append(StepTrace(step_id=step.id, kind=step.kind, status="success"))

        try:
            output = self._resolve_value(
                spec.outputs,
                context,
                allow_private=False,
            )
            output = _json_safe_value(output)
        except CompositeWorkflowError as exc:
            return CompositeResult(
                trace=trace,
                is_error=True,
                error_code=exc.error_code,
                error=str(exc),
            )
        return CompositeResult(output=output, trace=trace)

    def _validate_output_schema(
        self,
        step: CompositeWorkflowStep,
        output: dict[str, Any],
    ) -> None:
        if step.output_schema is None:
            return
        try:
            from clawcodex_ext.tool_system.schema_validation import validate_json_schema

            validate_json_schema(
                output,
                step.output_schema,
                root_name=f"workflow.{step.id}.output",
            )
        except Exception as exc:
            raise CompositeWorkflowError(
                "workflow_output_schema_mismatch",
                str(exc),
            ) from exc

    def _run_step(self, step: CompositeWorkflowStep, args: dict[str, Any]) -> Any:
        if step.kind == "tool":
            if self._tool_runner is None:
                raise CompositeWorkflowError(
                    "workflow_step_failed",
                    f"tool step {step.id!r} has no tool runner",
                )
            return self._tool_runner(step.callable_ref, args)
        if step.kind not in {"python", "catalog"}:
            raise CompositeWorkflowError(
                "workflow_step_failed",
                f"unsupported step kind: {step.kind}",
            )
        return self._resolve_python_callable(step.callable_ref)(**args)

    def _resolve_python_callable(self, reference: str) -> Callable[..., Any]:
        registered = self._python_callables.get(reference)
        if registered is not None:
            return registered
        if reference not in self._DEFAULT_PYTHON_CALLABLES:
            raise CompositeWorkflowError(
                "workflow_step_failed",
                f"python callable is not allowlisted: {reference}",
            )
        try:
            module_name, attr_name = reference.split(":", 1)
            func = getattr(importlib.import_module(module_name), attr_name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise CompositeWorkflowError(
                "workflow_step_failed",
                f"unable to resolve python callable {reference}: {exc}",
            ) from exc
        if not callable(func):
            raise CompositeWorkflowError(
                "workflow_step_failed",
                f"python callable is not callable: {reference}",
            )
        return func

    def _resolve_value(
        self,
        value: Any,
        context: Mapping[str, Any],
        *,
        allow_private: bool,
    ) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            return self._resolve_reference(value, context, allow_private=allow_private)
        if isinstance(value, dict):
            return {key: self._resolve_value(item, context, allow_private=allow_private) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(item, context, allow_private=allow_private) for item in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_value(item, context, allow_private=allow_private) for item in value)
        return value

    def _resolve_reference(
        self,
        reference: str,
        context: Mapping[str, Any],
        *,
        allow_private: bool,
    ) -> Any:
        segments = [segment for segment in reference[1:].split(".") if segment]
        allowed_roots = {"input", "steps", "resources"}
        if allow_private:
            allowed_roots.add("private")
        if not segments or segments[0] not in allowed_roots:
            raise CompositeWorkflowError(
                "workflow_binding_missing",
                f"unsupported workflow binding: {reference}",
            )
        current: Any = context
        for segment in segments:
            if isinstance(current, Mapping):
                if segment not in current:
                    raise CompositeWorkflowError(
                        "workflow_binding_missing",
                        f"workflow binding not found: {reference}",
                    )
                current = current[segment]
                continue
            if isinstance(current, (list, tuple)) and segment.isdigit():
                index = int(segment)
                if index >= len(current):
                    raise CompositeWorkflowError(
                        "workflow_binding_missing",
                        f"workflow binding not found: {reference}",
                    )
                current = current[index]
                continue
            if not segment.startswith("_") and hasattr(current, segment):
                current = getattr(current, segment)
                continue
            raise CompositeWorkflowError(
                "workflow_binding_missing",
                f"workflow binding not found: {reference}",
            )
        return current


def _json_safe_value(value: Any) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CompositeWorkflowError(
            "workflow_output_unserializable",
            f"workflow output is not JSON serializable: {exc}",
        ) from exc


def normalize_workflow_output(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        normalized = _json_safe_value(dict(value))
        if not isinstance(normalized, dict):
            raise CompositeWorkflowError(
                "workflow_output_unserializable",
                "mapping output did not normalize to an object",
            )
        return normalized
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return _json_safe_value(parsed)
        return {"text": value, "value": value}
    return {"value": _json_safe_value(value)}


__all__ = [
    "CompositeResult",
    "CompositeWorkflowError",
    "CompositeWorkflowRunner",
    "CompositeWorkflowSpec",
    "CompositeWorkflowStep",
    "StepVisibility",
    "StepTrace",
    "normalize_workflow_output",
]
