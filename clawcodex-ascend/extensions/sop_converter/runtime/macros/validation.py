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

"""Validate MacroDefinition before convert persist/register / session overlay."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec

from ..composite_runtime import CompositeWorkflowSpec, CompositeWorkflowStep
from .errors import MacroConvertError
from .models import MacroDefinition

_INPUT_REF = re.compile(r"^\$input\.([A-Za-z_][\w]*)$")
_STEP_OUTPUT_REF = re.compile(r"^\$steps\.([A-Za-z_][\w]*)\.output(?:\.([A-Za-z_][\w]*))?$")
_FORBIDDEN_STEP_TOOLS = frozenset(
    {
        "register-macro-workflow",
        "RegisterMacroWorkflow",
        "register-macro-from-trace",
        "RegisterMacroFromTrace",
        "promote-macro-workflow",
        "PromoteMacroWorkflow",
    }
)
_KEBAB_NAME = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


@dataclass(frozen=True)
class ValidatedSessionMacro:
    """Normalized session macro ready for plan/confirm/overlay commit."""

    definition: MacroDefinition
    workflow: CompositeWorkflowSpec
    tool_spec: AgentToolSpec


def workflow_dict_to_spec(macro: MacroDefinition) -> CompositeWorkflowSpec:
    """Convert MacroDefinition.workflow mapping into CompositeWorkflowSpec."""
    workflow = macro.workflow
    inputs = workflow.get("inputs") or {}
    if not isinstance(inputs, dict):
        raise MacroConvertError(
            "macro_schema_invalid",
            "workflow.inputs must be a mapping",
            manifest=str(macro.provenance.get("manifest") or macro.name),
            field="workflow.inputs",
        )
    steps_raw = workflow.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise MacroConvertError(
            "macro_schema_invalid",
            "workflow.steps must be a non-empty list",
            manifest=str(macro.provenance.get("manifest") or macro.name),
            field="workflow.steps",
        )
    steps: list[CompositeWorkflowStep] = []
    for index, raw in enumerate(steps_raw):
        if not isinstance(raw, dict):
            raise MacroConvertError(
                "macro_schema_invalid",
                f"workflow.steps[{index}] must be a mapping",
                manifest=str(macro.provenance.get("manifest") or macro.name),
                field=f"workflow.steps[{index}]",
            )
        step_id = str(raw.get("id") or "").strip()
        if not step_id:
            raise MacroConvertError(
                "macro_schema_invalid",
                "step id is required",
                manifest=str(macro.provenance.get("manifest") or macro.name),
                field=f"workflow.steps[{index}].id",
            )
        kind = str(raw.get("kind") or "tool")
        if kind != "tool":
            raise MacroConvertError(
                "macro_step_kind_forbidden",
                "bundle macros may only contain kind=tool steps",
                manifest=str(macro.provenance.get("manifest") or macro.name),
                step_id=step_id,
                field="kind",
            )
        callable_ref = str(raw.get("callable_ref") or "").strip()
        if not callable_ref:
            raise MacroConvertError(
                "macro_callable_missing",
                "callable_ref is required",
                manifest=str(macro.provenance.get("manifest") or macro.name),
                step_id=step_id,
                field="callable_ref",
            )
        args = raw.get("args") or {}
        if not isinstance(args, dict):
            raise MacroConvertError(
                "macro_schema_invalid",
                "step args must be a mapping",
                manifest=str(macro.provenance.get("manifest") or macro.name),
                step_id=step_id,
                field="args",
            )
        output_schema = raw.get("output_schema")
        if output_schema is not None and not isinstance(output_schema, dict):
            raise MacroConvertError(
                "macro_schema_invalid",
                "output_schema must be a mapping when present",
                manifest=str(macro.provenance.get("manifest") or macro.name),
                step_id=step_id,
                field="output_schema",
            )
        steps.append(
            CompositeWorkflowStep(
                id=step_id,
                kind="tool",
                callable_ref=callable_ref,
                args=dict(args),
                visibility="public",
                output_schema=output_schema,
            )
        )
    outputs = workflow.get("outputs") or {}
    if not isinstance(outputs, dict):
        raise MacroConvertError(
            "macro_schema_invalid",
            "workflow.outputs must be a mapping",
            manifest=str(macro.provenance.get("manifest") or macro.name),
            field="workflow.outputs",
        )
    normalized_inputs: dict[str, dict[str, Any]] = {}
    for key, schema in inputs.items():
        if isinstance(schema, dict):
            normalized_inputs[str(key)] = dict(schema)
        else:
            normalized_inputs[str(key)] = {"type": "string", "required": False}
    return CompositeWorkflowSpec(
        name=macro.name,
        description=macro.description or macro.name,
        inputs=normalized_inputs,
        steps=tuple(steps),
        outputs=dict(outputs),
        trusted=False,
    )


def _collect_binding_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, str) and value.startswith("$"):
        refs.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            refs.extend(_collect_binding_refs(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            refs.extend(_collect_binding_refs(child))
    return refs


def _workflow_input_schema(spec: CompositeWorkflowSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, schema in spec.inputs.items():
        prop = dict(schema)
        prop.setdefault("type", "string")
        is_required = bool(prop.pop("required", False))
        properties[name] = prop
        if is_required:
            required.append(name)
    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    return out


def validate_macro_core(
    macro: MacroDefinition,
    *,
    tool_index: Iterable[str] | None = None,
    forbid_workflow_tools: Iterable[str] | None = None,
    max_steps: int = 16,
) -> CompositeWorkflowSpec:
    """Shared structural validation for bundle and session macros.

    ``tool_index`` is allowlist-shaped: current bundle/agent allowlist ∪
    explicit base tools ∪ non-session names already on ``options.tools`` —
    **not** the full global registry.
    """
    manifest = str(macro.provenance.get("manifest") or macro.name)
    spec = workflow_dict_to_spec(macro)
    if len(spec.steps) > max_steps:
        raise MacroConvertError(
            "macro_step_limit",
            f"macro exceeds default max of {max_steps} steps",
            manifest=manifest,
        )

    seen_ids: set[str] = set()
    available = {str(name) for name in (tool_index or [])}
    forbidden = set(_FORBIDDEN_STEP_TOOLS)
    if forbid_workflow_tools is not None:
        forbidden.update(str(name) for name in forbid_workflow_tools)

    for step in spec.steps:
        if step.id in seen_ids:
            raise MacroConvertError(
                "macro_step_duplicate",
                f"duplicate step id: {step.id}",
                manifest=manifest,
                step_id=step.id,
            )
        seen_ids.add(step.id)
        if step.callable_ref in forbidden:
            raise MacroConvertError(
                "macro_step_forbidden",
                f"step may not call forbidden tool: {step.callable_ref}",
                manifest=manifest,
                step_id=step.id,
                field="callable_ref",
            )
        if available and step.callable_ref not in available:
            raise MacroConvertError(
                "macro_callable_unresolved",
                f"callable_ref not in tool index: {step.callable_ref}",
                manifest=manifest,
                step_id=step.id,
                field="callable_ref",
            )

    # Forward-only binding checks
    completed: set[str] = set()
    input_names = set(spec.inputs)
    for step in spec.steps:
        for ref in _collect_binding_refs(step.args):
            input_match = _INPUT_REF.match(ref)
            if input_match:
                name = input_match.group(1)
                if name not in input_names:
                    raise MacroConvertError(
                        "macro_binding_invalid",
                        f"unknown input binding: {ref}",
                        manifest=manifest,
                        step_id=step.id,
                        field="args",
                    )
                continue
            step_match = _STEP_OUTPUT_REF.match(ref)
            if step_match:
                prior = step_match.group(1)
                if prior not in completed:
                    raise MacroConvertError(
                        "macro_binding_forward",
                        f"binding references unfinished step: {ref}",
                        manifest=manifest,
                        step_id=step.id,
                        field="args",
                    )
                continue
            if ref.startswith("$private."):
                raise MacroConvertError(
                    "macro_private_forbidden",
                    "macros cannot use private bindings",
                    manifest=manifest,
                    step_id=step.id,
                    field="args",
                )
            if ref.startswith("$"):
                # Allow $resources.* without deep validation in Phase 4 MVP
                if ref.startswith("$resources."):
                    continue
                raise MacroConvertError(
                    "macro_binding_invalid",
                    f"unsupported binding: {ref}",
                    manifest=manifest,
                    step_id=step.id,
                    field="args",
                )
        completed.add(step.id)

    for ref in _collect_binding_refs(spec.outputs):
        if _INPUT_REF.match(ref):
            continue
        step_match = _STEP_OUTPUT_REF.match(ref)
        if step_match and step_match.group(1) in completed:
            continue
        if ref.startswith("$resources."):
            continue
        raise MacroConvertError(
            "macro_binding_invalid",
            f"invalid output binding: {ref}",
            manifest=manifest,
            field="workflow.outputs",
        )

    return spec


def _validate_covered_tools(
    macro: MacroDefinition,
    *,
    tool_index: Iterable[str] | None,
    manifest: str,
) -> None:
    if not macro.routing.covered_tools or tool_index is None:
        return
    from ...core.tool_retrieval import resolve_tool_references

    try:
        covered = resolve_tool_references(
            macro.routing.covered_tools,
            tool_index,
            require_unique=True,
        )
    except ValueError as exc:
        raise MacroConvertError(
            "macro_coverage_unresolved",
            str(exc),
            manifest=manifest,
            field="routing.covered_tools",
        ) from exc
    target_norm = macro.routing.target_tool.replace("_", "-").replace(".", "-").lower()
    if any(name.replace("_", "-").replace(".", "-").lower() == target_norm for name in covered):
        raise MacroConvertError(
            "macro_coverage_self_reference",
            "macro may not cover itself",
            manifest=manifest,
            field="routing.covered_tools",
        )


def validate_macro_definition(
    macro: MacroDefinition,
    *,
    tool_index: Iterable[str] | None = None,
) -> CompositeWorkflowSpec:
    """Validate and normalize a bundle MacroDefinition; return workflow IR.

    Side effect: unverified ``selection=exclusive`` is downgraded to ``prefer``.
    """
    manifest = str(macro.provenance.get("manifest") or macro.name)
    if not macro.enabled:
        raise MacroConvertError(
            "macro_disabled",
            f"macro is disabled: {macro.name}",
            manifest=manifest,
        )
    if macro.scope != "bundle":
        raise MacroConvertError(
            "macro_scope_unsupported",
            "Phase 4 convert only accepts scope=bundle macros",
            manifest=manifest,
            field="scope",
        )

    # §6.3: exclusive requires verified, else prefer
    routing_updates: dict[str, Any] = {"scope": "bundle"}
    if macro.routing.selection == "exclusive" and not macro.routing.verified:
        routing_updates["selection"] = "prefer"
    if not macro.routing.target_tool:
        routing_updates["target_tool"] = macro.name
    macro.routing = replace(macro.routing, **routing_updates)

    # Structural exclusive suppression needs a narrow intent and an
    # explicit atomic coverage relation.  Phrase-only exclusive routes are
    # still readable at runtime for compatibility, but new bundle manifests
    # must not claim verified exclusive without the structural contract.
    if macro.routing.selection == "exclusive":
        if not macro.routing.intent_key:
            raise MacroConvertError(
                "macro_retrieval_intent_missing",
                "verified exclusive macro requires routing.intent_key",
                manifest=manifest,
                field="routing.intent_key",
            )
        if not macro.routing.covered_tools:
            raise MacroConvertError(
                "macro_coverage_missing",
                "verified exclusive macro requires routing.covered_tools",
                manifest=manifest,
                field="routing.covered_tools",
            )

    _validate_covered_tools(macro, tool_index=tool_index, manifest=manifest)
    return validate_macro_core(macro, tool_index=tool_index)


def validate_session_macro_definition(
    macro: MacroDefinition,
    *,
    tool_index: Iterable[str] | None = None,
    forbid_workflow_tools: Iterable[str] | None = None,
) -> ValidatedSessionMacro:
    """Validate a session MacroDefinition; never silent-downgrade exclusive.

    ``tool_index`` is allowlist-shaped (see :func:`validate_macro_core`).
    """
    manifest = str(macro.provenance.get("manifest") or macro.name)
    if not macro.enabled:
        raise MacroConvertError(
            "macro_disabled",
            f"macro is disabled: {macro.name}",
            manifest=manifest,
        )
    if macro.scope != "session":
        raise MacroConvertError(
            "macro_scope_unsupported",
            "session validate requires scope=session",
            manifest=manifest,
            field="scope",
        )
    if not _KEBAB_NAME.match(macro.name):
        raise MacroConvertError(
            "macro_name_invalid",
            f"session macro name must be kebab-case: {macro.name!r}",
            manifest=manifest,
            field="name",
        )
    if macro.routing.selection == "exclusive":
        raise MacroConvertError(
            "macro_selection_forbidden",
            "session macros may not use selection=exclusive",
            manifest=manifest,
            field="routing.selection",
        )
    target = (macro.routing.target_tool or "").strip()
    if target and target != macro.name:
        raise MacroConvertError(
            "macro_target_mismatch",
            f"routing.target_tool must equal name ({macro.name!r}), got {target!r}",
            manifest=manifest,
            field="routing.target_tool",
        )
    session_updates: dict[str, Any] = {"target_tool": macro.name, "scope": "session"}
    if not macro.routing.selection:
        session_updates["selection"] = "prefer"
    macro.routing = replace(macro.routing, **session_updates)

    _validate_covered_tools(macro, tool_index=tool_index, manifest=manifest)
    workflow = validate_macro_core(
        macro,
        tool_index=tool_index,
        forbid_workflow_tools=forbid_workflow_tools,
    )
    tool_spec = AgentToolSpec(
        name=macro.name,
        description=macro.description or workflow.description or macro.name,
        input_schema=_workflow_input_schema(workflow),
        call_type="workflow",
        call_impl={"catalog_id": f"session:{macro.name}"},
        tags=("macro", "workflow", "session"),
        source="session-macro",
    )
    return ValidatedSessionMacro(
        definition=macro,
        workflow=workflow,
        tool_spec=tool_spec,
    )
