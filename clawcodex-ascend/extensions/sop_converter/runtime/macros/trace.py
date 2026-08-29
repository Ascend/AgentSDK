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

"""Phase B — extract successful tool traces into MacroDefinition drafts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .errors import MacroConvertError

_KEBAB = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")

_SKIP_TOOL_NAMES = frozenset(
    {
        "register-macro-workflow",
        "RegisterMacroWorkflow",
        "register-macro-from-trace",
        "RegisterMacroFromTrace",
        "promote-macro-workflow",
        "PromoteMacroWorkflow",
    }
)


@dataclass(frozen=True)
class TraceToolStep:
    """One successful tool call extracted from session messages."""

    tool_name: str
    input: Mapping[str, Any]
    output: Any
    tool_use_id: str = ""


def _iter_content_blocks(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, list):
        return list(content)
    return [content]


def _block_type(block: Any) -> str:
    if isinstance(block, Mapping):
        return str(block.get("type") or "")
    return str(getattr(block, "type", "") or "")


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, Mapping):
        return block.get(key, default)
    return getattr(block, key, default)


def extract_successful_tool_steps(
    messages: Sequence[Any] | None,
    *,
    max_steps: int = 16,
    skip_names: Iterable[str] | None = None,
) -> list[TraceToolStep]:
    """Pair ToolUse with successful ToolResult; return the latest contiguous run.

    Walks messages in order, collects successful (tool_name, input, output)
    pairs, then returns the trailing contiguous successful sequence (capped
    at ``max_steps``). Failed results break continuity for the "latest run"
    window: only the suffix after the last failure is kept.
    """
    if max_steps < 1:
        raise MacroConvertError(
            "macro_schema_invalid",
            "max_steps must be >= 1",
            field="max_steps",
        )
    skip = set(_SKIP_TOOL_NAMES)
    if skip_names:
        skip.update(str(n) for n in skip_names)

    pending: dict[str, tuple[str, Mapping[str, Any]]] = {}
    successful: list[TraceToolStep] = []
    # Segments separated by failures; we keep the last non-empty segment.
    segments: list[list[TraceToolStep]] = [[]]

    for message in list(messages or []):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, Mapping):
            content = message.get("content")
        tool_use_result = getattr(message, "toolUseResult", None)
        if tool_use_result is None and isinstance(message, Mapping):
            tool_use_result = message.get("toolUseResult")

        for block in _iter_content_blocks(content):
            btype = _block_type(block)
            if btype == "tool_use":
                use_id = str(_block_get(block, "id") or "")
                name = str(_block_get(block, "name") or "").strip()
                raw_input = _block_get(block, "input") or {}
                if not isinstance(raw_input, Mapping):
                    raw_input = {}
                if use_id and name and name not in skip:
                    pending[use_id] = (name, dict(raw_input))
                continue

            if btype != "tool_result":
                continue
            use_id = str(_block_get(block, "tool_use_id") or "")
            is_error = bool(_block_get(block, "is_error", False))
            if use_id not in pending:
                continue
            name, args = pending.pop(use_id)
            if is_error:
                if segments[-1]:
                    segments.append([])
                continue
            output = tool_use_result if tool_use_result is not None else _block_get(block, "content", "")
            step = TraceToolStep(
                tool_name=name,
                input=args,
                output=output,
                tool_use_id=use_id,
            )
            segments[-1].append(step)
            successful.append(step)

    run = segments[-1] if segments else []
    if not run and successful:
        run = successful
    if len(run) > max_steps:
        run = run[-max_steps:]
    return list(run)


def _values_equal(a: Any, b: Any) -> bool:
    try:
        return a == b
    except Exception:
        return False


def _promote_inputs_and_bindings(
    steps: list[TraceToolStep],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Build workflow.inputs, step dicts, and outputs with conservative bindings."""
    inputs: dict[str, dict[str, Any]] = {}
    step_dicts: list[dict[str, Any]] = []
    prev_outputs: list[tuple[str, Any]] = []

    for index, step in enumerate(steps):
        step_id = f"step{index + 1}"
        args: dict[str, Any] = {}
        for key, value in dict(step.input).items():
            bound = False
            for prev_id, prev_out in prev_outputs:
                if _values_equal(value, prev_out):
                    args[key] = f"$steps.{prev_id}.output"
                    bound = True
                    break
            if bound:
                continue

            if key in inputs and _values_equal(inputs[key].get("_value"), value):
                args[key] = f"$input.{key}"
                continue

            input_name = key
            if key in inputs and not _values_equal(inputs[key].get("_value"), value):
                input_name = f"{step_id}_{key}"

            if index == 0:
                inputs[input_name] = {
                    "type": "string" if isinstance(value, str) else "object",
                    "required": True,
                    "_value": value,
                }
                args[key] = f"$input.{input_name}"
            else:
                args[key] = value

        step_dicts.append(
            {
                "id": step_id,
                "kind": "tool",
                "callable_ref": step.tool_name,
                "args": args,
            }
        )
        prev_outputs.append((step_id, step.output))

    clean_inputs = {name: {k: v for k, v in schema.items() if not k.startswith("_")} for name, schema in inputs.items()}
    if not clean_inputs:
        clean_inputs = {"query": {"type": "string", "required": False}}

    last_id = step_dicts[-1]["id"] if step_dicts else "step1"
    outputs = {"result": f"$steps.{last_id}.output"}
    return clean_inputs, step_dicts, outputs


def trace_steps_to_definition_dict(
    steps: Sequence[TraceToolStep],
    *,
    name: str,
    description: str = "",
    phrases: Sequence[str] | None = None,
    keywords: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a session MacroDefinition dict from extracted trace steps."""
    name = str(name or "").strip()
    if not _KEBAB.match(name):
        raise MacroConvertError(
            "macro_name_invalid",
            f"session macro name must be kebab-case: {name!r}",
            field="name",
        )
    if not steps:
        raise MacroConvertError(
            "macro_trace_empty",
            "no successful tool steps found in session trace",
        )
    if len(steps) > 16:
        raise MacroConvertError(
            "macro_schema_invalid",
            "trace exceeds 16 steps",
        )

    inputs, step_dicts, outputs = _promote_inputs_and_bindings(list(steps))
    phrase_list = [str(p) for p in (phrases or []) if str(p).strip()]
    if not phrase_list:
        phrase_list = [f"run {name}"]
    keyword_list = [str(k) for k in (keywords or []) if str(k).strip()]
    if not keyword_list:
        keyword_list = [name]

    return {
        "version": 1,
        "name": name,
        "description": description or f"Session macro from tool trace: {name}",
        "scope": "session",
        "enabled": True,
        "workflow": {
            "inputs": inputs,
            "steps": step_dicts,
            "outputs": outputs,
        },
        "routing": {
            "phrases": phrase_list,
            "keywords": keyword_list,
            "selection": "prefer",
            "priority": 100,
            "target_tool": name,
        },
        "provenance": {
            "kind": "session_trace",
            "source_tools": [s.tool_name for s in steps],
        },
    }


__all__ = [
    "TraceToolStep",
    "extract_successful_tool_steps",
    "trace_steps_to_definition_dict",
]
