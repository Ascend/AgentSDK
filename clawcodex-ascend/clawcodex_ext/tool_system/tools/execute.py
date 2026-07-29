"""ExecuteTool — delegate execution to another tool by name.

Mirrors ``claude-code-best/packages/builtin-tools/.../ExecuteTool.ts``.
The model invokes deferred / extra tools (those discovered via
SearchExtraTools) through this dispatcher: it takes a ``tool_name`` and
a ``params`` dict, looks the tool up in the active registry / tool pool,
runs schema validation, permission checks, and ``validate_input``, then
delegates the call.

Why a central dispatcher: deferred tools are not loaded into the model's
schema by default. Once discovered, the model still needs a stable
handle to invoke them. ExecuteTool is that handle — it centralizes the
validation/permission guard so each deferred tool does not have to
defensively re-implement it.

The lookup order is:
1. ``context.tool_registry`` (the active ``ToolRegistry``) — preferred,
   authoritative.
2. ``context.options.tools`` (the tool pool passed to the query loop) —
   fallback for contexts that build a pool without a registry.

NOT concurrency-safe (delegated tool may mutate state). Read-only-ness
is inherited from the target tool.
"""

from __future__ import annotations

import json
from typing import Any

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..schema_validation import coerce_tool_input, validate_json_schema


def _resolve_target(name: str, context: ToolContext) -> Tool | None:
    from extensions.sop_converter.runtime.macros.resolve_tool import (
        resolve_tool_for_context,
    )

    registry = getattr(context, "tool_registry", None)
    return resolve_tool_for_context(context, name, base_registry=registry)


def _wrap_user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _execute_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    tool_name = tool_input.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ToolInputError("tool_name must be a non-empty string")
    params = tool_input.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ToolInputError("params must be an object")

    target = _resolve_target(tool_name, context)
    if target is None:
        return ToolResult(
            name="Execute",
            output={"result": None, "tool_name": tool_name},
            new_messages=[
                _wrap_user_message(f'Tool "{tool_name}" not found. Use SearchExtraTools to discover available tools.')
            ],
        )

    # isEnabled gate — deferred tools may be disabled (e.g. Remote Control
    # not connected). Mirror the TS guard rather than calling blindly.
    is_enabled = getattr(target, "is_enabled", None)
    if callable(is_enabled) and not is_enabled():
        return ToolResult(
            name="Execute",
            output={"result": None, "tool_name": tool_name},
            new_messages=[_wrap_user_message(f'Tool "{tool_name}" is currently disabled.')],
        )

    # Schema-validate params against the target tool BEFORE delegating.
    # The deferred dispatch path passes raw model output straight through;
    # a wrong field name or missing required field would otherwise crash
    # inside the target tool's first .trim()/.length/.split().
    schema = getattr(target, "input_schema", None)
    coerced = params
    if isinstance(schema, dict):
        try:
            coerced = coerce_tool_input(params, schema, root_name=target.name)
            validate_json_schema(coerced, schema, root_name=target.name)
        except Exception as exc:  # noqa: BLE001 — surface validation errors to the model
            return ToolResult(
                name="Execute",
                output={"result": None, "tool_name": tool_name},
                new_messages=[_wrap_user_message(f'Invalid parameters for tool "{tool_name}": {exc}')],
            )

    # Target-level input validation (semantic checks beyond JSON schema).
    validate_input = getattr(target, "validate_input", None)
    if callable(validate_input):
        validation = validate_input(coerced, context)
        if validation is not None and not getattr(validation, "result", True):
            return ToolResult(
                name="Execute",
                output={"result": None, "tool_name": tool_name},
                new_messages=[
                    _wrap_user_message(
                        f'Invalid parameters for tool "{tool_name}": {getattr(validation, "message", "")}'
                    )
                ],
            )

    # Permission check — delegate to the target tool's own check so the
    # ask/deny UI and persisted rules match a direct invocation.
    check_permissions = getattr(target, "check_permissions", None)
    if callable(check_permissions):
        decision = check_permissions(coerced, context)
        behavior = getattr(decision, "behavior", None)
        if behavior == "deny":
            msg = getattr(decision, "message", None) or "Permission denied"
            return ToolResult(
                name="Execute",
                output={"result": None, "tool_name": tool_name},
                new_messages=[_wrap_user_message(f'Permission denied for tool "{tool_name}": {msg}')],
            )

    # Delegate execution to the target tool. We call the tool's ``call``
    # directly (the registry's ``dispatch`` would re-run schema/permission
    # checks we already performed). The target's result is wrapped so the
    # model sees the ExecuteTool output shape.
    try:
        inner: ToolResult = target.call(coerced, context)
    except Exception as exc:  # noqa: BLE001 — surface target errors to the model
        return ToolResult(
            name="Execute",
            output={"result": None, "tool_name": tool_name},
            is_error=True,
            new_messages=[_wrap_user_message(f'Tool "{tool_name}" raised: {exc}')],
        )

    return ToolResult(
        name="Execute",
        output={"result": inner.output, "tool_name": tool_name},
        is_error=getattr(inner, "is_error", False),
        new_messages=getattr(inner, "new_messages", None),
    )


def _map_result_to_api(output: Any, tool_use_id: str) -> dict[str, Any]:
    content: str | list[dict[str, Any]]
    if isinstance(output, (dict, list)):
        content = json.dumps(output, ensure_ascii=False)
    else:
        content = str(output)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


ExecuteTool: Tool = build_tool(
    name="Execute",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tool_name": {
                "type": "string",
                "description": (
                    'The exact name of the target tool to execute (e.g. "CronCreate", "mcp__server__action").'
                ),
            },
            "params": {
                "type": "object",
                "description": "The parameters to pass to the target tool.",
                "additionalProperties": True,
            },
        },
        "required": ["tool_name"],
    },
    call=_execute_call,
    prompt=(
        "Execute: invoke another tool by name with the given parameters. "
        "Use this for deferred / extra tools discovered via SearchExtraTools — "
        "the model specifies tool_name and params, and this tool handles "
        "schema validation, permission checks, and delegation. If the target "
        "tool is not found or disabled, an explanatory user message is "
        "returned. Permission decisions are delegated to the target tool."
    ),
    description=(
        "Execute a deferred tool by name with parameters. Delegates schema "
        "validation and permission checks to the target tool."
    ),
    search_hint="execute run invoke call a deferred tool by name with parameters",
    max_result_size_chars=100_000,
    should_defer=False,
    is_concurrency_safe=lambda _input: False,
    is_read_only=lambda _input: False,
    map_result_to_api=_map_result_to_api,
    to_auto_classifier_input=lambda input_data: (f"Execute {input_data.get('tool_name', '')}"),
)
