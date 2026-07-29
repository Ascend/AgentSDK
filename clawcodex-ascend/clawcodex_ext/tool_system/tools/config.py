from __future__ import annotations

from typing import Any

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult


def _config_classifier_input(input_data: dict) -> str:
    """Return a stable, reviewable summary for the auto classifier."""
    data = input_data or {}
    setting = data.get("setting", "") or ""
    operation = data.get("operation")
    if operation == "get" or ("value" not in data and operation is None):
        return setting
    if not any(key in data for key in ("scope", "domain", "operation")):
        return f"{setting} = {data.get('value')}"
    return f"{operation or 'set'} {data.get('scope', 'user')}:{setting} = {data.get('value')}"


def _config_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    from clawcodex_ext.configuration import (
        ConfigMutationRequest,
        ConfigurationError,
        mutate_configuration,
    )

    setting = tool_input["setting"]
    if not isinstance(setting, str) or not setting:
        raise ToolInputError("setting must be a non-empty string")

    request_kwargs: dict[str, Any] = {
        "setting": setting,
        "scope": tool_input.get("scope", "user"),
        "domain": tool_input.get("domain"),
        "operation": tool_input.get("operation"),
    }
    if "value" in tool_input:
        request_kwargs["value"] = tool_input.get("value")
    try:
        result = mutate_configuration(ConfigMutationRequest(**request_kwargs), context)
    except ConfigurationError as exc:
        raise ToolInputError(str(exc)) from exc
    return ToolResult(name="Config", output=result.to_dict())


ConfigTool: Tool = build_tool(
    name="Config",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "setting": {"type": "string"},
            "value": {},
            "scope": {"type": "string", "enum": ["user", "project", "local"]},
            "domain": {"type": "string", "enum": ["app", "settings"]},
            "operation": {
                "type": "string",
                "enum": ["get", "set", "unset", "append_unique", "remove"],
            },
        },
        "required": ["setting"],
    },
    call=_config_call,
    prompt=(
        "Read or incrementally mutate scoped ClawCodex app/settings configuration. "
        "Supports get, set, unset, append_unique, and remove."
    ),
    description="Read or safely mutate scoped ClawCodex configuration values.",
    max_result_size_chars=100_000,
    is_destructive=lambda data: (
        data.get("operation") != "get"
        and ("value" in data or data.get("operation") in {"unset", "append_unique", "remove"})
    ),
    to_auto_classifier_input=_config_classifier_input,
)
