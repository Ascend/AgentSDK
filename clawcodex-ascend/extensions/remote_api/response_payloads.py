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

"""OpenAI/Hermes payload builders and request-field validation."""

from __future__ import annotations

# Relative imports bind local contracts; target subpackages are exposed dynamically.
# pylint: disable=relative-beyond-top-level,no-name-in-module

import json
import time
from typing import Any

from clawcodex_ext.types.content_blocks import ImageBlock, TextBlock

from .errors import RemoteAPIError
from .runner import (
    RemoteRunResult,
    RemoteToolCall,
    RemoteToolResult,
)

API_MODEL_NAME = "clawcodex-agent"


def _chat_completion_payload(
    run_id: str,
    model: str,
    result: RemoteRunResult,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": _openai_usage(result.usage),
    }


def _responses_payload(
    *,
    response_id: str,
    model: str,
    result: RemoteRunResult,
    previous_response_id: str | None,
    conversation: str | None,
    store: bool,
    instructions: str | None = None,
    output: list[dict[str, Any]] | None = None,
    created_at: int | None = None,
) -> dict[str, Any]:
    response_output = output if output is not None else _response_output_items(response_id, result.text, result.events)
    response_created_at = created_at or int(time.time())
    payload = _responses_base_payload(
        response_id=response_id,
        model=model,
        previous_response_id=previous_response_id,
        conversation=conversation,
        store=store,
        status="completed",
        completed_at=int(time.time()),
        output=response_output,
        output_text=result.text,
        usage=_responses_usage(result.usage),
        created_at=response_created_at,
        instructions=instructions,
    )
    return payload


def _responses_base_payload(
    *,
    response_id: str,
    model: str,
    previous_response_id: str | None,
    conversation: str | None,
    store: bool,
    status: str,
    completed_at: int | None,
    output: list[dict[str, Any]],
    output_text: str,
    usage: dict[str, Any] | None,
    created_at: int | None = None,
    instructions: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at or int(time.time()),
        "status": status,
        "completed_at": completed_at,
        "error": error,
        "incomplete_details": None,
        "instructions": instructions,
        "max_output_tokens": None,
        "model": model,
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": True,
        "previous_response_id": previous_response_id,
        "reasoning": {"effort": None, "summary": None},
        "store": store,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": usage,
        "user": None,
        "metadata": _response_metadata(),
    }
    if conversation:
        payload["conversation"] = {"id": conversation}
    return payload


def _response_output_items(
    response_id: str,
    text: str,
    events: list[Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    pending_call_ids: dict[str, list[str]] = {}
    for event in events:
        if isinstance(event, RemoteToolCall):
            item = _response_tool_call_item(event, response_id, len(output))
            output.append(item)
            _remember_pending_call(pending_call_ids, event, item["call_id"])
        elif isinstance(event, RemoteToolResult):
            fallback_call_id = _take_pending_call(pending_call_ids, event)
            output.append(
                _response_tool_result_item(
                    event,
                    response_id,
                    len(output),
                    fallback_call_id=fallback_call_id,
                )
            )
    output.append(_response_message_item(response_id, text))
    return output


def _response_message_item(response_id: str, text: str) -> dict[str, Any]:
    return {
        "id": f"msg_{response_id.removeprefix('resp_')}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def _response_input_items(response_id: str, messages: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    suffix = response_id.removeprefix("resp_")
    for index, message in enumerate(messages):
        role = getattr(message, "role", "user")
        if role not in {"user", "assistant", "system"}:
            role = "user"
        items.append(
            {
                "id": f"msg_{suffix}_input_{index}",
                "type": "message",
                "role": role,
                "content": _response_input_content(getattr(message, "content", "")),
            }
        )
    return items


def _response_input_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]
    out: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextBlock):
            out.append({"type": "input_text", "text": block.text})
        elif isinstance(block, ImageBlock):
            source = block.source
            media_type = str(source.get("media_type", "image/png"))
            data = str(source.get("data", ""))
            out.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{data}",
                }
            )
        else:
            text = getattr(block, "text", None)
            out.append({"type": "input_text", "text": str(text if text is not None else block)})
    return out


def _response_tool_call_item(
    event: RemoteToolCall,
    response_id: str,
    output_index: int,
) -> dict[str, Any]:
    suffix = response_id.removeprefix("resp_")
    return {
        "id": f"fc_{suffix}_{output_index}",
        "type": "function_call",
        "status": "completed",
        "name": event.tool_name,
        "arguments": _json_or_string(event.params),
        "call_id": event.tool_use_id or f"call_{suffix}_{output_index}",
    }


def _remember_pending_call(
    pending_call_ids: dict[str, list[str]],
    event: RemoteToolCall,
    call_id: str,
) -> None:
    if event.tool_name:
        pending_call_ids.setdefault(event.tool_name, []).append(call_id)


def _take_pending_call(
    pending_call_ids: dict[str, list[str]],
    event: RemoteToolResult,
) -> str | None:
    """Consume a call pairing, preferring the event's exact tool-use ID."""
    call_ids = pending_call_ids.get(event.tool_name, [])
    if event.tool_use_id:
        try:
            call_ids.remove(event.tool_use_id)
        except ValueError:
            pass
        return None
    return call_ids.pop(0) if call_ids else None


def _response_tool_result_item(
    event: RemoteToolResult,
    response_id: str,
    output_index: int,
    *,
    fallback_call_id: str | None = None,
) -> dict[str, Any]:
    suffix = response_id.removeprefix("resp_")
    # event.result is a wrapper dict {"output": …, "is_error": …}.
    # Open WebUI consumes Responses tool output as an array of input content
    # parts and calls ``part.get(...)`` on every entry.  The Responses API
    # permits either a string or content-part array here; use the array form
    # so tool output renders in Open WebUI and remains safe on the next turn.
    actual_output = event.result.get("output", "") if isinstance(event.result, dict) else event.result
    return {
        "id": f"fco_{suffix}_{output_index}",
        "type": "function_call_output",
        "status": "completed",
        "call_id": event.tool_use_id or fallback_call_id or f"call_{suffix}_{output_index}",
        "output": [
            {
                "type": "input_text",
                "text": _json_or_string(actual_output),
            }
        ],
    }


def _json_or_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _openai_usage(usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = _int_usage(usage, "input_tokens")
    output_tokens = _int_usage(usage, "output_tokens")
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _responses_usage(usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = _int_usage(usage, "input_tokens")
    output_tokens = _int_usage(usage, "output_tokens")
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": _first_int_usage(
                usage,
                "cached_tokens",
                "cache_read_input_tokens",
            )
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": _int_usage(usage, "reasoning_tokens")},
        "total_tokens": input_tokens + output_tokens,
    }


def _int_usage(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _first_int_usage(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in usage:
            return _int_usage(usage, key)
    return 0


def _optional_string_field(body: dict[str, Any], field: str) -> str | None:
    value = body.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RemoteAPIError(400, f"{field} must be a string")
    return value or None


def _validate_common_request_fields(body: dict[str, Any]) -> None:
    model = body.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise RemoteAPIError(400, "model must be a non-empty string")
    for field in ("stream", "store"):
        value = body.get(field)
        if value is not None and not isinstance(value, bool):
            raise RemoteAPIError(400, f"{field} must be a boolean")


def _chat_include_usage(body: dict[str, Any]) -> bool:
    options = body.get("stream_options")
    if options is None:
        return False
    if not isinstance(options, dict):
        raise RemoteAPIError(400, "stream_options must be an object")
    include_usage = options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise RemoteAPIError(400, "stream_options.include_usage must be a boolean")
    if body.get("stream") is not True:
        raise RemoteAPIError(400, "stream_options is only supported when stream is true")
    return include_usage


def _optional_conversation_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        raw_id = value.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            return raw_id.strip()
    raise RemoteAPIError(400, "conversation must be a string or object with an id")


def _stream_error_payload(exc: RemoteAPIError) -> dict[str, Any]:
    return {
        "error": {
            "message": exc.detail,
            "type": exc.error_type,
            "code": exc.code,
        }
    }


def _resolve_query_model(request_model: Any, service_model: str | None) -> str | None:
    if isinstance(request_model, str) and request_model and request_model != API_MODEL_NAME:
        return request_model
    if service_model and service_model != API_MODEL_NAME:
        return service_model
    return None


def _response_metadata() -> dict[str, Any]:
    try:
        from .state_reporter import current_automation_state

        return {"automation_state": current_automation_state()}
    except Exception:
        return {}
