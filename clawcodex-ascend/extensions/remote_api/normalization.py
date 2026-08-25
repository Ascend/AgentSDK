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

"""Normalize OpenAI/Hermes request bodies into ClawCodex message objects."""

# Pylint cannot infer dynamically exposed clawcodex_ext subpackages.
# pylint: disable=no-name-in-module

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from clawcodex_ext.types.content_blocks import (
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from clawcodex_ext.types.messages import AssistantMessage, Message, UserMessage

from .errors import RemoteAPIError


_DATA_IMAGE_RE = re.compile(r"^data:(image/[a-z0-9.+_-]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)
_DATA_URI_RE = re.compile(r"^data:", re.IGNORECASE)
_REMOTE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


FORBIDDEN_WORKSPACE_FIELDS = {"cwd", "workspace", "workdir", "working_dir", "root_dir"}


@dataclass
class NormalizedMessages:
    messages: list[Message] = field(default_factory=list)
    instructions: str = ""


def reject_workspace_override(body: dict[str, Any]) -> None:
    forbidden = sorted(FORBIDDEN_WORKSPACE_FIELDS.intersection(body))
    if forbidden:
        fields = ", ".join(forbidden)
        raise RemoteAPIError(400, f"workspace override is not supported: {fields}")


def normalize_chat_messages(messages: Any) -> NormalizedMessages:
    if not isinstance(messages, list) or not messages:
        raise RemoteAPIError(400, "messages must be a non-empty array")

    out: list[Message] = []
    system_parts: list[str] = []
    has_user = False
    for raw in messages:
        if not isinstance(raw, dict):
            raise RemoteAPIError(400, "each message must be a JSON object")
        role = str(raw.get("role", "")).strip()
        content = raw.get("content", "")
        if _has_file_reference(raw):
            raise _unsupported_content("uploaded files are not supported")
        if role in {"system", "developer"}:
            text = _content_to_text(content)
            if text:
                system_parts.append(text)
            continue
        if role == "user":
            has_user = True
            out.append(UserMessage(content=_normalize_content(content)))
            continue
        if role == "assistant":
            out.append(_normalize_assistant_message(raw))
            continue
        if role == "tool":
            text = _content_to_text(content)
            tool_call_id = str(raw.get("tool_call_id", "")).strip()
            if not tool_call_id:
                raise RemoteAPIError(400, "tool messages must include tool_call_id")
            out.append(
                UserMessage(
                    content=[
                        ToolResultBlock(
                            tool_use_id=tool_call_id,
                            content=text or "[tool result]",
                        )
                    ],
                    origin="tool_result",
                )
            )
            continue
        if role == "function":
            out.append(UserMessage(content=_content_to_text(content) or "[function result]"))
            continue
        raise RemoteAPIError(400, f"unsupported message role: {role}")

    if not has_user:
        raise RemoteAPIError(400, "messages must include a user message")
    return NormalizedMessages(messages=out, instructions="\n\n".join(system_parts))


def normalize_responses_input(raw_input: Any) -> NormalizedMessages:
    if isinstance(raw_input, str):
        text = raw_input.strip()
        if not text:
            raise RemoteAPIError(400, "input must not be empty")
        return NormalizedMessages(messages=[UserMessage(content=text)])

    if not isinstance(raw_input, list) or not raw_input:
        raise RemoteAPIError(400, "input must be a string or non-empty array")

    # Responses accepts either message objects or a bare content-part array.
    if all(isinstance(item, dict) and "role" not in item for item in raw_input):
        return NormalizedMessages(messages=[UserMessage(content=_normalize_content(raw_input))])

    out: list[Message] = []
    system_parts: list[str] = []
    has_user = False
    for raw in raw_input:
        if not isinstance(raw, dict):
            raise RemoteAPIError(400, "input messages must be JSON objects")
        if _has_file_reference(raw):
            raise _unsupported_content("uploaded files are not supported")
        role = str(raw.get("role", "user")).strip() or "user"
        content = raw.get("content", raw.get("text", ""))
        if role in {"system", "developer"}:
            text = _content_to_text(content)
            if text:
                system_parts.append(text)
        elif role == "user":
            has_user = True
            out.append(UserMessage(content=_normalize_content(content)))
        elif role == "assistant":
            out.append(AssistantMessage(content=_normalize_content(content, allow_images=False)))
        else:
            raise RemoteAPIError(400, f"unsupported input role: {role}")
    if not has_user:
        raise RemoteAPIError(400, "input must include a user message")
    return NormalizedMessages(messages=out, instructions="\n\n".join(system_parts))


def _normalize_assistant_message(raw: dict[str, Any]) -> AssistantMessage:
    content = raw.get("content")
    tool_calls = raw.get("tool_calls")
    if not tool_calls:
        return AssistantMessage(content=_normalize_content(content, allow_images=False))
    if not isinstance(tool_calls, list):
        raise RemoteAPIError(400, "assistant tool_calls must be an array")

    blocks: list[Any] = []
    normalized_content = _normalize_content(content, allow_images=False)
    if isinstance(normalized_content, str):
        if normalized_content:
            blocks.append(TextBlock(text=normalized_content))
    else:
        blocks.extend(normalized_content)

    for raw_call in tool_calls:
        if not isinstance(raw_call, dict):
            raise RemoteAPIError(400, "assistant tool_calls entries must be objects")
        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise RemoteAPIError(400, "assistant tool_calls must include function objects")
        call_id = str(raw_call.get("id", "")).strip()
        name = str(function.get("name", "")).strip()
        if not call_id or not name:
            raise RemoteAPIError(400, "assistant tool_calls must include id and function.name")
        arguments = function.get("arguments", "{}")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError as exc:
            raise RemoteAPIError(400, "assistant tool call arguments must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise RemoteAPIError(400, "assistant tool call arguments must decode to an object")
        blocks.append(ToolUseBlock(id=call_id, name=name, input=parsed))
    return AssistantMessage(content=blocks)


def merge_instructions(*parts: Any) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, str) and part.strip():
            text_parts.append(part.strip())
    return "\n\n".join(text_parts)


def _normalize_content(content: Any, *, allow_images: bool = True) -> str | list[Any]:
    if content is None:
        return ""
    if isinstance(content, str):
        if _DATA_URI_RE.match(content):
            raise _unsupported_content("bare data URLs are not supported")
        return content
    if not isinstance(content, list):
        return str(content)

    blocks: list[Any] = []
    for part in content:
        if isinstance(part, str):
            blocks.append(TextBlock(text=part))
            continue
        if not isinstance(part, dict):
            blocks.append(TextBlock(text=str(part)))
            continue
        if _has_file_reference(part):
            raise _unsupported_content("uploaded files are not supported")
        ptype = str(part.get("type", "")).strip()
        if ptype in {"text", "input_text", "output_text"}:
            blocks.append(TextBlock(text=str(part.get("text", ""))))
            continue
        if ptype in {"image_url", "input_image"}:
            if not allow_images:
                raise _unsupported_content("assistant image content is not supported")
            blocks.append(_image_part_to_block(part))
            continue
        if ptype in {"file", "input_file"}:
            raise _unsupported_content("uploaded files are not supported")
        raise _unsupported_content(f"unsupported content part type: {ptype}")
    text_blocks = [block for block in blocks if isinstance(block, TextBlock) and block.text]
    non_text_blocks = [block for block in blocks if not isinstance(block, TextBlock)]
    if not non_text_blocks:
        return "\n".join(block.text for block in text_blocks).strip()
    return blocks


def _content_to_text(content: Any) -> str:
    normalized = _normalize_content(content, allow_images=False)
    if isinstance(normalized, str):
        return normalized.strip()
    parts: list[str] = []
    for block in normalized:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "\n".join(parts).strip()


def _image_part_to_block(part: dict[str, Any]) -> ImageBlock:
    raw = part.get("image_url")
    url = raw.get("url") if isinstance(raw, dict) else raw
    if not isinstance(url, str) or not url:
        raise _unsupported_content("image_url must include a url")
    if _REMOTE_URL_RE.match(url):
        raise _unsupported_content("remote image URLs are not supported")
    match = _DATA_IMAGE_RE.match(url)
    if not match:
        if _DATA_URI_RE.match(url):
            raise _unsupported_content("non-image data URLs are not supported")
        raise _unsupported_content("image_url must be a data:image URL")
    media_type, data = match.group(1), match.group(2)
    if not data:
        raise _unsupported_content("image data URL must include base64 data")
    return ImageBlock(
        source={
            "type": "base64",
            "media_type": media_type.lower(),
            "data": data,
        }
    )


def _has_file_reference(value: dict[str, Any]) -> bool:
    if "file_id" in value or "input_file" in value:
        return True
    if value.get("type") in {"file", "input_file"}:
        return True
    return False


def _unsupported_content(detail: str) -> RemoteAPIError:
    return RemoteAPIError(
        400,
        detail,
        code="unsupported_content_type",
        error_type="invalid_request_error",
    )
