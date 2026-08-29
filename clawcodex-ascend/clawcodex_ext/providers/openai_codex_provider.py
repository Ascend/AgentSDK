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

# pylint: disable=attribute-defined-outside-init
"""OpenAI Codex provider backed by ChatGPT OAuth tokens."""

from __future__ import annotations

import json
import os
import queue
import threading
from typing import Any, Generator, Optional

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None

from src.auth.codex_oauth import CODEX_BASE_URL, resolve_codex_runtime_credentials
from clawcodex_ext.providers.base import ChatResponse, MessageInput
from clawcodex_ext.providers.codex_models import CODEX_FALLBACK_MODELS, get_codex_model_ids
from clawcodex_ext.providers._stream_abort import StreamAbortGuard
from clawcodex_ext.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _convert_anthropic_messages_to_openai,
    _convert_to_openai_tool_schema,
)

_INTERNAL_CHAT_KWARGS = {"model", "tools", "abort_signal", "stream", "on_thinking_chunk"}


class OpenAICodexProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(api_key, base_url or CODEX_BASE_URL, model or CODEX_FALLBACK_MODELS[0])

    def _create_client(self) -> Any:
        if OpenAI is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "openai package is not installed. Install optional dependencies to use OpenAICodexProvider."
            )
        credentials = resolve_codex_runtime_credentials()
        self.api_key = credentials.api_key
        self.base_url = self.base_url or credentials.base_url
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": 60.0,
        }
        if os.environ.get("CLAWCODEX_SSL_VERIFY", "").lower() in ("0", "false", "no"):
            import httpx

            kwargs["http_client"] = httpx.Client(verify=False, timeout=60.0)
        return OpenAI(**kwargs)

    @property
    def client(self) -> Any:
        credentials = resolve_codex_runtime_credentials()
        if credentials.api_key != self.api_key:
            self.api_key = credentials.api_key
            self._client = None
        if self.base_url != credentials.base_url:
            self.base_url = credentials.base_url
            self._client = None
        return super().client

    def _prepare_responses_input(
        self, messages: list[MessageInput], *, model: str | None = None
    ) -> tuple[str | None, list[dict[str, Any]]]:
        provider_messages = _convert_anthropic_messages_to_openai(super()._prepare_messages(messages, model=model))
        instructions: list[str] = []
        responses_input: list[dict[str, Any]] = []

        for message in provider_messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                _append_system_instruction(instructions, content)
                continue
            if role == "tool":
                responses_input.append(_tool_message_to_response_input(message, content))
                continue
            if role == "assistant" and message.get("tool_calls"):
                responses_input.extend(_assistant_tool_calls_to_response_input(message, content))
                continue
            if role in {"user", "assistant"}:
                responses_input.append({"role": role, "content": _content_to_responses_parts(content, role=role)})

        return ("\n\n".join(instructions) or None), responses_input

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        model, request_kwargs = self._build_responses_request(messages, tools, **kwargs)
        stream = self.client.responses.create(**request_kwargs)
        return _collect_responses_stream(stream, model=model)

    def chat_stream(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        response = self.chat(messages, tools, **kwargs)
        if response.content:
            yield response.content

    def chat_stream_response(  # pylint: disable=arguments-renamed
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk=None,
        on_thinking_chunk=None,
        abort_signal: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        guard = StreamAbortGuard(abort_signal)
        guard.raise_if_pre_aborted()

        model, request_kwargs = self._build_responses_request(messages, tools, **kwargs)
        stream_queue: queue.Queue = queue.Queue()
        done = object()

        def _drain_responses_stream() -> None:
            stream = None
            try:
                stream = self.client.responses.create(**request_kwargs)
                for event in stream:
                    stream_queue.put(event)
            except BaseException as exc:  # noqa: BLE001 - surface to polling thread
                stream_queue.put(exc)
            finally:
                stream_queue.put(done)

        worker = threading.Thread(
            target=_drain_responses_stream,
            daemon=True,
            name=f"openai-codex-responses-{id(self)}",
        )
        worker.start()
        events: list[Any] = []
        while True:
            try:
                item = stream_queue.get(timeout=0.1)
            except queue.Empty:
                if guard.aborted:
                    guard.raise_if_post_aborted()
                continue
            if item is done:
                break
            if isinstance(item, BaseException):
                if isinstance(item, Exception):
                    guard.reraise_if_aborted(item)
                    raise item
                raise item
            if guard.aborted:
                guard.raise_if_post_aborted()
            delta = _response_event_text_delta(item)
            if delta and on_text_chunk is not None:
                on_text_chunk(delta)
            events.append(item)

        guard.raise_if_post_aborted()
        return _collect_responses_stream(events, model=model)

    def _build_responses_request(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        model = self._get_model(**kwargs)
        instructions, responses_input = self._prepare_responses_input(messages, model=model)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "input": responses_input,
            "store": False,
            "stream": True,
        }
        if instructions:
            request_kwargs["instructions"] = instructions
        converted_tools = _responses_tools(tools)
        if converted_tools:
            request_kwargs["tools"] = converted_tools
        request_kwargs.update({k: v for k, v in kwargs.items() if k not in _INTERNAL_CHAT_KWARGS})
        return model, request_kwargs

    def get_available_models(self) -> list[str]:
        """Return the non-blocking local fallback used before cache refresh."""
        return list(CODEX_FALLBACK_MODELS)

    def model_catalog_cache_scope(self) -> str:
        """Return a stable account scope without persisting OAuth credentials."""
        from clawcodex_ext.bridge.jwt_utils import decode_jwt_payload

        payload = decode_jwt_payload(self.api_key)
        if payload is not None:
            auth = payload.get("https://api.openai.com/auth")
            if isinstance(auth, dict):
                account_id = auth.get("chatgpt_account_id")
                if isinstance(account_id, str) and account_id:
                    return f"chatgpt-account:{account_id}"
            subject = payload.get("sub")
            if isinstance(subject, str) and subject:
                return f"oauth-subject:{subject}"
        return f"oauth-token:{self.api_key}"

    def discover_available_models(self) -> list[str]:
        """Use the dedicated ChatGPT Codex catalog, not SDK ``/models``."""
        credentials = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        if not credentials.api_key.strip():
            raise RuntimeError("Codex model discovery requires a valid access token.")
        return get_codex_model_ids(credentials.api_key, raise_on_error=True)


def _append_system_instruction(instructions: list[str], content: Any) -> None:
    text = _content_to_text(content)
    if text:
        instructions.append(text)


def _tool_message_to_response_input(message: dict[str, Any], content: Any) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": message.get("tool_call_id", ""),
        "output": _content_to_text(content),
    }


def _assistant_tool_calls_to_response_input(message: dict[str, Any], content: Any) -> list[dict[str, Any]]:
    responses_input: list[dict[str, Any]] = []
    text = _content_to_text(content)
    if text:
        responses_input.append({"role": "assistant", "content": text})

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function", {})
        responses_input.append(
            {
                "type": "function_call",
                "call_id": tool_call.get("id", ""),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            }
        )
    return responses_input


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts = [_content_part_to_text(item) for item in content]
    return "\n".join(part for part in parts if part)


def _content_part_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)

    text = item.get("text")
    if isinstance(text, str):
        return text
    if item.get("type") == "image_url":
        return "[image]"
    if item.get("type") == "file":
        return "[file]"
    return json.dumps(item)


def _content_to_responses_parts(content: Any, *, role: str) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _content_to_text(content)

    parts: list[dict[str, Any]] = []
    text_type = "output_text" if role == "assistant" else "input_text"
    for item in content:
        if isinstance(item, str):
            parts.append({"type": text_type, "text": item})
            continue
        if not isinstance(item, dict):
            parts.append({"type": text_type, "text": str(item)})
            continue
        item_type = item.get("type")
        if item_type in {"text", "input_text", "output_text"}:
            text = item.get("text", "")
            if text:
                parts.append({"type": text_type, "text": text})
        elif item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and image_url.get("url"):
                parts.append({"type": "input_image", "image_url": image_url["url"]})
        elif item_type == "file":
            parts.append({"type": text_type, "text": "[file]"})
        else:
            parts.append({"type": text_type, "text": json.dumps(item)})
    return parts or ""


def _responses_tools(tools: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        openai_tool = _convert_to_openai_tool_schema(tool)
        if not openai_tool:
            continue
        function = openai_tool.get("function", {})
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
                "strict": False,
            }
        )
    return converted or None


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _response_event_text_delta(event: Any) -> str:
    event_type = _get_attr_or_key(event, "type", "")
    if event_type not in {"response.output_text.delta", "output_text.delta"}:
        return ""
    delta = _get_attr_or_key(event, "delta", "")
    return delta if isinstance(delta, str) else str(delta or "")


def _collect_responses_stream(stream: Any, *, model: str) -> ChatResponse:
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    response_model = model
    finish_reason = "stop"
    final_response: Any = None

    for event in stream:
        event_type = _get_attr_or_key(event, "type", "")
        if event_type in {"response.output_text.delta", "output_text.delta"}:
            delta = _get_attr_or_key(event, "delta", "")
            if delta:
                text_parts.append(delta)
        elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
            final_response = _get_attr_or_key(event, "response", None)
            if final_response is not None:
                response_model = _get_attr_or_key(final_response, "model", response_model)
                usage = _build_responses_usage_dict(_get_attr_or_key(final_response, "usage", None))
                finish_reason = _get_attr_or_key(final_response, "status", finish_reason)
                final_text, final_tool_uses = _parse_responses_output(final_response)
                if final_text:
                    text_parts = [final_text]
                if final_tool_uses:
                    tool_uses = final_tool_uses
        elif event_type in {"response.output_item.done", "output_item.done"}:
            item = _get_attr_or_key(event, "item", None)
            if item is not None and _get_attr_or_key(item, "type") == "function_call":
                tool_uses.append(_parse_function_call(item))
        elif event_type == "":
            final_response = event

    if final_response is not None and not usage:
        response_model = _get_attr_or_key(final_response, "model", response_model)
        usage = _build_responses_usage_dict(_get_attr_or_key(final_response, "usage", None))
        finish_reason = _get_attr_or_key(final_response, "status", finish_reason)
        final_text, final_tool_uses = _parse_responses_output(final_response)
        if final_text:
            text_parts = [final_text]
        if final_tool_uses:
            tool_uses = final_tool_uses

    return ChatResponse(
        content="".join(text_parts),
        model=response_model,
        usage=usage,
        finish_reason=finish_reason,
        tool_uses=tool_uses or None,
    )


def _parse_responses_output(response: Any) -> tuple[str, list[dict[str, Any]] | None]:
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for item in _get_attr_or_key(response, "output", []) or []:
        item_type = _get_attr_or_key(item, "type")
        if item_type == "message":
            text_parts.extend(_parse_message_text_parts(item))
        elif item_type == "function_call":
            tool_uses.append(_parse_function_call(item))
    return "\n".join(text_parts), tool_uses or None


def _parse_message_text_parts(item: Any) -> list[str]:
    text_parts: list[str] = []
    for part in _get_attr_or_key(item, "content", []) or []:
        part_type = _get_attr_or_key(part, "type")
        if part_type not in {"output_text", "text"}:
            continue
        text = _get_attr_or_key(part, "text", "")
        if text:
            text_parts.append(text)
    return text_parts


def _parse_function_call(item: Any) -> dict[str, Any]:
    return {
        "id": _get_attr_or_key(item, "call_id") or _get_attr_or_key(item, "id", ""),
        "name": _get_attr_or_key(item, "name", ""),
        "input": _parse_function_call_arguments(_get_attr_or_key(item, "arguments", "{}") or "{}"),
    }


def _parse_function_call_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except Exception:
            return {}
    else:
        parsed_arguments = arguments

    if isinstance(parsed_arguments, dict):
        return parsed_arguments
    return {"value": parsed_arguments}


def _build_responses_usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    input_tokens = _get_attr_or_key(usage, "input_tokens", _get_attr_or_key(usage, "prompt_tokens", 0))
    output_tokens = _get_attr_or_key(usage, "output_tokens", _get_attr_or_key(usage, "completion_tokens", 0))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _get_attr_or_key(usage, "total_tokens", input_tokens + output_tokens),
    }
