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

"""Native Grok / xAI provider.

xAI's Grok API is OpenAI-compatible: the same ``/chat/completions``
shape, the same ``tools`` array, the same ``response_format`` JSON
Schema option. The plan therefore says we can implement the
adapter by reusing the ``openai`` SDK against ``https://api.x.ai/v1``
rather than pulling in a Grok-specific client.

Differences from :class:`NativeOpenAIProvider`:

* Default ``base_url`` is the xAI endpoint, not ``api.openai.com``.
* Default model is a Grok family member (``grok-3`` at the time of
  writing). The exact model list is shorter than OpenAI's and is
  enumerated in :meth:`get_available_models`.
* Grok is more conservative on streaming-tools — the SDK supports
  it, but we mark the capability honestly: xAI's docs do not promise
  tool-call *deltas* mid-stream, so we advertise
  :data:`CAP_STREAMING_TOOLS` but recommend the non-streaming path
  for tool-calling-heavy flows.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None

from .base import NativeProvider
from .capabilities import (
    CAP_REASONING,
    CAP_STREAMING_TOOLS,
    CAP_STRUCTURED_OUTPUT,
    CAP_VISION,
)


# xAI's public endpoint. The ``/v1`` suffix is required; the SDK
# appends ``/chat/completions`` to whatever ``base_url`` we hand it.
_DEFAULT_BASE_URL = "https://api.x.ai/v1"

# Default model kept in sync with the values exposed in
# :meth:`get_available_models`. If the user passes a model via
# ``__init__`` we trust them.
_DEFAULT_MODEL = "grok-3"


def _ensure_sdk() -> None:
    if OpenAI is None:
        raise ModuleNotFoundError(
            "openai package is not installed. Run `pip install openai` "
            "to use the native Grok adapter (xAI exposes an OpenAI-"
            "compatible endpoint)."
        )


class NativeGrokProvider(NativeProvider):
    """Native Grok / xAI adapter built on the ``openai`` SDK."""

    capabilities = {
        CAP_STRUCTURED_OUTPUT,
        CAP_STREAMING_TOOLS,
        CAP_VISION,
        CAP_REASONING,
    }

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(api_key, base_url, model or _DEFAULT_MODEL)
        _ensure_sdk()
        # The base URL is required — without it the SDK targets
        # OpenAI's endpoint, which is almost never what the user
        # wants when they say "grok".
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url or _DEFAULT_BASE_URL,
            "timeout": 60.0,
        }
        self.client = OpenAI(**kwargs)

    def get_provider_name(self) -> str:
        return "grok"

    # ---- request shape ----

    def _build_request(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        # Grok uses the same Anthropic→OpenAI translation path as
        # the native OpenAI adapter. We deliberately re-import here
        # rather than holding a class-level reference so the import
        # cost only pays when a Grok request actually happens.
        from clawcodex_ext.providers.openai_compatible import (
            convert_anthropic_messages_to_openai as _convert_anthropic_messages_to_openai,
        )
        from src.models.capabilities import supports_vision as _supports_vision

        resolved_model = self._get_model(**kwargs)
        sv = _supports_vision(resolved_model) if resolved_model else None

        request: dict[str, Any] = {
            "model": resolved_model,
            "messages": _convert_anthropic_messages_to_openai(
                self._prepare_messages(messages),
                supports_vision=sv,
            ),
        }
        if tools:
            request["tools"] = tools
        # xAI's accepted parameters are a subset of OpenAI's; we
        # forward the common ones. ``response_format`` works for
        # JSON-mode but not the strict-schema variant on every
        # model — that limitation is exposed via the capability
        # registry (callers can check ``CAP_STRUCTURED_OUTPUT``).
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "response_format",
            "stop",
            "user",
        ):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        return request

    # ---- public API ----

    def chat(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        request = self._build_request(messages, tools, kwargs)
        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        tool_uses: Optional[list[dict[str, Any]]] = None
        if getattr(message, "tool_calls", None):
            import json

            tool_uses = []
            for tc in message.tool_calls:
                fn = getattr(tc, "function", None)
                raw = getattr(fn, "arguments", "") or "{}"
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(parsed, dict):
                        parsed = {"_raw": parsed}
                except (ValueError, TypeError):
                    parsed = {"_raw": raw}
                tool_uses.append(
                    {
                        "id": str(getattr(tc, "id", "") or ""),
                        "name": str(getattr(fn, "name", "") or ""),
                        "input": parsed,
                    }
                )
        usage = getattr(response, "usage", None)
        usage_dict: dict[str, Any] = {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        }
        from src.providers.base import ChatResponse

        return ChatResponse(
            content=str(getattr(message, "content", "") or ""),
            model=str(getattr(response, "model", request["model"])),
            usage=usage_dict,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            tool_uses=tool_uses,
        )

    def chat_stream(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        request = self._build_request(messages, tools, kwargs)
        request["stream"] = True
        stream = self.client.chat.completions.create(**request)
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            piece = getattr(choices[0].delta, "content", None)
            if piece:
                yield piece

    def chat_stream_response(  # pylint: disable=arguments-differ
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk=None,
        **kwargs: Any,
    ):
        response = self.chat(messages, tools, **kwargs)
        if on_text_chunk is not None and response.content:
            on_text_chunk(response.content)
        return response

    def get_available_models(self) -> list[str]:
        return [
            "grok-3",
            "grok-3-mini",
            "grok-2",
            "grok-2-mini",
        ]


__all__ = ["NativeGrokProvider"]
