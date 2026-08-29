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

# pylint: disable=ungrouped-imports
"""Minimax AI provider (Anthropic-compatible) — clawcodex_ext canonical implementation.

Contains both:

* :class:`MinimaxProvider` — the base provider.  Minimax exposes an
  Anthropic-compatible endpoint at ``api.minimaxi.com/anthropic`` and
  is driven by the ``anthropic`` SDK.  The base implementation handles
  the lazy SDK import (degrades to a stub if the optional dependency
  is missing) and the standard chat / chat_stream / chat_stream_response
  surfaces.
* :class:`ClawcodexMinimaxProvider` — subclass that overrides
  :meth:`chat_stream_response` to route the synchronous ``text_stream``
  iteration through :func:`clawcodex_ext.providers._stream_drain.
  drain_text_stream_with_abort_poll`.  This bounds cancel latency to
  ~100ms on platforms where ``response.close()`` from another thread is
  advisory only and does NOT interrupt the blocking httpx read.

Registered in :mod:`clawcodex_ext.providers` via
``register_provider("minimax", …)``; the hook in
``src.providers.get_provider_class`` makes the override win over the
upstream hardcoded branch.

Backward-compat: ``src.providers.minimax_provider`` is a
``sys.modules`` swap facade that resolves to this module, so test
imports like ``from src.providers.minimax_provider import
MinimaxProvider`` continue to work transparently.
"""

from __future__ import annotations

from typing import Any, Generator, Optional, TYPE_CHECKING

from clawcodex_ext.providers._stream_drain import drain_text_stream_with_abort_poll

try:
    import anthropic  # type: ignore
except ModuleNotFoundError:  # pragma: no cover

    class _MissingAnthropic:
        class Anthropic:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs):
                raise ModuleNotFoundError(
                    "anthropic package is not installed. Install optional dependencies to use MinimaxProvider."
                )

    anthropic = _MissingAnthropic()

from clawcodex_ext.providers.base import (
    BaseProvider,
    ChatResponse,
    MessageInput,
    TextChunkCallback,
    ThinkingChunkCallback,
)

if TYPE_CHECKING:
    from src.utils.abort_controller import AbortSignal


def _usage_token_count(usage: Any, field: str) -> int:
    value = getattr(usage, field, 0)
    return int(value) if isinstance(value, (int, float)) else 0


class MinimaxProvider(BaseProvider):
    """Minimax AI provider using Anthropic-compatible API.

    Minimax provides an Anthropic-compatible endpoint at api.minimax.io/anthropic.
    Uses the Anthropic SDK with Minimax-specific models.
    """

    DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None):
        """Initialize Minimax provider.

        Args:
            api_key: Minimax API key
            base_url: Base URL (optional, defaults to Minimax Anthropic-compatible endpoint)
            model: Default model (default: MiniMax-M3)
        """
        resolved_base_url = base_url or self.DEFAULT_BASE_URL
        super().__init__(api_key, resolved_base_url, model or "MiniMax-M3")

        self._client_kwargs: dict[str, Any] = {"api_key": api_key}
        if resolved_base_url:
            self._client_kwargs["base_url"] = resolved_base_url
        self.client = None

    def _ensure_client(self):
        if self.client is not None:
            return self.client
        self.client = anthropic.Anthropic(**self._client_kwargs)
        return self.client

    def _prepare_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Base preparation + removal of foreign passthrough blocks.

        Minimax speaks the Anthropic wire format, which rejects unknown
        content-block types — a mid-session ``/model`` switch away from
        the ChatGPT-subscription provider must not leak its
        ``openai_responses_item`` replay blocks here. Same strip as
        ``AnthropicProvider._prepare_messages``.
        """
        prepared = super()._prepare_messages(messages)
        from .openai_responses import strip_responses_item_blocks  # pylint: disable=no-name-in-module

        return strip_responses_item_blocks(prepared)

    def _build_chat_response(
        self,
        response: Any,
        *,
        request_service_tier: str = "standard",
    ) -> ChatResponse:
        content_text = ""
        tool_uses: list[dict[str, Any]] = []

        for block in response.content:
            block_type = getattr(block, "type", "text")
            if block_type == "text":
                text_val = getattr(block, "text", "")
                if text_val is not None:
                    content_text += str(text_val)
            elif block_type == "tool_use":
                tool_uses.append(
                    {
                        "id": str(getattr(block, "id", "")),
                        "name": str(getattr(block, "name", "")),
                        "input": dict(getattr(block, "input", {})),
                    }
                )

        usage = getattr(response, "usage", None)
        response_service_tier = getattr(usage, "service_tier", None)
        service_tier = (
            response_service_tier if response_service_tier in ("standard", "priority") else request_service_tier
        )
        return ChatResponse(
            content=content_text,
            model=getattr(response, "model", self.model or ""),
            usage={
                "input_tokens": _usage_token_count(usage, "input_tokens"),
                "output_tokens": _usage_token_count(usage, "output_tokens"),
                "cache_creation_input_tokens": _usage_token_count(usage, "cache_creation_input_tokens"),
                "cache_read_input_tokens": _usage_token_count(usage, "cache_read_input_tokens"),
                "service_tier": service_tier,
            },
            finish_reason=str(getattr(response, "stop_reason", "stop")),
            tool_uses=tool_uses if tool_uses else None,
        )

    def chat(
        self, messages: list[MessageInput], tools: Optional[list[dict[str, Any]]] = None, **kwargs
    ) -> ChatResponse:
        """Synchronous chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas
            **kwargs: Additional parameters

        Returns:
            Chat response
        """
        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)
        request_service_tier = "priority" if kwargs.get("service_tier") == "priority" else "standard"

        system = kwargs.pop("system", None)

        # Convert messages
        minimax_messages = self._prepare_messages(messages)

        # Make API call
        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=minimax_messages,
            **({"system": system} if system else {}),
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
        )

        return self._build_chat_response(
            response,
            request_service_tier=request_service_tier,
        )

    def chat_stream(
        self, messages: list[MessageInput], tools: Optional[list[dict[str, Any]]] = None, **kwargs
    ) -> Generator[str, None, None]:
        """Streaming chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas
            **kwargs: Additional parameters

        Yields:
            Chunks of response content
        """
        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)

        # Convert messages
        minimax_messages = self._prepare_messages(messages)

        # Stream API call
        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=minimax_messages,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
        ) as stream:
            yield from stream.text_stream

    def chat_stream_response(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk: TextChunkCallback | None = None,
        on_thinking_chunk: ThinkingChunkCallback | None = None,
        abort_signal: Any = None,
        **kwargs,
    ) -> ChatResponse:
        """Stream Minimax response with abort-signal-aware cancellation.

        Minimax wraps the anthropic SDK against its compatible
        endpoint, so the same response-close listener pattern
        AnthropicProvider uses works here too. The bookkeeping lives
        in ``StreamAbortGuard``; this provider only owns the
        SDK-specific iteration shape (``with client.messages.stream``
        + ``stream.text_stream`` + ``get_final_message``).

        ``on_thinking_chunk`` is accepted for the common provider
        interface.  The Anthropic-compatible MiniMax stream currently
        exposes text only, so there is no thinking delta to emit.
        """
        from clawcodex_ext.providers._stream_abort import StreamAbortGuard

        guard = StreamAbortGuard(abort_signal)
        guard.raise_if_pre_aborted()

        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)
        request_service_tier = "priority" if kwargs.get("service_tier") == "priority" else "standard"
        system = kwargs.pop("system", None)
        minimax_messages = self._prepare_messages(messages)

        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        streamed_text = ""
        final_message: Any = None
        try:
            with (
                client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    messages=minimax_messages,
                    **({"system": system} if system else {}),
                    **extra_kwargs,
                    **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
                ) as stream,
                guard.attach(stream),
            ):
                for text in stream.text_stream:
                    if not text:
                        continue
                    streamed_text += text
                    if on_text_chunk is not None:
                        on_text_chunk(text)
                try:
                    final_message = stream.get_final_message()
                except Exception:
                    final_message = None
        except Exception as streaming_exc:
            guard.reraise_if_aborted(streaming_exc)
            raise

        # Stream exited normally but abort may have fired between
        # ``__exit__`` and here.
        guard.raise_if_post_aborted()

        if final_message is not None:
            return self._build_chat_response(
                final_message,
                request_service_tier=request_service_tier,
            )

        return ChatResponse(
            content=streamed_text,
            model=model,
            usage={},
            finish_reason="stop",
            tool_uses=None,
        )

    def get_available_models(self) -> list[str]:
        """Get list of available Minimax models.

        Returns:
            List of model names
        """
        return [
            "MiniMax-M3",
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
            "M2-her",
            # Historical
            "MiniMax-M2.1",
            "MiniMax-M2.1-highspeed",
            "MiniMax-M2",
        ]


class ClawcodexMinimaxProvider(MinimaxProvider):
    """Minimax AI provider (Anthropic-compatible) with cancel-latency fix.

    Inherits ``__init__``, ``_ensure_client``, ``_get_model``,
    ``_prepare_messages``, ``_build_chat_response``, and ``chat`` from
    the parent. Overrides ``chat_stream_response`` to route the
    synchronous ``text_stream`` iteration through
    :func:`drain_text_stream_with_abort_poll`.
    """

    def chat_stream_response(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk: TextChunkCallback | None = None,
        on_thinking_chunk: ThinkingChunkCallback | None = None,
        abort_signal: "AbortSignal | None" = None,
        **kwargs,
    ) -> ChatResponse:
        from clawcodex_ext.providers._stream_abort import StreamAbortGuard

        guard = StreamAbortGuard(abort_signal)
        guard.raise_if_pre_aborted()

        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)
        system = kwargs.pop("system", None)
        minimax_messages = self._prepare_messages(messages)

        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        streamed_text_parts: list[str] = []

        def _on_text(text: str) -> None:
            streamed_text_parts.append(text)
            if on_text_chunk is not None:
                on_text_chunk(text)

        final_message: Any = None
        try:
            with (
                client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    messages=minimax_messages,
                    **({"system": system} if system else {}),
                    **extra_kwargs,
                    **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
                ) as stream,
                guard.attach(stream),
            ):
                # The only behavioral change vs. the parent: route
                # the synchronous ``text_stream`` iteration through
                # the worker-thread + 100ms queue-poll drain so a
                # user cancel lands within ~100ms instead of the
                # worst-case HTTP read timeout.
                drain_text_stream_with_abort_poll(
                    stream,
                    guard=guard,
                    on_text=_on_text,
                    stream_name="minimax-stream",
                )
                try:
                    final_message = stream.get_final_message()
                except Exception:
                    final_message = None
        except Exception as streaming_exc:
            guard.reraise_if_aborted(streaming_exc)
            raise

        # Stream exited normally but abort may have fired between
        # ``__exit__`` and here.
        guard.raise_if_post_aborted()

        streamed_text = "".join(streamed_text_parts)

        if final_message is not None:
            return self._build_chat_response(final_message)

        return ChatResponse(
            content=streamed_text,
            model=model,
            usage={},
            finish_reason="stop",
            tool_uses=None,
        )


__all__ = ["MinimaxProvider", "ClawcodexMinimaxProvider"]
