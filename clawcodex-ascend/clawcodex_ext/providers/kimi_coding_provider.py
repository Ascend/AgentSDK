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

"""Kimi Coding provider — Anthropic Messages API endpoint.

``sk-kimi-*`` keys for Kimi's coding plan route through
``https://api.kimi.com/coding`` and speak the Anthropic Messages API,
not the OpenAI chat-completions protocol.  This provider subclasses
``ClawcodexAnthropicProvider`` so it inherits the cancel-latency
bound and cache-state handling, while adding the endpoint-specific
quirks required by Kimi:

* ``User-Agent: claude-code/0.1.0`` is required, otherwise the endpoint
  returns 403.
* The native Anthropic ``thinking`` kwarg is not accepted and must be
  stripped before the request is sent.
* ``temperature`` is managed by the endpoint, so we omit it.
"""

from __future__ import annotations

from typing import Any, Generator, Optional

from clawcodex_ext.providers.anthropic_provider import ClawcodexAnthropicProvider
from clawcodex_ext.providers.base import ChatResponse, MessageInput, TextChunkCallback


class KimiCodingProvider(ClawcodexAnthropicProvider):
    """Kimi Coding provider using the Anthropic SDK against api.kimi.com/coding."""

    DEFAULT_BASE_URL = "https://api.kimi.com/coding"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize Kimi Coding provider.

        Args:
            api_key: Kimi API key (``sk-kimi-*``).
            base_url: Base URL (optional, defaults to https://api.kimi.com/coding).
            model: Default model (default: ``kimi-code``).
        """
        super().__init__(
            api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            model=model or "kimi-code",
        )
        # The /coding endpoint refuses requests without a Claude Code user-agent.
        self._client_kwargs.setdefault("default_headers", {})
        self._client_kwargs["default_headers"]["User-Agent"] = "claude-code/0.1.0"

    @staticmethod
    def _apply_kimi_coding_quirks(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Apply Kimi /coding request quirks.

        - Drop ``temperature`` — Kimi manages sampling internally.
        - Drop ``thinking`` — the Anthropic Messages API ``thinking`` kwarg
          is not accepted by Kimi's /coding endpoint.
        """
        return {k: v for k, v in kwargs.items() if k not in ("temperature", "thinking")}

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        kwargs = self._apply_kimi_coding_quirks(kwargs)
        return super().chat(messages, tools=tools, **kwargs)

    def chat_stream(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        kwargs = self._apply_kimi_coding_quirks(kwargs)
        return super().chat_stream(messages, tools=tools, **kwargs)

    def chat_stream_response(  # pylint: disable=arguments-differ
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk: TextChunkCallback | None = None,
        abort_signal: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        kwargs = self._apply_kimi_coding_quirks(kwargs)
        return super().chat_stream_response(
            messages,
            tools=tools,
            on_text_chunk=on_text_chunk,
            abort_signal=abort_signal,
            **kwargs,
        )

    def get_available_models(self) -> list[str]:
        """Return Kimi Coding models."""
        return [
            "kimi-code",
        ]
