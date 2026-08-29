#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Tests for the native OpenAI adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from clawcodex_ext.providers.base import ChatMessage
from src.providers.native.capabilities import (
    CAP_REASONING,
    CAP_STREAMING_TOOLS,
    CAP_STRUCTURED_OUTPUT,
    CAP_VISION,
)
from src.providers.native.openai_adapter import (
    NativeOpenAIProvider,
    _safe_json_loads,
)


class _StubChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class _StubUsage:
    prompt_tokens = 11
    completion_tokens = 7


class _StubResponse:
    model = "gpt-5.4"
    usage = _StubUsage()
    choices = [_StubChoice(SimpleNamespace(content="hi", tool_calls=None))]


class _StubStreamChunk:
    def __init__(self, content):
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]


def _make_provider(monkeypatch: pytest.MonkeyPatch) -> tuple[NativeOpenAIProvider, MagicMock]:
    client_mock = MagicMock()
    client_mock.chat.completions.create.return_value = _StubResponse()
    fake_openai = MagicMock()
    fake_openai.return_value = client_mock
    monkeypatch.setattr("src.providers.native.openai_adapter.OpenAI", fake_openai)
    provider = NativeOpenAIProvider(api_key="sk-test", base_url=None, model="gpt-5.4")
    return provider, client_mock


def test_capabilities_match_f72_plan() -> None:
    """The OpenAI adapter must declare exactly the capabilities the
    adapter contract attributes to it (structured output, vision, streaming
    tools, reasoning).
    """
    expected = {CAP_STRUCTURED_OUTPUT, CAP_STREAMING_TOOLS, CAP_VISION, CAP_REASONING}
    assert NativeOpenAIProvider.capabilities == expected


def test_get_provider_name() -> None:
    assert NativeOpenAIProvider(api_key="k").get_provider_name() == "openai"


def test_get_available_models_is_non_empty() -> None:
    """The model list must include both the GPT-5.4 family and the
    legacy GPT-4 line so existing callers don't regress.
    """
    models = NativeOpenAIProvider(api_key="k").get_available_models()
    assert "gpt-5.4" in models
    assert "gpt-4o" in models
    assert "gpt-3.5-turbo" in models


def test_chat_returns_chat_response(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client = _make_provider(monkeypatch)
    response = provider.chat(
        [ChatMessage(role="user", content="hello")],
    )
    assert response.content == "hi"
    assert response.model == "gpt-5.4"
    assert response.finish_reason == "stop"
    assert response.usage == {"input_tokens": 11, "output_tokens": 7}
    client.chat.completions.create.assert_called_once()
    # The Anthropic-style message was translated to OpenAI's
    # ``{"role": ..., "content": ...}`` shape.
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert call_kwargs["model"] == "gpt-5.4"


def test_chat_forwards_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native path must surface ``response_format`` so callers
    can request JSON-schema structured output — the headline
    capability that LiteLLM generalises away.
    """
    provider, client = _make_provider(monkeypatch)
    provider.chat(
        [ChatMessage(role="user", content="hi")],
        response_format={"type": "json_schema", "json_schema": {"name": "X"}},
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "X"},
    }


def test_chat_stream_yields_content_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client = _make_provider(monkeypatch)
    client.chat.completions.create.return_value = iter(
        [
            _StubStreamChunk("Hel"),
            _StubStreamChunk("lo"),
            _StubStreamChunk(None),
        ]
    )
    chunks = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    assert chunks == ["Hel", "lo"]


def test_chat_stream_response_includes_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """The non-streaming path's convenience wrapper must call the
    text callback once with the full content.
    """
    provider, _ = _make_provider(monkeypatch)
    captured: list[str] = []
    response = provider.chat_stream_response(
        [ChatMessage(role="user", content="hi")],
        on_text_chunk=captured.append,
    )
    assert response.content == "hi"
    assert captured == ["hi"]


def test_safe_json_loads_parses_dict() -> None:
    assert _safe_json_loads('{"x": 1}') == {"x": 1}


def test_safe_json_loads_handles_empty() -> None:
    assert _safe_json_loads("") == {}


def test_safe_json_loads_handles_malformed() -> None:
    assert _safe_json_loads("{not json") == {"_raw": "{not json"}


def test_safe_json_loads_wraps_non_object() -> None:
    """A bare list/array is not a valid tool input; surface it as
    ``{"_raw": ...}`` rather than rejecting the call.
    """
    result = _safe_json_loads("[1, 2, 3]")
    assert result == {"_raw": [1, 2, 3]}


def test_construction_raises_when_openai_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the SDK is unavailable at construction time the
    adapter must raise ``ModuleNotFoundError`` rather than
    silently producing a broken client. We simulate the
    missing-SDK case by binding ``OpenAI`` to ``None`` in the
    module's namespace — same observable behaviour as
    ``openai`` not being installed.
    """
    monkeypatch.setattr("src.providers.native.openai_adapter.OpenAI", None)
    with pytest.raises(ModuleNotFoundError):
        NativeOpenAIProvider(api_key="k")
