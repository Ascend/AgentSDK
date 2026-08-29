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

"""Tests for the native Grok / xAI adapter (P72-C)."""

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
from src.providers.native.grok_adapter import _DEFAULT_BASE_URL, NativeGrokProvider


def test_default_base_url_points_to_xai() -> None:
    """The plan calls out that the Grok adapter must use the
    xAI endpoint by default — otherwise it would silently target
    OpenAI's API.
    """
    assert "x.ai" in _DEFAULT_BASE_URL


def test_capabilities_match_f72_plan() -> None:
    """Grok advertises the OpenAI-compatible capability set: structured
    output, streaming tools, vision, reasoning. It does NOT claim
    safety_settings (xAI doesn't expose Gemini-style safety config).
    """
    expected = {CAP_STRUCTURED_OUTPUT, CAP_STREAMING_TOOLS, CAP_VISION, CAP_REASONING}
    assert NativeGrokProvider.capabilities == expected


def test_get_provider_name() -> None:
    assert NativeGrokProvider(api_key="k").get_provider_name() == "grok"


def test_get_available_models_lists_grok_families() -> None:
    models = NativeGrokProvider(api_key="k").get_available_models()
    assert "grok-3" in models
    assert "grok-2" in models


def test_construction_uses_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting ``base_url`` must route the SDK at the xAI endpoint."""
    captured: dict[str, object] = {}
    fake_openai = MagicMock()

    def _capture(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    fake_openai.side_effect = _capture
    monkeypatch.setattr("src.providers.native.grok_adapter.OpenAI", fake_openai)
    NativeGrokProvider(api_key="xai-key")
    assert captured["base_url"] == _DEFAULT_BASE_URL
    assert captured["api_key"] == "xai-key"


def test_construction_respects_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    fake_openai = MagicMock()

    def _capture(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    fake_openai.side_effect = _capture
    monkeypatch.setattr("src.providers.native.grok_adapter.OpenAI", fake_openai)
    NativeGrokProvider(api_key="xai-key", base_url="https://proxy.example/v1")
    assert captured["base_url"] == "https://proxy.example/v1"


def test_chat_translates_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Grok adapter must run the same Anthropic→OpenAI
    translation as the native OpenAI adapter, so a Grok request
    that uses an Anthropic-shape message list (the rest of
    clawcodex's default) doesn't blow up at the SDK boundary.
    """
    response = SimpleNamespace(
        model="grok-3",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ack", tool_calls=None),
                finish_reason="stop",
            )
        ],
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    fake_openai = MagicMock(return_value=client)
    monkeypatch.setattr("src.providers.native.grok_adapter.OpenAI", fake_openai)

    provider = NativeGrokProvider(api_key="xai-key")
    result = provider.chat([ChatMessage(role="user", content="ping")])

    assert result.content == "ack"
    assert result.model == "grok-3"
    assert result.usage == {"input_tokens": 4, "output_tokens": 6}
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == [{"role": "user", "content": "ping"}]
    assert call_kwargs["model"] == "grok-3"


def test_chat_stream_yields_text(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = iter(
        [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="He"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="llo"))]),
        ]
    )
    fake_openai = MagicMock(return_value=client)
    monkeypatch.setattr("src.providers.native.grok_adapter.OpenAI", fake_openai)

    provider = NativeGrokProvider(api_key="xai-key")
    chunks = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    assert chunks == ["He", "llo"]
