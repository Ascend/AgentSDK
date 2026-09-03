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

"""Tests for the Kimi Coding (Anthropic Messages API) provider."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from clawcodex_ext.providers.base import ChatMessage
from clawcodex_ext.providers.kimi_coding_provider import KimiCodingProvider


class TestKimiCodingProvider(unittest.TestCase):
    """Test Kimi Coding provider."""

    def test_initialization(self):
        """Test provider initialization with defaults."""
        provider = KimiCodingProvider(api_key="test_key")
        self.assertEqual(provider.api_key, "test_key")
        self.assertEqual(provider.model, "kimi-code")
        self.assertEqual(provider.base_url, "https://api.kimi.com/coding")
        self.assertEqual(
            provider._client_kwargs["default_headers"]["User-Agent"],
            "claude-code/0.1.0",
        )

    def test_custom_model(self):
        """Test provider with custom model."""
        provider = KimiCodingProvider(api_key="test_key", model="kimi-code-highspeed")
        self.assertEqual(provider.model, "kimi-code-highspeed")

    def test_custom_base_url(self):
        """Test provider with custom base URL."""
        provider = KimiCodingProvider(
            api_key="test_key",
            base_url="https://kimi-coding.example.com/anthropic",
        )
        self.assertEqual(provider.base_url, "https://kimi-coding.example.com/anthropic")

    def test_get_available_models(self):
        """Test getting available models."""
        provider = KimiCodingProvider(api_key="test_key")
        models = provider.get_available_models()
        self.assertIn("kimi-code", models)

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat(self, mock_anthropic):
        """Test synchronous chat."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello!"
        mock_response.content = [text_block]
        mock_response.model = "kimi-code"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = KimiCodingProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hi")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "kimi-code")
        self.assertEqual(response.finish_reason, "end_turn")
        mock_client.messages.create.assert_called_once()

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_strips_temperature_and_thinking(self, mock_anthropic):
        """Kimi /coding rejects temperature and Anthropic thinking kwarg."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"
        mock_response.content = [text_block]
        mock_response.model = "kimi-code"
        mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = KimiCodingProvider(api_key="test_key")
        provider.chat(
            [ChatMessage(role="user", content="Hi")],
            temperature=0.5,
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertNotIn("temperature", call_kwargs)
        self.assertNotIn("thinking", call_kwargs)

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_stream_response_strips_temperature_and_thinking(self, mock_anthropic):
        """Streaming path also strips temperature and thinking."""
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = False
        mock_stream.__iter__.return_value = iter(
            [SimpleNamespace(delta=SimpleNamespace(text=chunk, thinking=None)) for chunk in ("Hello", " world")]
        )

        final_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello world"
        final_response.content = [text_block]
        final_response.model = "kimi-code"
        final_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        final_response.stop_reason = "end_turn"
        mock_stream.get_final_message.return_value = final_response

        mock_client.messages.stream.return_value = mock_stream
        mock_anthropic.return_value = mock_client

        provider = KimiCodingProvider(api_key="test_key")
        chunks: list[str] = []
        response = provider.chat_stream_response(
            [ChatMessage(role="user", content="Hi")],
            on_text_chunk=chunks.append,
            temperature=0.7,
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        self.assertEqual("".join(chunks), "Hello world")
        self.assertEqual(response.content, "Hello world")
        stream_kwargs = mock_client.messages.stream.call_args.kwargs
        self.assertNotIn("temperature", stream_kwargs)
        self.assertNotIn("thinking", stream_kwargs)


if __name__ == "__main__":
    unittest.main()
