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

"""Tests for the Kimi (Moonshot AI) provider."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from clawcodex_ext.providers.base import ChatMessage
from clawcodex_ext.providers.kimi_provider import KimiProvider


class TestKimiProvider(unittest.TestCase):
    """Test Kimi provider."""

    def test_initialization(self):
        """Test provider initialization with defaults."""
        provider = KimiProvider(api_key="test_key")
        self.assertEqual(provider.api_key, "test_key")
        self.assertEqual(provider.model, "kimi-k2.6")
        self.assertEqual(provider.base_url, "https://api.moonshot.ai/v1")

    def test_custom_model(self):
        """Test provider with custom model."""
        provider = KimiProvider(api_key="test_key", model="kimi-k2.5")
        self.assertEqual(provider.model, "kimi-k2.5")

    def test_custom_base_url(self):
        """Test provider with custom base URL."""
        provider = KimiProvider(
            api_key="test_key",
            base_url="https://kimi.example.com/v1",
        )
        self.assertEqual(provider.base_url, "https://kimi.example.com/v1")

    def test_get_available_models(self):
        """Test getting available models."""
        provider = KimiProvider(api_key="test_key")
        models = provider.get_available_models()
        self.assertIn("kimi-k2.6", models)
        self.assertIn("kimi-k2.5", models)
        self.assertIn("moonshot-v1-8k", models)
        self.assertIn("moonshot-v1-128k-vision-preview", models)

    @patch("clawcodex_ext.providers.kimi_provider.OpenAI")
    def test_create_client(self, mock_openai):
        """Test OpenAI SDK client creation."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        provider = KimiProvider(api_key="test_key")
        client = provider._create_client()

        self.assertIs(client, mock_client)
        mock_openai.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.moonshot.ai/v1",
            timeout=60.0,
        )

    @patch.dict(os.environ, {"CLAWCODEX_SSL_VERIFY": "false"}, clear=False)
    @patch("clawcodex_ext.providers.kimi_provider.OpenAI")
    @patch("httpx.Client")
    def test_create_client_with_ssl_bypass(self, mock_httpx_client, mock_openai):
        """Test SSL verification bypass when CLAWCODEX_SSL_VERIFY=false."""
        mock_insecure_client = MagicMock()
        mock_httpx_client.return_value = mock_insecure_client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        provider = KimiProvider(api_key="test_key")
        client = provider._create_client()

        self.assertIs(client, mock_client)
        mock_httpx_client.assert_called_once_with(verify=False, timeout=60.0)
        mock_openai.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.moonshot.ai/v1",
            timeout=60.0,
            http_client=mock_insecure_client,
        )

    @patch("clawcodex_ext.providers.kimi_provider.OpenAI")
    def test_chat(self, mock_openai):
        """Test synchronous chat."""
        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.model = "kimi-k2.6"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = KimiProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hi")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "kimi-k2.6")
        self.assertEqual(response.usage["total_tokens"], 15)
        self.assertEqual(response.finish_reason, "stop")

    @patch("clawcodex_ext.providers.kimi_provider.OpenAI")
    def test_chat_accepts_dict_messages(self, mock_openai):
        """Test synchronous chat with dict messages."""
        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.model = "kimi-k2.6"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = KimiProvider(api_key="test_key")
        messages = [{"role": "user", "content": "Hi"}]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        mock_client.chat.completions.create.assert_called_once()
        self.assertEqual(
            mock_client.chat.completions.create.call_args.kwargs["messages"],
            messages,
        )

    @patch("clawcodex_ext.providers.kimi_provider.OpenAI")
    def test_chat_stream_response_rebuilds_tool_calls(self, mock_openai):
        """Streaming chunks are rebuilt into a final response with tool calls."""
        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.model = "kimi-k2.6"
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].finish_reason = None
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.reasoning_content = None
        chunk1.choices[0].delta.tool_calls = []

        tool_call_delta = MagicMock()
        tool_call_delta.index = 0
        tool_call_delta.id = "call_1"
        tool_call_delta.function = MagicMock(name="function")
        tool_call_delta.function.name = "Read"
        tool_call_delta.function.arguments = '{"file_path":"README.md"}'

        chunk2 = MagicMock()
        chunk2.model = "kimi-k2.6"
        chunk2.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].finish_reason = "tool_calls"
        chunk2.choices[0].delta.content = None
        chunk2.choices[0].delta.reasoning_content = None
        chunk2.choices[0].delta.tool_calls = [tool_call_delta]

        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])
        mock_openai.return_value = mock_client

        provider = KimiProvider(api_key="test_key")
        chunks: list[str] = []
        response = provider.chat_stream_response(
            [ChatMessage(role="user", content="Hi")],
            tools=[{"name": "Read", "description": "", "input_schema": {"type": "object"}}],
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "Hello")
        self.assertEqual(response.content, "Hello")
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_uses[0]["name"], "Read")
        self.assertEqual(response.usage["total_tokens"], 15)

    def test_apply_kimi_request_quirks_drops_temperature(self):
        """Temperature is omitted from Moonshot requests."""
        provider = KimiProvider(api_key="test_key")
        quirks = provider._apply_kimi_request_quirks({"temperature": 0.5})
        self.assertNotIn("temperature", quirks)

    def test_apply_kimi_request_quirks_default_thinking_enabled(self):
        """Default thinking is enabled via extra_body when no reasoning config."""
        provider = KimiProvider(api_key="test_key")
        quirks = provider._apply_kimi_request_quirks({})
        self.assertEqual(quirks["extra_body"]["thinking"], {"type": "enabled"})

    def test_apply_kimi_request_quirks_reasoning_effort(self):
        """Recognized effort levels are sent as reasoning_effort."""
        provider = KimiProvider(api_key="test_key")
        quirks = provider._apply_kimi_request_quirks({"reasoning_config": {"enabled": True, "effort": "high"}})
        self.assertEqual(quirks["reasoning_effort"], "high")
        self.assertNotIn("extra_body", quirks)

    def test_apply_kimi_request_quirks_thinking_disabled(self):
        """Disabled reasoning_config emits disabled thinking block."""
        provider = KimiProvider(api_key="test_key")
        quirks = provider._apply_kimi_request_quirks({"reasoning_config": {"enabled": False}})
        self.assertEqual(quirks["extra_body"]["thinking"], {"type": "disabled"})


if __name__ == "__main__":
    unittest.main()
