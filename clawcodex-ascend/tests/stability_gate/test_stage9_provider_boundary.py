#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""P1 P0 Tests for stage9 provider boundary."""

from __future__ import annotations

from clawcodex_ext.providers.base import BaseProvider, ChatMessage, ChatResponse


class TestStage9ChatResponseBoundary:
    """P1 Tests for TestStage9ChatResponseBoundary."""

    def test_chat_response_empty_content(self):
        """Verify chat response empty content."""
        resp = ChatResponse(content="", model="test-model", usage={}, finish_reason="stop")
        assert resp.content == ""
        assert resp.model == "test-model"
        assert resp.finish_reason == "stop"
        assert resp.tool_uses is None

    def test_chat_response_missing_optional_fields(self):
        """Verify chat response missing optional fields."""
        resp = ChatResponse(content="hello", model="m", usage={}, finish_reason="stop")
        assert resp.reasoning_content is None
        assert resp.tool_uses is None
        assert resp.raw_content_blocks is None

    def test_chat_response_with_tool_uses(self):
        """Verify chat response with tool uses."""
        tool_uses = [
            {"id": "tu_001", "name": "Read", "input": {"file_path": "/tmp/x"}},
        ]
        resp = ChatResponse(
            content="Using tool",
            model="m",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="tool_use",
            tool_uses=tool_uses,
        )
        assert resp.finish_reason == "tool_use"
        assert len(resp.tool_uses) == 1
        assert resp.tool_uses[0]["name"] == "Read"

    def test_chat_response_empty_usage(self):
        """Verify chat response empty usage."""
        resp = ChatResponse(content="x", model="m", usage={}, finish_reason="stop")
        assert resp.usage == {}
        assert resp.usage.get("input_tokens", 0) == 0

    def test_chat_response_zero_tokens(self):
        """Verify chat response zero tokens."""
        resp = ChatResponse(
            content="",
            model="m",
            usage={"input_tokens": 0, "output_tokens": 0},
            finish_reason="stop",
        )
        assert resp.usage["input_tokens"] == 0
        assert resp.usage["output_tokens"] == 0


class TestStage9ChatMessageBoundary:
    """Tests for TestStage9ChatMessageBoundary."""

    def test_chat_message_empty_content(self):
        """Verify chat message empty content."""
        msg = ChatMessage(role="user", content="")
        assert msg.role == "user"
        assert msg.content == ""
        d = msg.to_dict()
        assert d == {"role": "user", "content": ""}

    def test_chat_message_long_content(self):
        """Verify chat message long content."""
        long_text = "x" * 100_000
        msg = ChatMessage(role="user", content=long_text)
        assert len(msg.content) == 100_000


class TestStage9FakeProviderBoundary:
    """P1 Tests for TestStage9FakeProviderBoundary."""

    def test_fake_provider_first_chat_structure(self):
        """Verify fake provider first chat structure."""
        from tests.stability_gate._fake_provider import FakeProvider

        provider = FakeProvider(api_key="test-key")
        resp = provider.chat([{"role": "user", "content": "hello"}])
        assert isinstance(resp, ChatResponse)
        assert resp.content == "Hello from stability gate smoke test."
        assert resp.finish_reason == "stop"
        assert resp.tool_uses is None
        assert resp.usage["input_tokens"] == 5

    def test_fake_provider_second_chat_tool_use(self):
        """Verify fake provider second chat tool use."""
        from tests.stability_gate._fake_provider import FakeProvider

        provider = FakeProvider(api_key="test-key")
        provider.chat([{"role": "user", "content": "first"}])
        resp = provider.chat([{"role": "user", "content": "second"}])
        assert resp.finish_reason == "tool_use"
        assert resp.tool_uses is not None
        assert resp.tool_uses[0]["name"] == "Write"

    def test_write_tool_provider_first_chat(self):
        """Verify write tool provider first chat."""
        from tests.stability_gate._fake_provider import WriteToolProvider

        provider = WriteToolProvider(api_key="test-key")
        resp = provider.chat([{"role": "user", "content": "write it"}])
        assert resp.finish_reason == "tool_use"
        assert resp.tool_uses is not None
        assert resp.tool_uses[0]["name"] == "Write"
        assert "file_path" in resp.tool_uses[0]["input"]

    def test_write_tool_provider_second_chat_stop(self):
        """Verify write tool provider second chat stop."""
        from tests.stability_gate._fake_provider import WriteToolProvider

        provider = WriteToolProvider(api_key="test-key")
        provider.chat([{"role": "user", "content": "first"}])
        resp = provider.chat([{"role": "user", "content": "second"}])
        assert resp.finish_reason == "stop"
        assert resp.content == "File written successfully."


class TestStage9BaseProvider:
    """Tests for TestStage9BaseProvider."""

    def test_prepare_messages_empty(self):
        """Verify prepare messages empty."""

        class _MinimalProvider(BaseProvider):
            def chat(self, messages, tools=None, **kwargs):
                return ChatResponse(content="", model="m", usage={}, finish_reason="stop")

            def chat_stream(self, messages, tools=None, **kwargs):
                return iter(())

            def get_available_models(self):
                return []

        p = _MinimalProvider(api_key="k", base_url="https://example.com", model="m")
        result = p._prepare_messages([])
        assert result == []

    def test_prepare_messages_basic(self):
        """Verify prepare messages basic."""

        class _MinimalProvider(BaseProvider):
            def chat(self, messages, tools=None, **kwargs):
                return ChatResponse(content="", model="m", usage={}, finish_reason="stop")

            def chat_stream(self, messages, tools=None, **kwargs):
                return iter(())

            def get_available_models(self):
                return []

        p = _MinimalProvider(api_key="k", model="m")
        result = p._prepare_messages([ChatMessage(role="user", content="hello")])
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hello"}

    def test_provider_chat_stream_response_not_implemented(self):
        """Verify provider chat stream response not implemented."""

        class _MinimalProvider(BaseProvider):
            def chat(self, messages, tools=None, **kwargs):
                return ChatResponse(content="", model="m", usage={}, finish_reason="stop")

            def chat_stream(self, messages, tools=None, **kwargs):
                return iter(())

            def get_available_models(self):
                return []

        p = _MinimalProvider(api_key="k", model="m")
        import pytest

        with pytest.raises(NotImplementedError):
            p.chat_stream_response([])
