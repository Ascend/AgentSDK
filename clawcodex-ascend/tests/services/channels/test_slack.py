#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Slack channel tests: payload shape and HTTP semantics."""

from __future__ import annotations

import json

import pytest

from src.services.channels import (
    ChannelConfig,
    ChannelMessage,
    ChannelType,
    SlackChannel,
)


class _FakeResponse:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status = status
        self.body = body


class _FakeTransport:
    def __init__(self, status: int = 200, body: bytes = b"") -> None:
        self.status = status
        self.body = body
        self.calls: list[dict] = []

    async def post(self, url, body, *, headers=None, timeout=10.0):
        self.calls.append({"url": url, "body": body, "headers": dict(headers or {}), "timeout": timeout})
        return _FakeResponse(self.status, self.body)


def _config(name: str = "slack-1") -> ChannelConfig:
    return ChannelConfig(
        type=ChannelType.SLACK,
        webhook_url="https://hooks.slack.com/services/T0000/B0000/abcdef0123456789",
        name=name,
    )


def test_format_plain_text_payload() -> None:
    transport = _FakeTransport()
    channel = SlackChannel(_config(), transport=transport)
    body, headers = channel.format_message(ChannelMessage(text="hello"))
    payload = json.loads(body.decode("utf-8"))
    assert payload == {"text": "hello"}
    assert headers["Content-Type"] == "application/json"


def test_format_blocks_payload_when_markdown_and_title() -> None:
    transport = _FakeTransport()
    channel = SlackChannel(_config(), transport=transport)
    body, _ = channel.format_message(ChannelMessage(text="body text", title="Headline", markdown=True))
    payload = json.loads(body.decode("utf-8"))
    assert payload["text"] == "Headline"
    blocks = payload["blocks"]
    assert len(blocks) == 1
    section = blocks[0]
    assert section["type"] == "section"
    assert section["text"]["type"] == "mrkdwn"
    assert "Headline" in section["text"]["text"]
    assert "body text" in section["text"]["text"]


def test_format_text_payload_when_markdown_false() -> None:
    transport = _FakeTransport()
    channel = SlackChannel(_config(), transport=transport)
    body, _ = channel.format_message(ChannelMessage(text="hi", title="T", markdown=False))
    payload = json.loads(body.decode("utf-8"))
    # Markdown disabled falls back to plain text.
    assert payload == {"text": "hi"}


def test_format_payload_unicode_safe() -> None:
    transport = _FakeTransport()
    channel = SlackChannel(_config(), transport=transport)
    body, _ = channel.format_message(ChannelMessage(text="你好 world"))
    assert "你好 world".encode("utf-8") in body


@pytest.mark.asyncio
async def test_send_returns_true_on_http_200_empty_body() -> None:
    transport = _FakeTransport(status=200, body=b"")
    channel = SlackChannel(_config(), transport=transport)
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is True


@pytest.mark.asyncio
async def test_send_returns_true_on_ok_true_body() -> None:
    transport = _FakeTransport(status=204, body=b'{"ok": true}')
    channel = SlackChannel(_config(), transport=transport)
    # Even with non-200 status, an ok=true envelope must count as success.
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is True


@pytest.mark.asyncio
async def test_send_returns_false_on_http_error() -> None:
    transport = _FakeTransport(status=500, body=b"oops")
    channel = SlackChannel(_config(), transport=transport)
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is False


@pytest.mark.asyncio
async def test_send_returns_false_on_ok_false_body() -> None:
    transport = _FakeTransport(status=200, body=b'{"ok": false, "error": "invalid_auth"}')
    channel = SlackChannel(_config(), transport=transport)
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is False


@pytest.mark.asyncio
async def test_send_ignores_invalid_json_response() -> None:
    # If body is not JSON, status code 200 alone is enough to succeed.
    transport = _FakeTransport(status=200, body=b"<html>")
    channel = SlackChannel(_config(), transport=transport)
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is True


@pytest.mark.asyncio
async def test_send_uses_configured_webhook_url() -> None:
    transport = _FakeTransport(status=200, body=b"")
    channel = SlackChannel(_config("slack-x"), transport=transport)
    await channel.send(ChannelMessage(text="hi"))
    assert transport.calls[0]["url"].startswith("https://hooks.slack.com/")


def test_message_level_does_not_affect_payload() -> None:
    transport = _FakeTransport()
    channel = SlackChannel(_config(), transport=transport)
    body, _ = channel.format_message(
        ChannelMessage(text="hi", title="T", level="error")  # type: ignore[arg-type]
    )
    payload = json.loads(body.decode("utf-8"))
    assert "level" not in payload
