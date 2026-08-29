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

"""Discord channel tests: payload shape and HTTP semantics."""

from __future__ import annotations

import json

import pytest

from src.services.channels import (
    ChannelConfig,
    ChannelMessage,
    ChannelType,
    DiscordChannel,
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


def _config(name: str = "discord-1") -> ChannelConfig:
    return ChannelConfig(
        type=ChannelType.DISCORD,
        webhook_url="https://discord.com/api/webhooks/123456/abcdef0123456789",
        name=name,
    )


def test_format_content_without_title() -> None:
    transport = _FakeTransport()
    channel = DiscordChannel(_config(), transport=transport)
    body, headers = channel.format_message(ChannelMessage(text="hi"))
    payload = json.loads(body.decode("utf-8"))
    assert payload == {"content": "hi"}
    assert "embeds" not in payload
    assert headers["Content-Type"] == "application/json"


def test_format_content_with_title() -> None:
    transport = _FakeTransport()
    channel = DiscordChannel(_config(), transport=transport)
    body, _ = channel.format_message(ChannelMessage(text="body", title="Headline", markdown=True))
    payload = json.loads(body.decode("utf-8"))
    assert payload["content"] == "**Headline**\nbody"


def test_format_embeds_when_attachments_present() -> None:
    transport = _FakeTransport()
    channel = DiscordChannel(_config(), transport=transport)
    body, _ = channel.format_message(
        ChannelMessage(
            text="with embeds",
            attachments=[{"title": "link", "url": "https://example.com"}],
        )
    )
    payload = json.loads(body.decode("utf-8"))
    assert payload["content"] == "with embeds"
    assert payload["embeds"] == [{"title": "link", "url": "https://example.com"}]


def test_format_unicode_safe() -> None:
    transport = _FakeTransport()
    channel = DiscordChannel(_config(), transport=transport)
    body, _ = channel.format_message(ChannelMessage(text="你好"))
    payload = json.loads(body.decode("utf-8"))
    assert payload["content"] == "你好"


@pytest.mark.asyncio
async def test_send_returns_true_on_2xx_empty_body() -> None:
    transport = _FakeTransport(status=204, body=b"")
    channel = DiscordChannel(_config(), transport=transport)
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is True


@pytest.mark.asyncio
async def test_send_returns_false_on_4xx() -> None:
    transport = _FakeTransport(status=400, body=b'{"code": 10008, "message": "Unknown Channel"}')
    channel = DiscordChannel(_config(), transport=transport)
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is False


@pytest.mark.asyncio
async def test_send_returns_false_on_error_envelope() -> None:
    transport = _FakeTransport(
        status=200,
        body=json.dumps({"code": 10008, "message": "Unknown Channel"}).encode("utf-8"),
    )
    channel = DiscordChannel(_config(), transport=transport)
    ok = await channel.send(ChannelMessage(text="hi"))
    # Discord returns 200 with a non-zero code envelope to indicate failure.
    assert ok is False


@pytest.mark.asyncio
async def test_send_returns_true_on_envelope_with_zero_code() -> None:
    transport = _FakeTransport(status=200, body=json.dumps({"code": 0, "message": "ok"}).encode("utf-8"))
    channel = DiscordChannel(_config(), transport=transport)
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is True


@pytest.mark.asyncio
async def test_send_uses_configured_webhook_url() -> None:
    transport = _FakeTransport(status=204, body=b"")
    channel = DiscordChannel(_config("discord-x"), transport=transport)
    await channel.send(ChannelMessage(text="hi"))
    assert transport.calls[0]["url"].startswith("https://discord.com/api/webhooks/")


@pytest.mark.asyncio
async def test_send_ignores_invalid_json_envelope() -> None:
    # If body is not JSON, the 2xx status alone is enough.
    transport = _FakeTransport(status=204, body=b"<html>oops</html>")
    channel = DiscordChannel(_config(), transport=transport)
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is True


@pytest.mark.asyncio
async def test_send_records_timeout_in_call() -> None:
    transport = _FakeTransport(status=204, body=b"")
    channel = DiscordChannel(_config(), transport=transport)
    await channel.send(ChannelMessage(text="hi"))
    assert transport.calls[0]["timeout"] > 0
