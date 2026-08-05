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

"""Feishu channel tests: HMAC signing, payload shape, mocked transport."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest

from src.services.channels import (
    ChannelConfig,
    ChannelMessage,
    ChannelType,
    FeishuChannel,
    MessageLevel,
    WebhookSecretMissingError,
    sign_feishu,
)
from src.services.channels.feishu import FEISHU_SUCCESS_CODE


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


def _config(name: str = "feishu-1", secret: str | None = None) -> ChannelConfig:
    return ChannelConfig(
        type=ChannelType.FEISHU,
        webhook_url="https://open.feishu.cn/hook/abcdef0123456789",
        name=name,
        extra={"secret": secret} if secret else None,
    )


def test_sign_feishu_matches_reference_vector() -> None:
    # Reference vector from the Feishu docs:
    # https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN
    secret = "secret_value"
    timestamp = "1620000000"
    string_to_sign = f"{timestamp}\n{secret}"
    expected = base64.b64encode(
        hmac.new(
            secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    assert sign_feishu(secret, timestamp) == expected


def test_sign_feishu_rejects_empty_secret() -> None:
    with pytest.raises(WebhookSecretMissingError):
        sign_feishu("", "123")
    with pytest.raises(WebhookSecretMissingError):
        sign_feishu(None, "123")  # type: ignore[arg-type]


def test_format_text_payload_without_title() -> None:
    transport = _FakeTransport()
    channel = FeishuChannel(_config("f1", secret="s3cret"), transport=transport)  # type: ignore[arg-type]
    body, headers = channel.format_message(ChannelMessage(text="hello"))
    payload = json.loads(body.decode("utf-8"))
    assert payload["msg_type"] == "text"
    assert json.loads(payload["content"]) == {"text": "hello"}
    assert "timestamp" in payload
    assert "sign" in payload
    assert headers["Content-Type"] == "application/json"


def test_format_interactive_payload_with_title_and_markdown() -> None:
    transport = _FakeTransport()
    channel = FeishuChannel(_config("f1", secret="s3cret"), transport=transport)  # type: ignore[arg-type]
    body, _ = channel.format_message(ChannelMessage(text="body", title="Headline", markdown=True))
    payload = json.loads(body.decode("utf-8"))
    assert payload["msg_type"] == "interactive"
    card = payload["card"]
    assert card["header"]["title"]["content"] == "Headline"
    assert card["elements"][0]["content"] == "body"


def test_format_text_payload_without_secret_skips_signing() -> None:
    transport = _FakeTransport()
    channel = FeishuChannel(_config("f1"), transport=transport)
    body, _ = channel.format_message(ChannelMessage(text="hi"))
    payload = json.loads(body.decode("utf-8"))
    assert "timestamp" not in payload
    assert "sign" not in payload


def test_format_uses_unicode_safe_encoding() -> None:
    transport = _FakeTransport()
    channel = FeishuChannel(_config("f1", secret="x"), transport=transport)  # type: ignore[arg-type]
    body, _ = channel.format_message(ChannelMessage(text="你好"))
    payload = json.loads(body.decode("utf-8"))
    assert payload["msg_type"] == "text"
    assert json.loads(payload["content"]) == {"text": "你好"}


@pytest.mark.asyncio
async def test_send_returns_true_on_code_zero() -> None:
    transport = _FakeTransport(status=200, body=json.dumps({"code": FEISHU_SUCCESS_CODE, "msg": "ok"}).encode("utf-8"))
    channel = FeishuChannel(_config("f1"), transport=transport)  # type: ignore[arg-type]
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is True


@pytest.mark.asyncio
async def test_send_returns_false_on_nonzero_code() -> None:
    transport = _FakeTransport(status=200, body=json.dumps({"code": 999, "msg": "bad"}).encode("utf-8"))
    channel = FeishuChannel(_config("f1"), transport=transport)  # type: ignore[arg-type]
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is False


@pytest.mark.asyncio
async def test_send_returns_false_on_non_200_status() -> None:
    transport = _FakeTransport(status=500, body=b"server error")
    channel = FeishuChannel(_config("f1"), transport=transport)  # type: ignore[arg-type]
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is False


@pytest.mark.asyncio
async def test_send_transport_call_includes_url_body_and_timeout() -> None:
    transport = _FakeTransport(status=200, body=b'{"code":0}')
    channel = FeishuChannel(_config("f1"), transport=transport)  # type: ignore[arg-type]
    await channel.send(ChannelMessage(text="hi"))
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"].startswith("https://open.feishu.cn/hook/")
    assert json.loads(call["body"].decode("utf-8"))["msg_type"] == "text"
    assert call["timeout"] > 0


@pytest.mark.asyncio
async def test_send_treats_empty_body_as_success() -> None:
    transport = _FakeTransport(status=200, body=b"")
    channel = FeishuChannel(_config("f1"), transport=transport)  # type: ignore[arg-type]
    ok = await channel.send(ChannelMessage(text="hi"))
    assert ok is True


@pytest.mark.asyncio
async def test_send_raises_on_non_json_response() -> None:
    transport = _FakeTransport(status=200, body=b"<html>not json</html>")
    channel = FeishuChannel(_config("f1"), transport=transport)  # type: ignore[arg-type]
    from src.services.channels import TransportError

    with pytest.raises(TransportError):
        await channel.send(ChannelMessage(text="hi"))


def test_message_level_passes_through() -> None:
    # Sanity: the level field on ChannelMessage does not affect payload
    # (Feishu is plain text/interactive), but ensure the channel accepts
    # all enum values without raising.
    transport = _FakeTransport()
    channel = FeishuChannel(_config("f1"), transport=transport)
    for level in MessageLevel:
        body, _ = channel.format_message(ChannelMessage(text="x", level=level))
        json.loads(body.decode("utf-8"))  # must parse
