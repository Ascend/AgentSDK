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

"""Tests for the WeChat iLink adapter, client, auth store, and pairing."""
# pylint: disable=no-name-in-module, redefined-outer-name, reimported
# pylint: disable=too-many-lines, use-implicit-booleaness-not-comparison

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse

import pytest

from clawcodex_ext.services.channels import wechat_ilink as wechat_module
from clawcodex_ext.services.channels.exceptions import TransportError
from clawcodex_ext.services.channels.models import ChannelConfig, ChannelMessage, ChannelType
from clawcodex_ext.services.channels.results import CircuitState, ErrorCategory, SendStatus
from clawcodex_ext.services.channels.transport import ChannelTransport, TransportResponse
from clawcodex_ext.services.channels.wechat_ilink import (
    PAIRING_CODE_TTL_SECONDS,
    WeChatAuthRecord,
    WeChatIlinkAuthStore,
    WeChatIlinkChannelAdapter,
    WeChatIlinkClient,
    WeChatPairingStore,
)
from clawcodex_ext.services.im_gateway.config import ReliabilityConfig
from clawcodex_ext.services.im_gateway.store import ReliabilityStore


# -- fake iLink transport ----------------------------------------------


class FakeIlinkTransport(ChannelTransport):
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.inbound_queue: list[dict] = []
        self.next_buf: str = "cur-1"
        self.login_token: str = "bot_tok_123"
        self.login_account_id: str = "bot_account_1"
        self.login_base_url: str = "https://ilink-region.weixin.qq.com"
        self.login_user_id: str = "bot_user_1"
        self.status_override: dict[str, int] = {}
        # path -> response body dict returned with HTTP 200 (used to inject
        # iLink body-level errors like {"ret": -14, "errcode": -14, ...}).
        self.payload_override: dict[str, dict] = {}
        self.getupdates_calls: list[dict] = []
        self.send_calls: list[dict] = []
        self.post_calls: list[dict] = []
        self.get_calls: list[str] = []
        self.getupdates_response_key = "msgs"  # real iLink response key
        self.status_transport_errors_before_success = 0

    async def get(self, url, *, headers=None, timeout=10.0) -> TransportResponse:  # type: ignore[override]
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        self.get_calls.append(f"{path}?{parsed.query}" if parsed.query else path)
        if path == "/ilink/bot/get_bot_qrcode":
            assert query.get("bot_type") == ["3"]
            resp = {
                "qrcode": "qr_token_abc",
                "qrcode_img_content": "https://ilinkai.weixin.qq.com/qr/abc",
            }
            return TransportResponse(200, json.dumps(resp).encode("utf-8"), {})
        if path == "/ilink/bot/get_qrcode_status":
            assert query.get("qrcode") == ["qr_token_abc"]
            if self.status_transport_errors_before_success > 0:
                self.status_transport_errors_before_success -= 1
                raise TransportError("transport timeout: The read operation timed out")
            resp = {
                "status": "confirmed",
                "ilink_bot_id": self.login_account_id,
                "bot_token": self.login_token,
                "baseurl": self.login_base_url,
                "ilink_user_id": self.login_user_id,
            }
            return TransportResponse(200, json.dumps(resp).encode("utf-8"), {})
        return TransportResponse(404, b"{}", {})

    async def post(self, url, body, *, headers=None, timeout=10.0) -> TransportResponse:  # type: ignore[override]
        path = urllib.parse.urlparse(url).path
        payload = json.loads(body.decode("utf-8")) if body else {}
        self.post_calls.append({"path": path, "payload": payload, "headers": dict(headers or {}), "timeout": timeout})
        if path in self.status_override:
            return TransportResponse(self.status_override[path], b'{"msg":"err"}', {})
        if path in self.payload_override:
            resp = self.payload_override[path]
            return TransportResponse(200, json.dumps(resp).encode("utf-8"), {})
        if path in {"/getupdates", "/ilink/bot/getupdates"}:
            self.getupdates_calls.append(payload)
            msgs = self.inbound_queue
            self.inbound_queue = []
            resp = {self.getupdates_response_key: msgs, "get_updates_buf": self.next_buf}
            return TransportResponse(200, json.dumps(resp).encode("utf-8"), {})
        if path in {"/sendmessage", "/ilink/bot/sendmessage"}:
            self.send_calls.append(payload)
            message = payload.get("msg") if isinstance(payload.get("msg"), dict) else payload
            self.sent.append(message)
            resp = {
                "message_id": f"srv_{len(self.sent)}",
                "client_id": message.get("client_id"),
            }
            return TransportResponse(200, json.dumps(resp).encode("utf-8"), {})
        return TransportResponse(404, b"{}", {})


def _cfg(name: str = "wechat-main") -> ChannelConfig:
    return ChannelConfig(
        type=ChannelType.WECHAT,
        webhook_url="https://ilinkai.weixin.qq.com/dummy",
        name=name,
        enabled=True,
        extra={"base_url": "https://ilinkai.weixin.qq.com", "account_id": "default"},
    )


def _make_adapter(tmp_path, transport=None, *, allowed_users=None, max_failures=10, pairing=None):
    transport = transport or FakeIlinkTransport()
    store = ReliabilityStore(tmp_path, ReliabilityConfig())
    auth_path = tmp_path / "auth.json"
    adapter = WeChatIlinkChannelAdapter(
        _cfg(),
        auth_store=WeChatIlinkAuthStore(auth_path),
        store=store,
        pairing=pairing,
        transport=transport,
        allowed_users=allowed_users or [],
        max_consecutive_failures=max_failures,
    )
    # save credentials so load_credentials() arms the client
    adapter._auth_store.save(
        WeChatAuthRecord(
            bot_token="bot_tok_123",
            account_id="default",
            base_url="https://ilinkai.weixin.qq.com",
            user_id="bot_user_1",
        )
    )
    adapter.load_credentials()
    return adapter, store, transport


def _sent_text(message: dict) -> str:
    return str((message["item_list"][0].get("text_item") or {}).get("text") or "")


# -- auth store ---------------------------------------------------------


def test_auth_store_encrypt_decrypt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_IM_SECRET", _fernet_key())
    store = WeChatIlinkAuthStore(tmp_path / "auth.json")
    store.save(
        WeChatAuthRecord(
            bot_token="secret_token",
            account_id="default",
            base_url="https://x",
            user_id="u1",
        )
    )
    # file is not plaintext
    raw = (tmp_path / "auth.json").read_text(encoding="utf-8")
    assert "secret_token" not in raw
    loaded = store.load()
    assert loaded is not None
    assert loaded.bot_token == "secret_token"
    assert loaded.user_id == "u1"


def test_auth_store_fallback_key_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CLAWCODEX_IM_SECRET", raising=False)
    store = WeChatIlinkAuthStore(tmp_path / "auth.json")
    store.save(WeChatAuthRecord(bot_token="t", account_id="default", base_url="https://x"))
    # key file created with 0o600; token still not plaintext
    key_file = tmp_path / "auth.key"
    assert key_file.exists()
    raw = (tmp_path / "auth.json").read_text(encoding="utf-8")
    assert '"t"' not in raw and "bot_token_enc" in raw
    assert store.load().bot_token == "t"


def test_auth_store_fallback_key_file_does_not_warn_for_local_cli(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.delenv("CLAWCODEX_IM_SECRET", raising=False)
    caplog.set_level(logging.WARNING)
    store = WeChatIlinkAuthStore(tmp_path / "auth.json")

    store.save(WeChatAuthRecord(bot_token="t", account_id="default", base_url="https://x"))

    assert not [record for record in caplog.records if "CLAWCODEX_IM_SECRET not set" in record.getMessage()]


def test_last_known_sender_falls_back_to_loaded_login_user_after_rescan(tmp_path) -> None:
    adapter, store, _ = _make_adapter(tmp_path)

    assert store.wechat_context_users("default") == []
    assert adapter.last_known_sender() == "bot_user_1"


def test_load_credentials_clears_stale_last_sender_when_account_changes(tmp_path) -> None:
    adapter, _, _ = _make_adapter(tmp_path)
    adapter._last_from_user_id = "old_user"
    adapter._auth_store.save(
        WeChatAuthRecord(
            bot_token="new_token",
            account_id="new_account",
            base_url="https://ilinkai.weixin.qq.com",
            user_id="new_user",
        )
    )

    adapter.load_credentials()

    assert adapter.last_known_sender() == "new_user"


def test_auth_store_wrong_key_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_IM_SECRET", _fernet_key())
    store = WeChatIlinkAuthStore(tmp_path / "auth.json")
    store.save(WeChatAuthRecord(bot_token="t", account_id="d", base_url="https://x"))
    monkeypatch.setenv("CLAWCODEX_IM_SECRET", _fernet_key())  # different key
    assert store.load() is None


def _fernet_key() -> str:
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode("utf-8")


# -- pairing -----------------------------------------------------------


def test_pairing_generate_and_consume(tmp_path) -> None:
    p = WeChatPairingStore(tmp_path / "pair.json")
    code = p.generate()
    assert len(code) >= 22  # 128-bit urlsafe
    assert p.consume(code, user_id="user_a") is True
    # single consume
    assert p.consume(code, user_id="user_b") is False
    assert p.is_allowed("user_a") is True


def test_pairing_expired_rejected(tmp_path, monkeypatch) -> None:
    p = WeChatPairingStore()
    code = p.generate()
    # backdate
    p._codes[code].created_at = time.time() - PAIRING_CODE_TTL_SECONDS - 1
    assert p.consume(code, user_id="user_a") is False


def test_pairing_already_bound_rejected(tmp_path) -> None:
    p = WeChatPairingStore()
    p.add_allowed("user_a")
    code = p.generate()
    assert p.consume(code, user_id="user_a") is False


def test_pairing_wrong_code_rejected(tmp_path) -> None:
    p = WeChatPairingStore()
    p.generate()
    assert p.consume("totally-wrong", user_id="u") is False


# -- client -------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_getupdates_cursor_and_headers(tmp_path) -> None:
    transport = FakeIlinkTransport()
    transport.inbound_queue = [
        {
            "msg_id": "m1",
            "from_user_id": "u1",
            "to_user_id": "bot",
            "msg_type": "TEXT",
            "text": "hi",
        }
    ]
    client = WeChatIlinkClient(
        base_url="https://ilinkai.weixin.qq.com",
        bot_token="tok",
        transport=transport,
    )
    msgs, buf = await client.getupdates("cur-0")
    assert buf == "cur-1"
    assert len(msgs) == 1
    assert msgs[0].text == "hi"
    # cursor echoed back to the server
    assert transport.getupdates_calls[0]["get_updates_buf"] == "cur-0"
    assert transport.send_calls == []
    # authenticated POST headers (AuthorizationType + Bearer + X-WECHAT-UIN)
    headers = transport.post_calls[0]["headers"]
    assert headers["AuthorizationType"] == "ilink_bot_token"
    assert headers["Authorization"] == "Bearer tok"
    assert headers["X-WECHAT-UIN"]
    assert headers["iLink-App-Id"] == "bot"


@pytest.mark.asyncio
async def test_client_sendmessage_receipt(tmp_path) -> None:
    transport = FakeIlinkTransport()
    client = WeChatIlinkClient(base_url="https://ilinkai.weixin.qq.com", bot_token="tok", transport=transport)
    result = await client.sendmessage(to_user_id="u1", text="hello")
    assert result.ok is True
    assert result.provider_receipt is not None
    assert transport.sent[0]["to_user_id"] == "u1"


@pytest.mark.asyncio
async def test_client_runtime_uses_ilink_bot_endpoints_and_message_shape(tmp_path) -> None:
    transport = FakeIlinkTransport()
    client = WeChatIlinkClient(base_url="https://ilinkai.weixin.qq.com", bot_token="tok", transport=transport)

    await client.getupdates(None)
    await client.sendmessage(to_user_id="u1", text="hello", context_token="ctx")

    assert [call["path"] for call in transport.post_calls] == [
        "/ilink/bot/getupdates",
        "/ilink/bot/sendmessage",
    ]
    # Every POST carries base_info.channel_version (iLink contract) ...
    # First-poll get_updates_buf is "" (NOT null) — iLink requires a string
    # cursor or the server silently drops the message stream.
    assert transport.post_calls[0]["payload"] == {
        "get_updates_buf": "",
        "base_info": {"channel_version": "2.2.0"},
    }
    send_payload = transport.post_calls[1]["payload"]
    assert send_payload["base_info"] == {"channel_version": "2.2.0"}
    assert send_payload["msg"]["to_user_id"] == "u1"
    assert send_payload["msg"]["message_type"] == 2
    assert send_payload["msg"]["message_state"] == 2
    assert send_payload["msg"]["context_token"] == "ctx"
    assert send_payload["msg"]["item_list"] == [{"type": 1, "text_item": {"text": "hello"}}]
    # ... and the X-WECHAT-UIN + AuthorizationType headers required by the
    # server on authenticated POSTs.
    for call in transport.post_calls:
        assert call["headers"]["AuthorizationType"] == "ilink_bot_token"
        assert call["headers"]["Authorization"] == "Bearer tok"
        assert call["headers"]["X-WECHAT-UIN"]
        assert call["headers"]["iLink-App-Id"] == "bot"
    # each request gets a fresh random UIN
    assert transport.post_calls[0]["headers"]["X-WECHAT-UIN"] != transport.post_calls[1]["headers"]["X-WECHAT-UIN"]


@pytest.mark.asyncio
async def test_client_getupdates_accepts_ilink_msgs_item_list_shape(tmp_path) -> None:
    transport = FakeIlinkTransport()
    transport.getupdates_response_key = "msgs"
    transport.inbound_queue = [
        {
            "message_id": "m1",
            "from_user_id": "u1",
            "to_user_id": "bot",
            "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
            "context_token": "ctx_abc",
        }
    ]
    client = WeChatIlinkClient(base_url="https://ilinkai.weixin.qq.com", bot_token="tok", transport=transport)

    msgs, buf = await client.getupdates("cur-0")

    assert buf == "cur-1"
    assert len(msgs) == 1
    assert msgs[0].message_id == "m1"
    assert msgs[0].text == "hi"
    assert msgs[0].context_token == "ctx_abc"


@pytest.mark.asyncio
async def test_qr_login_uses_bot_qrcode_status_and_persists_credentials(tmp_path) -> None:
    transport = FakeIlinkTransport()
    auth_store = WeChatIlinkAuthStore(tmp_path / "auth.json")
    adapter = WeChatIlinkChannelAdapter(
        _cfg(),
        auth_store=auth_store,
        store=None,  # type: ignore[arg-type]
        transport=transport,
    )

    result = await adapter.qr_login()

    assert transport.get_calls == [
        "/ilink/bot/get_bot_qrcode?bot_type=3",
        "/ilink/bot/get_qrcode_status?qrcode=qr_token_abc",
    ]
    assert result["code_url"] == "https://ilinkai.weixin.qq.com/qr/abc"
    assert result["bot_token"] == "bot_tok_123"
    record = auth_store.load()
    assert record is not None
    assert record.account_id == "bot_account_1"
    assert record.bot_token == "bot_tok_123"
    assert record.base_url == "https://ilink-region.weixin.qq.com"
    assert record.user_id == "bot_user_1"


@pytest.mark.asyncio
async def test_qr_login_treats_status_transport_timeout_as_transient(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wechat_module, "ILINK_QR_POLL_INTERVAL_SECONDS", 0)
    transport = FakeIlinkTransport()
    transport.status_transport_errors_before_success = 1
    auth_store = WeChatIlinkAuthStore(tmp_path / "auth.json")
    adapter = WeChatIlinkChannelAdapter(
        _cfg(),
        auth_store=auth_store,
        store=None,  # type: ignore[arg-type]
        transport=transport,
    )

    result = await adapter.qr_login()

    assert result["status"] == "confirmed"
    assert result["bot_token"] == "bot_tok_123"
    assert transport.get_calls.count("/ilink/bot/get_qrcode_status?qrcode=qr_token_abc") == 2


# -- adapter contract ---------------------------------------------------


def test_adapter_capabilities(tmp_path) -> None:
    from clawcodex_ext.services.channels.capabilities import ChannelCapability as CC

    adapter, _, _ = _make_adapter(tmp_path)
    caps = adapter.capabilities
    assert caps.has(CC.OUTBOUND_TEXT)
    assert {
        CC.OUTBOUND_TEXT,
        CC.INBOUND_POLLING,
        CC.CONTEXT_REPLY,
        CC.LOGIN_MANAGED,
    } <= caps.capabilities
    desc = caps.descriptor(CC.OUTBOUND_TEXT)
    assert desc.supports_markdown is False
    assert desc.max_text_length == 4000


def test_adapter_validate_config_ok(tmp_path) -> None:
    adapter, _, _ = _make_adapter(tmp_path)
    assert adapter.validate_config().ok is True


@pytest.mark.asyncio
async def test_adapter_send_success_with_receipt(tmp_path) -> None:
    adapter, _, transport = _make_adapter(tmp_path)
    result = await adapter.send(ChannelMessage(text="hello"), target="u1")
    assert result.ok is True
    assert result.channel_id == adapter.channel_id
    assert result.provider_receipt is not None
    assert transport.sent[0]["to_user_id"] == "u1"


@pytest.mark.asyncio
async def test_adapter_send_splits_long_text(tmp_path) -> None:
    adapter, _, transport = _make_adapter(tmp_path)
    long_text = "x" * 9000
    await adapter.send(ChannelMessage(text=long_text), target="u1")
    # 9000 / 4000 -> 3 chunks
    assert len(transport.sent) == 3


@pytest.mark.asyncio
async def test_adapter_send_throttles_consecutive_sendmessage_calls(tmp_path, monkeypatch) -> None:
    adapter, _, transport = _make_adapter(tmp_path)
    sleeps: list[float] = []
    clock = {"now": 100.0}

    def _monotonic() -> float:
        return clock["now"]

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)
        clock["now"] += delay

    monkeypatch.setattr(wechat_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(wechat_module.asyncio, "sleep", _sleep)

    await adapter.send(ChannelMessage(text="first"), target="u1")
    await adapter.send(ChannelMessage(text="second"), target="u1")

    assert len(transport.sent) == 2
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_adapter_reply_after_inbound_uses_monotonic_delay(tmp_path, monkeypatch) -> None:
    adapter, _, transport = _make_adapter(tmp_path)
    sleeps: list[float] = []
    clock = {"monotonic": 100.0, "wall": 1_800_000_000.0}

    def _monotonic() -> float:
        return clock["monotonic"]

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)
        clock["monotonic"] += delay

    monkeypatch.setattr(wechat_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(wechat_module.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(wechat_module.asyncio, "sleep", _sleep)
    transport.inbound_queue = [
        {
            "msg_id": "m1",
            "from_user_id": "u1",
            "to_user_id": "bot",
            "msg_type": "TEXT",
            "text": "hello",
        }
    ]
    adapter.set_inbound_handler(lambda _message: _noop())

    await adapter._poll_once()
    result = await adapter.send(ChannelMessage(text="reply"), target="u1")

    assert result.ok is True
    assert sleeps == [1.0]
    assert adapter._last_inbound_at == clock["wall"]
    assert len(transport.sent) == 1


@pytest.mark.asyncio
async def test_adapter_long_text_chunks_are_throttled_between_sendmessage_calls(tmp_path, monkeypatch) -> None:
    adapter, _, transport = _make_adapter(tmp_path)
    sleeps: list[float] = []
    clock = {"now": 200.0}

    def _monotonic() -> float:
        return clock["now"]

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)
        clock["now"] += delay

    monkeypatch.setattr(wechat_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(wechat_module.asyncio, "sleep", _sleep)

    await adapter.send(ChannelMessage(text="x" * 9000), target="u1")

    assert len(transport.sent) == 3
    # inter-chunk delay is 1.5s (hermes _send_chunk_delay_seconds). Each
    # inter-chunk sleep (1.5s) exceeds the 1.0s min-interval, so no extra
    # min-interval sleep fires between chunks → only the two inter-chunk sleeps.
    assert len(sleeps) == 2
    assert sleeps[0] == pytest.approx(1.5)  # inter-chunk delay after chunk1
    assert sleeps[1] == pytest.approx(1.5)  # inter-chunk delay after chunk2


@pytest.mark.asyncio
async def test_adapter_send_local_window_waits_before_transport(tmp_path, monkeypatch) -> None:
    adapter, _, transport = _make_adapter(tmp_path)
    adapter._send_min_interval_seconds = 0
    sleeps: list[float] = []
    clock = {"now": 300.0}

    def _monotonic():
        return clock["now"]

    async def _sleep(delay):
        sleeps.append(delay)
        clock["now"] += delay

    monkeypatch.setattr(wechat_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(wechat_module.asyncio, "sleep", _sleep)

    for idx in range(5):
        result = await adapter.send(ChannelMessage(text=f"msg {idx}"), target="u1")
        assert result.ok is True

    result = await adapter.send(ChannelMessage(text="sixth"), target="u1")

    assert result.ok is True
    assert sleeps == [10.0]
    assert len(transport.sent) == 6


@pytest.mark.asyncio
async def test_adapter_send_rate_limit_retries_then_succeeds(tmp_path, monkeypatch) -> None:
    adapter, _, transport = _make_adapter(tmp_path)
    adapter._send_min_interval_seconds = 0
    transport.payload_override["/ilink/bot/sendmessage"] = {
        "ret": -2,
        "errcode": -2,
        "errmsg": "freq limit",
    }

    async def _noop_sleep(_delay):
        pass

    monkeypatch.setattr(wechat_module.asyncio, "sleep", _noop_sleep)

    first = await adapter.send(ChannelMessage(text="first"), target="u1")
    transport.payload_override.pop("/ilink/bot/sendmessage")
    # Exhausting rate-limit retries opens the send-side circuit breaker for
    # 30s; simulate it closing before the next send so the second message
    # reaches the transport (and now succeeds with the override cleared).
    adapter._reset_send_rate_limit_circuit()
    second = await adapter.send(ChannelMessage(text="second"), target="u1")

    assert first.ok is False
    assert first.status is SendStatus.RATE_LIMITED
    assert first.error_category is ErrorCategory.RATE_LIMIT
    assert second.ok is True
    # first: 5 retry attempts; second: 1 send
    send_posts = [call for call in transport.post_calls if call["path"] == "/ilink/bot/sendmessage"]
    assert len(send_posts) == 6


@pytest.mark.asyncio
async def test_adapter_send_rate_limit_does_not_poison_next_send(tmp_path, monkeypatch) -> None:
    adapter, _, transport = _make_adapter(tmp_path)
    adapter._send_min_interval_seconds = 0
    transport.payload_override["/ilink/bot/sendmessage"] = {
        "ret": -2,
        "errcode": -2,
        "errmsg": "freq limit",
    }

    async def _noop_sleep(_delay):
        pass

    monkeypatch.setattr(wechat_module.asyncio, "sleep", _noop_sleep)

    first = await adapter.send(ChannelMessage(text="first"), target="u1")
    transport.payload_override.pop("/ilink/bot/sendmessage")
    # Circuit breaker opens after retries exhaust; clear it so the next send
    # is not short-circuited (simulates the 30s circuit elapsing).
    adapter._reset_send_rate_limit_circuit()
    second = await adapter.send(ChannelMessage(text="second"), target="u1")

    assert first.ok is False
    assert first.status is SendStatus.RATE_LIMITED
    assert first.error_category is ErrorCategory.RATE_LIMIT
    assert second.ok is True
    send_posts = [call for call in transport.post_calls if call["path"] == "/ilink/bot/sendmessage"]
    assert len(send_posts) == 6  # 5 rate-limit retries + 1 success


@pytest.mark.asyncio
async def test_adapter_poll_rate_limit_backoff_caps_at_300_seconds(tmp_path) -> None:
    adapter, store, transport = _make_adapter(tmp_path)
    transport.payload_override["/ilink/bot/getupdates"] = {
        "ret": -2,
        "errcode": -2,
        "errmsg": "freq limit",
    }

    retry_after_values: list[float] = []
    for _ in range(7):
        await adapter._poll_once()
        cooldowns = [e for e in store.audit_entries() if e["event_type"] == "wechat_rate_limit_cooldown"]
        retry_after_values.append(cooldowns[-1]["retry_after_seconds"])
        adapter._poll_rate_limit_cooldown_until = None

    assert retry_after_values == [2, 30, 30, 30, 30, 30, 30]
    assert adapter._poll_rate_limit_cooldown_seconds == 30


@pytest.mark.asyncio
async def test_adapter_send_requires_target(tmp_path) -> None:
    adapter, _, _ = _make_adapter(tmp_path)
    result = await adapter.send(ChannelMessage(text="hi"))
    assert result.ok is False
    assert result.error_category is ErrorCategory.CLIENT_ERROR


@pytest.mark.asyncio
async def test_adapter_send_session_expired_retries_without_context_token(tmp_path) -> None:
    """A stale peer context is evicted and retried once without invalidating bot auth."""
    transport = FakeIlinkTransport()
    _call_count = {"n": 0}
    _original_post = transport.post
    sent_payloads: list[dict] = []

    async def _sequential_post(url, body, *, headers=None, timeout=10.0):
        path = urllib.parse.urlparse(url).path
        if path in {"/sendmessage", "/ilink/bot/sendmessage"}:
            _call_count["n"] += 1
            payload = json.loads(body.decode("utf-8")) if body else {}
            msg = payload.get("msg") if isinstance(payload.get("msg"), dict) else payload
            sent_payloads.append(msg)
            if _call_count["n"] == 1:
                # First call: return session-expired error
                resp = {"ret": -14, "errcode": -14, "errmsg": "session expired"}
                return TransportResponse(200, json.dumps(resp).encode("utf-8"), {})
        return await _original_post(url, body, headers=headers, timeout=timeout)

    transport.post = _sequential_post  # type: ignore[assignment]

    adapter, store, _ = _make_adapter(tmp_path, transport=transport)
    adapter._send_min_interval_seconds = 0
    # Store a context_token for the target user
    store.set_context_token("default", "u1", "stale_token_abc")

    result = await adapter.send(ChannelMessage(text="hello"), target="u1")
    second = await adapter.send(ChannelMessage(text="again"), target="u1")

    assert result.ok is True
    assert adapter._account_status == "logged_in"
    assert second.ok is True
    assert len(sent_payloads) == 3
    assert sent_payloads[0].get("context_token") == "stale_token_abc"
    assert "context_token" not in sent_payloads[1]
    assert "context_token" not in sent_payloads[2]
    assert store.get_context_token("default", "u1") is None


# -- inbound: anti-loop, unsupported media, context_token --------------


@pytest.mark.asyncio
async def test_inbound_anti_loop_drops_self_message(tmp_path) -> None:
    adapter, store, transport = _make_adapter(tmp_path)
    # iLink marks a bot-originated message with from_user_id == the bot's
    # account_id (...@im.bot), NOT its wechat user_id. The adapter's account_id
    # is "default" (see _make_adapter), so a self-message comes from "default".
    transport.inbound_queue = [
        {
            "msg_id": "m1",
            "from_user_id": "default",
            "to_user_id": "default",
            "msg_type": "TEXT",
            "text": "self",
        }
    ]
    received: list = []
    adapter.set_inbound_handler(lambda m: received.append(m) or _noop())
    await adapter._poll_once()
    assert received == []
    assert any(e["event_type"] == "wechat_self_message_dropped" for e in store.audit_entries())


async def _noop() -> None:
    return None


async def _noop_sleep(_delay: float) -> None:
    return None


@pytest.mark.asyncio
async def test_inbound_unsupported_media_records_and_replies(tmp_path, monkeypatch) -> None:
    adapter, store, transport = _make_adapter(tmp_path, allowed_users=["u1"])
    transport.inbound_queue = [
        {
            "msg_id": "m1",
            "from_user_id": "u1",
            "to_user_id": "bot",
            "msg_type": "IMAGE",
            "text": None,
            "size": 12345,
        }
    ]
    received: list = []
    adapter.set_inbound_handler(lambda m: received.append(m) or _noop())
    # The unsupported-media reply is dispatched fire-and-forget so it does
    # not block the poll loop. No-op sleep lets the spawned reply task
    # complete within the event loop turn after _poll_once returns.
    monkeypatch.setattr(wechat_module.asyncio, "sleep", _noop_sleep)
    await adapter._poll_once()
    # The reply is dispatched fire-and-forget; drain spawned tasks before
    # asserting on transport.sent.
    for task in [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
        await task
    assert not received  # media not delivered to agent
    # unsupported_inbound recorded
    assert any(e.get("media_type") == "IMAGE" for e in _unsupported_entries(store))
    assert any(e.get("size") == 12345 for e in _unsupported_entries(store))
    # reply sent
    assert len(transport.sent) == 1
    assert "仅支持文本" in _sent_text(transport.sent[0])


def _unsupported_entries(store):
    from pathlib import Path

    p = Path(store.state_dir) / "unsupported_inbound.ndjson"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_inbound_text_delivered_with_context_token(tmp_path) -> None:
    adapter, store, transport = _make_adapter(tmp_path, allowed_users=["u1"])
    transport.inbound_queue = [
        {
            "msg_id": "m1",
            "from_user_id": "u1",
            "to_user_id": "bot",
            "msg_type": "TEXT",
            "text": "hi",
            "context_token": "ctx_abc",
        }
    ]
    received: list = []
    adapter.set_inbound_handler(lambda m: received.append(m) or _noop())
    await adapter._poll_once()
    assert len(received) == 1
    assert received[0].text == "hi"
    assert received[0].context_token == "ctx_abc"
    # context_token persisted
    assert store.get_context_token("default", "u1") == "ctx_abc"


@pytest.mark.asyncio
async def test_inbound_cursor_persisted_across_polls(tmp_path) -> None:
    adapter, _, transport = _make_adapter(tmp_path, allowed_users=["u1"])
    transport.inbound_queue = [
        {"msg_id": "m1", "from_user_id": "u1", "to_user_id": "bot", "msg_type": "TEXT", "text": "a"}
    ]
    adapter.set_inbound_handler(lambda m: _noop())
    await adapter._poll_once()
    assert adapter._get_updates_buf == "cur-1"
    # second poll sends the saved buf
    await adapter._poll_once()
    assert transport.getupdates_calls[1]["get_updates_buf"] == "cur-1"


@pytest.mark.asyncio
async def test_inbound_cursor_not_overwritten_when_server_omits_buf(tmp_path) -> None:
    """An empty/absent get_updates_buf must NOT clobber a valid cursor."""
    adapter, _, transport = _make_adapter(tmp_path, allowed_users=["u1"])
    transport.inbound_queue = [
        {"msg_id": "m1", "from_user_id": "u1", "to_user_id": "bot", "msg_type": "TEXT", "text": "a"}
    ]
    adapter.set_inbound_handler(lambda m: _noop())
    await adapter._poll_once()
    assert adapter._get_updates_buf == "cur-1"
    # server now returns no cursor — existing cursor must be preserved
    transport.next_buf = ""
    transport.inbound_queue = []
    await adapter._poll_once()
    assert adapter._get_updates_buf == "cur-1"
    assert transport.getupdates_calls[1]["get_updates_buf"] == "cur-1"


@pytest.mark.asyncio
async def test_inbound_cursor_restored_across_adapter_restarts(tmp_path) -> None:
    adapter, _, transport = _make_adapter(tmp_path, allowed_users=["u1"])
    transport.inbound_queue = [
        {"msg_id": "m1", "from_user_id": "u1", "to_user_id": "bot", "msg_type": "TEXT", "text": "a"}
    ]
    adapter.set_inbound_handler(lambda m: _noop())
    await adapter._poll_once()
    # first poll uses "" cursor (iLink requires a string, not null)
    assert transport.getupdates_calls[0]["get_updates_buf"] == ""

    restarted_transport = FakeIlinkTransport()
    restarted, _, _ = _make_adapter(tmp_path, transport=restarted_transport, allowed_users=["u1"])
    restarted.set_inbound_handler(lambda m: _noop())
    await restarted._poll_once()
    assert restarted_transport.getupdates_calls[0]["get_updates_buf"] == "cur-1"


@pytest.mark.asyncio
async def test_inbound_text_from_unpaired_user_delivered_without_pairing(tmp_path) -> None:
    adapter, store, transport = _make_adapter(tmp_path, allowed_users=[])
    transport.inbound_queue = [
        {
            "msg_id": "m1",
            "from_user_id": "stranger",
            "to_user_id": "bot",
            "msg_type": "TEXT",
            "text": "hi",
        }
    ]
    received: list = []
    adapter.set_inbound_handler(lambda m: received.append(m) or _noop())
    await adapter._poll_once()
    assert len(received) == 1
    assert received[0].from_user_id == "stranger"
    assert received[0].text == "hi"
    assert transport.sent == []


@pytest.mark.asyncio
async def test_inbound_pairing_code_text_is_delivered_like_normal_message(tmp_path) -> None:
    pairing = WeChatPairingStore(tmp_path / "pairing.json")
    code = pairing.generate()
    adapter, _, transport = _make_adapter(tmp_path, allowed_users=[], pairing=pairing)
    transport.inbound_queue = [
        {
            "msg_id": "m1",
            "from_user_id": "new_user",
            "to_user_id": "bot",
            "msg_type": "TEXT",
            "text": code,
        }
    ]
    received: list = []
    adapter.set_inbound_handler(lambda m: received.append(m) or _noop())
    await adapter._poll_once()
    assert len(received) == 1
    assert received[0].text == code
    assert pairing.is_allowed("new_user") is False
    assert transport.sent == []


def test_pairing_code_single_consume_across_store_instances(tmp_path) -> None:
    path = tmp_path / "pairing.json"
    first = WeChatPairingStore(path)
    code = first.generate()
    second = WeChatPairingStore(path)
    assert first.consume(code, user_id="user_a") is True
    assert second.consume(code, user_id="user_b") is False
    reloaded = WeChatPairingStore(path)
    assert reloaded.is_allowed("user_a") is True
    assert reloaded.is_allowed("user_b") is False


# -- circuit breaker / 401 ---------------------------------------------


@pytest.mark.asyncio
async def test_circuit_opens_after_consecutive_failures(tmp_path) -> None:
    adapter, store, transport = _make_adapter(tmp_path, max_failures=3)
    transport.status_override["/ilink/bot/getupdates"] = 500
    for _ in range(3):
        await adapter._poll_once()
    assert adapter._circuit_state is CircuitState.OPEN
    assert adapter._account_status == "circuit_open"
    assert any(e["event_type"] == "wechat_circuit_open" for e in store.audit_entries())
    # further poll_once is a no-op while open
    await adapter._poll_once()


@pytest.mark.asyncio
async def test_401_marks_session_expired_without_consuming_breaker(tmp_path) -> None:
    """HTTP 401 during poll marks session expired (permanent) instead of rate-limit cooldown."""
    adapter, store, transport = _make_adapter(tmp_path, max_failures=3)
    transport.status_override["/ilink/bot/getupdates"] = 401
    await adapter._poll_once()
    await adapter._poll_once()
    assert adapter._account_status == "session_expired"
    assert adapter._consecutive_failures == 0
    assert adapter._circuit_state is CircuitState.CLOSED
    expired = [e for e in store.audit_entries() if e["event_type"] == "wechat_session_expired"]
    assert expired and expired[0].get("reason") == "http_401"
    poll_posts = [call for call in transport.post_calls if call["path"] == "/ilink/bot/getupdates"]
    assert len(poll_posts) == 1


@pytest.mark.asyncio
async def test_getupdates_session_expired_body_marks_session_expired(tmp_path) -> None:
    """iLink signals session expiry as ret/errcode=-14 inside HTTP 200."""
    adapter, store, transport = _make_adapter(tmp_path, max_failures=3)
    transport.payload_override["/ilink/bot/getupdates"] = {
        "ret": -14,
        "errcode": -14,
        "errmsg": "session expired",
    }
    await adapter._poll_once()
    assert adapter._account_status == "session_expired"
    assert adapter._consecutive_failures == 0
    assert adapter._circuit_state is CircuitState.CLOSED
    assert "session expired" in (adapter._last_error or "")
    expired = [e for e in store.audit_entries() if e["event_type"] == "wechat_session_expired"]
    assert expired and expired[0].get("reason") == "session_expired"


@pytest.mark.asyncio
async def test_session_expired_skips_polling_and_sending_until_rescan(tmp_path) -> None:
    """Session-expired blocks subsequent polls and send returns AUTH error."""
    adapter, _, transport = _make_adapter(tmp_path, max_failures=3)
    transport.payload_override["/ilink/bot/getupdates"] = {
        "ret": -14,
        "errcode": -14,
        "errmsg": "session expired",
    }

    await adapter._poll_once()
    transport.payload_override.pop("/ilink/bot/getupdates")
    await adapter._poll_once()
    send_result = await adapter.send(ChannelMessage(text="hello"), target="u1")

    assert adapter._account_status == "session_expired"
    poll_posts = [call for call in transport.post_calls if call["path"] == "/ilink/bot/getupdates"]
    assert len(poll_posts) == 1
    assert len(transport.send_calls) == 0
    assert send_result.ok is False
    assert send_result.error_category is ErrorCategory.AUTH
    assert "session expired" in send_result.message.lower() or "re-scan" in send_result.message.lower()


@pytest.mark.asyncio
async def test_load_credentials_clears_session_expired_for_immediate_resume(tmp_path) -> None:
    """A fresh credential load (initial start, or a QR re-scan) clears session-expired status."""
    adapter, _, _ = _make_adapter(tmp_path)
    adapter._mark_session_expired("session expired; re-scan required", reason="session_expired")
    assert adapter._account_status == "session_expired"
    assert adapter._rate_limit_remaining() == 0.0

    adapter.load_credentials()

    assert adapter._account_status == "logged_in"
    assert adapter._poll_rate_limit_cooldown_until is None
    assert adapter._rate_limit_remaining() == 0.0
    assert adapter._poll_rate_limit_cooldown_seconds == wechat_module.WECHAT_RATE_LIMIT_RETRY_DELAY_SECONDS


@pytest.mark.asyncio
async def test_getupdates_stale_session_ret2_unknown_error_marks_session_expired(tmp_path) -> None:
    """ret=-2 + errmsg 'unknown error' is a stale-session signal (hermes)."""
    adapter, _, transport = _make_adapter(tmp_path, max_failures=3)
    transport.payload_override["/ilink/bot/getupdates"] = {
        "ret": -2,
        "errmsg": "unknown error",
    }
    await adapter._poll_once()
    assert adapter._account_status == "session_expired"
    assert adapter._consecutive_failures == 0


@pytest.mark.asyncio
async def test_getupdates_rate_limit_enters_cooldown_not_breaker(tmp_path) -> None:
    """ret=-2 with a non-'unknown error' errmsg is a real rate limit cooldown."""
    adapter, _, transport = _make_adapter(tmp_path, max_failures=3)
    transport.payload_override["/ilink/bot/getupdates"] = {
        "ret": -2,
        "errcode": -2,
        "errmsg": "freq limit",
    }
    await adapter._poll_once()
    assert adapter._account_status == "logged_in"
    assert adapter._consecutive_failures == 0
    assert adapter._rate_limit_remaining() > 0


@pytest.mark.asyncio
async def test_send_rate_limit_does_not_block_inbound_polling(tmp_path, monkeypatch) -> None:
    """Send-side rate-limit failures must not freeze getupdates and delay commands."""
    adapter, _, transport = _make_adapter(tmp_path, allowed_users=["u1"])
    adapter._send_min_interval_seconds = 0
    transport.payload_override["/ilink/bot/sendmessage"] = {
        "ret": -2,
        "errcode": -2,
        "errmsg": "freq limit",
    }

    async def _noop_sleep(_delay):
        pass

    monkeypatch.setattr(wechat_module.asyncio, "sleep", _noop_sleep)
    rate_limited = await adapter.send(ChannelMessage(text="first"), target="u1")
    transport.payload_override.pop("/ilink/bot/sendmessage")
    transport.inbound_queue = [
        {
            "msg_id": "m1",
            "from_user_id": "u1",
            "to_user_id": "bot",
            "msg_type": "TEXT",
            "text": "/stop",
        }
    ]
    received: list = []
    adapter.set_inbound_handler(lambda m: received.append(m) or _noop())

    await adapter._poll_once()

    assert rate_limited.error_category is ErrorCategory.RATE_LIMIT
    assert rate_limited.status is SendStatus.RATE_LIMITED
    # Send-side rate limit opens the send circuit (30s) but must NOT freeze
    # inbound polling: getupdates still ran and delivered the inbound message.
    assert adapter._send_rate_limit_remaining() > 0
    assert adapter._poll_rate_limit_remaining() == 0.0
    assert len(transport.getupdates_calls) == 1
    assert len(received) == 1
    assert received[0].text == "/stop"


@pytest.mark.asyncio
async def test_getupdates_generic_platform_error_records_failure(tmp_path) -> None:
    adapter, _, transport = _make_adapter(tmp_path, max_failures=3)
    transport.payload_override["/ilink/bot/getupdates"] = {
        "ret": -99,
        "errmsg": "boom",
    }
    await adapter._poll_once()
    assert adapter._account_status == "logged_in"
    assert adapter._consecutive_failures == 1


@pytest.mark.asyncio
async def test_sendmessage_session_expired_raises_for_adapter_retry(tmp_path) -> None:
    """Session-expired errors are re-raised (not returned) so the adapter
    can evict the stale context_token and perform the tokenless retry.
    """
    from clawcodex_ext.services.channels.wechat_ilink import _IlinkPlatformError

    transport = FakeIlinkTransport()
    transport.payload_override["/ilink/bot/sendmessage"] = {
        "ret": -14,
        "errcode": -14,
        "errmsg": "session expired",
    }
    client = WeChatIlinkClient(base_url="https://ilinkai.weixin.qq.com", bot_token="tok", transport=transport)
    with pytest.raises(_IlinkPlatformError) as exc_info:
        await client.sendmessage(to_user_id="u1", text="hi")
    assert exc_info.value.is_session_expired is True


@pytest.mark.asyncio
async def test_sendmessage_rate_limited_returns_retryable(tmp_path) -> None:
    transport = FakeIlinkTransport()
    transport.payload_override["/ilink/bot/sendmessage"] = {
        "ret": -2,
        "errcode": -2,
        "errmsg": "freq limit",
    }
    client = WeChatIlinkClient(base_url="https://ilinkai.weixin.qq.com", bot_token="tok", transport=transport)
    result = await client.sendmessage(to_user_id="u1", text="hi")
    assert result.ok is False
    assert result.error_category is ErrorCategory.RATE_LIMIT
    assert result.retryable is True


def test_parse_ilink_response_raises_on_nonzero_ret() -> None:
    from clawcodex_ext.services.channels.wechat_ilink import (
        _IlinkPlatformError,
        _parse_ilink_response,
    )
    from clawcodex_ext.services.channels.transport import TransportResponse

    resp = TransportResponse(200, json.dumps({"ret": -14, "errcode": -14, "errmsg": "session expired"}).encode(), {})
    with pytest.raises(_IlinkPlatformError) as exc_info:
        _parse_ilink_response(resp)
    assert exc_info.value.is_session_expired is True
    assert exc_info.value.errmsg == "session expired"


def test_parse_ilink_response_success_when_ret_zero_or_absent() -> None:
    from clawcodex_ext.services.channels.wechat_ilink import _parse_ilink_response
    from clawcodex_ext.services.channels.transport import TransportResponse

    for body in ({"msgs": [], "get_updates_buf": "x"}, {"ret": 0, "errcode": 0, "msgs": []}):
        resp = TransportResponse(200, json.dumps(body).encode(), {})
        assert _parse_ilink_response(resp) == body


def test_reset_circuit_restores_closed(tmp_path) -> None:
    adapter, _, _ = _make_adapter(tmp_path, max_failures=1)
    adapter._record_failure("boom")
    assert adapter._circuit_state is CircuitState.OPEN
    adapter.reset_circuit()
    assert adapter._circuit_state is CircuitState.CLOSED
    assert adapter._consecutive_failures == 0


@pytest.mark.asyncio
async def test_health_check_reflects_state(tmp_path) -> None:
    adapter, _, _ = _make_adapter(tmp_path)
    health = await adapter.health_check()
    assert health.channel_id == "wechat-main"
    assert health.circuit_state == "closed"
    assert health.account_status == "logged_in"


# -- context_token round-trip on outbound ------------------------------


@pytest.mark.asyncio
async def test_outbound_loads_saved_context_token(tmp_path) -> None:
    adapter, store, transport = _make_adapter(tmp_path)
    store.set_context_token("default", "u1", "ctx_xyz")
    await adapter.send(ChannelMessage(text="reply"), target="u1")
    assert transport.sent[0]["context_token"] == "ctx_xyz"


# -- session-expired vs rate-limit separation ---------------------------


@pytest.mark.asyncio
async def test_rate_limit_cooldown_does_not_mark_session_expired(tmp_path) -> None:
    """Genuine rate limit (ret=-2, non-stale-session errmsg) enters cooldown
    but keeps account_status as logged_in — unlike session expiry.
    """
    adapter, store, transport = _make_adapter(tmp_path, max_failures=3)
    transport.payload_override["/ilink/bot/getupdates"] = {
        "ret": -2,
        "errcode": -2,
        "errmsg": "freq limit",
    }
    await adapter._poll_once()
    assert adapter._account_status == "logged_in"
    assert adapter._rate_limit_remaining() > 0
    assert adapter._consecutive_failures == 0
    assert adapter._circuit_state is CircuitState.CLOSED
    cooldowns = [e for e in store.audit_entries() if e["event_type"] == "wechat_rate_limit_cooldown"]
    assert cooldowns and cooldowns[0].get("reason") == "rate_limit"


@pytest.mark.asyncio
async def test_session_expired_health_check_reports_unhealthy(tmp_path) -> None:
    """health_check must report healthy=False when session is expired."""
    adapter, _, transport = _make_adapter(tmp_path)
    transport.status_override["/ilink/bot/getupdates"] = 401
    await adapter._poll_once()
    health = await adapter.health_check()
    assert health.healthy is False
    assert health.account_status == "session_expired"
