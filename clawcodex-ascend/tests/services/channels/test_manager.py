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

"""ChannelManager dispatch + thread-safety tests."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import pytest

from src.services.channels import (
    BaseChannel,
    ChannelConfig,
    ChannelDisabledError,
    ChannelManager,
    ChannelMessage,
    ChannelNotFoundError,
    ChannelType,
    ChannelTransport,
    TransportResponse,
)
from src.services.channels.base import default_headers
from src.services.channels.transport import (
    DEFAULT_TIMEOUT_SECONDS,
    encode_json_body,
)


class _RecordingTransport(ChannelTransport):
    def __init__(self, status: int = 200, body: bytes = b"") -> None:
        self.status = status
        self.body = body
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    async def post(
        self,
        url: str,
        body: bytes,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> TransportResponse:
        with self._lock:
            self.calls.append({"url": url, "body": body, "headers": dict(headers or {}), "timeout": timeout})
        return TransportResponse(status=self.status, body=self.body, headers={})


class _StubChannel(BaseChannel):
    """A no-op channel that records every send."""

    def __init__(
        self,
        config: ChannelConfig,
        transport: _RecordingTransport,
        *,
        should_raise: BaseException | None = None,
    ) -> None:
        super().__init__(config, transport=transport)
        self._should_raise = should_raise
        self.received: list[ChannelMessage] = []

    def format_message(self, message: ChannelMessage) -> tuple[bytes, dict[str, str]]:
        return encode_json_body({"text": message.text}), default_headers()

    async def send(self, message: ChannelMessage) -> bool:
        self.received.append(message)
        if self._should_raise is not None:
            raise self._should_raise
        body, headers = self.format_message(message)
        await self._transport.post(self._config.webhook_url, body, headers=headers, timeout=DEFAULT_TIMEOUT_SECONDS)
        return self._transport.status == 200


def _make_config(name: str, url: str = "https://hooks.example.com/x") -> ChannelConfig:
    return ChannelConfig(type=ChannelType.SLACK, webhook_url=url, name=name)


@pytest.mark.asyncio
async def test_manager_register_and_lookup() -> None:
    transport = _RecordingTransport()
    channel = _StubChannel(_make_config("a"), transport)
    manager = ChannelManager()
    manager.register(channel)
    assert manager.names() == ["a"]
    assert manager.get("a") is channel
    assert manager.get("missing") is None


def test_manager_unregister_is_silent_on_missing() -> None:
    manager = ChannelManager()
    manager.unregister("nope")  # must not raise


@pytest.mark.asyncio
async def test_manager_send_to_dispatches_to_named_channel() -> None:
    transport = _RecordingTransport()
    channel = _StubChannel(_make_config("a"), transport)
    manager = ChannelManager()
    manager.register(channel)
    ok = await manager.send_to("a", ChannelMessage(text="hi"))
    assert ok is True
    assert len(transport.calls) == 1
    assert channel.received[0].text == "hi"


@pytest.mark.asyncio
async def test_manager_send_to_raises_when_missing() -> None:
    manager = ChannelManager()
    with pytest.raises(ChannelNotFoundError):
        await manager.send_to("nope", ChannelMessage(text="hi"))


@pytest.mark.asyncio
async def test_manager_send_to_raises_when_disabled() -> None:
    transport = _RecordingTransport()
    cfg = _make_config("a")
    cfg.enabled = False
    channel = _StubChannel(cfg, transport)
    manager = ChannelManager()
    manager.register(channel)
    with pytest.raises(ChannelDisabledError):
        await manager.send_to("a", ChannelMessage(text="hi"))
    assert not transport.calls


@pytest.mark.asyncio
async def test_broadcast_dispatches_to_all_channels() -> None:
    transport_a = _RecordingTransport()
    transport_b = _RecordingTransport()
    channel_a = _StubChannel(_make_config("a"), transport_a)
    channel_b = _StubChannel(_make_config("b"), transport_b)
    manager = ChannelManager()
    manager.register(channel_a)
    manager.register(channel_b)
    results = await manager.broadcast(ChannelMessage(text="hi"))
    assert results == {"a": True, "b": True}
    assert len(transport_a.calls) == 1
    assert len(transport_b.calls) == 1


@pytest.mark.asyncio
async def test_broadcast_continues_on_per_channel_failure(caplog) -> None:
    transport_a = _RecordingTransport()
    transport_b = _RecordingTransport()
    transport_c = _RecordingTransport()
    channel_a = _StubChannel(_make_config("a"), transport_a)
    channel_b = _StubChannel(
        _make_config("b"),
        transport_b,
        should_raise=RuntimeError("boom token=channel-secret"),
    )
    channel_c = _StubChannel(_make_config("c"), transport_c)
    manager = ChannelManager()
    manager.register(channel_a)
    manager.register(channel_b)
    manager.register(channel_c)
    with caplog.at_level(logging.WARNING, logger="clawcodex_ext.services.channels.base"):
        results = await manager.broadcast(ChannelMessage(text="hi"))
    # a and c succeed, b is recorded as failure but doesn't crash the call.
    assert results == {"a": True, "b": False, "c": True}
    # The other channels must still receive the call.
    assert len(transport_a.calls) == 1
    assert len(transport_c.calls) == 1
    assert "channel=b" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert "error_fingerprint=" in caplog.text
    assert "boom" not in caplog.text
    assert "channel-secret" not in caplog.text


@pytest.mark.asyncio
async def test_broadcast_empty_manager_returns_empty_dict() -> None:
    manager = ChannelManager()
    results = await manager.broadcast(ChannelMessage(text="hi"))
    assert results == {}


@pytest.mark.asyncio
async def test_broadcast_runs_channels_in_parallel() -> None:
    # Three channels that each sleep briefly; broadcast should not be serial.
    delays = [0.1, 0.1, 0.1]

    class _SlowChannel(BaseChannel):
        def __init__(self, name: str, delay: float) -> None:
            super().__init__(_make_config(name, "https://hooks.example.com/x"))
            self.delay = delay
            self.calls = 0

        def format_message(self, message: ChannelMessage) -> tuple[bytes, dict[str, str]]:
            return encode_json_body({"text": message.text}), default_headers()

        async def send(self, message: ChannelMessage) -> bool:
            await asyncio.sleep(self.delay)
            self.calls += 1
            return True

    manager = ChannelManager()
    channels = [_SlowChannel(f"c{i}", d) for i, d in enumerate(delays)]
    for ch in channels:
        manager.register(ch)

    import time

    start = time.monotonic()
    results = await manager.broadcast(ChannelMessage(text="hi"))
    elapsed = time.monotonic() - start
    assert results == {f"c{i}": True for i in range(len(delays))}
    # Serial execution would take ~0.3s; parallel should be much closer to
    # 0.1s. Use a generous upper bound to avoid flakiness on slow CI.
    assert elapsed < sum(delays) * 0.6


def test_manager_register_is_thread_safe() -> None:
    manager = ChannelManager()
    transport = _RecordingTransport()
    n = 100

    def register_one(i: int) -> None:
        ch = _StubChannel(_make_config(f"c{i}"), transport)
        manager.register(ch)

    threads = [threading.Thread(target=register_one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # All 100 distinct names must be present after concurrent registration.
    assert set(manager.names()) == {f"c{i}" for i in range(n)}
    assert len(manager.names()) == n


@pytest.mark.asyncio
async def test_manager_send_to_returns_false_on_http_error() -> None:
    transport = _RecordingTransport(status=500)
    channel = _StubChannel(_make_config("a"), transport)
    manager = ChannelManager()
    manager.register(channel)
    ok = await manager.send_to("a", ChannelMessage(text="hi"))
    assert ok is False
