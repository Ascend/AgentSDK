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

"""Tests for GatewayIpcClient auto-reconnect (exp backoff + re-register).

Covers the three P5 residual-risk contract boundaries:

  1. A redelivered ``delivery_id`` after reconnect hits the gateway's
     server-side dedup (``InboundDispatcher`` keys on ``message_id``) —
     reconnect must not duplicate.
  2. After the gateway daemon restarts, reconnect re-registers so the
     binding is rebuilt (origin visible as active again).
  3. During the disconnect gap the gateway does NOT replay inbound that
     arrived while the opt-in target was offline (no offline payload
     store — contract boundary). The test asserts the gap message is
     rejected as ``target_offline`` / not delivered to the handler.

Plus the basic backoff-escalation happy path.
"""

from __future__ import annotations

# This split test branch is linted without the separately submitted gateway implementation.
# pylint: disable=no-name-in-module

import asyncio
import logging

import pytest

from clawcodex_ext.services.im_gateway.binding import BindingPolicy
from clawcodex_ext.services.im_gateway.config import GatewayConfig
from clawcodex_ext.services.im_gateway.gateway import MessageGateway
from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient
from clawcodex_ext.services.im_gateway.ipc_protocol import GatewayFrame
from clawcodex_ext.services.im_gateway.ipc_server import GatewayIpcServer
from clawcodex_ext.services.im_gateway.models import (
    AckLayer,
    AckReceipt,
    InboundMessage,
    SessionTarget,
)


class _RecordingHandler:
    """Real dispatcher handler — records every message that survives dedup."""

    def __init__(self) -> None:
        self.received: list[InboundMessage] = []

    async def __call__(self, message: InboundMessage):
        self.received.append(message)
        return AckReceipt(message.message_id or "d1", AckLayer.ENQUEUED, "enqueued")


@pytest.mark.asyncio
async def test_reconnect_logs_safe_exception_category(tmp_path, monkeypatch, caplog) -> None:
    client = GatewayIpcClient(tmp_path / "gateway.sock", instance_id="repl_main")

    async def denied_connect() -> None:
        raise PermissionError("token=exception-secret")

    monkeypatch.setattr(client, "connect", denied_connect)
    with caplog.at_level(logging.DEBUG, logger="clawcodex_ext.services.im_gateway.ipc_client"):
        response = await client.reconnect_until_registered(
            session_id="repl_main",
            origin="o1",
            base_delay=0,
            max_delay=0,
            max_attempts=1,
        )

    assert response is None
    assert "exception_type=PermissionError" in caplog.text
    assert "error_fingerprint=" in caplog.text
    assert "exception-secret" not in caplog.text


@pytest.mark.asyncio
async def test_reconnect_logs_safe_registration_nack(tmp_path, monkeypatch, caplog) -> None:
    client = GatewayIpcClient(tmp_path / "gateway.sock", instance_id="repl_main")

    async def connected() -> None:
        return None

    async def rejected(**_kwargs) -> GatewayFrame:
        return GatewayFrame.nack(delivery_id="register", reason="password=nack-secret")

    monkeypatch.setattr(client, "connect", connected)
    monkeypatch.setattr(client, "register", rejected)
    with caplog.at_level(logging.WARNING, logger="clawcodex_ext.services.im_gateway.ipc_client"):
        response = await client.reconnect_until_registered(
            session_id="repl_main",
            origin="o1",
            base_delay=0,
            max_delay=0,
            max_attempts=1,
        )

    assert response is None
    assert "response_type=nack ack_layer=none" in caplog.text
    assert "reason_fingerprint=" in caplog.text
    assert "nack-secret" not in caplog.text


class _FakeGateway:
    """Minimal gateway stand-in for binding/reload tests (no real dispatcher)."""

    def __init__(self):
        self.reloaded = []
        self.binding = BindingPolicy()

    async def receive(self, message):
        return AckReceipt(message.message_id or "d1", AckLayer.ENQUEUED, "enqueued")

    def reload_channel(self, name):
        self.reloaded.append(name)
        return name != "missing"

    async def health(self):
        return {"running": True, "channels": ["wechat-main"], "peers": 0}


def _real_gateway(tmp_path) -> tuple[MessageGateway, _RecordingHandler]:
    """A real MessageGateway whose InboundDispatcher actually dedupes."""
    gw = MessageGateway(GatewayConfig(state_dir=str(tmp_path)))
    handler = _RecordingHandler()
    gw.inbound.set_handler(handler)
    return gw, handler


@pytest.mark.asyncio
async def test_reconnect_redeliver_hits_server_dedup(tmp_path) -> None:
    """Boundary 1: redelivering the same delivery_id after reconnect is
    deduped server-side (InboundDispatcher keys on message_id), not
    double-delivered to the handler.
    """
    gw, handler = _real_gateway(tmp_path)
    sock = tmp_path / "gw.sock"
    server = GatewayIpcServer(sock, gw)
    await server.start()
    try:
        async with GatewayIpcClient(sock, instance_id="repl_main") as client:
            await client.register(session_id="repl_main", origin="o1")
            r1 = await client.deliver(delivery_id="d1", session_id="repl_main", origin="o1", text="hello")
            assert r1 is not None and r1.ack_layer == "enqueued"
            # simulate a naive reconnect that lost client-side dedup state: the
            # same delivery_id becomes eligible to be sent again.
            client._seen_delivery_ids.clear()
            r2 = await client.deliver(delivery_id="d1", session_id="repl_main", origin="o1", text="hello")
            # server-side dedup rejects the duplicate (accepted "duplicate; skipped")
            assert r2 is not None
            assert r2.ack_layer == "accepted"
            assert "duplicate" in (r2.reason or "")
        # handler received the message exactly once — end-to-end idempotent
        assert len(handler.received) == 1
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_client_delivery_cache_is_bounded_and_evicted_ids_can_retry(tmp_path) -> None:
    client = GatewayIpcClient(tmp_path / "gw.sock", delivery_cache_size=2)
    calls: list[str] = []

    async def accepted(frame):
        calls.append(frame.delivery_id)
        return GatewayFrame.ack(delivery_id=frame.delivery_id, layer="accepted")

    client._send = accepted  # type: ignore[method-assign]
    for delivery_id in ("d1", "d2", "d3", "d1"):
        await client.deliver(
            delivery_id=delivery_id,
            session_id="repl-1",
            origin="wechat:direct:a:u",
            text="hello",
        )

    assert calls == ["d1", "d2", "d3", "d1"]
    assert list(client._seen_delivery_ids) == ["d3", "d1"]


@pytest.mark.asyncio
async def test_server_deliver_id_collision_does_not_consume_pending_request(tmp_path) -> None:
    delivered: list[GatewayFrame] = []
    client = GatewayIpcClient(tmp_path / "gw.sock", on_deliver=delivered.append)
    pending = asyncio.get_running_loop().create_future()
    client._pending["same-id"] = pending

    client._dispatch_incoming(
        GatewayFrame.deliver(
            delivery_id="same-id",
            session_id="repl-1",
            origin="wechat:direct:a:u",
            text="hello",
        )
    )

    assert [frame.delivery_id for frame in delivered] == ["same-id"]
    assert client._pending["same-id"] is pending
    assert not pending.done()


@pytest.mark.asyncio
async def test_reconnect_re_registers_after_server_restart(tmp_path) -> None:
    """Boundary 2: after the gateway daemon restarts, the client's
    reconnect loop re-establishes the connection and re-registers so the
    origin binding is rebuilt as active.
    """
    gw = _FakeGateway()
    sock = tmp_path / "gw.sock"

    server = GatewayIpcServer(sock, gw)
    await server.start()
    client = GatewayIpcClient(sock, instance_id="repl_main")
    await client.connect()
    await client.register(session_id="repl_main", origin="o1")
    assert gw.binding.get("o1").connection_state == "active"

    # gateway restarts: server tears down, client's socket is dead, binding offline
    await server.close()
    # the client's existing connection is now broken
    client._writer = None
    client._reader = None

    server2 = GatewayIpcServer(sock, gw)
    await server2.start()
    try:
        # the reconnect loop re-connects + re-registers against the new server
        resp = await client.reconnect_until_registered(
            session_id="repl_main",
            origin="o1",
            capabilities=["outbound_text"],
            base_delay=0.01,
            max_delay=0.05,
            max_attempts=5,
        )
        assert resp is not None and resp.ack_layer == "accepted"
        assert gw.binding.get("o1").connection_state == "active"
    finally:
        await client.close()
        await server2.close()


@pytest.mark.asyncio
async def test_no_offline_replay_during_disconnect_gap(tmp_path) -> None:
    """Boundary 3: inbound arriving while the opt-in target is offline is
    rejected (target_offline), NOT queued/replayed by the gateway. Reconnect
    must not surface gap messages.
    """
    gw, handler = _real_gateway(tmp_path)
    sock = tmp_path / "gw.sock"
    server = GatewayIpcServer(sock, gw)
    await server.start()
    try:
        client = GatewayIpcClient(sock, instance_id="repl_main")
        await client.connect()
        await client.register(session_id="repl_main", origin="o1")

        # take the target offline (simulates heartbeat gap / disconnect)
        gw.binding.mark_offline("o1")
        assert gw.binding.get("o1").connection_state == "offline"

        # a DIFFERENT connected client (e.g. a channel adapter) delivers to the
        # gateway while o1's target is offline.
        other = GatewayIpcClient(sock, instance_id="chan")
        await other.connect()
        await other.register(session_id="chan", origin="chan")
        gap_resp = await other.deliver(delivery_id="gap1", session_id="repl_main", origin="o1", text="during-gap")
        # rejected as target_offline (accepted layer, not enqueued) — NOT delivered
        assert gap_resp is not None
        assert gap_resp.ack_layer == "accepted"
        assert "target_offline" in (gap_resp.reason or "")
        assert all(m.text != "during-gap" for m in handler.received)

        # reconnect o1's target: re-bind as active, but the gap message is gone
        gw.binding.bind("o1", SessionTarget(session_id="repl_main", host_type="repl"))
        await other.close()
        await client.close()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_backoff_escalates_on_repeated_connect_failures(monkeypatch, tmp_path) -> None:
    """Happy path: repeated connect failures escalate backoff (1s→...→capped)
    and the loop keeps trying up to max attempts without raising into caller.
    """
    client = GatewayIpcClient(tmp_path / "nope.sock", instance_id="repl_main")

    sleeps: list[float] = []
    connect_calls = [0]

    async def _fake_connect():
        connect_calls[0] += 1
        raise ConnectionRefusedError("no server")

    def _fake_sleep(seconds):
        sleeps.append(seconds)

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(client, "connect", _fake_connect)
    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    await client.reconnect_until_registered(
        session_id="repl_main",
        origin="o1",
        capabilities=["outbound_text"],
        base_delay=1.0,
        max_delay=30.0,
        max_attempts=4,
    )

    # tried max_attempts times, each with escalating (≥ base, ≤ max) backoff
    assert connect_calls[0] == 4
    assert all(1.0 <= s <= 30.0 for s in sleeps)
    # strictly non-decreasing (exponential growth, never shrinks)
    assert sleeps == sorted(sleeps)
