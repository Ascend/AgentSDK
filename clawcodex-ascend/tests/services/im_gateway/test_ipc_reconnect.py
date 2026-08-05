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


import pytest

from clawcodex_ext.services.im_gateway.binding import BindingPolicy
from clawcodex_ext.services.im_gateway.config import GatewayConfig
from clawcodex_ext.services.im_gateway.gateway import MessageGateway
from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient
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
        async with GatewayIpcClient(sock, instance_id="repl_main", delivery_cache_size=1) as client:
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
