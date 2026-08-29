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

"""Tests for IPC push deliver / client outbound / peer lifecycle (split from test_ipc.py)."""

from __future__ import annotations

# IPC and channel dependencies are migrated in sibling feature branches;
# suppress E0611 until the full clawcodex-ascend series lands.
# pylint: disable=no-name-in-module

import asyncio
import logging
from types import SimpleNamespace

import pytest

from clawcodex_ext.services.im_gateway.binding import BindingPolicy
from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient
from clawcodex_ext.services.im_gateway.ipc_protocol import FrameType, GatewayFrame
from clawcodex_ext.services.im_gateway.ipc_server import GatewayIpcServer
from clawcodex_ext.services.im_gateway.models import (
    AckLayer,
    AckReceipt,
)
from clawcodex_ext.services.channels.capabilities import ProcessingOutcome


class _FakeGateway:
    def __init__(self):
        self.received = []
        self.reloaded = []
        self.sent = []  # outbound messages from OUTBOUND frames
        self.binding = BindingPolicy()

    async def receive(self, message):
        self.received.append(message)
        return AckReceipt(message.message_id or "d1", AckLayer.ENQUEUED, "enqueued")

    def reload_channel(self, name):
        self.reloaded.append(name)
        return name != "missing"

    async def health(self):
        return {"running": True, "channels": ["wechat"], "peers": 0}

    async def send(self, message):
        """OutboundDispatcher stand-in: records OUTBOUND-driven sends."""
        self.sent.append(message)
        from clawcodex_ext.services.channels.results import ChannelSendResult

        return ChannelSendResult.success(getattr(message, "channel", "wechat"))


class _FakeProcessingStatus:
    def __init__(self, *, message_id: str, origin: str) -> None:
        self.entry = SimpleNamespace(message_id=message_id, origin=origin)
        self.completed: list[tuple[str, ProcessingOutcome, str | None]] = []

    def pending(self, message_id: str):
        return self.entry if message_id == self.entry.message_id else None

    async def complete(self, message_id, outcome, *, origin=None):
        self.completed.append((message_id, outcome, origin))
        return True


@pytest.mark.asyncio
async def test_ipc_peer_online_after_register_then_offline(tmp_path) -> None:
    clock = [1000.0]
    gw = _FakeGateway()
    server = GatewayIpcServer(tmp_path / "gw.sock", gw, clock=lambda: clock[0])
    await server.start()
    try:
        async with GatewayIpcClient(tmp_path / "gw.sock", instance_id="repl_main") as client:
            await client.register(session_id="repl_main", origin="o1")
            assert server.is_online("repl_main") is True
            clock[0] = 1000.0 + 120  # beyond heartbeat timeout
            assert server.is_online("repl_main") is False
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_ipc_server_pushes_deliver_to_registered_client(tmp_path) -> None:
    """server.push_deliver writes a DELIVER frame the client receives via on_deliver."""
    gw = _FakeGateway()
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    await server.start()
    delivered: list[GatewayFrame] = []
    try:
        async with GatewayIpcClient(
            tmp_path / "gw.sock",
            instance_id="repl_main",
            on_deliver=lambda f: asyncio.ensure_future(_append(f)),
        ) as client:

            async def _append(f):
                delivered.append(f)

            await client.register(session_id="repl_main", origin="wechat:direct:acct:user_zhao")
            # server pushes an inbound message to that origin
            await server.push_deliver(
                origin="wechat:direct:acct:user_zhao",
                delivery_id="d1",
                text="hello from wechat",
                semantic="newPrompt",
                context_token="ctx_abc",
            )
            await asyncio.sleep(0.1)  # let the read loop dispatch
        assert len(delivered) == 1
        assert delivered[0].type.value == "deliver"
        assert delivered[0].text == "hello from wechat"
        assert delivered[0].origin == "wechat:direct:acct:user_zhao"
        assert delivered[0].context_token == "ctx_abc"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_ipc_push_deliver_noop_when_origin_not_registered(tmp_path) -> None:
    gw = _FakeGateway()
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    await server.start()
    try:
        # no client registered for this origin — push must not raise
        await server.push_deliver(origin="wechat:direct:acct:nobody", delivery_id="d1", text="x")
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_ipc_client_send_outbound_calls_gateway_send(tmp_path) -> None:
    """OUTBOUND frame (client→server) routes to gateway.send → WeChat."""
    gw = _FakeGateway()
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    await server.start()
    try:
        async with GatewayIpcClient(tmp_path / "gw.sock", instance_id="repl_main") as client:
            await client.register(session_id="repl_main", origin="wechat:direct:acct:user_zhao")
            await client.send_outbound(origin="wechat:direct:acct:user_zhao", text="reply from agent")
        await asyncio.sleep(0.05)
        assert len(gw.sent) == 1
        assert gw.sent[0].text == "reply from agent"
        assert gw.sent[0].channel == "wechat"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_ipc_client_send_outbound_accepts_metadata(tmp_path) -> None:
    gw = _FakeGateway()
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    await server.start()
    try:
        async with GatewayIpcClient(tmp_path / "gw.sock", instance_id="repl_main") as client:
            await client.register(session_id="repl_main", origin="wechat:direct:acct:user_zhao")
            await client.send_outbound(
                origin="wechat:direct:acct:user_zhao",
                text="permission prompt",
                metadata={
                    "intent": "permission_approval",
                    "permission": {"options": [{"value": "s", "label": "allow session", "decision": "allow"}]},
                },
                semantic_tags=["approval"],
            )
        await asyncio.sleep(0.05)
        assert len(gw.sent) == 1
        assert gw.sent[0].metadata == {
            "intent": "permission_approval",
            "permission": {"options": [{"value": "s", "label": "allow session", "decision": "allow"}]},
        }
        assert gw.sent[0].semantic_tags == ["approval"]
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_ipc_client_send_outbound_preserves_context_token(tmp_path) -> None:
    gw = _FakeGateway()
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    await server.start()
    try:
        async with GatewayIpcClient(tmp_path / "gw.sock", instance_id="repl_main") as client:
            await client.register(session_id="repl_main", origin="feishu:dm:cli_app:ou_user")
            await client.send_outbound(
                origin="feishu:dm:cli_app:ou_user",
                text="reply from agent",
                context_token="oc_chat",
            )
        await asyncio.sleep(0.05)
        assert len(gw.sent) == 1
        assert gw.sent[0].channel == "feishu"
        assert gw.sent[0].target == "ou_user"
        assert gw.sent[0].context_token == "oc_chat"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_ipc_client_send_outbound_returns_nack_for_wechat_rate_limit(
    tmp_path,
) -> None:
    """WeChat rate-limit sends are observable failures, not hidden ACK/enqueued."""
    from clawcodex_ext.services.channels.results import ChannelSendResult, ErrorCategory

    gw = _FakeGateway()

    async def _rate_limited_send(message):
        gw.sent.append(message)
        return ChannelSendResult.retryable_error(
            "wechat",
            message="rate limited",
            category=ErrorCategory.RATE_LIMIT,
            raw={"retry_after_seconds": 600},
        )

    gw.send = _rate_limited_send
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    await server.start()
    try:
        async with GatewayIpcClient(tmp_path / "gw.sock", instance_id="repl_main") as client:
            await client.register(session_id="repl_main", origin="wechat:direct:acct:user_zhao")
            response = await client.send_outbound(origin="wechat:direct:acct:user_zhao", text="reply from agent")

        assert response is not None
        assert response.type is FrameType.NACK
        assert "rate limited" in (response.reason or "")
        assert len(gw.sent) == 1
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_ipc_client_send_outbound_returns_nack_for_unresolvable_origin(tmp_path) -> None:
    """OUTBOUND NACKs are observable by the client."""
    gw = _FakeGateway()
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    await server.start()
    try:
        async with GatewayIpcClient(tmp_path / "gw.sock", instance_id="orch") as client:
            response = await client.send_outbound(origin="slack:dm:T123:U456", text="reply")

        assert response is not None
        assert response.type is FrameType.NACK
        assert "unresolvable origin" in (response.reason or "")
        assert not gw.sent
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_ipc_client_send_returns_none_on_broken_pipe(tmp_path) -> None:
    """_send must catch ConnectionError (gateway stopped) and return None.

    The gateway and orchestrator are decoupled — either may be stopped
    independently. When the gateway socket is gone, _send must not
    propagate BrokenPipeError; it returns None so the caller can
    reconnect gracefully without a traceback.
    """
    gw = _FakeGateway()
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    await server.start()
    try:
        async with GatewayIpcClient(tmp_path / "gw.sock", instance_id="orch") as client:
            await client.register(session_id="orch", origin="wechat:direct:*:*")
            # Simulate gateway gone: close the server socket so the client's
            # writer.drain() raises BrokenPipeError on the next send.
            await server.close()
            await asyncio.sleep(0.05)  # let the OS propagate the closed socket

            # heartbeat must return None, not raise BrokenPipeError.
            response = await client.heartbeat()
            assert response is None
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_handle_outbound_logs_warning_when_send_exceeds_client_ack_timeout(tmp_path, caplog, monkeypatch) -> None:
    """A gateway.send slower than the client ACK timeout (5s) must log at
    WARNING, not INFO, so gateway.log reconciles with the client's
    "OUTBOUND timed out" line instead of silently showing success.
    """
    from clawcodex_ext.services.im_gateway import ipc_server as ipc_server_mod
    from clawcodex_ext.services.im_gateway.ipc_server import IPC_CLIENT_ACK_TIMEOUT_SECONDS

    gw = _FakeGateway()

    async def _slow_send(message):
        gw.sent.append(message)
        from clawcodex_ext.services.channels.results import ChannelSendResult

        return ChannelSendResult.success(getattr(message, "channel", "wechat"))

    gw.send = _slow_send
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    # Drive send_elapsed past the client ACK timeout via a fake monotonic clock:
    # first call (send_started) returns 100.0, second (elapsed) returns
    # 100.0 + timeout + 1. The actual gateway.send is instantaneous.
    ticks = {"n": 0}

    def _fake_monotonic() -> float:
        ticks["n"] += 1
        if ticks["n"] == 1:
            return 100.0
        return 100.0 + IPC_CLIENT_ACK_TIMEOUT_SECONDS + 1.0

    monkeypatch.setattr(ipc_server_mod.time, "monotonic", _fake_monotonic)
    frame = GatewayFrame.outbound(origin="wechat:direct:acct:user_zhao", text="slow reply")
    with caplog.at_level(logging.WARNING, logger="clawcodex_ext.services.im_gateway.ipc_server"):
        await server._handle_outbound(frame)
    assert len(gw.sent) == 1
    assert any("OUTBOUND send slow" in rec.message and "client ACK timeout" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_handle_outbound_logs_info_when_send_within_client_ack_timeout(tmp_path, caplog) -> None:
    """A fast gateway.send logs at INFO (the normal path), not WARNING."""
    gw = _FakeGateway()
    server = GatewayIpcServer(tmp_path / "gw.sock", gw)
    frame = GatewayFrame.outbound(origin="wechat:direct:acct:user_zhao", text="fast reply")
    with caplog.at_level(logging.INFO, logger="clawcodex_ext.services.im_gateway.ipc_server"):
        await server._handle_outbound(frame)
    assert len(gw.sent) == 1
    assert any("OUTBOUND → send" in rec.message for rec in caplog.records)
    assert not any("OUTBOUND send slow" in rec.message for rec in caplog.records)
