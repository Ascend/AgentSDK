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

# ruff: noqa: UP009

"""Focused migration tests for orchestrator gateway delivery outcomes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from clawcodex_ext.services.im_gateway.ipc_client import (
    GatewayIpcClient,
    OutboundSendOutcome,
)
from clawcodex_ext.services.im_gateway.ipc_protocol import FrameType, GatewayFrame
from extensions.orchestrator.im_gateway_client import (
    OrchestratorGatewayClient,
    OrchestratorHandlers,
)


def _handlers() -> OrchestratorHandlers:
    return OrchestratorHandlers(
        queue_pending_message=lambda _issue, _text: None,
        control_verb=lambda _verb, _issue: None,
        issue_inject=lambda _issue, _hint: None,
        operator_hints=lambda _issue, _text: None,
        agent_intent=lambda _verb, _issue: None,
        issue_cli=lambda _verb, _issue, _payload: None,
        bridge_interrupt=lambda _issue, _payload: None,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pending_outbound_limit": 0},
        {"pending_flush_batch_size": 0},
        {"pending_flush_batch_size": True},
        {"pending_flush_timeout_seconds": 0},
        {"pending_flush_timeout_seconds": float("inf")},
        {"pending_retry_base_seconds": -1},
        {"pending_retry_max_seconds": float("nan")},
        {"pending_retry_base_seconds": 2, "pending_retry_max_seconds": 1},
    ],
)
def test_public_limits_are_validated(kwargs) -> None:
    with pytest.raises(ValueError):
        OrchestratorGatewayClient(_handlers(), **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "queued"),
    [
        ("not_sent", True),
        ("ambiguous_timeout", False),
    ],
)
async def test_retry_policy_uses_typed_transport_outcome(outcome: str, queued: bool) -> None:
    class _Ipc:
        def __init__(self) -> None:
            self.on_deliver = None

        async def send_outbound_result(self, *, origin, text):
            return SimpleNamespace(outcome=SimpleNamespace(value=outcome), response=None)

    client = OrchestratorGatewayClient(_handlers(), ipc_client=_Ipc(), origin="im:direct:*:*")

    await client.send_outbound("event text")

    assert bool(client._pending_outbound) is queued


@pytest.mark.asyncio
async def test_pending_flush_continues_until_empty() -> None:
    class _Ipc:
        def __init__(self) -> None:
            self.on_deliver = None
            self.sent: list[str] = []

        async def send_outbound(self, *, origin, text):
            self.sent.append(text)
            return GatewayFrame.ack(delivery_id=text, layer="processed", message="sent")

    ipc = _Ipc()
    client = OrchestratorGatewayClient(
        _handlers(),
        ipc_client=ipc,
        origin="im:direct:*:*",
        pending_flush_batch_size=2,
    )
    for index in range(5):
        client._queue_pending_outbound(f"event {index}")

    await client._flush_pending_outbound()
    for _ in range(50):
        if not client._pending_outbound:
            break
        await asyncio.sleep(0.01)

    assert ipc.sent == [f"event {index}" for index in range(5)]
    assert list(client._pending_outbound) == []


@pytest.mark.asyncio
async def test_nack_schedules_retry_without_new_event() -> None:
    class _Ipc:
        def __init__(self) -> None:
            self.on_deliver = None
            self.calls = 0

        async def send_outbound(self, *, origin, text):
            self.calls += 1
            if self.calls == 1:
                return GatewayFrame.nack(delivery_id=text, reason="retry")
            return GatewayFrame.ack(delivery_id=text, layer="processed", message="sent")

    ipc = _Ipc()
    client = OrchestratorGatewayClient(
        _handlers(),
        ipc_client=ipc,
        origin="im:direct:*:*",
        pending_retry_base_seconds=0.001,
        pending_retry_max_seconds=0.001,
    )

    await client.send_outbound("event text")
    for _ in range(50):
        if ipc.calls == 2:
            break
        await asyncio.sleep(0.01)

    assert ipc.calls == 2
    assert list(client._pending_outbound) == []


@pytest.mark.asyncio
async def test_flush_timeout_keeps_event_queued() -> None:
    class _Ipc:
        def __init__(self) -> None:
            self.on_deliver = None
            self.blocker = asyncio.Event()
            self.cancelled = False

        async def send_outbound(self, *, origin, text):
            try:
                await self.blocker.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return GatewayFrame.ack(delivery_id=text, layer="processed", message="sent")

    ipc = _Ipc()
    client = OrchestratorGatewayClient(
        _handlers(),
        ipc_client=ipc,
        origin="im:direct:*:*",
        pending_flush_timeout_seconds=0.01,
    )
    client._queue_pending_outbound("event text")

    await client._flush_pending_outbound()

    assert list(client._pending_outbound) == ["event text"]
    assert not ipc.cancelled

    ipc.blocker.set()
    task = client._active_flush_task
    assert task is not None
    await asyncio.wait_for(task, timeout=1)

    assert list(client._pending_outbound) == []


def test_command_reply_is_english_and_bounded() -> None:
    reply = OrchestratorGatewayClient._format_command_reply("/issue list", 0, "x" * 7000, "")
    failure = OrchestratorGatewayClient._format_command_reply("/issue list", 2, "", "")

    assert reply.startswith("Command completed: /issue list")
    assert reply.endswith("...")
    assert len(reply) < 6100
    assert failure == "Command failed (2): /issue list"


@pytest.mark.asyncio
async def test_ipc_result_is_not_sent_while_disconnected(tmp_path) -> None:
    client = GatewayIpcClient(tmp_path / "gw.sock")

    result = await client.send_outbound_result(origin="wechat:direct:a:u", text="reply")

    assert result.outcome is OutboundSendOutcome.NOT_SENT
    assert result.response is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (None, OutboundSendOutcome.AMBIGUOUS_TIMEOUT),
        (
            GatewayFrame.nack(delivery_id="d1", reason="private rejection"),
            OutboundSendOutcome.REJECTED,
        ),
        (
            GatewayFrame.ack(delivery_id="d1", layer="processed", message="sent"),
            OutboundSendOutcome.ACCEPTED,
        ),
    ],
)
async def test_ipc_result_classifies_response(tmp_path, monkeypatch, response, expected) -> None:
    client = GatewayIpcClient(tmp_path / "gw.sock")
    client._writer = SimpleNamespace()

    async def _send(_frame, *, raise_on_connection_error=False):
        assert raise_on_connection_error
        return response

    monkeypatch.setattr(client, "_send", _send)

    result = await client.send_outbound_result(origin="wechat:direct:a:u", text="reply")

    assert result.outcome is expected
    assert result.response is response


@pytest.mark.asyncio
async def test_send_cancellation_cleans_pending_keys(tmp_path) -> None:
    class _Writer:
        def write(self, _data) -> None:
            return None

        async def drain(self) -> None:
            return None

    client = GatewayIpcClient(tmp_path / "gw.sock")
    client._writer = _Writer()
    client._write_lock = asyncio.Lock()
    task = asyncio.create_task(client._send(GatewayFrame.heartbeat(session_id="cancel")))
    for _ in range(20):
        if client._pending:
            break
        await asyncio.sleep(0)

    assert client._pending
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client._pending == {}


@pytest.mark.asyncio
async def test_rejection_log_redacts_remote_reason(tmp_path, monkeypatch, caplog) -> None:
    client = GatewayIpcClient(tmp_path / "gw.sock")
    client._writer = SimpleNamespace()

    async def _send(_frame, *, raise_on_connection_error=False):
        return GatewayFrame.nack(delivery_id="d1", reason="private rejection")

    monkeypatch.setattr(client, "_send", _send)
    with caplog.at_level("WARNING", logger="clawcodex_ext.services.im_gateway.ipc_client"):
        result = await client.send_outbound_result(origin="wechat:direct:a:u", text="reply")

    assert result.response is not None and result.response.type is FrameType.NACK
    assert "reason_fingerprint=" in caplog.text
    assert "private rejection" not in caplog.text
