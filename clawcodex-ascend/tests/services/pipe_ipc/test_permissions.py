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

# pylint: disable=no-name-in-module

from __future__ import annotations


import pytest

from clawcodex_ext.services.pipe_ipc import PipeMessage, PipeMessageType
from clawcodex_ext.services.pipe_ipc.permissions import PipePermissionForwarder


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[PipeMessage] = []
        self.forwarder: PipePermissionForwarder | None = None

    async def send(self, message: PipeMessage) -> None:
        self.sent.append(message)

    async def grant(self, message: PipeMessage) -> None:
        self.sent.append(message)
        assert self.forwarder is not None
        self.forwarder.handle_permission_response(
            PipeMessage(
                type=PipeMessageType.PERMISSION_GRANT,
                source_id="bob",
                target_id="alice",
                payload={"request_id": message.payload["request_id"]},
            )
        )

    async def deny(self, message: PipeMessage) -> None:
        self.sent.append(message)
        assert self.forwarder is not None
        self.forwarder.handle_permission_response(
            PipeMessage(
                type=PipeMessageType.PERMISSION_DENY,
                source_id="bob",
                target_id="alice",
                payload={"request_id": message.payload["request_id"]},
            )
        )


@pytest.mark.asyncio
async def test_permission_grant_returns_true() -> None:
    transport = FakeTransport()
    forwarder = PipePermissionForwarder("alice", transport.grant)
    transport.forwarder = forwarder

    allowed = await forwarder.request_permission("bob", {"tool": "Bash"}, timeout=0.1)

    assert allowed is True
    assert transport.sent[0].type is PipeMessageType.PERMISSION_REQ
    assert transport.sent[0].payload["request_id"]
    assert forwarder.pending_count == 0


@pytest.mark.asyncio
async def test_permission_deny_returns_false() -> None:
    transport = FakeTransport()
    forwarder = PipePermissionForwarder("alice", transport.deny)
    transport.forwarder = forwarder

    allowed = await forwarder.request_permission("bob", {"tool": "Bash"}, timeout=0.1)

    assert allowed is False


@pytest.mark.asyncio
async def test_permission_timeout_returns_false() -> None:
    transport = FakeTransport()
    forwarder = PipePermissionForwarder("alice", transport.send)

    allowed = await forwarder.request_permission("bob", {"tool": "Bash"}, timeout=0.01)

    assert allowed is False
    assert forwarder.pending_count == 0


@pytest.mark.asyncio
async def test_send_exception_is_not_swallowed() -> None:
    async def failing_send(_: PipeMessage) -> None:
        raise RuntimeError("send failed")

    forwarder = PipePermissionForwarder("alice", failing_send)

    with pytest.raises(RuntimeError, match="send failed"):
        await forwarder.request_permission("bob", {"tool": "Bash"}, timeout=0.1)
    assert forwarder.pending_count == 0


def test_non_permission_response_is_ignored() -> None:
    async def unused_send(_: PipeMessage) -> None:
        raise AssertionError("not called")

    forwarder = PipePermissionForwarder("alice", unused_send)

    handled = forwarder.handle_permission_response(PipeMessage(type=PipeMessageType.REPLY, source_id="bob", payload={}))

    assert handled is False
