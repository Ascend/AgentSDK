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

import asyncio
import contextlib
from pathlib import Path

import pytest

from clawcodex_ext.services.pipe_ipc import (
    PipeMessage,
    PipeMessageType,
    UdsPipeClient,
    UdsPipeServer,
)
from clawcodex_ext.services.pipe_ipc.codec import PipeJsonCodec

pytestmark = pytest.mark.skipif(
    not hasattr(asyncio, "start_unix_server"),
    reason="Unix Domain Sockets are not available on this platform",
)


@pytest.mark.asyncio
async def test_client_server_peer_join_and_unicast(tmp_path: Path) -> None:
    server_seen: list[PipeMessage] = []
    server = UdsPipeServer(tmp_path / "pipe.sock", on_message=server_seen.append)
    await server.start()
    alice = UdsPipeClient(tmp_path / "pipe.sock", "alice")
    bob = UdsPipeClient(tmp_path / "pipe.sock", "bob")

    try:
        await alice.connect()
        await bob.connect()
        await asyncio.sleep(0)

        await alice.send(
            PipeMessage(
                type=PipeMessageType.COMMAND,
                source_id="alice",
                target_id="bob",
                payload={"command": "status"},
            )
        )
        received = await bob.receive(timeout=1.0)

        assert received.type is PipeMessageType.COMMAND
        assert received.source_id == "alice"
        assert received.target_id == "bob"
        assert received.payload == {"command": "status"}
        assert any(msg.type is PipeMessageType.PEER_JOIN for msg in server_seen)
    finally:
        await alice.close()
        await bob.close()
        await server.close()


@pytest.mark.asyncio
async def test_broadcast_routes_to_other_clients(tmp_path: Path) -> None:
    server = UdsPipeServer(tmp_path / "pipe.sock")
    await server.start()
    alice = UdsPipeClient(tmp_path / "pipe.sock", "alice")
    bob = UdsPipeClient(tmp_path / "pipe.sock", "bob")
    carol = UdsPipeClient(tmp_path / "pipe.sock", "carol")

    try:
        await alice.connect()
        await bob.connect()
        await carol.connect()
        await asyncio.sleep(0)

        await alice.send(
            PipeMessage(
                type=PipeMessageType.BROADCAST,
                source_id="alice",
                payload={"text": "hello"},
            )
        )

        bob_msg = await bob.receive(timeout=1.0)
        carol_msg = await carol.receive(timeout=1.0)
        assert bob_msg.payload == {"text": "hello"}
        assert carol_msg.payload == {"text": "hello"}
    finally:
        await alice.close()
        await bob.close()
        await carol.close()
        await server.close()


@pytest.mark.asyncio
async def test_server_send_broadcasts_to_clients(tmp_path: Path) -> None:
    server = UdsPipeServer(tmp_path / "pipe.sock")
    await server.start()
    alice = UdsPipeClient(tmp_path / "pipe.sock", "alice")

    try:
        await alice.connect()
        await asyncio.sleep(0)

        await server.send(
            PipeMessage(
                type=PipeMessageType.AGENT_STREAM,
                source_id="master",
                payload={"delta": "hi"},
            )
        )

        received = await alice.receive(timeout=1.0)
        assert received.type is PipeMessageType.AGENT_STREAM
        assert received.payload == {"delta": "hi"}
    finally:
        await alice.close()
        await server.close()


@pytest.mark.asyncio
async def test_server_drops_malformed_frame_and_keeps_connection(tmp_path: Path) -> None:
    seen: list[PipeMessage] = []
    server = UdsPipeServer(tmp_path / "pipe.sock", on_message=seen.append)
    await server.start()

    try:
        reader, writer = await asyncio.open_unix_connection(str(tmp_path / "pipe.sock"))
        try:
            writer.write(b"this-is-not-json\n")
            valid = PipeMessage(
                type=PipeMessageType.HEARTBEAT,
                source_id="probe",
                payload={"k": "v"},
            )
            writer.write(PipeJsonCodec.encode_message(valid))
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

        deadline = asyncio.get_running_loop().time() + 1.0
        while asyncio.get_running_loop().time() < deadline:
            if any(msg.source_id == "probe" for msg in seen):
                break
            await asyncio.sleep(0.02)

        assert any(msg.source_id == "probe" for msg in seen), "server should still process frames after a bad line"
        assert not any(
            msg.type is PipeMessageType.HEARTBEAT and msg.source_id == "probe" and msg.payload.get("error")
            for msg in seen
        ), "user callback must not be invoked for decode failures"
    finally:
        await server.close()
