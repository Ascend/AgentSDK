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

# pylint: disable=no-name-in-module
from __future__ import annotations

from clawcodex_ext.services.pipe_ipc import PipeMessage, PipeMessageType, PipePeer


def test_message_round_trip_serializes_enum_as_string() -> None:
    msg = PipeMessage(
        id="msg-1",
        type=PipeMessageType.COMMAND,
        source_id="alice",
        target_id="bob",
        payload={"command": "status"},
        timestamp=123.0,
        ttl=3,
        permission_token="token",
    )

    data = msg.to_dict()
    assert data["type"] == "command"

    restored = PipeMessage.from_dict(data)
    assert restored == msg


def test_message_defaults_are_safe_to_deserialize() -> None:
    restored = PipeMessage.from_dict({"type": "heartbeat", "source_id": "alice"})

    assert restored.type is PipeMessageType.HEARTBEAT
    assert restored.source_id == "alice"
    assert restored.target_id is None
    assert restored.payload == {}
    assert restored.ttl == 16
    assert restored.id


def test_peer_round_trip() -> None:
    peer = PipePeer(
        instance_id="peer-1",
        hostname="host",
        pid=123,
        version="2026.6.24",
        addr="/tmp/pipe.sock",
        transport="uds",
        last_seen=42.0,
        is_master=True,
        capabilities=["pipes", "permissions"],
    )

    assert PipePeer.from_dict(peer.to_dict()) == peer


def test_peer_defaults_are_safe_to_deserialize() -> None:
    peer = PipePeer.from_dict({"instance_id": "peer-1", "hostname": "host", "pid": 123})

    assert peer.version == ""
    assert peer.addr == ""
    assert peer.transport == "uds"
    assert peer.is_master is False
    assert peer.capabilities == []
