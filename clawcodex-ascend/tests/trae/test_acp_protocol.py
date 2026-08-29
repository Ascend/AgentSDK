#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""P66-A Tests for acp protocol."""

from __future__ import annotations

from extensions.capabilities.acp_protocol import (
    ACPMessage,
    ACPMessageRole,
    ACPMessageType,
    ACPServer,
    ACPSession,
    ACPToolSpec,
    ACPTransport,
)


def test_message_type_enum_values() -> None:
    """Verify message type enum values."""
    assert ACPMessageType.SESSION_CREATE.value == "session/create"
    assert ACPMessageType.MESSAGE_STREAM.value == "message/stream"
    assert ACPMessageType.TOOL_CALL.value == "tool/call"
    assert ACPMessageType.SKILL_INVOKE.value == "skill/invoke"


def test_message_role_enum_values() -> None:
    assert ACPMessageRole.USER.value == "user"
    assert ACPMessageRole.ASSISTANT.value == "assistant"


def test_message_round_trip_dict() -> None:
    """Verify message round trip dict."""
    msg = ACPMessage(
        type=ACPMessageType.MESSAGE_SEND,
        id="m1",
        session_id="s1",
        role=ACPMessageRole.USER,
        content="hello",
        metadata={"k": "v"},
    )
    d = msg.to_dict()
    assert d["type"] == "message/send"
    assert d["role"] == "user"
    assert d["content"] == "hello"

    restored = ACPMessage.from_dict(d)
    assert restored.type == ACPMessageType.MESSAGE_SEND
    assert restored.id == "m1"
    assert restored.role == ACPMessageRole.USER
    assert restored.content == "hello"
    assert restored.metadata == {"k": "v"}


def test_message_from_dict_tolerates_unknown_type() -> None:
    """Verify message from dict tolerates unknown type."""
    msg = ACPMessage.from_dict({"type": "future/method", "id": "x", "content": "c"})
    assert msg.type == ACPMessageType.ERROR
    assert msg.metadata["unknown_type"] == "future/method"
    assert msg.content == "c"


def test_message_from_dict_tolerates_unknown_role() -> None:
    """Verify message from dict tolerates unknown role."""
    msg = ACPMessage.from_dict({"type": "message/send", "role": "developer"})
    assert msg.role == ACPMessageRole.USER


def test_message_default_timestamp_present() -> None:
    """Verify message default timestamp present."""
    msg = ACPMessage(type=ACPMessageType.SESSION_CREATE)
    assert isinstance(msg.timestamp, str)
    assert len(msg.timestamp) > 0
    assert "T" in msg.timestamp


def test_session_append_records_history() -> None:
    """Verify session append records history."""
    session = ACPSession(id="s1", workspace_path="/tmp/ws")
    assert session.messages == []
    msg = ACPMessage(type=ACPMessageType.MESSAGE_SEND, session_id="s1", content="hi")
    session.append(msg)
    assert len(session.messages) == 1
    assert session.messages[0] is msg


def test_session_to_dict_serializes_messages() -> None:
    session = ACPSession(id="s1", workspace_path="/tmp/ws", skills=["a", "b"])
    session.append(ACPMessage(type=ACPMessageType.MESSAGE_SEND, content="x"))
    d = session.to_dict()
    assert d["id"] == "s1"
    assert d["workspace_path"] == "/tmp/ws"
    assert d["skills"] == ["a", "b"]
    assert len(d["messages"]) == 1
    assert d["messages"][0]["type"] == "message/send"


def test_tool_spec_defaults() -> None:
    spec = ACPToolSpec(name="t", description="d")
    assert spec.name == "t"
    assert spec.input_schema == {}


def test_transport_and_server_are_runtime_checkable_protocols() -> None:
    """Verify transport and server are runtime checkable protocols."""
    assert hasattr(ACPTransport, "connect")
    assert hasattr(ACPServer, "create_session")
    assert hasattr(ACPServer, "process_message")
    assert hasattr(ACPServer, "invoke_skill")
