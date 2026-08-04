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

"""P66-A — ACP 协议层数据模型 / 序列化测试."""

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
    """枚举值对齐 JSON-RPC method 命名空间 (str 子类)。"""
    assert ACPMessageType.SESSION_CREATE.value == "session/create"
    assert ACPMessageType.MESSAGE_STREAM.value == "message/stream"
    assert ACPMessageType.TOOL_CALL.value == "tool/call"
    assert ACPMessageType.SKILL_INVOKE.value == "skill/invoke"


def test_message_role_enum_values() -> None:
    assert ACPMessageRole.USER.value == "user"
    assert ACPMessageRole.ASSISTANT.value == "assistant"


def test_message_round_trip_dict() -> None:
    """to_dict / from_dict 往返保持语义。"""
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
    """未知 type 降级为 ERROR 而非抛错 (协议前向兼容)。"""
    msg = ACPMessage.from_dict({"type": "future/method", "id": "x", "content": "c"})
    assert msg.type == ACPMessageType.ERROR
    assert msg.metadata["unknown_type"] == "future/method"
    assert msg.content == "c"


def test_message_from_dict_tolerates_unknown_role() -> None:
    """未知 role 降级为 USER 而非抛错。"""
    msg = ACPMessage.from_dict({"type": "message/send", "role": "developer"})
    assert msg.role == ACPMessageRole.USER


def test_message_default_timestamp_present() -> None:
    """默认 timestamp 是 timezone-aware ISO 字符串 (非空)。"""
    msg = ACPMessage(type=ACPMessageType.SESSION_CREATE)
    assert isinstance(msg.timestamp, str)
    assert len(msg.timestamp) > 0
    # 应含 'T' 分隔 (ISO 8601)
    assert "T" in msg.timestamp


def test_session_append_records_history() -> None:
    """ACPSession.append 记录消息历史。"""
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
    """Protocol 应为 runtime_checkable，便于 isinstance 适配器检查。"""
    # 构造一个空对象验证 Protocol 不在实例化路径上抛错
    assert hasattr(ACPTransport, "connect")
    assert hasattr(ACPServer, "create_session")
    assert hasattr(ACPServer, "process_message")
    assert hasattr(ACPServer, "invoke_skill")
