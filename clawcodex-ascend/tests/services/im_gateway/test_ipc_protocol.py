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

"""Tests for the Gateway IPC frame protocol."""

from __future__ import annotations

import pytest

from clawcodex_ext.services.im_gateway.ipc_protocol import (
    PROTOCOL_VERSION,
    FrameType,
    GatewayFrame,
    constant_time_eq,
)


def test_register_frame_roundtrip() -> None:
    f = GatewayFrame.register(
        session_id="repl_main",
        origin="wechat:direct:default:user_gz",
        capabilities=["outbound_text"],
        token="t0",
    )
    raw = f.encode()
    assert raw.endswith(b"\n")
    back = GatewayFrame.decode(raw)
    assert back.type is FrameType.REGISTER
    assert back.session_id == "repl_main"
    assert back.origin == "wechat:direct:default:user_gz"
    assert back.capabilities == ["outbound_text"]
    assert back.token == "t0"
    assert back.protocol_version == PROTOCOL_VERSION


def test_deliver_and_ack_frames() -> None:
    d = GatewayFrame.deliver(
        delivery_id="d1",
        session_id="s1",
        origin="o1",
        text="hi",
        semantic="followUp",
        deadline_ms=5000,
    )
    assert d.type is FrameType.DELIVER
    assert d.deadline_ms == 5000
    a = GatewayFrame.ack(delivery_id="d1", layer="enqueued", message="q pos 2")
    assert a.type is FrameType.ACK
    assert a.ack_layer == "enqueued"
    n = GatewayFrame.nack(delivery_id="d1", reason="target_offline")
    assert n.type is FrameType.NACK
    assert n.reason == "target_offline"


def test_heartbeat_and_event_and_unregister() -> None:
    assert GatewayFrame.heartbeat(session_id="s").type is FrameType.HEARTBEAT
    e = GatewayFrame.event(event_type="issue.failed", payload={"id": "AGENTSDK-15"})
    assert e.type is FrameType.EVENT
    assert e.payload == {"id": "AGENTSDK-15"}
    assert GatewayFrame(type=FrameType.UNREGISTER, session_id="s").type is FrameType.UNREGISTER


def test_outbound_frame_roundtrip() -> None:
    """OUTBOUND (client→server) carries a reply text back to a WeChat origin."""
    f = GatewayFrame.outbound(
        origin="wechat:direct:acct:user_zhao",
        text="reply text",
        metadata={"intent": "permission_approval"},
        semantic_tags=["approval"],
        in_reply_to="om_inbound",
    )
    assert f.type is FrameType.OUTBOUND
    assert f.origin == "wechat:direct:acct:user_zhao"
    assert f.text == "reply text"
    assert f.metadata == {"intent": "permission_approval"}
    assert f.semantic_tags == ["approval"]
    assert f.in_reply_to == "om_inbound"
    # round-trips through encode/decode
    back = GatewayFrame.decode(f.encode())
    assert back.type is FrameType.OUTBOUND
    assert back.origin == f.origin
    assert back.text == f.text
    assert back.metadata == f.metadata
    assert back.semantic_tags == f.semantic_tags
    assert back.in_reply_to == "om_inbound"


def test_processing_complete_event_roundtrip() -> None:
    frame = GatewayFrame.processing_complete(
        message_id="om_inbound",
        outcome="cancelled",
        reason="user stopped",
    )

    back = GatewayFrame.decode(frame.encode())
    assert back.type is FrameType.EVENT
    assert back.event_type == "processing.complete"
    assert back.payload == {
        "message_id": "om_inbound",
        "outcome": "cancelled",
        "reason": "user stopped",
    }
    assert back.protocol_version == PROTOCOL_VERSION


def test_decode_rejects_bad_type() -> None:
    with pytest.raises(ValueError):
        GatewayFrame.decode(b'{"type": "bogus"}')
    # valid type decodes fine
    assert GatewayFrame.decode(b'{"type": "register", "session_id": "s"}').type is FrameType.REGISTER


def test_decode_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        GatewayFrame.decode(b"[1,2,3]")


def test_constant_time_eq() -> None:
    assert constant_time_eq("abc", "abc") is True
    assert constant_time_eq("abc", "abx") is False
    assert constant_time_eq(None, None) is True
    assert constant_time_eq("abc", None) is False


def test_frame_omits_empty_fields() -> None:
    f = GatewayFrame(type=FrameType.HEARTBEAT, session_id="s")
    d = f.to_dict()
    assert "capabilities" not in d
    assert "text" not in d
    assert d["type"] == "heartbeat"
