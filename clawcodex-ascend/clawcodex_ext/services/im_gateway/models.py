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

"""Data models for the IM Message Gateway.

These are the in-process value objects shared by the inbound dispatcher,
session router, outbound dispatcher, and reliability store. They are
deliberately plain dataclasses so they can be serialized to the file-based
reliability store (ndjson/json) without coupling to a particular backend.
"""

from __future__ import annotations

# Sibling clawcodex_ext packages (channels) are migrated in separate branches;
# suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from clawcodex_ext.services.channels.results import CircuitState


class MessageSemantics(str, Enum):
    """The six v1 message semantics plus the unsupported-media non-semantic."""

    NEW_PROMPT = "newPrompt"
    COMMAND = "command"
    FOLLOW_UP = "followUp"
    APPROVAL = "approval"
    INTERRUPT = "interrupt"
    CONTEXT_ONLY = "contextOnly"
    UNSUPPORTED_MEDIA = "unsupportedMedia"


class AckLayer(str, Enum):
    """Layered acknowledgement for inbound delivery."""

    ACCEPTED = "accepted"  # gateway received the inbound
    ENQUEUED = "enqueued"  # target host enqueued it
    PROCESSED = "processed"  # target runtime confirmed processing


WECHAT_DIRECT_ALL_ORIGIN = "wechat:direct:*:*"
FEISHU_DM_ALL_ORIGIN = "feishu:dm:*:*"
IM_DIRECT_ALL_ORIGIN = "im:direct:*:*"


@dataclass(frozen=True)
class OriginKey:
    """Unique inbound origin, e.g. ``wechat:direct:default:user_gz``."""

    value: str

    @classmethod
    def wechat(cls, account_id: str, from_user_id: str) -> OriginKey:
        return cls(f"wechat:direct:{account_id}:{from_user_id}")

    @classmethod
    def wechat_all_direct(cls) -> OriginKey:
        """All private/direct WeChat senders for the single configured WeChat channel."""
        return cls(WECHAT_DIRECT_ALL_ORIGIN)

    @classmethod
    def feishu_all_dm(cls) -> OriginKey:
        """All private/direct Feishu DM senders for the single configured Feishu channel."""
        return cls(FEISHU_DM_ALL_ORIGIN)

    @classmethod
    def im_all_direct(cls) -> OriginKey:
        """All supported private/direct IM senders."""
        return cls(IM_DIRECT_ALL_ORIGIN)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SessionTarget:
    """Where an origin's inbound is routed."""

    session_id: str
    host_type: str = "default"  # default | repl | orchestrator


@dataclass
class InboundMessage:
    """Normalized inbound message from a channel adapter."""

    origin: str
    text: str
    message_id: str
    channel: str
    context_token: str | None = None
    from_user_id: str | None = None
    received_at: float = field(default_factory=time.time)
    semantic: MessageSemantics | None = None
    semantic_tags: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "text": self.text,
            "message_id": self.message_id,
            "channel": self.channel,
            "context_token": self.context_token,
            "from_user_id": self.from_user_id,
            "received_at": self.received_at,
            "semantic": self.semantic.value if self.semantic else None,
            "semantic_tags": list(self.semantic_tags),
            "raw": dict(self.raw) if self.raw else None,
        }


@dataclass
class OutboundMessage:
    """Outbound message the gateway sends to a channel."""

    text: str
    channel: str
    target: str | None = None
    context_token: str | None = None
    level: str = "info"
    title: str | None = None
    markdown: bool = True
    idempotency_key: str | None = None
    semantic_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "channel": self.channel,
            "target": self.target,
            "context_token": self.context_token,
            "level": self.level,
            "title": self.title,
            "markdown": self.markdown,
            "idempotency_key": self.idempotency_key,
            "semantic_tags": list(self.semantic_tags),
            "metadata": dict(self.metadata) if self.metadata else None,
        }


@dataclass
class AckReceipt:
    """Layered ack returned to the inbound caller."""

    delivery_id: str
    layer: AckLayer
    message: str = ""
    notify_user: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivery_id": self.delivery_id,
            "layer": self.layer.value,
            "message": self.message,
            "notify_user": self.notify_user,
        }


__all__ = [
    "AckLayer",
    "AckReceipt",
    "CircuitState",
    "FEISHU_DM_ALL_ORIGIN",
    "IM_DIRECT_ALL_ORIGIN",
    "InboundMessage",
    "MessageSemantics",
    "OriginKey",
    "OutboundMessage",
    "SessionTarget",
    "WECHAT_DIRECT_ALL_ORIGIN",
]
