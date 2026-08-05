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

"""Channels data models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


_CHANNEL_MESSAGE_TEXT_MAX_LENGTH = 30_000


class ChannelType(str, Enum):
    FEISHU = "feishu"
    SLACK = "slack"
    DISCORD = "discord"
    WECHAT = "wechat"
    MCP_PUSH = "mcp_push"


class MessageLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class ChannelMessage:
    text: str
    level: MessageLevel = MessageLevel.INFO
    title: str | None = None
    markdown: bool = True
    attachments: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("ChannelMessage.text must be a string")
        if not self.text:
            raise ValueError("ChannelMessage.text must be non-empty")
        if len(self.text) > _CHANNEL_MESSAGE_TEXT_MAX_LENGTH:
            raise ValueError(f"ChannelMessage.text exceeds {_CHANNEL_MESSAGE_TEXT_MAX_LENGTH} character safety cap")
        if self.title is not None and not isinstance(self.title, str):
            raise TypeError("ChannelMessage.title must be a string or None")
        if self.title is not None and len(self.title) > 200:
            raise ValueError("ChannelMessage.title exceeds 200 character safety cap")
        if self.attachments is not None and not isinstance(self.attachments, list):
            raise TypeError("ChannelMessage.attachments must be a list or None")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("ChannelMessage.metadata must be a dict or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "level": self.level.value,
            "title": self.title,
            "markdown": self.markdown,
            "attachments": list(self.attachments) if self.attachments is not None else None,
            "metadata": dict(self.metadata) if self.metadata is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelMessage:
        if not isinstance(data, dict):
            raise ValueError("ChannelMessage data must be a dict")
        level = data.get("level", MessageLevel.INFO.value)
        if not isinstance(level, str):
            raise ValueError("ChannelMessage.level must be a string")
        return cls(
            text=str(data["text"]),
            level=MessageLevel(level),
            title=data.get("title"),
            markdown=bool(data.get("markdown", True)),
            attachments=data.get("attachments"),
            metadata=data.get("metadata"),
        )


CHANNEL_NAME_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"
_NAME_RE = re.compile(CHANNEL_NAME_PATTERN)
FEISHU_CONNECTION_MODES = frozenset({"webhook", "websocket"})


@dataclass
class ChannelConfig:
    type: ChannelType
    webhook_url: str = ""
    name: str = ""
    enabled: bool = True
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, ChannelType):
            raise TypeError("ChannelConfig.type must be a ChannelType")
        if not isinstance(self.webhook_url, str):
            raise ValueError("ChannelConfig.webhook_url must be a string")
        if self.extra is not None and not isinstance(self.extra, dict):
            raise TypeError("ChannelConfig.extra must be a dict or None")
        mode = str((self.extra or {}).get("connection_mode") or "webhook").lower()
        if self.type is ChannelType.FEISHU and mode not in FEISHU_CONNECTION_MODES:
            raise ValueError("Feishu connection_mode must be one of: " + ", ".join(sorted(FEISHU_CONNECTION_MODES)))
        if not self.webhook_url:
            if not (self.type is ChannelType.FEISHU and mode == "websocket"):
                raise ValueError("ChannelConfig.webhook_url must be a non-empty string")
        if not _NAME_RE.match(self.name or ""):
            raise ValueError(f"ChannelConfig.name must match {CHANNEL_NAME_PATTERN}; got: {self.name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "webhook_url": self.webhook_url,
            "name": self.name,
            "enabled": self.enabled,
            "extra": dict(self.extra) if self.extra is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelConfig:
        if not isinstance(data, dict):
            raise ValueError("ChannelConfig data must be a dict")
        return cls(
            type=ChannelType(str(data["type"])),
            webhook_url=str(data.get("webhook_url") or ""),
            name=str(data["name"]),
            enabled=bool(data.get("enabled", True)),
            extra=data.get("extra"),
        )


__all__ = [
    "CHANNEL_NAME_PATTERN",
    "FEISHU_CONNECTION_MODES",
    "ChannelConfig",
    "ChannelMessage",
    "ChannelType",
    "MessageLevel",
]
