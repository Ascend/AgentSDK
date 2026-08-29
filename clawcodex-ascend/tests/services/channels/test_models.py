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

"""ChannelMessage and ChannelConfig validation tests."""

from __future__ import annotations

import pytest

from clawcodex_ext.services.channels import (
    ChannelConfig,
    ChannelMessage,
    ChannelType,
    MessageLevel,
)


def test_channel_message_defaults() -> None:
    msg = ChannelMessage(text="hello")
    assert msg.text == "hello"
    assert msg.level is MessageLevel.INFO
    assert msg.title is None
    assert msg.markdown is True
    assert msg.attachments is None
    assert msg.metadata is None


def test_channel_message_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        ChannelMessage(text="")


def test_channel_message_rejects_non_string_text() -> None:
    with pytest.raises(TypeError):
        ChannelMessage(text=123)  # type: ignore[arg-type]


def test_channel_message_rejects_oversized_text() -> None:
    with pytest.raises(ValueError):
        ChannelMessage(text="x" * 30_001)


def test_channel_message_rejects_oversized_title() -> None:
    with pytest.raises(ValueError):
        ChannelMessage(text="ok", title="t" * 201)


def test_channel_message_rejects_non_list_attachments() -> None:
    with pytest.raises(TypeError):
        ChannelMessage(text="ok", attachments="not-a-list")  # type: ignore[arg-type]


def test_channel_message_rejects_non_dict_metadata() -> None:
    with pytest.raises(TypeError):
        ChannelMessage(text="ok", metadata=[1, 2, 3])  # type: ignore[arg-type]


def test_channel_message_round_trip() -> None:
    msg = ChannelMessage(
        text="hello world",
        level=MessageLevel.WARN,
        title="alert",
        markdown=False,
        attachments=[{"color": "red"}],
        metadata={"trace_id": "abc"},
    )
    data = msg.to_dict()
    assert data == {
        "text": "hello world",
        "level": "warn",
        "title": "alert",
        "markdown": False,
        "attachments": [{"color": "red"}],
        "metadata": {"trace_id": "abc"},
    }
    assert ChannelMessage.from_dict(data) == msg


def test_channel_message_from_dict_defaults_level() -> None:
    msg = ChannelMessage.from_dict({"text": "hi"})
    assert msg.level is MessageLevel.INFO
    assert msg.markdown is True


def test_channel_message_from_dict_rejects_non_dict() -> None:
    with pytest.raises(ValueError):
        ChannelMessage.from_dict("not-a-dict")  # type: ignore[arg-type]


def test_channel_message_from_dict_rejects_bad_level() -> None:
    with pytest.raises(ValueError):
        ChannelMessage.from_dict({"text": "x", "level": "nope"})


def test_channel_config_defaults() -> None:
    cfg = ChannelConfig(
        type=ChannelType.SLACK,
        webhook_url="https://hooks.example.com/services/T0000/B0000/abcdef0123456789",
        name="alerts",
    )
    assert cfg.enabled is True
    assert cfg.extra is None


def test_channel_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        ChannelConfig(
            type=ChannelType.SLACK,
            webhook_url="https://hooks.example.com/x",
            name="has spaces",
        )
    with pytest.raises(ValueError):
        ChannelConfig(
            type=ChannelType.SLACK,
            webhook_url="https://hooks.example.com/x",
            name="",
        )
    with pytest.raises(ValueError):
        ChannelConfig(
            type=ChannelType.SLACK,
            webhook_url="https://hooks.example.com/x",
            name="a" * 65,
        )


def test_channel_config_rejects_non_string_url() -> None:
    with pytest.raises(ValueError):
        ChannelConfig(
            type=ChannelType.SLACK,
            webhook_url="",  # type: ignore[arg-type]
            name="x",
        )
    with pytest.raises(ValueError):
        ChannelConfig(
            type=ChannelType.SLACK,
            webhook_url=123,  # type: ignore[arg-type]
            name="x",
        )


def test_channel_config_allows_empty_url_for_non_webhook_channel() -> None:
    cfg = ChannelConfig(
        type=ChannelType.FEISHU,
        webhook_url="",
        name="feishu",
        extra={
            "connection_mode": "websocket",
            "app_id": "cli_a",
            "app_secret": "secret",
            "allowed_user_open_id": "ou_user",
        },
    )

    assert cfg.webhook_url == ""
    assert cfg.to_dict()["webhook_url"] == ""


def test_feishu_channel_config_rejects_unknown_connection_mode() -> None:
    with pytest.raises(ValueError, match="connection_mode"):
        ChannelConfig(
            type=ChannelType.FEISHU,
            webhook_url="",
            name="feishu",
            extra={"connection_mode": "typo"},
        )


def test_feishu_webhook_mode_requires_webhook_url() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ChannelConfig(
            type=ChannelType.FEISHU,
            webhook_url="",
            name="feishu",
            extra={"connection_mode": "webhook"},
        )


def test_channel_config_rejects_non_dict_extra() -> None:
    with pytest.raises(TypeError):
        ChannelConfig(
            type=ChannelType.SLACK,
            webhook_url="https://hooks.example.com/x",
            name="x",
            extra=[1, 2],  # type: ignore[arg-type]
        )


def test_channel_config_rejects_wrong_type_enum() -> None:
    with pytest.raises(TypeError):
        ChannelConfig(
            type="slack",  # type: ignore[arg-type]
            webhook_url="https://hooks.example.com/x",
            name="x",
        )


def test_channel_config_round_trip() -> None:
    cfg = ChannelConfig(
        type=ChannelType.DISCORD,
        webhook_url="https://discord.com/api/webhooks/1/abcdef0123456789",
        name="bot-1",
        enabled=False,
        extra={"thread_id": "42"},
    )
    data = cfg.to_dict()
    assert data == {
        "type": "discord",
        "webhook_url": "https://discord.com/api/webhooks/1/abcdef0123456789",
        "name": "bot-1",
        "enabled": False,
        "extra": {"thread_id": "42"},
    }
    assert ChannelConfig.from_dict(data) == cfg


def test_channel_config_from_dict_rejects_non_dict() -> None:
    with pytest.raises(ValueError):
        ChannelConfig.from_dict(None)  # type: ignore[arg-type]


def test_channel_type_enum_values() -> None:
    assert ChannelType.FEISHU.value == "feishu"
    assert ChannelType.SLACK.value == "slack"
    assert ChannelType.DISCORD.value == "discord"
    assert ChannelType.WECHAT.value == "wechat"
    assert ChannelType.MCP_PUSH.value == "mcp_push"


def test_message_level_enum_values() -> None:
    assert MessageLevel.INFO.value == "info"
    assert MessageLevel.WARN.value == "warn"
    assert MessageLevel.ERROR.value == "error"
    assert MessageLevel.SUCCESS.value == "success"
