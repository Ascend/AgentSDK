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

"""Feishu SDK→gateway inbound translation tests.

``translate_inbound`` consumes the SDK ``InboundMessage`` (already deduped /
content-parsed by ``FeishuChannel``'s ``InboundPipeline``) and applies the V1
p2p-only / optional-allowlist admission before producing a gateway
``InboundMessage``.
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

from types import SimpleNamespace

from clawcodex_ext.services.channels.feishu_events import translate_inbound
from clawcodex_ext.services.channels.feishu_settings import FeishuAppSettings


def _settings() -> FeishuAppSettings:
    return FeishuAppSettings(
        channel_id="feishu",
        connection_mode="websocket",
        app_id="cli_app",
        app_secret="secret",
        allowed_user_open_id="ou_allowed",
        bot_open_id="ou_bot",
    )


def _sdk_inbound(
    *,
    message_id: str = "om_msg_1",
    chat_id: str = "oc_chat",
    chat_type: str = "p2p",
    open_id: str = "ou_allowed",
    text: str = "hello",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        content_text=text,
        create_time=123,
        raw_content_type="text",
        conversation=SimpleNamespace(chat_id=chat_id, chat_type=chat_type),
        sender=SimpleNamespace(open_id=open_id),
    )


def test_feishu_translate_inbound_maps_sdk_inbound_to_gateway_inbound() -> None:
    inbound = translate_inbound(_sdk_inbound(), _settings())

    assert inbound is not None
    assert inbound.origin == "feishu:dm:cli_app:ou_allowed"
    assert inbound.channel == "feishu"
    assert inbound.message_id == "om_msg_1"
    assert inbound.text == "hello"
    assert inbound.context_token == "oc_chat"
    assert inbound.from_user_id == "ou_allowed"
    assert inbound.raw["chat_id"] == "oc_chat"
    assert inbound.raw["create_time"] == 123


def test_feishu_translate_inbound_drops_non_p2p() -> None:
    assert translate_inbound(_sdk_inbound(chat_type="group"), _settings()) is None


def test_feishu_translate_inbound_applies_optional_allowlist() -> None:
    assert translate_inbound(_sdk_inbound(open_id="ou_other"), _settings()) is None


def test_feishu_translate_inbound_drops_self_echo() -> None:
    assert translate_inbound(_sdk_inbound(open_id="ou_bot"), _settings()) is None


def test_feishu_translate_inbound_drops_empty_text() -> None:
    assert translate_inbound(_sdk_inbound(text="   "), _settings()) is None


def test_feishu_translate_inbound_allowlist_empty_admits_all() -> None:
    settings = FeishuAppSettings(
        channel_id="feishu",
        connection_mode="websocket",
        app_id="cli_app",
        app_secret="secret",
        allowed_user_open_id="",
        bot_open_id="ou_bot",
    )
    assert translate_inbound(_sdk_inbound(open_id="ou_anyone"), settings) is not None
