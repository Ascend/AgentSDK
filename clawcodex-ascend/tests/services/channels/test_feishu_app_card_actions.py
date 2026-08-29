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
from types import SimpleNamespace

import pytest

from clawcodex_ext.services.channels.feishu_app import FeishuAppChannelAdapter
from clawcodex_ext.services.channels.models import ChannelConfig, ChannelType


class _FakeChannel:
    def __init__(self) -> None:
        self.updated_cards: list[dict] = []

    async def update_card(self, message_id: str, card: dict) -> None:
        self.updated_cards.append({"message_id": message_id, "card": card})


def _config() -> ChannelConfig:
    return ChannelConfig(
        type=ChannelType.FEISHU,
        webhook_url="",
        name="feishu",
        extra={
            "connection_mode": "websocket",
            "app_id": "cli_app",
            "app_secret": "secret",
            "allowed_user_open_id": "ou_allowed",
        },
    )


def _card_action_event(*, approval_id: str, nonce: str, choice: str = "y") -> SimpleNamespace:
    return SimpleNamespace(
        message_id="om_card",
        chat_id="oc_chat",
        operator=SimpleNamespace(open_id="ou_allowed"),
        action=SimpleNamespace(
            tag="button",
            value={
                "clawcodex_action": "permission_approval",
                "approval_id": approval_id,
                "nonce": nonce,
                "choice": choice,
            },
        ),
    )


def test_feishu_adapter_exposes_public_inbound_activity_context() -> None:
    adapter = FeishuAppChannelAdapter(_config(), channel_factory=lambda _settings: _FakeChannel())

    assert adapter.last_inbound_context() is None
    adapter._remember_inbound(SimpleNamespace(message_id="om_activity", chat_id="oc_chat"))

    context = adapter.last_inbound_context()
    assert context is not None
    assert context.message_id == "om_activity"
    assert context.chat_id == "oc_chat"


@pytest.mark.asyncio
async def test_feishu_card_click_updates_card_before_slow_gateway_handler() -> None:
    channel = _FakeChannel()
    started = asyncio.Event()
    release = asyncio.Event()
    delivered = []
    adapter = FeishuAppChannelAdapter(_config(), channel_factory=lambda _settings: channel)
    adapter._channel = channel
    adapter._main_loop = asyncio.get_running_loop()

    async def _slow_handler(message) -> None:
        delivered.append(message)
        started.set()
        await release.wait()

    adapter.set_inbound_handler(_slow_handler)
    pending = adapter.approval_manager.create_pending(
        origin="feishu:dm:cli_app:ou_allowed",
        chat_id="oc_chat",
        allowed_user_open_id="ou_allowed",
        choices={"y"},
        ttl_seconds=60,
    )

    task = asyncio.create_task(
        adapter._on_card_action(_card_action_event(approval_id=pending.approval_id, nonce=pending.nonce))
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    try:
        assert channel.updated_cards, "card should be resolved before gateway delivery finishes"
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=1.0)

    assert len(delivered) == 1


@pytest.mark.asyncio
async def test_feishu_session_approval_click_updates_card_as_allowed() -> None:
    channel = _FakeChannel()
    delivered = []
    adapter = FeishuAppChannelAdapter(_config(), channel_factory=lambda _settings: channel)
    adapter._channel = channel
    adapter._main_loop = asyncio.get_running_loop()
    adapter.set_inbound_handler(delivered.append)
    pending = adapter.approval_manager.create_pending(
        origin="feishu:dm:cli_app:ou_allowed",
        chat_id="oc_chat",
        allowed_user_open_id="ou_allowed",
        choices={"y", "s", "n"},
        allow_choices={"y", "s"},
        ttl_seconds=60,
    )

    await adapter._on_card_action(
        _card_action_event(
            approval_id=pending.approval_id,
            nonce=pending.nonce,
            choice="s",
        )
    )

    assert len(delivered) == 1
    assert delivered[0].text == "s"
    assert channel.updated_cards[0]["card"]["header"]["template"] == "green"
    assert "已允许" in channel.updated_cards[0]["card"]["header"]["title"]["content"]
