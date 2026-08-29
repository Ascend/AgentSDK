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

"""Tests for the stub guidance inbound handler (default host agent placeholder)."""

from __future__ import annotations

import pytest

from clawcodex_ext.services.channels.capabilities import (
    CapabilityDescriptor,
    ChannelAdapter,
    ChannelCapability,
    ChannelCapabilitySet,
)
from clawcodex_ext.services.channels.registry import ChannelAdapterRegistry
from clawcodex_ext.services.channels.results import (
    ChannelHealth,
    ChannelSendResult,
    ValidationResult,
)
from clawcodex_ext.services.im_gateway.capability_gate import CapabilityGate
from clawcodex_ext.services.im_gateway.config import GatewayConfig, ReliabilityConfig
from clawcodex_ext.services.im_gateway.models import InboundMessage
from clawcodex_ext.services.im_gateway.outbound import OutboundDispatcher
from clawcodex_ext.services.im_gateway.store import ReliabilityStore
from clawcodex_ext.services.im_gateway.stub_agent import make_stub_handler


class _FakeOutAdapter(ChannelAdapter):
    def __init__(self, name: str = "wechat") -> None:
        self._name = name
        self._caps = ChannelCapabilitySet.of(
            ChannelCapability.OUTBOUND_TEXT,
            descriptors={
                ChannelCapability.OUTBOUND_TEXT: CapabilityDescriptor(
                    ChannelCapability.OUTBOUND_TEXT, supports_markdown=False
                )
            },
        )
        self.sends: list[tuple] = []  # (message, target)

    @property
    def channel_id(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ChannelCapabilitySet:
        return self._caps

    def validate_config(self) -> ValidationResult:
        return ValidationResult.ok_result()

    async def health_check(self) -> ChannelHealth:
        return ChannelHealth(healthy=True, channel_id=self._name)

    async def send(self, message, *, target=None, context_token=None) -> ChannelSendResult:
        self.sends.append((message, target))
        return ChannelSendResult.success(self._name, provider_receipt="mid_1")


async def _noop_sleep(_delay: float) -> None:
    return None


def _make_outbound(tmp_path, adapter):
    reg = ChannelAdapterRegistry()
    reg.register(adapter)
    store = ReliabilityStore(tmp_path, ReliabilityConfig())
    config = GatewayConfig(state_dir=str(tmp_path), reliability=store._reliability)
    gate = CapabilityGate(reg)
    return OutboundDispatcher(reg, gate, store, config, sleep=_noop_sleep), adapter


@pytest.mark.asyncio
async def test_stub_handler_guides_user_to_bind_repl_or_orchestrator(tmp_path) -> None:
    outbound, adapter = _make_outbound(tmp_path, _FakeOutAdapter("wechat"))
    handler = make_stub_handler(outbound)

    msg = InboundMessage(
        origin="wechat:direct:acct:user_zhao",
        text="你好",
        message_id="m1",
        channel="wechat",
        from_user_id="user_zhao",
    )
    receipt = await handler(msg)

    assert receipt.layer.value == "processed"
    assert len(adapter.sends) == 1
    sent_msg, target = adapter.sends[0]
    assert target == "user_zhao"
    assert "请通过 REPL 或 orchestrator 对 gateway 进行连接配置" in sent_msg.text
    assert "你好" not in sent_msg.text


@pytest.mark.asyncio
async def test_stub_handler_skips_reply_when_target_missing(tmp_path) -> None:
    outbound, adapter = _make_outbound(tmp_path, _FakeOutAdapter("wechat"))
    handler = make_stub_handler(outbound)

    msg = InboundMessage(
        origin="wechat:direct:acct:",
        text="hi",
        message_id="m2",
        channel="wechat",
        from_user_id=None,
    )
    receipt = await handler(msg)
    assert receipt.layer.value == "accepted"
    assert adapter.sends == []  # no reply attempted
