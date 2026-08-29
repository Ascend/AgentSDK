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

"""Tests for the capability model and ChannelAdapter contract."""

from __future__ import annotations

import pytest

from clawcodex_ext.services.channels.capabilities import (
    CapabilityDescriptor,
    CapabilityNotDeclaredError,
    CardUpdateCapability,
    ChannelAdapter,
    ChannelCapability,
    ChannelCapabilitySet,
    InboundActivityContext,
    OutboundCapability,
)
from clawcodex_ext.services.channels.results import ChannelHealth, ValidationResult


def test_capability_enum_values() -> None:
    assert ChannelCapability.OUTBOUND_TEXT.value == "outbound_text"
    assert ChannelCapability.INBOUND_POLLING.value == "inbound_polling"
    assert ChannelCapability.INBOUND_WEBHOOK.value == "inbound_webhook"
    assert ChannelCapability.CONTEXT_REPLY.value == "context_reply"
    assert ChannelCapability.LOGIN_MANAGED.value == "login_managed"


def test_capability_set_of_and_has() -> None:
    caps = ChannelCapabilitySet.of(
        ChannelCapability.OUTBOUND_TEXT,
        ChannelCapability.CONTEXT_REPLY,
    )
    assert caps.has(ChannelCapability.OUTBOUND_TEXT)
    assert caps.has(ChannelCapability.CONTEXT_REPLY)
    assert not caps.has(ChannelCapability.INBOUND_POLLING)


def test_capability_set_descriptor_lookup() -> None:
    desc = CapabilityDescriptor(
        ChannelCapability.OUTBOUND_TEXT,
        supports_markdown=False,
        max_text_length=4000,
    )
    caps = ChannelCapabilitySet.of(
        ChannelCapability.OUTBOUND_TEXT,
        descriptors={ChannelCapability.OUTBOUND_TEXT: desc},
    )
    got = caps.descriptor(ChannelCapability.OUTBOUND_TEXT)
    assert got is not None
    assert got.supports_markdown is False
    assert got.max_text_length == 4000
    assert caps.descriptor(ChannelCapability.INBOUND_POLLING) is None


def test_capability_set_rejects_orphan_descriptor() -> None:
    desc = CapabilityDescriptor(ChannelCapability.INBOUND_POLLING)
    with pytest.raises(ValueError):
        ChannelCapabilitySet.of(
            ChannelCapability.OUTBOUND_TEXT,
            descriptors={ChannelCapability.INBOUND_POLLING: desc},
        )


class _FakeAdapter(ChannelAdapter):
    def __init__(self, caps: ChannelCapabilitySet) -> None:
        self._caps = caps

    @property
    def channel_id(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> ChannelCapabilitySet:
        return self._caps

    def validate_config(self) -> ValidationResult:
        return ValidationResult.ok_result()

    async def health_check(self) -> ChannelHealth:
        return ChannelHealth(healthy=True, channel_id="fake")


def test_channel_adapter_require_capability_passes_when_declared() -> None:
    adapter = _FakeAdapter(ChannelCapabilitySet.of(ChannelCapability.OUTBOUND_TEXT))
    adapter.require_capability(ChannelCapability.OUTBOUND_TEXT)  # no raise


def test_channel_adapter_require_capability_fail_closed() -> None:
    adapter = _FakeAdapter(ChannelCapabilitySet.of(ChannelCapability.OUTBOUND_TEXT))
    with pytest.raises(CapabilityNotDeclaredError):
        adapter.require_capability(ChannelCapability.MEDIA_IMAGE)


def test_outbound_capability_is_structural_protocol() -> None:
    class _Out:
        channel_id = "x"

        async def send(self, message, *, target=None, context_token=None): ...

    assert isinstance(_Out(), OutboundCapability)


def test_card_update_capability_is_structural_protocol() -> None:
    class _Cards:
        channel_id = "cards"
        capabilities = ChannelCapabilitySet.of(ChannelCapability.CARD_UPDATE)

        def last_inbound_context(self):
            return InboundActivityContext(message_id="om_1", chat_id="oc_1")

        async def send_placeholder_card(self, chat_id, card):
            return "om_placeholder"

        async def update_progress_card(self, message_id, card):
            return True

    assert isinstance(_Cards(), CardUpdateCapability)


def test_channel_adapter_default_retry_policy() -> None:
    adapter = _FakeAdapter(ChannelCapabilitySet.of(ChannelCapability.OUTBOUND_TEXT))
    assert adapter.retry_policy.max_attempts >= 1
