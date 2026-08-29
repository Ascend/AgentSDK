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

"""Tests for the unified ChannelAdapterRegistry and WebhookChannelAdapter."""

from __future__ import annotations

# Sibling clawcodex_ext packages (channels source) are migrated in separate
# branches; suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module


import pytest

from clawcodex_ext.services.channels.capabilities import (
    CapabilityNotDeclaredError,
    ChannelCapability,
)
from clawcodex_ext.services.channels.models import ChannelConfig, ChannelMessage, ChannelType
from clawcodex_ext.services.channels.registry import (
    WebhookChannelAdapter,
    build_default_registry,
)
from clawcodex_ext.services.channels.results import ErrorCategory, SendStatus
from clawcodex_ext.services.channels.transport import (
    ChannelTransport,
    TransportError,
    TransportResponse,
)


def _cfg(name: str = "ops", ctype: ChannelType = ChannelType.SLACK) -> ChannelConfig:
    return ChannelConfig(
        type=ctype,
        webhook_url="https://hooks.example.com/services/T0/B0/abcdef0123456789",
        name=name,
    )


class _FakeTransport(ChannelTransport):
    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b'{"ok":true}',
        raise_exc: BaseException | None = None,
    ):
        self._status = status
        self._body = body
        self._raise = raise_exc
        self.calls: list[tuple[str, bytes]] = []

    async def post(self, url, body, *, headers=None, timeout=10.0) -> TransportResponse:  # type: ignore[override]
        self.calls.append((url, body))
        if self._raise is not None:
            raise self._raise
        return TransportResponse(status=self._status, body=self._body, headers=dict(headers or {}))


def test_registry_lists_default_types() -> None:
    reg = build_default_registry()
    assert set(reg.list_types()) == {"feishu", "slack", "discord"}


def test_registry_create_and_get_webhook_adapter() -> None:
    reg = build_default_registry()
    adapter = reg.create(_cfg("alerts", ChannelType.SLACK))
    assert adapter.channel_id == "alerts"
    assert reg.get("alerts") is adapter
    assert "alerts" in reg.names()
    # webhook channels declare outbound_text only
    assert adapter.capabilities.has(ChannelCapability.OUTBOUND_TEXT)
    assert not adapter.capabilities.has(ChannelCapability.INBOUND_POLLING)


def test_registry_create_unknown_type_raises() -> None:
    reg = build_default_registry()
    with pytest.raises(KeyError):
        reg.create(_cfg("x", ChannelType.WECHAT))


def test_registry_require_capability_fail_closed() -> None:
    reg = build_default_registry()
    reg.create(_cfg("alerts", ChannelType.SLACK))
    reg.require_capability("alerts", ChannelCapability.OUTBOUND_TEXT)
    with pytest.raises(CapabilityNotDeclaredError):
        reg.require_capability("alerts", ChannelCapability.MEDIA_IMAGE)
    with pytest.raises(CapabilityNotDeclaredError):
        reg.require_capability("alerts", ChannelCapability.INBOUND_POLLING)


def test_registry_inbound_adapters_empty_for_webhook_only() -> None:
    reg = build_default_registry()
    reg.create(_cfg("a", ChannelType.SLACK))
    reg.create(_cfg("b", ChannelType.DISCORD))
    assert reg.inbound_adapters() == []


def test_webhook_adapter_validate_config_ok() -> None:
    reg = build_default_registry()
    adapter = reg.create(_cfg("alerts", ChannelType.SLACK))
    result = adapter.validate_config()
    assert result.ok is True


def test_webhook_adapter_validate_config_rejects_bad_url() -> None:
    cfg = ChannelConfig(
        type=ChannelType.SLACK,
        webhook_url="not-a-url",
        name="bad",
    )
    reg = build_default_registry()
    # BaseChannel construction validates the URL and raises before the adapter
    # is even built; the registry surfaces that as a ValueError.
    with pytest.raises(Exception):
        reg.create(cfg)


@pytest.mark.asyncio
async def test_webhook_adapter_send_success_returns_result() -> None:
    from clawcodex_ext.services.channels.feishu import FeishuChannel

    transport = _FakeTransport(status=200, body=b'{"code":0}')
    base = FeishuChannel(_cfg("feishu1", ChannelType.FEISHU), transport=transport)
    adapter = WebhookChannelAdapter(_cfg("feishu1", ChannelType.FEISHU), base)
    result = await adapter.send(ChannelMessage(text="hello"))
    assert result.ok is True
    assert result.status is SendStatus.SUCCESS
    assert result.channel_id == "feishu1"


@pytest.mark.asyncio
async def test_webhook_adapter_send_transport_error_is_retryable() -> None:
    from clawcodex_ext.services.channels.feishu import FeishuChannel

    transport = _FakeTransport(raise_exc=TransportError("network down"))
    base = FeishuChannel(_cfg("feishu1", ChannelType.FEISHU), transport=transport)
    adapter = WebhookChannelAdapter(_cfg("feishu1", ChannelType.FEISHU), base)
    result = await adapter.send(ChannelMessage(text="hello"))
    assert result.ok is False
    assert result.retryable is True
    assert result.error_category is ErrorCategory.NETWORK


@pytest.mark.asyncio
async def test_webhook_adapter_send_timeout_is_retryable() -> None:
    from clawcodex_ext.services.channels.feishu import FeishuChannel

    transport = _FakeTransport(raise_exc=TransportError("transport timeout: slow"))
    base = FeishuChannel(_cfg("feishu1", ChannelType.FEISHU), transport=transport)
    adapter = WebhookChannelAdapter(_cfg("feishu1", ChannelType.FEISHU), base)
    result = await adapter.send(ChannelMessage(text="hello"))
    assert result.retryable is True
    assert result.error_category is ErrorCategory.TIMEOUT


@pytest.mark.asyncio
async def test_webhook_adapter_send_false_is_nonretryable() -> None:
    from clawcodex_ext.services.channels.feishu import FeishuChannel

    # Feishu returns code != 0 -> BaseChannel.send returns False
    transport = _FakeTransport(status=200, body=b'{"code":19021,"msg":"bad"}')
    base = FeishuChannel(_cfg("feishu1", ChannelType.FEISHU), transport=transport)
    adapter = WebhookChannelAdapter(_cfg("feishu1", ChannelType.FEISHU), base)
    result = await adapter.send(ChannelMessage(text="hello"))
    assert result.ok is False
    assert result.retryable is False


@pytest.mark.asyncio
async def test_webhook_adapter_health_check() -> None:
    from clawcodex_ext.services.channels.slack import SlackChannel

    base = SlackChannel(_cfg("s1", ChannelType.SLACK))
    adapter = WebhookChannelAdapter(_cfg("s1", ChannelType.SLACK), base)
    health = await adapter.health_check()
    assert health.healthy is True
    assert health.channel_id == "s1"
    assert health.circuit_state == "closed"


def test_registry_remove() -> None:
    reg = build_default_registry()
    reg.create(_cfg("alerts", ChannelType.SLACK))
    assert reg.remove("alerts") is True
    assert reg.get("alerts") is None
    assert reg.remove("alerts") is False
