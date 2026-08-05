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

"""Tests for the MessageGateway facade."""

from __future__ import annotations

# pylint: disable=no-name-in-module  # Split migration branches provide gateway modules.

import pytest

from clawcodex_ext.services.channels.capabilities import (
    CapabilityDescriptor,
    ChannelAdapter,
    ChannelCapability,
    ChannelCapabilitySet,
)
from clawcodex_ext.services.channels.models import ChannelConfig, ChannelMessage, ChannelType
from clawcodex_ext.services.channels.registry import ChannelAdapterRegistry
from clawcodex_ext.services.channels.results import (
    ChannelHealth,
    ChannelSendResult,
    ValidationResult,
)
from clawcodex_ext.services.im_gateway.config import CommandAllowlistConfig, GatewayConfig
from clawcodex_ext.services.im_gateway.gateway import MessageGateway
from clawcodex_ext.services.im_gateway.models import (
    IM_DIRECT_ALL_ORIGIN,
    InboundMessage,
    MessageSemantics,
    OutboundMessage,
    SessionTarget,
)


class _FakeAdapter(ChannelAdapter):
    def __init__(self, name: str = "fake") -> None:
        self._name = name
        self._caps = ChannelCapabilitySet.of(
            ChannelCapability.OUTBOUND_TEXT,
            descriptors={
                ChannelCapability.OUTBOUND_TEXT: CapabilityDescriptor(
                    ChannelCapability.OUTBOUND_TEXT, supports_markdown=False
                )
            },
        )
        self.sends: list[ChannelMessage] = []
        self.send_calls: list[dict] = []

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
        self.sends.append(message)
        self.send_calls.append({"message": message, "target": target, "context_token": context_token})
        return ChannelSendResult.success(self._name, provider_receipt="r")


def _gateway(tmp_path, *, adapter: _FakeAdapter | None = None) -> MessageGateway:
    reg = ChannelAdapterRegistry()
    if adapter is not None:
        reg.register(adapter)
    cfg = GatewayConfig(state_dir=str(tmp_path))
    return MessageGateway(cfg, registry=reg)


def test_gateway_applies_configured_command_allowlists(tmp_path) -> None:
    cfg = GatewayConfig(
        state_dir=str(tmp_path),
        command_allowlists=CommandAllowlistConfig(
            repl=("/model",),
            orchestrator=("/issue takeover",),
        ),
    )

    gateway = MessageGateway(cfg, registry=ChannelAdapterRegistry())

    assert gateway.inbound._repl_allowed_commands == {"/model"}
    assert gateway.inbound._orchestrator_allowed_commands == {"/issue takeover"}


class _FakeInboundAdapter(_FakeAdapter):
    """Inbound adapter whose account_status flips to connected after N polls."""

    def __init__(self, name: str = "fake-in", *, connect_after: int = 0) -> None:
        super().__init__(name)
        self._caps = ChannelCapabilitySet.of(
            ChannelCapability.OUTBOUND_TEXT,
            ChannelCapability.INBOUND_POLLING,
        )
        self._polls = 0
        self._connect_after = connect_after

    def set_inbound_handler(self, handler) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health_check(self) -> ChannelHealth:
        self._polls += 1
        status = "websocket:connected" if self._polls > self._connect_after else "websocket:reconnecting"
        return ChannelHealth(
            healthy=self._polls > self._connect_after,
            channel_id=self._name,
            account_status=status,
        )


@pytest.mark.asyncio
async def test_gateway_wait_channels_ready_returns_when_connected(tmp_path) -> None:
    adapter = _FakeInboundAdapter("feishu", connect_after=2)
    gw = _gateway(tmp_path, adapter=adapter)
    # Simulate gateway having attached it as inbound.
    gw._inbound_adapters.append(adapter)

    result = await gw.wait_channels_ready(timeout=5.0)

    assert result == {"feishu": "websocket:connected"}


@pytest.mark.asyncio
async def test_gateway_wait_channels_ready_times_out_degraded(tmp_path) -> None:
    adapter = _FakeInboundAdapter("feishu", connect_after=1000)  # never connects
    gw = _gateway(tmp_path, adapter=adapter)
    gw._inbound_adapters.append(adapter)

    result = await gw.wait_channels_ready(timeout=1.5)

    assert result["feishu"] == "websocket:reconnecting"


@pytest.mark.asyncio
async def test_gateway_wait_channels_ready_does_not_match_disconnected_substring(tmp_path) -> None:
    adapter = _FakeInboundAdapter("feishu", connect_after=1000)

    async def disconnected_health() -> ChannelHealth:
        return ChannelHealth(
            healthy=False,
            channel_id="feishu",
            account_status="websocket:disconnected",
        )

    adapter.health_check = disconnected_health  # type: ignore[method-assign]
    gw = _gateway(tmp_path, adapter=adapter)
    gw._inbound_adapters.append(adapter)

    result = await gw.wait_channels_ready(timeout=0)

    assert result == {"feishu": "websocket:disconnected"}


@pytest.mark.asyncio
async def test_gateway_send_uses_outbound_dispatcher(tmp_path) -> None:
    adapter = _FakeAdapter("wechat-main")
    gw = _gateway(tmp_path, adapter=adapter)
    result = await gw.send(OutboundMessage(text="hello", channel="wechat-main"))
    assert result.ok is True
    assert len(adapter.sends) == 1


@pytest.mark.asyncio
async def test_gateway_broadcast(tmp_path) -> None:
    a1 = _FakeAdapter("a")
    a2 = _FakeAdapter("b")
    reg = ChannelAdapterRegistry()
    reg.register(a1)
    reg.register(a2)
    gw = MessageGateway(GatewayConfig(state_dir=str(tmp_path)), registry=reg)
    results = await gw.broadcast(OutboundMessage(text="hi", channel="a"))
    assert results["a"].ok and results["b"].ok


@pytest.mark.asyncio
async def test_gateway_unhandled_inbound_remains_retryable_and_classifies(tmp_path) -> None:
    gw = _gateway(tmp_path)
    msg = InboundMessage(origin="wechat:direct:default:u", text="hello", message_id="m1", channel="c")
    r1 = await gw.receive(msg)
    assert r1.message != "duplicate; skipped"
    r2 = await gw.receive(msg)
    assert r2.message != "duplicate; skipped"
    assert msg.semantic is MessageSemantics.NEW_PROMPT


@pytest.mark.asyncio
async def test_gateway_inbound_pushes_to_opt_in_bound_origin(tmp_path) -> None:
    """When an origin is bound to an opt-in peer, dispatch pushes via IPC,
    NOT the default stub handler.
    """
    gw = _gateway(tmp_path)
    pushed: list[InboundMessage] = []

    async def _push(msg):
        pushed.append(msg)
        return True

    gw.set_push_handler(_push)
    # bind the origin to a REPL opt-in target
    gw.binding.bind(
        "wechat:direct:default:u",
        SessionTarget(session_id="repl_main", host_type="repl"),
    )
    handler_calls: list[InboundMessage] = []
    gw.set_handler(lambda m: handler_calls.append(m) or _ack())

    async def _ack():
        from clawcodex_ext.services.im_gateway.models import AckLayer, AckReceipt

        return AckReceipt("d", AckLayer.PROCESSED, "stub")

    msg = InboundMessage(origin="wechat:direct:default:u", text="hi", message_id="m1", channel="wechat-main")
    await gw.receive(msg)
    assert len(pushed) == 1  # pushed to the opt-in peer
    assert pushed[0].text == "hi"
    assert not handler_calls  # default handler NOT called (opt-in overrides)


@pytest.mark.asyncio
async def test_gateway_inbound_pushes_feishu_to_generic_opt_in_binding(tmp_path) -> None:
    """A channel-neutral opt-in binding must catch Feishu DM origins."""
    gw = _gateway(tmp_path)
    pushed: list[InboundMessage] = []

    async def _push(msg):
        pushed.append(msg)
        return True

    gw.set_push_handler(_push)
    gw.binding.bind(
        IM_DIRECT_ALL_ORIGIN,
        SessionTarget(session_id="repl_main", host_type="repl"),
    )

    msg = InboundMessage(
        origin="feishu:dm:cli_app:ou_user",
        text="hi",
        message_id="m-feishu",
        channel="feishu",
        context_token="oc_chat",
    )
    ack = await gw.receive(msg)

    assert ack.message == "pushed to opt-in peer"
    assert len(pushed) == 1
    assert pushed[0].origin == "feishu:dm:cli_app:ou_user"
    assert pushed[0].context_token == "oc_chat"


@pytest.mark.asyncio
async def test_gateway_notifies_feishu_sender_when_repl_command_is_blocked(tmp_path) -> None:
    adapter = _FakeAdapter("feishu")
    gw = _gateway(tmp_path, adapter=adapter)
    pushed: list[InboundMessage] = []

    async def _push(msg):
        pushed.append(msg)
        return True

    gw.set_push_handler(_push)
    gw.binding.bind(
        IM_DIRECT_ALL_ORIGIN,
        SessionTarget(session_id="repl_main", host_type="repl"),
    )
    msg = InboundMessage(
        origin="feishu:dm:cli_app:ou_user",
        text="/exit",
        message_id="m-feishu-blocked",
        channel="feishu",
        context_token="oc_chat",
        from_user_id="ou_user",
    )

    ack = await gw._on_inbound(msg)

    assert not pushed
    assert getattr(ack, "notify_user", False) is True
    assert len(adapter.send_calls) == 1
    call = adapter.send_calls[0]
    assert call["target"] == "ou_user"
    assert call["context_token"] == "oc_chat"
    assert "/exit" in call["message"].text


@pytest.mark.asyncio
async def test_gateway_notifies_feishu_sender_when_orchestrator_command_is_blocked(
    tmp_path,
) -> None:
    adapter = _FakeAdapter("feishu")
    gw = _gateway(tmp_path, adapter=adapter)
    pushed: list[InboundMessage] = []

    async def _push(msg):
        pushed.append(msg)
        return True

    gw.set_push_handler(_push)
    gw.binding.bind(
        IM_DIRECT_ALL_ORIGIN,
        SessionTarget(session_id="orch_main", host_type="orchestrator"),
    )
    msg = InboundMessage(
        origin="feishu:dm:cli_app:ou_user",
        text="/server stop",
        message_id="m-feishu-orch-blocked",
        channel="feishu",
        context_token="oc_chat",
        from_user_id="ou_user",
    )

    ack = await gw._on_inbound(msg)

    assert not pushed
    assert getattr(ack, "notify_user", False) is True
    assert ack.message == "Command /server stop is not supported."
    assert len(adapter.send_calls) == 1
    call = adapter.send_calls[0]
    assert call["target"] == "ou_user"
    assert call["context_token"] == "oc_chat"
    assert call["message"].text == "Command /server stop is not supported."


@pytest.mark.asyncio
async def test_gateway_inbound_default_origin_still_uses_handler(tmp_path) -> None:
    """Unbound (default) origins still go to the stub/handler, not push."""
    gw = _gateway(tmp_path)
    pushed: list[InboundMessage] = []

    async def _push(msg):
        pushed.append(msg)
        return True

    gw.set_push_handler(_push)
    handler_calls: list[InboundMessage] = []
    gw.set_handler(lambda m: handler_calls.append(m) or _ack())

    async def _ack():
        from clawcodex_ext.services.im_gateway.models import AckLayer, AckReceipt

        return AckReceipt("d", AckLayer.PROCESSED, "stub")

    msg = InboundMessage(origin="wechat:direct:default:u", text="hi", message_id="m1", channel="wechat-main")
    await gw.receive(msg)
    assert not pushed  # no opt-in binding → no push
    assert len(handler_calls) == 1  # default handler called


@pytest.mark.asyncio
async def test_gateway_inbound_classifies_slash_as_command(tmp_path) -> None:
    gw = _gateway(tmp_path)
    msg = InboundMessage(origin="o", text="/agent retry AGENTSDK-15", message_id="m1", channel="c")
    await gw.receive(msg)
    assert msg.semantic is MessageSemantics.COMMAND


@pytest.mark.asyncio
async def test_gateway_reload_channel_rebuilds(tmp_path) -> None:
    # registry with a fake factory for "slack" type
    reg = ChannelAdapterRegistry()

    def _factory(cfg: ChannelConfig) -> _FakeAdapter:
        return _FakeAdapter(cfg.name)

    reg.register_type(ChannelType.SLACK, _factory)
    cfg = GatewayConfig(state_dir=str(tmp_path))
    cfg.channels.append(
        ChannelConfig(
            type=ChannelType.SLACK,
            webhook_url="https://hooks.example.com/x",
            name="slack-ops",
        )
    )
    gw = MessageGateway(cfg, registry=reg)
    assert gw.reload_channel("slack-ops") is True
    assert gw.registry.get("slack-ops") is not None
    assert gw.reload_channel("nope") is False


def test_gateway_normalizes_duplicate_channel_types_before_runtime_load(tmp_path) -> None:
    reg = ChannelAdapterRegistry()

    def _factory(cfg: ChannelConfig) -> _FakeAdapter:
        return _FakeAdapter(cfg.name)

    reg.register_type(ChannelType.SLACK, _factory)
    cfg = GatewayConfig(state_dir=str(tmp_path))
    cfg.channels = [
        ChannelConfig(
            type=ChannelType.SLACK,
            webhook_url="https://hooks.example.com/old",
            name="slack-old",
        ),
        ChannelConfig(
            type=ChannelType.SLACK,
            webhook_url="https://hooks.example.com/new",
            name="slack-new",
        ),
    ]

    gw = MessageGateway(cfg, registry=reg)

    assert gw.registry.names() == ["slack-new"]
    assert [c.name for c in gw.config.channels] == ["slack-new"]


@pytest.mark.asyncio
async def test_gateway_health(tmp_path) -> None:
    gw = _gateway(tmp_path, adapter=_FakeAdapter("wechat-main"))
    health = await gw.health()
    assert health["running"] is False
    assert "wechat-main" in health["channels"]
    assert health["outbox_pending"] == 0


@pytest.mark.asyncio
async def test_gateway_stop_logs_stopped_once_when_called_concurrently(tmp_path, caplog) -> None:
    class _SlowStopAdapter(_FakeInboundAdapter):
        async def stop(self) -> None:
            await __import__("asyncio").sleep(0.05)

    adapter = _SlowStopAdapter("feishu")
    gw = _gateway(tmp_path, adapter=adapter)
    gw._inbound_adapters.append(adapter)
    await gw.start()

    caplog.set_level("INFO", logger="clawcodex_ext.services.im_gateway.gateway")
    await __import__("asyncio").gather(gw.stop(), gw.stop())

    stopped = [
        record
        for record in caplog.records
        if record.name == "clawcodex_ext.services.im_gateway.gateway" and record.getMessage() == "gateway stopped"
    ]
    assert len(stopped) == 1


def test_gateway_loads_wechat_channel_from_config(tmp_path) -> None:
    cfg = GatewayConfig(state_dir=str(tmp_path))
    cfg.channels.append(
        ChannelConfig(
            type=ChannelType.WECHAT,
            webhook_url="https://ilinkai.weixin.qq.com/dummy",
            name="wechat",
            enabled=True,
            extra={
                "base_url": "https://ilinkai.weixin.qq.com",
                "account_id": "default",
                "allowed_users": [],
            },
        )
    )
    gw = MessageGateway(cfg)
    adapter = gw.registry.get("wechat")
    assert adapter is not None
    assert adapter.capabilities.has(ChannelCapability.INBOUND_POLLING)
    # WeChat adapter is registered as an inbound adapter
    assert any(a.channel_id == "wechat" for a in gw._inbound_adapters)
    # not logged in (no saved auth) but registered
    assert adapter._account_status == "unconfigured"


def test_gateway_loads_wechat_and_feishu_inbound_channels_together(tmp_path) -> None:
    cfg = GatewayConfig(state_dir=str(tmp_path))
    cfg.channels.extend(
        [
            ChannelConfig(
                type=ChannelType.WECHAT,
                webhook_url="https://ilinkai.weixin.qq.com/dummy",
                name="wechat",
                enabled=True,
                extra={"account_id": "default"},
            ),
            ChannelConfig(
                type=ChannelType.FEISHU,
                webhook_url="",
                name="feishu",
                enabled=True,
                extra={
                    "connection_mode": "websocket",
                    "app_id": "cli_test",
                    "app_secret": "test-secret",
                },
            ),
        ]
    )

    gateway = MessageGateway(cfg)

    assert set(gateway.registry.names()) == {"wechat", "feishu"}
    assert {adapter.channel_id for adapter in gateway._inbound_adapters} == {
        "wechat",
        "feishu",
    }


def test_gateway_normalizes_legacy_wechat_name_and_reuses_legacy_auth(tmp_path) -> None:
    from clawcodex_ext.services.channels.wechat_ilink import WeChatAuthRecord, WeChatIlinkAuthStore

    wechat_dir = tmp_path / "wechat"
    old_auth = wechat_dir / "wechat-main_auth.json"
    WeChatIlinkAuthStore(old_auth).save(
        WeChatAuthRecord(
            bot_token="bot_tok_123",
            account_id="acct",
            base_url="https://ilinkai.weixin.qq.com",
            user_id="bot_user",
        )
    )
    cfg = GatewayConfig(state_dir=str(tmp_path))
    cfg.channels.append(
        ChannelConfig(
            type=ChannelType.WECHAT,
            webhook_url="https://ilinkai.weixin.qq.com/dummy",
            name="wechat-main",
            enabled=True,
            extra={"base_url": "https://ilinkai.weixin.qq.com", "account_id": "default"},
        )
    )

    gw = MessageGateway(cfg)

    assert gw.registry.get("wechat-main") is None
    adapter = gw.registry.get("wechat")
    assert adapter is not None
    assert adapter._account_status == "logged_in"
    assert adapter._account_id == "acct"


def test_gateway_skips_disabled_channels(tmp_path) -> None:
    cfg = GatewayConfig(state_dir=str(tmp_path))
    cfg.channels.append(
        ChannelConfig(
            type=ChannelType.WECHAT,
            webhook_url="https://ilinkai.weixin.qq.com/dummy",
            name="wechat-off",
            enabled=False,
        )
    )
    gw = MessageGateway(cfg)
    assert gw.registry.get("wechat-off") is None
