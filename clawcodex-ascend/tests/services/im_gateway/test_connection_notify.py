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

"""Tests for connection-state notifications.

When REPL or orchestrator connects/disconnects to the gateway via IPC,
the binding auditor fires and the gateway sends a best-effort notification
to the WeChat user (``"clawcodex-REPL connected"`` / ``"clawcodex-orchestrator disconnected"`` etc.).
These tests verify the notification logic in isolation by driving the
binding policy directly and checking the outbound dispatcher's sends.
"""

from __future__ import annotations

# pylint: disable=no-name-in-module  # Split migration branches provide gateway modules.

import asyncio
import logging
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
    ErrorCategory,
    ChannelHealth,
    ChannelSendResult,
    ValidationResult,
)
from clawcodex_ext.services.im_gateway.config import GatewayConfig
from clawcodex_ext.services.im_gateway.gateway import MessageGateway, _collect_broadcast_targets
from clawcodex_ext.services.im_gateway.models import SessionTarget
from clawcodex_ext.services.im_gateway.store import ReliabilityStore


class _FakeWeChatAdapter(ChannelAdapter):
    """Minimal WeChat adapter for notification tests."""

    def __init__(self, name: str = "wechat", account_id: str = "default") -> None:
        self._name = name
        self._account_id = account_id
        self._config = ChannelConfig(
            type=ChannelType.WECHAT,
            webhook_url="https://ilinkai.weixin.qq.com/dummy",
            name=name,
            enabled=True,
        )
        self._caps = ChannelCapabilitySet.of(
            ChannelCapability.OUTBOUND_TEXT,
            ChannelCapability.INBOUND_POLLING,
            ChannelCapability.CONTEXT_REPLY,
            descriptors={
                ChannelCapability.OUTBOUND_TEXT: CapabilityDescriptor(
                    ChannelCapability.OUTBOUND_TEXT, supports_markdown=False
                )
            },
        )
        self.sends: list[tuple[ChannelMessage, str | None]] = []
        self._last_sender: str | None = None

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
        return ChannelSendResult.success(self._name, provider_receipt="r")

    def last_known_sender(self) -> str | None:
        return self._last_sender

    def set_inbound_handler(self, handler) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _gateway_with_wechat(
    tmp_path, *, sender: str | None = "operator@im.wechat"
) -> tuple[MessageGateway, _FakeWeChatAdapter]:
    """Build a real gateway with a fake WeChat adapter that has a known sender.

    Returns (gateway, adapter) so the caller can inspect ``adapter.sends``.
    """
    adapter = _FakeWeChatAdapter("wechat")
    adapter._last_sender = sender
    reg = ChannelAdapterRegistry()
    reg.register(adapter)
    cfg = GatewayConfig(state_dir=str(tmp_path))
    gw = MessageGateway(cfg, registry=reg)
    return gw, adapter


def _drive_notifications(gw: MessageGateway) -> None:
    """Allow any scheduled create_task notifications to complete."""
    # The _audit_binding schedules notifications via loop.create_task.
    # We need to yield to the event loop so those tasks run.
    # In async tests, awaiting asyncio.sleep(0) suffices.


@pytest.mark.asyncio
async def test_notify_on_connect_repl(tmp_path) -> None:
    """Verify notify on connect repl."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)  # let create_task notification run

    texts = [msg.text for msg, _ in adapter.sends]
    assert "clawcodex-REPL connected" in texts


@pytest.mark.asyncio
async def test_notify_on_connect_orchestrator(tmp_path) -> None:
    """Verify notify on connect orchestrator."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="orchestrator-1", host_type="orchestrator"),
    )
    await asyncio.sleep(0.05)

    texts = [msg.text for msg, _ in adapter.sends]
    assert "clawcodex-orchestrator connected; Orchestrator currently supports commands only" in texts


@pytest.mark.asyncio
async def test_notify_on_disconnect_offline(tmp_path) -> None:
    """Verify notify on disconnect offline."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)
    adapter.sends.clear()

    gw.binding.mark_offline("wechat:direct:*:*", session_id="repl-1")
    await asyncio.sleep(0.05)

    texts = [msg.text for msg, _ in adapter.sends]
    assert "clawcodex-REPL disconnected" in texts


@pytest.mark.asyncio
async def test_repeated_offline_transition_does_not_duplicate_disconnect(tmp_path) -> None:
    """A stale socket cleanup must not announce the same disconnect twice."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="orchestrator-1", host_type="orchestrator"),
    )
    await asyncio.sleep(0.05)
    adapter.sends.clear()

    gw.binding.mark_offline("wechat:direct:*:*", session_id="orchestrator-1")
    gw.binding.mark_offline("wechat:direct:*:*", session_id="orchestrator-1")
    await asyncio.sleep(0.05)

    texts = [msg.text for msg, _ in adapter.sends]
    assert texts.count("clawcodex-orchestrator disconnected") == 1


@pytest.mark.asyncio
async def test_notify_on_disconnect_terminated(tmp_path) -> None:
    """Verify notify on disconnect terminated."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="orch-1", host_type="orchestrator"),
    )
    await asyncio.sleep(0.05)
    adapter.sends.clear()

    gw.binding.terminate("wechat:direct:*:*", session_id="orch-1")
    await asyncio.sleep(0.05)

    texts = [msg.text for msg, _ in adapter.sends]
    assert "clawcodex-orchestrator disconnected" in texts


@pytest.mark.asyncio
async def test_notify_on_override_active_previous(tmp_path) -> None:
    """Verify notify on override active previous."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="orch-1", host_type="orchestrator"),
    )
    await asyncio.sleep(0.05)
    adapter.sends.clear()

    # REPL replaces orchestrator (both in wechat:direct exclusive group)
    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)

    texts = [msg.text for msg, _ in adapter.sends]
    assert "clawcodex-orchestrator disconnected" in texts
    assert "clawcodex-REPL connected" in texts
    # Disconnect notification should come before connect notification
    assert texts.index("clawcodex-orchestrator disconnected") < texts.index("clawcodex-REPL connected")


@pytest.mark.asyncio
async def test_notify_on_override_offline_previous(tmp_path) -> None:
    """Verify notify on override offline previous."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="orch-1", host_type="orchestrator"),
    )
    await asyncio.sleep(0.05)
    # Mark offline first (e.g. socket closed before REPL registers)
    gw.binding.mark_offline("wechat:direct:*:*", session_id="orch-1")
    await asyncio.sleep(0.05)
    adapter.sends.clear()

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)

    texts = [msg.text for msg, _ in adapter.sends]
    assert "clawcodex-REPL connected" in texts
    assert "clawcodex-orchestrator disconnected" not in texts  # already offline, no duplicate


@pytest.mark.asyncio
async def test_notify_skipped_when_origin_unresolvable(tmp_path) -> None:
    """If origin can't be resolved (no known sender), no notification is sent."""
    gw, adapter = _gateway_with_wechat(tmp_path, sender=None)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)

    assert not adapter.sends  # no notification — can't address the user


@pytest.mark.asyncio
async def test_notify_best_effort_does_not_raise(tmp_path) -> None:
    """If outbound.send raises, the notification is swallowed (best-effort)."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    # Make send raise
    original_send = adapter.send

    async def _raising_send(message, *, target=None, context_token=None):
        raise RuntimeError("simulated send failure")

    adapter.send = _raising_send

    # Should not raise
    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)

    # Restore and verify gateway is still functional
    adapter.send = original_send


@pytest.mark.asyncio
async def test_notify_failed_send_result_is_not_logged_as_sent(tmp_path, caplog) -> None:
    """A non-ok send result is a failed notification, not a delivered one."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    async def _failed_send(message, *, target=None, context_token=None):
        adapter.sends.append((message, target))
        return ChannelSendResult.nonretryable_error(
            "wechat",
            message="session expired",
            category=ErrorCategory.AUTH,
        )

    adapter.send = _failed_send
    caplog.set_level(logging.INFO, logger="clawcodex_ext.services.im_gateway.gateway")

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="orch-1", host_type="orchestrator"),
    )
    await asyncio.sleep(0.05)

    assert "connection notify: send failed" in caplog.text
    assert "connection notify: sent 'clawcodex-orchestrator connected'" not in caplog.text


@pytest.mark.asyncio
async def test_notify_enqueued_send_result_is_not_logged_as_sent(tmp_path, caplog) -> None:
    """A deferred notification is accepted for later delivery, not delivered."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    async def _enqueued_send(message, *, target=None, context_token=None):
        adapter.sends.append((message, target))
        return ChannelSendResult.enqueued(
            "wechat",
            message="deferred due to rate limit",
            raw={"retry_after_seconds": 10},
        )

    adapter.send = _enqueued_send
    caplog.set_level(logging.INFO, logger="clawcodex_ext.services.im_gateway.gateway")

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="orch-1", host_type="orchestrator"),
    )
    await asyncio.sleep(0.05)

    assert "connection notify: enqueued" in caplog.text
    assert "connection notify: sent 'clawcodex-orchestrator宸茶繛鎺?" not in caplog.text


@pytest.mark.asyncio
async def test_notify_concrete_origin(tmp_path) -> None:
    """Concrete origin wechat:direct:{account}:{user} resolves and sends."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    gw.binding.bind(
        "wechat:direct:default:operator@im.wechat",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)

    texts = [msg.text for msg, _ in adapter.sends]
    assert "clawcodex-REPL connected" in texts
    # Verify it was sent to the correct target
    targets = [t for _, t in adapter.sends if t is not None]
    assert "operator@im.wechat" in targets


@pytest.mark.asyncio
async def test_notify_terminate_matching_sends_for_each(tmp_path) -> None:
    """Verify notify terminate matching sends for each."""
    gw, adapter = _gateway_with_wechat(tmp_path)

    gw.binding.bind(
        "wechat:direct:default:user_a",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)
    adapter.sends.clear()

    # terminate_matching removes all bindings in the wechat:direct group
    removed = gw.binding.terminate_matching("wechat:direct:*:*")
    assert len(removed) >= 1
    await asyncio.sleep(0.05)

    texts = [msg.text for msg, _ in adapter.sends]
    assert "clawcodex-REPL disconnected" in texts


class _FakeFeishuAdapter(ChannelAdapter):
    """Minimal Feishu adapter for notification broadcast tests."""

    def __init__(self, name: str = "feishu", last_sender: str | None = "ou_feishu_user") -> None:
        self._name = name
        self._config = ChannelConfig(
            type=ChannelType.FEISHU,
            webhook_url="",
            name=name,
            enabled=True,
            extra={"connection_mode": "websocket"},
        )
        self._caps = ChannelCapabilitySet.of(
            ChannelCapability.OUTBOUND_TEXT,
            ChannelCapability.INBOUND_POLLING,
            ChannelCapability.CONTEXT_REPLY,
            descriptors={
                ChannelCapability.OUTBOUND_TEXT: CapabilityDescriptor(
                    ChannelCapability.OUTBOUND_TEXT, supports_markdown=True
                )
            },
        )
        self.sends: list[tuple[ChannelMessage, str | None]] = []
        self._last_sender: str | None = last_sender

    @property
    def channel_id(self) -> str:
        return self._name

    @property
    def config(self) -> ChannelConfig:
        return self._config

    @property
    def capabilities(self) -> ChannelCapabilitySet:
        return self._caps

    def validate_config(self) -> ValidationResult:
        return ValidationResult.ok_result()

    async def health_check(self) -> ChannelHealth:
        return ChannelHealth(healthy=True, channel_id=self._name)

    async def send(self, message, *, target=None, context_token=None) -> ChannelSendResult:
        self.sends.append((message, target))
        return ChannelSendResult.success(self._name, provider_receipt="r")

    def last_known_sender(self) -> str | None:
        return self._last_sender

    def set_inbound_handler(self, handler) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _gateway_with_feishu(
    tmp_path, *, sender: str | None = "ou_feishu_user"
) -> tuple[MessageGateway, _FakeFeishuAdapter]:
    """Build a real gateway with only a fake Feishu adapter (no WeChat)."""
    adapter = _FakeFeishuAdapter("feishu")
    adapter._last_sender = sender
    reg = ChannelAdapterRegistry()
    reg.register(adapter)
    cfg = GatewayConfig(state_dir=str(tmp_path))
    gw = MessageGateway(cfg, registry=reg)
    return gw, adapter


@pytest.mark.asyncio
async def test_notify_broadcasts_to_feishu_when_no_wechat(tmp_path) -> None:
    """Bug 1: REPL connect notification must reach Feishu when no WeChat adapter.

    Previously the wildcard origin ``wechat:direct:*:*`` hard-coded WeChat-only
    resolution: without a WeChat adapter, ``resolve_origin`` returned
    ``(None, None)`` and the notification was silently skipped — even when a
    Feishu channel was connected and ready. The notification should broadcast
    to all connected outbound channels, not just WeChat.
    """
    gw, adapter = _gateway_with_feishu(tmp_path)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)

    texts = [msg.text for msg, _ in adapter.sends]
    assert "clawcodex-REPL connected" in texts


def test_collect_broadcast_targets_uses_persisted_feishu_sender_after_restart(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    store.set_feishu_last_sender("feishu", "oc_chat")
    cfg = GatewayConfig(state_dir=str(tmp_path))
    cfg.replace_channel(
        ChannelConfig(
            type=ChannelType.FEISHU,
            webhook_url="",
            name="feishu",
            enabled=True,
            extra={
                "connection_mode": "websocket",
                "app_id": "cli_app",
                "app_secret": "secret",
            },
        )
    )

    gw = MessageGateway(cfg, store=store)

    assert _collect_broadcast_targets(gw.registry) == [("feishu", "oc_chat")]


def test_collect_broadcast_targets_uses_scanner_before_first_inbound(tmp_path) -> None:
    cfg = GatewayConfig(state_dir=str(tmp_path))
    cfg.replace_channel(
        ChannelConfig(
            type=ChannelType.FEISHU,
            webhook_url="",
            name="feishu",
            enabled=True,
            extra={
                "connection_mode": "websocket",
                "app_id": "cli_app",
                "app_secret": "secret",
                "allowed_user_open_id": "ou_scanner",
            },
        )
    )

    gw = MessageGateway(cfg)

    assert _collect_broadcast_targets(gw.registry) == [("feishu", "ou_scanner")]


@pytest.mark.asyncio
async def test_notify_broadcasts_to_all_connected_channels(tmp_path) -> None:
    """Bug 1: REPL connect notification broadcasts to WeChat AND Feishu."""
    wechat = _FakeWeChatAdapter("wechat")
    wechat._last_sender = "user_wx"
    feishu = _FakeFeishuAdapter("feishu")
    feishu._last_sender = "ou_feishu_user"
    reg = ChannelAdapterRegistry()
    reg.register(wechat)
    reg.register(feishu)
    cfg = GatewayConfig(state_dir=str(tmp_path))
    gw = MessageGateway(cfg, registry=reg)

    gw.binding.bind(
        "wechat:direct:*:*",
        SessionTarget(session_id="repl-1", host_type="repl"),
    )
    await asyncio.sleep(0.05)

    wechat_texts = [msg.text for msg, _ in wechat.sends]
    feishu_texts = [msg.text for msg, _ in feishu.sends]
    assert "clawcodex-REPL connected" in wechat_texts
    assert "clawcodex-REPL connected" in feishu_texts
