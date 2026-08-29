#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Unified channel adapter registry.

Every channel — WeChat and the legacy Feishu/Slack/Discord webhook
channels — is built through :class:`ChannelAdapterRegistry` and
registered with its :class:`ChannelCapabilitySet`. The gateway looks
up adapters by name and gates every call through
``require_capability``.

The legacy webhook channels (``BaseChannel`` subclasses) are wrapped by
:class:`WebhookChannelAdapter`, which adapts their ``send -> bool``
interface to the uniform ``ChannelSendResult`` contract. This keeps the
existing channels and their tests untouched while bringing them
under the common contract.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from .base import BaseChannel
from .capabilities import (
    CapabilityDescriptor,
    CapabilityNotDeclaredError,
    ChannelAdapter,
    ChannelCapability,
    ChannelCapabilitySet,
)
from .discord import DiscordChannel
from .feishu import FeishuChannel
from .feishu_app import FeishuAppChannelAdapter
from .models import ChannelConfig, ChannelMessage, ChannelType
from .results import (
    ChannelHealth,
    ChannelSendResult,
    ErrorCategory,
    ValidationResult,
)
from .retry import DEFAULT_RETRY_POLICY, RetryPolicy, classify_exception
from .slack import SlackChannel
from .transport import TransportError, validate_webhook_url

ChannelAdapterFactory = Callable[[ChannelConfig], ChannelAdapter]

_WEBHOOK_CAPABILITIES = ChannelCapabilitySet.of(
    ChannelCapability.OUTBOUND_TEXT,
    descriptors={
        ChannelCapability.OUTBOUND_TEXT: CapabilityDescriptor(
            ChannelCapability.OUTBOUND_TEXT,
            supports_markdown=True,
        )
    },
)


class WebhookChannelAdapter(ChannelAdapter):
    """Adapts a legacy ``BaseChannel`` (webhook push) to the channel contract.

    Declares only ``outbound_text``. Error classification is coarse
    because ``BaseChannel.send`` returns a bare ``bool``; precise
    classification (per HTTP status) happens in adapters that own their
    transport, e.g. the WeChat iLink adapter.
    """

    def __init__(
        self,
        config: ChannelConfig,
        base_channel: BaseChannel,
        *,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    ) -> None:
        self._config = config
        self._base = base_channel
        self._retry_policy = retry_policy

    @property
    def channel_id(self) -> str:
        return self._config.name

    @property
    def capabilities(self) -> ChannelCapabilitySet:
        return _WEBHOOK_CAPABILITIES

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry_policy

    @property
    def config(self) -> ChannelConfig:
        return self._config

    @property
    def base_channel(self) -> BaseChannel:
        return self._base

    def validate_config(self) -> ValidationResult:
        errors: list[str] = []
        try:
            validate_webhook_url(self._config.webhook_url)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"webhook_url: {exc}")
        if not self._config.name:
            errors.append("name must be non-empty")
        if errors:
            return ValidationResult.fail(errors)
        return ValidationResult.ok_result()

    async def health_check(self) -> ChannelHealth:
        return ChannelHealth(
            healthy=bool(self._config.enabled),
            channel_id=self.channel_id,
            circuit_state="closed",
            account_status="webhook",
        )

    async def send(
        self,
        message: ChannelMessage,
        *,
        target: str | None = None,
        context_token: str | None = None,
    ) -> ChannelSendResult:
        # ``target`` / ``context_token`` are meaningless for one-way webhook
        # push; they are accepted to satisfy the OutboundCapability signature.
        try:
            ok = await self._base.send(message)
        except TransportError as exc:
            category = classify_exception(exc)
            if category in self._retry_policy.retryable_categories:
                return ChannelSendResult.retryable_error(self.channel_id, message=str(exc), category=category)
            return ChannelSendResult.nonretryable_error(self.channel_id, message=str(exc), category=category)
        except Exception as exc:  # noqa: BLE001
            return ChannelSendResult.nonretryable_error(
                self.channel_id,
                message=f"send raised: {exc}",
                category=ErrorCategory.UNKNOWN,
            )
        if ok:
            return ChannelSendResult.success(self.channel_id)
        # Bare ``False`` from a webhook channel is an ambiguous business
        # rejection (non-200 or platform error code). Treat as non-retryable
        # to avoid hammering a rejecting endpoint; the operator can resend.
        return ChannelSendResult.nonretryable_error(
            self.channel_id,
            message="webhook returned non-success",
            category=ErrorCategory.UNKNOWN,
        )


class ChannelAdapterRegistry:
    """Registry of channel factories and live adapter instances."""

    def __init__(self) -> None:
        self._types: dict[str, ChannelAdapterFactory] = {}
        self._instances: dict[str, ChannelAdapter] = {}
        self._lock = threading.RLock()

    def register_type(self, channel_type: ChannelType | str, factory: ChannelAdapterFactory) -> None:
        key = channel_type.value if isinstance(channel_type, ChannelType) else str(channel_type)
        with self._lock:
            self._types[key] = factory

    def list_types(self) -> list[str]:
        with self._lock:
            return sorted(self._types.keys())

    def create(self, config: ChannelConfig) -> ChannelAdapter:
        """Build an adapter from ``config`` and register it by name."""
        key = config.type.value
        with self._lock:
            factory = self._types.get(key)
        if factory is None:
            raise KeyError(f"no factory registered for channel type {key!r}")
        adapter = factory(config)
        self.register(adapter)
        return adapter

    def register(self, adapter: ChannelAdapter) -> None:
        with self._lock:
            self._instances[adapter.channel_id] = adapter

    def get(self, name: str) -> ChannelAdapter | None:
        with self._lock:
            return self._instances.get(name)

    def require(self, name: str) -> ChannelAdapter:
        adapter = self.get(name)
        if adapter is None:
            raise KeyError(f"no channel registered as {name!r}")
        return adapter

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._instances.keys())

    def all_adapters(self) -> list[ChannelAdapter]:
        with self._lock:
            return list(self._instances.values())

    def inbound_adapters(self) -> list[ChannelAdapter]:
        """Adapters that declare ``inbound_polling`` (started by the gateway)."""
        return [a for a in self.all_adapters() if a.capabilities.has(ChannelCapability.INBOUND_POLLING)]

    def remove(self, name: str) -> bool:
        with self._lock:
            return self._instances.pop(name, None) is not None

    def require_capability(self, adapter: ChannelAdapter | str, capability: ChannelCapability) -> ChannelAdapter:
        if isinstance(adapter, str):
            adapter = self.require(adapter)
        if not adapter.capabilities.has(capability):
            raise CapabilityNotDeclaredError(f"channel {adapter.channel_id!r} does not declare {capability.value!r}")
        return adapter


def _webhook_factory(channel_cls: type[BaseChannel]) -> ChannelAdapterFactory:
    def factory(config: ChannelConfig) -> WebhookChannelAdapter:
        base = channel_cls(config)
        return WebhookChannelAdapter(config, base)

    return factory


def _feishu_factory(config: ChannelConfig) -> ChannelAdapter:
    mode = str((config.extra or {}).get("connection_mode") or "").strip().lower()
    if not mode and config.webhook_url:
        mode = "webhook"
    if mode == "webhook":
        return WebhookChannelAdapter(config, FeishuChannel(config))
    if mode == "websocket":
        return FeishuAppChannelAdapter(config)
    raise ValueError("feishu connection_mode must be websocket or webhook")


def build_default_registry() -> ChannelAdapterRegistry:
    """Registry with the legacy webhook channels pre-registered."""
    registry = ChannelAdapterRegistry()
    registry.register_type(ChannelType.FEISHU, _feishu_factory)
    registry.register_type(ChannelType.SLACK, _webhook_factory(SlackChannel))
    registry.register_type(ChannelType.DISCORD, _webhook_factory(DiscordChannel))
    return registry


__all__ = [
    "ChannelAdapterRegistry",
    "ChannelAdapterFactory",
    "WebhookChannelAdapter",
    "build_default_registry",
]
