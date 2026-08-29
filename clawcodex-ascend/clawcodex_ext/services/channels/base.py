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

"""Abstract base classes and the manager for Channels."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from .exceptions import ChannelDisabledError, ChannelNotFoundError
from .models import ChannelConfig, ChannelMessage
from .transport import (
    DEFAULT_TIMEOUT_SECONDS,
    ChannelTransport,
    UrllibChannelTransport,
    default_headers,
    validate_webhook_url,
)


SendCallable = Callable[[ChannelMessage], Awaitable[bool]]
logger = logging.getLogger(__name__)


def _exception_fingerprint(exc: Exception) -> str:
    """Return a stable diagnostic identifier without exposing exception text."""
    try:
        detail = str(exc)
    except Exception:  # noqa: BLE001
        detail = "<unprintable>"
    material = f"{type(exc).__name__}:{detail}".encode("utf-8", errors="replace")
    return hashlib.sha256(material).hexdigest()[:12]


class BaseChannel(ABC):
    """Abstract channel. Subclasses implement :meth:`format_message` and
    :meth:`send` for a specific platform.
    """

    def __init__(
        self,
        config: ChannelConfig,
        transport: ChannelTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or UrllibChannelTransport()
        self._owns_transport = transport is None
        self._validate_url()

    @property
    def config(self) -> ChannelConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def transport(self) -> ChannelTransport:
        return self._transport

    def _validate_url(self) -> None:
        validate_webhook_url(self._config.webhook_url)

    @abstractmethod
    def format_message(self, message: ChannelMessage) -> tuple[bytes, dict[str, str]]:
        """Return the serialized body and request headers for ``message``."""

    @abstractmethod
    async def send(self, message: ChannelMessage) -> bool: ...

    async def close(self) -> None:
        # Hook for transports that hold resources; the default urllib transport
        # is stateless so this is a no-op.
        return None


class ChannelManager:
    """Dispatches messages to a set of registered channels."""

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: ChannelTransport | None = None,
    ) -> None:
        self._channels: dict[str, BaseChannel] = {}
        self._lock = threading.RLock()
        self._default_timeout = default_timeout
        self._default_transport = transport

    def register(self, channel: BaseChannel) -> None:
        with self._lock:
            self._channels[channel.name] = channel

    def unregister(self, name: str) -> None:
        with self._lock:
            self._channels.pop(name, None)

    def get(self, name: str) -> BaseChannel | None:
        with self._lock:
            return self._channels.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._channels.keys())

    async def broadcast(self, message: ChannelMessage) -> dict[str, bool]:
        with self._lock:
            channels = list(self._channels.values())
        results: dict[str, bool] = {}
        coros = [self._safe_send(channel, message, results) for channel in channels]
        if coros:
            await asyncio.gather(*coros)
        return results

    async def send_to(self, name: str, message: ChannelMessage) -> bool:
        with self._lock:
            channel = self._channels.get(name)
        if channel is None:
            raise ChannelNotFoundError(f"no channel registered as {name!r}")
        if not channel.enabled:
            raise ChannelDisabledError(f"channel {name!r} is disabled")
        return await channel.send(message)

    @property
    def default_timeout(self) -> float:
        return self._default_timeout

    @staticmethod
    async def _safe_send(
        channel: BaseChannel,
        message: ChannelMessage,
        results: dict[str, bool],
    ) -> None:
        try:
            results[channel.name] = await channel.send(message)
        except Exception as exc:  # noqa: BLE001
            # Broadcast must not crash on a single channel failure; record
            # ``False`` and let the caller inspect the results map.
            results[channel.name] = False
            logger.warning(
                "channel broadcast failed: channel=%s error_type=%s error_fingerprint=%s",
                channel.name,
                type(exc).__name__,
                _exception_fingerprint(exc),
            )


def build_default_timeout(timeout: float | None = None) -> float:
    return DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout


__all__ = [
    "BaseChannel",
    "ChannelManager",
    "build_default_timeout",
    "default_headers",
]
