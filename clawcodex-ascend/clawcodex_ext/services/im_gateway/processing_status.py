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

"""Coordinate channel-visible processing state across gateway runtimes."""

from __future__ import annotations

# Split migration branches are linted before all package modules and initializers are aggregated.
# pylint: disable=no-name-in-module,relative-beyond-top-level

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from clawcodex_ext.services.channels.capabilities import (
    ChannelCapability,
    ProcessingOutcome,
)

from .models import InboundMessage

logger = logging.getLogger(__name__)

DEFAULT_PENDING_PROCESSING_LIMIT = 1024


@dataclass(frozen=True)
class PendingProcessing:
    message_id: str
    channel: str
    origin: str


class ProcessingStatusManager:
    """Best-effort lifecycle registry keyed by original inbound message ID."""

    def __init__(self, registry: Any, *, max_pending: int = DEFAULT_PENDING_PROCESSING_LIMIT) -> None:
        self._registry = registry
        self._max_pending = max(1, int(max_pending))
        self._pending: OrderedDict[str, PendingProcessing] = OrderedDict()

    def has_pending(self, message_id: str) -> bool:
        return bool(message_id) and message_id in self._pending

    def pending(self, message_id: str) -> PendingProcessing | None:
        return self._pending.get(message_id)

    def is_busy(self, origin: str) -> bool:
        """Return whether ``origin`` has an unfinished inbound message."""
        return bool(origin) and any(entry.origin == origin for entry in self._pending.values())

    async def start(self, message: InboundMessage) -> bool:
        message_id = str(message.message_id or "")
        if not message_id or not message.channel:
            return False
        existing = self._pending.get(message_id)
        if existing is not None:
            self._pending.move_to_end(message_id)
            return True
        adapter = self._processing_adapter(message.channel)
        if adapter is None:
            return False
        self._pending[message_id] = PendingProcessing(
            message_id=message_id,
            channel=message.channel,
            origin=message.origin,
        )
        self._pending.move_to_end(message_id)
        self._trim()
        try:
            return bool(await adapter.on_processing_start(message_id))
        except Exception:  # noqa: BLE001
            self._pending.pop(message_id, None)
            logger.warning(
                "processing status start failed: channel=%s message_id=%s",
                message.channel,
                message_id[:16],
                exc_info=True,
            )
            return False

    async def complete(
        self,
        message_id: str,
        outcome: ProcessingOutcome,
        *,
        origin: str | None = None,
    ) -> bool:
        entry = self._pending.get(str(message_id or ""))
        if entry is None:
            return False
        if origin is not None and origin != entry.origin:
            logger.warning(
                "processing status completion origin mismatch: message_id=%s",
                entry.message_id[:16],
            )
            return False
        adapter = self._processing_adapter(entry.channel)
        if adapter is None:
            self._pending.pop(entry.message_id, None)
            return False
        try:
            completed = bool(await adapter.on_processing_complete(entry.message_id, outcome))
        except Exception:  # noqa: BLE001
            logger.warning(
                "processing status completion failed: channel=%s message_id=%s outcome=%s",
                entry.channel,
                entry.message_id[:16],
                outcome.value,
                exc_info=True,
            )
            return False
        if completed:
            self._pending.pop(entry.message_id, None)
        return completed

    def _processing_adapter(self, channel: str) -> Any | None:
        adapter = self._registry.get(channel)
        if adapter is None:
            return None
        capabilities = getattr(adapter, "capabilities", None)
        if capabilities is None or not capabilities.has(ChannelCapability.PROCESSING_STATUS):
            return None
        start = getattr(adapter, "on_processing_start", None)
        complete = getattr(adapter, "on_processing_complete", None)
        return adapter if callable(start) and callable(complete) else None

    def _trim(self) -> None:
        while len(self._pending) > self._max_pending:
            self._pending.popitem(last=False)


__all__ = [
    "DEFAULT_PENDING_PROCESSING_LIMIT",
    "PendingProcessing",
    "ProcessingStatusManager",
]
