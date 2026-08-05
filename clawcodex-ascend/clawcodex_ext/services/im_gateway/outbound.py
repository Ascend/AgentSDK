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

"""Outbound dispatcher.

Gateway → CapabilityGate → ChannelAdapterRegistry → OutboundCapability.send().
Writes an outbox entry (with idempotency_key) before every send, records
the provider receipt on success, and moves non-retryable failures to the
dead-letter store. Markdown→plain-text fallback and long-message
LiveView fallback are applied per the channel's capability descriptor.

The full retry loop, storm aggregation, and compaction land in P4; v1
does a single attempt (plus the platform-rejection plain-text retry)
and records the result for P4 to replay.
"""

from __future__ import annotations

# pylint: disable=no-name-in-module  # Split migration branches provide these modules.

import asyncio
import logging
import time
import uuid
from typing import cast

from clawcodex_ext.services.channels.capabilities import ChannelCapability, OutboundCapability
from clawcodex_ext.services.channels.models import ChannelMessage
from clawcodex_ext.services.channels.results import ChannelSendResult, ErrorCategory, SendStatus

from .capability_gate import CapabilityGate
from .config import GatewayConfig
from .models import OutboundMessage
from .store import ReliabilityStore
from .text import maybe_truncate_with_liveview, strip_markdown

logger = logging.getLogger(__name__)


class OutboundDispatcher:
    def __init__(
        self,
        registry,
        gate: CapabilityGate,
        store: ReliabilityStore,
        config: GatewayConfig,
        *,
        sleep=asyncio.sleep,
    ) -> None:
        self._registry = registry
        self._gate = gate
        self._store = store
        self._config = config
        self._sleep = sleep
        self._idempotency_lock = asyncio.Lock()

    def _resolve_descriptor(self, channel: str):
        adapter = self._registry.get(channel)
        if adapter is None:
            return None, None
        cap = ChannelCapability.OUTBOUND_TEXT
        if not adapter.capabilities.has(cap):
            return adapter, None
        return adapter, adapter.capabilities.descriptor(cap)

    def _prepare_text(self, channel: str, text: str) -> tuple[list[str], bool]:
        """Return (chunks, stripped) after markdown/long-message handling."""
        _adapter, descriptor = self._resolve_descriptor(channel)
        supports_markdown = bool(descriptor and descriptor.supports_markdown)
        max_chunks = self._config.reliability.long_message_threshold_chunks
        effective = text
        stripped = False
        if self._config.reliability.markdown_fallback and not supports_markdown:
            effective = strip_markdown(effective)
            stripped = True
        chunks = maybe_truncate_with_liveview(effective, max_chunks=max_chunks)
        return chunks, stripped

    async def send(self, message: OutboundMessage) -> ChannelSendResult:
        idem = message.idempotency_key or f"out:{uuid.uuid4()}"
        async with self._idempotency_lock:
            completed = self._completed_result(idem, message.channel)
            if completed is not None:
                return completed
            return await self._send_once(message, idem)

    async def _send_once(self, message: OutboundMessage, idem: str) -> ChannelSendResult:
        channel = message.channel

        adapter = self._registry.get(channel)
        if adapter is None:
            self._store.append_outbox(
                {
                    "idempotency_key": idem,
                    "channel": channel,
                    "status": "dead",
                    "error": "channel_not_found",
                    "at": time.time(),
                }
            )
            return ChannelSendResult.nonretryable_error(
                channel,
                message=f"channel {channel!r} not registered",
                category=ErrorCategory.NOT_FOUND,
            )
        logger.debug("outbound send: channel=%s idem=%s len=%d", channel, idem[:16], len(message.text))

        # fail-closed capability gate
        try:
            adapter = self._gate.require_outbound(channel)
        except Exception as exc:  # noqa: BLE001
            self._store.append_outbox(
                {
                    "idempotency_key": idem,
                    "channel": channel,
                    "status": "dead",
                    "error": f"capability_gate: {exc}",
                    "at": time.time(),
                }
            )
            logger.warning("outbound send: capability gate rejected channel=%s: %s", channel, exc)
            return ChannelSendResult.unsupported(channel, message=str(exc))

        sender = cast(OutboundCapability, adapter)
        chunks, stripped = self._prepare_text(channel, message.text)
        if not chunks:
            self._store.append_outbox(
                {
                    "idempotency_key": idem,
                    "channel": channel,
                    "status": "dead",
                    "error": "empty_message",
                    "at": time.time(),
                }
            )
            return ChannelSendResult.nonretryable_error(
                channel,
                message="outbound text must not be empty",
                category=ErrorCategory.CLIENT_ERROR,
            )
        last_result: ChannelSendResult | None = None
        for idx, chunk in enumerate(chunks):
            chunk_idem = idem if len(chunks) == 1 else f"{idem}#{idx}"
            self._store.append_outbox(
                {
                    "idempotency_key": chunk_idem,
                    "channel": channel,
                    "target": message.target,
                    "payload_size": len(chunk),
                    "status": "pending",
                    "at": time.time(),
                }
            )
            last_result = await self._send_chunk_with_retry(
                sender,
                chunk,
                message,
                chunk_idem,
                stripped,
                channel,
            )
            if not last_result.ok:
                logger.warning(
                    "outbound send: chunk %d failed channel=%s status=%s error=%s",
                    idx,
                    channel,
                    last_result.status.value if last_result.status else "?",
                    last_result.message or "",
                )
                return last_result
        logger.info("outbound send: ok channel=%s", channel)
        self._store.append_outbox(
            {
                "idempotency_key": idem,
                "channel": channel,
                "status": "delivered",
                "provider_receipt": last_result.provider_receipt if last_result else None,
                "attempts": last_result.attempts if last_result else 1,
                "at": time.time(),
            }
        )
        return last_result or ChannelSendResult.success(channel)

    def _completed_result(self, idempotency_key: str, channel: str) -> ChannelSendResult | None:
        for entry in reversed(self._store.outbox_entries()):
            if entry.get("idempotency_key") != idempotency_key:
                continue
            if entry.get("status") != "delivered":
                return None
            return ChannelSendResult.success(
                channel,
                provider_receipt=entry.get("provider_receipt"),
                attempts=max(1, int(entry.get("attempts") or 1)),
            )
        return None

    async def _send_chunk_with_retry(
        self,
        sender: OutboundCapability,
        chunk: str,
        message: OutboundMessage,
        chunk_idem: str,
        stripped: bool,
        channel: str,
    ) -> ChannelSendResult:
        """Send one chunk with retry-policy backoff; dead-letter on terminal failure."""
        from clawcodex_ext.services.channels.retry import compute_backoff

        policy = self._adapter_retry_policy(channel)
        max_attempts = policy.max_attempts
        attempt = 1
        payload = ChannelMessage(
            text=chunk,
            title=message.title,
            markdown=message.markdown and not stripped,
            metadata=message.metadata,
        )
        while True:
            result = await sender.send(
                payload,
                target=message.target,
                context_token=message.context_token,
            )
            # Platform rejected the stripped text — retry once as plain text.
            if not result.ok and stripped and result.status is SendStatus.NONRETRYABLE_ERROR and attempt == 1:
                plain = strip_markdown(chunk) if chunk != strip_markdown(chunk) else chunk
                payload = ChannelMessage(
                    text=plain,
                    title=message.title,
                    markdown=False,
                    metadata=message.metadata,
                )
                result = await sender.send(
                    payload,
                    target=message.target,
                    context_token=message.context_token,
                )
            if result.ok:
                self._store.append_outbox(
                    {
                        "idempotency_key": chunk_idem,
                        "channel": channel,
                        "status": "delivered",
                        "provider_receipt": result.provider_receipt,
                        "attempts": attempt,
                        "at": time.time(),
                    }
                )
                return result
            # terminal non-retryable → dead-letter
            if not result.retryable or attempt >= max_attempts:
                self._store.append_outbox(
                    {
                        "idempotency_key": chunk_idem,
                        "channel": channel,
                        "status": "failed",
                        "error_category": result.error_category.value,
                        "message": result.message,
                        "attempts": attempt,
                        "at": time.time(),
                    }
                )
                self._store.append_dead_letter(
                    {
                        "idempotency_key": chunk_idem,
                        "channel": channel,
                        "target": message.target,
                        "error_category": result.error_category.value,
                        "message": result.message,
                        "attempts": attempt,
                        "at": time.time(),
                    }
                )
                if not result.retryable:
                    logger.warning(
                        "outbound dead-letter: channel=%s idem=%s category=%s message=%s",
                        channel,
                        chunk_idem[:16],
                        result.error_category.value,
                        result.message or "",
                    )
                else:
                    logger.warning(
                        "outbound exhausted: channel=%s idem=%s attempts=%d/%d last=%s",
                        channel,
                        chunk_idem[:16],
                        attempt,
                        max_attempts,
                        result.message or "",
                    )
                return result
            # retryable + under budget → backoff and retry
            self._store.append_outbox(
                {
                    "idempotency_key": chunk_idem,
                    "channel": channel,
                    "status": "retry_pending",
                    "error_category": result.error_category.value,
                    "attempt": attempt,
                    "at": time.time(),
                }
            )
            delay = compute_backoff(attempt, policy)
            logger.info(
                "outbound retry: channel=%s idem=%s attempt=%d/%d delay=%.1fs category=%s",
                channel,
                chunk_idem[:16],
                attempt,
                max_attempts,
                delay,
                result.error_category.value,
            )
            await self._sleep(delay)
            attempt += 1

    def _adapter_retry_policy(self, channel: str):
        adapter = self._registry.get(channel)
        if adapter is not None:
            return adapter.retry_policy
        from clawcodex_ext.services.channels.retry import DEFAULT_RETRY_POLICY

        return DEFAULT_RETRY_POLICY

    async def broadcast(
        self, message: OutboundMessage, *, channels: list[str] | None = None
    ) -> dict[str, ChannelSendResult]:
        names = channels or self._registry.names()
        broadcast_key = message.idempotency_key or f"broadcast:{uuid.uuid4()}"
        results: dict[str, ChannelSendResult] = {}
        for name in names:
            try:
                results[name] = await self.send(
                    OutboundMessage(
                        text=message.text,
                        channel=name,
                        target=message.target,
                        context_token=message.context_token,
                        level=message.level,
                        title=message.title,
                        markdown=message.markdown,
                        metadata=message.metadata,
                        semantic_tags=list(message.semantic_tags),
                        idempotency_key=f"{broadcast_key}:{name}",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("outbound broadcast failed: channel=%s: %s", name, exc)
                results[name] = ChannelSendResult.nonretryable_error(
                    name, message=f"broadcast raised: {exc}", category=ErrorCategory.UNKNOWN
                )
        return results


__all__ = ["OutboundDispatcher"]
