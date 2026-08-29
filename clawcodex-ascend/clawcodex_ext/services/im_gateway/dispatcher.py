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

"""Inbound message routing and delivery.

The dispatcher owns the gateway-side pipeline: reserve the idempotency key,
route, classify, apply the opt-in command gate, start processing status, and
deliver through IPC or the local fallback handler. Plain text for an opt-in
runtime is classified by that runtime when the gateway cannot prove it is
busy; this preserves the peer's real-time follow-up/steering decision.
"""

from __future__ import annotations

# Sibling clawcodex_ext packages (messaging, channels) are migrated in separate
# branches; suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

import asyncio
import logging
import math
import threading
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable

from clawcodex_ext.messaging.semantics import MessageClassifier
from clawcodex_ext.services.channels.capabilities import ProcessingOutcome

from .config import CommandAllowlistConfig, DEFAULT_PUSH_TIMEOUT_SECONDS
from .models import (
    AckLayer,
    AckReceipt,
    InboundMessage,
    MessageSemantics,
    SessionTarget,
)
from .repl_command_gate import check_orchestrator_command, check_repl_command
from .router import SessionRouter
from .store import ReliabilityStore

if TYPE_CHECKING:
    from .processing_status import ProcessingStatusManager

logger = logging.getLogger(__name__)

InboundHandler = Callable[[InboundMessage], Awaitable[AckReceipt | None]]
PushHandler = Callable[[InboundMessage], Awaitable[bool]]

LOG_PREVIEW_LIMIT = 32
_AUDIT_COMMAND_LIMIT = 64
_OPT_IN_HOST_TYPES = frozenset({"repl", "orchestrator", "opt_in"})
_EXPLICIT_SEMANTIC_VALUES = frozenset(
    semantic.value for semantic in MessageSemantics if semantic is not MessageSemantics.UNSUPPORTED_MEDIA
)


class InboundDispatcher:
    def __init__(
        self,
        store: ReliabilityStore,
        router: SessionRouter,
        *,
        classifier: MessageClassifier | None = None,
        command_allowlists: CommandAllowlistConfig | None = None,
        processing_status: "ProcessingStatusManager | None" = None,
        push_timeout_seconds: float = DEFAULT_PUSH_TIMEOUT_SECONDS,
    ) -> None:
        if not math.isfinite(push_timeout_seconds) or push_timeout_seconds <= 0:
            raise ValueError("push_timeout_seconds must be a finite value greater than zero")
        self._store = store
        self._router = router
        self._classifier = classifier or MessageClassifier()
        effective_allowlists = command_allowlists or CommandAllowlistConfig()
        self._repl_allowed_commands = frozenset(effective_allowlists.repl)
        self._orchestrator_allowed_commands = frozenset(effective_allowlists.orchestrator)
        self._processing_status = processing_status
        self._push_timeout_seconds = float(push_timeout_seconds)
        self._handler: InboundHandler | None = None
        self._push_handler: PushHandler | None = None
        self._dedupe_lock = threading.Lock()
        self._in_flight: set[str] = set()

    def set_handler(self, handler: InboundHandler) -> None:
        self._handler = handler

    def set_push_handler(self, handler: PushHandler) -> None:
        """Register the IPC push callback used for opt-in origins."""
        self._push_handler = handler

    def classify(
        self,
        message: InboundMessage,
        *,
        is_busy: bool = False,
        has_pending_wait: bool = False,
    ) -> MessageSemantics:
        return self._classifier.classify(
            message,
            is_busy=is_busy,
            has_pending_wait=has_pending_wait,
        )

    async def process(self, message: InboundMessage) -> AckReceipt:
        delivery_id = str(uuid.uuid4())
        dedupe_key = self._dedupe_key(message)
        if not self._reserve_delivery(dedupe_key):
            logger.debug(
                "im_gateway: duplicate inbound skipped origin=%s",
                self._preview(message.origin),
            )
            return AckReceipt(delivery_id, AckLayer.ACCEPTED, message="duplicate; skipped")

        completed = False
        try:
            target, early_receipt = self._route_message(message, delivery_id)
            if early_receipt is not None:
                return early_receipt

            self._classify_message(message, target)
            self._log_route(message, target)

            gate_receipt = self._apply_command_gate(message, target, delivery_id)
            if gate_receipt is not None:
                self._record_completed(dedupe_key, message)
                completed = True
                return gate_receipt

            await self._start_processing(message)

            if await self._push_to_opt_in(message, target):
                self._record_completed(dedupe_key, message)
                completed = True
                return AckReceipt(
                    delivery_id,
                    AckLayer.ENQUEUED,
                    message="pushed to opt-in peer",
                )

            handler_receipt = await self._dispatch_to_handler(message)
            if handler_receipt is not None:
                self._record_completed(dedupe_key, message)
                completed = True
                return handler_receipt

            await self._complete_processing(message, ProcessingOutcome.FAILURE)
            return AckReceipt(delivery_id, AckLayer.ACCEPTED, message="accepted")
        finally:
            if not completed:
                self._release_delivery(dedupe_key)

    @staticmethod
    def _dedupe_key(message: InboundMessage) -> str:
        return message.message_id or f"{message.origin}:{message.text}"

    def _reserve_delivery(self, key: str) -> bool:
        """Reserve ``key`` without persisting a completed-delivery record."""
        with self._dedupe_lock:
            if key in self._in_flight or self._store.is_duplicate(key):
                return False
            self._in_flight.add(key)
            return True

    def _record_completed(self, key: str, message: InboundMessage) -> None:
        """Persist a successful delivery and release its in-flight slot."""
        with self._dedupe_lock:
            self._store.record_processed(key, message_id=message.message_id)
            self._in_flight.discard(key)

    def _release_delivery(self, key: str) -> None:
        with self._dedupe_lock:
            self._in_flight.discard(key)

    def _route_message(
        self,
        message: InboundMessage,
        delivery_id: str,
    ) -> tuple[SessionTarget, AckReceipt | None]:
        if self._router.is_offline(message.origin):
            self._store.audit(
                "target_offline",
                delivery_id=delivery_id,
                origin=message.origin,
                message_id=message.message_id,
            )
            logger.info(
                "im_gateway: target offline origin=%s; accepted but not delivered",
                self._preview(message.origin),
            )
            return (
                self._router.route(message.origin),
                AckReceipt(
                    delivery_id,
                    AckLayer.ACCEPTED,
                    message="target_offline; rebind or use default session",
                ),
            )
        return self._router.route(message.origin), None

    def _classify_message(self, message: InboundMessage, target: SessionTarget) -> None:
        if message.semantic is not None:
            return
        classified = self.classify(message, is_busy=self._origin_is_busy(message.origin))
        if (
            target.host_type in _OPT_IN_HOST_TYPES
            and classified is MessageSemantics.NEW_PROMPT
            and not self._has_explicit_semantic(message)
        ):
            # The gateway has no authoritative view of a peer's active query.
            # Leave ordinary opt-in text unclassified so the receiving runtime
            # can choose newPrompt vs followUp from its current busy state.
            return
        message.semantic = classified

    def _origin_is_busy(self, origin: str) -> bool:
        if self._processing_status is None:
            return False
        try:
            return bool(self._processing_status.is_busy(origin))
        except Exception:  # noqa: BLE001
            logger.debug(
                "im_gateway: busy probe failed origin=%s",
                self._preview(origin),
                exc_info=True,
            )
            return False

    @staticmethod
    def _has_explicit_semantic(message: InboundMessage) -> bool:
        raw_value = message.raw.get("deliverAs") if isinstance(message.raw, dict) else None
        if raw_value in _EXPLICIT_SEMANTIC_VALUES:
            return True
        return any(tag in _EXPLICIT_SEMANTIC_VALUES for tag in message.semantic_tags)

    def _log_route(self, message: InboundMessage, target: SessionTarget) -> None:
        logger.info(
            "im_gateway: route origin=%s semantic=%s target=%s host_type=%s",
            self._preview(message.origin),
            message.semantic.value if message.semantic else None,
            self._preview(target.session_id),
            target.host_type,
        )

    def _apply_command_gate(
        self,
        message: InboundMessage,
        target: SessionTarget,
        delivery_id: str,
    ) -> AckReceipt | None:
        """Apply the shared slash-command gate for opt-in runtime hosts."""
        if target.host_type == "repl":
            checker = check_repl_command
            allowed_commands = self._repl_allowed_commands
        elif target.host_type == "orchestrator":
            checker = check_orchestrator_command
            allowed_commands = self._orchestrator_allowed_commands
        else:
            return None

        allowed, reason = checker(
            message.text or "",
            allowed_commands=allowed_commands,
        )
        if allowed:
            return None

        event_type = f"{target.host_type}_command_blocked"
        self._store.audit(
            event_type,
            delivery_id=delivery_id,
            origin=message.origin,
            command=(message.text or "")[:_AUDIT_COMMAND_LIMIT],
            reason=reason,
        )
        logger.info(
            "im_gateway: %s command blocked origin=%s cmd=%s",
            target.host_type,
            self._preview(message.origin),
            self._preview(message.text),
        )
        return AckReceipt(
            delivery_id,
            AckLayer.ACCEPTED,
            message=reason,
            notify_user=True,
        )

    async def _start_processing(self, message: InboundMessage) -> None:
        if self._processing_status is not None:
            await self._processing_status.start(message)

    async def _push_to_opt_in(
        self,
        message: InboundMessage,
        target: SessionTarget,
    ) -> bool:
        if target.host_type not in _OPT_IN_HOST_TYPES or self._push_handler is None:
            return False
        try:
            delivered = await asyncio.wait_for(
                self._push_handler(message),
                timeout=self._push_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "im_gateway: opt-in push timed out origin=%s timeout=%.3fs",
                self._preview(message.origin),
                self._push_timeout_seconds,
            )
            delivered = False
        except Exception:  # noqa: BLE001
            logger.exception(
                "im_gateway: push_handler error origin=%s",
                self._preview(message.origin),
            )
            delivered = False
        if not delivered:
            logger.warning(
                "im_gateway: opt-in push failed origin=%s; falling back to default",
                self._preview(message.origin),
            )
        return bool(delivered)

    async def _dispatch_to_handler(self, message: InboundMessage) -> AckReceipt | None:
        if self._handler is None:
            return None
        try:
            result = await self._handler(message)
        except Exception:
            await self._complete_processing(message, ProcessingOutcome.FAILURE)
            raise
        if result is None:
            return None
        outcome = ProcessingOutcome.SUCCESS if result.layer is AckLayer.PROCESSED else ProcessingOutcome.FAILURE
        await self._complete_processing(message, outcome)
        return result

    async def _complete_processing(
        self,
        message: InboundMessage,
        outcome: ProcessingOutcome,
    ) -> None:
        if self._processing_status is None:
            return
        await self._processing_status.complete(
            message.message_id,
            outcome,
            origin=message.origin,
        )

    @staticmethod
    def _preview(value: str | None) -> str:
        return (value or "")[:LOG_PREVIEW_LIMIT]


__all__ = [
    "DEFAULT_PUSH_TIMEOUT_SECONDS",
    "InboundDispatcher",
    "InboundHandler",
    "LOG_PREVIEW_LIMIT",
]
