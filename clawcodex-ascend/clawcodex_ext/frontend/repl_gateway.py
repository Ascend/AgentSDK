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

"""ReplGatewayClient — REPL opt-in UDS client wrapping ``_enqueue_prompt`` (P5).

Lives in ``clawcodex_ext/frontend/`` and mounts via the existing
``install_repl_extensions`` hook (no src change). It:

  * connects to the gateway daemon UDS and registers the REPL session
    as the opt-in target for an origin (overriding the default route)
  * heartbeats + reconnects (exp backoff) so the gateway sees it online
  * on inbound ``followUp``/``newPrompt``, checks the REPL prompt queue
    capacity and dedups by ``delivery_id`` before calling
    ``repl._enqueue_prompt(text)`` — never silently drops when the
    ``deque(maxlen=100)`` is full (rejects + ack instead)
  * acks in layers: ``accepted`` on receive, ``enqueued`` after a
    successful ``_enqueue_prompt``

The wrapper takes injectable ``enqueue`` / ``queue_size`` callables so it
is unit-testable without a live REPL.
"""

from __future__ import annotations
# pylint: disable=E0611

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable

from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient
from clawcodex_ext.services.im_gateway.ipc_protocol import GatewayFrame

logger = logging.getLogger(__name__)

EnqueueFn = Callable[[str], None]
QueueSizeFn = Callable[[], int]
WakeFn = Callable[[], None]
ControlHandlerFn = Callable[[str, str | None], bool]
PermissionProbeFn = Callable[[str], bool]

_REPL_CONTROL_COMMANDS = frozenset({"/stop"})


@dataclass(frozen=True)
class PendingImReply:
    delivery_id: str
    origin: str
    context_token: str | None = None


class ReplGatewayClient:
    def __init__(
        self,
        socket_path: str,
        *,
        session_id: str,
        origin: str,
        enqueue: EnqueueFn,
        queue_size: QueueSizeFn,
        queue_capacity: int = 100,
        instance_id: str | None = None,
        wake: WakeFn | None = None,
        control_handler: ControlHandlerFn | None = None,
        permission_probe: PermissionProbeFn | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._session_id = session_id
        self._origin = origin
        self._enqueue = enqueue
        self._queue_size = queue_size
        self._capacity = queue_capacity
        self._instance_id = instance_id or session_id
        self._permission_probe = permission_probe
        # Wake the REPL's blocked prompt loop so an enqueued IM prompt is
        # drained on the next loop iteration. Without this the prompt sits in
        # ``_queued_prompts`` while the main loop is stuck in
        # ``prompt_async('❯ ')`` (the only other wake, ``_watch_outbox``,
        # fires solely on cron outbox events) — the message is never
        # displayed, processed, or replied to.
        self._wake: WakeFn | None = wake
        self._control_handler = control_handler
        # The IPC client's on_deliver fires when the gateway pushes an inbound
        # WeChat message for this origin. Route it to self.deliver, which does
        # dedup / capacity check / enqueue / ack.
        self._client = GatewayIpcClient(
            socket_path,
            instance_id=self._instance_id,
            on_deliver=self._on_pushed_deliver,
        )
        self._seen: set[str] = set()
        self._pending_replies: deque[PendingImReply] = deque()
        self._heartbeat_task: asyncio.Task[None] | None = None

    @property
    def socket_path(self) -> str:
        return self._socket_path

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def origin(self) -> str:
        return self._origin

    @property
    def is_connected(self) -> bool:
        return self._client._writer is not None

    async def _on_pushed_deliver(self, frame) -> None:
        """Server-pushed DELIVER callback: enqueue into the REPL prompt queue."""
        delivery_id = frame.delivery_id or frame.message_id or ""
        text = frame.text or ""
        logger.info(
            "repl_gateway: inbound push delivery_id=%s len=%d → enqueue",
            delivery_id[:16],
            len(text),
        )
        try:
            await self.deliver(
                delivery_id=delivery_id,
                text=text,
                origin=frame.origin,
                semantic=frame.semantic,
                context_token=frame.context_token,
            )
        except QueueFull:
            logger.warning("repl_gateway: queue full, rejected delivery_id=%s", delivery_id[:16])
            await self._complete_processing(
                message_id=delivery_id,
                outcome="failure",
                reason="REPL prompt queue full",
            )
        except Exception:  # noqa: BLE001
            logger.exception("repl_gateway: deliver failed delivery_id=%s", delivery_id[:16])
            await self._complete_processing(
                message_id=delivery_id,
                outcome="failure",
                reason="REPL delivery failed",
            )

    async def connect(self) -> GatewayFrame | None:
        await self._client.connect()
        response = await self._client.register(
            session_id=self._session_id, origin=self._origin, capabilities=["outbound_text"]
        )
        if response is None or response.ack_layer != "accepted":
            await self._client.close()
            raise ConnectionError("gateway registration failed")
        return response

    async def close(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
        with __import__("contextlib").suppress(RuntimeError, ConnectionError, OSError):
            await self._client.unregister(self._session_id)
        await self._client.close()

    async def start_heartbeat(self, interval: float = 30.0) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval))

    async def _heartbeat_loop(self, interval: float) -> None:
        while True:
            try:
                await self._client.heartbeat()
            except Exception:  # noqa: BLE001
                logger.debug("heartbeat failed", exc_info=True)
            await asyncio.sleep(interval)

    def can_enqueue(self) -> bool:
        """Reject before the deque(maxlen) silently drops the oldest message."""
        return self._queue_size() < self._capacity

    async def deliver(
        self,
        *,
        delivery_id: str,
        text: str,
        origin: str | None = None,
        semantic: str | None = None,
        context_token: str | None = None,
    ) -> GatewayFrame | None:
        """Enqueue a follow-up/newPrompt into the REPL, layered ack.

        Returns the ack frame from the gateway (or None if deduped).
        Raises ``QueueFull`` if the REPL prompt queue is at capacity so the
        gateway acks ``accepted`` (rejected) rather than silently dropping.
        """
        if delivery_id in self._seen:
            return None
        # A pending REPL permission wait accepts a WeChat reply (menu
        # number/letter, or /stop→deny) as the decision. This MUST take
        # priority over the control/enqueue paths so a "1"/"y" reply is
        # consumed as the permission choice instead of being queued as a
        # new prompt for the next turn.
        if self._permission_probe is not None and self._permission_probe(text):
            self._seen.add(delivery_id)
            response = await self._client.ack(delivery_id=delivery_id, layer="processed", message="permission reply")
            await self._complete_processing(
                message_id=delivery_id,
                outcome="success",
                reason="permission reply",
            )
            return response
        if self._is_priority_control(text, semantic):
            self._seen.add(delivery_id)
            handled = False
            if self._control_handler is not None:
                handled = bool(self._control_handler(text, origin))
            message = "control dispatched" if handled else "control ignored"
            response = await self._client.ack(delivery_id=delivery_id, layer="processed", message=message)
            await self._complete_processing(
                message_id=delivery_id,
                outcome="success" if handled else "failure",
                reason=message,
            )
            return response
        if not self.can_enqueue():
            raise QueueFull(f"REPL prompt queue at capacity ({self._capacity})")
        self._seen.add(delivery_id)
        self._enqueue(text)
        if origin:
            self._pending_replies.append(
                PendingImReply(
                    delivery_id=delivery_id,
                    origin=origin,
                    context_token=context_token,
                )
            )
        # Wake the REPL prompt loop so it iterates and drains the just-enqueued
        # prompt instead of staying blocked on ``prompt_async('❯ ')``.
        if self._wake is not None:
            try:
                self._wake()
            except Exception:  # noqa: BLE001
                logger.debug("repl_gateway: wake callback failed", exc_info=True)
        # acknowledge enqueued back to the gateway
        return await self._client.ack(delivery_id=delivery_id, layer="enqueued", message="enqueued")

    async def _complete_processing(
        self,
        *,
        message_id: str,
        outcome: str,
        reason: str,
    ) -> None:
        complete = getattr(self._client, "complete_processing", None)
        if callable(complete):
            try:
                await complete(message_id=message_id, outcome=outcome, reason=reason)
            except (ConnectionError, RuntimeError, OSError):
                logger.debug(
                    "repl processing completion skipped while disconnected: %s",
                    message_id[:16],
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "repl processing completion failed: %s",
                    message_id[:16],
                    exc_info=True,
                )

    @staticmethod
    def _is_priority_control(text: str, semantic: str | None) -> bool:
        normalized = (text or "").strip().split(maxsplit=1)[0].lower()
        if normalized in _REPL_CONTROL_COMMANDS:
            return True
        return (semantic or "").strip().lower() == "interrupt"

    def next_reply_origin(self, fallback: str) -> str:
        """Return the origin that should receive the next assistant reply."""
        if self._pending_replies:
            pending = self.pop_reply_delivery()
            return pending.origin if pending is not None else fallback
        return fallback

    def peek_reply_origin(self) -> str | None:
        """Return the current IM origin without consuming the final-reply slot."""
        if self._pending_replies:
            return self._pending_replies[0].origin
        return None

    def peek_reply_context_token(self) -> str | None:
        """Return the current IM context token without consuming the reply slot."""
        if self._pending_replies:
            return self._pending_replies[0].context_token
        return None

    def peek_reply_delivery_id(self) -> str | None:
        if self._pending_replies:
            return self._pending_replies[0].delivery_id
        return None

    def pop_reply_delivery(self) -> PendingImReply | None:
        if not self._pending_replies:
            return None
        return self._pending_replies.popleft()

    def pop_reply_context(self) -> tuple[str | None, str | None]:
        """Pop the IM origin and context token that triggered this turn."""
        pending = self.pop_reply_delivery()
        if pending is None:
            return None, None
        return pending.origin, pending.context_token

    def pop_reply_origin(self) -> str | None:
        """Pop the IM origin that triggered the current assistant turn.

        Returns None when the deque is empty — the caller can use this to
        distinguish IM-driven turns (populated by ``deliver()``) from
        keyboard-initiated turns (empty deque). Only IM-driven turns
        should send an OUTBOUND reply back to WeChat.
        """
        origin, _context_token = self.pop_reply_context()
        return origin


class QueueFull(Exception):
    """REPL prompt queue is at capacity — reject rather than silently drop."""


__all__ = [
    "EnqueueFn",
    "PermissionProbeFn",
    "PendingImReply",
    "QueueFull",
    "QueueSizeFn",
    "ReplGatewayClient",
    "WakeFn",
]
