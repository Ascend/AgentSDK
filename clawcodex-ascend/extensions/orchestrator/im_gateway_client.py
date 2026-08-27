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

# ruff: noqa: UP009

"""OrchestratorGatewayClient — orchestrator opt-in IM dispatch (P5).

Registered per issue/run session via the gateway UDS. Splits inbound
semantics to existing orchestrator entry points — never invents new
synonyms:

  * ``followUp`` → ``queue_pending_message`` (the existing pending queue)
  * ``pause/resume/stop`` → control socket verbs
  * ``inject`` / ``contextOnly`` → ``issue inject`` / ``.operator_hints.md``
    (NOT the control-socket no-op)
  * ``command`` (``/agent retry|follow-up|unblock``) → existing
    ``parse_agent_command`` path

The client is a pure dispatcher with injectable handlers so it is
unit-testable without a live orchestrator. The daemon wiring binds the
real handlers.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import logging
import math
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from extensions.orchestrator_runtime.adapters.clawcodex_compat import (
    CommandRouter,
    ControlBridge,
    InboundMessage,
    MessageSemantics,
)

logger = logging.getLogger(__name__)

_MAX_COMMAND_OUTPUT_CHARS = 6000
_DEFAULT_PENDING_FLUSH_BATCH_SIZE = 5
_DEFAULT_PENDING_FLUSH_TIMEOUT_SECONDS = 10.0


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_seconds(name: str, value: float, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # Public configuration rejects invalid types and values uniformly.
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004
    normalized = float(value)
    minimum_ok = normalized >= 0 if allow_zero else normalized > 0
    if not math.isfinite(normalized) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} finite number")
    return normalized


def _diagnostic_fingerprint(value: object) -> str:
    try:
        detail = str(value)
    except Exception:  # noqa: BLE001
        detail = "<unprintable>"
    return hashlib.sha256(detail.encode("utf-8", errors="replace")).hexdigest()[:12]


async def _invoke_async(callback: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
    """Invoke one dynamically discovered async capability."""
    return await callback(**kwargs)


class GatewayIpcClientProtocol(Protocol):
    """Required IPC wiring; send capabilities are discovered dynamically."""

    on_deliver: Callable[..., Awaitable[None] | None] | None


@dataclass
class OrchestratorHandlers:
    queue_pending_message: Callable[[str, str], None]  # (issue_id, text) -> None
    control_verb: Callable[[str, str], None]  # (verb, issue_id) -> None
    issue_inject: Callable[[str, str], None]  # (issue_id, hint) -> None
    operator_hints: Callable[[str, str], None]  # (issue_id, text) -> None
    agent_intent: Callable[[str, str], None]  # (verb, issue_id) -> None
    issue_cli: Callable[[str, str, str], None]  # (verb, issue_id, payload) -> None
    bridge_interrupt: Callable[[str, str], None]  # (issue_id, payload) -> None


class OrchestratorGatewayClient:
    def __init__(
        self,
        handlers: OrchestratorHandlers,
        *,
        ipc_client: GatewayIpcClientProtocol | None = None,
        origin: str = "",
        command_router: CommandRouter | None = None,
        control_bridge: ControlBridge | None = None,
        pending_outbound_limit: int = 20,
        pending_retry_base_seconds: float = 60.0,
        pending_retry_max_seconds: float = 300.0,
        pending_flush_batch_size: int = _DEFAULT_PENDING_FLUSH_BATCH_SIZE,
        pending_flush_timeout_seconds: float = _DEFAULT_PENDING_FLUSH_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        cli_runner: Callable[[list[str]], tuple[int, str, str]] | None = None,
    ) -> None:
        self._h = handlers
        self._commands = command_router or CommandRouter()
        self._control = control_bridge or ControlBridge()
        self._ipc: GatewayIpcClientProtocol | None = ipc_client
        self._origin = origin
        self._cli_runner = cli_runner
        self._pending_outbound: deque[str] = deque()
        self._pending_outbound_limit = _positive_int("pending_outbound_limit", pending_outbound_limit)
        self._pending_flush_batch_size = _positive_int("pending_flush_batch_size", pending_flush_batch_size)
        self._pending_flush_timeout_seconds = _finite_seconds(
            "pending_flush_timeout_seconds",
            pending_flush_timeout_seconds,
            allow_zero=False,
        )
        self._flush_lock = asyncio.Lock()
        self._active_flush_task: asyncio.Task[None] | None = None
        self._scheduled_flush: asyncio.TimerHandle | None = None
        self._scheduled_flush_task: asyncio.Task[None] | None = None
        self._clock = clock
        self._pending_retry_base_seconds = _finite_seconds(
            "pending_retry_base_seconds",
            pending_retry_base_seconds,
            allow_zero=True,
        )
        self._pending_retry_max_seconds = _finite_seconds(
            "pending_retry_max_seconds",
            pending_retry_max_seconds,
            allow_zero=False,
        )
        if self._pending_retry_max_seconds < self._pending_retry_base_seconds:
            raise ValueError("pending_retry_max_seconds must be greater than or equal to pending_retry_base_seconds")
        self._pending_retry_delay = self._pending_retry_base_seconds
        self._pending_next_flush_at = 0.0
        if ipc_client is not None:
            # Route server-pushed DELIVER frames through dispatch.
            ipc_client.on_deliver = self._on_pushed_deliver

    async def _on_pushed_deliver(self, frame) -> None:
        """Server-pushed DELIVER (gateway→orchestrator): classify + dispatch."""
        # Phase 3 cleanup: ``InboundMessage`` / ``MessageSemantics`` are
        # already top-level imported via ``clawcodex_compat`` shim, so the
        # redundant function-level lazy import is removed. Full migration to
        # the ``ImInbound`` Protocol dataclass is Phase 4+ (would change the
        # public ``dispatch()`` signature).
        semantic = None
        if frame.semantic:
            with contextlib.suppress(ValueError):
                semantic = MessageSemantics(frame.semantic)
        message = InboundMessage(
            origin=frame.origin or self._origin,
            text=frame.text or "",
            message_id=frame.delivery_id or "",
            channel="",
            semantic=semantic,
        )
        if semantic is None:
            message.semantic = self._classify(message)
            semantic = message.semantic
        try:
            status = self.dispatch(message, semantic)
            await self._flush_pending_outbound(force=True)
            await self._complete_processing(
                frame.delivery_id or "",
                "failure" if status in {"not_dispatched", "command_unroutable"} else "success",
                status,
            )
            logger.info(
                "orchestrator IM push dispatched: delivery_id=%s status=%s",
                (frame.delivery_id or "")[:16],
                status,
            )
        except Exception as exc:  # noqa: BLE001
            await self._complete_processing(
                frame.delivery_id or "",
                "failure",
                "orchestrator dispatch failed",
            )
            logger.warning(
                "orchestrator IM dispatch failed delivery_id=%s error_type=%s",
                (frame.delivery_id or "")[:16],
                type(exc).__name__,
            )

    async def _complete_processing(
        self,
        message_id: str,
        outcome: str,
        reason: str,
    ) -> None:
        complete = getattr(self._ipc, "complete_processing", None)
        if message_id and callable(complete):
            try:
                await _invoke_async(
                    cast(Callable[..., Awaitable[Any]], complete),
                    message_id=message_id,
                    outcome=outcome,
                    reason=reason,
                )
            except (ConnectionError, RuntimeError, OSError):
                logger.debug(
                    "orchestrator processing completion skipped while disconnected: %s",
                    message_id[:16],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "orchestrator processing completion failed message_id=%s error_type=%s",
                    message_id[:16],
                    type(exc).__name__,
                )

    def _classify(self, message):
        # Phase 3: ``MessageClassifier`` is *not* part of the ImChannel
        # Protocol surface (it produces upstream ``MessageSemantics`` enum
        # values consumed by ``dispatch()``). Keep as defensive lazy import
        # for backward compatibility — converting to ``ImInbound.metadata``
        # would change the public ``dispatch()`` signature (Phase 4+).
        from clawcodex_ext.messaging.semantics import MessageClassifier

        return MessageClassifier().classify(message)

    async def send_outbound(self, text: str) -> None:
        """Send a reply / event back to the IM origin via the OUTBOUND frame.

        The origin is the opt-in origin (``im:direct:*:*`` by default for
        orchestrator); the gateway resolves the wildcard to a concrete
        sender at OUTBOUND time. The event is queued only when the send
        cannot start right now (the IPC socket is not open yet) or when the
        gateway explicitly NACKs the send. If the IPC ACK times out, delivery
        is ambiguous: the gateway may already have sent the IM message but
        returned its ACK too late. In that case we do not auto-retry, because
        duplicate chat messages are worse than a best-effort dropped event.
        """
        if self._ipc is None or not self._origin:
            return
        if text in self._pending_outbound:
            logger.debug("orchestrator IM outbound deduped before send text_len=%d", len(text))
            await self._flush_pending_outbound()
            return
        if self._pending_outbound:
            self._queue_pending_outbound(text)
            await self._flush_pending_outbound()
            return
        sent = await self._send_to_origin(self._origin, text)
        if not sent:
            self._queue_pending_outbound(text)
            if self._pending_next_flush_at > self._clock():
                self._schedule_pending_flush(self._pending_next_flush_at - self._clock())

    async def _send_to_origin(self, origin: str, text: str) -> bool:
        if self._ipc is None:
            return False
        try:
            send_with_result = getattr(self._ipc, "send_outbound_result", None)
            if callable(send_with_result):
                result = await _invoke_async(
                    cast(Callable[..., Awaitable[Any]], send_with_result),
                    origin=origin,
                    text=text,
                )
                outcome = getattr(getattr(result, "outcome", None), "value", None)
                response = getattr(result, "response", None)
                if outcome == "not_sent":
                    logger.debug("orchestrator IM outbound was not sent; queueing")
                    return False
                if outcome == "ambiguous_timeout":
                    logger.warning("orchestrator IM outbound ACK timed out; not retrying to avoid duplicates")
                    self._reset_pending_flush_backoff()
                    return True
                if outcome not in {"accepted", "rejected"}:
                    logger.warning("orchestrator IM outbound returned an unknown send outcome")
                    return False
            else:
                legacy_send = getattr(self._ipc, "send_outbound", None)
                if not callable(legacy_send):
                    return False
                response = await _invoke_async(
                    cast(Callable[..., Awaitable[Any]], legacy_send),
                    origin=origin,
                    text=text,
                )
        except (ConnectionError, OSError, RuntimeError) as exc:
            # Not connected yet (heartbeat loop hasn't run / reconnected).
            # Queue so the post-register flush delivers it; never propagate
            # — IM must not break the orchestrator main loop.
            logger.debug(
                "orchestrator IM outbound not connected; queueing error_type=%s",
                type(exc).__name__,
            )
            return False
        if response is None:
            logger.warning("orchestrator IM outbound ACK timed out; not retrying to avoid duplicate IM delivery")
            self._reset_pending_flush_backoff()
            return True
        response_type = getattr(getattr(response, "type", None), "value", None)
        if response_type == "nack":
            reason = getattr(response, "reason", "") or ""
            logger.warning(
                "orchestrator IM outbound rejected reason_fingerprint=%s",
                _diagnostic_fingerprint(reason),
            )
            self._defer_pending_flush("nack")
            return False
        self._reset_pending_flush_backoff()
        return True

    def _queue_pending_outbound(self, text: str) -> None:
        # Skip exact duplicates already waiting in the queue — e.g. the
        # orchestrator emits "clawcodex-orchestrator: IM notifications
        # enabled" on every reconnect, and if the gateway can't resolve
        # the wildcard origin (operator hasn't messaged recently), each
        # copy would queue and all would flush at once when the operator
        # finally sends a message.
        if text in self._pending_outbound:
            logger.debug("orchestrator IM outbound deduped text_len=%d", len(text))
            return
        if len(self._pending_outbound) >= self._pending_outbound_limit:
            self._pending_outbound.popleft()
            logger.warning("orchestrator IM outbound pending queue full; dropped oldest event")
        self._pending_outbound.append(text)
        logger.info("orchestrator IM outbound queued (pending connection or send retry)")

    def _defer_pending_flush(self, reason_code: str) -> None:
        delay = self._pending_retry_delay
        self._pending_next_flush_at = self._clock() + delay
        self._pending_retry_delay = min(
            delay * 2 if delay > 0 else self._pending_retry_base_seconds,
            self._pending_retry_max_seconds,
        )
        logger.info(
            "orchestrator IM outbound retry deferred delay=%.1fs reason_code=%s",
            delay,
            reason_code,
        )

    def _reset_pending_flush_backoff(self) -> None:
        self._pending_retry_delay = self._pending_retry_base_seconds
        self._pending_next_flush_at = 0.0

    async def _flush_pending_outbound(self, *, force: bool = False) -> None:
        if force:
            self._reset_pending_flush_backoff()
        elif self._pending_next_flush_at > self._clock():
            return
        self._cancel_scheduled_flush()
        task = self._active_flush_task
        if task is None or task.done():
            task = asyncio.create_task(self._flush_pending_outbound_batch())
            self._active_flush_task = task
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._pending_flush_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("orchestrator IM pending outbound flush timed out")

    def _cancel_scheduled_flush(self) -> None:
        handle = self._scheduled_flush
        if handle is not None and not handle.cancelled():
            handle.cancel()
        self._scheduled_flush = None

    def _schedule_pending_flush(self, delay: float = 0.0) -> None:
        if not self._pending_outbound:
            return
        loop = asyncio.get_running_loop()
        scheduled_for = loop.time() + max(0.0, delay)
        handle = self._scheduled_flush
        if handle is not None and not handle.cancelled():
            if handle.when() <= scheduled_for:
                return
            handle.cancel()
        self._scheduled_flush = loop.call_later(max(0.0, delay), self._start_scheduled_flush)

    def _start_scheduled_flush(self) -> None:
        self._scheduled_flush = None
        self._scheduled_flush_task = asyncio.create_task(self._flush_pending_outbound())

    async def _flush_pending_outbound_batch(self) -> None:
        # Serialise flush calls — the heartbeat loop and the inbound push
        # handler can both call this concurrently; without a lock, both
        # peek self._pending_outbound[0] before an await, then both try
        # popleft(), causing IndexError: pop from an empty deque.
        async with self._flush_lock:
            if not self._pending_outbound:
                return
            origin = self._origin
            if not origin:
                return
            if self._pending_next_flush_at > self._clock():
                logger.debug(
                    "orchestrator IM pending outbound flush deferred for %.1fs",
                    self._pending_next_flush_at - self._clock(),
                )
                return
            # Wildcard origins are flushed too: the gateway resolves them at
            # OUTBOUND time (recent sender, else persisted context tokens).
            sent_count = 0
            while self._pending_outbound and sent_count < self._pending_flush_batch_size:
                text = self._pending_outbound[0]
                try:
                    sent = await self._send_to_origin(origin, text)
                    if not sent:
                        return
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "orchestrator IM pending outbound flush failed error_type=%s",
                        type(exc).__name__,
                    )
                    return
                self._pending_outbound.popleft()
                sent_count += 1
            if self._pending_outbound:
                self._schedule_pending_flush()

    def dispatch(self, message: InboundMessage, semantic: MessageSemantics) -> str:
        """Route ``message`` to the right existing orchestrator entry.

        Returns a short status string describing the dispatch (for ack).
        """
        issue_id = self._issue_id(message)
        if semantic is MessageSemantics.FOLLOW_UP:
            self._h.queue_pending_message(issue_id, message.text)
            return "followup_queued"
        if semantic is MessageSemantics.CONTEXT_ONLY:
            self._h.operator_hints(issue_id, message.text)
            return "context_only_recorded"
        if semantic is MessageSemantics.INTERRUPT:
            # interrupt maps to control verbs via the bridge
            ctrl = self._control.resolve(MessageSemantics.INTERRUPT, None)
            if ctrl is not None:
                self._h.bridge_interrupt(issue_id, ctrl.payload)
            return "interrupt_dispatched"
        if semantic is MessageSemantics.COMMAND:
            route = self._commands.route(message)
            if route is None:
                return "command_unroutable"
            if route.kind == "orchestrator_cli":
                return self._dispatch_orchestrator_cli(route)
            if route.kind == "agent_intent":
                self._h.agent_intent(route.verb, route.issue_hint or issue_id)
                return f"agent_{route.verb}"
            # control_verb
            ctrl = self._control.resolve(semantic, route)
            if ctrl is None:
                return "command_unroutable"
            if ctrl.surface == "control_socket":
                self._h.control_verb(ctrl.verb, ctrl.issue_hint or issue_id)
                return f"control_{ctrl.verb}"
            if ctrl.surface == "issue_inject":
                self._h.issue_inject(ctrl.issue_hint or issue_id, route.payload)
                return "inject_delivered"
            if ctrl.surface == "issue_cli":
                self._h.issue_cli(ctrl.verb, ctrl.issue_hint or issue_id, ctrl.payload)
                return f"issue_cli_{ctrl.verb}"
            return f"issue_cli_{ctrl.verb}"
        # newPrompt / approval → leave to the host agent / approval binding
        return "not_dispatched"

    def _dispatch_orchestrator_cli(self, route) -> str:
        argv = list(route.argv)
        if len(argv) < 2:
            self._queue_command_reply(route.payload, 2, "", "error: invalid orchestrator command")
            return "orchestrator_cli_invalid"

        noun, verb = argv[0], argv[1]
        if noun == "issue" and verb in {"stop", "pause", "resume"}:
            issue_id = route.issue_hint or self._arg_value(argv, "--id")
            if not issue_id:
                self._queue_command_reply(route.payload, 2, "", "error: --id is required")
                return f"orchestrator_cli_issue_{verb}"
            self._h.control_verb(verb, issue_id)
            self._queue_command_reply(
                route.payload,
                0,
                f"Control command '{verb}' sent for issue {issue_id}",
                "",
            )
            return f"orchestrator_cli_issue_{verb}"

        if noun == "issue" and verb == "tail":
            self._queue_command_reply(route.payload, 0, self._tail_notice(argv), "")
            return "orchestrator_cli_issue_tail"

        rc, stdout, stderr = self._run_orchestrator_cli(argv)
        self._queue_command_reply(route.payload, rc, stdout, stderr)
        return f"orchestrator_cli_{noun}_{verb}"

    def _run_orchestrator_cli(self, argv: list[str]) -> tuple[int, str, str]:
        if self._cli_runner is not None:
            return self._cli_runner(list(argv))

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                # Phase 3: ``run_orchestrator_subcommand`` lives in
                # ``clawcodex_ext.entrypoints.orchestrator`` — not part of
                # the ImChannel Protocol surface. Existing ``cli_runner``
                # injection (lines above) already provides the
                # Protocol-style replacement path; this fallback is kept as
                # an ImportError-safety net for tests that don't inject.
                from clawcodex_ext.entrypoints.orchestrator import (
                    run_orchestrator_subcommand,
                )

                rc = run_orchestrator_subcommand(list(argv))
            except SystemExit as exc:
                code = exc.code
                rc = code if isinstance(code, int) else 1
            except Exception as exc:  # noqa: BLE001
                print(
                    f"error: command execution failed ({type(exc).__name__})",
                    file=sys.stderr,
                )
                rc = 1
        return rc, stdout.getvalue(), stderr.getvalue()

    def _queue_command_reply(self, command_text: str, rc: int, stdout: str, stderr: str) -> None:
        text = self._format_command_reply(command_text, rc, stdout, stderr)
        self._queue_pending_outbound(text)

    @staticmethod
    def _format_command_reply(command_text: str, rc: int, stdout: str, stderr: str) -> str:
        command = (command_text or "").strip() or "<empty>"
        prefix = "Command completed" if rc == 0 else f"Command failed ({rc})"
        output = "\n".join(part.strip() for part in (stdout, stderr) if part and part.strip())
        if not output:
            return f"{prefix}: {command}"
        if len(output) > _MAX_COMMAND_OUTPUT_CHARS:
            output = output[:_MAX_COMMAND_OUTPUT_CHARS].rstrip() + "\n..."
        return f"{prefix}: {command}\n\n{output}"

    @staticmethod
    def _tail_notice(argv: list[str]) -> str:
        issue_id = OrchestratorGatewayClient._arg_value(argv, "--id") or "<issue-id>"
        return (
            f"/issue tail --id {issue_id} is a streaming command. "
            "IM returns this bounded notice instead of holding the gateway connection. "
            "Run `clawcodex-dev orchestrator issue tail --id "
            f"{issue_id}` locally for live tailing."
        )

    @staticmethod
    def _arg_value(argv: list[str], flag: str) -> str | None:
        for idx, token in enumerate(argv):
            if token == flag and idx + 1 < len(argv):
                return argv[idx + 1]
            if token.startswith(f"{flag}="):
                return token.split("=", 1)[1]
        return None

    @staticmethod
    def _issue_id(message: InboundMessage) -> str:
        if message.raw and isinstance(message.raw, dict):
            iid = message.raw.get("issue_id")
            if iid:
                return str(iid)
        return ""


__all__ = ["OrchestratorGatewayClient", "OrchestratorHandlers"]
