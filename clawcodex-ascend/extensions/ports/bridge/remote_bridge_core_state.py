# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#

"""Internal state machine and helpers for the env-less remote bridge.

Holds all mutable per-bridge state (``_BridgeState``) plus the transport
teardown/archive helpers used by the core initializer in
:mod:`extensions.ports.bridge.remote_bridge_core`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx

from clawcodex_ext.bridge.bounded_uuid_set import BoundedUUIDSet
from clawcodex_ext.bridge.code_session_api import (
    RemoteCredentials,
    fetch_remote_credentials,
)
from clawcodex_ext.bridge.env_less_bridge_config import (
    EnvLessBridgeConfig,
)
from clawcodex_ext.bridge.flush_gate import FlushGate
from clawcodex_ext.bridge.jwt_utils import TokenRefreshScheduler
from clawcodex_ext.bridge.messaging import (
    extract_title_text,
    handle_ingress_message,
    is_eligible_bridge_message,
    make_result_message,
)
from clawcodex_ext.bridge.messaging_handlers import (
    ServerControlRequestHandlers,
    handle_server_control_request,
)
from clawcodex_ext.bridge.repl_bridge_transport import (
    ReplBridgeTransport,
    V2TransportOptions,
)
from clawcodex_ext.bridge.session_id_compat import to_compat_session_id
from clawcodex_ext.bridge.work_secret import build_ccr_v2_sdk_url
from clawcodex_ext.types.messages import Message
from clawcodex_ext.utils.message_mappers import to_sdk_messages

if TYPE_CHECKING:
    from extensions.ports.bridge.remote_bridge_core import (
        BridgeState,
        EnvLessBridgeParams,
        OnPermissionResponse,
        OnStateChange,
    )

logger = logging.getLogger(__name__)

_TEARDOWN_RESULT_WRITE_TIMEOUT_SECONDS = 0.5
"""Cap on the teardown result-write enqueue wait. ``gracefulShutdown``
races teardown against a 2s budget; 0.5s leaves headroom for archive +
close while still containing back-pressure stalls."""


@dataclass
class _BridgeState:
    """All mutable state for one env-less bridge.

    Lives in its own class so the closure-heavy TS code maps cleanly to
    Python — each "inner function" in TS becomes a method that reads/
    writes ``self``.
    """

    params: EnvLessBridgeParams
    cfg: EnvLessBridgeConfig
    session_id: str
    credentials: RemoteCredentials
    transport: ReplBridgeTransport
    http_client: httpx.AsyncClient | None
    transport_factory: Callable[[V2TransportOptions], Awaitable[ReplBridgeTransport]]
    refresh: TokenRefreshScheduler | None = None

    recent_posted_uuids: BoundedUUIDSet = field(init=False)
    initial_message_uuids: set[str] = field(default_factory=set)
    recent_inbound_uuids: BoundedUUIDSet = field(init=False)
    flush_gate: FlushGate[Message] = field(default_factory=FlushGate)

    initial_flush_done: bool = False
    torn_down: bool = False
    auth_recovery_in_flight: bool = False
    user_message_callback_done: bool = False
    connect_cause: str = "initial"

    # Strong references to fire-and-forget background tasks. Without this,
    # ``asyncio.create_task`` results are only weakly held by the loop and
    # may be GC'd mid-flight; on teardown they also leak (the socket write
    # never gets cancelled). We track them and cancel on teardown.
    _pending_tasks: set[asyncio.Task[Any]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        # UUID dedup ring buffers, sized per env-less config.
        self.recent_posted_uuids = BoundedUUIDSet(self.cfg.uuid_dedup_buffer_size)
        self.recent_inbound_uuids = BoundedUUIDSet(self.cfg.uuid_dedup_buffer_size)

        # Seed dedup with initial-history UUIDs so server echoes of the
        # flushed history are recognized as our own.
        if self.params.initial_messages:
            for msg in self.params.initial_messages:
                self.initial_message_uuids.add(msg.uuid)
                self.recent_posted_uuids.add(msg.uuid)

        # Latch onUserMessage as already-done when no callback was wired.
        self.user_message_callback_done = self.params.on_user_message is None

    # ── Callback wiring ────────────────────────────────────────────────

    def wire_transport_callbacks(self) -> None:
        """Wire SSE setOnConnect / setOnData / setOnClose callbacks.

        Re-callable after a transport rebuild — captures ``self.transport``
        lazily via the closures so the new transport receives the wiring.
        """
        active_transport = self.transport
        self.transport.set_on_connect(lambda: self._on_connect(active_transport))
        self.transport.set_on_data(self._on_data)
        self.transport.set_on_close(self._on_close)

    def _on_connect(self, flush_transport: ReplBridgeTransport) -> None:
        """Fired when the transport handshake completes (CCR initialized).

        At this point ``CCRClient.is_initialized`` is True (the v2 transport
        wires this callback after ``await ccr.initialize()`` resolves), so
        it's safe to drain the flushGate — any queued writes will reach the
        uploader and be POSTed. Flushes initial history first if present.

        ``flush_transport`` is captured at wire time — if a 401/teardown
        swaps ``self.transport`` mid-flush, the stale callback is a no-op
        (matches TS guard pattern).
        """
        logger.debug("[remote-bridge] v2 transport connected")
        if not self.initial_flush_done and self.params.initial_messages and len(self.params.initial_messages) > 0:
            self.initial_flush_done = True
            self._spawn(self._flush_history_then_drain(flush_transport))
            return
        # No initial flush — drain any pre-connect queued writes
        # (``write_messages`` calls that landed during the handshake) and
        # close the gate so subsequent writes go straight to the uploader.
        if self.flush_gate.active:
            self._drain_flush_gate()
        _fire_state(self.params.on_state_change, "connected")

    async def _flush_history_then_drain(self, flush_transport: ReplBridgeTransport) -> None:
        """Async wrapper for flush + drain. Mirrors TS .finally() chain.

        Stale-transport guards: if the transport was swapped mid-flush
        (401 recovery, teardown), skip the drain — the new transport's
        re-wired onConnect will re-flush.
        """
        try:
            assert self.params.initial_messages is not None
            await self._flush_history(self.params.initial_messages)
        except Exception as err:  # noqa: BLE001  log + carry on
            logger.warning("[remote-bridge] flushHistory failed: %s", err)
        finally:
            if self.transport is flush_transport and not self.torn_down and not self.auth_recovery_in_flight:
                self._drain_flush_gate()
                _fire_state(self.params.on_state_change, "connected")

    def _on_data(self, data: str) -> None:
        """Route an SSE event line to the ingress handler."""
        params = self.params
        transport = self.transport

        on_permission_response_wrapped: OnPermissionResponse | None = None
        if params.on_permission_response is not None:
            inner = params.on_permission_response

            def wrapped(response: dict[str, Any]) -> None:
                # Remote client answered the prompt — turn resumes.
                # Without reportState('running'), the server stays on
                # ``requires_action`` until the next user message or
                # turn-end result.
                transport.report_state({"state": "running"})
                inner(response)

            on_permission_response_wrapped = wrapped

        def on_control_request(request: dict[str, Any]) -> None:
            handle_server_control_request(
                request,
                ServerControlRequestHandlers(
                    transport=transport,  # type: ignore[arg-type]
                    session_id=self.session_id,
                    outbound_only=params.outbound_only,
                    on_interrupt=params.on_interrupt,
                    on_set_model=params.on_set_model,
                    on_set_max_thinking_tokens=params.on_set_max_thinking_tokens,
                    on_set_permission_mode=params.on_set_permission_mode,
                ),
            )

        handle_ingress_message(
            data,
            self.recent_posted_uuids,
            self.recent_inbound_uuids,
            params.on_inbound_message,
            on_permission_response_wrapped,
            on_control_request,
        )

    def _on_close(self, code: int | None) -> None:
        """Fired when the transport closes terminally.

        Per TS comment: onClose fires only for TERMINAL failures —
        401 (JWT invalid), 4090 (epoch mismatch), 4091 (init failed),
        or SSE reconnect-budget exhausted. Transient drops are handled
        inside SSETransport. 401 is recoverable (fetch fresh JWT,
        rebuild transport); other codes are dead-ends.
        """
        if self.torn_down:
            return
        logger.debug("[remote-bridge] v2 transport closed (code=%s)", code)
        if code == 401 and not self.auth_recovery_in_flight:
            self._spawn(self._recover_from_auth_failure())
            return
        _fire_state(
            self.params.on_state_change,
            "failed",
            f"Transport closed (code {code})",
        )

    # ── Transport rebuild (shared by proactive refresh + 401 recovery) ──

    async def _rebuild_transport(
        self,
        fresh: RemoteCredentials,
        cause: str,
    ) -> None:
        """Replace the transport with a fresh JWT+epoch.

        Caller MUST set ``self.auth_recovery_in_flight = True`` before
        calling (synchronously, before any ``await``) and clear it in a
        ``finally``. Moving that here would be too late to prevent a
        double ``/bridge`` fetch.
        """
        self.connect_cause = cause
        # Queue writes during rebuild — once /bridge returns, the old
        # transport's epoch is stale and its next write/heartbeat 409s.
        # ``deactivate()`` (vs ``drop()``) on success leaves queued items
        # for the new transport's ``_on_connect`` to drain.
        self.flush_gate.start()
        success = False
        try:
            old_seq = self.transport.get_last_sequence_num()
            self.transport.close()
            self.transport = await self.transport_factory(
                V2TransportOptions(
                    session_url=build_ccr_v2_sdk_url(fresh.api_base_url, self.session_id),
                    ingress_token=fresh.worker_jwt,
                    session_id=self.session_id,
                    epoch=fresh.worker_epoch,
                    heartbeat_interval_seconds=(self.cfg.heartbeat_interval_ms / 1000.0),
                    heartbeat_jitter_fraction=(self.cfg.heartbeat_jitter_fraction),
                    initial_sequence_num=old_seq,
                    outbound_only=self.params.outbound_only,
                    get_auth_token=_freeze_token(fresh.worker_jwt),
                )
            )
            if self.torn_down:
                # Teardown fired during the async factory window — don't
                # wire/connect/schedule; we'd re-arm timers after cancelAll
                # and fire onInboundMessage into a torn-down bridge.
                self.transport.close()
                return
            self.wire_transport_callbacks()
            self.transport.connect()
            if self.refresh is not None:
                self.refresh.schedule_from_expires_in(self.session_id, fresh.expires_in)
            self.credentials = fresh
            # Leave the gate active — the new transport's ``_on_connect``
            # will drain it once CCR.initialize() resolves. Python
            # ``CCRClient.write_event`` silently drops while
            # ``_initialized is False`` (TS ``SerialBatchEventUploader``
            # queues instead — known divergence), so draining here would
            # lose every queued message between now and the SSE/CCR
            # handshake completing.
            success = True
        finally:
            # Failure path (exception unwound the try, or torn_down
            # short-circuit): drop the queued items because the new
            # transport isn't going to come up.
            if not success:
                self.flush_gate.drop()

    # ── 401 recovery ───────────────────────────────────────────────────

    async def _recover_from_auth_failure(self) -> None:
        """Recover from a 401 on the SSE stream.

        Refresh OAuth, re-fetch ``/bridge``, rebuild transport. Shared
        ``auth_recovery_in_flight`` flag with the proactive refresh path
        prevents double-bumping epoch.
        """
        if self.auth_recovery_in_flight:
            return
        self.auth_recovery_in_flight = True
        _fire_state(
            self.params.on_state_change,
            "reconnecting",
            "JWT expired — refreshing",
        )
        logger.debug("[remote-bridge] 401 on SSE — attempting JWT refresh")
        try:
            # getAccessToken() returns expired tokens as non-None strings
            # — unconditional OAuth refresh below catches that.
            stale = self.params.get_access_token()
            if self.params.on_auth_401 is not None:
                await self.params.on_auth_401(stale or "")
            oauth_token = self.params.get_access_token() or stale
            if not oauth_token or self.torn_down:
                if not self.torn_down:
                    _fire_state(
                        self.params.on_state_change,
                        "failed",
                        "JWT refresh failed: no OAuth token",
                    )
                return
            fresh = await _with_retry(
                lambda: fetch_remote_credentials(
                    self.session_id,
                    self.params.base_url,
                    oauth_token,
                    timeout_seconds=self.cfg.http_timeout_ms / 1000.0,
                    client=self.http_client,
                ),
                "fetchRemoteCredentials (recovery)",
                self.cfg,
            )
            if fresh is None or self.torn_down:
                if not self.torn_down:
                    _fire_state(
                        self.params.on_state_change,
                        "failed",
                        "JWT refresh failed after 401",
                    )
                return
            # If 401 interrupted the initial flush, writeBatch may have
            # silently no-op'd on the closed uploader. Reset so the new
            # onConnect re-flushes.
            self.initial_flush_done = False
            await self._rebuild_transport(fresh, "auth_401_recovery")
            logger.debug("[remote-bridge] Transport rebuilt after 401")
        except Exception as err:  # noqa: BLE001  log + surface
            logger.error("[remote-bridge] 401 recovery failed: %s", err)
            if not self.torn_down:
                _fire_state(
                    self.params.on_state_change,
                    "failed",
                    f"JWT refresh failed: {err}",
                )
        finally:
            self.auth_recovery_in_flight = False

    def on_jwt_refresh(self, session_id: str, oauth_token: str) -> None:
        """Called by ``TokenRefreshScheduler`` 5min before JWT expiry.

        Re-fetches ``/bridge`` and rebuilds the transport. Serialized
        against ``recover_from_auth_failure`` via ``auth_recovery_in_flight``
        so a laptop-wake double-fire doesn't bump epoch twice.
        """

        async def _do() -> None:
            if self.auth_recovery_in_flight or self.torn_down:
                logger.debug("[remote-bridge] Recovery already in flight, skipping proactive refresh")
                return
            self.auth_recovery_in_flight = True
            try:
                fresh = await _with_retry(
                    lambda: fetch_remote_credentials(
                        session_id,
                        self.params.base_url,
                        oauth_token,
                        timeout_seconds=self.cfg.http_timeout_ms / 1000.0,
                        client=self.http_client,
                    ),
                    "fetchRemoteCredentials (proactive)",
                    self.cfg,
                )
                if fresh is None or self.torn_down:
                    return
                await self._rebuild_transport(fresh, "proactive_refresh")
                logger.debug("[remote-bridge] Transport rebuilt (proactive refresh)")
            except Exception as err:  # noqa: BLE001
                logger.error(
                    "[remote-bridge] Proactive refresh rebuild failed: %s",
                    err,
                )
                if not self.torn_down:
                    _fire_state(
                        self.params.on_state_change,
                        "failed",
                        f"Refresh failed: {err}",
                    )
            finally:
                self.auth_recovery_in_flight = False

        self._spawn(_do())

    # ── History flush + drain ──────────────────────────────────────────

    async def _flush_history(self, msgs: list[Message]) -> None:
        """POST a capped, eligible-only slice of initial history.

        Cap via ``initial_history_cap``; filter via
        ``is_eligible_bridge_message``. The cap takes the *tail* to keep
        the most recent context. If the eligible tail is a user message
        (pre-cap), reports state ``'running'`` so the web UI shows the
        turn-in-progress indicator immediately on connect.
        """
        eligible = [m for m in msgs if is_eligible_bridge_message(_msg_dict(m))]
        cap = self.params.initial_history_cap
        capped = eligible[-cap:] if 0 < cap < len(eligible) else eligible
        if len(capped) < len(eligible):
            logger.debug(
                "[remote-bridge] Capped initial flush: %s -> %s (cap=%s)",
                len(eligible),
                len(capped),
                cap,
            )
        events = [{**m, "session_id": self.session_id} for m in to_sdk_messages(capped)]
        if not events:
            return
        # Pre-cap eligible tail decides reportState: the cap may truncate
        # to a user message even when the actual trailing message is
        # assistant. Match TS behavior exactly.
        if eligible and _msg_dict(eligible[-1]).get("type") == "user":
            self.transport.report_state({"state": "running"})
        logger.debug("[remote-bridge] Flushing %s history events", len(events))
        await self.transport.write_batch(events)

    def _drain_flush_gate(self) -> None:
        """Send any messages queued during the flush window."""
        msgs = self.flush_gate.end()
        if not msgs:
            return
        for msg in msgs:
            self.recent_posted_uuids.add(msg.uuid)
        events = [{**m, "session_id": self.session_id} for m in to_sdk_messages(msgs)]
        if any(_msg_dict(m).get("type") == "user" for m in msgs):
            self.transport.report_state({"state": "running"})
        logger.debug(
            "[remote-bridge] Drained %s queued message(s) after flush",
            len(msgs),
        )
        self._spawn(self.transport.write_batch(events))

    # ── Public handle methods ──────────────────────────────────────────

    def write_messages(self, messages: list[Message]) -> None:
        """Write a batch of local Message[] to the transport."""
        filtered = [
            m
            for m in messages
            if is_eligible_bridge_message(_msg_dict(m))
            and m.uuid not in self.initial_message_uuids
            and not self.recent_posted_uuids.has(m.uuid)
        ]
        if not filtered:
            return

        # Title derivation — scan BEFORE the flushGate check so prompts
        # queued during flush still count.
        if not self.user_message_callback_done:
            for m in filtered:
                text = extract_title_text(_msg_dict(m))
                cb = self.params.on_user_message
                if text is not None and cb is not None and cb(text, self.session_id):
                    self.user_message_callback_done = True
                    break

        if self.flush_gate.enqueue(*filtered):
            logger.debug(
                "[remote-bridge] Queued %s message(s) during flush",
                len(filtered),
            )
            return

        for msg in filtered:
            self.recent_posted_uuids.add(msg.uuid)
        events = [{**m, "session_id": self.session_id} for m in to_sdk_messages(filtered)]
        if any(_msg_dict(m).get("type") == "user" for m in filtered):
            self.transport.report_state({"state": "running"})
        logger.debug("[remote-bridge] Sending %s message(s)", len(filtered))
        self._spawn(self.transport.write_batch(events))

    def write_sdk_messages(self, messages: list[dict[str, Any]]) -> None:
        """Write pre-shaped SDK messages (used by the daemon path)."""
        filtered = [m for m in messages if not m.get("uuid") or not self.recent_posted_uuids.has(m["uuid"])]
        if not filtered:
            return
        for msg in filtered:
            uid = msg.get("uuid")
            if uid:
                self.recent_posted_uuids.add(uid)
        events = [{**m, "session_id": self.session_id} for m in filtered]
        self._spawn(self.transport.write_batch(events))

    def send_control_request(self, request: dict[str, Any]) -> None:
        if self.auth_recovery_in_flight:
            logger.debug(
                "[remote-bridge] Dropping control_request during 401 recovery: %s",
                request.get("request_id"),
            )
            return
        event = {**request, "session_id": self.session_id}
        inner = request.get("request") or {}
        if inner.get("subtype") == "can_use_tool":
            self.transport.report_state({"state": "requires_action"})
        self._spawn(self.transport.write(event))
        logger.debug(
            "[remote-bridge] Sent control_request request_id=%s",
            request.get("request_id"),
        )

    def send_control_response(self, response: dict[str, Any]) -> None:
        if self.auth_recovery_in_flight:
            logger.debug("[remote-bridge] Dropping control_response during 401 recovery")
            return
        event = {**response, "session_id": self.session_id}
        self.transport.report_state({"state": "running"})
        self._spawn(self.transport.write(event))
        logger.debug("[remote-bridge] Sent control_response")

    def send_cancel_request(self, request_id: str) -> None:
        if self.auth_recovery_in_flight:
            logger.debug(
                "[remote-bridge] Dropping control_cancel_request during 401 recovery: %s",
                request_id,
            )
            return
        event = {
            "type": "control_cancel_request",
            "request_id": request_id,
            "session_id": self.session_id,
        }
        # Hook/classifier/channel/recheck resolved the permission locally
        # — interactiveHandler calls only cancelRequest (no sendResponse)
        # on those paths, so without this the server stays on
        # requires_action.
        self.transport.report_state({"state": "running"})
        self._spawn(self.transport.write(event))
        logger.debug(
            "[remote-bridge] Sent control_cancel_request request_id=%s",
            request_id,
        )

    def send_result(self) -> None:
        if self.auth_recovery_in_flight:
            logger.debug("[remote-bridge] Dropping result during 401 recovery")
            return
        self.transport.report_state({"state": "idle"})
        self._spawn(self.transport.write(make_result_message(self.session_id)))
        logger.debug("[remote-bridge] Sent result")

    # ── Teardown ───────────────────────────────────────────────────────

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        """Schedule a fire-and-forget coroutine with lifecycle tracking.

        Keeps a strong reference until completion (prevents premature GC of
        the task) and registers it for cancellation on ``teardown``. The
        done-callback discards the reference so the set doesn't grow.
        """
        task = asyncio.ensure_future(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def _cancel_pending_tasks(self) -> None:
        """Cancel and await all in-flight fire-and-forget tasks.

        Snapshot the set first — the done-callback mutates it as tasks
        settle. Cancellation is best-effort; we swallow each task's outcome
        (including ``CancelledError``) so one stuck write can't block the
        teardown cap.
        """
        pending = list(self._pending_tasks)
        if not pending:
            return
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def teardown(self) -> None:
        """Idempotent shutdown.

        1. Cancel JWT refresh scheduler.
        2. Drop the flush gate (any queued messages are lost — transport
           is about to close).
        3. ``reportState('idle')`` + write result message (fire-and-forget;
           archive latency covers the uploader drain).
        4. Archive the session via the v2 compat archive endpoint, with a
           single 401 retry through ``on_auth_401``.
        5. Close the transport.
        """
        if self.torn_down:
            return
        self.torn_down = True
        if self.refresh is not None:
            self.refresh.cancel_all()
        self.flush_gate.drop()
        await self._cancel_pending_tasks()

        # Fire the result message before archive. ``transport.write()``
        # in Python awaits enqueue into ``CCRClient.SerialBatchEventUploader``
        # (fast on the happy path) but blocks up to ``producer_timeout_seconds``
        # (default 30s) when the queue is full. ``gracefulShutdown`` races
        # this against a 2s cap, so bound the wait tightly — losing the
        # result message under back-pressure is preferable to blocking
        # teardown past the cap.
        self.transport.report_state({"state": "idle"})
        try:
            await asyncio.wait_for(
                self.transport.write(make_result_message(self.session_id)),
                timeout=_TEARDOWN_RESULT_WRITE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[remote-bridge] Result write timed out after %ss (producer queue back-pressure)",
                _TEARDOWN_RESULT_WRITE_TIMEOUT_SECONDS,
            )
        except Exception as err:  # noqa: BLE001  log + carry on
            logger.warning("[remote-bridge] Result write failed: %s", err)

        token = self.params.get_access_token()
        archive_timeout = self.cfg.teardown_archive_timeout_ms / 1000.0
        status = await _archive_v2_session(
            self.session_id,
            self.params.base_url,
            token,
            self.params.org_uuid,
            archive_timeout,
            self.http_client,
        )

        # Single 401 retry — token might be stale post-laptop-wake.
        if status == 401 and self.params.on_auth_401 is not None:
            try:
                await self.params.on_auth_401(token or "")
                token = self.params.get_access_token()
                status = await _archive_v2_session(
                    self.session_id,
                    self.params.base_url,
                    token,
                    self.params.org_uuid,
                    archive_timeout,
                    self.http_client,
                )
            except Exception as err:  # noqa: BLE001
                logger.error("[remote-bridge] Teardown 401 retry threw: %s", err)

        self.transport.close()
        logger.debug("[remote-bridge] Torn down (archive=%s)", status)


# ── Helpers ───────────────────────────────────────────────────────────────


def _freeze_token(token: str) -> Callable[[], str | None]:
    """Capture the JWT in a closure so the transport reads it stably.

    Per-instance auth header source; bypasses
    ``CLAUDE_CODE_SESSION_ACCESS_TOKEN`` env var which mcp/client reads
    ungatedly. Frozen at construction — the transport is fully rebuilt
    on JWT refresh, so the closure is reissued each time.
    """
    return lambda: token


def _fire_state(
    cb: OnStateChange | None,
    state: BridgeState,
    detail: str | None = None,
) -> None:
    """Invoke ``on_state_change`` if wired, swallowing exceptions."""
    if cb is None:
        return
    try:
        if detail is None:
            cb(state)
        else:
            cb(state, detail)
    except Exception as err:  # noqa: BLE001
        logger.warning("[remote-bridge] on_state_change callback raised: %s", err)


def _msg_dict(msg: Message | dict[str, Any]) -> dict[str, Any]:
    """Coerce a Message dataclass to the wire-format dict shape.

    The dict-shaped predicates in ``bridge.messaging`` mirror the TS
    wire format where user/assistant messages have a nested
    ``message: {role, content}`` field. Python ``UserMessage`` /
    ``AssistantMessage`` keep ``role`` and ``content`` flat at the top
    level, so a shallow ``vars()`` would give ``extract_title_text``
    nothing to extract. We synthesize the nested ``message`` field
    on top of the flat ``vars()`` so both ``is_eligible_bridge_message``
    (reads ``type``/``isVirtual``/``subtype``) AND ``extract_title_text``
    (reads ``message.content``) work against the same dict.
    """
    if isinstance(msg, dict):
        return msg
    flat = vars(msg).copy()
    msg_type = flat.get("type")
    # Only user/assistant carry inner ``message.content`` in the wire
    # format; system and others stay flat.
    if msg_type in ("user", "assistant") and "message" not in flat:
        flat["message"] = {
            "role": flat.get("role", msg_type),
            "content": flat.get("content"),
        }
    return flat


async def _with_retry(
    fn: Callable[[], Awaitable[Any]],
    label: str,
    cfg: EnvLessBridgeConfig,
) -> Any:
    """Retry an async init call with exponential backoff + jitter.

    Mirrors TS ``withRetry`` on ``remoteBridgeCore.ts:891-913``. Returns
    the first non-``None`` result; ``None`` after all attempts indicates
    permanent failure (caller logs + degrades).
    """
    max_attempts = cfg.init_retry_max_attempts
    for attempt in range(1, max_attempts + 1):
        result = await fn()
        if result is not None:
            return result
        if attempt < max_attempts:
            base_ms = cfg.init_retry_base_delay_ms * (2 ** (attempt - 1))
            jitter_ms = base_ms * cfg.init_retry_jitter_fraction * (2 * random.random() - 1)  # nosec B311
            delay_ms = min(base_ms + jitter_ms, cfg.init_retry_max_delay_ms)
            logger.debug(
                "[remote-bridge] %s failed (attempt %s/%s), retrying in %sms",
                label,
                attempt,
                max_attempts,
                round(delay_ms),
            )
            await asyncio.sleep(delay_ms / 1000.0)
    return None


def _async_refresh_token(
    params: EnvLessBridgeParams,
) -> Callable[[], Awaitable[str | None]]:
    """Build the async token getter for the JWT refresh scheduler.

    Always attempts OAuth refresh first — ``get_access_token()`` returns
    expired tokens as non-None strings (no expiresAt check), so we can't
    rely on truthiness alone. Passes the stale token to ``on_auth_401``
    so keychain-comparison can detect parallel refresh (Phase 10 wiring).
    """

    async def _refresh() -> str | None:
        stale = params.get_access_token()
        if params.on_auth_401 is not None:
            await params.on_auth_401(stale or "")
        return params.get_access_token() or stale

    return _refresh


async def _safe_archive_session(
    session_id: str,
    base_url: str,
    access_token: str | None,
    org_uuid: str,
    timeout_seconds: float,
    http_client: httpx.AsyncClient | None,
) -> None:
    """Best-effort archive used by the pre-flight failure paths.

    Wraps ``_archive_v2_session`` and swallows all exceptions — the
    caller is already about to return ``None`` from init, so any archive
    failure shouldn't prevent that.
    """
    try:
        await _archive_v2_session(
            session_id,
            base_url,
            access_token,
            org_uuid,
            timeout_seconds,
            http_client,
        )
    except Exception as err:  # noqa: BLE001
        logger.debug("[remote-bridge] Pre-flight archive failed: %s", err)


_ARCHIVE_BETA_HEADER = "ccr-byoc-2025-07-29"
_ANTHROPIC_VERSION = "2023-06-01"


async def _archive_v2_session(
    session_id: str,
    base_url: str,
    access_token: str | None,
    org_uuid: str,
    timeout_seconds: float,
    http_client: httpx.AsyncClient | None,
) -> int | str:
    """Archive a v2 session via the compat ``/v1/sessions/{id}/archive`` endpoint.

    Mirrors TS ``archiveSession`` on ``remoteBridgeCore.ts:963-1008``.
    The v2 archive lives at the compat layer (``/v1/sessions/*``, not
    ``/v1/code/sessions/*``) and requires distinct headers vs.
    ``bridge_api.archive_session`` (which targets the environments API):

    * ``anthropic-beta: ccr-byoc-2025-07-29``  (compat byoc)
    * ``x-organization-uuid``                  (required by compat gateway)

    Returns the HTTP status code on success, or ``'no_token'`` /
    ``'timeout'`` / ``'error'`` on failure. Idempotent — 409 (already
    archived) is success.
    """
    if not access_token:
        return "no_token"
    compat_id = to_compat_session_id(session_id)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "anthropic-version": _ANTHROPIC_VERSION,
        "anthropic-beta": _ARCHIVE_BETA_HEADER,
        "x-organization-uuid": org_uuid,
    }
    url = f"{base_url.rstrip('/')}/v1/sessions/{compat_id}/archive"
    try:
        if http_client is not None:
            response = await http_client.post(
                url,
                headers=headers,
                json={},
                timeout=timeout_seconds,
            )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json={},
                    timeout=timeout_seconds,
                )
    except httpx.TimeoutException:
        logger.debug("[remote-bridge] Archive %s timed out", compat_id)
        return "timeout"
    except httpx.HTTPError as err:
        logger.debug("[remote-bridge] Archive %s failed: %s", compat_id, err)
        return "error"
    logger.debug("[remote-bridge] Archive %s status=%s", compat_id, response.status_code)
    return response.status_code
