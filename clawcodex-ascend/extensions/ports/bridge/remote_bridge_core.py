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

"""Env-less Remote Control bridge core — Phase 5 MVP.

Ports ``typescript/src/bridge/remoteBridgeCore.ts``.

"Env-less" means no Environments API layer — this connects directly to
the session-ingress (CCR v2) layer:

  1. POST ``/v1/code/sessions``              → ``cse_*`` session id
  2. POST ``/v1/code/sessions/{id}/bridge``  → worker JWT + epoch + TTL
  3. ``create_v2_repl_transport``            → SSE reads + CCRClient writes
  4. ``TokenRefreshScheduler``               → proactive ``/bridge`` re-call before expiry
  5. 401 on SSE                              → re-fetch ``/bridge`` and rebuild transport

No register/poll/ack/stop/heartbeat/deregister environment lifecycle.
Each ``/bridge`` call bumps ``worker_epoch`` server-side, so any refresh
path must rebuild the transport (a JWT-only swap leaves the old
CCRClient heartbeating with a stale epoch → 409 within 20s).

**MVP scope** (this Phase 5 port intentionally defers a few things):

* **No connect-timeout telemetry** — TS arms a ``setTimeout`` that
  emits ``tengu_bridge_repl_connect_timeout`` if neither onConnect nor
  onClose fires before ``cfg.connect_timeout_ms``. The Python port logs
  the deadline as a debug warning instead; analytics wiring lands in
  Phase 10.
* **No CCR mirror-mode telemetry** — the ``CCR_MIRROR`` feature flag
  branches on telemetry event names; we route everything through
  ``tengu_bridge_repl_*`` for now.
* **Trusted-device token** — passes ``None`` (no Phase 10 keychain).
* **``ConnectCause`` enum** — kept as a string for log clarity but not
  used for telemetry discriminator (no analytics in this build).

What IS ported in full:

* OAuth → ``/code/sessions`` → ``/bridge`` init with retry+jitter
* v2 transport build (SSE + CCRClient via existing factory)
* FlushGate + dual-set UUID dedup (echo + re-delivery)
* Proactive JWT refresh via ``TokenRefreshScheduler`` (epoch-bumping
  rebuild on fire)
* 401 SSE recovery (token refresh + rebuild)
* ``ReplBridgeHandle``-style return surface: ``write_messages``,
  ``write_sdk_messages``, ``send_control_request``,
  ``send_control_response``, ``send_cancel_request``, ``send_result``,
  ``teardown``
* Idempotent teardown: cancel scheduler → drop gate → reportState idle →
  write result → archive (with 401 retry) → close
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from clawcodex_ext.bridge.code_session_api import (
    create_code_session,
    fetch_remote_credentials,
)
from clawcodex_ext.bridge.env_less_bridge_config import (
    DEFAULT_ENV_LESS_BRIDGE_CONFIG,
    EnvLessBridgeConfig,
    get_env_less_bridge_config,
)
from clawcodex_ext.bridge.jwt_utils import TokenRefreshScheduler
from clawcodex_ext.bridge.repl_bridge_transport import (
    ReplBridgeTransport,
    V2TransportOptions,
    create_v2_repl_transport,
)
from clawcodex_ext.bridge.work_secret import build_ccr_v2_sdk_url
from clawcodex_ext.types.messages import Message

from extensions.ports.bridge.remote_bridge_core_state import (
    _BridgeState,
    _async_refresh_token,
    _fire_state,
    _freeze_token,
    _safe_archive_session,
    _with_retry,
)

logger = logging.getLogger(__name__)


_TEARDOWN_RESULT_WRITE_TIMEOUT_SECONDS = 0.5
"""Cap on the teardown result-write enqueue wait. ``gracefulShutdown``
races teardown against a 2s budget; 0.5s leaves headroom for archive +
close while still containing back-pressure stalls."""


# ── Public types ──────────────────────────────────────────────────────────


BridgeState = str
"""Lifecycle states emitted via ``on_state_change``: ``'ready'``,
``'connected'``, ``'reconnecting'``, ``'failed'``. Kept as ``str`` to
stay compatible with Phase 6+ orchestrators that share the same union.
"""


OnInboundMessage = Callable[[dict[str, Any]], Any]
OnUserMessage = Callable[[str, str], bool]
OnPermissionResponse = Callable[[dict[str, Any]], None]
OnInterrupt = Callable[[], None]
OnSetModel = Callable[[str | None], None]
OnSetMaxThinkingTokens = Callable[[int | None], None]
OnSetPermissionMode = Callable[[str], Any]
OnStateChange = Callable[..., None]
"""``on_state_change(state, detail=None)`` — kept as ``...`` so call
sites can omit detail."""

OnAuth401 = Callable[[str], Awaitable[bool]]
GetAccessToken = Callable[[], str | None]


@dataclass
class EnvLessBridgeParams:
    """Configuration for ``init_env_less_bridge_core``.

    Mirrors TS ``EnvLessBridgeParams`` on ``remoteBridgeCore.ts:89-131``.
    Required: ``base_url``, ``org_uuid``, ``title``, ``get_access_token``,
    ``initial_history_cap``. All callbacks are optional.
    """

    base_url: str
    org_uuid: str
    title: str
    get_access_token: GetAccessToken
    initial_history_cap: int
    initial_messages: list[Message] | None = None
    on_auth_401: OnAuth401 | None = None
    on_inbound_message: OnInboundMessage | None = None
    on_user_message: OnUserMessage | None = None
    on_permission_response: OnPermissionResponse | None = None
    on_interrupt: OnInterrupt | None = None
    on_set_model: OnSetModel | None = None
    on_set_max_thinking_tokens: OnSetMaxThinkingTokens | None = None
    on_set_permission_mode: OnSetPermissionMode | None = None
    on_state_change: OnStateChange | None = None
    outbound_only: bool = False
    tags: list[str] | None = None


@dataclass
class RemoteBridgeHandle:
    """Opaque handle returned by ``init_env_less_bridge_core``.

    Mirrors the consumer-facing surface of TS ``ReplBridgeHandle``
    (``remoteBridgeCore.ts:763-886``). All write methods are sync
    fire-and-forget — the underlying transport batches writes via
    ``SerialBatchEventUploader``. ``teardown`` is async and idempotent.
    """

    bridge_session_id: str
    environment_id: str  # always empty for env-less
    session_ingress_url: str
    write_messages: Callable[[list[Message]], None]
    write_sdk_messages: Callable[[list[dict[str, Any]]], None]
    send_control_request: Callable[[dict[str, Any]], None]
    send_control_response: Callable[[dict[str, Any]], None]
    send_cancel_request: Callable[[str], None]
    send_result: Callable[[], None]
    teardown: Callable[[], Awaitable[None]]


# ── Init ──────────────────────────────────────────────────────────────────


async def init_env_less_bridge_core(
    params: EnvLessBridgeParams,
    *,
    http_client: httpx.AsyncClient | None = None,
    config: EnvLessBridgeConfig | None = None,
    transport_factory: Callable[[V2TransportOptions], Awaitable[ReplBridgeTransport]] | None = None,
) -> RemoteBridgeHandle | None:
    """Create a session, fetch a worker JWT, connect the v2 transport.

    Returns ``None`` on any pre-flight failure (session create failed,
    ``/bridge`` failed, transport setup failed). Caller surfaces this as
    a generic "initialization failed" state.

    Test seams (kw-only — production callers omit):

    * ``http_client``: optional injected ``httpx.AsyncClient`` for the
      ``/code/sessions``, ``/bridge``, and ``archive`` calls.
    * ``config``: override ``EnvLessBridgeConfig`` (otherwise fetched
      via ``get_env_less_bridge_config()`` which currently returns
      defaults).
    * ``transport_factory``: override the v2 transport constructor for
      tests so they can inject a fake without hitting the SSE/CCR layer.
    """
    cfg = config if config is not None else await get_env_less_bridge_config()
    factory = transport_factory or create_v2_repl_transport

    # ── 1. OAuth pre-check ─────────────────────────────────────────────
    access_token = params.get_access_token()
    if not access_token:
        logger.debug("[remote-bridge] No OAuth token")
        _fire_state(params.on_state_change, "failed", "No OAuth token — see debug log")
        return None

    # ── 2. Create session (POST /v1/code/sessions) ─────────────────────
    timeout_seconds = cfg.http_timeout_ms / 1000.0
    session_id = await _with_retry(
        lambda: create_code_session(
            params.base_url,
            access_token,
            params.title,
            timeout_seconds=timeout_seconds,
            tags=params.tags,
            client=http_client,
        ),
        "createCodeSession",
        cfg,
    )
    if session_id is None:
        _fire_state(params.on_state_change, "failed", "Session creation failed — see debug log")
        return None
    logger.debug("[remote-bridge] Created session %s", session_id)

    # ── 3. Fetch bridge credentials ────────────────────────────────────
    credentials = await _with_retry(
        lambda: fetch_remote_credentials(
            session_id,
            params.base_url,
            access_token,
            timeout_seconds=timeout_seconds,
            client=http_client,
        ),
        "fetchRemoteCredentials",
        cfg,
    )
    if credentials is None:
        _fire_state(params.on_state_change, "failed", "Remote credentials fetch failed — see debug log")
        await _safe_archive_session(
            session_id,
            params.base_url,
            access_token,
            params.org_uuid,
            timeout_seconds,
            http_client,
        )
        return None
    logger.debug(
        "[remote-bridge] Fetched bridge credentials (expires_in=%ss)",
        credentials.expires_in,
    )

    # ── 4. Build v2 transport ──────────────────────────────────────────
    session_url = build_ccr_v2_sdk_url(credentials.api_base_url, session_id)
    try:
        transport = await factory(
            V2TransportOptions(
                session_url=session_url,
                ingress_token=credentials.worker_jwt,
                session_id=session_id,
                epoch=credentials.worker_epoch,
                heartbeat_interval_seconds=cfg.heartbeat_interval_ms / 1000.0,
                heartbeat_jitter_fraction=cfg.heartbeat_jitter_fraction,
                outbound_only=params.outbound_only,
                # Per-instance closure — keeps the worker JWT out of
                # ``CLAUDE_CODE_SESSION_ACCESS_TOKEN`` env which mcp/client
                # would otherwise leak to user-configured MCP servers.
                # Frozen at construction: transport is fully rebuilt on
                # refresh (rebuild_transport below) with a fresh closure.
                get_auth_token=_freeze_token(credentials.worker_jwt),
            )
        )
    except Exception as err:  # noqa: BLE001  surface as pre-flight failure
        logger.error("[remote-bridge] v2 transport setup failed: %s", err)
        _fire_state(params.on_state_change, "failed", f"Transport setup failed: {err}")
        await _safe_archive_session(
            session_id,
            params.base_url,
            access_token,
            params.org_uuid,
            timeout_seconds,
            http_client,
        )
        return None
    logger.debug(
        "[remote-bridge] v2 transport created (epoch=%s)",
        credentials.worker_epoch,
    )
    _fire_state(params.on_state_change, "ready")

    # ── 5. State (closures shared by all callbacks) ────────────────────
    state = _BridgeState(
        params=params,
        cfg=cfg,
        session_id=session_id,
        credentials=credentials,
        transport=transport,
        http_client=http_client,
        transport_factory=factory,
    )

    # ── 6. JWT refresh scheduler ───────────────────────────────────────
    refresh = TokenRefreshScheduler(
        get_access_token=_async_refresh_token(params),
        on_refresh=state.on_jwt_refresh,
        label="remote",
        refresh_buffer_ms=cfg.token_refresh_buffer_ms,
    )
    state.refresh = refresh
    refresh.schedule_from_expires_in(session_id, credentials.expires_in)

    # ── 7. Wire transport callbacks ────────────────────────────────────
    state.wire_transport_callbacks()

    # Start the flushGate BEFORE connect *unconditionally* so any
    # write_messages() / send_* calls that arrive during the handshake
    # are queued instead of dropped. The Python ``CCRClient.write_event``
    # silently drops messages while ``_initialized is False`` (unlike
    # TS's ``SerialBatchEventUploader`` which queues), so any pre-onConnect
    # write would be lost without this gate. The gate is drained in
    # ``_on_connect`` once the new transport's CCR is initialized.
    state.flush_gate.start()
    transport.connect()

    # ── 8. Return handle ───────────────────────────────────────────────
    return RemoteBridgeHandle(
        bridge_session_id=session_id,
        environment_id="",
        session_ingress_url=credentials.api_base_url,
        write_messages=state.write_messages,
        write_sdk_messages=state.write_sdk_messages,
        send_control_request=state.send_control_request,
        send_control_response=state.send_control_response,
        send_cancel_request=state.send_cancel_request,
        send_result=state.send_result,
        teardown=state.teardown,
    )


# ── Internal state machine (one instance per bridge) ──────────────────────
__all__ = [
    "BridgeState",
    "DEFAULT_ENV_LESS_BRIDGE_CONFIG",
    "EnvLessBridgeParams",
    "RemoteBridgeHandle",
    "init_env_less_bridge_core",
]
