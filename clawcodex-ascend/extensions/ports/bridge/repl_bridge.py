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

"""Env-based bridge orchestrator — Phase 6 MVP slice.

Ports the **public surface + happy path** of
``typescript/src/bridge/replBridge.ts`` (~2400 lines in TS).

**Scope decision**: A full Phase 6 port (perpetual mode, dual v1/v2
transport, multi-attempt env recreation on 404, crash-recovery pointer
integration, dropped-batch telemetry, deterministic poll-loop backoff,
work-id dedup across stale redeliveries, etc.) is 2-3 weeks per the
refactoring plan. For autonomous porting in one session, this module
implements the structural skeleton + single-session happy path:

* Register environment → create session
* Work-poll loop (basic, v2-only transport)
* Spawn session via Phase 4 ``session_runner``
* ``ReplBridgeHandle`` surface — write_messages / control / teardown
* Teardown — stop_work + archive + deregister

What is **explicitly deferred** (with TODOs at the call sites):

* **v1 transport** (``HybridTransport`` POST writes + WS reads) — v2 is
  the going-forward path; v1 is being deprecated server-side. Module
  raises ``NotImplementedError`` if work secrets indicate v1 only.
* **Perpetual mode** (crash-recovery pointer integration, env reuse via
  ``reuseEnvironmentId``). Caller must set ``perpetual=False``.
* **Env recreation** (the Strategy-1 / Strategy-2 reconnect dance after
  a poll 404). Module logs the 404, fires ``on_state_change('failed')``,
  and exits the poll loop. Phase 8 ``bridgeMain`` is the right place to
  build the full recreation flow.
* **JWT refresh integration with the spawned session** — the
  ``TokenRefreshScheduler`` exists; wiring it to ``session.update_access_token``
  on refresh is left to a follow-up. For now sessions use their initial
  JWT until expiry.
* **Multi-session** — the MVP handles one session at a time; second poll
  result is rejected. Phase 8 ``bridgeMain`` handles the multi-session
  daemon case via spawn-mode dispatch.
* **Backoff/give-up logic** — the poll loop uses a fixed interval from
  the config. The full TS backoff machinery (two-track error counters,
  process-suspension detection, 10-min give-up) lands in Phase 8.
* **Dropped-batch telemetry** + **work-id completion dedup** — both
  log-only enhancements; deferred.

What IS ported in full:

* Public types: ``ReplBridgeHandle``, ``BridgeState``, ``BridgeCoreParams``
* ``init_bridge_core(params, *, http_client?, api_client?, spawner?)`` — the factory
* Single-session lifecycle: register → poll → spawn → done → archive
* Idempotent teardown
* OAuth + env-secret auth via ``bridge_api``

This is sufficient to validate the bridge_api + session_runner + v2
transport integration end-to-end. Phase 8 will fill in the multi-
session + reconnect + perpetual surface.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from clawcodex_ext.bridge.bridge_api import (
    create_bridge_api_client,
)
from clawcodex_ext.bridge.bridge_pointer import (
    BridgePointer,
    clear_pointer,
    read_pointer,
    write_pointer,
)
from clawcodex_ext.bridge.poll_config_defaults import (
    DEFAULT_POLL_CONFIG,
    PollIntervalConfig,
)
from clawcodex_ext.bridge.session_id_compat import (
    to_infra_session_id,
)
from clawcodex_ext.bridge.types import (
    BridgeApiClient,
    BridgeConfig,
    SessionSpawner,
)
from extensions.ports.bridge.session_runner import (
    SessionSpawnerDeps,
    create_session_spawner,
)

from extensions.ports.bridge.repl_bridge_state import _BridgeState, _fire_state

logger = logging.getLogger(__name__)


# ── Public types ──────────────────────────────────────────────────────────


BridgeState = str
"""``'ready'`` | ``'connected'`` | ``'reconnecting'`` | ``'failed'``."""


# Forward references via Any so we don't have to pre-define types in this
# already-busy module. Real Message / SDK types live in their own modules.
OnInboundMessage = Callable[[dict[str, Any]], Any]
OnUserMessage = Callable[[str, str], bool]
OnPermissionResponse = Callable[[dict[str, Any]], None]
OnInterrupt = Callable[[], None]
OnSetModel = Callable[[str | None], None]
OnSetMaxThinkingTokens = Callable[[int | None], None]
OnSetPermissionMode = Callable[[str], Any]
OnStateChange = Callable[..., None]
OnAuth401 = Callable[[str], Awaitable[bool]]
GetAccessToken = Callable[[], str | None]


@dataclass
class BridgeCoreParams:
    """Explicit-param input to ``init_bridge_core``.

    Mirrors TS ``BridgeCoreParams`` on ``replBridge.ts:92-222``. Required
    fields first; everything optional defaults sensibly.
    """

    # Identity
    dir: str
    machine_name: str
    branch: str
    git_repo_url: str | None
    title: str

    # URLs
    base_url: str
    session_ingress_url: str
    worker_type: str

    # Auth
    get_access_token: GetAccessToken

    # Session creation (injected for daemon vs REPL flexibility)
    create_session: Callable[[dict[str, Any]], Awaitable[str | None]]
    """``async def create_session({environment_id, title, gitRepoUrl, branch})
    -> session_id | None``. Daemon/REPL wrappers pass distinct implementations
    that differ in how they build the org-scoped HTTP headers."""

    archive_session: Callable[[str], Awaitable[None]]
    """``async def archive_session(session_id)`` — best-effort archival
    on teardown; MUST NOT throw."""

    # Optional callbacks
    on_auth_401: OnAuth401 | None = None
    on_inbound_message: OnInboundMessage | None = None
    on_user_message: OnUserMessage | None = None
    on_permission_response: OnPermissionResponse | None = None
    on_interrupt: OnInterrupt | None = None
    on_set_model: OnSetModel | None = None
    on_set_max_thinking_tokens: OnSetMaxThinkingTokens | None = None
    on_set_permission_mode: OnSetPermissionMode | None = None
    on_state_change: OnStateChange | None = None

    # Config getters
    get_poll_interval_config: Callable[[], PollIntervalConfig] = lambda: DEFAULT_POLL_CONFIG
    get_current_title: Callable[[], str] | None = None

    # Identity for the env registration
    max_sessions: int = 1
    spawn_mode: str = "single-session"  # 'single-session' | 'same-dir' | 'worktree'
    bridge_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # MVP scope: perpetual mode is not yet supported.
    perpetual: bool = False

    # Initial history (currently unused by the MVP slice — recorded for
    # future Phase 6 work that integrates with remote_bridge_core's
    # flush_gate pattern).
    initial_messages: list[Any] | None = None
    initial_history_cap: int = 200

    # Max attempts to recreate the environment after a poll 404 / expired
    # error. Mirrors TS bridgeMain's 3-attempt envelope on
    # ``replBridge.ts:614-852``. Each attempt: re-register the env, then
    # create a fresh session, then resume polling.
    #
    # NOTE: the MVP slice does not implement environment recreation yet —
    # this field is reserved for the future Phase 6 port and is currently
    # unused by the code below.
    max_env_recreation_attempts: int = 3


@dataclass
class ReplBridgeHandle:
    """Opaque handle returned by ``init_bridge_core``.

    Mirrors TS ``ReplBridgeHandle`` on ``replBridge.ts:71-82``. All
    write methods are sync fire-and-forget; ``teardown`` is async and
    idempotent.
    """

    bridge_session_id: str
    environment_id: str
    session_ingress_url: str
    write_messages: Callable[[list[Any]], None]
    write_sdk_messages: Callable[[list[dict[str, Any]]], None]
    send_control_request: Callable[[dict[str, Any]], None]
    send_control_response: Callable[[dict[str, Any]], None]
    send_cancel_request: Callable[[str], None]
    send_result: Callable[[], None]
    teardown: Callable[[], Awaitable[None]]


# ── init_bridge_core ──────────────────────────────────────────────────────


async def init_bridge_core(
    params: BridgeCoreParams,
    *,
    http_client: httpx.AsyncClient | None = None,
    api_client: BridgeApiClient | None = None,
    spawner: SessionSpawner | None = None,
    runner_version: str = "py-bridge-mvp",
) -> ReplBridgeHandle | None:
    """Set up the env-based bridge: register → create session → start poll loop.

    Returns ``None`` on any pre-flight failure (no OAuth, env registration
    failed, initial session creation failed). The returned handle stays
    alive until ``teardown()`` is called or the (single) session ends.

    Test seams (kw-only):

    * ``http_client``: optional ``httpx.AsyncClient`` for the bridge API.
    * ``api_client``: pre-built ``BridgeApiClient`` (overrides
      ``http_client`` if provided). Tests use this to inject fakes.
    * ``spawner``: pre-built ``SessionSpawner``. Tests use this to skip
      the real subprocess.
    * ``runner_version``: header value for ``x-environment-runner-version``.
    """
    pointer: BridgePointer | None = None
    if params.perpetual:
        pointer = read_pointer(params.dir, machine_name=params.machine_name)

    if api_client is None:
        api_client = create_bridge_api_client(
            base_url=params.base_url,
            get_access_token=params.get_access_token,
            runner_version=runner_version,
            on_auth_401=params.on_auth_401,
            client=http_client,
        )

    # ── 1. Register environment ────────────────────────────────────────
    bridge_config = BridgeConfig(
        dir=params.dir,
        machine_name=params.machine_name,
        branch=params.branch,
        git_repo_url=params.git_repo_url,
        max_sessions=params.max_sessions,
        spawn_mode=_validated_spawn_mode(params.spawn_mode),
        verbose=False,
        sandbox=False,
        bridge_id=params.bridge_id,
        worker_type=params.worker_type,
        environment_id=params.bridge_id,  # client-generated; server may swap
        api_base_url=params.base_url,
        session_ingress_url=params.session_ingress_url,
        reuse_environment_id=pointer.environment_id if pointer else None,
    )
    try:
        registration = await api_client.register_bridge_environment(bridge_config)
    except Exception as err:  # noqa: BLE001 — best-effort, like the other
        # error paths in this module: 429/5xx/network exceptions must not
        # crash the caller, they get the same cleanup + None return.
        logger.error("[bridge:repl] Registration failed: %s", err)
        _fire_state(params.on_state_change, "failed", f"Registration failed: {err}")
        if params.perpetual:
            clear_pointer(params.dir)
        return None
    environment_id = registration["environment_id"]
    environment_secret = registration["environment_secret"]
    logger.debug("[bridge:repl] Registered environment_id=%s", environment_id)
    if pointer is not None and environment_id != pointer.environment_id:
        logger.info(
            "[bridge:repl] Perpetual: server did not resurrect env (pointer=%s, got=%s); creating fresh session",
            pointer.environment_id,
            environment_id,
        )
        clear_pointer(params.dir)
        pointer = None

    # ── 2. Create initial session ──────────────────────────────────────
    session_id: str | None = None
    if pointer is not None and pointer.session_id is not None:
        candidates = [pointer.session_id]
        infra_session_id = to_infra_session_id(pointer.session_id)
        if infra_session_id != pointer.session_id:
            candidates.append(infra_session_id)
        for candidate in candidates:
            try:
                await api_client.reconnect_session(
                    environment_id,
                    candidate,
                )
            except Exception as err:  # noqa: BLE001
                logger.debug(
                    "[bridge:repl] reconnect_session(%s) failed: %s",
                    candidate,
                    err,
                )
                continue
            session_id = pointer.session_id
            logger.debug(
                "[bridge:repl] Reconnected pointer session_id=%s",
                session_id,
            )
            break
        if session_id is None:
            logger.info(
                "[bridge:repl] Pointer session no longer reachable; creating fresh session",
            )
            clear_pointer(params.dir)
            pointer = None
    if session_id is None:
        try:
            session_id = await params.create_session(
                {
                    "environment_id": environment_id,
                    "title": params.title,
                    "gitRepoUrl": params.git_repo_url,
                    "branch": params.branch,
                }
            )
        except Exception as err:  # noqa: BLE001
            logger.error("[bridge:repl] Session creation threw: %s", err)
            session_id = None
    if session_id is None:
        _fire_state(params.on_state_change, "failed", "Session creation failed")
        if params.perpetual:
            clear_pointer(params.dir)
        try:
            await api_client.deregister_environment(environment_id)
        except Exception as err:  # noqa: BLE001
            logger.debug("[bridge:repl] Deregister-after-create-fail failed: %s", err)
        return None
    logger.debug("[bridge:repl] Created session_id=%s", session_id)
    if params.perpetual:
        write_pointer(
            params.dir,
            bridge_id=params.bridge_id,
            environment_id=environment_id,
            session_id=session_id,
            machine_name=params.machine_name,
            created_at_ms=pointer.created_at_ms if pointer else None,
        )

    # ── 3. Build the spawner (if not test-injected) ────────────────────
    if spawner is None:
        spawner = create_session_spawner(
            SessionSpawnerDeps(
                exec_path="claude",  # caller-overridable in future
                verbose=False,
                sandbox=False,
            )
        )

    # ── 4. State machine + poll loop ──────────────────────────────────
    state = _BridgeState(
        params=params,
        api=api_client,
        spawner=spawner,
        environment_id=environment_id,
        environment_secret=environment_secret,
        initial_session_id=session_id,
        bridge_config=bridge_config,
        pointer_created_at_ms=pointer.created_at_ms if pointer else None,
    )
    state.start_poll_loop()
    pointer_mtime_task: asyncio.Task[None] | None = None
    if params.perpetual:
        pointer_mtime_task = asyncio.create_task(
            state._pointer_mtime_refresh_loop(),
            name="bridge-pointer-mtime-refresh",
        )

    _fire_state(params.on_state_change, "ready")

    async def _teardown() -> None:
        # Cancel the pointer mtime refresh loop before tearing down the
        # state machine, so it cannot re-write a stale pointer mid-teardown.
        if pointer_mtime_task is not None:
            pointer_mtime_task.cancel()
            try:
                await pointer_mtime_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await state.teardown()

    return ReplBridgeHandle(
        bridge_session_id=session_id,
        environment_id=environment_id,
        session_ingress_url=params.session_ingress_url,
        write_messages=state.write_messages,
        write_sdk_messages=state.write_sdk_messages,
        send_control_request=state.send_control_request,
        send_control_response=state.send_control_response,
        send_cancel_request=state.send_cancel_request,
        send_result=state.send_result,
        teardown=_teardown,
    )


# ── Internal state machine ────────────────────────────────────────────────


def _validated_spawn_mode(mode: str) -> Any:
    """Cast a user-supplied spawn-mode string to the Literal type."""
    if mode not in ("single-session", "worktree", "same-dir"):
        raise ValueError(f"Invalid spawn_mode: {mode!r}")
    return mode


__all__ = [
    "BridgeCoreParams",
    "BridgeState",
    "ReplBridgeHandle",
    "init_bridge_core",
]
