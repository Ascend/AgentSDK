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

"""Multi-session bridge daemon CLI entry point.

Parses CLI args and drives the daemon loop implemented in
:mod:`extensions.ports.bridge.bridge_daemon`.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import uuid
from collections.abc import Callable

from clawcodex_ext.bridge.bridge_api import (
    BridgeFatalError,
    create_bridge_api_client,
)
from clawcodex_ext.bridge.types import BridgeApiClient, BridgeConfig, SessionSpawner

from extensions.ports.bridge.bridge_daemon import (
    BackoffConfig,
    BridgeHeadlessPermanentError,
    DEFAULT_BACKOFF,
    ParsedArgs,
    is_connection_error,
    is_server_error,
    parse_args,
    run_bridge_loop,
)
from extensions.ports.bridge.session_runner import (
    SessionSpawnerDeps,
    create_session_spawner,
)

logger = logging.getLogger(__name__)

# ── End-to-end entry point ──────────────────────────────────────────────


async def bridge_main(
    args: list[str],
    *,
    api: BridgeApiClient | None = None,
    spawner: SessionSpawner | None = None,
    get_access_token: Callable[[], str | None] = lambda: "tok-placeholder",
    runner_version: str = "py-bridge-mvp",
    base_url: str = "https://api.anthropic.com",
    machine_name: str = "localhost",
    branch: str = "main",
    git_repo_url: str | None = None,
    working_dir: str = ".",
    cancel_event: asyncio.Event | None = None,
) -> int:
    """End-to-end daemon entry: parse → register → run loop → shutdown.

    Returns a process exit code: 0 = clean shutdown, 1 = parse error /
    help, 2 = registration failed, 3 = permanent runtime error.

    Test seams:

    * ``api`` / ``spawner``: pre-built for tests.
    * ``get_access_token``: OAuth token getter.
    * ``cancel_event``: optional ``asyncio.Event`` so tests can ask the
      daemon to shut down without sending a real signal.
    """
    parsed = parse_args(args)
    if parsed.error is not None:
        logger.error("[bridge:main] %s", parsed.error)
        return 1
    if parsed.help:
        _print_usage()
        return 0

    spawn_mode = parsed.spawn_mode or "single-session"
    capacity = parsed.capacity or (1 if spawn_mode == "single-session" else 4)

    bridge_config = BridgeConfig(
        dir=working_dir,
        machine_name=machine_name,
        branch=branch,
        git_repo_url=git_repo_url,
        max_sessions=capacity,
        spawn_mode=spawn_mode,
        verbose=parsed.verbose,
        sandbox=parsed.sandbox,
        bridge_id=str(uuid.uuid4()),
        worker_type="claude_code",
        environment_id="",  # filled by registration
        api_base_url=base_url,
        session_ingress_url=base_url,
        debug_file=parsed.debug_file,
        session_timeout_ms=parsed.session_timeout_ms,
    )

    if api is None:
        api = create_bridge_api_client(
            base_url=base_url,
            get_access_token=get_access_token,
            runner_version=runner_version,
        )

    try:
        registration = await api.register_bridge_environment(bridge_config)
    except BridgeFatalError as err:
        logger.error("[bridge:main] Registration failed: %s", err)
        return 2
    environment_id = registration.get("environment_id")
    environment_secret = registration.get("environment_secret")
    if environment_id is None or environment_secret is None:
        logger.error(
            "[bridge:main] Registration response missing environment_id/environment_secret: %s",
            registration,
        )
        return 2
    logger.info(
        "[bridge:main] Registered environment_id=%s capacity=%s mode=%s",
        environment_id,
        capacity,
        spawn_mode,
    )

    if spawner is None:
        spawner = create_session_spawner(
            SessionSpawnerDeps(
                exec_path="claude",
                verbose=parsed.verbose,
                sandbox=parsed.sandbox,
                debug_file=parsed.debug_file,
                permission_mode=parsed.permission_mode,
            )
        )

    if cancel_event is None:
        cancel_event = asyncio.Event()
        _install_signal_handlers(cancel_event)

    try:
        await run_bridge_loop(
            bridge_config,
            environment_id,
            environment_secret,
            api,
            spawner,
            cancel_event,
        )
    except BridgeHeadlessPermanentError as err:
        logger.error("[bridge:main] Permanent error: %s", err)
        return 3
    return 0


def _install_signal_handlers(cancel_event: asyncio.Event) -> None:
    """Register SIGINT/SIGTERM handlers that set ``cancel_event``.

    No-op on platforms where ``loop.add_signal_handler`` isn't available
    (notably Windows). The MVP tolerates that by relying on the test
    seam ``cancel_event`` instead.
    """
    import sys

    if sys.platform == "win32":
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, cancel_event.set)
        except (NotImplementedError, RuntimeError):
            pass


def _print_usage() -> None:
    """Print a minimal usage banner. Mirrors TS help text shape."""
    usage = """Usage: claude remote-control [options]

Options:
  --verbose, -v               Enable verbose logging
  --sandbox                   Run children in sandbox
  --no-sandbox                Disable sandbox (default)
  --debug-file PATH           Write per-session debug log
  --session-timeout SECONDS   Per-session timeout (parsed but not yet enforced)
  --permission-mode MODE      Default permission mode for children
  --name NAME                 Friendly name for the registered environment
  --spawn {session,same-dir,worktree}
                              Spawn mode (worktree mode logs a warning)
  --capacity N                Max concurrent sessions (default 1 or 4)
  --create-session-in-dir     Override default session-in-dir behavior
  --no-create-session-in-dir  Disable session-in-dir behavior
  --help, -h                  Show this help

Note: --session-id / --continue (perpetual mode) are not yet supported.
"""
    print(usage)


__all__ = [
    "BackoffConfig",
    "BridgeHeadlessPermanentError",
    "DEFAULT_BACKOFF",
    "ParsedArgs",
    "bridge_main",
    "is_connection_error",
    "is_server_error",
    "parse_args",
    "run_bridge_loop",
]
