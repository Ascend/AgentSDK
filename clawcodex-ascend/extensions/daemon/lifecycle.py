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
"""Worker lifecycle helpers — spawn / restart / graceful shutdown.

This module contains the coroutines the supervisor calls to:

* Spawn a worker subprocess (with ``CLAWCODEX_DAEMON_*`` env injection).
* React to a worker exit — classify it (ok / permanent / transient),
  apply exponential backoff, detect rapid-failure bursts, and
  eventually restart.
* Shut down every worker gracefully — ``SIGTERM`` first, wait up
  to ``timeout_ms``, then ``SIGKILL`` stragglers.

The actual subprocess is launched with
``asyncio.create_subprocess_exec`` so the supervisor loop can wait
on multiple workers concurrently and propagate cancellation cleanly.
The worker entry point is ``extensions.daemon.worker_main`` (run as
``python -m extensions.daemon.worker_main <kind>``) — see
:mod:`extensions.daemon.worker_main`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


from extensions.daemon.constants import (
    BACKOFF_CAP_MS,
    BACKOFF_INITIAL_MS,
    BACKOFF_MULTIPLIER,
    ENV_VAR_DAEMON_CAPACITY,
    ENV_VAR_DAEMON_DIR,
    ENV_VAR_DAEMON_NAME,
    ENV_VAR_DAEMON_PERMISSION_MODE,
    ENV_VAR_DAEMON_SANDBOX,
    ENV_VAR_DAEMON_SESSION_KIND,
    ENV_VAR_DAEMON_SPAWN_MODE,
    ENV_VAR_DAEMON_TIMEOUT_MS,
    ENV_VAR_SUPERVISOR_PID,
    EXIT_CODE_OK,
    EXIT_CODE_PERMANENT,
    GRACEFUL_SHUTDOWN_TIMEOUT_MS,
    MAX_RAPID_FAILURES,
    RAPID_FAILURE_WINDOW_MS,
)
from extensions.daemon.errors import WorkerSpawnError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker runtime state (in-memory only — not persisted).
# ---------------------------------------------------------------------------


@dataclass
class WorkerRuntime:
    """Mutable per-worker runtime state held by the supervisor.

    ``process`` and ``restart_timer`` are intentionally typed as
    ``Optional`` so the dataclass can be constructed before the first
    spawn. The supervisor updates them in place.
    """

    kind: str
    failure_count: int = 0
    backoff_ms: int = BACKOFF_INITIAL_MS
    backoff_initial_ms: int = BACKOFF_INITIAL_MS
    backoff_cap_ms: int = BACKOFF_CAP_MS
    parked: bool = False
    total_restarts: int = 0
    last_exit_code: int | None = None
    last_start_monotonic: float = 0.0
    process: asyncio.subprocess.Process | None = None
    restart_timer: asyncio.Task | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    def reset_backoff(self) -> None:
        """Reset backoff after a clean run (lived longer than the
        rapid-failure window).
        """
        self.failure_count = 0
        self.backoff_ms = self.backoff_initial_ms


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------


def build_worker_env(
    *,
    supervisor_pid: int,
    name: str,
    dir_: Path,
    spawn_mode: str,
    capacity: int,
    permission_mode: str | None,
    sandbox: bool,
    timeout_ms: int,
) -> dict[str, str]:
    """Compose the ``CLAWCODEX_DAEMON_*`` environment for a worker.

    Inherits from ``os.environ`` so the worker sees the supervisor's
    ``PATH``, locale, and provider credentials.
    """
    env = dict(os.environ)
    env.update(
        {
            ENV_VAR_SUPERVISOR_PID: str(supervisor_pid),
            ENV_VAR_DAEMON_NAME: name,
            ENV_VAR_DAEMON_DIR: str(dir_),
            ENV_VAR_DAEMON_SPAWN_MODE: spawn_mode,
            ENV_VAR_DAEMON_CAPACITY: str(capacity),
            ENV_VAR_DAEMON_PERMISSION_MODE: permission_mode or "",
            ENV_VAR_DAEMON_SANDBOX: "1" if sandbox else "0",
            ENV_VAR_DAEMON_TIMEOUT_MS: str(timeout_ms),
            ENV_VAR_DAEMON_SESSION_KIND: "daemon-worker",
        }
    )
    return env


# ---------------------------------------------------------------------------
# Subprocess IO pump
# ---------------------------------------------------------------------------


async def _pump_stream(
    stream: asyncio.StreamReader | None,
    kind: str,
    level: int,
) -> None:
    """Forward subprocess *stream* lines into the supervisor logger."""
    if stream is None:
        return
    while True:
        try:
            line = await stream.readline()
        except ValueError:
            return  # stream closed
        if not line:
            return
        try:
            text = line.decode("utf-8", errors="replace").rstrip()
        except Exception:
            text = repr(line)
        logger.log(level, "[worker:%s] %s", kind, text)


# ---------------------------------------------------------------------------
# Spawn + restart
# ---------------------------------------------------------------------------


def _worker_argv(kind: str) -> list[str]:
    """Build the argv that runs the worker entry point."""
    return [sys.executable, "-m", "extensions.daemon.worker_main", kind]


async def spawn_worker(
    runtime: WorkerRuntime,
    *,
    supervisor_pid: int,
    name: str,
    dir_: Path,
    spawn_mode: str,
    capacity: int,
    permission_mode: str | None,
    sandbox: bool,
    timeout_ms: int,
    stop_event: asyncio.Event,
) -> None:
    """Spawn the worker subprocess and wait for it to exit.

    On exit, schedule a restart (with backoff) unless the supervisor
    is shutting down (``stop_event`` is set) or the worker is parked.
    """
    if stop_event.is_set() or runtime.parked:
        return
    if runtime.process is not None and runtime.process.returncode is None:
        # Already running — don't double-spawn.
        return

    env = build_worker_env(
        supervisor_pid=supervisor_pid,
        name=name,
        dir_=dir_,
        spawn_mode=spawn_mode,
        capacity=capacity,
        permission_mode=permission_mode,
        sandbox=sandbox,
        timeout_ms=timeout_ms,
    )
    argv = _worker_argv(runtime.kind)

    runtime.last_start_monotonic = time.monotonic()
    logger.info("[supervisor] spawning worker '%s'", runtime.kind)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(dir_),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise WorkerSpawnError(f"interpreter not found: {argv[0]!r} ({exc})") from exc
    except PermissionError as exc:
        raise WorkerSpawnError(f"interpreter not executable: {argv[0]!r} ({exc})") from exc
    except OSError as exc:
        raise WorkerSpawnError(f"failed to spawn {runtime.kind}: {exc}") from exc

    runtime.process = proc

    # Pump streams in the background; we don't await them here.
    asyncio.create_task(_pump_stream(proc.stdout, runtime.kind, logging.INFO))
    asyncio.create_task(_pump_stream(proc.stderr, runtime.kind, logging.ERROR))

    try:
        try:
            exit_code = await proc.wait()
        except asyncio.CancelledError:
            # Supervisor cancellation — terminate, wait for exit, then propagate.
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                pass
            raise
    finally:
        runtime.process = None
    runtime.last_exit_code = exit_code
    await _on_worker_exit(
        runtime,
        exit_code,
        stop_event=stop_event,
        supervisor_pid=supervisor_pid,
        name=name,
        dir_=dir_,
        spawn_mode=spawn_mode,
        capacity=capacity,
        permission_mode=permission_mode,
        sandbox=sandbox,
        timeout_ms=timeout_ms,
    )


async def _on_worker_exit(
    runtime: WorkerRuntime,
    exit_code: int,
    *,
    stop_event: asyncio.Event,
    supervisor_pid: int,
    name: str,
    dir_: Path,
    spawn_mode: str,
    capacity: int,
    permission_mode: str | None,
    sandbox: bool,
    timeout_ms: int,
) -> None:
    """Decide whether to restart, park, or stop after a worker exit.

    Decision table:

    * Permanent (78) → park.
    * Clean exit (0) → stop (no auto-restart under MVP).
    * Rapid burst (≥ MAX_RAPID_FAILURES inside the window) → park.
    * Otherwise → exponential backoff restart.
    """
    if stop_event.is_set():
        logger.info(
            "[supervisor] worker '%s' exited (code=%d) during shutdown",
            runtime.kind,
            exit_code,
        )
        return

    if exit_code == EXIT_CODE_PERMANENT:
        logger.error(
            "[supervisor] worker '%s' permanent failure (exit=%d) — parking",
            runtime.kind,
            exit_code,
        )
        runtime.parked = True
        return

    if exit_code == EXIT_CODE_OK:
        logger.info(
            "[supervisor] worker '%s' exited cleanly — leaving stopped",
            runtime.kind,
        )
        return

    # Transient: classify by run duration.
    run_duration_ms = (time.monotonic() - runtime.last_start_monotonic) * 1000.0
    if run_duration_ms < RAPID_FAILURE_WINDOW_MS:
        runtime.failure_count += 1
        if runtime.failure_count >= MAX_RAPID_FAILURES:
            logger.error(
                "[supervisor] worker '%s' failed %d times in %d ms — parking",
                runtime.kind,
                runtime.failure_count,
                RAPID_FAILURE_WINDOW_MS,
            )
            runtime.parked = True
            return
    else:
        # Worker lived long enough to count as healthy.
        runtime.reset_backoff()

    delay_ms = runtime.backoff_ms
    logger.info(
        "[supervisor] worker '%s' exited (code=%d); restarting in %d ms",
        runtime.kind,
        exit_code,
        delay_ms,
    )

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=delay_ms / 1000.0)
        return  # stop_event was set during the backoff sleep
    except asyncio.TimeoutError:
        pass

    if stop_event.is_set() or runtime.parked:
        return

    runtime.total_restarts += 1
    runtime.backoff_ms = min(
        runtime.backoff_ms * BACKOFF_MULTIPLIER,
        runtime.backoff_cap_ms,
    )

    try:
        await spawn_worker(
            runtime,
            supervisor_pid=supervisor_pid,
            name=name,
            dir_=dir_,
            spawn_mode=spawn_mode,
            capacity=capacity,
            permission_mode=permission_mode,
            sandbox=sandbox,
            timeout_ms=timeout_ms,
            stop_event=stop_event,
        )
    except WorkerSpawnError as exc:
        logger.error(
            "[supervisor] worker '%s' spawn failed: %s — parking",
            runtime.kind,
            exc,
        )
        runtime.parked = True


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


async def graceful_shutdown(
    runtimes: list[WorkerRuntime],
    *,
    timeout_ms: int = GRACEFUL_SHUTDOWN_TIMEOUT_MS,
) -> None:
    """Terminate every worker, escalate to ``SIGKILL`` on timeout.

    ``proc.terminate()`` maps to ``SIGTERM`` on POSIX and ``CTRL_BREAK``
    on Windows. ``proc.kill()`` is the hard-stop fallback.
    """
    live = [r for r in runtimes if r.process is not None and r.process.returncode is None]
    if not live:
        return

    # Phase 1 — polite termination.
    for r in live:
        proc = r.process
        assert proc is not None
        logger.info(
            "[supervisor] sending SIGTERM to worker '%s' (pid=%s)",
            r.kind,
            proc.pid,
        )
        try:
            proc.terminate()
        except ProcessLookupError:
            pass

    # Phase 2 — wait for graceful exit.
    pending: list[asyncio.Task] = []
    for r in live:
        proc = r.process
        assert proc is not None
        pending.append(asyncio.create_task(_wait_process(proc, r)))

    if pending:
        done, still = await asyncio.wait(
            pending,
            timeout=timeout_ms / 1000.0,
            return_when=asyncio.ALL_COMPLETED,
        )
    else:
        still = []

    # Phase 3 — SIGKILL anything still running.
    for task in still:
        r: WorkerRuntime = task._worker  # type: ignore[attr-defined]
        proc = r.process
        if proc is not None and proc.returncode is None:
            logger.warning(
                "[supervisor] worker '%s' did not exit gracefully — SIGKILL",
                r.kind,
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:  # nosec
                pass

    # Drain the completed tasks.
    for t in pending:
        try:
            await t
        except Exception:  # nosec
            pass


async def _wait_process(proc: asyncio.subprocess.Process, runtime: WorkerRuntime) -> None:
    """Wrapper that pins the runtime onto the task for post-timeout lookup."""
    # Stash the runtime on the task BEFORE awaiting, so graceful_shutdown
    # can find it even after ``asyncio.wait`` returns the incomplete set.
    asyncio.current_task()._worker = runtime  # type: ignore[attr-defined]
    await proc.wait()


__all__ = [
    "WorkerRuntime",
    "build_worker_env",
    "spawn_worker",
    "graceful_shutdown",
]
