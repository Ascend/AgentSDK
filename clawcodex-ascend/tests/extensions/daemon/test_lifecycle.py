#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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
#

"""Tests for ``extensions.daemon.lifecycle`` — spawn / restart / shutdown."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from extensions.daemon.constants import (
    BACKOFF_CAP_MS,
    BACKOFF_INITIAL_MS,
    ENV_VAR_DAEMON_SESSION_KIND,
)
from extensions.daemon.errors import WorkerSpawnError
from extensions.daemon.lifecycle import (
    WorkerRuntime,
    build_worker_env,
    graceful_shutdown,
    spawn_worker,
)


# ---------------------------------------------------------------------------
# Environment composition
# ---------------------------------------------------------------------------


def test_build_worker_env_includes_daemon_keys(tmp_path):
    env = build_worker_env(
        supervisor_pid=999,
        name="alpha",
        dir_=tmp_path,
        spawn_mode="worktree",
        capacity=8,
        permission_mode="bypassPermissions",
        sandbox=True,
        timeout_ms=12345,
    )
    assert env["CLAWCODEX_SUPERVISOR_PID"] == "999"
    assert env["CLAWCODEX_DAEMON_NAME"] == "alpha"
    assert env["CLAWCODEX_DAEMON_DIR"] == str(tmp_path)
    assert env["CLAWCODEX_DAEMON_SPAWN_MODE"] == "worktree"
    assert env["CLAWCODEX_DAEMON_CAPACITY"] == "8"
    assert env["CLAWCODEX_DAEMON_PERMISSION_MODE"] == "bypassPermissions"
    assert env["CLAWCODEX_DAEMON_SANDBOX"] == "1"
    assert env["CLAWCODEX_DAEMON_TIMEOUT_MS"] == "12345"
    assert env[ENV_VAR_DAEMON_SESSION_KIND] == "daemon-worker"
    # Inherited PATH
    assert "PATH" in env


def test_build_worker_env_sandbox_false_uses_zero():
    env = build_worker_env(
        supervisor_pid=1,
        name="x",
        dir_=Path("/tmp"),
        spawn_mode="same-dir",
        capacity=4,
        permission_mode=None,
        sandbox=False,
        timeout_ms=30000,
    )
    assert env["CLAWCODEX_DAEMON_SANDBOX"] == "0"
    assert env["CLAWCODEX_DAEMON_PERMISSION_MODE"] == ""


# ---------------------------------------------------------------------------
# WorkerRuntime basics
# ---------------------------------------------------------------------------


def test_worker_runtime_defaults():
    rt = WorkerRuntime(kind="x")
    assert rt.kind == "x"
    assert rt.failure_count == 0
    assert rt.backoff_ms == BACKOFF_INITIAL_MS
    assert rt.parked is False
    assert rt.process is None
    assert rt.restart_timer is None
    assert rt.last_exit_code is None
    assert not rt.cancel_event.is_set()


def test_worker_runtime_reset_backoff():
    rt = WorkerRuntime(kind="x", failure_count=3, backoff_ms=BACKOFF_CAP_MS)
    rt.reset_backoff()
    assert rt.failure_count == 0
    assert rt.backoff_ms == BACKOFF_INITIAL_MS


# ---------------------------------------------------------------------------
# spawn_worker — happy path (subprocess exits cleanly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_worker_clean_exit_no_restart(state_dir):
    """Worker that exits 0 should not be restarted."""
    runtime = WorkerRuntime(kind="echo")
    stop = asyncio.Event()
    # Use a python one-liner that exits 0 immediately.
    # We can't easily inject cmd into spawn_worker (it builds argv from kind),
    # so we test the public API by registering a script-style worker. Instead
    # we directly call the worker subprocess via a helper. Here we
    # short-circuit by simulating: write a tiny script to state_dir and
    # patch spawn_worker's _worker_argv via monkey-patch.
    import extensions.daemon.lifecycle as lifecycle_mod

    script = state_dir / "echo.py"
    script.write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    original_argv = lifecycle_mod._worker_argv

    def fake_argv(kind):
        return [sys.executable, str(script)]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lifecycle_mod, "_worker_argv", fake_argv)
    try:
        await spawn_worker(
            runtime,
            supervisor_pid=os.getpid(),
            name="echo",
            dir_=state_dir,
            spawn_mode="same-dir",
            capacity=4,
            permission_mode=None,
            sandbox=False,
            timeout_ms=30000,
            stop_event=stop,
        )
        # 0 = clean stop, no failure recorded.
        assert runtime.last_exit_code == 0
        assert runtime.failure_count == 0
        assert runtime.parked is False
    finally:
        monkey.undo()
        lifecycle_mod._worker_argv = original_argv


@pytest.mark.asyncio
async def test_spawn_worker_permanent_exit_parks(state_dir, short_backoff):
    runtime = WorkerRuntime(kind="bad")
    stop = asyncio.Event()
    script = state_dir / "bad.py"
    script.write_text("import sys; sys.exit(78)\n", encoding="utf-8")

    import extensions.daemon.lifecycle as lifecycle_mod

    def fake_argv(kind):
        return [sys.executable, str(script)]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lifecycle_mod, "_worker_argv", fake_argv)
    try:
        await spawn_worker(
            runtime,
            supervisor_pid=os.getpid(),
            name="bad",
            dir_=state_dir,
            spawn_mode="same-dir",
            capacity=4,
            permission_mode=None,
            sandbox=False,
            timeout_ms=30000,
            stop_event=stop,
        )
        assert runtime.last_exit_code == 78
        assert runtime.parked is True
    finally:
        monkey.undo()


@pytest.mark.asyncio
async def test_spawn_worker_transient_then_stop(state_dir, short_backoff):
    """Transient failure (exit 1) should be scheduled for restart,
    but if stop_event is set during the backoff, the worker stays stopped.
    """
    runtime = WorkerRuntime(kind="loop")
    stop = asyncio.Event()

    # Backoff is short — schedule stop_event to fire DURING the backoff
    # so the supervisor aborts the restart attempt.
    async def kick_stop():
        await asyncio.sleep(0.05)
        stop.set()

    script = state_dir / "loop.py"
    script.write_text("import sys; sys.exit(1)\n", encoding="utf-8")

    import extensions.daemon.lifecycle as lifecycle_mod

    def fake_argv(kind):
        return [sys.executable, str(script)]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lifecycle_mod, "_worker_argv", fake_argv)
    try:
        # Keep backoff long enough that the stopper definitely fires first.
        runtime.backoff_ms = 2_000
        stopper_task = asyncio.create_task(kick_stop())
        await spawn_worker(
            runtime,
            supervisor_pid=os.getpid(),
            name="loop",
            dir_=state_dir,
            spawn_mode="same-dir",
            capacity=4,
            permission_mode=None,
            sandbox=False,
            timeout_ms=30000,
            stop_event=stop,
        )
        await stopper_task
        # Exit code was recorded; restart was NOT attempted.
        assert runtime.last_exit_code == 1
        assert runtime.total_restarts == 0
    finally:
        monkey.undo()


@pytest.mark.asyncio
async def test_rapid_failure_park(state_dir, short_backoff):
    """Three back-to-back short-lived failures should park the worker
    (short_backoff fixture sets MAX_RAPID_FAILURES=3).
    """
    runtime = WorkerRuntime(kind="crashloop")
    stop = asyncio.Event()
    script = state_dir / "crash.py"
    # Exit 1 quickly — counts as a "rapid" failure under the
    # shortened backoff window.
    script.write_text("import sys; sys.exit(1)\n", encoding="utf-8")

    import extensions.daemon.lifecycle as lifecycle_mod

    def fake_argv(kind):
        return [sys.executable, str(script)]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lifecycle_mod, "_worker_argv", fake_argv)
    try:
        # Reduce backoff further so the test finishes in <5s.
        runtime.backoff_ms = 1
        await spawn_worker(
            runtime,
            supervisor_pid=os.getpid(),
            name="crashloop",
            dir_=state_dir,
            spawn_mode="same-dir",
            capacity=4,
            permission_mode=None,
            sandbox=False,
            timeout_ms=30000,
            stop_event=stop,
        )
        # After MAX_RAPID_FAILURES (=3) the worker should be parked.
        assert runtime.parked is True
        assert runtime.failure_count >= 3
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# spawn_worker — error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_worker_missing_interpreter_raises(state_dir):
    runtime = WorkerRuntime(kind="missing")
    stop = asyncio.Event()
    import extensions.daemon.lifecycle as lifecycle_mod

    def fake_argv(kind):
        return ["/nonexistent/python-xyz", "irrelevant"]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lifecycle_mod, "_worker_argv", fake_argv)
    try:
        with pytest.raises(WorkerSpawnError):
            await spawn_worker(
                runtime,
                supervisor_pid=os.getpid(),
                name="missing",
                dir_=state_dir,
                spawn_mode="same-dir",
                capacity=4,
                permission_mode=None,
                sandbox=False,
                timeout_ms=30000,
                stop_event=stop,
            )
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# graceful_shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graceful_shutdown_no_workers():
    # No runtimes — should be a no-op, no raise.
    await graceful_shutdown([], timeout_ms=10)


@pytest.mark.asyncio
async def test_graceful_shutdown_terminates_live_process(state_dir):
    """Spawn a long-running subprocess, then call graceful_shutdown and
    confirm the process is gone within the timeout.
    """
    import extensions.daemon.lifecycle as lifecycle_mod

    script = state_dir / "long.py"
    # Sleep longer than the graceful timeout we'll pass.
    script.write_text("import time, sys; time.sleep(60); sys.exit(0)\n", encoding="utf-8")

    def fake_argv(kind):
        return [sys.executable, str(script)]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lifecycle_mod, "_worker_argv", fake_argv)
    try:
        runtime = WorkerRuntime(kind="long")
        stop = asyncio.Event()
        # Spawn the worker in the background.
        task = asyncio.create_task(
            spawn_worker(
                runtime,
                supervisor_pid=os.getpid(),
                name="long",
                dir_=state_dir,
                spawn_mode="same-dir",
                capacity=4,
                permission_mode=None,
                sandbox=False,
                timeout_ms=30000,
                stop_event=stop,
            )
        )
        # Wait until the subprocess is up.
        for _ in range(50):
            if runtime.process is not None and runtime.process.returncode is None:
                break
            await asyncio.sleep(0.02)
        assert runtime.process is not None
        assert runtime.process.returncode is None

        await graceful_shutdown([runtime], timeout_ms=2_000)

        # Process should be gone.
        assert runtime.process is None or runtime.process.returncode is not None
        # The spawn_worker task may still be running waiting on the
        # backoff — cancel it.
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass  # The test explicitly cancels this lingering worker task.
    finally:
        monkey.undo()
