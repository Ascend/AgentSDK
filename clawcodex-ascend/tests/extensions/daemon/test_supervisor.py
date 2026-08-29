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

"""Tests for ``extensions.daemon.supervisor``.

Supervisor tests use **in-process worker factories** rather than
spawning real subprocesses. We monkey-patch
``extensions.daemon.supervisor.spawn_worker`` to invoke the
registered factory's ``run(env)`` coroutine directly. This keeps the
tests fast (no ``python -m extensions.daemon.worker_main`` round-trip)
and avoids contention on ``asyncio.add_signal_handler`` which
behaves differently inside pytest.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from extensions.daemon.config import DaemonConfig
from extensions.daemon.constants import (
    EXIT_CODE_OK,
    EXIT_CODE_PERMANENT,
)
from extensions.daemon.errors import DaemonAlreadyRunningError
from extensions.daemon.state import make_state, write_daemon_state
from extensions.daemon.supervisor import Supervisor
from extensions.daemon.worker_registry import WorkerRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(tmp_path, *, workers=("cron",)) -> DaemonConfig:
    return DaemonConfig(
        name="sup-test",
        dir=tmp_path,
        worker_kinds=workers,
        spawn_mode="same-dir",
        capacity=2,
        backoff_initial_ms=50,
        backoff_cap_ms=200,
    )


class _InProcWorker:
    """Test worker that runs in-process and returns a configurable code.

    Each call to ``run`` increments ``call_count`` so tests can assert
    how many times the supervisor invoked the worker.
    """

    kind = "cron"

    def __init__(self, exit_code: int = EXIT_CODE_OK) -> None:
        self.exit_code = exit_code
        self.call_count = 0
        self.last_env: dict[str, str] | None = None

    async def run(self, env: dict[str, str]) -> int:
        self.call_count += 1
        self.last_env = dict(env)
        return self.exit_code


def _patch_supervisor_to_run_inproc(monkeypatch, workers: dict[str, _InProcWorker]):
    """Replace ``supervisor.spawn_worker`` with an in-process runner.

    The in-process runner looks up the worker kind in *workers* and
    awaits its ``run`` method. If the kind is not in *workers*, the
    runner falls back to a 0-exit no-op (matches the "clean stop"
    branch in :func:`lifecycle.spawn_worker`).
    """

    async def fake_spawn_worker(runtime, **_kwargs):
        worker = workers.get(runtime.kind)
        if worker is None:
            runtime.last_exit_code = 0
            return
        env = {
            "CLAWCODEX_SUPERVISOR_PID": str(os.getpid()),
            "CLAWCODEX_DAEMON_NAME": "sup-test",
            "CLAWCODEX_DAEMON_SPAWN_MODE": "same-dir",
            "CLAWCODEX_DAEMON_CAPACITY": "2",
            "CLAWCODEX_DAEMON_PERMISSION_MODE": "",
            "CLAWCODEX_DAEMON_SANDBOX": "0",
            "CLAWCODEX_DAEMON_TIMEOUT_MS": "30000",
            "CLAWCODEX_SESSION_KIND": "daemon-worker",
        }
        rc = await worker.run(env)
        runtime.last_exit_code = rc
        if rc == EXIT_CODE_PERMANENT:
            runtime.parked = True
        elif rc == EXIT_CODE_OK:
            return  # no restart

    monkeypatch.setattr("extensions.daemon.supervisor.spawn_worker", fake_spawn_worker)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    WorkerRegistry.reset()
    from extensions.daemon.workers import build_cron_worker, build_remote_control_worker

    WorkerRegistry.register("remoteControl", build_remote_control_worker)
    WorkerRegistry.register("cron", build_cron_worker)
    yield
    WorkerRegistry.reset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_supervisor_validates_config(tmp_path):
    cfg = _build_config(tmp_path, workers=("cron",))
    cfg.validate()  # no raise


@pytest.mark.asyncio
async def test_supervisor_rejects_when_already_running(state_dir, tmp_path):
    cfg = _build_config(tmp_path)
    state = make_state(pid=os.getpid(), worker_kinds=cfg.worker_kinds, name=cfg.name)
    write_daemon_state(state, state_dir=state_dir)
    sup = Supervisor(cfg, state_dir=state_dir)
    with pytest.raises(DaemonAlreadyRunningError):
        await sup.run()


@pytest.mark.asyncio
async def test_supervisor_rejects_unknown_kind(tmp_path, state_dir, monkeypatch):
    cfg = DaemonConfig(
        name="x",
        dir=tmp_path,
        worker_kinds=("not-a-real-kind",),
        spawn_mode="same-dir",
        capacity=1,
        backoff_initial_ms=10,
    )
    _patch_supervisor_to_run_inproc(monkeypatch, {})
    sup = Supervisor(cfg, state_dir=state_dir)
    with pytest.raises(ValueError, match="not-a-real-kind"):
        await sup.run()


@pytest.mark.asyncio
async def test_supervisor_cleans_state_on_exit(tmp_path, state_dir, monkeypatch):
    cfg = _build_config(tmp_path, workers=("cron",))
    worker = _InProcWorker(exit_code=EXIT_CODE_OK)
    _patch_supervisor_to_run_inproc(monkeypatch, {"cron": worker})

    sup = Supervisor(cfg, state_dir=state_dir)
    rc = await sup.run()
    assert rc == 0
    assert not (state_dir / f"{cfg.name}.json").exists()
    assert worker.call_count == 1


@pytest.mark.asyncio
async def test_supervisor_request_stop_sets_event(state_dir, tmp_path):
    cfg = _build_config(tmp_path)
    sup = Supervisor(cfg, state_dir=state_dir)
    assert not sup.stop_event.is_set()
    sup.request_stop()
    assert sup.stop_event.is_set()


@pytest.mark.asyncio
async def test_supervisor_emits_state_change_callback(tmp_path, state_dir, monkeypatch):
    cfg = _build_config(tmp_path, workers=("cron",))
    worker = _InProcWorker(exit_code=EXIT_CODE_OK)
    _patch_supervisor_to_run_inproc(monkeypatch, {"cron": worker})

    seen: list = []
    sup = Supervisor(
        cfg,
        state_dir=state_dir,
        on_state_change=lambda s: seen.append(s),
    )
    await sup.run()
    assert len(seen) >= 1
    assert seen[0] is not None
    assert seen[-1] is None


@pytest.mark.asyncio
async def test_supervisor_stops_when_requested(tmp_path, state_dir, monkeypatch):
    """Worker that blocks forever should be interruptible via stop_event."""
    cfg = _build_config(tmp_path, workers=("cron",))

    class _BlockingWorker(_InProcWorker):
        def __init__(self) -> None:
            super().__init__()
            self.stop_event: asyncio.Event | None = None

        async def run(self, env):
            self.call_count += 1
            # Block until the supervisor requests stop — genuinely
            # exercises the stop_event interruption path.
            assert self.stop_event is not None
            await self.stop_event.wait()
            return EXIT_CODE_OK

    worker = _BlockingWorker()
    _patch_supervisor_to_run_inproc(monkeypatch, {"cron": worker})

    sup = Supervisor(cfg, state_dir=state_dir)
    worker.stop_event = sup.stop_event

    async def stop_after():
        await asyncio.sleep(0.1)
        sup.request_stop()

    stopper = asyncio.create_task(stop_after())
    rc = await sup.run()
    await stopper
    assert rc == 0
    # Worker should have been invoked at least once.
    assert worker.call_count >= 1


@pytest.mark.asyncio
async def test_supervisor_persistent_park(tmp_path, state_dir, monkeypatch):
    """Worker that returns PERMANENT should be parked (no further runs)."""
    cfg = _build_config(tmp_path, workers=("cron",))
    worker = _InProcWorker(exit_code=EXIT_CODE_PERMANENT)
    _patch_supervisor_to_run_inproc(monkeypatch, {"cron": worker})

    sup = Supervisor(cfg, state_dir=state_dir)
    rc = await sup.run()
    assert rc == 0
    # Worker ran once, then was parked.
    assert worker.call_count == 1
    runtime = sup.runtimes["cron"]
    assert runtime.parked is True
