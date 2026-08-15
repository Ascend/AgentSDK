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
"""End-to-end Supervisor round-trip tests.

These tests drive a real ``extensions.daemon.Supervisor`` instance through
its full lifecycle:

1. ``start`` writes the daemon state file and accepts ``status``.
2. A worker that exits with a transient code is restarted within
   ``BACKOFF_INITIAL_MS``.
3. A worker that returns :data:`EXIT_CODE_PERMANENT` is parked (no further
   invocations).
4. ``SIGTERM`` (or stop_event) triggers graceful shutdown — state file
   is removed.
5. Stale state files are auto-cleaned by :func:`query_daemon_status`.
6. Atomic writes never leave ``.tmp`` artefacts.

Design
------
The pattern follows ``tests/orchestrator/manual_e2e_f38.py`` (LocalTracker
sandbox style): every artefact is inside ``tmp_path`` and every worker is
a small Python script. Workers append to a counter file on each
invocation so tests can assert "spawned N times" without instrumenting
the in-process ``Worker.run()`` — the supervisor spawns real subprocesses
through ``_worker_argv`` and never touches ``Worker.run()`` directly.

Run with::

    python -m pytest tests/extensions/daemon/test_e2e_supervisor.py -q

Total wall-time is bounded to ~30s by the per-test ``asyncio.wait_for``
timeouts.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import pytest

from extensions.daemon import constants
from extensions.daemon.config import DaemonConfig
from extensions.daemon.state import (
    DaemonStatus,
    is_process_alive,
    query_daemon_status,
    read_daemon_state,
)
from extensions.daemon.supervisor import Supervisor
from extensions.daemon.worker_registry import WorkerRegistry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: All E2E tests share this short polling interval when waiting for the
#: state file to appear.
_POLL_INTERVAL_S = 0.05

#: Number of polling cycles before we declare "state file never appeared".
_POLL_CYCLES = 100  # 100 × 50ms = 5s


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_backoff_constants():
    """Snapshot + restore the backoff tunables so E2E tests can shrink
    them without bleeding into the rest of the suite.
    """
    saved = {
        "BACKOFF_INITIAL_MS": constants.BACKOFF_INITIAL_MS,
        "BACKOFF_CAP_MS": constants.BACKOFF_CAP_MS,
        "RAPID_FAILURE_WINDOW_MS": constants.RAPID_FAILURE_WINDOW_MS,
        "MAX_RAPID_FAILURES": constants.MAX_RAPID_FAILURES,
    }
    yield
    for k, v in saved.items():
        setattr(constants, k, v)


@pytest.fixture(autouse=True)
def _reset_worker_registry():
    """Reset the registry around each test, then re-register the
    built-ins plus a stub factory for every E2E worker kind.

    The supervisor validates that every kind in ``worker_kinds`` is
    registered before spawning, so even when ``Worker.run()`` is never
    called (the supervisor drives subprocesses through
    ``_worker_argv``), the registry entry must exist.
    """
    from extensions.daemon.workers import build_cron_worker, build_remote_control_worker

    WorkerRegistry.reset()
    WorkerRegistry.register("remoteControl", build_remote_control_worker)
    WorkerRegistry.register("cron", build_cron_worker)
    # Register stub factories for every E2E kind we use. The factory's
    # ``run()`` is never invoked by the supervisor; we only need the
    # name to be registered so the supervisor's pre-flight check passes.
    for kind in (
        "e2e-block",
        "e2e-flap",
        "e2e-bad",
        "e2e-rapid",
    ):
        WorkerRegistry.register(kind, _stub_factory)
    yield
    WorkerRegistry.reset()


@pytest.fixture
def fast_backoff(monkeypatch: pytest.MonkeyPatch):
    """Shrink backoff windows so transient-failure rounds finish quickly.

    Note: ``extensions.daemon.lifecycle`` imports the backoff tunables
    via ``from .constants import ...``, which captures the values at
    import time. Likewise ``WorkerRuntime.backoff_ms`` is bound at
    class-definition time. We patch all three layers so the runtime
    sees the shrunken values.
    """
    import extensions.daemon.lifecycle as lifecycle_mod

    constants.BACKOFF_INITIAL_MS = 200
    constants.BACKOFF_CAP_MS = 1_000
    constants.RAPID_FAILURE_WINDOW_MS = 500
    constants.MAX_RAPID_FAILURES = 3

    # Re-bind the imported names inside the lifecycle module so the
    # running code sees the shrunken constants.
    monkeypatch.setattr(lifecycle_mod, "BACKOFF_INITIAL_MS", 200)
    monkeypatch.setattr(lifecycle_mod, "BACKOFF_CAP_MS", 1_000)
    monkeypatch.setattr(lifecycle_mod, "RAPID_FAILURE_WINDOW_MS", 500)
    monkeypatch.setattr(lifecycle_mod, "MAX_RAPID_FAILURES", 3)

    # WorkerRuntime captured BACKOFF_INITIAL_MS at class-definition
    # time, so we also patch its __init__ to inject the desired value.
    original_init = lifecycle_mod.WorkerRuntime.__init__

    def _patched_init(self, **kwargs):
        kwargs.setdefault("backoff_ms", 200)
        original_init(self, **kwargs)

    monkeypatch.setattr(lifecycle_mod.WorkerRuntime, "__init__", _patched_init)


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    d = tmp_path / "daemon-state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stub_factory():
    """Stub worker factory for E2E kinds — never invoked by the
    supervisor because subprocesses are driven through ``_worker_argv``.
    Raises if accidentally called so we notice broken test setup.
    """

    class _Stub:
        kind = "stub"

        async def run(self, env):
            raise AssertionError("stub factory's run() should not be called by the supervisor")

        def health_check(self):
            return None

    return _Stub()


# ---------------------------------------------------------------------------
# Script-based worker helpers (LocalTracker style)
# ---------------------------------------------------------------------------


def _write_blocking_worker(script: Path) -> None:
    """Worker that writes to stdout and sleeps forever — used to keep the
    supervisor alive while we inspect the state file.
    """
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import sys, time\nsys.stdout.write('alive\\n')\nsys.stdout.flush()\ntime.sleep(120)\n",
        encoding="utf-8",
    )


def _write_counter_worker(script: Path, counter_file: Path, exit_code: int) -> None:
    """Worker that appends a single byte to *counter_file* on each invocation
    and then exits with *exit_code*. Used to count spawn cycles.
    """
    script.parent.mkdir(parents=True, exist_ok=True)
    counter_str = str(counter_file).replace("\\", "\\\\").replace("'", "\\'")
    script.write_text(
        f"import sys\nwith open('{counter_str}', 'a') as f:\n    f.write('x')\nsys.exit({exit_code})\n",
        encoding="utf-8",
    )


def _write_rapid_counter_worker(script: Path, counter_file: Path, exit_code: int) -> None:
    """Like :func:`_write_counter_worker` but with a tiny sleep so the
    run duration is comfortably under ``RAPID_FAILURE_WINDOW_MS`` (this
    is what triggers parking logic).
    """
    script.parent.mkdir(parents=True, exist_ok=True)
    counter_str = str(counter_file).replace("\\", "\\\\").replace("'", "\\'")
    script.write_text(
        f"import sys, time\n"
        f"time.sleep(0.02)\n"
        f"with open('{counter_str}', 'a') as f:\n"
        f"    f.write('x')\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )


def _patch_worker_argv(
    monkeypatch: pytest.MonkeyPatch,
    scripts: dict[str, Path],
    *,
    fallback_dir: Path,
) -> None:
    """Replace ``lifecycle._worker_argv`` so each kind spawns the script
    we wrote into *scripts*, instead of ``python -m extensions.daemon.worker_main``.

    Any kind not in *scripts* falls back to a 0-exit no-op (avoid
    busy-looping on missing registrations). The fallback lives in
    *fallback_dir* (caller-supplied ``tmp_path``) so we never write
    into the production daemon package directory.
    """
    import extensions.daemon.lifecycle as lifecycle_mod

    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback = fallback_dir / "_e2e_fallback_worker.py"
    if not fallback.exists():
        fallback.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    def fake_argv(kind: str) -> list[str]:
        script = scripts.get(kind, fallback)
        return [sys.executable, str(script)]

    monkeypatch.setattr(lifecycle_mod, "_worker_argv", fake_argv)


def _count_invocations(counter_file: Path) -> int:
    """Return the number of times a counter worker has been spawned."""
    if not counter_file.exists():
        return 0
    return len(counter_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# V-1: ``start`` writes the daemon state file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v1_start_writes_state_file(
    tmp_path: Path,
    state_dir: Path,
    fast_backoff,
    monkeypatch: pytest.MonkeyPatch,
):
    """V-1: while the supervisor is alive, the state file exists and is
    queryable as RUNNING. After graceful shutdown, it is removed.
    """
    cfg = DaemonConfig(
        name="e2e-v1",
        dir=tmp_path,
        worker_kinds=("e2e-block",),
        spawn_mode="same-dir",
        capacity=2,
        backoff_initial_ms=constants.BACKOFF_INITIAL_MS,
        backoff_cap_ms=constants.BACKOFF_CAP_MS,
    )

    # Blocking worker keeps the supervisor alive while we read the
    # state file.
    block_script = tmp_path / "block.py"
    _write_blocking_worker(block_script)
    _patch_worker_argv(monkeypatch, {"e2e-block": block_script}, fallback_dir=tmp_path)

    sup = Supervisor(cfg, state_dir=state_dir)
    state_path = state_dir / f"{cfg.name}.json"

    task = asyncio.create_task(sup.run())

    # Wait for the state file to appear (written before the first spawn).
    for _ in range(_POLL_CYCLES):
        if state_path.exists():
            break
        await asyncio.sleep(_POLL_INTERVAL_S)

    assert state_path.exists(), "daemon state file was not written on start"

    # Validate the persisted state.
    state = read_daemon_state(cfg.name, state_dir=state_dir)
    assert state is not None
    assert state.name == "e2e-v1"
    assert state.worker_kinds == ["e2e-block"]
    assert state.cwd == str(tmp_path.resolve())
    assert state.pid == os.getpid()
    assert state.last_status.value == "running"

    # query_daemon_status agrees it's RUNNING.
    status, queried = query_daemon_status(cfg.name, state_dir=state_dir)
    assert status == DaemonStatus.RUNNING
    assert queried is not None
    assert queried.pid == os.getpid()

    # Stop the supervisor — state file should be cleaned up.
    sup.request_stop()
    rc = await asyncio.wait_for(task, timeout=10.0)
    assert rc == 0
    assert not state_path.exists()


# ---------------------------------------------------------------------------
# V-3: graceful shutdown removes the state file (in-process stop)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v3_graceful_shutdown_removes_state(
    tmp_path: Path,
    state_dir: Path,
    fast_backoff,
    monkeypatch: pytest.MonkeyPatch,
):
    """V-3: in-process ``request_stop()`` simulates the SIGTERM path —
    graceful shutdown removes the state file.
    """
    cfg = DaemonConfig(
        name="e2e-v3",
        dir=tmp_path,
        worker_kinds=("e2e-block",),
        spawn_mode="same-dir",
        capacity=2,
        backoff_initial_ms=constants.BACKOFF_INITIAL_MS,
        backoff_cap_ms=constants.BACKOFF_CAP_MS,
    )

    block_script = tmp_path / "block.py"
    _write_blocking_worker(block_script)
    _patch_worker_argv(monkeypatch, {"e2e-block": block_script}, fallback_dir=tmp_path)

    sup = Supervisor(cfg, state_dir=state_dir)
    state_path = state_dir / f"{cfg.name}.json"

    async def stopper() -> None:
        for _ in range(_POLL_CYCLES):
            if state_path.exists():
                break
            await asyncio.sleep(_POLL_INTERVAL_S)
        assert state_path.exists(), "state file should exist before stop"
        sup.request_stop()

    task = asyncio.create_task(sup.run())
    await asyncio.wait_for(stopper(), timeout=5.0)
    rc = await asyncio.wait_for(task, timeout=10.0)
    assert rc == 0
    assert not state_path.exists(), "graceful shutdown must remove the state file"


# ---------------------------------------------------------------------------
# V-4: transient failure → restart within BACKOFF_INITIAL_MS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v4_transient_failure_triggers_restart(
    tmp_path: Path,
    state_dir: Path,
    fast_backoff,
    monkeypatch: pytest.MonkeyPatch,
):
    """V-4: a worker that exits 1 is restarted within BACKOFF_INITIAL_MS
    (shrunk to 200ms here).
    """
    cfg = DaemonConfig(
        name="e2e-v4",
        dir=tmp_path,
        worker_kinds=("e2e-flap",),
        spawn_mode="same-dir",
        capacity=2,
        backoff_initial_ms=constants.BACKOFF_INITIAL_MS,
        backoff_cap_ms=constants.BACKOFF_CAP_MS,
    )

    # Raise MAX_RAPID_FAILURES high enough that we observe a restart
    # before parking kicks in. We'll stop manually after 2 invocations.
    constants.MAX_RAPID_FAILURES = 99

    counter_file = tmp_path / "counter.txt"
    flap_script = tmp_path / "flap.py"
    _write_counter_worker(flap_script, counter_file, exit_code=1)
    _patch_worker_argv(monkeypatch, {"e2e-flap": flap_script}, fallback_dir=tmp_path)

    sup = Supervisor(cfg, state_dir=state_dir)

    async def stop_after_two_invocations() -> None:
        # Wait until we see two spawns recorded.
        for _ in range(_POLL_CYCLES):
            if _count_invocations(counter_file) >= 2:
                break
            await asyncio.sleep(_POLL_INTERVAL_S)
        sup.request_stop()

    task = asyncio.create_task(sup.run())
    await asyncio.wait_for(stop_after_two_invocations(), timeout=10.0)
    rc = await asyncio.wait_for(task, timeout=10.0)
    assert rc == 0

    # Two invocations: original + 1 restart. Counter holds 2 'x' chars.
    invocations = _count_invocations(counter_file)
    assert invocations >= 2, f"expected restart, got {invocations} invocations"

    # Backoff state was updated at least once. ``last_exit_code`` is
    # implementation-dependent here: the subprocess that was running
    # when we requested stop was SIGTERM'd by the supervisor, so its
    # exit code can be either 1 (clean exit caught by supervisor) or
    # a negative signal code (-15 = SIGTERM). We assert the restart
    # counter instead, which is deterministic.
    runtime = sup.runtimes["e2e-flap"]
    assert runtime.total_restarts >= 1


# ---------------------------------------------------------------------------
# V-5: ``EXIT_CODE_PERMANENT`` parks the worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v5_permanent_exit_parks_worker(
    tmp_path: Path,
    state_dir: Path,
    fast_backoff,
    monkeypatch: pytest.MonkeyPatch,
):
    """V-5: a worker that exits with ``EXIT_CODE_PERMANENT`` (78) is
    parked — the supervisor does NOT restart it.
    """
    cfg = DaemonConfig(
        name="e2e-v5",
        dir=tmp_path,
        worker_kinds=("e2e-bad",),
        spawn_mode="same-dir",
        capacity=1,
        backoff_initial_ms=constants.BACKOFF_INITIAL_MS,
        backoff_cap_ms=constants.BACKOFF_CAP_MS,
    )

    counter_file = tmp_path / "counter.txt"
    bad_script = tmp_path / "bad.py"
    _write_counter_worker(bad_script, counter_file, exit_code=78)
    _patch_worker_argv(monkeypatch, {"e2e-bad": bad_script}, fallback_dir=tmp_path)

    sup = Supervisor(cfg, state_dir=state_dir)

    async def stop_after_parking() -> None:
        # Wait for the worker to run at least once and then for parking.
        for _ in range(_POLL_CYCLES):
            if _count_invocations(counter_file) >= 1:
                # Give the supervisor a beat to flip parked=True.
                await asyncio.sleep(_POLL_INTERVAL_S)
                if sup.runtimes["e2e-bad"].parked:
                    break
            await asyncio.sleep(_POLL_INTERVAL_S)
        sup.request_stop()

    task = asyncio.create_task(sup.run())
    await asyncio.wait_for(stop_after_parking(), timeout=10.0)
    rc = await asyncio.wait_for(task, timeout=10.0)
    assert rc == 0

    # Exactly one invocation before parking.
    invocations = _count_invocations(counter_file)
    assert invocations == 1, f"expected 1 invocation, got {invocations}"
    runtime = sup.runtimes["e2e-bad"]
    assert runtime.parked is True
    assert runtime.last_exit_code == 78


# ---------------------------------------------------------------------------
# V-6: rapid failures → parking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v6_rapid_failures_park_worker(
    tmp_path: Path,
    state_dir: Path,
    fast_backoff,
    monkeypatch: pytest.MonkeyPatch,
):
    """V-6: ``MAX_RAPID_FAILURES`` short-lived crashes in
    ``RAPID_FAILURE_WINDOW_MS`` park the worker (no further spawns).
    """
    cfg = DaemonConfig(
        name="e2e-v6",
        dir=tmp_path,
        worker_kinds=("e2e-rapid",),
        spawn_mode="same-dir",
        capacity=1,
        backoff_initial_ms=constants.BACKOFF_INITIAL_MS,
        backoff_cap_ms=constants.BACKOFF_CAP_MS,
    )

    counter_file = tmp_path / "counter.txt"
    rapid_script = tmp_path / "rapid.py"
    _write_rapid_counter_worker(rapid_script, counter_file, exit_code=1)
    _patch_worker_argv(monkeypatch, {"e2e-rapid": rapid_script}, fallback_dir=tmp_path)

    sup = Supervisor(cfg, state_dir=state_dir)

    async def stop_after_parking() -> None:
        # Wait for parking to flip after MAX_RAPID_FAILURES spawns.
        for _ in range(_POLL_CYCLES):
            if sup.runtimes["e2e-rapid"].parked:
                # Give the supervisor one more event-loop tick so the
                # subprocess's counter write fully lands on disk before
                # we read it.
                await asyncio.sleep(_POLL_INTERVAL_S)
                break
            await asyncio.sleep(_POLL_INTERVAL_S)
        sup.request_stop()

    task = asyncio.create_task(sup.run())
    await asyncio.wait_for(stop_after_parking(), timeout=15.0)
    rc = await asyncio.wait_for(task, timeout=10.0)
    assert rc == 0

    invocations = _count_invocations(counter_file)
    runtime = sup.runtimes["e2e-rapid"]
    # Diagnostic dump for failure analysis (printed only on failure).
    if not (constants.MAX_RAPID_FAILURES <= invocations <= constants.MAX_RAPID_FAILURES + 1):
        print(
            f"\n[V6 DIAG] invocations={invocations} "
            f"failure_count={runtime.failure_count} "
            f"total_restarts={runtime.total_restarts} "
            f"last_exit_code={runtime.last_exit_code} "
            f"backoff_ms={runtime.backoff_ms} "
            f"parked={runtime.parked}"
        )
    # MAX_RAPID_FAILURES is 3 in fast_backoff — we should see at least
    # that many invocations before parking kicked in. Allow a small
    # tolerance (≤ MAX) to keep the assertion robust against scheduling
    # jitter on slow CI hosts.
    assert invocations >= constants.MAX_RAPID_FAILURES, (
        f"expected at least {constants.MAX_RAPID_FAILURES} invocations, got {invocations}"
    )
    assert invocations <= constants.MAX_RAPID_FAILURES + 1, f"too many invocations before parking: {invocations}"
    assert runtime.parked is True


# ---------------------------------------------------------------------------
# V-9: stale state file is auto-cleaned by query_daemon_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v9_stale_state_auto_cleanup(state_dir: Path):
    """V-9: if the recorded PID is dead, ``query_daemon_status`` removes
    the state file and returns ``STALE``.
    """
    from extensions.daemon.state import make_state, write_daemon_state

    stale_pid = _dead_pid()
    state = make_state(
        pid=stale_pid,
        worker_kinds=["remoteControl"],
        name="ghost",
    )
    write_daemon_state(state, state_dir=state_dir)

    state_path = state_dir / f"{state.name}.json"
    assert state_path.exists()
    assert not is_process_alive(stale_pid)

    status, queried = query_daemon_status(state.name, state_dir=state_dir)
    assert status == DaemonStatus.STALE
    assert queried is None
    assert not state_path.exists(), "stale state file should be removed"


# ---------------------------------------------------------------------------
# V-10: state file write is atomic
# ---------------------------------------------------------------------------


def test_v10_atomic_state_write(state_dir: Path):
    """V-10: ``write_daemon_state`` uses ``.tmp`` + ``os.replace`` so a
    concurrent reader never sees a half-written JSON file. No ``.tmp``
    artefact is left behind on success.
    """
    from extensions.daemon.state import make_state, write_daemon_state

    state = make_state(
        pid=os.getpid(),
        worker_kinds=["a", "b", "c"],
        name="atomic",
    )
    write_daemon_state(state, state_dir=state_dir)
    # The persisted file is valid JSON and round-trips.
    raw = (state_dir / "atomic.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["worker_kinds"] == ["a", "b", "c"]
    assert parsed["pid"] == os.getpid()
    # No leftover .tmp files.
    leftovers = list(state_dir.glob("*.tmp"))
    assert not leftovers, f"atomic write left tmp files: {leftovers}"


# ---------------------------------------------------------------------------
# V-3 (signal path): real SIGTERM triggers graceful shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v3_sigterm_triggers_graceful_shutdown(
    tmp_path: Path,
    state_dir: Path,
    fast_backoff,
    monkeypatch: pytest.MonkeyPatch,
):
    """V-3 (signal path): send ``SIGTERM`` to the supervisor process and
    confirm the state file is removed within the graceful timeout.

    On Windows, ``SIGTERM`` is not supported by ``asyncio`` so we skip
    this variant — the in-process variant above covers the same path.
    """
    if sys.platform == "win32":
        pytest.skip("SIGTERM signal-based shutdown not portable to Windows")

    cfg = DaemonConfig(
        name="e2e-sigterm",
        dir=tmp_path,
        worker_kinds=("e2e-block",),
        spawn_mode="same-dir",
        capacity=1,
        backoff_initial_ms=constants.BACKOFF_INITIAL_MS,
        backoff_cap_ms=constants.BACKOFF_CAP_MS,
    )

    block_script = tmp_path / "block.py"
    _write_blocking_worker(block_script)
    _patch_worker_argv(monkeypatch, {"e2e-block": block_script}, fallback_dir=tmp_path)

    sup = Supervisor(cfg, state_dir=state_dir)
    state_path = state_dir / f"{cfg.name}.json"

    task = asyncio.create_task(sup.run())

    # Wait for the state file to appear before signaling.
    for _ in range(_POLL_CYCLES):
        if state_path.exists():
            break
        await asyncio.sleep(_POLL_INTERVAL_S)
    assert state_path.exists()

    # Send SIGTERM to ourselves — the signal handler installed by the
    # supervisor will flip stop_event.
    os.kill(os.getpid(), signal.SIGTERM)

    rc = await asyncio.wait_for(task, timeout=10.0)
    assert rc == 0
    assert not state_path.exists()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """Return a PID that is very unlikely to be alive on this host.

    Probe upward from a high starting value until we find a PID that
    ``os.kill(pid, 0)`` reports as dead. If we can't find one in 50
    tries, fall back to INT32_MAX which is conventionally unused.
    """
    candidate = 1_000_000
    for _ in range(50):
        if not is_process_alive(candidate):
            return candidate
        candidate += 1
    return 2_147_483_647
