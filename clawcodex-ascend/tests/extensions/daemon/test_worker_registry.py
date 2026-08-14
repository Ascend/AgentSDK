"""Tests for ``extensions.daemon.worker_registry`` and built-in workers."""

from __future__ import annotations

import pytest

from extensions.daemon.errors import UnknownWorkerKindError
from extensions.daemon.worker_registry import WorkerRegistry
from extensions.daemon.workers import (
    build_cron_worker,
    build_remote_control_worker,
    build_task_server_worker,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    WorkerRegistry.reset()
    # Re-register built-in workers (same as ``extensions.daemon.workers``).
    WorkerRegistry.register("remoteControl", build_remote_control_worker)
    WorkerRegistry.register("task_server", build_task_server_worker)
    WorkerRegistry.register("cron", build_cron_worker)
    yield
    WorkerRegistry.reset()


def test_known_kinds_contains_builtins():
    kinds = WorkerRegistry.known_kinds()
    assert "remoteControl" in kinds
    assert "task_server" in kinds
    assert "cron" in kinds


def test_register_decorator_returns_factory():
    @WorkerRegistry.register("decorated")
    def _factory():
        return object()

    assert WorkerRegistry.has_kind("decorated")


def test_register_overrides_existing():
    sentinel_a = object()
    sentinel_b = object()

    def factory_a():
        return sentinel_a

    def factory_b():
        return sentinel_b

    WorkerRegistry.register("dup", factory_a)
    WorkerRegistry.register("dup", factory_b)
    assert WorkerRegistry.create("dup") is sentinel_b


def test_register_rejects_empty_kind():
    with pytest.raises(ValueError):
        WorkerRegistry.register("", lambda: None)


def test_register_rejects_non_callable():
    with pytest.raises(TypeError):
        WorkerRegistry.register("bad", 42)  # type: ignore[arg-type]


def test_create_unknown_kind_raises():
    with pytest.raises(UnknownWorkerKindError) as ei:
        WorkerRegistry.create("not-here")
    assert ei.value.kind == "not-here"


def test_unregister_returns_true_when_present():
    WorkerRegistry.register("tmp", lambda: None)
    assert WorkerRegistry.unregister("tmp") is True
    assert WorkerRegistry.has_kind("tmp") is False


def test_unregister_returns_false_when_absent():
    assert WorkerRegistry.unregister("absent") is False


def test_reset_clears_everything():
    WorkerRegistry.register("a", lambda: None)
    WorkerRegistry.register("b", lambda: None)
    WorkerRegistry.reset()
    assert WorkerRegistry.known_kinds() == []


def test_remote_control_factory_returns_worker():
    from extensions.daemon.workers import RemoteControlWorker

    worker = WorkerRegistry.create("remoteControl")
    assert isinstance(worker, RemoteControlWorker)
    assert worker.kind == "remoteControl"


def test_task_server_factory_returns_worker():
    from extensions.daemon.workers import TaskServerWorker

    worker = WorkerRegistry.create("task_server")
    assert isinstance(worker, TaskServerWorker)
    assert worker.kind == "task_server"


def test_task_server_health_check():
    worker = WorkerRegistry.create("task_server")
    # Before run(), health_check returns None (not started).
    assert worker.health_check() is None


def test_remote_control_is_task_server_compat():
    """``remoteControl`` remains a ``TaskServerWorker`` subclass for back-compat."""
    from extensions.daemon.workers import TaskServerWorker

    rc = WorkerRegistry.create("remoteControl")
    ts = WorkerRegistry.create("task_server")
    assert isinstance(rc, TaskServerWorker)
    assert isinstance(ts, TaskServerWorker)


def test_cron_factory_returns_worker():
    from extensions.daemon.workers import CronWorker

    worker = WorkerRegistry.create("cron")
    assert isinstance(worker, CronWorker)
    assert worker.kind == "cron"


def test_health_check_returns_dict_for_built_in():
    worker = WorkerRegistry.create("cron")
    snap = worker.health_check()
    assert isinstance(snap, dict)
    assert snap["kind"] == "cron"
    assert snap["uptime_s"] >= 0.0
    assert snap.get("stub") is True


def test_base_worker_reads_daemon_env():
    from extensions.daemon.workers.base import BaseWorker

    env = {
        "CLAWCODEX_DAEMON_NAME": "alpha",
        "CLAWCODEX_DAEMON_DIR": "/tmp/x",
        "CLAWCODEX_DAEMON_SPAWN_MODE": "worktree",
        "CLAWCODEX_DAEMON_CAPACITY": "7",
        "CLAWCODEX_DAEMON_PERMISSION_MODE": "bypassPermissions",
        "CLAWCODEX_DAEMON_SANDBOX": "1",
        "CLAWCODEX_DAEMON_TIMEOUT_MS": "12000",
    }
    cfg = BaseWorker.read_daemon_env(env)
    assert cfg["name"] == "alpha"
    assert cfg["dir"] == "/tmp/x"
    assert cfg["spawn_mode"] == "worktree"
    assert cfg["capacity"] == 7
    assert cfg["permission_mode"] == "bypassPermissions"
    assert cfg["sandbox"] is True
    assert cfg["timeout_ms"] == 12000


def test_base_worker_env_defaults_when_missing():
    from extensions.daemon.workers.base import BaseWorker

    cfg = BaseWorker.read_daemon_env({})
    assert cfg["capacity"] == 4
    assert cfg["sandbox"] is False
    assert cfg["permission_mode"] is None
    assert cfg["timeout_ms"] == 30_000
    assert cfg["spawn_mode"] == "same-dir"


def test_base_worker_env_handles_garbage_values():
    from extensions.daemon.workers.base import BaseWorker

    cfg = BaseWorker.read_daemon_env(
        {
            "CLAWCODEX_DAEMON_CAPACITY": "not-a-number",
            "CLAWCODEX_DAEMON_TIMEOUT_MS": "NaN",
            "CLAWCODEX_DAEMON_SANDBOX": "FALSE",
        }
    )
    assert cfg["capacity"] == 4
    assert cfg["timeout_ms"] == 30_000
    assert cfg["sandbox"] is False
