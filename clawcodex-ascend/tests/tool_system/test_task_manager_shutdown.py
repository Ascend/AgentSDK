from __future__ import annotations


# pylint: disable=E0611
import threading
import time

import pytest

from clawcodex_ext.tool_system.task_manager import TaskManager, TaskManagerClosedError


def test_shutdown_signals_callbacks_and_joins_workers() -> None:
    manager = TaskManager()
    callback_called = threading.Event()
    worker_stopped = threading.Event()

    def target(stop_event: threading.Event) -> None:
        while not stop_event.wait(0.01):
            pass
        worker_stopped.set()

    manager.start(
        name="shutdown-test",
        target=target,
        on_stop=callback_called.set,
    )

    assert manager.shutdown(timeout=1.0) is True
    assert callback_called.is_set()
    assert worker_stopped.is_set()
    assert manager.list() == []


def test_shutdown_is_bounded_when_worker_does_not_cooperate() -> None:
    manager = TaskManager()
    release = threading.Event()

    def target(_stop_event: threading.Event) -> None:
        release.wait(2.0)

    manager.start(name="uncooperative-test", target=target)
    started = time.monotonic()
    try:
        assert manager.shutdown(timeout=0.05) is False
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        deadline = time.monotonic() + 1.0
        while manager.list() and time.monotonic() < deadline:
            time.sleep(0.01)


def test_stop_callback_runs_once_and_shutdown_rejects_new_work() -> None:
    manager = TaskManager()
    release = threading.Event()
    callback_count = 0

    def target(_stop_event: threading.Event) -> None:
        release.wait(1.0)

    def on_stop() -> None:
        nonlocal callback_count
        callback_count += 1

    task = manager.start(name="one-shot-stop", target=target, on_stop=on_stop)
    try:
        assert manager.stop(task.task_id) is True
        assert manager.stop(task.task_id) is True
        assert manager.shutdown(timeout=0.01) is False
        assert callback_count == 1
        with pytest.raises(TaskManagerClosedError):
            manager.start(name="too-late", target=target)
    finally:
        release.set()
