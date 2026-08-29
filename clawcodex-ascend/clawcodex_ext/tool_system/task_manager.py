#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional


class TaskManagerClosedError(RuntimeError):
    """Raised when work is submitted after session shutdown has begun."""


@dataclass(frozen=True)
class ManagedTask:
    task_id: str
    name: str
    started_at: float
    stop_event: threading.Event
    thread: threading.Thread
    on_stop: Callable[[], None] | None = None


class TaskManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, ManagedTask] = {}
        self._stop_notified: set[str] = set()
        self._closed = False

    def start(
        self,
        *,
        name: str,
        target: Callable[[threading.Event], None],
        on_stop: Callable[[], None] | None = None,
    ) -> ManagedTask:
        task_id = str(uuid.uuid4())
        stop_event = threading.Event()

        def runner() -> None:
            try:
                target(stop_event)
            finally:
                with self._lock:
                    self._tasks.pop(task_id, None)
                    self._stop_notified.discard(task_id)

        thread = threading.Thread(target=runner, name=f"tool-task:{name}:{task_id}", daemon=True)
        task = ManagedTask(
            task_id=task_id,
            name=name,
            started_at=time.time(),
            stop_event=stop_event,
            thread=thread,
            on_stop=on_stop,
        )
        with self._lock:
            if self._closed:
                raise TaskManagerClosedError("task manager is shutting down")
            self._tasks[task_id] = task
            try:
                # Start while holding the same lock used by shutdown so
                # shutdown cannot miss a worker accepted concurrently.
                thread.start()
            except BaseException:
                self._tasks.pop(task_id, None)
                raise
        return task

    def stop(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            task.stop_event.set()
            callback = None
            if task_id not in self._stop_notified:
                self._stop_notified.add(task_id)
                callback = task.on_stop
        if callback is not None:
            try:
                callback()
            except Exception:  # nosec
                # Cancellation is best-effort. The stop flag remains set even
                # when a task-specific callback fails.
                pass
        return True

    def shutdown(self, *, timeout: float = 2.0) -> bool:
        """Stop every managed task and wait briefly for worker convergence.

        The manager owns daemon threads, but relying on interpreter teardown
        leaves subprocesses and provider streams alive while an interactive
        session is trying to exit. Signal all tasks first, then join against a
        single bounded deadline. Returns ``True`` when every worker stopped.
        """

        with self._lock:
            self._closed = True
            tasks = list(self._tasks.values())
        for task in tasks:
            self.stop(task.task_id)

        deadline = time.monotonic() + max(0.0, timeout)
        current = threading.current_thread()
        for task in tasks:
            if task.thread is current:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            task.thread.join(remaining)
        return not self.list()

    def get(self, task_id: str) -> Optional[ManagedTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[ManagedTask]:
        with self._lock:
            return list(self._tasks.values())


__all__ = ["ManagedTask", "TaskManager", "TaskManagerClosedError"]
