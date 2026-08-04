#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tick scheduler: periodic fire-and-deliver.

The :class:`TickScheduler` runs a background daemon thread that fires
:class:`TickEvent`s at a configured interval. Callers register one or
more callbacks; each callback is invoked synchronously per tick, with
exceptions caught and logged (a failing subscriber never takes down the
scheduler). The scheduler supports pause / resume, jitter, and a clean
shutdown path that joins the thread within a configurable deadline.

Threading model:

* Inherits from :class:`PeriodicDaemon` so the daemon thread, stop
  event, idempotent :meth:`start` / :meth:`stop`, and context manager
  are shared with :mod:`src.services.swarm.mailbox_poller`. TickScheduler
  only adds the *cadence* — drift-free fire, symmetric jitter, pause
  flag, callback fan-out, and tick-number accounting.
* Wall-clock time is read from :func:`time.monotonic` for scheduling
  math and :func:`time.time` for event payloads, so drift in the system
  clock does not accumulate.
* Pause is a soft flag — the thread keeps running but skips callback
  delivery. This keeps the wake-up cadence stable when the host is
  briefly busy.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable

from ..periodic import PeriodicDaemon
from .exceptions import SchedulerStateError
from .models import TickConfig, TickEvent

logger = logging.getLogger(__name__)

TickCallback = Callable[[TickEvent], None]


class TickScheduler(PeriodicDaemon):  # pylint: disable=abstract-method
    """Periodic tick scheduler with pause / resume / jitter.

    Lifecycle is inherited from :class:`PeriodicDaemon`:

    * :meth:`start` is idempotent — calling twice returns ``False`` on
      the second call instead of raising.
    * :meth:`stop` joins the thread up to ``timeout`` seconds and
      returns whether the join completed.
    * :meth:`__enter__` / :meth:`__exit__` mirror ``start`` / ``stop``.

    Subclass additions:

    * :meth:`subscribe` / :meth:`unsubscribe` register tick callbacks.
    * :meth:`pause` / :meth:`resume` halt / restore callback delivery.
    * :meth:`tick_count` exposes the monotonic fire counter.
    """

    def __init__(
        self,
        config: TickConfig,
        *,
        time_func: Callable[[], float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if not isinstance(config, TickConfig):
            raise TypeError("TickScheduler requires a TickConfig")
        super().__init__(
            name=f"kairos-tick-{config.id}",
            tick_seconds=config.interval_seconds,
            logger_obj=logger,
        )
        self._config = config
        self._time = time_func or time.monotonic
        self._wall = time.time
        self._rng = rng or random.Random()  # nosec B311
        self._callbacks: list[TickCallback] = []
        self._paused = threading.Event()  # set == paused
        self._tick_number = 0
        # Captured on start() (or lazily on first tick) so _fire() can
        # translate monotonic scheduling time to wall-clock timestamps.
        self._start_monotonic: float | None = None
        self._start_wall: float | None = None
        if config.enabled:
            self.start()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> TickConfig:
        return self._config

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def tick_count(self) -> int:
        with self._lock:
            return self._tick_number

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def subscribe(self, callback: TickCallback) -> None:
        """Register a tick callback. May be called before or after start."""
        if not callable(callback):
            raise TypeError("subscribe() expects a callable")
        with self._lock:
            self._callbacks.append(callback)

    def unsubscribe(self, callback: TickCallback) -> None:
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError as exc:
                raise SchedulerStateError("callback not registered") from exc

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the scheduler. Idempotent — second call returns ``False``."""
        # Capture scheduling anchors *before* the daemon thread is
        # launched. Set under PeriodicDaemon's lock so we never race the
        # thread creation branch in super().start().
        with self._lock:
            already = self._thread is not None and self._thread.is_alive()
            if already:
                return False
            self._stop_event.clear()
            self._paused.clear()
            self._start_monotonic = self._time()
            self._start_wall = self._wall()
            self._thread = threading.Thread(
                target=self._loop,
                name=self._name,
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, *, timeout: float | None = 5.0) -> bool:
        """Signal shutdown and join the thread."""
        return super().stop(timeout=timeout if timeout is not None else 5.0)

    def pause(self) -> None:
        """Halt callback delivery without stopping the thread."""
        self._paused.set()

    def resume(self) -> None:
        """Resume callback delivery."""
        self._paused.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Drift-free loop: next fire time advances from the *scheduled*
        cadence, not from when the previous tick actually completed.

        Overrides :meth:`PeriodicDaemon._loop` to (a) advance by
        ``interval`` rather than measuring wall-clock drift, and (b)
        apply symmetric jitter around the scheduled fire time.
        """
        interval = self._tick_seconds
        next_fire = self._time() + interval
        while not self._stop_event.is_set():
            now = self._time()
            wait = next_fire - now
            if wait > 0:
                # Sleep in small slices so stop() is responsive.
                if self._stop_event.wait(timeout=min(wait, 0.5)):
                    return
                continue
            jitter_applied = 0.0
            if self._config.jitter_fraction > 0:
                # Symmetric jitter in [-interval*frac, +interval*frac].
                jitter_applied = self._rng.uniform(
                    -interval * self._config.jitter_fraction,
                    interval * self._config.jitter_fraction,
                )
            scheduled_monotonic = next_fire
            actual_monotonic = scheduled_monotonic + jitter_applied
            try:
                self._fire(scheduled_monotonic, actual_monotonic, jitter_applied)
            except Exception:
                self._logger.exception(
                    "tick scheduler %r fire raised; continuing",
                    self._name,
                )
            # Advance from the *scheduled* cadence so jitter does not
            # accumulate drift into the next interval.
            next_fire = scheduled_monotonic + interval

    def _fire(
        self,
        scheduled_monotonic: float,
        actual_monotonic: float,
        jitter_applied: float,
    ) -> None:
        with self._lock:
            self._tick_number += 1
            tick_number = self._tick_number
            callbacks = list(self._callbacks)
            start_wall = self._start_wall
            start_monotonic = self._start_monotonic
        if self._paused.is_set():
            return
        if start_wall is None or start_monotonic is None:
            return
        scheduled_at = start_wall + (scheduled_monotonic - start_monotonic)
        actual_at = start_wall + (actual_monotonic - start_monotonic)
        event = TickEvent(
            scheduler_id=self._config.id,
            tick_number=tick_number,
            scheduled_at=scheduled_at,
            actual_at=actual_at,
            jitter_applied=jitter_applied,
        )
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception(
                    "tick callback %r raised; continuing",
                    getattr(cb, "__qualname__", repr(cb)),
                )


__all__ = [
    "TickCallback",
    "TickScheduler",
]
