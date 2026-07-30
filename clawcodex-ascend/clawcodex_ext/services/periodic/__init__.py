"""Reusable periodic daemon base class.

This package provides a single, well-tested implementation of the
"background daemon thread that wakes up every N seconds and runs a
hook" pattern that several services need:

* :class:`PeriodicDaemon` — owns the daemon thread, the stop event,
  the join deadline, and the per-tick hook. Subclasses override
  :meth:`PeriodicDaemon._do_tick`.

Existing callers in the codebase that already implement this pattern
manually (notably :mod:`src.services.swarm.mailbox_poller` and the
:class:`TickScheduler` in :mod:`src.services.kairos.scheduler`) can
either inherit from :class:`PeriodicDaemon` directly or stay as-is
and gradually migrate. The base class does not force a particular
scheduling model — TickScheduler adds pause / resume / jitter on top
without disturbing the lifecycle primitives here.
"""

from __future__ import annotations

import logging
import threading

from typing_extensions import Self

logger = logging.getLogger(__name__)


class PeriodicDaemon:
    """Base class for a named periodic daemon thread.

    Lifecycle:

    * :meth:`start` is **idempotent** — calling it twice is a no-op
      while the daemon is running. This mirrors the behaviour of
      :func:`src.services.swarm.mailbox_poller.start_mailbox_poller`.
    * :meth:`stop` sets the stop event, joins the thread up to
      ``timeout`` seconds, and clears the thread reference.
    * Subclasses override :meth:`_do_tick` to do per-tick work; the
      default raises ``NotImplementedError``.
    * Exceptions raised by ``_do_tick`` are logged but do not stop
      the daemon — a flaky tick should not take down the cadence.

    The daemon thread is a ``daemon=True`` thread, so a process exit
    will not be blocked by a still-running daemon. Callers that need
    graceful shutdown should still call :meth:`stop` explicitly.
    """

    def __init__(
        self,
        *,
        name: str,
        tick_seconds: float,
        logger_obj: logging.Logger | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(tick_seconds, (int, float)) or tick_seconds <= 0:
            raise ValueError(f"tick_seconds must be a positive number (got {tick_seconds!r})")
        self._name = name
        self._tick_seconds = float(tick_seconds)
        self._logger = logger_obj or logger
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def tick_seconds(self) -> float:
        return self._tick_seconds

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the daemon if it is not already running.

        Returns ``True`` if the daemon was started by this call,
        ``False`` if it was already running (idempotent).
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name=self._name,
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, *, timeout: float = 2.0) -> bool:
        """Signal the daemon to exit and join the thread.

        Returns ``True`` if the thread joined within ``timeout`` or if
        no thread was active. Returns ``False`` if the join timed out.
        Safe to call when no daemon is running.
        """
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        self._stop_event.set()
        thread.join(timeout=timeout)
        joined = not thread.is_alive()
        if not joined:
            self._logger.warning(
                "periodic daemon %r did not stop within %s seconds",
                self._name,
                timeout,
            )
        with self._lock:
            self._thread = None
        return joined

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _do_tick(self) -> None:
        """Override this in a subclass to perform per-tick work."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self._tick_seconds):
                return
            try:
                self._do_tick()
            except Exception:
                self._logger.exception(
                    "periodic daemon %r tick raised; continuing",
                    self._name,
                )


__all__ = [
    "PeriodicDaemon",
]
