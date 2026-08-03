"""Cron scheduler lifecycle independent of frontends (F-22-G1/G5/G8/G7)."""

from __future__ import annotations

import asyncio
import atexit
import logging
import signal
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, MutableMapping

from .lock import CronTaskLock
from .models import CronJitterConfig, CronTask, load_jitter_config
from .notifications import build_missed_task_notification
from .runs import CronRun, create_queued_run_for_task, finalize_cron_run
from .tasks import (
    find_due_tasks,
    find_missed_tasks,
    mark_cron_tasks_fired,
    now_ms,
    prune_expired_recurring_tasks,
    read_cron_tasks_cached,
    remove_missed_tasks,
)

_log = logging.getLogger(__name__)

# Optional event hook signatures for F-22-G7 (analytics reservation).
# Defaults are no-ops; callers (REPL, daemon) can pass observability sinks.
FireEventHook = Callable[[dict], None]
MissedEventHook = Callable[[dict], None]
ExpiredEventHook = Callable[[dict], None]


def _noop_event(_payload: dict) -> None:  # pragma: no cover - default hook
    return None


def _loop_thread_id(loop: asyncio.AbstractEventLoop) -> int | None:
    """Best-effort: return the thread id that the given loop runs on."""
    # asyncio loops store the running thread on the internal ``_thread_id``
    # attribute (CPython). Fall back to None when unavailable.
    return getattr(loop, "_thread_id", None)


@dataclass
class CronScheduler:
    workspace_root: Path
    on_fire: Callable[[str], None]
    on_fire_task: Callable[[CronTask, CronRun], None] | None = None
    on_missed: Callable[[list[CronTask], str], None] | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    check_interval_seconds: float = 1.0
    # F-22-G1: polled before each tick; when True, scheduler stops firing.
    is_killed: Callable[[], bool] | None = None
    # F-22-G2: per-tick config loader. When None, falls back to
    # ``load_jitter_config(workspace_root)`` so live edits to
    # ``.clawcodex/cron/jitter_config.json`` (or legacy ``.claude/cron_jitter_config.json``)
    # and ``CLAWCODEX_CRON_*`` env
    # vars take effect on the next ``check_once`` without restarting the
    # CLI. Pass an explicit callable to inject a GrowthBook-style
    # remote-config source.
    load_jitter_config: Callable[[], CronJitterConfig] | None = None
    # F-22-G7: optional event hooks for analytics. No-op by default.
    on_fire_event: FireEventHook = _noop_event
    on_missed_event: MissedEventHook = _noop_event
    on_expired_event: ExpiredEventHook = _noop_event
    # Phase A-2: in-memory store for durable=False tasks. When provided,
    # ``check_once`` will scan these alongside file-backed tasks. The
    # store is process-local; only the lock-owning process fires session
    # tasks (see ``check_once`` lock gate in Phase B-2).
    session_store: MutableMapping[str, CronTask | dict] | None = None
    # F-22-G9: optional daemon-mode overrides. When provided, the
    # scheduler operates independently of the bootstrap session state:
    #   - dir_override — overrides workspace_root for all file I/O
    #     (tasks, locks, jitter config, run storage).
    #   - lock_identity — overrides session_id as the lock owner
    #     identity, enabling headless/daemon schedulers to coexist
    #     without sharing a session.
    # Both default to None, preserving the existing bootstrap-dependent
    # behaviour.
    dir_override: Path | None = None
    lock_identity: str | None = None
    is_loading: Callable[[], bool] | None = None
    assistant_mode: bool = False
    # F-22-F: agent ownership — when set, only fire tasks belonging to this agent
    # or global tasks (agent_id=None). None means no filtering (all agents).
    agent_id: str | None = None
    # F-22-F-4: optional hook fired when the owning agent is reported
    # to have exited (crash / SIGKILL / clean shutdown) via
    # :meth:`notify_owner_exited`. The teammate subsystem is not yet
    # wired, so this stays as an optional callable the caller may set
    # once teammate lifecycle is integrated.
    on_owner_exited: Callable[[str], None] | None = None
    # F-22-F-5: optional provider returning the set of currently active
    # agent ids. When set, :meth:`cleanup_orphaned_tasks` uses it to
    # find tasks whose owning agent is no longer active. The scheduler
    # does NOT poll this provider automatically — teammate subsystem
    # integration will wire the call site once :class:`TeammateManager`
    # is connected. Until then, callers can invoke
    # :meth:`cleanup_orphaned_tasks` manually from outside.
    active_agents_provider: Callable[[], set[str]] | None = None

    _thread: threading.Thread | None = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: CronTaskLock | None = field(default=None, init=False)
    # F-22-G2: cache the most recent jitter config so downstream callers
    # (e.g. ``prune_expired_recurring_tasks``) can pick up the live
    # ``recurring_max_age_ms`` without re-reading the loader twice.
    _last_jitter_config: CronJitterConfig | None = field(default=None, init=False)
    # F-22-G8: thread-safe in-flight set to prevent double-fire on async
    # mark_fired / remove windows.
    _in_flight: set[str] = field(default_factory=set, init=False)
    _in_flight_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    # F-22-G2: per-tick I/O throttle — jitter config and expired-task
    # cleanup only execute every _THROTTLE_INTERVAL ticks instead of
    # every second, cutting ~80% of disk reads when no schedule change.
    _THROTTLE_INTERVAL: int = 60
    _jitter_tick_counter: int = field(default=0, init=False)
    _prune_tick_counter: int = field(default=0, init=False)

    # F-22-G5: track whether we registered atexit/signal cleanup so stop()
    # is idempotent.
    _atexit_registered: bool = field(default=False, init=False)
    _signal_registered: bool = field(default=False, init=False)
    _previous_sigterm: signal._HANDLER | None = field(default=None, init=False)
    _previous_sigint: signal._HANDLER | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """F-22-G9: apply daemon-mode overrides.

        When ``dir_override`` is set, replace ``workspace_root`` so that
        *all* downstream I/O (tasks, locks, jitter config, runs) uses the
        overridden path without requiring per-method changes.
        """
        if self.dir_override is not None:
            self.workspace_root = self.dir_override

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        lock_owner = self.lock_identity if self.lock_identity is not None else self.session_id
        lock = CronTaskLock(self.workspace_root, lock_owner)
        if not lock.acquire():
            return False
        self._lock = lock
        self._stop_event.clear()
        self._register_cleanup_hooks()
        self._thread = threading.Thread(target=self._run_loop, name="clawcodex-cron-scheduler", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                _log.warning(
                    "cron scheduler thread did not stop within 2s; holding lock "
                    "to avoid dual-scheduler window (thread may still fire callbacks)"
                )
                self._unregister_cleanup_hooks()
                return
        if self._lock:
            self._lock.release()
            self._lock = None
        self._unregister_cleanup_hooks()

    def load(self) -> list[CronTask]:
        return read_cron_tasks_cached(self.workspace_root)

    def is_disabled(self) -> bool:
        """F-22-G1: True if the kill switch is engaged this tick."""
        if self.is_killed is None:
            return False
        try:
            return bool(self.is_killed())
        except Exception:  # pragma: no cover - defensive
            return False

    def _is_loading_gate(self) -> bool:
        """True if agent is busy AND assistant_mode is off.

        Mirrors TS ``check()``: ``if isLoading() && !assistantMode: return``.
        When the gate is active, the entire tick is skipped — no tasks are
        evaluated, no fires occur. Overdue tasks fire on the first tick
        where the gate opens (their ``next_fire_at`` is already in the past).
        """
        if self.assistant_mode:
            return False
        if self.is_loading is None:
            return False
        try:
            return bool(self.is_loading())
        except Exception:  # pragma: no cover - defensive
            return False

    def _agent_owned_only(self, tasks: list[CronTask]) -> list[CronTask]:
        """F-22-F-2: filter due tasks by agent ownership.

        Returns the input unchanged when ``self.agent_id`` is ``None``
        (single-agent mode, no ownership tracking). Otherwise keeps global
        tasks (``agent_id=None``) plus tasks whose ``agent_id`` matches
        the scheduler's identity. Extracted from :meth:`check_once` so the
        ownership contract is testable without driving the full scheduler
        loop.
        """
        if self.agent_id is None:
            return list(tasks)
        return [task for task in tasks if task.agent_id is None or task.agent_id == self.agent_id]

    def check_once(self, at_ms: int | None = None) -> list[CronTask]:
        if self.is_disabled():
            return []

        if self._is_loading_gate():
            return []

        timestamp = at_ms if at_ms is not None else now_ms()

        # F-22-G2: refresh jitter config on every _THROTTLE_INTERVAL-th
        # tick instead of every second.  Live edits to cron_jitter_config
        # take effect within ~60 s instead of immediately — an acceptable
        # trade-off for cutting ~98 % of background I/O when idle.
        if self._jitter_tick_counter % self._THROTTLE_INTERVAL == 0:
            try:
                self._last_jitter_config = self._resolve_jitter_config()
            except Exception as exc:  # pragma: no cover - defensive
                _log.warning("jitter config reload failed: %s; using cached", exc)
                if self._last_jitter_config is None:
                    self._last_jitter_config = load_jitter_config(self.workspace_root)
        self._jitter_tick_counter += 1

        if self._prune_tick_counter % self._THROTTLE_INTERVAL == 0:
            prune_expired_recurring_tasks(
                self.workspace_root,
                timestamp,
                max_age_ms=self._last_jitter_config.recurring_max_age_ms,
                session_store=self.session_store,
            )
        self._prune_tick_counter += 1
        try:
            due = find_due_tasks(self.workspace_root, timestamp, session_store=self.session_store)
        except OSError as exc:
            _log.warning("check_once: find_due_tasks failed with OSError: %s; skipping tick", exc)
            return []
        if not due:
            return []
        # Dedupe by task.id: if the same id lives in both file and session
        # (defensive), fire it once. The earlier entry wins (session is
        # scanned first; file is fallback).
        seen_ids: set[str] = set()
        deduped_due: list[CronTask] = []
        for task in due:
            if task.id in seen_ids:
                continue
            seen_ids.add(task.id)
            deduped_due.append(task)
        due = deduped_due
        # F-22-F: agent ownership filtering — only fire tasks belonging to the
        # current agent or global tasks (agent_id=None). Delegated to
        # :meth:`_agent_owned_only` so the contract is testable.
        due = self._agent_owned_only(due)
        if not due:
            return []
        # Phase B-2: session-only tasks live in this process's memory.
        # Another process cannot see them, so only the lock-owning
        # process should fire them — otherwise two instances holding
        # the workspace scheduler lock could each fire the same task.
        owns_scheduler_lock = self._lock is not None and self._lock.is_acquired
        fired: list[CronTask] = []
        for task in due:
            if self._in_flight_contains(task.id):
                continue
            if not task.durable and not owns_scheduler_lock:
                # Defer to the lock-owning process; do not queue.
                continue
            self._in_flight_add(task.id)
            try:
                run = create_queued_run_for_task(self.workspace_root, task, queued_at=timestamp)
                if run is None:
                    continue
                fired.append(task)
                # F-22-G7: fire event.
                self.on_fire_event(
                    {
                        "type": "fire",
                        "task_id": task.id,
                        "recurring": task.recurring,
                        "fire_at": timestamp,
                    }
                )
                try:
                    if self.on_fire_task is not None:
                        self.on_fire_task(task, run)
                    else:
                        self.on_fire(task.prompt)
                except Exception as exc:  # pragma: no cover - defensive
                    _log.exception("cron fire callback failed for task %s", task.id)
                    finalize_cron_run(
                        self.workspace_root,
                        run.id,
                        "failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
            finally:
                self._in_flight_remove(task.id)
        mark_cron_tasks_fired(
            self.workspace_root,
            fired,
            timestamp,
            session_store=self.session_store,
        )
        return fired

    def _resolve_jitter_config(self) -> CronJitterConfig:
        if self.load_jitter_config is not None:
            return self.load_jitter_config()
        return load_jitter_config(self.workspace_root)

    def get_jitter_config(self) -> CronJitterConfig:
        """Return the most recently loaded jitter config (refreshes if needed)."""
        if self._last_jitter_config is None:
            self._last_jitter_config = self._resolve_jitter_config()
        return self._last_jitter_config

    def notify_missed_once(self, at_ms: int | None = None) -> list[CronTask]:
        if self.is_disabled():
            return []
        missed = find_missed_tasks(self.workspace_root, at_ms, session_store=self.session_store)
        if missed:
            remove_missed_tasks(self.workspace_root, missed, session_store=self.session_store)
            # F-22-G7: missed event.
            self.on_missed_event(
                {
                    "type": "missed",
                    "count": len(missed),
                    "task_ids": [t.id for t in missed],
                }
            )
            if self.on_missed is not None:
                self.on_missed(missed, build_missed_task_notification(missed))
        return missed

    # ---- F-22-F-4 / F-22-F-5 stubs (awaiting teammate subsystem) ----
    def notify_owner_exited(self, owner_agent_id: str) -> list[str]:
        """F-22-F-4 stub: notify the scheduler that an owning agent exited.

        Currently a no-op with two side effects:

        1. If ``on_owner_exited`` is set, invoke it with the agent id.
           Callers (typically the future teammate subsystem) get a
           single hook point to track agent lifecycle without coupling
           to scheduler internals.
        2. Log the exit at debug level. No runs are auto-finalized
           here yet — once :class:`TeammateManager` exposes an
           ``active_agents`` API, this method will additionally walk
           any in-flight cron runs owned by ``owner_agent_id`` and
           finalize them as ``failed`` with reason ``owner_exited``.

        Returns the list of run ids that were finalized as failed
        (currently always empty).
        """
        if not owner_agent_id:
            return []
        _log.debug("notify_owner_exited(agent=%s) — teammate integration pending", owner_agent_id)
        if self.on_owner_exited is not None:
            try:
                self.on_owner_exited(owner_agent_id)
            except Exception:  # pragma: no cover - defensive
                _log.exception("on_owner_exited hook raised for agent %s", owner_agent_id)
        return []

    def cleanup_orphaned_tasks(self) -> list[CronTask]:
        """F-22-F-5 stub: find tasks whose owning agent is no longer active.

        Requires ``active_agents_provider`` to be set; otherwise the
        caller has no way to know which agents are alive. The scheduler
        does NOT poll this provider automatically — that wiring will
        land once the teammate subsystem (``TeammateManager``) is
        connected. Until then, callers (REPL/headless shutdown,
        orchestrator drain) can invoke this method directly when they
        have a fresh snapshot of the active agent set.

        Returns the list of orphaned tasks. The caller decides whether
        to delete them — this method is read-only.
        """
        from .tasks import cleanup_orphaned_tasks as _cleanup

        if self.active_agents_provider is None:
            return []
        try:
            active = self.active_agents_provider()
        except Exception:  # pragma: no cover - defensive
            _log.exception("active_agents_provider raised; skipping cleanup")
            return []
        return _cleanup(
            self.workspace_root,
            active,
            session_store=self.session_store,
        )

    def get_next_fire_time(self) -> int | None:
        if self.is_disabled():
            return None
        values = [
            task.next_fire_at for task in read_cron_tasks_cached(self.workspace_root) if task.next_fire_at is not None
        ]
        return min(values) if values else None

    # ---- F-22-G8 in-flight helpers ----
    def _in_flight_contains(self, task_id: str) -> bool:
        with self._in_flight_lock:
            return task_id in self._in_flight

    def _in_flight_add(self, task_id: str) -> None:
        with self._in_flight_lock:
            self._in_flight.add(task_id)

    def _in_flight_remove(self, task_id: str) -> None:
        with self._in_flight_lock:
            self._in_flight.discard(task_id)

    # ---- F-22-G5 cleanup registry ----
    def _register_cleanup_hooks(self) -> None:
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True
        if not self._signal_registered and threading.current_thread() is threading.main_thread():
            try:
                self._previous_sigterm = signal.signal(signal.SIGTERM, self._signal_cleanup)
                self._previous_sigint = signal.signal(signal.SIGINT, self._signal_cleanup)
                self._signal_registered = True
            except (ValueError, OSError):  # not main thread / unsupported
                pass

    def _unregister_cleanup_hooks(self) -> None:
        if self._atexit_registered:
            try:
                atexit.unregister(self.stop)
            except Exception:  # pragma: no cover  # nosec B110 - best-effort atexit cleanup
                pass
            self._atexit_registered = False
        if self._signal_registered:
            try:
                signal.signal(signal.SIGTERM, self._previous_sigterm or signal.SIG_DFL)
                signal.signal(signal.SIGINT, self._previous_sigint or signal.SIG_DFL)
            except (ValueError, OSError):  # pragma: no cover
                pass
            self._signal_registered = False

    def _signal_cleanup(self, signum, frame):  # pragma: no cover - signal path
        self.stop()
        prev = self._previous_sigterm if signum == signal.SIGTERM else self._previous_sigint
        if prev and prev not in (signal.SIG_DFL, signal.SIG_IGN, None):
            try:
                prev(signum, frame)
            except Exception:  # pragma: no cover  # nosec B110 - best-effort signal cleanup
                pass

    def _run_loop(self) -> None:
        self.notify_missed_once()
        while not self._stop_event.is_set():
            if self.is_disabled():
                self._stop_event.wait(1.0)
                continue
            try:
                self.check_once()
            except Exception:
                _log.exception("cron scheduler tick failed; continuing after brief sleep")
                self._stop_event.wait(1.0)
                continue
            # Dynamic sleep: wait until the next known fire time (capped
            # at 60 s) instead of a fixed 1 s.  When no task is due
            # within 60 s the scheduler stays quiet, cutting idle I/O
            # by ~98 % without missing any deadline.
            next_fire = self.get_next_fire_time()
            now = now_ms()
            if next_fire is not None and next_fire > now:
                wait = min((next_fire - now) / 1000.0, 60.0)
            else:
                wait = 1.0
            self._stop_event.wait(wait)


class AsyncCronScheduler(CronScheduler):
    """asyncio-native variant of :class:`CronScheduler`.

    Uses an `asyncio.Task` instead of a `threading.Thread` for the run
    loop, eliminating the thread→coroutine bridge needed by Orchestrator
    callers.  Methods ``start()``, ``stop()`` and ``_run_loop()`` are
    overridden; all other logic (``check_once``, jitter, fire hooks) is
    inherited unchanged.

    Start/stop are **not** thread-safe — call only from the asyncio event
    loop thread.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._async_task: asyncio.Task | None = None

    def start(self) -> bool:
        """Start in a non-blocking manner (returns immediately)."""

        if self._async_task is not None and not self._async_task.done():
            return True

        # Acquire the filesystem lock synchronously; it's fast and
        # needs no I/O loop.
        lock_owner = self.lock_identity if self.lock_identity is not None else self.session_id
        lock = CronTaskLock(self.workspace_root, lock_owner)
        if not lock.acquire():
            return False
        self._lock = lock
        self._stop_event.clear()
        self._register_cleanup_hooks()
        self._async_task = asyncio.create_task(self._run_loop_async())
        return True

    async def stop_async(self) -> None:
        """Coroutine that cancels the run loop and releases the lock."""
        self._stop_event.set()
        if self._async_task is not None and not self._async_task.done():
            self._async_task.cancel()
            try:
                await self._async_task
            except asyncio.CancelledError:
                pass
            self._async_task = None
        if self._lock:
            self._lock.release()
            self._lock = None

    def stop(self) -> None:
        """Synchronous stop — submits the coroutine and waits briefly.

        When called from inside the running event loop thread, we cannot
        block on ``future.result()`` (that would deadlock the loop).
        Instead we schedule ``stop_async`` as a fire-and-forget task and
        return immediately. When called from outside the loop thread,
        ``run_coroutine_threadsafe`` + ``future.result()`` is safe.

        When there is no running loop at all (e.g. signal handler / atexit
        context), we set the stop event and release the lock synchronously
        so resources are not leaked.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._stop_event.set()
            if self._async_task is not None and not self._async_task.done():
                self._async_task.cancel()
                self._async_task = None
            if self._lock:
                self._lock.release()
                self._lock = None
            return
        if loop.is_running():
            if threading.current_thread() is threading.main_thread() and _loop_thread_id(loop) == threading.get_ident():
                asyncio.ensure_future(self.stop_async())
            else:
                future = asyncio.run_coroutine_threadsafe(self.stop_async(), loop)
                try:
                    future.result(timeout=2.0)
                except Exception:  # nosec B110 - ignore cross-thread stop timeout
                    pass
        else:
            loop.run_until_complete(self.stop_async())

    async def _run_loop_async(self) -> None:
        """Async run loop with same dynamic sleep as the threaded variant."""
        self.notify_missed_once()
        while not self._stop_event.is_set():
            if self.is_disabled():
                await asyncio.sleep(1.0)
                continue
            try:
                self.check_once()
            except Exception:
                _log.exception("cron async scheduler tick failed; continuing after brief sleep")
                await asyncio.sleep(1.0)
                continue
            next_fire = self.get_next_fire_time()
            now = now_ms()
            if next_fire is not None and next_fire > now:
                wait = min((next_fire - now) / 1000.0, 60.0)
            else:
                wait = 1.0
            await asyncio.sleep(wait)
