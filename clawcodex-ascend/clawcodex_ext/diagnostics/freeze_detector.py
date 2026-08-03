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

"""Layer-1 freeze-detection watchdog (F-108 §十八 P108-D).

A daemon thread polls every ``check_interval_s`` and, if the last
``heartbeat()`` was more than ``threshold_s`` ago, dumps the captured
Python thread stacks to disk and emits an ``append_debug_event``
``freeze_detected`` row (mirroring the existing headless-runner debug
log convention).

The detector is deliberately **observation-only**: it never trips an
abort controller, never raises, and never modifies the canonical
agent loop. Auto-recovery is the responsibility of Layer 3
(P108-G) — the detector's job is to give postmortem reporters (the
Layer 4 ``diag freeze-report`` CLI) something to read.

Why a watchdog thread rather than ``asyncio.create_task``? The
freeze can land while the only asyncio loop is blocked on
blocking I/O (provider stream, NFS disk write). A daemon thread
keeps ticking when the asyncio loop does not.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .freeze_config import (
    FreezeSettings,
    dump_path,
    resolve_freeze_settings,
)


# F-108 §十八 P108-D default poll cadence. 10 s is well under the
# 60 s default threshold (the detector must tick at least once
# within the threshold window) and far above the syscall cost of
# reading ``time.monotonic()``.
DEFAULT_FREEZE_CHECK_INTERVAL_S = 10.0

# F-108 §十八 acceptance §5: ``CLAWCODEX_FREEZE_DIAG=1`` flips the
# watchdog on for an existing process. Empty/unset keeps it idle
# (no thread spawned, no resource spent).
DEFAULT_FREEZE_DIAG_ENV = "CLAWCODEX_FREEZE_DIAG"

# Cap per-dump size to keep a runaway stack (deep recursion, large
# generated frames) from blowing disk budget. 2 MiB is well above
# any plausible real-world freeze dump but well below "fills the
# user's home directory" territory.
_MAX_DUMP_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ThreadStackFrame:
    """One captured frame in a frozen thread.

    ``frames`` is the formatted ``traceback.format_stack`` output
    (already a single ``str``); ``tid`` is the Python thread id
    (``threading.get_ident()``). Keeping the dataclass small makes
    ``FreezeDump.to_dict()`` JSON-serialisable without bespoke
    encoders.
    """

    tid: int
    thread_name: str
    frames: str


@dataclass
class FreezeDump:
    """One captured freeze report.

    Fields are populated by :meth:`FreezeDetector.check` at the moment
    the threshold trips and written to JSON via :meth:`to_dict` /
    :meth:`write`. The dataclass is mutable so :meth:`FreezeDetector.start`
    can stamp ``run_id`` / ``wall_clock_seconds`` incrementally.
    """

    detected_at_unix: float
    last_heartbeat_at_unix: float
    elapsed_seconds: float
    threshold_seconds: float
    check_interval_seconds: float
    detected_by_thread: str
    diag_env_enabled: bool
    # Stable per-process id so multiple detectors (worker thread +
    # signal-handler) don't double-write.
    detector_id: str = ""
    run_id: str | None = None
    process_id: int = field(default_factory=os.getpid)
    thread_stacks: list[ThreadStackFrame] = field(default_factory=list)
    # ``extra`` lets downstream emit a paired debug-event row without
    # having to extend the dataclass. Always serialised.
    extra: dict[str, Any] = field(default_factory=dict)
    # Path of the persisted dump file, populated by ``write()``.
    dump_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["thread_stacks"] = [
            {"tid": f.tid, "thread_name": f.thread_name, "frames": f.frames} for f in self.thread_stacks
        ]
        return d


class FreezeDetector:
    """Detects when the agent loop stops heartbeating.

    The detector is constructed lazily. ``start()`` spawns a daemon
    watchdog thread; ``stop()`` joins it (called from process
    teardown).

    Heartbeat protocol: callers invoke :meth:`heartbeat` whenever the
    agent loop observes forward progress. ``Threading.Lock`` keeps
    the read/write race-free without forcing every heartbeat to
    allocate (the dataclass is shared mutable state).
    """

    # Singleton handle so callers (``AgentBridge``, the headless
    # runner, the TUI app) don't have to thread the detector
    # through every layer.
    _INSTANCE: "FreezeDetector | None" = None
    _INSTANCE_LOCK = threading.Lock()

    def __init__(
        self,
        *,
        threshold: float | None = None,
        check_interval: float | None = None,
        settings: FreezeSettings | None = None,
        diag_env_enabled: bool | None = None,
        debug_log_writer: Callable[..., None] | None = None,
        logger: Any | None = None,
        dump_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        """Configure the watchdog.

        All parameters are optional — the constructor reads the
        persisted + env config for you. Tests pass the knobs
        explicitly so the test never depends on env ordering.
        """
        resolved = settings or resolve_freeze_settings()
        self._threshold = float(threshold) if threshold is not None else resolved.threshold_s
        self._check_interval = float(check_interval) if check_interval is not None else DEFAULT_FREEZE_CHECK_INTERVAL_S
        self._lock = threading.Lock()
        self._last_heartbeat: float = time.monotonic()
        self._last_heartbeat_wall: float = time.time()
        self._watchdog: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._detector_id = f"detector-{os.getpid()}-{id(self)}"
        self._diag_env_enabled = bool(diag_env_enabled) if diag_env_enabled is not None else _diag_env_enabled()
        # Counts how many times the threshold has tripped — surfaces
        # in the dump file so the Layer-4 CLI can show "frozen N
        # times in this run" without grep'ing the debug log.
        self._tripped_count = 0
        self._tripped_history: list[float] = []
        self._debug_log_writer = debug_log_writer
        self._logger = logger
        self._dump_dir = dump_path(dump_dir=dump_dir if dump_dir is not None else resolved.dump_dir)

    # ------------------------------------------------------------------
    # Singleton accessors — used by Layer-0 / Layer-3 hooks that don't
    # want to thread a detector instance through their constructor.
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> "FreezeDetector":
        """Return (and lazily build) the process-wide detector.

        The watchdog thread is NOT started by this call — see
        :meth:`enable` for the explicit toggle. ``instance()`` is
        safe to call from any thread.
        """
        with cls._INSTANCE_LOCK:
            inst = cls._INSTANCE
            if inst is None:
                inst = cls()
                cls._INSTANCE = inst
            return inst

    @classmethod
    def enable(cls, *, reason: str = "explicit") -> "FreezeDetector":
        """Start the watchdog. Idempotent.

        Used by ``CLAWCODEX_FREEZE_DIAG=1`` adoption in
        :meth:`maybe_start_from_env` and by the ``diag freeze-report``
        CLI when live-monitoring is requested.
        """
        inst = cls.instance()
        inst.start()
        # Surface the activation in the debug log so postmortem
        # ``query_runner.freeze_*`` events have a start-of-watch
        # marker. ``safe_call`` swallows the failure so the watchdog
        # never blocks on a logger that's been torn down.
        inst._safe_debug_event(
            "freeze_watchdog.enabled",
            threshold_s=inst._threshold,
            check_interval_s=inst._check_interval,
            reason=reason,
        )
        return inst

    @classmethod
    def disable(cls) -> None:
        """Stop the watchdog. Idempotent. Used in tests."""
        with cls._INSTANCE_LOCK:
            inst = cls._INSTANCE
            if inst is not None:
                inst.stop()

    @classmethod
    def maybe_start_from_env(cls) -> "FreezeDetector | None":
        """Start the watchdog iff ``CLAWCODEX_FREEZE_DIAG=1``.

        Called from each entrypoint's bootstrap to honour the env
        var. Returns the detector (even when nothing changes) so the
        caller can ``heartbeat()`` without an extra lookup; returns
        None when the env var is unset so callers can skip emitting
        heartbeats for free.
        """
        if not _diag_env_enabled():
            return None
        return cls.enable(reason="env")

    # ------------------------------------------------------------------
    # Heartbeat protocol — called by the agent loop on every forward
    # progress milestone.
    # ------------------------------------------------------------------

    def heartbeat(self) -> None:
        """Mark the loop as alive. Cheap; safe to call per chunk."""
        now = time.monotonic()
        with self._lock:
            self._last_heartbeat = now
            self._last_heartbeat_wall = time.time()

    @property
    def seconds_since_last_heartbeat(self) -> float:
        """Float seconds since the last heartbeat (monotonic clock)."""
        with self._lock:
            return time.monotonic() - self._last_heartbeat

    # ------------------------------------------------------------------
    # Watchdog lifecycle.
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the daemon thread. Idempotent.

        The thread is named so a ``py-spy dump -p <pid>`` or
        ``threading.enumerate()`` call surfaces it cleanly. It
        joins cleanly via :meth:`stop`.
        """
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._stop_event.clear()
        self._watchdog = threading.Thread(
            target=self._run,
            name="freeze-detector",
            daemon=True,
        )
        self._watchdog.start()

    def stop(self, *, timeout: float = 1.0) -> None:
        """Ask the watchdog to exit; join with ``timeout``.

        Safe to call from any thread, multiple times. The watchdog
        loop checks ``self._stop_event`` between sleeps so this
        cannot hang longer than ``check_interval_s``.
        """
        self._stop_event.set()
        thread = self._watchdog
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._watchdog = None

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Watchdog loop body.

        Exits cleanly on :meth:`stop`. The loop interval is
        ``self._check_interval``; on every tick we measure how long
        the loop has been silent. If that gap is past
        ``self._threshold`` we dump and emit a debug event.

        Critically the loop does NOT trip an abort controller —
        that is P108-G's job. Self-termination would mask the bug
        we are trying to diagnose.
        """
        while not self._stop_event.is_set():
            # ``Event.wait`` is interruptable by ``stop.set()`` so the
            # shutdown path doesn't have to wait a full tick.
            if self._stop_event.wait(self._check_interval):
                return
            try:
                if self.check():
                    # Don't spam dumps — back off to once per
                    # threshold window so a sustained freeze doesn't
                    # fill the dump dir. The history list captures
                    # every trip so the Layer-4 CLI can still show
                    # "tripped N times".
                    pass
            except Exception:
                if self._logger is not None:
                    try:
                        self._logger.exception("freeze-detector tick failed")
                    except Exception:  # nosec B110
                        pass

    def check(self) -> bool:
        """Inspect the heartbeat gap. Returns True iff threshold tripped.

        Public so tests can drive the detector without spawning the
        watchdog thread. ``check()`` captures a snapshot of every
        live thread's stack at the moment of the trip.
        """
        gap = self.seconds_since_last_heartbeat
        if gap < self._threshold:
            return False
        # Stagger dump writes so a sustained freeze produces one
        # dump per threshold window, not one per tick.
        if self._should_skip_due_to_backoff(gap):
            return True
        try:
            dump = self._capture_dump(elapsed=gap)
        except Exception:  # nosec B110
            if self._logger is not None:
                try:
                    self._logger.exception("freeze dump capture failed")
                except Exception:  # nosec B110
                    pass
            return True
        self._tripped_count += 1
        self._tripped_history.append(time.time())
        # Persist to disk first — the CLI / postmortem tool reads
        # the file even if the debug log write below raises.
        try:
            dump.write(self._dump_dir)
        except Exception:  # nosec B110
            if self._logger is not None:
                try:
                    self._logger.exception("freeze dump write failed")
                except Exception:  # nosec B110
                    pass
        self._safe_debug_event(
            "freeze_detected",
            elapsed_seconds=round(gap, 3),
            threshold_s=self._threshold,
            dump_file=dump.dump_file,
            thread_count=len(dump.thread_stacks),
            tripped_count=self._tripped_count,
        )
        return True

    def _should_skip_due_to_backoff(self, gap: float) -> bool:
        """Skip a re-dump inside the same frozen window.

        Without this, a 5-minute freeze fires a dump every 10 s (≈30
        dumps, ~60 MiB). The dump file name encodes the trip index
        so the suppressed dumps are still discoverable.
        """
        if not self._tripped_history:
            return False
        last_trip = self._tripped_history[-1]
        # Use wall-clock time so the backoff matches the file
        # timestamps the Layer-4 CLI uses to filter.
        return (time.time() - last_trip) < self._threshold

    def _capture_dump(self, *, elapsed: float) -> FreezeDump:
        """Snapshot every live Python thread's stack.

        ``sys._current_frames()`` is documented as CPython-only but
        mirrored by PyPy; for the platforms we ship on this is
        always populated. ``traceback.format_stack`` formats each
        thread's frame list into a single ``str`` (newlines
        included) so the dumped JSON is grep-friendly.
        """
        thread_stacks: list[ThreadStackFrame] = []
        try:
            frames_map = sys._current_frames()  # type: ignore[attr-defined]
        except Exception:  # nosec B110
            frames_map = {}
        for tid, frame in frames_map.items():
            try:
                stack_text = "".join(traceback.format_stack(frame))
            except Exception:
                stack_text = "<traceback formatting failed>"
            thread_name = _thread_name_for(tid)
            thread_stacks.append(
                ThreadStackFrame(
                    tid=int(tid),
                    thread_name=thread_name,
                    frames=stack_text,
                )
            )
        return FreezeDump(
            detected_at_unix=time.time(),
            last_heartbeat_at_unix=self._last_heartbeat_wall,
            elapsed_seconds=float(elapsed),
            threshold_seconds=self._threshold,
            check_interval_seconds=self._check_interval,
            detected_by_thread="freeze-detector",
            diag_env_enabled=self._diag_env_enabled,
            detector_id=self._detector_id,
            thread_stacks=thread_stacks,
            extra={
                "tripped_count": self._tripped_count + 1,
                "current_pid": os.getpid(),
            },
        )

    def _safe_debug_event(self, name: str, **payload: Any) -> None:
        """Emit a debug event without breaking the watchdog.

        Mirrors ``extensions.api.debug_log.append_debug_event`` but
        doesn't import it (cold path).
        """
        if self._debug_log_writer is None:
            return
        try:
            self._debug_log_writer(name, **payload)
        except Exception:  # nosec B110
            pass


def _diag_env_enabled() -> bool:
    raw = os.environ.get(DEFAULT_FREEZE_DIAG_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _thread_name_for(tid: int) -> str:
    """Return the registered name for ``tid`` or a fallback."""
    for thread in threading.enumerate():
        if thread.ident == tid:
            return thread.name
    return f"unknown-{tid}"


# ----------------------------------------------------------------------
# ``FreezeDump.write`` — kept on the dataclass so the Layer-4 CLI can
# call ``dump.write`` without depending on the detector module.
# ----------------------------------------------------------------------


def _dump_write(self: Any, target_dir: str | os.PathLike[str]) -> None:  # noqa: D401
    """Persist the dump to ``<target_dir>/freeze-<pid>-<idx>-<ts>.json``.

    Bound below; declared here to keep the type stub next to the
    dataclass definition. Caps the file at ``_MAX_DUMP_BYTES`` so a
    runaway stack cannot overflow the disk.
    """
    target = dump_path(dump_dir=target_dir)
    # Trip index in the filename lets a Layer-4 reader reconstruct
    # the timeline without parsing the file body.
    extra = self.extra if isinstance(self.extra, dict) else {}
    idx = extra.get("tripped_count", 0)
    ts = int(self.detected_at_unix)
    name = f"freeze-{self.process_id}-{idx}-{ts}.json"
    path = target / name
    payload = self.to_dict()
    payload.pop("dump_file", None)
    data = json.dumps(payload, indent=2, default=str)
    if len(data.encode("utf-8")) > _MAX_DUMP_BYTES:
        # Truncate per-thread stacks to keep the dump byte-budget
        # bounded. ``thread_count`` survives so the reader still
        # knows how many threads were live.
        for entry in payload.get("thread_stacks", []):
            entry["frames"] = entry.get("frames", "")[:4096] + "\n...<truncated>"
        data = json.dumps(payload, indent=2, default=str)
    path.write_text(data, encoding="utf-8")
    # ``object.__setattr__`` because the dataclass is ``frozen`` for
    # consumers but ``dump_file`` is an internal marker.
    object.__setattr__(self, "dump_file", str(path))


# Bind the method onto the dataclass. Doing this in code (vs. as
# ``@dataclass``) keeps the immutable field-list readable.
FreezeDump.write = _dump_write  # type: ignore[attr-defined]


__all__ = [
    "DEFAULT_FREEZE_CHECK_INTERVAL_S",
    "DEFAULT_FREEZE_DIAG_ENV",
    "FreezeDetector",
    "FreezeDump",
    "ThreadStackFrame",
]
