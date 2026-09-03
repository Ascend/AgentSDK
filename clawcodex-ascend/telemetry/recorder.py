"""TelemetryRecorder — the single chokepoint for telemetry event capture.

The recorder is the only public surface business code talks to:

* :func:`get_recorder` returns either a real :class:`TelemetryRecorder`
  or a :class:`_NullRecorder` whose methods are no-ops. The default
  is :class:`_NullRecorder` because :attr:`TelemetryConfig.enabled` is
  ``False`` until a user opts in.
* All public methods are best-effort: any exception inside the
  recorder is logged and swallowed. Business code MUST NOT be
  affected by telemetry failures.
* The recorder is process-global and lazily initialized. Re-entry
  from threads is safe because each method takes its own short
  critical section; the underlying :class:`LocalJsonlStorage` writes
  are atomic per file.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Final, Optional

from .aggregator import DailyAggregator
from .config import TelemetryConfig, load_config
from .events import EventType, TelemetryEvent
from .fingerprint import compute_fingerprint
from .redaction import Redactor
from .reporters import CompositeReporter, DryRunReporter, LocalFileReporter
from .storage import LocalJsonlStorage

logger = logging.getLogger(__name__)

_REDACTION_PROJECT_ROOTS: Final[tuple[str, ...]] = (
    # Resolved at construction time; updated lazily by ``_build_recorder``.
)


def _default_project_roots() -> tuple[str, ...]:
    roots: list[str] = []
    try:
        from src.config import GLOBAL_CONFIG_DIR  # type: ignore[import-not-found]

        roots.append(str(GLOBAL_CONFIG_DIR))
    except Exception:  # nosec B110
        # The optional global config root may be unavailable in partial installs.
        pass
    try:
        from pathlib import Path

        cwd = str(Path.cwd())
        if cwd and cwd not in roots:
            roots.append(cwd)
    except Exception:  # nosec B110
        # Current-working-directory discovery is only a redaction hint.
        pass
    return tuple(roots)


def _short_session_id(raw: str) -> str:
    """Return a 16-char stable-ish hash of a session id.

    Telemetry is a low-frequency aggregate stream; the full session
    id is overkill for a join key and would re-introduce the
    cardinality problem the spec warns against.
    """
    if not raw:
        raw = uuid.uuid4().hex
    return hashlib.sha1(raw.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Null recorder (zero-cost when disabled)
# ---------------------------------------------------------------------------


class _NullRecorder:
    """All methods are no-ops; used when telemetry is disabled."""

    enabled: bool = False

    def record_session_start(self, **_: Any) -> None:
        return None

    def record_session_end(self, **_: Any) -> None:
        return None

    def record_command_run(self, **_: Any) -> None:
        return None

    def record_error(self, **_: Any) -> None:
        return None

    def record_tool_summary(self, **_: Any) -> None:
        return None

    def record_turn(self, **_: Any) -> None:
        return None

    def record_usage(self, **_: Any) -> None:
        return None

    def record_event(self, event: TelemetryEvent, kind: str = "events") -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def config(self) -> TelemetryConfig:
        return TelemetryConfig()

    @property
    def dry_run_reporter(self) -> DryRunReporter:
        return DryRunReporter()

    def build_report_for(self, date: str) -> str:
        return ""


# ---------------------------------------------------------------------------
# Real recorder
# ---------------------------------------------------------------------------


@dataclass
class _BuildResult:
    recorder: "_TelemetryRecorderImpl | _NullRecorder"
    cleanup: "list[Any]"


class _TelemetryRecorderImpl:
    enabled: bool = True

    def __init__(
        self,
        cfg: TelemetryConfig,
        storage: LocalJsonlStorage,
        aggregator: DailyAggregator,
        redactor: Redactor,
        reporters: CompositeReporter,
    ) -> None:
        self._cfg = cfg
        self._storage = storage
        self._aggregator = aggregator
        self._redactor = redactor
        self._reporters = reporters
        self._lock = threading.Lock()
        self._closed = False
        self._dry_run = DryRunReporter()

    @property
    def config(self) -> TelemetryConfig:
        return self._cfg

    @property
    def dry_run_reporter(self) -> DryRunReporter:
        return self._dry_run

    def build_report_for(self, date: str) -> str:
        summary = self._storage.read_latest_summary(date) or {}
        if not summary:
            # Build on demand if today's aggregator hasn't run yet.
            self._aggregator.aggregate(date)
            summary = self._storage.read_latest_summary(date) or {}
        return self._dry_run.render(summary, date)

    # -- public recorders ----------------------------------------------

    def record_session_start(
        self,
        *,
        session_id: str,
        entrypoint: str,
        client_type: str = "cli",
        is_non_interactive: bool = False,
        platform: str | None = None,
        python_version: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        os_version: str | None = None,
        ide_type: str | None = None,
        ide_version: str | None = None,
        is_resume: bool | None = None,
        start_time: float | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Best-effort session_start.

        ``os_version`` / ``ide_type`` / ``ide_version`` /
        ``is_resume`` / ``start_time`` / ``extra`` are bridged from
        :class:`src.services.analytics.metadata.SessionAnalyticsMetadata`.
        When any of them is ``None`` (or absent) the recorder falls back
        to :func:`collect_session_metadata` so a caller can stay on the
        old 8-kw signature and still get the analytics fields for free.
        """
        if self._closed:
            return
        if platform is None:
            try:
                import platform as _platform

                platform = _platform.system() or "unknown"
            except Exception:
                platform = "unknown"
        if python_version is None:
            try:
                import platform as _platform

                python_version = _platform.python_version() or "unknown"
            except Exception:
                python_version = "unknown"

        # When the analytics fields are not supplied, derive them
        # from the same source the analytics layer uses. Any failure is
        # swallowed — telemetry must never break business code.
        analytics_fields = (os_version, ide_type, ide_version, is_resume, start_time, extra)
        if any(value is None for value in analytics_fields):
            try:
                from clawcodex_ext.services.analytics.metadata import collect_session_metadata

                meta = collect_session_metadata(
                    session_id=session_id,
                    model=model or "",
                    ide_type=ide_type or "",
                    ide_version=ide_version or "",
                    is_non_interactive=bool(is_non_interactive),
                    is_resume=bool(is_resume) if is_resume is not None else False,
                )
            except Exception:
                meta = None
            if meta is not None:
                os_version = os_version if os_version is not None else meta.os_version
                ide_type = ide_type if ide_type is not None else meta.ide_type
                ide_version = ide_version if ide_version is not None else meta.ide_version
                if is_resume is None:
                    is_resume = meta.is_resume
                if start_time is None:
                    start_time = meta.start_time
                if extra is None:
                    extra = dict(meta.extra) if meta.extra else None

        event = TelemetryEvent(
            type=EventType.SESSION_START,
            timestamp=time.time(),
            session_id=_short_session_id(session_id),
            fields={
                "entrypoint": entrypoint,
                "client_type": client_type,
                "is_non_interactive": bool(is_non_interactive),
                "platform": platform,
                "os_version": os_version or "",
                "python_version": python_version,
                "provider": provider or "unknown",
                "model": model or "unknown",
                "ide_type": ide_type or "",
                "ide_version": ide_version or "",
                "is_resume": bool(is_resume) if is_resume is not None else False,
                "start_time": float(start_time) if start_time is not None else 0.0,
                "extra": dict(extra) if extra else {},
                "app_version": _safe_app_version(),
            },
        )
        self._enqueue_event(event)

    def record_session_end(
        self,
        *,
        session_id: str,
        duration_s: float,
        exit_status: int,
        outcome: str | None = None,
    ) -> None:
        if self._closed:
            return
        event = TelemetryEvent(
            type=EventType.SESSION_END,
            timestamp=time.time(),
            session_id=_short_session_id(session_id),
            fields={
                "duration_s": float(duration_s),
                "exit_status": int(exit_status),
                "outcome": outcome or ("completed" if exit_status == 0 else "failed"),
            },
        )
        self._enqueue_event(event)

    def record_command_run(
        self,
        *,
        session_id: str,
        command_name: str,
        mode: str = "non_interactive",
        success: bool = True,
        duration_s: float = 0.0,
        exit_status: int | None = None,
    ) -> None:
        if self._closed:
            return
        event = TelemetryEvent(
            type=EventType.COMMAND_RUN,
            timestamp=time.time(),
            session_id=_short_session_id(session_id),
            fields={
                "command_name": command_name,
                "mode": mode,
                "success": bool(success),
                "duration_s": float(duration_s),
                "exit_status": int(exit_status) if exit_status is not None else None,
            },
        )
        self._enqueue_event(event)

    def record_tool_summary(
        self,
        *,
        session_id: str,
        tool_name: str,
        success: bool = True,
        duration_s: float = 0.0,
        timed_out: bool = False,
    ) -> None:
        if self._closed:
            return
        event = TelemetryEvent(
            type=EventType.TOOL_SUMMARY,
            timestamp=time.time(),
            session_id=_short_session_id(session_id),
            fields={
                "tool_name": tool_name,
                "success": bool(success),
                "duration_s": float(duration_s),
                "timed_out": bool(timed_out),
            },
        )
        self._enqueue_event(event)

    def record_turn(self, *, session_id: str, success: bool = True, duration_s: float = 0.0) -> None:
        """Record one completed agent turn without retaining its content."""
        if self._closed:
            return
        self._enqueue_event(
            TelemetryEvent(
                type=EventType.TURN,
                timestamp=time.time(),
                session_id=_short_session_id(session_id),
                fields={"success": bool(success), "duration_s": float(duration_s)},
            )
        )

    def record_usage(
        self,
        *,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record aggregate usage for a session; no prompts or outputs."""
        if self._closed:
            return
        self._enqueue_event(
            TelemetryEvent(
                type=EventType.USAGE,
                timestamp=time.time(),
                session_id=_short_session_id(session_id),
                fields={
                    "input_tokens": max(0, int(input_tokens)),
                    "output_tokens": max(0, int(output_tokens)),
                    "cache_read_tokens": max(0, int(cache_read_tokens)),
                    "cache_creation_tokens": max(0, int(cache_creation_tokens)),
                    "cost_usd": max(0.0, float(cost_usd)),
                },
            )
        )

    def record_error(
        self,
        *,
        session_id: str,
        exc: BaseException,
    ) -> None:
        if self._closed:
            return
        try:
            fingerprint = compute_fingerprint(exc, project_roots=tuple(self._redactor.project_roots))
        except Exception:  # noqa: BLE001
            fingerprint = "unknown"
        try:
            stacktrace = self._redactor.truncate_stacktrace(exc)
        except Exception:  # noqa: BLE001
            stacktrace = []

        error_event = TelemetryEvent(
            type=EventType.ERROR,
            timestamp=time.time(),
            session_id=_short_session_id(session_id),
            fields={
                "error_class": type(exc).__name__,
                "fingerprint": fingerprint,
                "stacktrace": stacktrace,
            },
        )
        self._enqueue_event(error_event, kind="events")
        self._enqueue_event(error_event, kind="crashes")

    # -- lifecycle ------------------------------------------------------

    def flush(self) -> None:
        """Force aggregator to run for today; safe to call repeatedly."""
        if self._closed:
            return
        try:
            from .storage import utc_date, utc_now

            date = utc_date(utc_now())
            summary = self._aggregator.aggregate(date)
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: flush aggregator failed: %s", exc)
            return
        if not summary:
            return
        if not self._cfg.reporting.reporting_enabled:
            return
        try:
            rendered = self._dry_run.render(summary, date)
            self._reporters.emit(rendered, date=date)
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: flush reporter failed: %s", exc)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._storage.retention_sweep()
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: retention sweep failed during close: %s", exc)

    # -- internals ------------------------------------------------------

    def record_event(self, event: TelemetryEvent, kind: str = "events") -> None:
        """Submit a pre-built :class:`TelemetryEvent` for redaction + storage.

        This is the public chokepoint for callers that need to attach
        rich, type-specific payloads that the typed ``record_*()``
        helpers cannot express — e.g. the analytics → telemetry bridge
        routing ``IMAGE_PROCESSING`` events with arbitrary ``subtype``
        and per-call fields.

        Failures are swallowed (debug-logged) so the caller is never
        blocked by telemetry errors.
        """
        if self._closed:
            return
        self._enqueue_event(event, kind=kind)

    def _enqueue_event(self, event: TelemetryEvent, kind: str = "events") -> None:
        try:
            redacted = self._redactor.redact_event(event)
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: redact failed (%s), dropping", exc)
            return
        try:
            self._storage.append(kind, redacted.to_dict())
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: append failed (%s)", exc)
        try:
            self._aggregator.aggregate_today_if_stale()
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: aggregator failed (%s)", exc)


def _safe_app_version() -> str:
    try:
        from .version import __version__

        return __version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_RECORDER_LOCK = threading.Lock()
_RECORDER: _NullRecorder | _TelemetryRecorderImpl | None = None


def get_recorder() -> _NullRecorder | _TelemetryRecorderImpl:
    """Return the process-global recorder, building it on first call.

    The result is a :class:`_NullRecorder` when
    :attr:`TelemetryConfig.enabled` is ``False`` (the default); the
    real :class:`_TelemetryRecorderImpl` is returned otherwise.
    """
    global _RECORDER
    if _RECORDER is not None:
        return _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is not None:
            return _RECORDER
        _RECORDER = _build_recorder()
        return _RECORDER


def _build_recorder() -> _NullRecorder | _TelemetryRecorderImpl:
    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        logger.debug("telemetry: load_config failed (%s), using defaults", exc)
        cfg = TelemetryConfig()
    if not cfg.enabled:
        return _NullRecorder()

    try:
        storage = LocalJsonlStorage(cfg.storage_dir, cfg.retention_days)
    except Exception as exc:  # noqa: BLE001
        logger.debug("telemetry: storage init failed (%s)", exc)
        return _NullRecorder()

    aggregator = DailyAggregator(storage)
    redactor = Redactor(cfg.redaction, _default_project_roots())
    reporters = CompositeReporter()
    if cfg.reporting.reporting_enabled:
        _configure_reporters(cfg, storage, redactor, reporters)

    return _TelemetryRecorderImpl(
        cfg=cfg,
        storage=storage,
        aggregator=aggregator,
        redactor=redactor,
        reporters=reporters,
    )


def _configure_reporters(
    cfg: TelemetryConfig,
    storage: LocalJsonlStorage,
    redactor: Redactor,
    reporters: CompositeReporter,
) -> None:
    kind = (cfg.reporting.kind or "local_file").strip().lower()
    mode = (cfg.reporting.mode or "create_daily").strip().lower()
    if kind == "dry_run":
        reporters.add(DryRunReporter())
        return
    if kind == "issue" and mode != "local_file":
        try:
            from .reporters.issue import IssueReporter

            reporters.add(
                IssueReporter(
                    storage=storage,
                    redactor=redactor,
                    config=cfg.reporting,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: issue reporter init failed: %s", exc)
            storage.append(
                "reporter_errors",
                {
                    "timestamp": time.time(),
                    "kind": "issue",
                    "reason": "init_failed",
                    "platform": cfg.reporting.platform,
                    "mode": cfg.reporting.mode,
                    "error": str(exc),
                },
            )
        return
    reporters.add(LocalFileReporter(storage, redactor))


def reset_recorder_for_tests() -> None:
    """Reset the global recorder. Tests only."""
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is not None and not isinstance(_RECORDER, _NullRecorder):
            try:
                _RECORDER.close()
            except Exception:  # nosec B110
                # Test reset must always clear the process-global recorder.
                pass
        _RECORDER = None


def override_recorder(recorder: Optional["_NullRecorder | _TelemetryRecorderImpl"]) -> None:
    """Tests and embedders may pin a specific recorder instance.

    ``None`` restores the lazy default.
    """
    global _RECORDER
    with _RECORDER_LOCK:
        _RECORDER = recorder
