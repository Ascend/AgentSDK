"""Independent telemetry package.

Public surface (the only symbols business code should reach for):

* :func:`get_recorder`           — process-global, lazy, returns a real
  recorder or a no-op :class:`_NullRecorder` based on
  :attr:`TelemetryConfig.enabled`.
* :func:`record_session_start`   — fire-and-forget convenience wrappers
* :func:`record_session_end`
* :func:`record_command_run`
* :func:`record_error`
* :func:`record_tool_summary`
* :class:`TelemetryConfig`      — config dataclass + :func:`load_config`
* :class:`TelemetryEvent`        — event dataclass + :class:`EventType`
* :class:`AnalyticsTelemetrySink` — drop-in :class:`AnalyticsSink`
  that routes ``src.services.analytics`` events into the live
  recorder. Installed by :func:`install_analytics_bridge`.

Anything else is an implementation detail and may change between
minor releases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .recorder import _NullRecorder, _TelemetryRecorderImpl

from .aggregator import DailyAggregator
from .bridge import (
    AnalyticsTelemetrySink,
    get_analytics_bridge,
    install_analytics_bridge,
)
from .config import (
    ReportingConfig,
    TelemetryConfig,
    load_config,
)
from .events import SCHEMA_VERSION, EventType, TelemetryEvent
from .fingerprint import compute_fingerprint
from .redaction import RedactionConfig, Redactor
from .reporters import (
    CompositeReporter,
    DryRunReporter,
    LocalFileReporter,
    Reporter,
)
from .storage import LocalJsonlStorage
from .version import __version__

__all__ = [
    "AnalyticsTelemetrySink",
    "CompositeReporter",
    "DailyAggregator",
    "DryRunReporter",
    "EventType",
    "LocalFileReporter",
    "LocalJsonlStorage",
    "RedactionConfig",
    "Redactor",
    "Reporter",
    "ReportingConfig",
    "SCHEMA_VERSION",
    "TelemetryConfig",
    "TelemetryEvent",
    "__version__",
    "compute_fingerprint",
    "get_analytics_bridge",
    "install_analytics_bridge",
    "load_config",
    "record_command_run",
    "record_error",
    "record_session_end",
    "record_session_start",
    "record_tool_summary",
    "record_turn",
    "record_usage",
]


# ---------------------------------------------------------------------------
# Convenience wrappers around ``get_recorder()``
#
# These exist so that business code (CLI dispatch, headless, REPL) can call
# ``record_session_start(...)`` etc. directly without reaching into the
# recorder object. The wrappers are zero-cost when telemetry is disabled
# because they route through the null recorder's no-op methods.
# ---------------------------------------------------------------------------


def _recorder() -> _NullRecorder | _TelemetryRecorderImpl:
    from .recorder import get_recorder as _get

    return _get()


def record_session_start(
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
    """Record the start of a telemetry session."""
    _recorder().record_session_start(
        session_id=session_id,
        entrypoint=entrypoint,
        client_type=client_type,
        is_non_interactive=is_non_interactive,
        platform=platform,
        python_version=python_version,
        provider=provider,
        model=model,
        os_version=os_version,
        ide_type=ide_type,
        ide_version=ide_version,
        is_resume=is_resume,
        start_time=start_time,
        extra=extra,
    )


def record_session_end(
    *,
    session_id: str,
    duration_s: float,
    exit_status: int,
    outcome: str | None = None,
) -> None:
    """Record the end of a telemetry session."""
    _recorder().record_session_end(
        session_id=session_id,
        duration_s=duration_s,
        exit_status=exit_status,
        outcome=outcome,
    )


def record_command_run(
    *,
    session_id: str,
    command_name: str,
    mode: str = "non_interactive",
    success: bool = True,
    duration_s: float = 0.0,
    exit_status: int | None = None,
) -> None:
    """Record a command execution summary."""
    _recorder().record_command_run(
        session_id=session_id,
        command_name=command_name,
        mode=mode,
        success=success,
        duration_s=duration_s,
        exit_status=exit_status,
    )


def record_error(*, session_id: str, exc: BaseException) -> None:
    """Record a redacted error fingerprint for a session."""
    _recorder().record_error(session_id=session_id, exc=exc)


def record_tool_summary(
    *,
    session_id: str,
    tool_name: str,
    success: bool = True,
    duration_s: float = 0.0,
    timed_out: bool = False,
) -> None:
    """Record aggregate metadata for one tool execution."""
    _recorder().record_tool_summary(
        session_id=session_id,
        tool_name=tool_name,
        success=success,
        duration_s=duration_s,
        timed_out=timed_out,
    )


def record_turn(
    *,
    session_id: str,
    success: bool = True,
    duration_s: float = 0.0,
) -> None:
    """Record one completed agent turn."""
    _recorder().record_turn(
        session_id=session_id,
        success=success,
        duration_s=duration_s,
    )


def record_usage(
    *,
    session_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Record aggregate token and cost usage for a session."""
    _recorder().record_usage(
        session_id=session_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cost_usd=cost_usd,
    )
