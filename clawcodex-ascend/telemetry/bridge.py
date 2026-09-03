"""Bridge: route ``src.services.analytics`` events into telemetry.

ClawCodex has two parallel event systems:

* :mod:`src.services.analytics` — older, pluggable :class:`AnalyticsSink`
  abstraction (``NullSink``/``ConsoleSink``/``FileSink``) consumed by
  ``image_processor.py``/``image_validation.py``/``pdf_extraction.py``.
* :mod:`telemetry` — newer, opt-in system with local
  JSONL storage, redaction, daily aggregation and opt-in Issue
  reporting. Redaction runs on every event in
  :meth:`_TelemetryRecorderImpl._enqueue_event` so any field set on a
  :class:`TelemetryEvent` is automatically scrubbed for secrets, paths,
  prompts, outputs, etc.

By default the analytics global sink is :class:`NullSink` — every
``log_event()`` call from the image / PDF pipeline is silently
discarded. The :class:`AnalyticsTelemetrySink` installed by
:func:`install_analytics_bridge` replaces that no-op with one that
forwards events into the live recorder. When telemetry is disabled
the live recorder is :class:`_NullRecorder` and the bridge is a no-op
pass-through, so this is safe to leave installed permanently.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from clawcodex_ext.services.analytics.events import AnalyticsEvent, set_analytics_sink
from clawcodex_ext.services.analytics.sink import AnalyticsSink
from .events import EventType, TelemetryEvent
from .fingerprint import _normalize_message
from .recorder import get_recorder

logger = logging.getLogger(__name__)

_FINGERPRINT_LENGTH = 16

# Analytics EventType values that have no telemetry semantic and must
# not be forwarded. ``PERMISSION_*`` is dropped to avoid leaking
# permission prompts; ``MODEL_SWITCH`` is dropped to avoid
# fingerprinting user model choice; ``COMPACT`` is internal to the
# analytics layer.
_DROPPED_ANALYTICS_TYPES: frozenset[str] = frozenset(
    {
        "compact",
        "permission_prompt",
        "permission_decision",
        "model_switch",
    }
)

# Analytics event types that become completed-turn telemetry events.
_COMMAND_RUN_SUBTYPES: frozenset[str] = frozenset(
    {
        "agent_spawn",
        "agent_complete",
    }
)


def _short_session_id(raw: str) -> str:
    """Mirror :func:`telemetry.recorder._short_session_id`.

    The recorder hashes raw session ids to a stable 16-char SHA1
    prefix for storage. The bridge must use the same join key so
    analytics events and native telemetry events merge on the same
    session in the daily summary.
    """
    if not raw:
        return ""
    digest = hashlib.sha1(raw.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()
    return digest[:_FINGERPRINT_LENGTH]


def _string_fingerprint(message: str) -> str:
    """Return a stable 16-char hash for an error message string.

    Analytics ``ERROR`` events carry ``data`` dictionaries rather than
    live :class:`BaseException` instances, so
    :func:`compute_fingerprint` (which walks ``__traceback__``) cannot
    be used. We instead normalize the message with the same volatile
    token stripper as the exception path and hash the result.
    """
    normalized = _normalize_message(message or "") or "unknown"
    digest = hashlib.sha1(normalized.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()
    return digest[:_FINGERPRINT_LENGTH]


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


class AnalyticsTelemetrySink(AnalyticsSink):
    """Drop-in :class:`AnalyticsSink` that forwards events to telemetry.

    The class is stateless apart from the optional ``project_roots``
    hint used for stack-trace truncation. All actual storage happens
    inside the live recorder, so constructing a new instance per
    process is cheap.
    """

    enabled: bool = True

    def __init__(self, project_roots: tuple[str, ...] | None = None) -> None:
        self._project_roots = project_roots

    # ------------------------------------------------------------------
    # AnalyticsSink ABC
    # ------------------------------------------------------------------

    def emit(self, event: AnalyticsEvent) -> None:  # type: ignore[override]
        """Translate ``event`` into a :class:`TelemetryEvent` and enqueue it.

        All exceptions are caught and logged at ``DEBUG`` so a malformed
        analytics event never breaks the calling pipeline (image,
        PDF, etc.). The pipeline code is fire-and-forget and trusts
        the global sink to swallow errors.
        """
        try:
            telemetry_event = self._translate(event)
            if telemetry_event is None:
                return
            recorder = get_recorder()
            if not getattr(recorder, "enabled", True):
                return
            recorder.record_event(telemetry_event)
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: analytics bridge emit failed: %s", exc)

    def flush(self) -> None:  # type: ignore[override]
        try:
            get_recorder().flush()
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: analytics bridge flush failed: %s", exc)

    def close(self) -> None:  # type: ignore[override]
        # The recorder has its own lifetime; do not close it from here.
        return None

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def _translate(self, event: AnalyticsEvent) -> TelemetryEvent | None:
        type_name = event.type.value if hasattr(event.type, "value") else str(event.type)
        if type_name in _DROPPED_ANALYTICS_TYPES:
            return None

        # Snapshot data so we never mutate the caller's dict.
        data: dict[str, Any] = dict(event.data) if event.data else {}
        # Preserve provenance and model info without leaking it as a
        # "prompt" / "output" / "stacktrace" key (those get auto-dropped
        # by the redactor).
        data.setdefault("analytics_event_type", type_name)
        if event.model:
            data.setdefault("model", event.model)

        session_id = _short_session_id(event.session_id or "")
        timestamp = float(event.timestamp) if event.timestamp else time.time()

        if type_name == "image_processing":
            subtype = _coerce_str(data.pop("subtype", "")) or "unknown"
            success = data.pop("success", True)
            duration_s = data.pop("duration_s", 0.0)
            tool_name = _coerce_str(data.pop("tool_name", "")) or "image_processing"
            fields: dict[str, Any] = {
                "tool_name": tool_name,
                "subtype": subtype,
                "success": bool(success),
                "duration_s": float(duration_s),
            }
            fields.update(data)
            return TelemetryEvent(
                type=EventType.TOOL_SUMMARY,
                timestamp=timestamp,
                session_id=session_id,
                fields=fields,
            )

        if type_name in ("session_start", "session_end"):
            return TelemetryEvent(
                type=EventType(type_name),
                timestamp=timestamp,
                session_id=session_id,
                fields=data,
            )

        if type_name in _COMMAND_RUN_SUBTYPES:
            success = data.pop("success", True)
            duration_s = data.pop("duration_s", 0.0)
            fields = {
                "command_name": type_name,
                "subtype": type_name,
                "mode": "non_interactive",
                "success": bool(success),
                "duration_s": float(duration_s),
            }
            fields.update(data)
            return TelemetryEvent(
                type=EventType.COMMAND_RUN,
                timestamp=timestamp,
                session_id=session_id,
                fields=fields,
            )

        if type_name == "turn_end":
            success = data.pop("success", True)
            duration_s = data.pop("duration_s", 0.0)
            return TelemetryEvent(
                type=EventType.TURN,
                timestamp=timestamp,
                session_id=session_id,
                fields={"success": bool(success), "duration_s": float(duration_s)},
            )
        if type_name == "turn_start":
            return None

        if type_name in ("tool_use", "tool_result"):
            tool_name = _coerce_str(data.pop("tool", "")) or type_name
            success = data.pop("success", True)
            duration_s = data.pop("duration_s", 0.0)
            fields = {
                "tool_name": tool_name,
                "subtype": type_name,
                "success": bool(success),
                "duration_s": float(duration_s),
            }
            fields.update(data)
            return TelemetryEvent(
                type=EventType.TOOL_SUMMARY,
                timestamp=timestamp,
                session_id=session_id,
                fields=fields,
            )

        if type_name == "error":
            message = _coerce_str(data.pop("message", None) or data.pop("error", None) or data.pop("msg", None) or "")
            error_class = _coerce_str(data.pop("error_class", "")) or "AnalyticsError"
            fields = {
                "error_class": error_class,
                "fingerprint": _string_fingerprint(message),
                "stacktrace": [],
            }
            fields.update(data)
            return TelemetryEvent(
                type=EventType.ERROR,
                timestamp=timestamp,
                session_id=session_id,
                fields=fields,
            )

        # Unknown type — drop with a debug log rather than ship a
        # mystery event.
        logger.debug("telemetry: analytics bridge dropping unknown type %r", type_name)
        return None


_BRIDGE: AnalyticsTelemetrySink | None = None


def install_analytics_bridge(*, project_roots: tuple[str, ...] | None = None) -> AnalyticsTelemetrySink:
    """Install :class:`AnalyticsTelemetrySink` as the global analytics sink.

    Idempotent: calling it twice returns the existing instance. Safe to
    call before :func:`telemetry.recorder.get_recorder` is
    first invoked — the bridge lazily resolves the recorder on each
    ``emit()``. When telemetry is disabled the recorder is a no-op and
    the bridge becomes one too, so this can be installed permanently
    without any extra lifecycle management.

    Returns the live bridge instance for test introspection.
    """
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = AnalyticsTelemetrySink(project_roots=project_roots)
    set_analytics_sink(_BRIDGE)
    return _BRIDGE


def get_analytics_bridge() -> AnalyticsTelemetrySink | None:
    """Return the installed bridge, or ``None`` if not yet installed."""
    return _BRIDGE


def reset_analytics_bridge_for_tests() -> None:
    """Drop the installed bridge; the next ``install_analytics_bridge``
    call will construct a fresh instance. Test-only helper.
    """
    global _BRIDGE
    _BRIDGE = None
