"""Built-in source factories wired into the default registry.

Each entry registers one :class:`RecordableSource` factory under a
canonical ``source_id`` string. The factory is invoked by the CLI when
the user passes ``--sources <id>``.

Adding a new subsystem:

1. Implement the adapter (a class with ``source_id`` plus ``open`` /
   ``close``) in the subsystem's own directory.
2. Add a module-level factory function here that takes the capture
   handle and returns a configured adapter.
3. Call :func:`extensions.recording.register_source` at import time
   with the factory.

The factories are deliberately tiny — the heavy lifting (capture
handle, file format, threading) lives in :mod:`extensions.recording`.
"""

from __future__ import annotations

from typing import Any

from extensions.capabilities.recorder import AsciicastCapture
from extensions.recording.registry import register_source

__all__ = []


def _orchestrator_factory(capture: AsciicastCapture) -> Any:
    """Return the orchestrator-side :class:`RecordableSource` shim.

    The orchestrator exposes its sink via a constructor kwarg on
    :class:`Orchestrator` (see
    ``extensions/orchestrator/orchestrator.py:184``) rather than a
    standalone adapter class, so the factory returns a tiny shim that
    holds the capture and exposes ``source_id`` + ``close``.
    """
    from extensions.orchestrator.asciicast_sink import AsciicastSink

    class _OrchestratorShim:
        source_id = "orchestrator"
        _sink: AsciicastSink | None

        def __init__(self, cap: AsciicastCapture) -> None:
            # The sink itself is bound to the orchestrator's
            # ``_build_session_sink`` path; here we only stash the
            # capture so ``close`` can drop a final marker.
            self._capture = cap

        def open(self, capture: AsciicastCapture) -> None:  # noqa: ARG002
            self._capture = capture
            self._capture.marker(
                "orchestrator:recording_started",
                text="Orchestrator asciicast capture open",
            )

        def close(self) -> None:
            try:
                self._capture.marker("orchestrator:recording_closed")
            except Exception:  # nosec B110
                # Recorder shutdown markers are best-effort by design.
                pass

    return _OrchestratorShim(capture)


def _sop_factory(capture: AsciicastCapture) -> Any:
    """Return the SOP converter :class:`RecordableSource` shim.

    The SOP adapter is invoked from the CLI wrapper
    (``clawcodex_ext/cli/sop_cmd/commands.py``); the shim here just
    lets ``clawcodex record --sources sop`` produce a standalone
    capture for an ad-hoc run.
    """

    class _SopShim:
        source_id = "sop"

        def __init__(self, cap: AsciicastCapture) -> None:
            self._capture = cap

        def open(self, capture: AsciicastCapture) -> None:  # noqa: ARG002
            self._capture = capture
            self._capture.marker("sop:recording_started")

        def close(self) -> None:
            try:
                self._capture.marker("sop:recording_closed")
            except Exception:  # nosec B110
                # Recorder shutdown markers are best-effort by design.
                pass

    return _SopShim(capture)


def _visualizer_factory(capture: AsciicastCapture) -> Any:
    """Return the visualizer dashboard :class:`RecordableSource` shim.

    The adapter now lives under
    :mod:`extensions.recording.visualizer_dashboard_source` (its real
    consumer). We import it lazily so a partial checkout that lacks
    the visualizer package keeps ``clawcodex record --sources ...``
    working for non-visualizer sources.
    """
    from extensions.recording.visualizer_dashboard_source import (
        AsciicastDashboardSource,
    )

    class _VisualizerShim:
        source_id = "visualizer"

        def __init__(self, cap: AsciicastCapture) -> None:
            self._adapter = AsciicastDashboardSource()

        def open(self, capture: AsciicastCapture) -> None:  # noqa: ARG002
            # Recording the live dashboard is driven by the CLI tick
            # loop, not by the source itself; the shim just emits a
            # start marker here.
            capture.marker("visualizer:recording_started")

        def close(self) -> None:
            # The shim owns the adapter reference for its complete lifetime.
            return None

    return _VisualizerShim(capture)


def _cron_factory(capture: AsciicastCapture) -> Any:
    """Return the cron observer :class:`RecordableSource` shim."""

    class _CronShim:
        source_id = "cron"

        def __init__(self, cap: AsciicastCapture) -> None:
            self._capture = cap

        def open(self, capture: AsciicastCapture) -> None:  # noqa: ARG002
            self._capture = capture
            self._capture.marker("cron:recording_started")

        def close(self) -> None:
            try:
                self._capture.marker("cron:recording_closed")
            except Exception:  # nosec B110
                # Recorder shutdown markers are best-effort by design.
                pass

    return _CronShim(capture)


def _query_factory(capture: AsciicastCapture) -> Any:
    """Return the query-loop :class:`RecordableSource` shim."""

    class _QueryShim:
        source_id = "query"

        def __init__(self, cap: AsciicastCapture) -> None:
            self._capture = cap

        def open(self, capture: AsciicastCapture) -> None:  # noqa: ARG002
            self._capture = capture
            self._capture.marker("query:recording_started")

        def close(self) -> None:
            try:
                self._capture.marker("query:recording_closed")
            except Exception:  # nosec B110
                # Recorder shutdown markers are best-effort by design.
                pass

    return _QueryShim(capture)


# Register the factories at module import time. Tests can call
# ``reset_default_registry()`` to drop these and isolate themselves.
register_source("orchestrator", _orchestrator_factory)
register_source("sop", _sop_factory)
register_source("visualizer", _visualizer_factory)
register_source("cron", _cron_factory)
register_source("query", _query_factory)
