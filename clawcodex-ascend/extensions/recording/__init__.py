"""Asciicast v2 recorder for ClawCodex subsystems.

Public surface re-exports the writer, capture handle, source registry,
and self-contained validator. Adapters live next to the subsystems they
instrument (e.g. ``extensions/orchestrator/asciicast_sink.py``).
"""

from __future__ import annotations

from extensions.recording.asciicast_writer import (
    AsciicastCapture,
    AsciicastWriter,
)
from extensions.recording.config import RecordingConfig
from extensions.recording.registry import (
    RecordableSourceRegistry,
    get_default_registry,
    register_source,
)
from extensions.recording.repl_source import (
    ReplConsoleSource,
    install_repl_capture,
)
from extensions.recording.validate_cast import validate_cast

# Register the well-known source factories at import time so callers
# can ask ``get_default_registry().get("orchestrator")`` without
# having to wire up the adapter themselves. Each adapter is imported
# lazily — a subsystem that hasn't been touched yet doesn't pay any
# import cost.
from extensions.recording import _factories  # noqa: F401,E402

__all__ = [
    "AsciicastCapture",
    "AsciicastWriter",
    "RecordableSourceRegistry",
    "RecordingConfig",
    "ReplConsoleSource",
    "get_default_registry",
    "install_repl_capture",
    "register_source",
    "validate_cast",
]
