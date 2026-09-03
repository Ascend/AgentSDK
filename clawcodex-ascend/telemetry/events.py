"""Telemetry event types and core data model.

Telemetry deliberately keeps the event model small and stable. The event
table and privacy boundary are defined in the telemetry data model.
Prompts, model outputs, transcripts, file contents, API keys, env vars,
absolute paths and full shell args are **never** recorded here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

# ``SCHEMA_VERSION`` is the *read* default — events missing the
# ``schema_version`` field are treated as v1 so legacy on-disk data
# still loads. New events are written with the value re-exported from
# ``migration.py`` (see :data:`SCHEMA_VERSION_V2`).
# ``SCHEMA_VERSION`` is the *read* default — events missing the
# ``schema_version`` field are treated as v1 so legacy on-disk data
# still loads. Newly-written events are stamped with
# :data:`SCHEMA_VERSION_V2` via the ``TelemetryEvent`` field default
# below.
SCHEMA_VERSION: Final[int] = 1
SCHEMA_VERSION_V2: Final[int] = 2
_CURRENT_SCHEMA_VERSION: Final[int] = SCHEMA_VERSION_V2


class EventType(str, Enum):
    """Stable event type names persisted in JSONL.

    Names are persisted verbatim; renaming is a breaking change. Adding a
    new value is safe and must be paired with a schema version bump in
    the aggregator if the on-disk shape changes.
    """

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    COMMAND_RUN = "command_run"
    TOOL_SUMMARY = "tool_summary"
    TURN = "turn"
    USAGE = "usage"
    ERROR = "error"
    CRASH = "crash"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class TelemetryEvent:
    """A single telemetry event.

    The dataclass is intentionally narrow: ``fields`` carries per-type
    structured data (already redacted) and the contract is that callers
    MUST NOT put sensitive payloads into ``fields`` directly — the
    :class:`Redactor` is the single chokepoint for redaction.
    """

    type: EventType
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    schema_version: int = SCHEMA_VERSION_V2
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "fields": dict(self.fields),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TelemetryEvent":
        type_str = payload.get("type", "")
        try:
            event_type = EventType(type_str)
        except ValueError:
            # Unknown event type — keep the original value as a string so
            # forward compatibility is not lost when the on-disk shape
            # grows new event kinds before this binary learns about them.
            class _Passthrough(str, Enum):
                UNKNOWN = type_str

            event_type = _Passthrough.UNKNOWN  # type: ignore[assignment]
        return cls(
            type=event_type,  # type: ignore[arg-type]
            timestamp=float(payload.get("timestamp", time.time())),
            session_id=str(payload.get("session_id", "")),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            fields=dict(payload.get("fields", {}) or {}),
        )
