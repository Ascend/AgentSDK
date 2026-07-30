"""Data model for the Kairos tick scheduler and brief generator.

The scheduler runs in a background thread, firing :class:`TickEvent`s at
a configured interval. Callers subscribe by registering a callable that
receives the event. The brief generator consumes a status snapshot and
returns a deterministic, low-ceremony summary suitable for CLI / status
bar display.

The model is intentionally minimal — there is no dependency on the
rest of the agent runtime. Higher-level layers (CLI, SleepTool,
conversation-stream injection) wrap this primitive.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Reuse the path-safe id pattern from ultraplan/templates.
_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


def _validate_id(value: str, *, what: str = "id") -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")
    if not _ID_RE.match(value):
        raise ValueError(f"{what} has invalid characters or length: {value!r} (expected [A-Za-z0-9._-]{{1,64}})")


def _validate_positive_interval(value: float, *, what: str) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a number")
    if value <= 0:
        raise ValueError(f"{what} must be positive (got {value!r})")


def _validate_jitter(value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError("jitter must be a number")
    if value < 0:
        raise ValueError(f"jitter must be non-negative (got {value!r})")
    if value > 1:
        raise ValueError(f"jitter is a fraction of the interval; must be in [0, 1] (got {value!r})")


@dataclass(frozen=True)
class TickConfig:
    """Configuration for the periodic tick scheduler.

    Attributes:
        id: Stable identifier for this scheduler instance (used in logs
            and event payloads). Same id pattern as plans / templates.
        interval_seconds: Nominal interval between ticks. Must be > 0.
        enabled: Whether the scheduler starts in a running state. A
            scheduler can be enabled but paused (see
            :attr:`TickScheduler.paused`); pausing halts event delivery
            without stopping the underlying thread.
        jitter_fraction: Optional fraction of ``interval_seconds`` used
            to randomize each tick's actual delay, in the range
            ``[0, 1]``. ``0`` (the default) disables jitter. ``0.1``
            means each tick fires within +/- 10% of the interval.
        name: Human-readable name (defaults to ``id`` when not given).
        metadata: Optional free-form metadata for log enrichment.
    """

    id: str
    interval_seconds: float
    enabled: bool = True
    jitter_fraction: float = 0.0
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id(self.id)
        _validate_positive_interval(self.interval_seconds, what="interval_seconds")
        _validate_jitter(self.jitter_fraction)
        if self.name is not None and not isinstance(self.name, str):
            raise ValueError("name must be a string when provided")
        if self.metadata and not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping when provided")
        if self.metadata:
            for key in self.metadata:
                if not isinstance(key, str) or not key:
                    raise ValueError(f"metadata keys must be non-empty strings: {key!r}")

    @property
    def display_name(self) -> str:
        return self.name or self.id


@dataclass(frozen=True)
class TickEvent:
    """A single tick fired by the scheduler.

    Attributes:
        scheduler_id: The id of the :class:`TickConfig` that fired.
        tick_number: Monotonic counter, starting at 1 for the first tick.
        scheduled_at: Wall-clock timestamp the tick was *intended* to
            fire (in epoch seconds). With jitter, ``actual_at`` may
            differ slightly; ``scheduled_at`` is the canonical cadence.
        actual_at: Wall-clock timestamp the tick actually fired.
        jitter_applied: How much jitter (in seconds) was added to the
            scheduled time to produce the actual fire time. Zero when
            jitter is disabled.
    """

    scheduler_id: str
    tick_number: int
    scheduled_at: float
    actual_at: float
    jitter_applied: float = 0.0

    @property
    def drift(self) -> float:
        """Actual fire time minus scheduled time (in seconds)."""
        return self.actual_at - self.scheduled_at


@dataclass(frozen=True)
class BriefSummarySnapshot:
    """Snapshot of agent state used by :class:`BriefSummaryBuilder`.

    Attributes:
        agent_id: Identifier of the agent emitting the brief.
        session_id: Identifier of the current session.
        tick_number: The most recent tick the agent has processed.
        pending_tasks: Optional list of in-flight task descriptions.
        last_action: Optional description of the most recent action.
        metadata: Optional free-form metadata for log enrichment.
        captured_at: Wall-clock timestamp the snapshot was captured
            (in epoch seconds). Defaults to ``time.time()`` when not
            provided, but tests can pin it for deterministic output.
    """

    agent_id: str
    session_id: str
    tick_number: int = 0
    pending_tasks: tuple[str, ...] = ()
    last_action: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    captured_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _validate_id(self.agent_id, what="agent_id")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.tick_number, int) or self.tick_number < 0:
            raise ValueError(f"tick_number must be a non-negative int (got {self.tick_number!r})")
        if self.pending_tasks and not isinstance(self.pending_tasks, tuple):
            # Coerce silently for ergonomics; the public API expects a tuple.
            object.__setattr__(self, "pending_tasks", tuple(self.pending_tasks))
        if self.last_action is not None and not isinstance(self.last_action, str):
            raise ValueError("last_action must be a string when provided")
        if self.metadata and not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping when provided")


@dataclass(frozen=True)
class DailyLogEntry:
    """One entry to be appended to a daily log file.

    Attributes:
        timestamp: ISO 8601 local timestamp string. The caller is
            responsible for formatting it (the writer does not pin a
            timezone so callers can choose local or UTC).
        body: Markdown body of the entry (may contain newlines).
        tags: Optional tuple of tag strings, joined with ``#`` for
            filtering downstream.
    """

    timestamp: str
    body: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, str) or not self.timestamp:
            raise ValueError("timestamp must be a non-empty ISO 8601 string")
        if not isinstance(self.body, str):
            raise TypeError("body must be a string")
        if self.tags and not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(self.tags))
        for tag in self.tags:
            if not isinstance(tag, str) or not tag:
                raise ValueError("tags must be non-empty strings")

    def render(self) -> str:
        """Render the entry as a Markdown fragment."""
        out = f"## {self.timestamp}\n\n{self.body.rstrip()}"
        if self.tags:
            tag_str = " ".join(f"#{t}" for t in self.tags)
            out += f"\n\n{tag_str}"
        return out + "\n"


def format_local_timestamp(when: float | None = None) -> str:
    """Format a wall-clock timestamp as an ISO 8601 local string."""
    if when is None:
        when = time.time()
    return datetime.fromtimestamp(when, tz=timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "BriefSummarySnapshot",
    "DailyLogEntry",
    "TickConfig",
    "TickEvent",
    "format_local_timestamp",
]
