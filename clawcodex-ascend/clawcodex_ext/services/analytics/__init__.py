"""Analytics subsystem.

Event logging, session metadata, and event sinks.
Mirrors TypeScript analytics/ directory.
"""

from __future__ import annotations

from .events import AnalyticsEvent, EventType, get_analytics_sink, log_event, set_analytics_sink
from .metadata import SessionAnalyticsMetadata, collect_session_metadata
from .sink import AnalyticsSink, ConsoleSink, FileSink, NullSink

__all__ = [
    "AnalyticsEvent",
    "AnalyticsSink",
    "ConsoleSink",
    "EventType",
    "FileSink",
    "NullSink",
    "SessionAnalyticsMetadata",
    "collect_session_metadata",
    "get_analytics_sink",
    "log_event",
    "set_analytics_sink",
]
