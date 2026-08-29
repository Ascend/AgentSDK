#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""P65-B — :class:`LangfuseSink` analytics sink.

Translates :class:`AnalyticsEvent` records into Langfuse traces,
spans, and generations, and keeps a local in-memory buffer of the
same records so :class:`TrainingDataExporter` can serialise them
to JSONL for SFT / DPO pipelines.

The sink is **degradation-safe**: when the ``langfuse`` SDK is
not importable, the credentials are not set, or the live Langfuse
HTTP call raises, the sink silently falls through to a no-op while
still appending to the local buffer. This matches the
:class:`NullSink` contract — observers of the global analytics
sink must never see a crash from this implementation.

Trace model
-----------
A Langfuse **trace** is the top-level grouping; the sink creates
one trace per :data:`EventType.SESSION_START` and reuses it for
all events with the same ``session_id`` until :data:`EventType.SESSION_END`
fires. This mirrors how a single user session maps to one
Langfuse trace, with sub-events becoming spans / generations.

Spans that have explicit start + end events (``AGENT_SPAWN`` →
``AGENT_COMPLETE``) are paired by ``agent_id`` so the resulting
span has both timestamps.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from ..analytics.events import AnalyticsEvent, EventType
from ..analytics.sink import AnalyticsSink
from .client import get_langfuse_client, is_langfuse_available

if TYPE_CHECKING:
    from langfuse import Langfuse  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


# Hard cap on the local buffer so a long session cannot leak memory.
# 50k records is roughly 1k turns * 50 events/turn — generous, but
# bounded. The training-data exporter should be invoked before the
# buffer is full for real workloads.
_BUFFER_MAXLEN: int = 50_000


class LangfuseSink(AnalyticsSink):
    """Forwards :class:`AnalyticsEvent` into a Langfuse project.

    Parameters
    ----------
    client:
        Optional pre-initialised ``langfuse.Langfuse`` instance.
        When ``None`` (the default) the sink resolves the client
        lazily on the first :meth:`emit` via
        :func:`get_langfuse_client`.
    buffer_maxlen:
        Override the default in-memory buffer cap. Tests use a
        small value to keep fixtures tight.
    """

    def __init__(
        self,
        client: Langfuse | None = None,
        *,
        buffer_maxlen: int = _BUFFER_MAXLEN,
    ) -> None:
        self._explicit_client = client
        self._lock = threading.RLock()
        self._traces: dict[str, Any] = {}
        self._pending_agent_spans: dict[str, dict[str, Any]] = {}
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_maxlen)

    # -- Client resolution --------------------------------------------------

    def _resolve_client(self) -> Any:
        """Return the active Langfuse client, preferring the
        explicit one passed at construction time.
        """
        if self._explicit_client is not None:
            return self._explicit_client
        return get_langfuse_client()

    # -- emit ---------------------------------------------------------------

    def emit(self, event: AnalyticsEvent) -> None:
        """Dispatch ``event`` to the right Langfuse primitive.

        Failures are logged at warning level and swallowed — the
        sink must never crash the caller. The local buffer is
        updated regardless of whether the live Langfuse call
        succeeded, so :class:`TrainingDataExporter` always has
        data to work with.
        """
        record = self._record_from_event(event)
        with self._lock:
            self._buffer.append(record)

        client = self._resolve_client()
        if client is None:
            return

        try:
            self._dispatch(event, record, client)
        except Exception as exc:
            logger.warning(
                "langfuse dispatch failed for event %s: %s",
                event.type.value,
                exc,
            )

    def _dispatch(
        self,
        event: AnalyticsEvent,
        record: dict[str, Any],
        client: Any,
    ) -> None:
        """Map ``event`` onto the correct Langfuse primitive."""
        session_id = event.session_id
        trace = self._trace_for(client, session_id)

        etype = event.type
        if etype is EventType.TURN_END:
            self._emit_generation(trace, event, record)
        elif etype is EventType.AGENT_SPAWN:
            self._open_agent_span(trace, event, record)
        elif etype is EventType.AGENT_COMPLETE:
            self._close_agent_span(trace, event, record)
        elif etype is EventType.TOOL_USE:
            self._emit_span(trace, event, record, kind="tool_use")
        elif etype is EventType.SESSION_START:
            # The trace was created lazily by _trace_for; emit a
            # domain event so the trace's event list is not empty.
            self._emit_event(trace, event, record)
        elif etype is EventType.SESSION_END:
            # Drop the cached trace so the next SESSION_START creates
            # a fresh one.
            self._close_session_trace(session_id)
            self._emit_event(trace, event, record)
        else:
            self._emit_event(trace, event, record)

    # -- trace lifecycle ----------------------------------------------------

    def _trace_for(self, client: Any, session_id: str) -> Any | None:
        """Return (and lazily create) the Langfuse trace for ``session_id``.

        When ``session_id`` is empty, we still create a trace with
        a synthetic id so events that lack a session context are
        not silently dropped.
        """
        if not session_id:
            session_id = f"ad-hoc-{int(time.time() * 1000)}"

        with self._lock:
            existing = self._traces.get(session_id)
            if existing is not None:
                return existing

        trace = client.trace(
            name="clawcodex-session",
            session_id=session_id,
            metadata={"source": "clawcodex.analytics"},
        )
        with self._lock:
            self._traces[session_id] = trace
        return trace

    def _close_session_trace(self, session_id: str) -> None:
        """Forget the cached trace for ``session_id`` (best-effort)."""
        with self._lock:
            self._traces.pop(session_id, None)

    # -- primitive emitters -------------------------------------------------

    def _emit_generation(
        self,
        trace: Any,
        event: AnalyticsEvent,
        record: dict[str, Any],
    ) -> None:
        """Translate a TURN_END event into a Langfuse generation.

        The generation surfaces:

        * model name (from ``event.model``)
        * prompt (from ``data["prompt"]``)
        * completion (from ``data["completion"]``)
        * usage tokens (from ``data["usage"]`` dict, optional)
        * latency (from ``data["latency_ms"]`` or computed from
          ``event.timestamp`` minus a stored start — fall back to 0).
        """
        usage = event.data.get("usage") or {}
        metadata = {
            "model": event.model,
            **{k: v for k, v in event.data.items() if k not in {"prompt", "completion", "usage", "latency_ms"}},
        }
        trace.generation(
            name=event.data.get("name", "model-call"),
            model=event.model or event.data.get("model", "unknown"),
            input=event.data.get("prompt"),
            output=event.data.get("completion"),
            usage=usage if isinstance(usage, dict) else None,
            metadata=metadata,
            start_time=record.get("start_time"),
            end_time=record.get("end_time"),
        )

    def _emit_span(
        self,
        trace: Any,
        event: AnalyticsEvent,
        record: dict[str, Any],
        *,
        kind: str,
    ) -> None:
        """Generic span emission (TOOL_USE, ad-hoc spans)."""
        trace.span(
            name=event.data.get("name", kind),
            input=event.data.get("input"),
            output=event.data.get("output"),
            metadata={"kind": kind, **event.data},
        )

    def _emit_event(
        self,
        trace: Any,
        event: AnalyticsEvent,
        record: dict[str, Any],
    ) -> None:
        """Domain-level event (anything that isn't a generation or span)."""
        trace.event(
            name=event.type.value,
            input=event.data,
            metadata={"session_id": event.session_id, "model": event.model},
        )

    def _open_agent_span(
        self,
        trace: Any,
        event: AnalyticsEvent,
        record: dict[str, Any],
    ) -> None:
        """Start an agent span; remembered by ``agent_id`` so the
        matching :data:`EventType.AGENT_COMPLETE` can close it.
        """
        agent_id = event.data.get("agent_id") or event.session_id
        span = trace.span(
            name=event.data.get("name", "agent"),
            input=event.data.get("input"),
            metadata={"agent_id": agent_id, "kind": "agent"},
            start_time=event.timestamp,
        )
        with self._lock:
            self._pending_agent_spans[agent_id] = {
                "span": span,
                "trace": trace,
                "start_time": event.timestamp,
            }

    def _close_agent_span(
        self,
        trace: Any,
        event: AnalyticsEvent,
        record: dict[str, Any],
    ) -> None:
        """End the agent span opened by the matching ``AGENT_SPAWN``."""
        agent_id = event.data.get("agent_id") or event.session_id
        with self._lock:
            pending = self._pending_agent_spans.pop(agent_id, None)
        if pending is None:
            # No matching open — fall through to a free-floating span
            # so the data still shows up in Langfuse.
            self._emit_span(trace, event, record, kind="agent_complete")
            return

        span = pending["span"]
        # The Langfuse SDK's Span exposes ``end`` for explicit closure.
        end = getattr(span, "end", None)
        if callable(end):
            end(
                output=event.data.get("output"),
                status_message=event.data.get("status_message"),
                metadata=event.data,
            )

    # -- buffer helpers -----------------------------------------------------

    @staticmethod
    def _record_from_event(event: AnalyticsEvent) -> dict[str, Any]:
        """Project an :class:`AnalyticsEvent` into the dict shape
        the :class:`TrainingDataExporter` consumes.
        """
        start = event.data.get("start_time")
        end = event.timestamp
        latency_ms = event.data.get("latency_ms")
        if latency_ms is None and isinstance(start, (int, float)):
            latency_ms = max(0.0, (end - float(start)) * 1000.0)

        return {
            "type": event.type.value,
            "session_id": event.session_id,
            "model": event.model,
            "name": event.data.get("name", event.type.value),
            "input": event.data.get("input") or event.data.get("prompt"),
            "output": event.data.get("output") or event.data.get("completion"),
            "usage": event.data.get("usage") or {},
            "metadata": {
                k: v
                for k, v in event.data.items()
                if k
                not in {
                    "name",
                    "input",
                    "output",
                    "prompt",
                    "completion",
                    "usage",
                    "latency_ms",
                    "start_time",
                }
            },
            "agent_id": event.data.get("agent_id", ""),
            "start_time": start if isinstance(start, (int, float)) else None,
            "end_time": end,
            "latency_ms": latency_ms,
            "level": event.data.get("level", "DEFAULT"),
            "status_message": event.data.get("status_message", ""),
            "timestamp": event.timestamp,
        }

    # -- buffer access (for the exporter) -----------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the current buffer.

        The exporter should call this rather than reaching into
        ``_buffer`` directly so the lock is acquired.
        """
        with self._lock:
            return list(self._buffer)

    def clear_buffer(self) -> None:
        """Drop the in-memory buffer (used by tests + after export)."""
        with self._lock:
            self._buffer.clear()

    # -- lifecycle ----------------------------------------------------------

    def flush(self) -> None:
        """Forward ``flush()`` to the live client if available."""
        client = self._resolve_client()
        flush = getattr(client, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception as exc:
                logger.warning("langfuse flush failed: %s", exc)

    def close(self) -> None:
        """Flush + shut the live client (best-effort)."""
        self.flush()
        client = self._resolve_client()
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                logger.warning("langfuse shutdown failed: %s", exc)

    # -- introspection helpers ---------------------------------------------

    def is_live(self) -> bool:
        """True iff a live Langfuse client is currently reachable.

        An explicit client passed at construction time is always
        considered live — the caller is asserting "I have a real
        client, just use it". The default (no explicit client)
        path defers to :func:`is_langfuse_available` which checks
        the SDK + credentials.
        """
        if self._explicit_client is not None:
            return True
        return is_langfuse_available() and self._resolve_client() is not None


__all__ = ["LangfuseSink"]
