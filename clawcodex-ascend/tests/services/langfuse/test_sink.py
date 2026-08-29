#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for src/services/langfuse/sink.py (P65-B)."""
# pylint: disable=no-name-in-module, use-implicit-booleaness-not-comparison

from __future__ import annotations

import threading
from typing import Any

import pytest

from clawcodex_ext.services.analytics.events import AnalyticsEvent, EventType
from src.services.langfuse.sink import LangfuseSink


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSpan:
    def __init__(self, trace: "_FakeTrace", name: str) -> None:
        self.trace = trace
        self.name = name
        self.ended = False
        self.end_kwargs: dict[str, Any] = {}

    def end(self, **kwargs: Any) -> None:
        self.ended = True
        self.end_kwargs = kwargs


class _FakeTrace:
    def __init__(self, *, name: str, session_id: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.session_id = session_id
        self.metadata = metadata
        self.generations: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._spans: list[_FakeSpan] = []

    def generation(self, **kwargs: Any) -> None:
        self.generations.append(kwargs)

    def span(self, **kwargs: Any) -> _FakeSpan:
        self.spans.append(kwargs)
        span = _FakeSpan(self, kwargs.get("name", ""))
        self._spans.append(span)
        return span

    def event(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class _FakeClient:
    def __init__(self) -> None:
        self.traces: list[_FakeTrace] = []
        self.flush_count = 0
        self.shutdown_count = 0
        self.flush_exc: Exception | None = None
        self.shutdown_exc: Exception | None = None

    def trace(self, *, name: str, session_id: str, metadata: dict[str, Any]) -> _FakeTrace:
        trace = _FakeTrace(name=name, session_id=session_id, metadata=metadata)
        self.traces.append(trace)
        return trace

    def flush(self) -> None:
        self.flush_count += 1
        if self.flush_exc is not None:
            raise self.flush_exc

    def shutdown(self) -> None:
        self.shutdown_count += 1
        if self.shutdown_exc is not None:
            raise self.shutdown_exc


# ---------------------------------------------------------------------------
# Tests — emit & buffer
# ---------------------------------------------------------------------------


def test_emit_appends_to_buffer_without_live_client() -> None:
    """No client passed → buffer still grows; live dispatch is a no-op."""
    sink = LangfuseSink()
    event = AnalyticsEvent(type=EventType.TURN_END, session_id="s1", model="claude-x")
    sink.emit(event)
    snap = sink.snapshot()
    assert len(snap) == 1
    assert snap[0]["type"] == "turn_end"
    assert snap[0]["session_id"] == "s1"


def test_emit_dispatches_turn_end_as_generation() -> None:
    fake = _FakeClient()
    sink = LangfuseSink(client=fake)
    sink.emit(
        AnalyticsEvent(
            type=EventType.TURN_END,
            session_id="s1",
            model="claude-x",
            data={
                "prompt": "hello",
                "completion": "world",
                "usage": {"input": 5, "output": 3},
                "latency_ms": 120.0,
            },
        )
    )
    # We need a SESSION_START or _trace_for to fire to attach the trace.
    sink.emit(AnalyticsEvent(type=EventType.TURN_END, session_id="s1", model="claude-x"))
    assert len(fake.traces) >= 1
    trace = fake.traces[0]
    assert len(trace.generations) >= 1
    gen = trace.generations[0]
    assert gen["model"] == "claude-x"
    assert gen["input"] == "hello"
    assert gen["output"] == "world"
    assert gen["usage"] == {"input": 5, "output": 3}


def test_emit_dispatches_tool_use_as_span() -> None:
    fake = _FakeClient()
    sink = LangfuseSink(client=fake)
    sink.emit(
        AnalyticsEvent(
            type=EventType.TOOL_USE,
            session_id="s1",
            data={"name": "Read", "input": {"file": "x.py"}, "output": "..."},
        )
    )
    trace = fake.traces[0]
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span["name"] == "Read"
    assert span["metadata"]["kind"] == "tool_use"


def test_emit_dispatches_other_events_as_event() -> None:
    fake = _FakeClient()
    sink = LangfuseSink(client=fake)
    sink.emit(
        AnalyticsEvent(
            type=EventType.MODEL_SWITCH,
            session_id="s1",
            data={"from": "haiku", "to": "opus"},
        )
    )
    trace = fake.traces[0]
    assert len(trace.events) == 1
    ev = trace.events[0]
    assert ev["name"] == "model_switch"
    assert ev["input"] == {"from": "haiku", "to": "opus"}


def test_agent_spawn_then_complete_pairs() -> None:
    fake = _FakeClient()
    sink = LangfuseSink(client=fake)

    sink.emit(
        AnalyticsEvent(
            type=EventType.AGENT_SPAWN,
            session_id="s1",
            data={"agent_id": "a1", "name": "Explore", "input": "go"},
        )
    )
    sink.emit(
        AnalyticsEvent(
            type=EventType.AGENT_COMPLETE,
            session_id="s1",
            data={"agent_id": "a1", "output": "found 2", "status_message": "ok"},
        )
    )

    trace = fake.traces[0]
    # Two spans total: the open + the agent_complete event (we currently
    # pass the AGENT_COMPLETE through _emit_span as a free-floating span
    # in the unmatched case — for the matched case it's a span.end()).
    matched_span = trace._spans[0]
    assert matched_span.ended is True
    assert matched_span.end_kwargs["output"] == "found 2"
    assert matched_span.end_kwargs["status_message"] == "ok"


def test_agent_complete_without_spawn_falls_through() -> None:
    """An orphan AGENT_COMPLETE emits a free-floating span rather
    than crashing.
    """
    fake = _FakeClient()
    sink = LangfuseSink(client=fake)
    sink.emit(
        AnalyticsEvent(
            type=EventType.AGENT_COMPLETE,
            session_id="s1",
            data={"agent_id": "ghost", "output": "..."},
        )
    )
    trace = fake.traces[0]
    assert len(trace.spans) == 1
    assert trace.spans[0]["metadata"]["kind"] == "agent_complete"


def test_buffer_is_bounded() -> None:
    sink = LangfuseSink(buffer_maxlen=3)
    for i in range(5):
        sink.emit(AnalyticsEvent(type=EventType.TURN_END, session_id=f"s{i}", model="claude-x"))
    snap = sink.snapshot()
    assert len(snap) == 3
    # The most recent three events should be retained.
    assert snap[-1]["session_id"] == "s4"


def test_snapshot_returns_defensive_copy() -> None:
    sink = LangfuseSink()
    sink.emit(AnalyticsEvent(type=EventType.TURN_END, session_id="s1"))
    snap = sink.snapshot()
    snap.clear()
    # The internal buffer is unchanged.
    assert len(sink.snapshot()) == 1


def test_clear_buffer_drops_all_records() -> None:
    sink = LangfuseSink()
    sink.emit(AnalyticsEvent(type=EventType.TURN_END, session_id="s1"))
    sink.clear_buffer()
    assert sink.snapshot() == []


# ---------------------------------------------------------------------------
# Tests — dispatch resilience
# ---------------------------------------------------------------------------


def test_emit_swallows_dispatch_errors() -> None:
    """A live-client that raises must not crash the sink or the buffer."""

    class _RaisingClient(_FakeClient):
        def trace(self, **_kwargs: Any) -> _FakeTrace:  # type: ignore[override]
            raise RuntimeError("kaboom")

    sink = LangfuseSink(client=_RaisingClient())
    sink.emit(AnalyticsEvent(type=EventType.TURN_END, session_id="s1"))
    # Buffer still received the record.
    assert len(sink.snapshot()) == 1


def test_session_lifecycle_creates_and_drops_traces() -> None:
    """SESSION_START creates a trace, SESSION_END drops it from the cache."""
    fake = _FakeClient()
    sink = LangfuseSink(client=fake)
    sink.emit(AnalyticsEvent(type=EventType.SESSION_START, session_id="s1"))
    sink.emit(AnalyticsEvent(type=EventType.TURN_END, session_id="s1", model="m"))
    sink.emit(AnalyticsEvent(type=EventType.SESSION_END, session_id="s1"))
    sink.emit(AnalyticsEvent(type=EventType.SESSION_START, session_id="s1"))
    # Two distinct traces for s1 (SESSION_END cleared the cache).
    assert len(fake.traces) == 2


def test_record_carries_input_output_and_usage() -> None:
    sink = LangfuseSink()
    sink.emit(
        AnalyticsEvent(
            type=EventType.TURN_END,
            session_id="s1",
            model="claude-x",
            data={
                "prompt": "hi",
                "completion": "hello",
                "usage": {"input": 1, "output": 2},
                "extra": "keep-me",
            },
        )
    )
    record = sink.snapshot()[0]
    assert record["input"] == "hi"
    assert record["output"] == "hello"
    assert record["usage"] == {"input": 1, "output": 2}
    assert record["metadata"]["extra"] == "keep-me"


def test_record_computes_latency_when_only_start_given() -> None:
    sink = LangfuseSink()
    sink.emit(
        AnalyticsEvent(
            type=EventType.TURN_END,
            session_id="s1",
            timestamp=1000.0,
            data={"start_time": 999.0, "prompt": "x", "completion": "y"},
        )
    )
    record = sink.snapshot()[0]
    # 1 second gap → 1000 ms.
    assert record["latency_ms"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Tests — flush / close / is_live
# ---------------------------------------------------------------------------


def test_flush_calls_client_flush() -> None:
    fake = _FakeClient()
    sink = LangfuseSink(client=fake)
    sink.flush()
    assert fake.flush_count == 1


def test_flush_swallows_client_errors() -> None:
    fake = _FakeClient()
    fake.flush_exc = RuntimeError("flush failed")
    sink = LangfuseSink(client=fake)
    sink.flush()  # must not raise


def test_close_flushes_and_shuts_down() -> None:
    fake = _FakeClient()
    sink = LangfuseSink(client=fake)
    sink.close()
    assert fake.flush_count == 1
    assert fake.shutdown_count == 1


def test_is_live_true_with_explicit_client() -> None:
    sink = LangfuseSink(client=_FakeClient())
    assert sink.is_live() is True


def test_is_live_false_without_client() -> None:
    sink = LangfuseSink()
    assert sink.is_live() is False


# ---------------------------------------------------------------------------
# Tests — thread safety
# ---------------------------------------------------------------------------


def test_concurrent_emit_does_not_corrupt_buffer() -> None:
    sink = LangfuseSink(buffer_maxlen=10_000)
    n = 200
    errors: list[BaseException] = []

    def _worker(start: int) -> None:
        try:
            for i in range(start, start + n):
                sink.emit(
                    AnalyticsEvent(
                        type=EventType.TURN_END,
                        session_id=f"s{i}",
                        model="claude-x",
                    )
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(t * n,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # 4 * 200 events buffered (bounded at 10k so all retained).
    assert len(sink.snapshot()) == n * 4
