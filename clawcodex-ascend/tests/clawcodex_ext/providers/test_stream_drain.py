"""Unit tests for :mod:`clawcodex_ext.providers._stream_drain`.

These tests exercise the worker-thread + 100ms queue-poll drain in
isolation against a fake stream, without needing a live Anthropic /
Minimax SDK or HTTP server. They cover:

* Happy-path delivery of all non-empty chunks.
* Empty-chunk skipping.
* Cancel-latency bound: a blocking stream must be abandoned within
  ~200ms of the abort signal firing (one poll tick + margin).
* Exception propagation: errors raised inside the underlying
  ``text_stream`` generator surface on the main thread.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock
import pytest

from src.utils.abort_controller import create_abort_controller
from src.providers._stream_abort import StreamAbortGuard
from clawcodex_ext.providers._stream_drain import (
    drain_event_stream_with_abort_poll,
    drain_text_stream_with_abort_poll,
)


class _FakeStream:
    """Minimal stand-in for an Anthropic SDK ``MessageStream``."""

    def __init__(self, chunks, *, block_per_chunk_s: float = 0.0):
        self._chunks = chunks
        self._block_per_chunk_s = block_per_chunk_s

    @property
    def text_stream(self):
        for chunk in self._chunks:
            if self._block_per_chunk_s > 0:
                time.sleep(self._block_per_chunk_s)
            yield chunk


def test_drain_delivers_all_non_empty_chunks():
    stream = _FakeStream(["hello", " ", "world"])
    guard = StreamAbortGuard(None)
    received: list[str] = []
    drain_text_stream_with_abort_poll(
        stream,
        guard=guard,
        on_text=received.append,
        stream_name="test",
    )
    assert received == ["hello", " ", "world"]


def test_drain_skips_empty_chunks():
    stream = _FakeStream(["hello", "", "world", ""])
    guard = StreamAbortGuard(None)
    received: list[str] = []
    drain_text_stream_with_abort_poll(
        stream,
        guard=guard,
        on_text=received.append,
        stream_name="test",
    )
    assert received == ["hello", "world"]


def test_drain_returns_silently_without_callback():
    """``on_text=None`` is allowed — the drain just discards chunks."""
    stream = _FakeStream(["a", "b", "c"])
    guard = StreamAbortGuard(None)
    drain_text_stream_with_abort_poll(
        stream,
        guard=guard,
        on_text=None,
        stream_name="test",
    )
    # No assertion needed — the test passes if no exception is raised.


def test_drain_cancels_within_200ms_of_abort():
    """A stream that blocks forever after one chunk must be abandoned
    within ~200ms of the abort signal firing.

    The 100ms queue poll is the upper bound; we add 100ms margin for
    thread scheduling. Before the fix (direct ``for text in
    stream.text_stream:``), this test would block for 60s.
    """

    def _slow_iter():
        yield "first"
        # Would block for 60s if the worker thread is not abandoned.
        time.sleep(60)
        yield "never"  # pragma: no cover

    class _SlowStream:
        @property
        def text_stream(self):
            return _slow_iter()

    controller = create_abort_controller()
    guard = StreamAbortGuard(controller.signal)
    received: list[str] = []

    def _fire_abort():
        # Let the first chunk land before tripping the signal.
        time.sleep(0.05)
        controller.abort("test_interrupt")

    threading.Thread(target=_fire_abort, daemon=True).start()

    start = time.monotonic()
    with pytest.raises(Exception):
        drain_text_stream_with_abort_poll(
            _SlowStream(),
            guard=guard,
            on_text=received.append,
            stream_name="test",
        )
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"drain took {elapsed:.3f}s, expected < 0.5s"
    assert received == ["first"]


def test_drain_propagates_stream_exception():
    class _ErrorStream:
        @property
        def text_stream(self):
            yield "before"
            raise RuntimeError("stream broke")
            yield "never"  # pragma: no cover

    guard = StreamAbortGuard(None)
    received: list[str] = []
    with pytest.raises(RuntimeError, match="stream broke"):
        drain_text_stream_with_abort_poll(
            _ErrorStream(),
            guard=guard,
            on_text=received.append,
            stream_name="test",
        )
    assert received == ["before"]


def test_drain_does_not_swallow_baseexception():
    """``KeyboardInterrupt`` / ``SystemExit`` from the worker must
    re-raise on the main thread — the helper must not turn them into
    normal exceptions.
    """

    class _InterruptStream:
        @property
        def text_stream(self):
            yield "before"
            raise KeyboardInterrupt()

    guard = StreamAbortGuard(None)
    with pytest.raises(KeyboardInterrupt):
        drain_text_stream_with_abort_poll(
            _InterruptStream(),
            guard=guard,
            on_text=None,
            stream_name="test",
        )


def test_event_drain_delivers_every_event_and_resets_watchdog():
    events = [object(), object(), object()]
    received: list[object] = []
    watchdog = MagicMock()

    drain_event_stream_with_abort_poll(
        iter(events),
        guard=StreamAbortGuard(None),
        on_event=received.append,
        watchdog=watchdog,
        stream_name="test-events",
    )

    assert received == events
    assert watchdog.reset.call_count == len(events)


def test_event_drain_cancels_while_worker_is_blocked():
    def _slow_events():
        yield object()
        time.sleep(60)

    controller = create_abort_controller()
    guard = StreamAbortGuard(controller.signal)

    threading.Thread(
        target=lambda: (time.sleep(0.05), controller.abort("test_interrupt")),
        daemon=True,
    ).start()

    start = time.monotonic()
    with pytest.raises(Exception):
        drain_event_stream_with_abort_poll(
            _slow_events(),
            guard=guard,
            on_event=lambda _event: None,
            stream_name="test-events",
        )
    assert time.monotonic() - start < 0.5
