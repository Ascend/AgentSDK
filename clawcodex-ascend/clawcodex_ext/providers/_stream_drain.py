"""Shared worker-thread + queue-poll stream drain for cancel latency.

Both :class:`ClawcodexAnthropicProvider` and
:class:`ClawcodexMinimaxProvider` (and the upstream
``openai_compatible.py``) wrap the Anthropic / OpenAI SDK's
``messages.stream`` context manager, whose ``text_stream`` is a
synchronous generator backed by a blocking httpx socket read.

``response.close()`` from the keypress thread is best-effort: on
some platforms and LiteLLM-proxied connections the blocking read
does NOT return when the response is closed, and the user would
otherwise wait the full HTTP read timeout (~60s) before a Shift+Tab
or Ctrl+C cancel lands.

This module hoists the synchronous iteration onto a daemon worker
thread and lets the main thread poll a queue with a 100ms timeout,
re-checking the abort guard on every tick. Cancel latency is
bounded to ~100ms regardless of whether the underlying socket close
is honored — the same pattern that ``openai_compatible.py:622-660``
ships with.
"""

from __future__ import annotations

import queue as _queue
import threading as _threading
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from clawcodex_ext.providers._stream_abort import StreamAbortGuard
    from src.utils.stream_watchdog import StreamWatchdog


def drain_text_stream_with_abort_poll(
    stream: Any,
    *,
    guard: "StreamAbortGuard",
    on_text: Callable[[str], None] | None = None,
    watchdog: "StreamWatchdog | None" = None,
    stream_name: str = "stream",
) -> None:
    """Drain ``stream.text_stream`` on a worker thread; poll the main
    thread with a 100ms timeout so abort signals land promptly.

    Args:
        stream: Any object exposing a synchronous ``text_stream``
            generator (Anthropic / Minimax SDK ``MessageStream``).
        guard: :class:`StreamAbortGuard` wrapping the cancel signal.
            ``guard.aborted`` is checked on every queue-poll tick; on
            True we call ``guard.raise_if_post_aborted()`` so the
            controller's abort reason (``"user_interrupt"`` etc.) is
            preserved.
        on_text: Optional callback invoked once per non-empty chunk.
            Runs on the **main** thread, not the worker.
        watchdog: Optional :class:`StreamWatchdog`; ``reset()`` is
            called on every delivered chunk so the idle deadline only
            fires when the stream genuinely stalls.
        stream_name: Used in the worker thread's ``name`` attribute
            for debuggability (e.g. ``anthropic-stream-140234``).

    Returns:
        None. Raises ``AbortError`` (via ``guard``) on user cancel,
        or re-raises any exception surfaced by the underlying
        ``text_stream`` generator (network errors, SDK errors, etc.).
    """
    _DONE = object()
    chunk_queue: _queue.Queue = _queue.Queue()

    def _drain() -> None:
        try:
            for text in stream.text_stream:
                chunk_queue.put(text)
        except BaseException as exc:  # noqa: BLE001 — surface to consumer
            chunk_queue.put(exc)
        finally:
            chunk_queue.put(_DONE)

    worker = _threading.Thread(
        target=_drain,
        daemon=True,
        name=f"{stream_name}-{id(stream)}",
    )
    worker.start()
    while True:
        try:
            item = chunk_queue.get(timeout=0.1)
        except _queue.Empty:
            # No chunk in the last 100ms — re-check abort and loop.
            # This is what bounds cancel latency to ~100ms.
            if guard.aborted:
                guard.raise_if_post_aborted()
            continue
        if item is _DONE:
            break
        if isinstance(item, BaseException):
            if isinstance(item, Exception):
                guard.reraise_if_aborted(item)
                raise item
            # KeyboardInterrupt / SystemExit — re-raise as-is.
            raise item
        text = item
        if watchdog is not None:
            watchdog.reset()
        if not text:
            continue
        if on_text is not None:
            on_text(text)


def drain_event_stream_with_abort_poll(
    stream: Any,
    *,
    guard: "StreamAbortGuard",
    on_event: Callable[[Any], None],
    watchdog: "StreamWatchdog | None" = None,
    stream_name: str = "event-stream",
) -> None:
    """Drain the full SDK event stream while polling for user aborts.

    Every event is delivered on the consumer thread. This lets thinking,
    tool-input and lifecycle events reset the watchdog while the caller alone
    decides which deltas belong in the visible text channel.
    """

    done = object()
    event_queue: _queue.Queue = _queue.Queue()

    def _drain() -> None:
        try:
            for event in stream:
                event_queue.put(event)
        except BaseException as exc:  # noqa: BLE001 - surface to consumer
            event_queue.put(exc)
        finally:
            event_queue.put(done)

    worker = _threading.Thread(
        target=_drain,
        daemon=True,
        name=f"{stream_name}-{id(stream)}",
    )
    worker.start()
    while True:
        try:
            item = event_queue.get(timeout=0.1)
        except _queue.Empty:
            if guard.aborted:
                guard.raise_if_post_aborted()
            continue
        if item is done:
            break
        if isinstance(item, BaseException):
            if isinstance(item, Exception):
                guard.reraise_if_aborted(item)
                raise item
            raise item
        if watchdog is not None:
            watchdog.reset()
        on_event(item)


__all__ = ["drain_event_stream_with_abort_poll", "drain_text_stream_with_abort_poll"]
