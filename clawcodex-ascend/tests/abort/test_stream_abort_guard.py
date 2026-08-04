#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Unit tests for ``src/providers/_stream_abort.py``.

The provider-level tests in ``test_provider_abort_signal.py``,
``test_openai_compat_abort_signal.py``, and
``test_minimax_abort_signal.py`` already cover the end-to-end behavior
through each provider's ``chat_stream_response`` path. This file pins
the helper's contract directly so a future refactor that changes one
provider but forgets to update the helper (or vice versa) fails fast
at the unit level.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.providers._stream_abort import StreamAbortGuard, _close_response_safely
from src.utils.abort_controller import AbortController, AbortError


def _make_stream() -> MagicMock:
    """Build a stub stream with a ``response.close`` we can assert on."""
    stream = MagicMock()
    stream.response = MagicMock()
    return stream


# ---------------------------------------------------------------------------
# Pre-call / post-call fast-paths


def test_raise_if_pre_aborted_no_signal_is_noop() -> None:
    """A guard with ``abort_signal=None`` does not raise on the pre-call check.

    Providers can build a guard unconditionally — callers that don't
    pass an abort signal just get a guard that does nothing.
    """
    StreamAbortGuard(None).raise_if_pre_aborted()  # no exception


def test_raise_if_pre_aborted_signal_clear_is_noop() -> None:
    controller = AbortController()
    StreamAbortGuard(controller.signal).raise_if_pre_aborted()  # no exception


def test_raise_if_pre_aborted_signal_set_raises_abort_error() -> None:
    controller = AbortController()
    controller.abort("user_interrupt")
    guard = StreamAbortGuard(controller.signal)
    with pytest.raises(AbortError) as exc_info:
        guard.raise_if_pre_aborted()
    assert exc_info.value.reason == "user_interrupt"


def test_raise_if_post_aborted_mirrors_pre_check() -> None:
    """Post-stream recheck has the same shape as the pre-call check.

    Providers call this after the SDK's ``with``-block exits to catch
    a signal that fired between ``__exit__`` and the function return.
    """
    controller = AbortController()
    guard = StreamAbortGuard(controller.signal)
    guard.raise_if_post_aborted()  # clean — no exception

    controller.abort("user_interrupt")
    with pytest.raises(AbortError):
        guard.raise_if_post_aborted()


# ---------------------------------------------------------------------------
# Aborted property — used by the in-loop check inside OpenAI-compat


def test_aborted_property_reflects_signal_state() -> None:
    controller = AbortController()
    guard = StreamAbortGuard(controller.signal)
    assert guard.aborted is False
    controller.abort("user_interrupt")
    assert guard.aborted is True


def test_aborted_property_is_false_when_no_signal() -> None:
    """``aborted`` is False when ``abort_signal=None`` — never tripped."""
    assert StreamAbortGuard(None).aborted is False


# ---------------------------------------------------------------------------
# attach() — the listener-lifecycle context manager


def test_attach_no_signal_is_noop_context() -> None:
    """With no signal, ``attach`` returns a context that does nothing.

    Lets providers wrap the iteration unconditionally without branching
    on whether the caller passed an abort_signal.
    """
    stream = _make_stream()
    guard = StreamAbortGuard(None)
    with guard.attach(stream):
        pass  # no listener registered, nothing to clean up
    stream.response.close.assert_not_called()


def test_attach_registers_listener_and_detaches_on_exit() -> None:
    """The listener exists while attached and is gone after exit.

    Pins the long-running-controller invariant: a single
    AbortController reused across many turns must not accumulate
    listeners pointing at gone streams.
    """
    controller = AbortController()
    stream = _make_stream()
    guard = StreamAbortGuard(controller.signal)

    assert controller.signal._listeners == []
    with guard.attach(stream):
        assert len(controller.signal._listeners) == 1
    assert controller.signal._listeners == []


def test_attach_fires_close_when_signal_trips_after_enter() -> None:
    """A signal that fires mid-attach calls ``stream.response.close()``."""
    controller = AbortController()
    stream = _make_stream()
    guard = StreamAbortGuard(controller.signal)

    with guard.attach(stream):
        stream.response.close.assert_not_called()
        controller.abort("user_interrupt")
        stream.response.close.assert_called_once()


def test_attach_fires_close_when_signal_already_tripped_at_enter() -> None:
    """Race-recovery: signal fired before ``attach`` calls ``__enter__``.

    The naive "check then register" sequence has a sub-microsecond
    race where ``_fire`` can snapshot the listener list before our
    ``add_listener`` append; the listener would be silently dropped.
    The helper's ``register-then-recheck`` ordering closes the gap:
    after ``add_listener`` we re-check ``aborted`` and call the close
    callback directly if the signal is already tripped.
    """
    controller = AbortController()
    controller.abort("user_interrupt")
    stream = _make_stream()
    guard = StreamAbortGuard(controller.signal)

    with guard.attach(stream):
        # The recheck after add_listener fired close() directly.
        stream.response.close.assert_called()


def test_attach_close_failures_do_not_propagate() -> None:
    """A raising ``stream.response.close()`` is swallowed.

    The listener fires from whichever thread tripped the abort (UI
    thread, SIGINT handler) — letting close() raise there would crash
    that thread without delivering the cancel.
    """
    controller = AbortController()
    stream = _make_stream()
    stream.response.close.side_effect = RuntimeError("simulated close failure")
    guard = StreamAbortGuard(controller.signal)

    with guard.attach(stream):
        # Should not raise even though close() throws.
        controller.abort("user_interrupt")

    # And the listener detach in __exit__ also tolerates the
    # already-fired state (the once=True wrapper has already
    # self-detached).


def test_attach_no_response_attribute_is_safe() -> None:
    """A stream without a ``response`` attribute is silently skipped.

    Future SDKs may name the response differently; the helper should
    degrade to "no close happens" rather than raising AttributeError
    inside the listener thread.
    """
    controller = AbortController()
    stream = MagicMock(spec=[])  # no response attribute
    guard = StreamAbortGuard(controller.signal)

    with guard.attach(stream):
        controller.abort("user_interrupt")  # no AttributeError raised

    # Stream lacks a response attribute; helper just no-ops.


def test_exit_closes_stream_when_signal_aborted_no_listener_fire() -> None:
    """``__exit__`` closes the response if the signal aborted but the listener never fired.

    Pins the race-recovery guarantee. ``AbortSignal._fire`` snapshots
    the listener list before iterating, so a narrow window exists
    where the consumer thread can:
    1. Observe ``aborted == True`` (set by ``_fire`` BEFORE the
       listener iteration starts),
    2. Break out of the iteration,
    3. Exit the ``with`` block — ``__exit__`` runs, detaches the
       listener,
    4. The original ``_fire`` thread resumes, ``list(self._listeners)``
       is now empty, the close never fires.

    Without the ``__exit__`` close fallback, the underlying httpx
    response would leak open. We simulate the race by setting
    ``_aborted = True`` directly (bypasses ``_fire``'s listener
    iteration) so the listener is guaranteed to have NOT fired.
    """
    controller = AbortController()
    stream = _make_stream()
    guard = StreamAbortGuard(controller.signal)

    with guard.attach(stream):
        # Trip the signal WITHOUT going through _fire, so no listener
        # is invoked — mimics the race window above where the abort
        # thread set ``_aborted=True`` but the listener iteration
        # races with our ``__exit__``.
        controller.signal._aborted = True
        controller.signal._reason = "user_interrupt"
        stream.response.close.assert_not_called()
    # ``__exit__`` must close even when the listener never fired.
    stream.response.close.assert_called()


def test_exit_does_not_close_when_signal_not_aborted() -> None:
    """The fallback close only fires on abort — clean exits don't trigger it.

    Regression guard: a stream that exits the attach context after
    natural iterator exhaustion (no abort) must not get a redundant
    ``close()`` call. The SDK's own ``__exit__`` is responsible for
    cleanup on the happy path.
    """
    controller = AbortController()
    stream = _make_stream()
    guard = StreamAbortGuard(controller.signal)

    with guard.attach(stream):
        pass  # no abort, no break — clean exit

    stream.response.close.assert_not_called()


# ---------------------------------------------------------------------------
# reraise_if_aborted — exception translation


def test_reraise_if_aborted_no_abort_is_noop() -> None:
    """If the signal didn't fire, leave the original exception alone."""
    controller = AbortController()
    guard = StreamAbortGuard(controller.signal)
    orig = RuntimeError("genuine network error")
    # No raise — the caller's subsequent ``raise`` re-raises ``orig``.
    guard.reraise_if_aborted(orig)


def test_reraise_if_aborted_translates_to_abort_error_with_cause() -> None:
    """When the signal fired, translate to AbortError preserving the cause.

    The SDK / httpx layer can raise several different exception
    classes when the underlying response is closed mid-read; the
    guard uses the signal state (not the exception class) as the
    authoritative abort indicator. The original exception is
    chained via ``raise ... from`` so observers can still see what
    the SDK reported.
    """
    controller = AbortController()
    controller.abort("user_interrupt")
    guard = StreamAbortGuard(controller.signal)
    orig = ConnectionError("socket closed mid-read")

    with pytest.raises(AbortError) as exc_info:
        guard.reraise_if_aborted(orig)
    assert exc_info.value.reason == "user_interrupt"
    assert exc_info.value.__cause__ is orig


def test_reraise_if_aborted_no_signal_is_noop() -> None:
    """``abort_signal=None`` guards always treat the exception as non-abort."""
    StreamAbortGuard(None).reraise_if_aborted(RuntimeError("anything"))


# ---------------------------------------------------------------------------
# F-99 方案2: transport.close() — interrupt the underlying socket read
#
# ``response.close()`` is advisory on some platforms and does not
# interrupt the in-flight blocking read on the worker thread. Closing
# the httpx transport closes the TCP socket, which makes the next
# read return immediately with an exception. The provider then
# translates that exception to ``AbortError`` via
# ``reraise_if_aborted``. These tests pin the close behaviour.


def test_close_response_safely_closes_transport_on_non_windows(monkeypatch) -> None:
    """F-99: ``_close_response_safely`` calls ``response._transport.close()``.

    Mock ``sys.platform`` so the Windows skip branch doesn't fire and
    assert both ``response.close`` AND ``response._transport.close``
    are called. The transport close is the actual interrupt that
    bounds cancel latency on platforms where ``response.close()`` is
    advisory.
    """
    monkeypatch.setattr("src.providers._stream_abort.sys.platform", "linux")
    stream = _make_stream()
    # The MagicMock auto-creates _transport with a callable close;
    # explicitly verify it gets called.
    _close_response_safely(stream)
    stream.response.close.assert_called_once()
    stream.response._transport.close.assert_called_once()


def test_close_response_safely_skips_transport_on_windows(monkeypatch) -> None:
    """F-99: Windows skips transport.close to avoid kernel-level deadlock risk.

    On Windows, forcibly closing a socket mid-read can deadlock some
    Winsock code paths. The F-99 plan degrades to ``response.close()``
    only on Windows, relying on ``read_timeout=5s`` (方案1) for the
    cancel latency bound instead.
    """
    monkeypatch.setattr("src.providers._stream_abort.sys.platform", "win32")
    stream = _make_stream()
    _close_response_safely(stream)
    stream.response.close.assert_called_once()
    # ``_transport`` should NOT have been touched on Windows.
    stream.response._transport.close.assert_not_called()


def test_close_response_safely_handles_missing_transport(monkeypatch) -> None:
    """F-99: missing ``_transport`` attribute is a silent no-op, not an error.

    httpx is allowed to rename the attribute in a future major
    version; the close must degrade to ``response.close()``-only
    semantics rather than raising AttributeError from the listener
    thread (which would mask the abort).
    """
    monkeypatch.setattr("src.providers._stream_abort.sys.platform", "linux")

    class _NoTransport:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    stream = MagicMock()
    stream.response = _NoTransport()
    # Should not raise even though _transport does not exist.
    _close_response_safely(stream)
    assert stream.response.closed is True


def test_close_response_safely_handles_transport_close_failure(monkeypatch) -> None:
    """F-99: ``_transport.close()`` failures are swallowed, not propagated.

    The listener fires from the keypress / SIGINT thread — letting
    the close raise there would crash that thread without delivering
    the cancel. The helper's contract is ``never raises``; both
    application-level and transport-level close paths share it.
    """
    monkeypatch.setattr("src.providers._stream_abort.sys.platform", "linux")
    stream = _make_stream()
    stream.response._transport.close.side_effect = RuntimeError("simulated transport close failure")
    # Should not raise.
    _close_response_safely(stream)
    stream.response.close.assert_called_once()


def test_attach_listener_closes_transport_on_abort(monkeypatch) -> None:
    """F-99 end-to-end: abort fires → both response.close AND _transport.close run.

    Pins the full listener path: the ``attach`` context's listener
    closes ``stream.response`` AND the underlying transport when the
    signal fires, so the SDK's blocking read returns immediately.
    """
    monkeypatch.setattr("src.providers._stream_abort.sys.platform", "linux")
    controller = AbortController()
    stream = _make_stream()
    guard = StreamAbortGuard(controller.signal)

    with guard.attach(stream):
        controller.abort("user_interrupt")
        stream.response.close.assert_called_once()
        stream.response._transport.close.assert_called_once()


def test_attach_transport_close_failure_does_not_propagate(monkeypatch) -> None:
    """F-99: transport.close() failures during abort must not raise.

    The listener thread cannot be allowed to die mid-abort; if it did,
    the underlying socket would leak and the agent loop would hang
    waiting for the read timeout. The defensive ``try/except`` in
    ``_close_transport_safely`` is the last line of defence.
    """
    monkeypatch.setattr("src.providers._stream_abort.sys.platform", "linux")
    controller = AbortController()
    stream = _make_stream()
    stream.response._transport.close.side_effect = OSError("simulated socket error")
    guard = StreamAbortGuard(controller.signal)

    with guard.attach(stream):
        controller.abort("user_interrupt")  # must not raise
