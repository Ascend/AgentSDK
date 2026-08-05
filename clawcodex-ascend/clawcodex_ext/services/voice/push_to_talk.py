#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Push-to-Talk controller — F-64 P64-B.

Mirrors TS ``src/hooks/useVoice.ts``: orchestrates the recording lifecycle
on a push-to-talk key event. The user holds a key (default: spacebar,
configured by the REPL/TUI); on press we start recording + open the STT
stream, on release we stop recording + await the final transcript.

Lifecycle
---------
1. :meth:`start` — arm the controller. Resolves the active provider from
   settings, instantiates it via the registry, opens a
   ``VoiceStreamConnection`` / ``DoubaoStreamConnection``, and starts the
   :class:`AudioRecorder`. The recorder thread pushes PCM frames into the
   connection's :class:`AudioChunkQueue`.
2. Interim transcripts arrive on the ``on_transcript`` callback during
   recording — the REPL/TUI renders them as live ghost text.
3. :meth:`stop` — user released the key. Stop the recorder, call
   ``finalize()`` on the connection (Anthropic blocks until the server
   emits the final transcript; doubao returns immediately), and return
   the final text for the REPL to insert + auto-submit.

Surface-agnostic
----------------
The controller is deliberately decoupled from the UI layer: it exposes
``on_transcript`` / ``on_error`` callbacks and a ``state`` property. The
REPL/TUI wires the push-to-talk hotkey → :meth:`start` / :meth:`stop` and
renders ``state`` + interim transcripts however it likes. This keeps the
voice stack testable without a real terminal.

Concurrency
-----------
``start`` / ``stop`` are *not* async — they're called from the UI thread
on a key event. The async connection machinery runs on the event loop
the controller binds at ``start`` time (the REPL's running loop). The
recorder runs on its own daemon thread and pushes into the thread-safe
:class:`AudioChunkQueue`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .audio_recorder import AudioRecorder, make_recorder
from .provider_registry import get_stt_provider
from .stt import STTConfig
from .voice_mode_enabled import (
    get_voice_provider,
    is_voice_available,
    is_voice_enabled,
)

logger = logging.getLogger(__name__)

__all__ = [
    "VoiceSessionState",
    "VoiceSessionResult",
    "PushToTalkController",
]


class VoiceSessionState(str, Enum):
    """Lifecycle states the controller cycles through per recording."""

    IDLE = "idle"  # not armed
    ARMED = "armed"  # armed, waiting for key press
    CONNECTING = "connecting"  # opening STT stream
    RECORDING = "recording"  # streaming audio, awaiting transcripts
    FINALIZING = "finalizing"  # key released, awaiting final transcript
    DONE = "done"  # final transcript ready
    ERROR = "error"


@dataclass
class VoiceSessionResult:
    """Outcome of one push-to-talk recording session."""

    text: str = ""
    error: Optional[str] = None
    provider: str = ""
    duration_ms: float = 0.0


class PushToTalkController:
    """Orchestrates one push-to-talk recording session.

    Constructed once per REPL/TUI session; :meth:`start` / :meth:`stop`
    are called on each key press/release cycle. The controller is not
    thread-safe across concurrent sessions — the REPL guarantees only one
    recording is active at a time (the hotkey is ignored while
    ``state != IDLE and state != ARMED``).
    """

    def __init__(
        self,
        *,
        recorder: Optional[AudioRecorder] = None,
        on_transcript: Optional[Callable[[str, bool], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[VoiceSessionState], None]] = None,
        config: Optional[STTConfig] = None,
    ) -> None:
        # Recorder is injected so tests can stub it; production uses
        # ``make_recorder()`` lazily at first ``start`` so we don't open
        # the mic until voice is actually armed.
        self._recorder_factory: Optional[Callable[[], AudioRecorder]]
        if recorder is not None:
            self._recorder_factory = lambda: recorder
        else:
            self._recorder_factory = None
        self._recorder: Optional[AudioRecorder] = None
        self._on_transcript = on_transcript
        self._on_error = on_error
        self._on_state_change = on_state_change
        self._config = config or STTConfig()
        self._state = VoiceSessionState.IDLE
        self._connection: Optional[object] = None  # VoiceStreamConnection | DoubaoStreamConnection
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._final_text = ""
        self._last_error: Optional[str] = None
        self._provider_name: str = ""

    @property
    def state(self) -> VoiceSessionState:
        return self._state

    @property
    def is_armed(self) -> bool:
        """True if a key press should begin recording."""
        return self._state in (VoiceSessionState.IDLE, VoiceSessionState.ARMED)

    def can_start(self) -> bool:
        """Gate check — should the hotkey arm at all right now?

        Combines the three F-64 layers: feature flag + kill-switch (via
        :func:`is_voice_available`) and the master on/off switch (via
        :func:`is_voice_enabled`). The per-provider OAuth check is
        deferred to :meth:`start` so the user gets a specific "run
        /login" message on the Anthropic path rather than the hotkey
        silently refusing.
        """
        return is_voice_available() and is_voice_enabled()

    def _set_state(self, new_state: VoiceSessionState) -> None:
        if new_state != self._state:
            self._state = new_state
            if self._on_state_change is not None:
                try:
                    self._on_state_change(new_state)
                except Exception:
                    logger.exception("on_state_change callback raised")

    def _emit_error(self, msg: str) -> None:
        self._last_error = msg
        self._set_state(VoiceSessionState.ERROR)
        if self._on_error is not None:
            try:
                self._on_error(msg)
            except Exception:
                logger.exception("on_error callback raised")

    def _emit_transcript(self, text: str, is_final: bool) -> None:
        if is_final:
            self._final_text = (self._final_text + " " + text).strip() if self._final_text else text
        if self._on_transcript is not None:
            try:
                self._on_transcript(text, is_final)
            except Exception:
                logger.exception("on_transcript callback raised")

    def start(self) -> bool:
        """Begin a recording session. Returns False if voice is unavailable.

        Called synchronously from the UI thread on key-down. Opens the
        STT connection (async, scheduled on the running loop) and starts
        the audio recorder. The recorder thread pushes PCM into the
        connection's audio queue; interim transcripts fire on
        ``on_transcript`` from the connection's pump tasks.
        """
        if not self.can_start():
            self._emit_error("Voice mode is disabled (feature flag / kill-switch / off)")
            return False
        if not self.is_armed:
            # Already recording — ignore the re-press (hotkey repeat).
            return False

        # Resolve the event loop now so the connection's background tasks
        # land on the REPL's loop. ``get_running_loop`` raises if called
        # from a non-async context (the UI thread) — fall back to
        # ``get_event_loop`` which returns the bound loop even from a
        # sync thread. On 3.12+ ``get_event_loop`` may also raise when
        # no loop is bound (e.g. in test runners between tests), so the
        # final fallback creates a fresh loop and binds it as the
        # thread's default. This loop is only used to schedule the
        # connection's background coroutines; the controller's own
        # ``stop()`` is awaited by the caller on whatever loop they
        # use (typically the REPL's).
        self._loop = None
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                try:
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
                except RuntimeError as exc:
                    self._emit_error(f"No event loop available for voice streaming: {exc}")
                    return False
        if self._loop is None:
            self._emit_error("No event loop available for voice streaming")
            return False

        self._provider_name = get_voice_provider()
        try:
            provider = get_stt_provider(self._provider_name)
        except KeyError:
            self._emit_error(f"Unknown voice provider {self._provider_name!r}. Run /voice anthropic or /voice doubao.")
            return False
        except ImportError as exc:
            self._emit_error(f"Voice backend {self._provider_name!r} unavailable: {exc}")
            return False

        self._set_state(VoiceSessionState.CONNECTING)
        self._final_text = ""
        self._last_error = None

        # Open the streaming connection on the event loop. The connection
        # object is returned synchronously by connect_stream (it schedules
        # the async run internally); we can start pushing audio as soon as
        # the recorder is up.
        try:
            connect_fn = getattr(provider, "connect_stream", None)
            if connect_fn is None:
                self._emit_error(f"Provider {self._provider_name!r} does not support streaming")
                return False
            self._connection = connect_fn(
                on_transcript=self._emit_transcript,
                on_error=self._emit_error,
                on_ready=lambda: self._set_state(VoiceSessionState.RECORDING),
                config=self._config,
            )
        except Exception as exc:
            self._emit_error(f"Failed to open voice stream: {exc}")
            return False

        # Start the recorder — push PCM into the connection's audio queue.
        recorder = self._recorder_factory() if self._recorder_factory else make_recorder()
        self._recorder = recorder
        try:
            recorder.start(self._on_audio_chunk)
        except RuntimeError as exc:
            # PyAudio open() failed — try SoX once before giving up.
            logger.warning("Primary recorder failed (%s); retrying with SoX", exc)
            from .audio_recorder import SoXRecorder

            recorder = SoXRecorder()
            self._recorder = recorder
            try:
                recorder.start(self._on_audio_chunk)
            except RuntimeError as exc2:
                self._emit_error(f"No audio recorder available: {exc2}")
                self._cleanup_connection()
                return False
        # Recorder is up — transition to RECORDING. (The connection's
        # on_ready callback may fire slightly later for the Anthropic
        # backend once the WebSocket handshake completes; that's fine,
        # the state will simply re-set to RECORDING.)
        self._set_state(VoiceSessionState.RECORDING)
        return True

    def _on_audio_chunk(self, chunk: bytes) -> None:
        """Recorder-thread callback: push PCM into the active connection."""
        if self._connection is not None:
            try:
                self._connection.feed_audio(chunk)
            except Exception:
                logger.exception("feed_audio failed on voice connection")

    async def stop(self) -> VoiceSessionResult:
        """End the recording session and await the final transcript.

        Called from the UI thread on key-up. Must be awaited (the REPL
        wraps it in ``asyncio.run_coroutine_threadsafe`` if the hotkey
        handler is sync). Returns the concatenated final text; on error
        ``result.error`` is set and ``result.text`` is empty.
        """
        if self._state in (VoiceSessionState.IDLE, VoiceSessionState.ARMED):
            return VoiceSessionResult(provider=self._provider_name)

        self._set_state(VoiceSessionState.FINALIZING)

        # Stop the recorder first so no more PCM is queued.
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:
                logger.exception("recorder.stop raised")
            self._recorder = None

        # Finalize the connection — Anthropic blocks until the server
        # emits the final transcript; doubao returns immediately.
        final_text = ""
        if self._connection is not None:
            try:
                finalize_fn = getattr(self._connection, "finalize", None)
                if finalize_fn is not None:
                    final_text = await finalize_fn()
                else:
                    final_text = self._final_text
            except Exception as exc:
                self._emit_error(f"Voice finalize failed: {exc}")
                final_text = ""
            self._cleanup_connection()

        if self._last_error is not None:
            self._set_state(VoiceSessionState.ERROR)
            return VoiceSessionResult(
                text="",
                error=self._last_error,
                provider=self._provider_name,
            )

        self._final_text = final_text or self._final_text
        self._set_state(VoiceSessionState.DONE)
        result = VoiceSessionResult(text=self._final_text, provider=self._provider_name)
        # Reset for the next session.
        self._final_text = ""
        self._set_state(VoiceSessionState.IDLE)
        return result

    def _cleanup_connection(self) -> None:
        if self._connection is None:
            return
        try:
            close_fn = getattr(self._connection, "close", None)
            if close_fn is not None and asyncio.iscoroutinefunction(close_fn):
                # close() is async — schedule it on the bound loop. We
                # don't await here because stop() may already be running
                # on that loop and we don't want a nested await; the
                # socket close is fire-and-forget.
                if self._loop is not None and self._loop.is_running():
                    self._loop.create_task(close_fn())  # pylint: disable=not-callable
                else:
                    asyncio.ensure_future(close_fn())  # pylint: disable=not-callable
            elif close_fn is not None:
                close_fn()
        except Exception:
            logger.exception("voice connection close raised")
        self._connection = None

    def disarm(self) -> None:
        """Force-stop any active session without awaiting transcripts.

        Used when the REPL is shutting down or the user disables voice
        mid-recording. Synchronous: stops the recorder and schedules a
        connection close on the event loop.
        """
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:  # nosec B110 - best-effort recorder stop during disarm
                pass
            self._recorder = None
        self._cleanup_connection()
        self._set_state(VoiceSessionState.IDLE)
