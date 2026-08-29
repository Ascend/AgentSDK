#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Full-duplex voice dialogue session management.

Owns the lifecycle of one full-duplex voice dialogue session:

```
IDLE ──> LISTENING ──> SPEAKING ──> INTERRUPTED ──> LISTENING
 │ │ │ │
 │ │ │ └──► reset to LISTENING once
 │ │ │ barge-in clears
 │ │ └──► back to LISTENING when audio stream ends
 │ └──► recording microphone frames, agent idle
 └──► start begins here; stop returns here on shutdown
```

Responsibilities
----------------
* Construct and tear down a :class:`FullDuplexDialogueProvider` (the
  provider interface). One manager is used per session.
* Own the audio recorder and player through :class:`AudioChunkQueue`,
  :class:`AudioOutQueue`, and :class:`AudioPlayer`.
* Run an :class:`InterruptDetector` over the recorded PCM. When the user
  barges in, call :meth:`AudioPlayer.stop_nowait`,
  :meth:`AudioOutQueue.clear`, and
  :meth:`FullDuplexDialogueProvider.interrupt` in that order so the device
  falls silent before the server cancels the in-flight response.
* Forward :class:`DialogueEvent` instances to user-supplied listeners (the
  agent pipeline): text to the transcript-event handler and audio to
  :class:`AudioOutQueue`.

Design notes
------------
* The manager is **session-scoped**, not process-scoped. A new instance
 per ``/dialogue start`` keeps each conversation isolated. There's no
 global session registry; the caller (``/dialogue`` command) stores the
 handle.
* Audio *playback* and *recording* run on the same event loop the
 provider's pump tasks use. PyAudio's blocking ``stream.write`` is
 offloaded to the default executor inside :class:`AudioPlayer`
 (``run_in_executor``) so the loop stays responsive to provider
 events.
* The interrupt policy uses VAD speech-start, a short cooldown, immediate
  stop and clear, and server ``response.cancel``. It does not perform
  semantic stop-word detection.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .audio_out_queue import AudioOutQueue
from .audio_player import AudioPlayer
from .audio_recorder import AudioRecorder, make_recorder
from .dialogue import (
    DialogueConfig,
    DialogueEvent,
    DialogueState,
    FullDuplexDialogueProvider,
)
from .interrupt import InterruptConfig, InterruptDecision, InterruptDetector
from .tts import TTSChunk

logger = logging.getLogger(__name__)

__all__ = [
    "DialogueSessionManager",
    "DialogueSessionState",
    "DialogueSessionCallbacks",
]


# ── public types ──────────────────────────────────────────────────────────


class DialogueSessionState:
    """Externalised state machine for the session.

    Strings rather than an enum so ``/dialogue status`` can ``str()``
    the field without ceremony. Mirrors :class:`DialogueState` so a
    caller-level mapping is unnecessary; the two surface the same names.
    """

    IDLE = DialogueState.IDLE
    LISTENING = DialogueState.LISTENING
    SPEAKING = DialogueState.SPEAKING
    INTERRUPTED = DialogueState.INTERRUPTED
    DONE = DialogueState.DONE
    ERROR = DialogueState.ERROR


@dataclass
class DialogueSessionCallbacks:
    """Callbacks the session invokes into the agent pipeline.

    * :attr:`on_user_transcript` — text the user just said (final).
    * :attr:`on_user_transcript_partial` — interim transcript while the
      user is still speaking (for UI echo / low-latency UIs).
    * :attr:`on_agent_text` — final agent text reply (modality=text and
      the server signals reply complete). Optional; the typical dialogue
      session routes audio straight to the speaker and only text when
      ``modality="text"``.
    * :attr:`on_error` — unhandled provider error.
    * :attr:`on_state_change` — state transition (idle/listening/...).
    * :attr:`on_interrupt` — barge-in detected by the VAD; the agent
      loop can use this to flush any pending tool calls.
    """

    on_user_transcript: Optional[Callable[[str], Awaitable[None] | None]] = None
    on_user_transcript_partial: Optional[Callable[[str], None]] = None
    on_agent_text: Optional[Callable[[str], Awaitable[None] | None]] = None
    on_error: Optional[Callable[[str], None]] = None
    on_state_change: Optional[Callable[[str], None]] = None
    on_interrupt: Optional[Callable[[], None]] = None


@dataclass
class DialogueSessionOptions:
    """Optional knobs for one session.

    Keeps :class:`DialogueSessionManager`'s signature flat — most users
    only need the provider + callbacks; everything else has a
    reasonable default.
    """

    config: Optional[DialogueConfig] = None
    interrupt_config: Optional[InterruptConfig] = None
    recorder: Optional[AudioRecorder] = None
    # When ``True`` (default) the manager owns and starts an AudioRecorder.
    # Pass ``False`` from tests that inject PCM frames directly via
    # :meth:`feed_audio`.
    use_recorder: bool = True


# ── the manager ───────────────────────────────────────────────────────────


class DialogueSessionManager:
    """Coordinates one full-duplex voice dialogue session.

    Construct → :meth:`start` → ... → :meth:`stop` (or :meth:`close`).
    A manager is single-use: destroying it tears down every resource it
    owns. Construct a fresh one for each ``/dialogue start`` invocation.
    """

    def __init__(
        self,
        provider: FullDuplexDialogueProvider,
        callbacks: Optional[DialogueSessionCallbacks] = None,
        options: Optional[DialogueSessionOptions] = None,
    ) -> None:
        self._provider = provider
        self._callbacks = callbacks or DialogueSessionCallbacks()
        self._options = options or DialogueSessionOptions()
        self._state: str = DialogueSessionState.IDLE

        # Audio pipeline. The player owns its own AudioOutQueue.
        self._out_queue = AudioOutQueue()
        self._player = AudioPlayer(queue=self._out_queue, sample_rate=24000)
        self._interrupt = InterruptDetector(self._options.interrupt_config)
        self._final_transcript_chunks: list[str] = []
        self._has_agent_audio = False  # any audio frame emitted this turn?
        self._recorder: Optional[AudioRecorder] = self._options.recorder
        self._recorder_task: Optional[asyncio.Task] = None

    # ── public surface ────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def out_queue(self) -> AudioOutQueue:
        """The playback queue — exposed for tests that drive audio frames in manually."""
        return self._out_queue

    @property
    def player(self) -> AudioPlayer:
        return self._player

    @property
    def interrupt_detector(self) -> InterruptDetector:
        return self._interrupt

    async def start(self) -> None:
        """Open the provider, start the player, begin capturing audio."""
        if self._state != DialogueSessionState.IDLE:
            return
        config = self._options.config
        # Provider handshake — fires events into our dispatcher.
        await self._provider.start(
            on_event=self._on_provider_event,
            config=config,
        )
        # Start the audio player task. Idempotent.
        self._player.start()
        # Begin recording (if a recorder is configured and the user opted in).
        if self._options.use_recorder and self._recorder is None:
            try:
                self._recorder = make_recorder(sample_rate=(config.sample_rate if config else 16000))
            except Exception as exc:
                # Recorder init failure is non-fatal: the session can still
                # run in text/scripted mode. Surface the error but stay up.
                logger.warning("Dialogue recorder init failed: %s", exc)
                self._recorder = None
        if self._recorder is not None:
            self._recorder_task = asyncio.create_task(self._recorder_loop(), name="dialogue-recorder-loop")
        self._set_state(DialogueSessionState.LISTENING)

    async def stop(self) -> str:
        """End the dialogue, return the concatenated final user transcript."""
        if self._state == DialogueSessionState.DONE:
            return " ".join(p for p in self._final_transcript_chunks if p).strip()
        final = await self._provider.stop()
        if final:
            self._final_transcript_chunks.append(final)
        await self._player.stop_nowait()
        # ``AudioOutQueue.clear`` is synchronous (returns a count) — it
        # can't be awaited. Calling it without ``await`` matches the
        # method's surface in :mod:`audio_out_queue`.
        self._out_queue.clear()
        if self._recorder_task is not None:
            self._recorder_task.cancel()
            try:
                await self._recorder_task
            except (asyncio.CancelledError, Exception):
                pass  # The task was explicitly cancelled; awaiting it only drains shutdown cleanup.
            self._recorder_task = None
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:  # nosec B110 - best-effort recorder stop during teardown
                pass  # Cleanup is best-effort and must not replace the primary operation result.
        await self._provider.close()
        self._set_state(DialogueSessionState.DONE)
        return " ".join(p for p in self._final_transcript_chunks if p).strip()

    async def feed_audio(self, pcm: bytes) -> None:
        """External audio injection — test/CLI override.

        Forwards the chunk to both the provider (real backend) and the
        local interrupt VAD. Use this from tests; the real path runs
        the recorder loop on its own.
        """
        if not pcm or self._state == DialogueSessionState.DONE:
            return
        await self._provider.feed_audio(pcm)
        decision = self._interrupt.feed_pcm(pcm)
        if decision == InterruptDecision.SPEECH_START and self._has_agent_audio:
            await self._handle_interrupt()

    async def close(self) -> None:
        """Best-effort cleanup. Idempotent."""
        try:
            await self._provider.close()
        except Exception:  # nosec B110 - best-effort cleanup; close() is idempotent
            pass  # Cleanup is best-effort and must not replace the primary operation result.
        try:
            await self._player.stop_and_close()
        except Exception:  # nosec B110 - best-effort cleanup; close() is idempotent
            pass  # Voice resource cleanup is best-effort during session shutdown.
        try:
            self._out_queue.clear()
        except Exception:  # nosec B110 - best-effort cleanup; close() is idempotent
            pass  # Cleanup is best-effort and must not replace the primary operation result.
        if self._recorder_task is not None and not self._recorder_task.done():
            self._recorder_task.cancel()
            try:
                await self._recorder_task
            except (asyncio.CancelledError, Exception):
                pass  # The task was explicitly cancelled; awaiting it only drains shutdown cleanup.
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:  # nosec B110 - best-effort recorder stop during teardown
                pass  # Cleanup is best-effort and must not replace the primary operation result.
        if self._state != DialogueSessionState.DONE:
            self._set_state(DialogueSessionState.DONE)

    # ── provider-event dispatch ───────────────────────────────────────────

    def _on_provider_event(self, event: DialogueEvent) -> None:
        """Router for :class:`DialogueEvent` from the backend.

        Synchronous dispatcher — async fanout (e.g. awaiting an LLM call)
        is the responsibility of the callback supplied in
        :class:`DialogueSessionCallbacks`. We schedule those callbacks
        on the loop but don't ``await`` them here because providers tend
        to fire events from a single recv task; awaiting inside it would
        stall the wire.
        """
        try:
            if event.type == "ready":
                return
            if event.type == "transcript":
                if event.is_final:
                    self._final_transcript_chunks.append(event.text)
                    cb = self._callbacks.on_user_transcript
                    if cb is not None:
                        self._schedule(cb, event.text)
                else:
                    cb = self._callbacks.on_user_transcript_partial
                    if cb is not None:
                        try:
                            cb(event.text)
                        except Exception:
                            logger.exception("Partial-transcript callback raised")
                return
            if event.type == "audio":
                # Forward PCM into the player's queue. The queue drops the
                # oldest frame on overflow, by design, so a slow
                # device never blocks the provider's recv task.
                if event.pcm:
                    self._has_agent_audio = True
                    self._set_state(DialogueSessionState.SPEAKING)
                    try:
                        self._out_queue.push_nowait(
                            TTSChunk(
                                pcm=event.pcm,
                                sample_rate=event.sample_rate or 24000,
                            )
                        )
                    except Exception:
                        logger.exception("Failed to enqueue TTS chunk")
                return
            if event.type == "done":
                # End of a response turn — back to listening.
                self._has_agent_audio = False
                text = event.text or ""
                if text:
                    cb = self._callbacks.on_agent_text
                    if cb is not None:
                        self._schedule(cb, text)
                if self._state != DialogueSessionState.INTERRUPTED:
                    self._set_state(DialogueSessionState.LISTENING)
                return
            if event.type == "interrupt":
                # The provider itself signalled an interrupt (e.g.
                # ``response.cancel`` ack from the server). No-op: the
                # barge-in already cleared local playback via
                # :meth:`_handle_interrupt`.
                return
            if event.type == "error":
                cb = self._callbacks.on_error
                if cb is not None:
                    try:
                        cb(event.message)
                    except Exception:
                        logger.exception("Error callback raised")
                self._set_state(DialogueSessionState.ERROR)
                return
            if event.type in ("speech_started", "speech_stopped"):
                # Optional server-side VAD hints — informational only; our
                # local :class:`InterruptDetector` is the source of truth
                # for barge-in timing.
                return
        except Exception:
            # Never let a callback or queue failure kill the recv loop.
            logger.exception("Dialogue event dispatch error")

    # ── helpers ──────────────────────────────────────────────────────────

    def _set_state(self, new_state: str) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        cb = self._callbacks.on_state_change
        if cb is not None:
            try:
                cb(new_state)
            except Exception:
                logger.exception("State-change callback raised")

    def _schedule(self, cb: Callable[..., Any], *args: Any) -> None:
        """Invoke an (optionally-async) callback from the event loop."""
        try:
            result = cb(*args)
        except Exception:
            logger.exception("Dialogue callback raised")
            return
        if asyncio.iscoroutine(result):
            # Fire-and-forget on the running loop.
            asyncio.create_task(self._await_callback(result))

    async def _await_callback(self, coro: Awaitable[Any]) -> None:
        try:
            await coro
        except Exception:
            logger.exception("Dialogue async callback raised")

    async def _handle_interrupt(self) -> None:
        """Barge-in: stop local playback, clear queue, notify provider."""
        self._set_state(DialogueSessionState.INTERRUPTED)
        try:
            await self._player.stop_nowait()
        except Exception:
            logger.exception("Player stop_nowait failed")
        try:
            self._out_queue.clear()
        except Exception:
            logger.exception("Out queue clear failed")
        try:
            await self._provider.interrupt()
        except Exception:
            logger.exception("Provider interrupt failed")
        cb = self._callbacks.on_interrupt
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.exception("Interrupt callback raised")
        # Return to listening; next provider event will flip state again.
        self._set_state(DialogueSessionState.LISTENING)

    # ── recorder pump ────────────────────────────────────────────────────

    async def _recorder_loop(self) -> None:
        """Read frames from the recorder, send them to provider + VAD.

        Runs until cancelled (on :meth:`stop` / :meth:`close`).
        Recorder implementations differ: PyAudio / SoX each expose a
        blocking ``read()`` returning one frame; we offload to a worker
        thread to keep the loop responsive.
        """
        recorder = self._recorder
        if recorder is None:
            return
        loop = asyncio.get_event_loop()
        try:
            recorder.start()
        except Exception as exc:
            logger.warning("Dialogue recorder.start failed: %s", exc)
            return
        try:
            while True:
                try:
                    chunk = await loop.run_in_executor(None, recorder.read_chunk)
                except Exception as exc:
                    logger.debug("Dialogue recorder read error: %s", exc)
                    await asyncio.sleep(0.05)
                    continue
                if not chunk:
                    continue
                await self.feed_audio(chunk)
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception:
            logger.exception("Dialogue recorder loop crashed")
