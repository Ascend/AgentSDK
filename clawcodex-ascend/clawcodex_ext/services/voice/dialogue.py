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

"""Full-duplex voice dialogue abstraction — F-65 P65-A.

Defines the interface every full-duplex voice backend (MiniMax Realtime,
OpenAI GPT-4o Voice, etc.) must implement. A full-duplex provider owns a
single bidirectional channel — usually a WebSocket — that simultaneously:

* receives PCM frames from the user microphone (``feed_audio``);
* emits incremental transcripts (``DialogueEvent(transcript, ...)``);
* emits TTS audio frames for the agent reply
  (``DialogueEvent(audio, pcm=...)``);
* accepts interrupt signals (``interrupt``) to cancel in-flight output
  without tearing down the session.

Why a separate ABC rather than reusing ``STTProvider`` / ``TTSProvider``
----------------------------------------------------------------------
MiniMax Realtime (and OpenAI Realtime) run ASR + LLM + TTS server-side
in one pipe. Pulling transcripts out of a transcript stream and pushing
text into a TTS stream would double the latency and lose the server's
VAD / turn-taking state. The single-pipe design is what makes
sub-second end-to-end response possible — so the abstraction mirrors it.
The standalone STT and TTS providers stay around as the F-64 half-duplex
/ 试听 fallback paths.

Lifecycle
---------
* :meth:`start` — open the underlying transport (WebSocket) and begin
  receiving the first ``DialogueEvent`` shortly after. Events stream
  through the ``on_event`` callback until ``stop`` is called.
* :meth:`feed_audio` — push PCM frames the user is currently saying.
  Non-blocking: returns after queueing the frame (the wire protocol
  task dispatches it on the event loop).
* :meth:`send_text` — inject a text message (e.g. an agent reply
  produced by an external LLM call) into the same conversation.
  Optional; some deployments only use server-side LLM.
* :meth:`interrupt` — user barges in. Cancels any in-flight TTS on the
  server side and stops local playback. The session stays open.
* :meth:`stop` — graceful shutdown: tell the server to flush its
  buffers, then disconnect. Returns the final transcript / summary.
* :meth:`close` — release resources (cancel tasks, close socket).
  Called automatically by ``stop``; not always needed by callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


__all__ = [
    "DialogueConfig",
    "DialogueEvent",
    "FullDuplexDialogueProvider",
    "DialogueModality",
    "DialogueState",
]


# ── config + event types ──────────────────────────────────────────────────

DialogueModality = str  # "text" | "audio" | "text_and_audio"


@dataclass
class DialogueConfig:
    """Full-duplex voice dialogue configuration.

    Defaults target the MiniMax Realtime API: ``speech-2.8-turbo`` model,  # gitleaks:allow -- not a secret, model name
    16 kHz mono input, 24 kHz mono output, and ``text`` modality so the
    caller receives incremental text and can route it through its own
    LLM / TTS pipeline. Set ``modality="audio"`` to get PCM TTS frames
    straight from the server (no second TTS call needed); text is still
    available via the transcript events in that mode.
    """

    model: str = "speech-2.8-turbo"
    sample_rate: int = 16000  # input sample rate (microphone)
    output_sample_rate: int = 24000  # output sample rate (speaker / TTS)
    voice: str = ""  # provider-specific TTS voice id
    modality: DialogueModality = "text"
    language: str = "zh"  # ASR language hint; provider-specific
    interim_results: bool = True  # stream partial transcripts
    # Provider-specific knobs (e.g. MiniMax voice_setting, OpenAI turn_detection).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueEvent:
    """A single event delivered by :class:`FullDuplexDialogueProvider`.

    The ``type`` field selects which other fields are populated; the
    others default to empty values so callers can branch on ``type``
    without checking each one. The set of event types mirrors the
    MiniMax / OpenAI Realtime event families trimmed down to the
    surface a full-duplex dialogue loop actually needs.
    """

    # ── discriminator ───────────────────────────────────────────────────
    type: str  # "transcript" | "audio" | "done" | "error" | "interrupt" |
    #           "ready" | "speech_started" | "speech_stopped"

    # ── payload (only some are set per type) ────────────────────────────
    text: str = ""  # transcript text (transcript/done)
    pcm: bytes = b""  # PCM16 mono frame (audio)
    sample_rate: int = 24000  # for audio frames
    is_final: bool = False  # transcript finalization flag (transcript)
    message: str = ""  # human-readable text (error)
    # Optional provider-specific passthrough (e.g. raw event id, item id).
    extra: dict[str, Any] = field(default_factory=dict)


# ── state enum (used by session manager, kept here for cohesion) ────────


class DialogueState(str):
    """Provider-agnostic state names. Strings, not enum, to keep the
    surface serializable for ``/dialogue status`` without extra plumbing.

    * ``"idle"`` — session not started yet.
    * ``"listening"`` — microphone open, no agent reply in flight.
    * ``"speaking"`` — agent audio currently streaming.
    * ``"interrupted"`` — barge-in detected, waiting for server ack.
    * ``"done"`` — ``stop()`` called; final transcript returned.
    * ``"error"`` — unrecoverable; caller should ``close()``.
    """

    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    DONE = "done"
    ERROR = "error"


# ── ABC ──────────────────────────────────────────────────────────────────


class FullDuplexDialogueProvider(ABC):
    """Abstract full-duplex voice dialogue backend.

    One instance per live session (construct → ``start`` → ``feed_audio`` /
    ``send_text`` in any order → ``stop`` → ``close``). The wire protocol
    itself (WebSocket, gRPC stream, etc.) is encapsulated by the concrete
    subclass; callers only see frames and events.

    Callbacks
    ---------
    ``start`` takes a single ``on_event`` callback that receives every
    :class:`DialogueEvent` the server emits (or a synthesised one for
    transport errors). Callback delivery is asynchronous — most concrete
    providers schedule it on the event loop. Exceptions in callbacks are
    logged, not re-raised (callbacks must not block the wire task).
    """

    @abstractmethod
    async def start(
        self,
        *,
        on_event: Callable[[DialogueEvent], None],
        config: Optional[DialogueConfig] = None,
    ) -> None:
        """Open the bidirectional channel and begin streaming events.

        Returns once the transport is established; ``on_event`` may fire
        before or after this returns (provider-specific). If the transport
        cannot be opened the callback receives a ``DialogueEvent(type=
        "error")`` describing the failure and the future raises.
        """

    @abstractmethod
    async def feed_audio(self, chunk: bytes) -> None:
        """Push a PCM frame to the server.

        ``chunk`` is mono PCM16 little-endian at :attr:`DialogueConfig
        .sample_rate` (defaults to 16 kHz). Non-blocking: the frame is
        enqueued for the wire task; the actual network send may lag by
        a few milliseconds under load.
        """

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Inject a text message into the conversation (optional path).

        Useful when an external LLM produces the agent reply and we
        want the server to render it as TTS (when ``modality="audio"``)
        or just acknowledge it (when ``modality="text"``). Providers
        that don't expose a text-in channel raise ``NotImplementedError``.
        """

    @abstractmethod
    async def interrupt(self) -> None:
        """User barge-in: cancel in-flight output, keep session open.

        Sends the provider's ``response.cancel`` (or equivalent) event
        to the server, drains any locally buffered output, and stops the
        audio player task. The next ``feed_audio`` frame starts a fresh
        response turn from the server.

        Idempotent: calling twice in a row is a no-op the second time.
        """

    @abstractmethod
    async def stop(self) -> str:
        """End the dialogue and return the final user transcript.

        Suggests the server flush its ASR buffer, waits briefly for the
        final transcript event, then closes the transport. The returned
        string is the concatenated final transcripts across all turns of
        this session (or empty if the server returned no final text).
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any remaining resources (cancel tasks, close sockets).

        Called automatically by :meth:`stop` once the transcript is
        captured. Safe to call more than once.
        """
