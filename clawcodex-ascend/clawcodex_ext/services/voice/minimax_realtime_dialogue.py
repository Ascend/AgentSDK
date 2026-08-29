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

"""MiniMax Realtime API full-duplex dialogue provider.

Implementation of :class:`FullDuplexDialogueProvider` that talks to
MiniMax's Realtime WebSocket endpoint (``wss://api.minimax.io/ws/realtime``
or its ``cn`` / ``uw`` regional siblings). The server-side stack is
ASR + LLM + TTS in a single pipe — this module mirrors the upstream
event names following the OpenAI Realtime-API convention, with a
MiniMax-specific quirk section at the bottom of :meth:`_handle_message`
that matches the partial protocol assumption in :mod:`minimax_stt`.

Design
------
One :class:`MiniMaxRealtimeDialogueProvider` per session lifecycle. The
constructor is cheap (no I/O); credentials + transport are resolved
inside :meth:`start` so editing
``~/.clawcodex/tts/minimax/credentials.json`` mid-session takes effect on
the next ``/dialogue start`` call.

The provider drives three background tasks once started:

* ``_recv_task`` — pumps server events, dispatches transcripts / audio /
 lifecycle signals through ``on_event``.
* ``_send_task`` — drains an internal async queue of outgoing JSON
 payloads (session.create + per-frame audio append + interrupt cancels
 + text injections). Keeping a single ordered send task simplifies
 back-pressure: ``feed_audio`` queues the frame and returns fast.
* ``_keepalive_task`` (optional) — periodic ping so idle sessions don't
 time out on networks that close silent sockets. Disabled by default;
 enabled via ``DialogueConfig.extra["keepalive_seconds"]`` if the user
 wants it.

Auth + protocol
---------------
MiniMax accepts the API key as a Bearer token; ``group_id`` rides in the
URL query for billing isolation (the same convention used by
:class:`MiniMaxStreamConnection`). The session is initialised by sending
``session.create`` with ``modalities`` reflecting the requested output
modality (``text`` / ``audio``) — the server's ``session.created`` response
flips the provider into "ready" and unblocks any caller
awaiting :meth:`feed_audio`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from .audio_chunk_queue import AudioChunkQueue
from .dialogue import (
    DialogueConfig,
    DialogueEvent,
    FullDuplexDialogueProvider,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MiniMaxCredentialsError",
    "MiniMaxRealtimeDialogueProvider",
    "MINIMAX_REALTIME_CREDENTIALS_PATH",
    "MINIMAX_REALTIME_ENDPOINTS",
]

# Same path as ``minimax_stt``: ~/.clawcodex/tts/minimax/credentials.json.
# We re-declare it locally rather than re-exporting to keep the dialogue
# adapter self-contained for tests that don't import the STT module.
MINIMAX_REALTIME_CREDENTIALS_PATH = Path("~/.clawcodex/tts/minimax/credentials.json")

# Realtime WebSocket endpoints (same as P64-D1).
MINIMAX_REALTIME_ENDPOINTS: dict[str, str] = {
    "global": "wss://api.minimax.io/ws/realtime",
    "cn": "wss://api.minimaxi.chat/ws/realtime",
    # No documented UW realtime endpoint — fall back to global if the
    # user explicitly picks uw for low-latency T2A only.
    "uw": "wss://api.minimax.io/ws/realtime",
}

# Default model — the same speech-2.8-turbo used by ``minimax_stt``; the
# realtime dial on MiniMax exposes both ASR and LLM-capable variants.
_MINIMAX_DEFAULT_MODEL = "speech-2.8-turbo"


class MiniMaxCredentialsError(RuntimeError):
    """Raised when MiniMax API key / group_id are missing.

    The dialogue session manager maps this to a user-friendly "configure
    MINIMAX_API_KEY / ~/.clawcodex/tts/minimax/credentials.json" hint.
    """


class MiniMaxRealtimeDialogueProvider(FullDuplexDialogueProvider):
    """MiniMax Realtime full-duplex voice dialogue provider.

    Lifecycle::

        provider = MiniMaxRealtimeDialogueProvider()
        await provider.start(on_event=my_handler, config=DialogueConfig())
        await provider.feed_audio(pcm_chunk)  # many of these
        await provider.send_text("hello")     # optional text injection
        await provider.interrupt()            # optional barge-in
        final = await provider.stop()
        await provider.close()

    Connection state survives :meth:`interrupt`; only :meth:`stop` /
    :meth:`close` tear down the WebSocket.
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        credentials_path: Optional[Path] = None,
    ) -> None:
        self._endpoint = endpoint
        self._credentials_path = credentials_path or MINIMAX_REALTIME_CREDENTIALS_PATH.expanduser()
        self._ws: Optional[Any] = None
        self._config: Optional[DialogueConfig] = None
        self._on_event: Optional[Callable[[DialogueEvent], None]] = None
        self._out_queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()
        self._in_queue = AudioChunkQueue()
        self._tasks: list[asyncio.Task] = []
        self._ready_event = asyncio.Event()
        self._closed = False
        self._session_started = False
        self._final_text_parts: list[str] = []
        # Drained ``response.audio.delta`` frames whose order matters
        # for playback. We hand them directly to ``on_event`` so the
        # session manager can choose to forward them to AudioOutQueue or
        # to a text-passthrough handler.
        self._audio_sample_rate = 24000

    # ── credentials (same pattern as minimax_stt) ─────────────────────────

    def _load_credentials_file(self) -> dict[str, Any]:
        path = self._credentials_path
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise MiniMaxCredentialsError(f"MiniMax credentials file {path} is unreadable: {exc}") from exc
        if not isinstance(data, dict):
            raise MiniMaxCredentialsError(f"MiniMax credentials file {path} must be a JSON object")
        return data

    def _resolve_credentials(self) -> tuple[str, str]:
        file_data = self._load_credentials_file()
        api_key = os.environ.get("MINIMAX_API_KEY") or file_data.get("api_key")
        group_id = os.environ.get("MINIMAX_GROUP_ID") or file_data.get("group_id") or ""
        if not api_key:
            raise MiniMaxCredentialsError(
                "MINIMAX_API_KEY not set. Configure the env var or "
                f'{self._credentials_path} ({{"api_key": "...", '
                '"group_id": "..."}}).'
            )
        return api_key, group_id

    def _resolve_endpoint(self) -> str:
        if self._endpoint:
            return self._endpoint
        file_data = self._load_credentials_file()
        region = (os.environ.get("MINIMAX_REGION") or file_data.get("endpoint_region") or "global").strip().lower()
        return MINIMAX_REALTIME_ENDPOINTS.get(region) or MINIMAX_REALTIME_ENDPOINTS["global"]

    # ── FullDuplexDialogueProvider surface ────────────────────────────────

    async def start(
        self,
        *,
        on_event: Callable[[DialogueEvent], None],
        config: Optional[DialogueConfig] = None,
    ) -> None:
        """Open the WebSocket and pump events through ``on_event``."""
        if self._ws is not None:
            raise RuntimeError("MiniMax realtime dialogue already started")
        self._on_event = on_event
        self._config = config or DialogueConfig()
        self._audio_sample_rate = self._config.output_sample_rate

        try:
            api_key, group_id = self._resolve_credentials()
        except MiniMaxCredentialsError as exc:
            # Surface via the event hook so the session manager can show
            # a settings hint, then re-raise for ``/dialogue start``.
            self._emit(DialogueEvent(type="error", message=str(exc)))
            raise

        endpoint = self._resolve_endpoint()
        if group_id and "?" not in endpoint:
            endpoint = f"{endpoint}?group_id={group_id}"
        elif group_id:
            endpoint = f"{endpoint}&group_id={group_id}"

        # Lazy import: websockets is an optional dep (same as).
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = "MiniMax dialogue needs the 'websockets' package: pip install websockets"
            self._emit(DialogueEvent(type="error", message=msg))
            raise RuntimeError(msg) from exc

        try:
            self._ws = await websockets.connect(  # type: ignore[attr-defined]
                endpoint,
                additional_headers={"Authorization": f"Bearer {api_key}"},
            )
        except Exception as exc:
            self._emit(DialogueEvent(type="error", message=f"MiniMax connect failed: {exc}"))
            self._ws = None
            raise

        # Tell the server what shape of session we want. ``output_modality``
        # maps onto ``modalities``: ``"text"`` → ``["text"]``;
        # ``"audio"`` → ``["text", "audio"]`` (text is still useful for
        # the dialog manager so the caller can echo / log it).
        modalities = ["text", "audio"] if self._config.modality == "audio" else ["text"]
        await self._send_now(
            {
                "type": "session.create",
                "session": {
                    "model": self._config.model,
                    "modalities": modalities,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "sample_rate": self._config.sample_rate,
                    "voice": self._config.voice or None,
                    "language": self._config.language,
                    "interim_results": self._config.interim_results,
                    "turn_detection": None,  # server VAD off; we manage turn-take ourselves
                    **(self._config.extra or {}),
                },
            }
        )

        # Kick off the three background tasks. Each catches its own
        # exceptions so a single one dying doesn't cascade — they call
        # ``_emit(error)`` and exit, the others keep running until
        # ``stop`` / ``close`` joins them.
        self._tasks = [
            asyncio.create_task(self._recv_loop(), name="minimax-dialogue-recv"),
            asyncio.create_task(self._send_loop(), name="minimax-dialogue-send"),
            asyncio.create_task(self._audio_pump_loop(), name="minimax-dialogue-audio"),
        ]

    async def feed_audio(self, chunk: bytes) -> None:
        """Push a PCM frame for the user microphone."""
        if self._closed:
            return
        # Hand the chunk to the audio pump task via AudioChunkQueue. The
        # producer (e.g. recorder thread) may call this from a sync
        # thread; the pump task is async and does the base64+JSON encode.
        self._in_queue.push(chunk)

    async def send_text(self, text: str) -> None:
        """Inject a text message into the conversation."""
        if not text:
            return
        if self._closed:
            return
        # OpenAI Realtime-style ``conversation.item.create`` with input_text.
        # The server treats it as if the user said it (for LLM context); the
        # provider's text path turns it into a response stream just like a
        # spoken turn would.
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def interrupt(self) -> None:
        """User barge-in: cancel the in-flight response on the server side.

        Sends ``response.cancel``; the server drops any unrendered audio /
        text in its current response. Locally the session manager must
        also stop its audio player and clear the output queue — this method
        is just the wire-side step.
        """
        if self._closed:
            return
        await self._send({"type": "response.cancel"})
        self._emit(DialogueEvent(type="interrupt"))

    async def stop(self) -> str:
        """Graceful shutdown: commit audio, drain, close.

        Returns the concatenated final transcript parts seen during this
        session. If no final transcript was emitted (the user never
        spoke, or the server only sent audio), the return is empty.
        """
        if self._closed:
            return self._final_text()
        # Tell the server to flush any partial audio / speech buffer.
        try:
            await self._send({"type": "input_audio_buffer.commit"})
        except Exception:  # nosec B110 - best-effort flush before close; failure is harmless
            pass  # Committing the final audio buffer is best-effort during shutdown.
        # Brief grace period for the server's final transcript event to
        # arrive (best-effort; the WS recv loop is bounded by ``close``).
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Cleanup is best-effort and must not replace the primary operation result.
        await self.close()
        return self._final_text()

    async def close(self) -> None:
        """Release the WebSocket and cancel all pump tasks. Idempotent."""
        if self._closed and not self._tasks:
            return
        self._closed = True
        # Signal end-of-stream to the audio pump so it stops forwarding.
        self._in_queue.push(None)
        # Wake the send task with a sentinel (None) so it exits.
        await self._out_queue.put(None)
        for task in self._tasks:
            if not task.done():
                task.cancel()
        # Drain the cancellation results (don't leak unawaited
        # CancelledErrors as warnings).
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # nosec B110 - best-effort websocket close during shutdown
                pass  # Cleanup is best-effort and must not replace the primary operation result.
            self._ws = None

    # ── helpers ───────────────────────────────────────────────────────────

    def _final_text(self) -> str:
        return " ".join(p for p in self._final_text_parts if p).strip()

    def _emit(self, event: DialogueEvent) -> None:
        cb = self._on_event
        if cb is None:
            return
        try:
            cb(event)
        except Exception:
            logger.exception("Dialogue on_event callback raised")

    async def _send(self, payload: dict[str, Any]) -> None:
        """Async enqueue an outgoing payload (frames are dispatched in order)."""
        await self._out_queue.put(payload)

    async def _send_now(self, payload: dict[str, Any]) -> None:
        """Send an outgoing payload directly, bypassing the queue.

        Used for the ``session.create`` handshake before any pump is
        started — we *must* deliver it before the recv loop begins
        otherwise the server can drop its ack on a full TCP buffer.
        """
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as exc:
            logger.debug("MiniMax dialogue direct send failed: %s", exc)

    # ── background tasks ─────────────────────────────────────────────────

    async def _send_loop(self) -> None:
        """Drain ``_out_queue`` and write each payload to the WebSocket."""
        if self._ws is None:
            return
        try:
            while True:
                payload = await self._out_queue.get()
                if payload is None:  # close sentinel
                    return
                try:
                    await self._ws.send(json.dumps(payload))
                except Exception as exc:
                    self._emit(DialogueEvent(type="error", message=f"MiniMax send failed: {exc}"))
                    return
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception as exc:
            if not self._closed:
                self._emit(DialogueEvent(type="error", message=f"Send loop error: {exc}"))

    async def _audio_pump_loop(self) -> None:
        """Forward PCM frames as ``input_audio_buffer.append`` events."""
        if self._ws is None:
            return
        try:
            async for chunk in self._in_queue:
                if not chunk:
                    continue
                try:
                    encoded = base64.b64encode(chunk).decode("ascii")
                    await self._send({"type": "input_audio_buffer.append", "audio": encoded})
                except Exception as exc:
                    self._emit(DialogueEvent(type="error", message=f"MiniMax audio send failed: {exc}"))
                    return
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception as exc:
            if not self._closed:
                self._emit(DialogueEvent(type="error", message=f"Audio pump error: {exc}"))

    async def _recv_loop(self) -> None:
        """Read server messages and dispatch ``DialogueEvent``s."""
        if self._ws is None:
            return
        try:
            async for raw in self._ws:  # type: ignore[attr-defined]
                await self._handle_message(raw)
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception as exc:
            if not self._closed:
                self._emit(DialogueEvent(type="error", message=f"Recv loop error: {exc}"))

    async def _handle_message(self, raw: object) -> None:
        """Parse one server message and dispatch a :class:`DialogueEvent`.

        Mirrors the ``minimax_stt._handle_message`` shape so the two
        adapters stay symmetric; the only added surface is the audio
        delta path (``response.audio.delta``).
        """
        try:
            payload = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                "MiniMax dialogue non-JSON message (type=%s, size=%d)",
                type(raw).__name__,
                len(raw) if isinstance(raw, (str, bytes)) else 0,
            )
            return
        if not isinstance(payload, dict):
            return

        msg_type = payload.get("type") or payload.get("event")
        if msg_type is None:
            return

        # ── session lifecycle ────────────────────────────────────────────
        if msg_type in ("session.created", "session.ready", "ready"):
            if not self._session_started:
                self._session_started = True
                self._ready_event.set()
                self._emit(DialogueEvent(type="ready"))
            return
        if msg_type in ("session.updated",):
            return  # informational

        # ── transcript surface ───────────────────────────────────────────
        # Interim chunk — accumulated into ``is_final=False`` events; the
        # session manager decides whether to display or buffer them.
        if msg_type == "response.text.delta":
            delta = payload.get("delta") or ""
            if delta:
                self._emit(DialogueEvent(type="transcript", text=delta, is_final=False))
            return
        # Final text for the current response turn.
        if msg_type in ("response.text.done", "response.done"):
            text = payload.get("text") or (payload.get("response") or {}).get("text") or ""
            if text:
                self._final_text_parts.append(text)
                self._emit(DialogueEvent(type="transcript", text=text, is_final=True))
                # If ``modality=="text"`` we treat the final transcript
                # as the agent reply boundary too — the session manager
                # gets the same ``done`` event so it can mark the turn
                # complete.
                if self._config is not None and self._config.modality == "text":
                    self._emit(DialogueEvent(type="done", text=text))
            return

        # ── audio surface ────────────────────────────────────────────────
        # Individual PCM delta frames — the session manager feeds them
        # into ``AudioOutQueue`` so ``AudioPlayer`` plays them in order.
        if msg_type == "response.audio.delta":
            delta_b64 = payload.get("delta") or ""
            if not delta_b64:
                return
            try:
                pcm = base64.b64decode(delta_b64)
            except Exception:
                logger.debug("MiniMax dialogue audio delta decode failed")
                return
            self._emit(
                DialogueEvent(
                    type="audio",
                    pcm=pcm,
                    sample_rate=self._audio_sample_rate,
                )
            )
            return
        # Marker for the end of the audio stream for one response turn.
        if msg_type == "response.audio.done":
            self._emit(DialogueEvent(type="done"))
            return

        # ── VAD surface (MiniMax-specific if exposed) ────────────────────
        if msg_type in (
            "input_audio_buffer.speech_started",
            "speech.started",
            "speech_started",
        ):
            self._emit(DialogueEvent(type="speech_started"))
            return
        if msg_type in (
            "input_audio_buffer.speech_stopped",
            "speech.stopped",
            "speech_stopped",
        ):
            self._emit(DialogueEvent(type="speech_stopped"))
            return

        # ── error ────────────────────────────────────────────────────────
        if msg_type == "error":
            err = payload.get("error") or {}
            msg = (
                (err.get("message") if isinstance(err, dict) else None)
                or payload.get("message")
                or "MiniMax Realtime error"
            )
            self._emit(DialogueEvent(type="error", message=msg))
            return

        logger.debug("MiniMax dialogue unknown message type: %r", msg_type)
