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

"""Anthropic STT provider — F-64 P64-A + P64-C.

Mirrors TS ``src/services/voiceStreamSTT.ts``: streams raw PCM audio over
a WebSocket to the Anthropic ``voice_stream`` endpoint (Nova 3 STT model)
and surfaces interim/final transcripts via callbacks.

Two sub-features converge here:
* **P64-A (runtime integration)** — :class:`AnthropicSTTProvider`
  implements the :class:`STTProvider` ABC, binding the OAuth bearer at
  construction and exposing ``start_streaming`` / ``feed_audio`` /
  ``stop_streaming``.
* **P64-C (WebSocket audio transport)** — :meth:`_connect_ws` opens the
  streaming socket; :meth:`_pump_audio` forwards PCM frames from the
  :class:`AudioChunkQueue` to the wire; :meth:`_pump_transcripts` reads
  server-side messages and dispatches interim/final callbacks.

Auth
----
Anthropic STT requires an **OAuth** token (claude.ai subscription), not a
sk-ant- API key. The token is read from the persisted OAuth store at
construction; absence raises :class:`VoiceAuthError` which the
push-to-talk controller surfaces as "run /login first".

Dependencies
------------
``websockets`` is an optional dependency (lazy-imported on connect). If
absent, :meth:`start_streaming` raises ``ImportError`` with a clear
"pip install websockets" hint rather than crashing at module import.
"""

from __future__ import annotations

# Sibling clawcodex_ext packages (auth) are migrated in separate branches;
# suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

import asyncio
import json
import logging
from typing import Callable, Optional

from .audio_chunk_queue import AudioChunkQueue
from .stt import STTConfig, STTProvider, STTResult

logger = logging.getLogger(__name__)

__all__ = [
    "VoiceAuthError",
    "VoiceStreamConnection",
    "AnthropicSTTProvider",
    "ANTHROPIC_VOICE_ENDPOINT",
]

# The voice_stream WebSocket endpoint. Host is claude.ai (OAuth-gated,
# not the api.anthropic.com API-key surface). Path matches the TS
# upstream ``voiceStreamSTT.ts`` constant.
ANTHROPIC_VOICE_ENDPOINT = "wss://claude.ai/api/voice_stream"


class VoiceAuthError(RuntimeError):
    """Raised when the Anthropic STT backend has no usable OAuth token.

    Distinct from a missing API key: the voice endpoint is OAuth-only.
    The push-to-talk controller catches this and surfaces "run /login
    first" rather than a raw auth stack trace.
    """


class VoiceStreamConnection:
    """Live streaming connection to the Anthropic voice endpoint.

    Mirrors TS ``VoiceStreamConnection``: an opaque handle the
    push-to-talk controller holds while recording. The handle exposes
    ``feed_audio`` (push PCM frames), ``finalize`` (await final
    transcript), and ``close`` (release the socket).

    Constructed by :meth:`AnthropicSTTProvider.connect_stream`; users
    should not instantiate it directly.
    """

    def __init__(
        self,
        *,
        on_transcript: Callable[[str, bool], None],
        on_error: Callable[[str], None],
        on_ready: Optional[Callable[[], None]] = None,
        config: Optional[STTConfig] = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_error = on_error
        self._on_ready = on_ready
        self._config = config or STTConfig()
        self._ws: Optional[object] = None  # websockets.WebSocketClientProtocol
        self._audio_queue = AudioChunkQueue()
        self._pump_tasks: list[asyncio.Task] = []
        self._final_text = ""
        self._ready_event = asyncio.Event()
        self._closed = False

    @property
    def audio_queue(self) -> AudioChunkQueue:
        """The push-side handle the recorder thread writes PCM frames to."""
        return self._audio_queue

    async def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block until the WebSocket handshake completes (or timeout).

        Returns ``True`` if ready, ``False`` on timeout. The
        push-to-talk controller uses this to decide whether to start
        feeding audio immediately or buffer briefly while the socket
        comes up. Mirrors TS ``onReady`` semantics.
        """
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def feed_audio(self, chunk: bytes) -> None:
        """Push a PCM frame from the recorder thread (thread-safe)."""
        self._audio_queue.push(chunk)

    async def finalize(self) -> str:
        """Signal end-of-stream and await the final transcript.

        Mirrors TS ``finalize``: pushes ``None`` into the audio queue
        (closing the producer side), waits for the server's final
        transcript message, then closes the socket. Returns the
        concatenated final text.
        """
        if self._closed:
            return self._final_text
        self._audio_queue.push(None)
        # Wait for the transcript pump to observe the final message and
        # complete the pump task.
        if self._pump_tasks:
            await asyncio.gather(*self._pump_tasks, return_exceptions=True)
        await self.close()
        return self._final_text

    async def close(self) -> None:
        """Release the WebSocket and cancel any pending pumps."""
        if self._closed:
            return
        self._closed = True
        for task in self._pump_tasks:
            if not task.done():
                task.cancel()
        self._pump_tasks.clear()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # nosec B110 - best-effort websocket close during shutdown
                pass
            self._ws = None

    async def _run(
        self,
        *,
        endpoint: str,
        bearer: str,
    ) -> None:
        """Open the socket, pump audio + transcripts, drive callbacks."""
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError as exc:
            self._on_error("Anthropic STT needs the 'websockets' package: pip install websockets")
            raise RuntimeError("websockets not installed") from exc

        try:
            self._ws = await websockets.connect(  # type: ignore[attr-defined]
                endpoint,
                additional_headers={"Authorization": f"Bearer {bearer}"},
            )
        except Exception as exc:
            self._on_error(f"Voice WebSocket connect failed: {exc}")
            return

        # Signal ready: the controller may now start pushing audio.
        self._ready_event.set()
        if self._on_ready is not None:
            try:
                self._on_ready()
            except Exception:
                logger.exception("on_ready callback raised")

        # Two concurrent pumps: audio (queue → socket) and transcripts
        # (socket → callbacks). Both stop when the audio queue closes
        # (finalize) or the socket drops.
        self._pump_tasks = [
            asyncio.create_task(self._pump_audio()),
            asyncio.create_task(self._pump_transcripts()),
        ]

    async def _pump_audio(self) -> None:
        """Forward PCM frames from the queue to the WebSocket."""
        if self._ws is None:
            return
        async for chunk in self._audio_queue:
            try:
                await self._ws.send(chunk)
            except Exception as exc:
                self._on_error(f"Voice audio send failed: {exc}")
                return

    async def _pump_transcripts(self) -> None:
        """Read server messages and dispatch transcript callbacks."""
        if self._ws is None:
            return
        try:
            async for raw in self._ws:  # type: ignore[attr-defined]
                await self._handle_message(raw)
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception as exc:
            if not self._closed:
                self._on_error(f"Voice transcript stream error: {exc}")

    async def _handle_message(self, raw: object) -> None:
        """Parse one server message → transcript callback.

        The Anthropic voice protocol emits JSON messages of the shape
        ``{"type": "interim"|"final"|"error", "text": "..."}``. We
        tolerate missing fields and unknown types (forward-compat) by
        logging at debug and continuing.
        """
        try:
            payload = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
        except (json.JSONDecodeError, TypeError):
            logger.debug("Voice stream non-JSON message: %r", raw)
            return
        msg_type = payload.get("type")
        text = payload.get("text", "")
        if msg_type == "interim":
            self._on_transcript(text, False)
        elif msg_type == "final":
            self._final_text = (self._final_text + " " + text).strip() if self._final_text else text
            self._on_transcript(text, True)
        elif msg_type == "error":
            self._on_error(payload.get("message") or text or "unknown voice error")
        else:
            logger.debug("Voice stream unknown message type: %r", msg_type)


class AnthropicSTTProvider(STTProvider):
    """Anthropic OAuth-gated STT backend (Nova 3 via WebSocket).

    The provider is a thin factory for :class:`VoiceStreamConnection`
    instances; the heavy lifting (socket, pumps) lives on the connection.
    One provider per recording session; construct a fresh one for each
    push-to-talk press.
    """

    def __init__(self, *, endpoint: str = ANTHROPIC_VOICE_ENDPOINT) -> None:
        self._endpoint = endpoint
        self._connection: Optional[VoiceStreamConnection] = None
        # Token resolved lazily on connect so a provider constructed before
        # login still works if the user logs in before pressing the key.
        self._bearer: Optional[str] = None

    def _resolve_bearer(self) -> str:
        """Read the OAuth bearer from the persisted store.

        Raises :class:`VoiceAuthError` if no token is available — the
        push-to-talk controller maps this to "run /login first".
        """
        if self._bearer:
            return self._bearer
        try:
            from clawcodex_ext.auth.codex_store import get_oauth_access_token

            token = get_oauth_access_token()
        except Exception as exc:
            raise VoiceAuthError("Anthropic STT needs an OAuth token (claude.ai login). Run /login first.") from exc
        if not token:
            raise VoiceAuthError("Anthropic STT needs an OAuth token (claude.ai login). Run /login first.")
        self._bearer = token
        return token

    def connect_stream(
        self,
        *,
        on_transcript: Callable[[str, bool], None],
        on_error: Callable[[str], None],
        on_ready: Optional[Callable[[], None]] = None,
        config: Optional[STTConfig] = None,
    ) -> VoiceStreamConnection:
        """Open a streaming connection and start the pumps.

        Returns a :class:`VoiceStreamConnection` ready to receive audio
        via ``feed_audio``. The caller drives the lifecycle: push frames
        while recording, then ``await finalize()`` to get the final
        transcript and close the socket.
        """
        bearer = self._resolve_bearer()
        conn = VoiceStreamConnection(
            on_transcript=on_transcript,
            on_error=on_error,
            on_ready=on_ready,
            config=config,
        )
        # Schedule the connection run on the current loop. The pumps
        # inside ``_run`` block on the audio queue, so this doesn't
        # busy-wait while waiting for the first frame.
        asyncio.ensure_future(conn._run(endpoint=self._endpoint, bearer=bearer))
        self._connection = conn
        return conn

    # ── STTProvider ABC ────────────────────────────────────────────────
    # The ABC's batch ``transcribe`` isn't the streaming path the
    # push-to-talk controller uses, but we implement it via a one-shot
    # stream so the provider is usable from non-interactive code paths
    # (tests, future SDK callers).

    async def transcribe(self, audio_data: bytes, config: STTConfig | None = None) -> STTResult:
        """One-shot batch transcription via a transient stream."""
        final_text = ""
        done = asyncio.Event()
        error: list[str] = []

        def _on_transcript(text: str, is_final: bool) -> None:
            nonlocal final_text
            if is_final:
                final_text = (final_text + " " + text).strip() if final_text else text

        def _on_error(msg: str) -> None:
            error.append(msg)
            done.set()

        conn = self.connect_stream(on_transcript=_on_transcript, on_error=_on_error, config=config)
        # Feed the whole clip then close the producer side.
        conn.feed_audio(audio_data)
        # Wait for finalize in a task so we can race against error/timeout.
        finalize_task = asyncio.create_task(conn.finalize())
        done_task = asyncio.create_task(done.wait())
        try:
            await asyncio.wait_for(
                asyncio.wait(
                    {finalize_task, done_task},
                    return_when=asyncio.FIRST_COMPLETED,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            await conn.close()
            return STTResult(text="", confidence=0.0, is_final=True, duration_ms=30000.0)
        finally:
            finalize_task.cancel()
            done_task.cancel()
        if error:
            return STTResult(text="", confidence=0.0, is_final=True)
        return STTResult(text=final_text, confidence=1.0, is_final=True)

    async def start_streaming(self, config: STTConfig | None = None) -> None:
        """No-op: streaming is initiated via :meth:`connect_stream`.

        Kept for ABC compliance; the push-to-talk controller uses
        ``connect_stream`` directly because it needs the callback wiring.
        """
        return None

    async def feed_audio(self, chunk: bytes) -> STTResult | None:
        """Not used on this provider — see ``connect_stream`` + ``feed_audio``."""
        return None

    async def stop_streaming(self) -> STTResult:
        """Finalize the active connection if one exists."""
        if self._connection is None:
            return STTResult(text="")
        text = await self._connection.finalize()
        return STTResult(text=text, is_final=True)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
