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

"""MiniMax Realtime STT provider.

Mirrors the Anthropic :class:`VoiceStreamConnection` surface but routes audio
through MiniMax's Realtime WebSocket API, which exposes a voice-in / text-out
dialogue channel (the server runs ASR + LLM + TTS in-band; we keep only the
ASR/transcript surface). This avoids waiting on MiniMax's standalone ASR
endpoint (not publicly exposed as of 2026-07-02) and lets users with an API
key + ``group_id`` get voice input without Anthropic OAuth or doubao creds.

Two concerns converge here:

* **Runtime integration** — :class:`MiniMaxSTTProvider` implements
  :class:`STTProvider`; :meth:`connect_stream` returns a
  :class:`MiniMaxStreamConnection` that the push-to-talk controller drives.
* **Protocol adapter** — :meth:`_handle_message` parses MiniMax Realtime
  events. The event names follow OpenAI Realtime-API style as an alpha
  assumption (MiniMax's exact names were not fully published when this was
  written); the parser is the single file to edit when the real protocol
  is confirmed, so callers are unaffected.

Auth
----
MiniMax uses an **API key** + **group_id** (HTTP Bearer). No OAuth. The
credentials are read from env vars first (``MINIMAX_API_KEY`` /
``MINIMAX_GROUP_ID``), then from
``~/.clawcodex/tts/minimax/credentials.json`` — same directory convention
as the doubao backend, so the user manages all voice credentials under
``~/.clawcodex/tts/<provider>/``.

Dependencies
------------
``websockets`` is an optional dependency (lazy-imported on connect). If absent,
:meth:`start_streaming` raises ``ImportError`` with a clear install hint.
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
from .stt import STTConfig, STTProvider, STTResult

logger = logging.getLogger(__name__)

__all__ = [
    "MiniMaxCredentialsError",
    "MiniMaxStreamConnection",
    "MiniMaxSTTProvider",
    "MINIMAX_CREDENTIALS_PATH",
    "MINIMAX_REALTIME_ENDPOINTS",
]

# Credentials file location — mirrors doubao: ~/.clawcodex/tts/<provider>/
# Resolved lazily so tests can pin $HOME without caching a stale path.
MINIMAX_CREDENTIALS_PATH = Path("~/.clawcodex/tts/minimax/credentials.json")

# Realtime WebSocket endpoints by region. T2A HTTP endpoints live in
# minimax_tts.py; this dict is STT-specific. Region selection: env
# ``MINIMAX_REGION`` (``global``/``cn``/``uw``) or credentials.json
# ``endpoint_region``. Default = ``global``.
MINIMAX_REALTIME_ENDPOINTS: dict[str, str] = {
    "global": "wss://api.minimax.io/ws/realtime",
    "cn": "wss://api.minimaxi.chat/ws/realtime",
    # MiniMax has not documented a UW realtime endpoint; fall back to global
    # if the user picks uw for low-latency T2A-only use.
    "uw": "wss://api.minimax.io/ws/realtime",
}

# Default Realtime model — MiniMax speech-2.8-turbo (low-latency). The user
# can override via STTConfig.model or credentials.json ``model``.
_MINIMAX_DEFAULT_MODEL = "speech-2.8-turbo"


class MiniMaxCredentialsError(RuntimeError):
    """Raised when MiniMax API key / group_id are missing or malformed.

    The push-to-talk controller maps this to "Configure MINIMAX_API_KEY env
    or ~/.clawcodex/tts/minimax/credentials.json" rather than a bare stack
    trace.
    """


class MiniMaxStreamConnection:
    """Live streaming connection to the MiniMax Realtime API.

    Same surface as :class:`VoiceStreamConnection` (Anthropic) and
    :class:`DoubaoStreamConnection` so the push-to-talk controller is
    provider-agnostic. Internal mechanics: open a WebSocket, send
    ``session.create`` once, then pump PCM frames as
    ``input_audio_buffer.append`` events; the server responds with
    transcript deltas and a final message we capture for ``finalize()``.

    The connection is single-use: one per push-to-talk press. Construct it
    via :meth:`MiniMaxSTTProvider.connect_stream`; do not instantiate
    directly.
    """

    def __init__(
        self,
        *,
        on_transcript: Callable[[str, bool], None],
        on_error: Callable[[str], None],
        on_ready: Optional[Callable[[], None]] = None,
        config: Optional[STTConfig] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        group_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_error = on_error
        self._on_ready = on_ready
        self._config = config or STTConfig()
        self._endpoint = endpoint
        self._api_key = api_key
        self._group_id = group_id
        self._model = model or _MINIMAX_DEFAULT_MODEL
        self._ws: Optional[object] = None
        self._audio_queue = AudioChunkQueue()
        self._pump_tasks: list[asyncio.Task] = []
        self._final_text = ""
        self._ready_event = asyncio.Event()
        self._closed = False
        self._session_started = False

    @property
    def audio_queue(self) -> AudioChunkQueue:
        """The push-side handle the recorder thread writes PCM frames to."""
        return self._audio_queue

    async def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block until the WebSocket handshake + ``session.create`` ack."""
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

        Sends ``input_audio_buffer.commit`` so the server flushes its ASR
        buffer and emits the final transcript, then waits for the pump
        tasks to drain. Mirrors Anthropic
        :meth:`VoiceStreamConnection.finalize` semantics so the controller
        needs no per-provider branch.
        """
        if self._closed:
            return self._final_text
        # Tell the server "user released the key" — flush pending audio.
        if self._ws is not None and not self._closed:
            try:
                await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            except Exception as exc:
                logger.debug("MiniMax commit send failed: %s", exc)
        # Close the producer side so the audio pump terminates.
        self._audio_queue.push(None)
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
                pass  # Cleanup is best-effort and must not replace the primary operation result.
            self._ws = None

    async def _run(self, *, endpoint: str, api_key: str, group_id: str) -> None:
        """Open the socket, send ``session.create``, drive the pumps."""
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            self._on_error("MiniMax STT needs the 'websockets' package: pip install websockets")
            raise RuntimeError("websockets not installed")

        # MiniMax accepts the API key as a Bearer token; group_id may be
        # passed as a URL query (preferred for billing isolation) or in the
        # session.create payload. We put it in the URL so it survives
        # reconnects and isn't replayed in every event.
        url = endpoint
        if group_id and "?" not in url:
            url = f"{url}?group_id={group_id}"
        elif group_id:
            url = f"{url}&group_id={group_id}"

        try:
            self._ws = await websockets.connect(  # type: ignore[attr-defined]
                url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
            )
        except Exception as exc:
            self._on_error(f"MiniMax WebSocket connect failed: {exc}")
            return

        # Send session.create — modalities text-only (we want transcripts,
        # not audio playback; the latter is the TTS path, P64-E).
        try:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "session.create",
                        "session": {
                            "model": self._model,
                            "modalities": ["text"],
                            "input_audio_format": "pcm16",
                            "sample_rate": self._config.sample_rate,
                            "turn_detection": None,  # server-side VAD off; we commit manually
                        },
                    }
                )
            )
        except Exception as exc:
            self._on_error(f"MiniMax session.create failed: {exc}")
            await self.close()
            return

        self._pump_tasks = [
            asyncio.create_task(self._pump_audio()),
            asyncio.create_task(self._pump_transcripts()),
        ]

    async def _pump_audio(self) -> None:
        """Forward PCM frames from the queue as ``input_audio_buffer.append``."""
        if self._ws is None:
            return
        async for chunk in self._audio_queue:
            try:
                encoded = base64.b64encode(chunk).decode("ascii")
                await self._ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": encoded,
                        }
                    )
                )
            except Exception as exc:
                self._on_error(f"MiniMax audio send failed: {exc}")
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
                self._on_error(f"MiniMax transcript stream error: {exc}")

    async def _handle_message(self, raw: object) -> None:
        """Parse one server message → transcript / ready / error callback.

        Single-file protocol adapter: when MiniMax's real event names are
        confirmed, only this method changes. We currently follow OpenAI
        Realtime-API style names (alpha). Tolerant of unknown types
        (forward-compat): logged at debug, no error.

        Event shapes handled:
        * ``session.created`` — session is live; fire ``on_ready``.
        * ``conversation.item.created`` (input) — user transcript item;
          treat as final if it carries text.
        * ``response.text.delta`` — interim transcript chunk.
        * ``response.text.done`` — final transcript for one response turn.
        * ``response.done`` — response turn complete (may carry final text).
        * ``error`` — surface via ``on_error``.
        """
        try:
            payload = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                "MiniMax stream non-JSON message (type=%s, size=%d)",
                type(raw).__name__,
                len(raw) if isinstance(raw, (str, bytes)) else 0,
            )
            return
        if not isinstance(payload, dict):
            logger.debug(
                "MiniMax stream non-object message (type=%s)",
                type(payload).__name__,
            )
            return

        msg_type = payload.get("type") or payload.get("event")
        if msg_type is None:
            logger.debug("MiniMax stream message without type (fields=%d)", len(payload))
            return

        # ── session lifecycle ───────────────────────────────────────────
        if msg_type in ("session.created", "session.ready", "ready"):
            if not self._session_started:
                self._session_started = True
                self._ready_event.set()
                if self._on_ready is not None:
                    try:
                        self._on_ready()
                    except Exception:
                        logger.exception("MiniMax on_ready callback raised")
            return
        if msg_type in ("session.updated", "session.updated"):
            return  # informational, no transcript

        # ── transcript surface ──────────────────────────────────────────
        if msg_type == "response.text.delta":
            delta = payload.get("delta") or ""
            if delta:
                self._on_transcript(delta, False)
            return
        if msg_type in ("response.text.done", "response.done"):
            text = payload.get("text") or (payload.get("response") or {}).get("text") or payload.get("delta") or ""
            if text:
                self._final_text = (self._final_text + " " + text).strip() if self._final_text else text
                self._on_transcript(text, True)
            return
        if msg_type == "conversation.item.created":
            item = payload.get("item") or {}
            text = item.get("content") or item.get("text") or ""
            if isinstance(text, list):
                # OpenAI-style content array → extract text parts
                text = " ".join((p.get("text", "") if isinstance(p, dict) else str(p)) for p in text).strip()
            if text:
                self._final_text = (self._final_text + " " + text).strip() if self._final_text else text
                self._on_transcript(text, True)
            return
        # MiniMax-specific transcript events (alpha guess — confirm on real
        # API; if present these win over OpenAI names).
        if msg_type in ("transcript.partial", "asr.partial"):
            text = payload.get("text") or payload.get("transcript") or ""
            if text:
                self._on_transcript(text, False)
            return
        if msg_type in ("transcript.final", "asr.final"):
            text = payload.get("text") or payload.get("transcript") or ""
            if text:
                self._final_text = (self._final_text + " " + text).strip() if self._final_text else text
                self._on_transcript(text, True)
            return

        # ── error ───────────────────────────────────────────────────────
        if msg_type == "error":
            err = payload.get("error") or {}
            msg = (
                (err.get("message") if isinstance(err, dict) else None)
                or payload.get("message")
                or "MiniMax Realtime error"
            )
            self._on_error(msg)
            return

        logger.debug("MiniMax stream unknown message type: %r", msg_type)


class MiniMaxSTTProvider(STTProvider):
    """MiniMax Realtime API STT backend — API key auth, no OAuth.

    One provider per recording session; construct a fresh one for each
    push-to-talk press (same lifecycle as :class:`AnthropicSTTProvider`).
    Credentials are resolved at :meth:`connect_stream` time so editing
    the credentials file or env mid-session takes effect on the next
    keypress.
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        credentials_path: Optional[Path] = None,
    ) -> None:
        self._endpoint = endpoint
        self._credentials_path = credentials_path or MINIMAX_CREDENTIALS_PATH.expanduser()
        self._connection: Optional[MiniMaxStreamConnection] = None

    # ── credentials resolution ──────────────────────────────────────────

    def _load_credentials_file(self) -> dict[str, Any]:
        """Read ~/.clawcodex/tts/minimax/credentials.json (best-effort).

        Returns ``{}`` if the file is absent (env-only configuration is
        valid). Raises :class:`MiniMaxCredentialsError` only if the file
        exists but is malformed — silent absence lets env-only users skip
        the file entirely.
        """
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
        """Resolve (api_key, group_id) — env first, file fallback.

        Raises :class:`MiniMaxCredentialsError` if no API key is available
        from either source. ``group_id`` may be empty if MiniMax accepts
        requests without it (some accounts are billed at the account
        level); we surface a hint but don't hard-fail on its absence.
        """
        file_data = self._load_credentials_file()
        api_key = os.environ.get("MINIMAX_API_KEY") or file_data.get("api_key")
        group_id = os.environ.get("MINIMAX_GROUP_ID") or file_data.get("group_id") or ""
        if not api_key:
            raise MiniMaxCredentialsError(
                "MINIMAX_API_KEY not set. Configure the env var or "
                f'{self._credentials_path} ({{"api_key": "...", '
                '"group_id": "..."}}).'
            )
        if not group_id:
            logger.debug("MiniMax group_id unset — some accounts allow this")
        return api_key, group_id

    def _resolve_endpoint(self) -> str:
        """Pick the Realtime WebSocket endpoint by region.

        Explicit ``endpoint`` ctor arg wins; else env ``MINIMAX_REGION``;
        else credentials.json ``endpoint_region``; else ``global``.
        """
        if self._endpoint:
            return self._endpoint
        file_data = self._load_credentials_file()
        region = (os.environ.get("MINIMAX_REGION") or file_data.get("endpoint_region") or "global").strip().lower()
        return MINIMAX_REALTIME_ENDPOINTS.get(region) or MINIMAX_REALTIME_ENDPOINTS["global"]

    # ── streaming surface ───────────────────────────────────────────────

    def connect_stream(
        self,
        *,
        on_transcript: Callable[[str, bool], None],
        on_error: Callable[[str], None],
        on_ready: Optional[Callable[[], None]] = None,
        config: Optional[STTConfig] = None,
    ) -> MiniMaxStreamConnection:
        """Open a MiniMax Realtime streaming session and start the pumps."""
        api_key, group_id = self._resolve_credentials()
        endpoint = self._resolve_endpoint()
        conn = MiniMaxStreamConnection(
            on_transcript=on_transcript,
            on_error=on_error,
            on_ready=on_ready,
            config=config,
            endpoint=endpoint,
            api_key=api_key,
            group_id=group_id,
        )
        asyncio.ensure_future(conn._run(endpoint=endpoint, api_key=api_key, group_id=group_id))
        self._connection = conn
        return conn

    # ── STTProvider ABC — batch path via a transient stream ─────────────

    async def transcribe(self, audio_data: bytes, config: STTConfig | None = None) -> STTResult:
        """One-shot batch transcription via a transient Realtime stream."""
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
        conn.feed_audio(audio_data)
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
        return None

    async def feed_audio(self, chunk: bytes) -> STTResult | None:
        return None

    async def stop_streaming(self) -> STTResult:
        if self._connection is None:
            return STTResult(text="")
        text = await self._connection.finalize()
        return STTResult(text=text, is_final=True)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
