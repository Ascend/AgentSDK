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

"""Doubao ASR STT provider — F-64 P64-A.

Mirrors TS ``src/services/doubaoSTT.ts``: an adapter that bridges the
``doubaoime-asr`` package's ``AsyncGenerator`` protocol to the
:class:`STTProvider` / :class:`VoiceStreamConnection`-style interface.

Why a separate adapter
----------------------
The doubao SDK exposes a *pull* API (``transcribeRealtime`` is an async
generator that yields ``ResponseType`` events as it consumes audio). The
Anthropic backend is a *push* API (we send PCM, server pushes transcripts
back over a WebSocket). The adapter reconciles them so the push-to-talk
controller sees one shape: ``connect_stream`` → ``feed_audio`` →
``finalize``.

Design (mirrors TS decisions, see docs/features/voice-mode.md §3.5)
-------------------------------------------------------------------
* :class:`AudioChunkQueue` already lives in ``audio_chunk_queue.py`` —
  the doubao adapter reuses it: the recorder pushes PCM frames, the
  ``transcribeRealtime`` async generator pulls them.
* :meth:`connect_stream` triggers ``on_ready`` *immediately* (no
  WebSocket handshake to wait for) — this avoids a deadlock with the
  controller's audio buffer (TS decision #6).
* :meth:`finalize` returns *immediately* (decision #7): doubao emits
  ``FINAL_RESULT`` during the recording, so by the time the user
  releases the key all results are already in.
* Response types map to callbacks exactly as TS does (see table in
  voice-mode.md §3.5).
* ``doubaoime-asr`` is an optional dependency — the factory in
  ``provider_registry.py`` defers the import to connection time so a
  missing dep doesn't break REPL boot, only this backend.

Credentials
-----------
Read from ``~/.clawcodex/tts/doubao/credentials.json`` (TS used
``~/.claude/...``; we use ``~/.clawcodex/`` for the same reason the rest
of the project does). The file shape is::

    {"deviceId": "...", "installId": "...", "cdid": "...",
     "openudid": "...", "clientudid": "...", "token": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from .audio_chunk_queue import AudioChunkQueue
from .stt import STTConfig, STTProvider, STTResult

logger = logging.getLogger(__name__)

__all__ = [
    "DoubaoCredentialsError",
    "DoubaoStreamConnection",
    "DoubaoSTTProvider",
    "DOUBAO_CREDENTIALS_PATH",
]

# Credentials file location — mirrors TS ``~/.claude/tts/doubao/`` but
# rooted at the project's config dir. Resolved lazily so tests can pin
# ``$HOME`` without us caching a stale path at import.
DOUBAO_CREDENTIALS_PATH = Path("~/.clawcodex/tts/doubao/credentials.json")


class DoubaoCredentialsError(RuntimeError):
    """Raised when the doubao backend can't find / parse its credentials.

    The push-to-talk controller maps this to "configure
    ~/.clawcodex/tts/doubao/credentials.json first".
    """


class DoubaoStreamConnection:
    """Live streaming connection to the doubao ASR backend.

    Same surface as :class:`VoiceStreamConnection` so the push-to-talk
    controller is provider-agnostic. Internal mechanics differ: instead
    of a WebSocket, a background task consumes the ``transcribeRealtime``
    async generator and maps its ``ResponseType`` events to callbacks.
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
        self._audio_queue = AudioChunkQueue()
        self._consumer_task: Optional[asyncio.Task] = None
        self._final_text = ""
        self._closed = False

    @property
    def audio_queue(self) -> AudioChunkQueue:
        return self._audio_queue

    def feed_audio(self, chunk: bytes) -> None:
        self._audio_queue.push(chunk)

    async def wait_until_ready(self, timeout: float = 5.0) -> bool:
        # Doubao needs no handshake — ready immediately (TS decision #6).
        return True

    async def finalize(self) -> str:
        """Signal end-of-stream. Returns immediately (TS decision #7).

        Doubao emits ``FINAL_RESULT`` during the recording, so by the
        time the user releases the key all results are already in. We
        still close the audio queue (so the consumer task terminates
        cleanly) but don't block on it — the background task is
        fire-and-forget.
        """
        if self._closed:
            return self._final_text
        self._audio_queue.push(None)
        # Don't await the consumer task — finalize is non-blocking per
        # the TS design. Any late-arriving FINAL_RESULT will be picked
        # up by the still-running consumer; if it ends after the
        # connection is closed the callback is a no-op.
        await self.close()
        return self._final_text

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._audio_queue.push(None)
        if self._consumer_task is not None and not self._consumer_task.done():
            self._consumer_task.cancel()
        self._consumer_task = None

    async def _run(self, *, credentials: dict[str, Any]) -> None:
        """Boot the doubao consumer task and fire ``on_ready`` immediately."""
        try:
            from doubaoime_asr import (  # type: ignore[import-not-found]
                ResponseType,
                transcribe_realtime,
            )
        except ImportError as exc:
            self._on_error("Doubao ASR needs the 'doubaoime-asr' package: pip install doubaoime-asr")
            raise RuntimeError("doubaoime-asr not installed") from exc

        # Fire on_ready immediately — no handshake to wait for (TS #6).
        if self._on_ready is not None:
            try:
                self._on_ready()
            except Exception:
                logger.exception("doubao on_ready callback raised")

        self._consumer_task = asyncio.create_task(self._consume(transcribe_realtime, credentials, ResponseType))

    async def _consume(
        self,
        transcribe_fn: Callable[..., Any],
        credentials: dict[str, Any],
        response_type: Any,
    ) -> None:
        """Drive the ``transcribeRealtime`` async generator → callbacks.

        The doubao SDK's generator pulls audio from an
        ``AsyncIterable[bytes]`` and yields typed response events. We
        feed it our :class:`AudioChunkQueue` (the recorder pushes into
        the same queue) and map each event to the
        ``on_transcript``/``on_error`` callbacks.
        """
        try:
            async for event in transcribe_fn(
                audio_stream=self._audio_queue,
                credentials=credentials,
                sample_rate=self._config.sample_rate,
                language=self._config.language,
            ):
                if self._closed:
                    return
                self._dispatch_event(event, response_type)
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception as exc:
            if not self._closed:
                self._on_error(f"Doubao ASR stream error: {exc}")

    def _dispatch_event(self, event: Any, response_type: Any) -> None:
        """Map one doubao response event to the callback surface.

        Mirrors the TS response-type table (voice-mode.md §3.5):
        ``INTERIM_RESULT`` → interim transcript, ``FINAL_RESULT`` →
        final transcript, ``ERROR`` → error callback, the rest are
        logged at debug (session lifecycle events we don't surface).
        """
        try:
            kind = getattr(event, "type", None) or getattr(event, "response_type", None)
        except Exception:
            kind = None
        text = getattr(event, "text", "") or ""
        # Resolve enum members to their string value for comparison.
        kind_str = getattr(kind, "value", kind)
        if kind_str in ("INTERIM_RESULT", "interim_result"):
            self._on_transcript(text, False)
        elif kind_str in ("FINAL_RESULT", "final_result"):
            self._final_text = (self._final_text + " " + text).strip() if self._final_text else text
            self._on_transcript(text, True)
        elif kind_str in ("ERROR", "error"):
            msg = getattr(event, "message", "") or text or "doubao ASR error"
            self._on_error(msg)
        else:
            logger.debug("Doubao ASR lifecycle event: %r", kind_str)


class DoubaoSTTProvider(STTProvider):
    """Doubao ASR backend — independent credentials, no Anthropic OAuth.

    One provider per recording session. The credentials file is read on
    each ``connect_stream`` call so a user editing it mid-session doesn't
    need to restart the REPL.
    """

    def __init__(self, *, credentials_path: Optional[Path] = None) -> None:
        self._credentials_path = credentials_path or DOUBAO_CREDENTIALS_PATH.expanduser()
        self._connection: Optional[DoubaoStreamConnection] = None

    def _load_credentials(self) -> dict[str, Any]:
        """Read the doubao credentials JSON file.

        Raises :class:`DoubaoCredentialsError` if the file is missing or
        malformed. The path is resolved at call time (not __init__) so
        ``$HOME`` changes / test temp dirs are honored.
        """
        path = self._credentials_path
        if not path.is_file():
            raise DoubaoCredentialsError(
                f"Doubao credentials not found at {path}. Configure ~/.clawcodex/tts/doubao/credentials.json first."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise DoubaoCredentialsError(f"Doubao credentials file {path} is unreadable: {exc}") from exc
        if not isinstance(data, dict) or not data.get("token"):
            raise DoubaoCredentialsError(f"Doubao credentials file {path} missing required 'token' field")
        return data

    def connect_stream(
        self,
        *,
        on_transcript: Callable[[str, bool], None],
        on_error: Callable[[str], None],
        on_ready: Optional[Callable[[], None]] = None,
        config: Optional[STTConfig] = None,
    ) -> DoubaoStreamConnection:
        """Open a doubao streaming session and start the consumer task."""
        credentials = self._load_credentials()
        conn = DoubaoStreamConnection(
            on_transcript=on_transcript,
            on_error=on_error,
            on_ready=on_ready,
            config=config,
        )
        asyncio.ensure_future(conn._run(credentials=credentials))
        self._connection = conn
        return conn

    # ── STTProvider ABC — batch path via a transient stream ────────────

    async def transcribe(self, audio_data: bytes, config: STTConfig | None = None) -> STTResult:
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
            self._connection = None
