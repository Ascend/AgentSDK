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

"""OpenAI TTS provider — F-64 P64-E2.

Reference implementation of :class:`TTSProvider` against OpenAI's
``POST /v1/audio/speech`` endpoint. The endpoint is *server-streaming*
(audio frames flow back as the model synthesizes) but *not token-level
input-streaming* — the client must submit the full text in the request
body. We reconcile this by buffering ``feed_text`` calls and submitting
once the synthesis is finalized (or a flush threshold is hit).

Auth
----
``OPENAI_API_KEY`` env var (or explicit ``api_key`` ctor arg). Raises
:class:`OpenAITTSCredentialsError` if absent — the ``/tts`` command maps
this to a friendly "set OPENAI_API_KEY" message.

Dependencies
------------
Uses :mod:`urllib.request` + :mod:`http.client` from the stdlib for the
batch path (no third-party dep) so the reference implementation works
even when ``openai`` SDK / ``httpx`` aren't installed. The streaming
path uses ``aiohttp`` if available, falling back to a thread-pool
wrapper around the stdlib batch POST (which still streams the response
body — we just read it in chunks from a worker thread).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from .tts import TTSChunk, TTSConfig, TTSProvider, TTSSynthesis

logger = logging.getLogger(__name__)

__all__ = [
    "OpenAITTSProvider",
    "OpenAITTSCredentialsError",
    "OPENAI_TTS_ENDPOINT",
]

OPENAI_TTS_ENDPOINT = "https://api.openai.com/v1/audio/speech"


class OpenAITTSCredentialsError(RuntimeError):
    """Raised when ``OPENAI_API_KEY`` is not configured."""


class _OpenAISynthesis(TTSSynthesis):
    """Extends :class:`TTSSynthesis` with a text buffer for batch submit."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._buffer: list[str] = []
        self._flush_event = asyncio.Event()
        self._final_text = ""

    async def feed_text(self, text: str) -> None:
        if self.is_cancelled or self.is_done:
            return
        self._buffer.append(text)
        # OpenAI is batch — we don't flush per-token; the run task waits
        # for finalize (close of feed) before POSTing.
        self._flush_event.set()

    async def finalize(self) -> str:
        """Signal no more text — triggers the POST. Returns accumulated text."""
        if self.is_cancelled:
            return ""
        self._final_text = "".join(self._buffer)
        self._buffer.clear()
        self._flush_event.set()
        return self._final_text

    @property
    def pending_text(self) -> str:
        return "".join(self._buffer)


class OpenAITTSProvider(TTSProvider):
    """OpenAI ``audio.speech`` TTS — reference implementation (P0).

    Streaming semantics: OpenAI accepts the full text up-front and
    streams audio frames back. We accumulate ``feed_text`` calls and
    POST once the caller finalizes (``TTSSynthesis.close`` /
    ``finalize``). For true token-level streaming use MiniMax Realtime
    or Gemini Live (P1 providers).
    """

    def __init__(self, *, api_key: Optional[str] = None, endpoint: str = OPENAI_TTS_ENDPOINT) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._endpoint = endpoint

    def _require_key(self) -> str:
        if not self._api_key:
            raise OpenAITTSCredentialsError(
                "OPENAI_API_KEY not set. Configure the env var to use the OpenAI TTS backend."
            )
        return self._api_key

    # ── streaming path ──────────────────────────────────────────────────

    def synthesize_stream(
        self,
        *,
        on_audio,
        on_error,
        on_done,
        config: Optional[TTSConfig] = None,
    ) -> TTSSynthesis:
        cfg = config or TTSConfig()
        syn = _OpenAISynthesis(on_audio=on_audio, on_error=on_error, on_done=on_done, config=cfg)
        try:
            api_key = self._require_key()
        except OpenAITTSCredentialsError as exc:
            syn._emit_error(str(exc))
            syn._mark_done()
            return syn
        asyncio.ensure_future(self._run(syn, api_key))
        return syn

    async def _run(self, syn: _OpenAISynthesis, api_key: str) -> None:
        """Wait for the caller to finalize text, POST, stream frames back."""
        # Wait until finalize() sets the flush event with a non-empty payload.
        while True:
            if syn.is_cancelled:
                return
            await syn._flush_event.wait()
            syn._flush_event.clear()
            text = syn._final_text
            if text:
                break
            # Spurious wake (feed_text then immediate clear) — keep waiting.
        if syn.is_cancelled:
            return
        try:
            pcm = await self._post_speech(text, syn.config, api_key)
        except Exception as exc:
            syn._emit_error(f"OpenAI TTS request failed: {exc}")
            syn._mark_done()
            return
        if syn.is_cancelled:
            return
        # Emit as a single chunk (OpenAI batch → we get the whole body).
        # A finer-grained chunking would require parsing the SSE stream;
        # for the P0 reference the single-chunk path is enough — the
        # AudioOutQueue buffers and plays incrementally anyway.
        syn._emit_audio(TTSChunk(pcm=pcm, sample_rate=syn.config.sample_rate, is_final=True))
        syn._mark_done()

    async def _post_speech(self, text: str, cfg: TTSConfig, api_key: str) -> bytes:
        """POST /v1/audio/speech → raw PCM bytes.

        Runs the blocking HTTP request in a worker thread (stdlib
        ``urllib``) so the event loop isn't held. The endpoint returns
        the full audio body; we read it in one shot. For incremental
        playback, the AudioOutQueue can chunk it client-side.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._post_speech_sync, text, cfg, api_key)

    def _post_speech_sync(self, text: str, cfg: TTSConfig, api_key: str) -> bytes:
        import urllib.request

        body = json.dumps(
            {
                "model": cfg.model,
                "input": text,
                "voice": cfg.voice,
                "response_format": "pcm",
                "speed": cfg.speed,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed OpenAI URL
            return resp.read()

    # ── batch path ──────────────────────────────────────────────────────

    async def synthesize(self, text: str, config: Optional[TTSConfig] = None) -> bytes:
        api_key = self._require_key()
        cfg = config or TTSConfig()
        return await self._post_speech(text, cfg, api_key)

    async def close(self) -> None:
        return None
