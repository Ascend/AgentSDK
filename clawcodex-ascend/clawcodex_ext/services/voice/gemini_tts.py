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

"""Gemini TTS provider — P64-E4.

Implements :class:`TTSProvider` against Google's Gemini ``generate_content``
API with output modality AUDIO (PCM16 24 kHz mono). Gemini is unique
among the P0/P1 providers in that TTS is a *model capability* rather
than a dedicated endpoint: the same ``generate_content`` call that
returns text can be asked to return audio by setting
``output_modality=AUDIO`` and a voice name.

Credentials
-----------
Google API key (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` env) or ADC
(Application Default Credentials via ``google-genai`` SDK). We prefer
the API-key path for parity with the OpenAI/MiniMax providers; ADC is
supported when the SDK picks it up automatically.

Dependencies
------------
The ``google-genai`` SDK is a *heavy* optional dep — the factory in
:mod:`provider_registry` lazy-imports it so REPL boot / Stage 6 perf
is unaffected. ``get_tts_provider("gemini")`` raises ``ImportError``
with a clear ``pip install google-genai`` hint when the SDK is absent.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from .tts import TTSChunk, TTSConfig, TTSProvider, TTSSynthesis

logger = logging.getLogger(__name__)

__all__ = [
    "GeminiTTSProvider",
    "GeminiTTSCredentialsError",
    "GEMINI_TTS_DEFAULT_MODEL",
    "GEMINI_TTS_DEFAULT_VOICE",
]

# Gemini 2.5 Flash Preview TTS is the publicly-available TTS model as of
# 2026-07. The user can override via TTSConfig.model.
GEMINI_TTS_DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_DEFAULT_VOICE = "Leda"


class GeminiTTSCredentialsError(RuntimeError):
    """Raised when no Google API key / ADC is available."""


class _GeminiSynthesis(TTSSynthesis):
    """Buffers text — Gemini generate_content is a single batch call."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._buffer: list[str] = []
        self._flush_event = asyncio.Event()
        self._final_text = ""

    async def feed_text(self, text: str) -> None:
        if self.is_cancelled or self.is_done:
            return
        self._buffer.append(text)
        self._flush_event.set()

    async def finalize(self) -> str:
        if self.is_cancelled:
            return ""
        self._final_text = "".join(self._buffer)
        self._buffer.clear()
        self._flush_event.set()
        return self._final_text


class GeminiTTSProvider(TTSProvider):
    """Gemini ``generate_content`` TTS — P1 provider.

    Gemini TTS is batch (one ``generate_content`` call per synthesis).
    We accumulate ``feed_text`` and submit on finalize, like the OpenAI
    and MiniMax HTTP providers. The response carries inline PCM16 data
    which we extract and emit as a single :class:`TTSChunk`.
    """

    def __init__(self, *, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def _require_key(self) -> str:
        if not self._api_key:
            raise GeminiTTSCredentialsError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. Configure the env var "
                "or run `gcloud auth application-default login` for ADC."
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
        cfg = config or TTSConfig(model=GEMINI_TTS_DEFAULT_MODEL, voice=GEMINI_TTS_DEFAULT_VOICE, sample_rate=24000)
        syn = _GeminiSynthesis(on_audio=on_audio, on_error=on_error, on_done=on_done, config=cfg)
        try:
            api_key = self._require_key()
        except GeminiTTSCredentialsError as exc:
            syn._emit_error(str(exc))
            syn._mark_done()
            return syn
        asyncio.ensure_future(self._run(syn, api_key))
        return syn

    async def _run(self, syn: _GeminiSynthesis, api_key: str) -> None:
        while True:
            if syn.is_cancelled:
                return
            await syn._flush_event.wait()
            syn._flush_event.clear()
            text = syn._final_text
            if text:
                break
        if syn.is_cancelled:
            return
        try:
            pcm = await self._generate(text, syn.config, api_key)
        except ImportError as exc:
            syn._emit_error(f"Gemini TTS needs the 'google-genai' package: pip install google-genai ({exc})")
            syn._mark_done()
            return
        except Exception as exc:
            syn._emit_error(f"Gemini TTS request failed: {exc}")
            syn._mark_done()
            return
        if syn.is_cancelled:
            return
        syn._emit_audio(TTSChunk(pcm=pcm, sample_rate=syn.config.sample_rate, is_final=True))
        syn._mark_done()

    async def _generate(self, text: str, cfg: TTSConfig, api_key: str) -> bytes:
        """Call ``client.models.generate_content`` with audio output modality.

        Runs in a worker thread because the ``google-genai`` SDK call is
        synchronous (it returns a single response object, not an async
        iterator, for the TTS modality). The response carries
        ``candidates[0].content.parts[*].inline_data.data`` with MIME
        ``audio/L16;rate=24000`` — we extract and return the raw bytes.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, text, cfg, api_key)

    def _generate_sync(self, text: str, cfg: TTSConfig, api_key: str) -> bytes:
        from google import genai  # type: ignore[import-not-found]
        from google.genai import types  # type: ignore[import-not-found]

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=cfg.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfigVoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=cfg.voice,
                    )
                ),
            ),
        )
        # Extract inline PCM data from the response parts.
        pcm = bytearray()
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is None:
                    continue
                data = getattr(inline, "data", None)
                if not data:
                    continue
                # ``data`` is ``bytes`` for audio parts.
                if isinstance(data, (bytes, bytearray)):
                    pcm.extend(data)
                elif isinstance(data, str):
                    # Some SDK versions return base64-str; decode defensively.
                    import base64

                    try:
                        pcm.extend(base64.b64decode(data))
                    except Exception:
                        logger.debug("Gemini TTS part data not base64-decodable, skipped")
        return bytes(pcm)

    # ── batch path ──────────────────────────────────────────────────────

    async def synthesize(self, text: str, config: Optional[TTSConfig] = None) -> bytes:
        api_key = self._require_key()
        cfg = config or TTSConfig(model=GEMINI_TTS_DEFAULT_MODEL, voice=GEMINI_TTS_DEFAULT_VOICE, sample_rate=24000)
        return await self._generate(text, cfg, api_key)

    async def close(self) -> None:
        return None
