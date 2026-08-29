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

"""MiniMax T2A TTS provider — P64-E3.

Implements :class:`TTSProvider` against MiniMax's ``POST /v1/t2a_v2``
HTTP endpoint (Text-to-Audio v2). MiniMax T2A supports 8 models
(``speech-2.8-hd``/``turbo``, ``2.6-hd``/``turbo``, ``02-hd``/``turbo``,
``01-hd``/``turbo``), 300+ voices, 30+ languages, and streaming output
with ``stream=true`` (hex-encoded PCM frames in a JSON envelope).

Credentials
-----------
Same as P64-D1 STT: ``MINIMAX_API_KEY`` + ``MINIMAX_GROUP_ID`` env, or
``~/.clawcodex/tts/minimax/credentials.json``. The two sub-features
share the credential file.

Streaming
---------
MiniMax T2A HTTP returns a stream of JSON objects each carrying a hex-
encoded audio frame under ``data.audio`` (when ``audio_setting.format=
pcm``) or an mp3 chunk. We decode hex → bytes at the boundary so the
``AudioOutQueue`` only sees PCM. For WebSocket streaming (lower TTFA)
see the P64-D Realtime path; this module sticks to the simpler HTTP
path for the P1 TTS scope.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from .tts import TTSChunk, TTSConfig, TTSProvider, TTSSynthesis

logger = logging.getLogger(__name__)

__all__ = [
    "MiniMaxTTSProvider",
    "MiniMaxTTSCredentialsError",
    "MINIMAX_T2A_ENDPOINTS",
    "MINIMAX_TTS_CREDENTIALS_PATH",
]

# Credentials path shared with P64-D1 STT (same ~/.clawcodex/tts/minimax/).
MINIMAX_TTS_CREDENTIALS_PATH = Path("~/.clawcodex/tts/minimax/credentials.json")

# T2A HTTP endpoints by region. ``uw`` is the low-latency Western US T2A
# endpoint (realtime WS for STT has no uw variant — see minimax_stt.py).
MINIMAX_T2A_ENDPOINTS: dict[str, str] = {
    "global": "https://api.minimax.io/v1/t2a_v2",
    "cn": "https://api.minimaxi.chat/v1/t2a_v2",
    "uw": "https://api-uw.minimax.io/v1/t2a_v2",
}

# Default model/voice for TTS — MiniMax speech-2.8-turbo (low-latency).
# Voice IDs follow MiniMax official system voice naming:
# - Chinese (Mandarin)_Warm_Girl: neutral warm female (default)
# - English_expressive_narrator: English expressive narrator
# Full list at https://platform.minimax.io/docs/faq/system-voice-id
# Override via TTSConfig.model / TTSConfig.voice.
_MINIMAX_TTS_DEFAULT_MODEL = "speech-2.8-turbo"
_MINIMAX_TTS_DEFAULT_VOICE = "Chinese (Mandarin)_Warm_Girl"

# Supported models (per official MiniMax API docs 2026-07):
# speech-2.8-hd, speech-2.8-turbo (current gen, recommended)
# speech-2.6-hd, speech-2.6-turbo (legacy, lower latency)
# speech-02-hd, speech-02-turbo, speech-01-hd, speech-01-turbo (legacy)
MINIMAX_SUPPORTED_MODELS: tuple[str, ...] = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
)

# MiniMax official system voice IDs (selected subset for common use cases).
# Full 332+ voice list at https://platform.minimax.io/docs/faq/system-voice-id
MINIMAX_SYSTEM_VOICES: dict[str, tuple[str, ...]] = {
    "Chinese (Mandarin)": (
        "Chinese (Mandarin)_Warm_Girl",
        "Chinese (Mandarin)_Gentleman",
        "Chinese (Mandarin)_News_Anchor",
        "Chinese (Mandarin)_Sweet_Lady",
        "Chinese (Mandarin)_Crisp_Girl",
        "Chinese (Mandarin)_Reliable_Executive",
        "Chinese (Mandarin)_Male_Announcer",
        "Chinese (Mandarin)_Cute_Spirit",
    ),
    "English": (
        "English_expressive_narrator",
        "English_radiant_girl",
        "English_magnetic_voiced_man",
        "English_captivating_female1",
        "English_Graceful_Lady",
        "English_CalmWoman",
        "English_Persuasive_Man",
        "English_FriendlyPerson",
    ),
    "Japanese": (
        "Japanese_IntellectualSenior",
        "Japanese_GentleButler",
        "Japanese_KindLady",
        "Japanese_CalmLady",
    ),
    "Korean": (
        "Korean_CharmingSister",
        "Korean_GentleWoman",
        "Korean_ReliableYouth",
    ),
    "Cantonese": (
        "Cantonese_ProfessionalHost (F)",
        "Cantonese_GentleLady",
        "Cantonese_CuteGirl",
    ),
}


class MiniMaxTTSCredentialsError(RuntimeError):
    """Raised when MiniMax API key / group_id are missing."""


class _MiniMaxSynthesis(TTSSynthesis):
    """Buffers text for batch POST (MiniMax T2A HTTP is server-streaming)."""

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


class MiniMaxTTSProvider(TTSProvider):
    """P1 P0 P64-D1 Provide text-to-speech through MiniMax."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        group_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        credentials_path: Optional[Path] = None,
    ) -> None:
        self._explicit_api_key = api_key
        self._explicit_group_id = group_id
        self._explicit_endpoint = endpoint
        self._credentials_path = credentials_path or MINIMAX_TTS_CREDENTIALS_PATH.expanduser()

    # ── credentials + endpoint resolution (shared logic with STT) ───────

    def _load_credentials_file(self) -> dict[str, Any]:
        path = self._credentials_path
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise MiniMaxTTSCredentialsError(f"MiniMax credentials file {path} is unreadable: {exc}") from exc
        if not isinstance(data, dict):
            raise MiniMaxTTSCredentialsError(f"MiniMax credentials file {path} must be a JSON object")
        return data

    def _resolve(self) -> tuple[str, str, str]:
        """Return (api_key, group_id, endpoint). Raises if no api_key."""
        file_data = self._load_credentials_file()
        api_key = self._explicit_api_key or os.environ.get("MINIMAX_API_KEY") or file_data.get("api_key")
        group_id = self._explicit_group_id or os.environ.get("MINIMAX_GROUP_ID") or file_data.get("group_id") or ""
        if not api_key:
            raise MiniMaxTTSCredentialsError(
                f"MINIMAX_API_KEY not set. Configure the env var or {self._credentials_path}."
            )
        if self._explicit_endpoint:
            endpoint = self._explicit_endpoint
        else:
            region = (os.environ.get("MINIMAX_REGION") or file_data.get("endpoint_region") or "global").strip().lower()
            endpoint = MINIMAX_T2A_ENDPOINTS.get(region) or MINIMAX_T2A_ENDPOINTS["global"]
        return api_key, group_id, endpoint

    # ── streaming path ──────────────────────────────────────────────────

    def synthesize_stream(
        self,
        *,
        on_audio,
        on_error,
        on_done,
        config: Optional[TTSConfig] = None,
    ) -> TTSSynthesis:
        cfg = config or TTSConfig(model=_MINIMAX_TTS_DEFAULT_MODEL, voice=_MINIMAX_TTS_DEFAULT_VOICE)
        syn = _MiniMaxSynthesis(on_audio=on_audio, on_error=on_error, on_done=on_done, config=cfg)
        try:
            api_key, group_id, endpoint = self._resolve()
        except MiniMaxTTSCredentialsError as exc:
            syn._emit_error(str(exc))
            syn._mark_done()
            return syn
        asyncio.ensure_future(self._run(syn, api_key, group_id, endpoint))
        return syn

    async def _run(self, syn: _MiniMaxSynthesis, api_key: str, group_id: str, endpoint: str) -> None:
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
            pcm = await self._post_t2a(text, syn.config, api_key, group_id, endpoint)
        except Exception as exc:
            syn._emit_error(f"MiniMax T2A request failed: {exc}")
            syn._mark_done()
            return
        if syn.is_cancelled:
            return
        syn._emit_audio(TTSChunk(pcm=pcm, sample_rate=syn.config.sample_rate, is_final=True))
        syn._mark_done()

    async def _post_t2a(self, text: str, cfg: TTSConfig, api_key: str, group_id: str, endpoint: str) -> bytes:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._post_t2a_sync, text, cfg, api_key, group_id, endpoint)

    def _post_t2a_sync(self, text: str, cfg: TTSConfig, api_key: str, group_id: str, endpoint: str) -> bytes:
        """POST /v1/t2a_v2 with stream=true → concatenated PCM bytes.

        MiniMax returns a stream of JSON objects (one per audio frame)
        when ``stream=true``. Each object's ``data.audio`` is a hex string
        of PCM bytes (when ``audio_setting.format=pcm``). We parse the
        stream line-by-line, decode hex, and concatenate. Non-streaming
        responses return a single JSON object with ``data.audio`` as the
        full hex payload — handled by the same parser.
        """
        import urllib.request

        body = json.dumps(
            {
                "model": cfg.model,
                "text": text,
                "stream": True,
                "voice_setting": {
                    "voice_id": cfg.voice,
                    "speed": cfg.speed,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {
                    "sample_rate": cfg.sample_rate,
                    "format": "pcm",
                    "channel": 1,
                },
                "pron_dict": {"tone": ["calm"]},
                "timber_weights": [],
            }
        ).encode("utf-8")
        url = endpoint
        if group_id:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}GroupId={group_id}"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        pcm = bytearray()
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed MiniMax URL
            for raw_line in resp:
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # MiniMax stream frame: {"data":{"audio":"<hex>"},"is_final":...}
                data = obj.get("data") or {}
                hex_audio = data.get("audio") if isinstance(data, dict) else None
                if hex_audio:
                    try:
                        pcm.extend(bytes.fromhex(hex_audio))
                    except ValueError:
                        logger.debug("MiniMax T2A non-hex audio frame skipped")
                extra_audio = obj.get("extra_info") or {}
                if isinstance(extra_audio, dict) and extra_audio.get("is_final"):
                    break
        return bytes(pcm)

    # ── batch path ──────────────────────────────────────────────────────

    async def synthesize(self, text: str, config: Optional[TTSConfig] = None) -> bytes:
        api_key, group_id, endpoint = self._resolve()
        cfg = config or TTSConfig(model=_MINIMAX_TTS_DEFAULT_MODEL, voice=_MINIMAX_TTS_DEFAULT_VOICE)
        return await self._post_t2a(text, cfg, api_key, group_id, endpoint)

    async def close(self) -> None:
        return None
