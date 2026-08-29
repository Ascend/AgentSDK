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

"""P64-E1 Support for tts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


__all__ = [
    "TTSConfig",
    "TTSChunk",
    "TTSSynthesis",
    "TTSProvider",
]


@dataclass
class TTSConfig:
    """Text-to-speech configuration.

    Defaults are tuned for the OpenAI reference path (``tts-1`` / ``alloy`` /
    24 kHz PCM). MiniMax and Gemini override these in their factories.
    """

    model: str = "tts-1"
    voice: str = "alloy"
    speed: float = 1.0  # 0.25 – 4.0 (provider-clamped)
    sample_rate: int = 24000
    encoding: str = "pcm_s16le"  # boundary format; providers decode native → this
    language: str = "en"  # MiniMax language_boost; OpenAI ignores; Gemini infers
    # Provider-specific extras (e.g. MiniMax ``audio_setting``/``voice_setting``
    # sub-objects, Gemini ``style_instructions``) live in ``extra`` so the
    # base config stays provider-agnostic.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSChunk:
    """One decoded PCM frame in a streaming synthesis.

    ``pcm`` is always mono PCM16 little-endian at ``TTSConfig.sample_rate``
    — providers do format conversion at the boundary. ``is_final`` marks
    the last frame so the player can release the device.
    """

    pcm: bytes
    sample_rate: int = 24000
    duration_ms: float = 0.0
    is_final: bool = False


class TTSSynthesis:
    """Handle for one streaming TTS session — symmetric to STT connections.

    Constructed by :meth:`TTSProvider.synthesize_stream`; the caller drives
    it by pushing text (``feed_text``) and eventually either lets the
    provider call ``on_done`` naturally or cancels via :meth:`cancel`.

    The callbacks (``on_audio`` / ``on_error`` / ``on_done``) are set once
    at construction; the provider's background task invokes them. We keep
    them as plain callables (not asyncio.Queue) so a non-async consumer
    (the audio player thread) can receive frames without a loop hop.
    """

    def __init__(
        self,
        *,
        on_audio: Callable[[TTSChunk], None],
        on_error: Callable[[str], None],
        on_done: Callable[[], None],
        config: Optional[TTSConfig] = None,
    ) -> None:
        self._on_audio = on_audio
        self._on_error = on_error
        self._on_done = on_done
        self._config = config or TTSConfig()
        self._cancelled = False
        self._done = False
        # Provider-specific state (e.g. an asyncio.Event the run task
        # awaits between feed_text calls). Providers cast this via the
        # factory; we keep it opaque here.
        self._state: Any = None

    @property
    def config(self) -> TTSConfig:
        return self._config

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def is_done(self) -> bool:
        return self._done

    def _mark_done(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._on_done()
        except Exception:  # nosec B110 - provider callback must not kill the run task
            # Provider's responsibility to log; swallow so a buggy
            # callback doesn't kill the run task.
            pass

    def _emit_audio(self, chunk: TTSChunk) -> None:
        if self._cancelled or self._done:
            return
        try:
            self._on_audio(chunk)
        except Exception:  # nosec B110 - provider callback must not kill the run task
            pass  # Intentional best-effort path; the surrounding fallback remains valid.

    def _emit_error(self, msg: str) -> None:
        if self._cancelled or self._done:
            return
        try:
            self._on_error(msg)
        except Exception:  # nosec B110 - provider callback must not kill the run task
            pass  # Intentional best-effort path; the surrounding fallback remains valid.

    async def feed_text(self, text: str) -> None:
        """Push a text fragment (LLM token / sentence) to the synthesizer.

        Providers may buffer internally until a flush threshold (OpenAI
        ``audio.speech`` is batch, so the OpenAI provider accumulates and
        submits on ``finalize``; MiniMax T2A WebSocket can stream per
        chunk). Subclasses override; this base is a no-op.
        """
        # Default no-op; providers override to actually drive their backend.

    async def cancel(self) -> None:
        """Abort the synthesis and stop the provider's background task."""
        self._cancelled = True

    async def close(self) -> None:
        """Alias for :meth:`cancel` (mirrors STT connection surface)."""
        await self.cancel()


class TTSProvider(ABC):
    """Abstract text-to-speech provider — mirrors :class:`STTProvider`."""

    @abstractmethod
    def synthesize_stream(
        self,
        *,
        on_audio: Callable[[TTSChunk], None],
        on_error: Callable[[str], None],
        on_done: Callable[[], None],
        config: Optional[TTSConfig] = None,
    ) -> TTSSynthesis:
        """Open a streaming synthesis session.

        Returns a :class:`TTSSynthesis` handle immediately; the provider's
        background coroutine pushes PCM frames via ``on_audio`` as they
        arrive from the backend. The caller pushes text via
        :meth:`TTSSynthesis.feed_text`; the provider decides whether to
        stream or batch internally.
        """

    @abstractmethod
    async def synthesize(self, text: str, config: Optional[TTSConfig] = None) -> bytes:
        """Synthesize speech with the configured provider."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider-level resources (HTTP session, etc.)."""
