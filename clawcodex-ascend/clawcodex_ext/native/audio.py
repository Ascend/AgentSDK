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

# pylint: disable=no-member
"""Microphone audio capture support."""

from __future__ import annotations

import io
import logging
import wave
from typing import AsyncIterator, Optional

from clawcodex_ext.native import NativeModuleRegistry

__all__ = ["AudioCaptureModule", "AudioFallback"]

_logger = logging.getLogger("clawcodex_ext.native.audio")


def _try_import_backend() -> Optional[str]:
    """Return the first available audio capture backend."""
    try:
        import pyaudio  # noqa: F401

        return "pyaudio"
    except ImportError:
        pass  # Optional integration is unavailable; keep the fallback.
    try:
        import sounddevice  # noqa: F401

        return "sounddevice"
    except ImportError:
        return None


@NativeModuleRegistry.register("audio_capture")
class AudioCaptureModule:
    """Capture microphone audio as PCM16 WAV data or streams."""

    name = "audio_capture"

    def __init__(self) -> None:
        self._backend = _try_import_backend()

    # -- NativeModule protocol --------------------------------------------

    def is_available(self) -> bool:
        return self._backend is not None

    def get_version(self) -> str:
        if self._backend == "pyaudio":
            try:
                import pyaudio

                return getattr(pyaudio, "__version__", "pyaudio-unknown")
            except ImportError:
                return "unavailable"
        if self._backend == "sounddevice":
            try:
                import sounddevice

                return getattr(sounddevice, "__version__", "sounddevice-unknown")
            except ImportError:
                return "unavailable"
        return "unavailable"

    # -- Recording API ----------------------------------------------------

    async def record(
        self,
        duration_sec: float = 5.0,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> bytes:
        """Record microphone audio and return WAV bytes."""
        if self._backend is None:
            from clawcodex_ext.native import NativeModuleError

            raise NativeModuleError("audio backend unavailable (install pyaudio or sounddevice)")
        if self._backend == "pyaudio":
            return await self._record_pyaudio(duration_sec, sample_rate, channels)
        return await self._record_sounddevice(duration_sec, sample_rate, channels)

    async def _record_pyaudio(self, duration_sec: float, sample_rate: int, channels: int) -> bytes:
        import pyaudio

        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024,
            )
            try:
                frames_per_buffer = 1024
                total_frames = int(sample_rate / frames_per_buffer * duration_sec)
                chunks = [stream.read(frames_per_buffer, exception_on_overflow=False) for _ in range(total_frames)]
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            p.terminate()
        return self._encode_wav(b"".join(chunks), sample_rate, channels, sampwidth=2)

    async def _record_sounddevice(self, duration_sec: float, sample_rate: int, channels: int) -> bytes:
        import numpy as np
        import sounddevice as sd

        # Callers in async contexts must wrap this blocking recording path
        # with ``asyncio.to_thread`` to avoid blocking the event loop.
        data = sd.rec(
            int(duration_sec * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
        )
        sd.wait()
        return self._encode_wav(
            np.ascontiguousarray(data).tobytes(),
            sample_rate,
            channels,
            sampwidth=2,
        )

    async def stream(self, sample_rate: int = 16000, channels: int = 1) -> AsyncIterator[bytes]:
        """Yield PCM16 audio frames."""
        if self._backend is None:
            from clawcodex_ext.native import NativeModuleError

            raise NativeModuleError("audio backend unavailable (install pyaudio or sounddevice)")
        if self._backend == "pyaudio":
            async for chunk in self._stream_pyaudio(sample_rate, channels):
                yield chunk
        else:
            async for chunk in self._stream_sounddevice(sample_rate, channels):
                yield chunk

    async def _stream_pyaudio(self, sample_rate: int, channels: int) -> AsyncIterator[bytes]:
        import pyaudio

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024,
        )
        try:
            while True:
                yield stream.read(1024, exception_on_overflow=False)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def _stream_sounddevice(self, sample_rate: int, channels: int) -> AsyncIterator[bytes]:
        import sounddevice as sd

        blocksize = 1024
        try:
            while True:
                block = sd.rec(
                    blocksize,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="int16",
                )
                sd.wait()
                yield block.tobytes()
        finally:
            pass  # sounddevice exposes no stream handle to close here.

    # -- WAV encoding helpers ---------------------------------------------

    @staticmethod
    def _encode_wav(pcm: bytes, sample_rate: int, channels: int, sampwidth: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    # -- fallback --------------------------------------------------

    @classmethod
    def fallback(cls) -> "AudioFallback":
        """Return a pure-Python fallback implementation."""
        return AudioFallback()


class AudioFallback:
    """Provide silent audio when no capture backend is available."""

    name = "audio_capture"

    def is_available(self) -> bool:
        return False

    def get_version(self) -> str:
        return "fallback-silent"

    async def record(
        self,
        duration_sec: float = 5.0,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> bytes:
        """Record microphone audio and return WAV bytes."""
        n_samples = int(duration_sec * sample_rate)
        silence = b"\x00\x00" * n_samples * channels
        return AudioCaptureModule._encode_wav(silence, sample_rate, channels, sampwidth=2)

    async def stream(self, sample_rate: int = 16000, channels: int = 1) -> AsyncIterator[bytes]:
        """Yield PCM16 audio frames."""
        silence_block = b"\x00\x00" * 1024 * channels
        while True:
            yield silence_block
