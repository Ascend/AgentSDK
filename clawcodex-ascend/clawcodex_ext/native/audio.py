#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=no-member
"""F-81.2: 麦克风音频捕获模块.

对标 CCB ``audio-capture-napi``，用 ``pyaudio``（首选）或 ``sounddevice``
实现 WAV 录音与实时音频流。两个后端都是可选依赖——缺失时
:meth:`AudioCaptureModule.is_available` 返回 ``False``，
:func:`clawcodex_ext.native.load_or_fallback` 会返回
:class:`_SilentFallback` 兜底实例（不产生音频、返回静音 WAV）。

前置依赖: F-64 Voice Mode.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import AsyncIterator, Optional

from clawcodex_ext.native import NativeModuleRegistry

__all__ = ["AudioCaptureModule", "AudioFallback"]

_logger = logging.getLogger("clawcodex_ext.native.audio")


def _try_import_backend() -> Optional[str]:
    """返回可用后端名：``"pyaudio"`` / ``"sounddevice"`` / ``None``."""
    try:
        import pyaudio  # noqa: F401

        return "pyaudio"
    except ImportError:
        pass
    try:
        import sounddevice  # noqa: F401

        return "sounddevice"
    except ImportError:
        return None


@NativeModuleRegistry.register("audio_capture")
class AudioCaptureModule:
    """PCM16 WAV 录音 + 实时流音频捕获.

    实现 :class:`clawcodex_ext.native.NativeModule` 协议.
    """

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

    # -- 录音 API ---------------------------------------------------------

    async def record(
        self,
        duration_sec: float = 5.0,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> bytes:
        """录制麦克风音频，返回完整 WAV 字节.

        Args:
            duration_sec: 录制时长（秒）.
            sample_rate: 采样率（Hz），默认 16000（适合语音识别）.
            channels: 声道数，默认 1（单声道）.

        Raises:
            NativeModuleError: 后端不可用或录制失败.
        """
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

        # 阻塞式录音 —— 在 async 上下文中调用方应使用 ``asyncio.to_thread``
        # 包裹以避免阻塞事件循环；这里保持同步语义与 pyaudio 路径一致，
        # 因为录音本身就是 I/O 密集且通常不并发。
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
        """实时音频流 —— 持续 yield PCM16 字节块.

        调用方负责在不需要时 ``break`` 退出 ``async for``，本生成器会在
        ``finally`` 中关闭流. 暂未集成 VAD（F-64 子任务），当前输出原始帧.
        """
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
            pass  # sounddevice 无显式 stream 句柄需关闭

    # -- WAV 编码工具 -----------------------------------------------------

    @staticmethod
    def _encode_wav(pcm: bytes, sample_rate: int, channels: int, sampwidth: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    # -- F-81.6 fallback --------------------------------------------------

    @classmethod
    def fallback(cls) -> "AudioFallback":
        """返回静音兜底实例（不依赖任何音频后端）."""
        return AudioFallback()


class AudioFallback:
    """F-81.6 fallback: 音频后端缺失时的兜底实现.

    所有录音/流操作返回静音 PCM，``is_available`` 恒为 ``False``，
    供 :func:`clawcodex_ext.native.load_or_fallback` 在依赖缺失场景使用.
    """

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
        """返回指定时长的静音 WAV 字节."""
        n_samples = int(duration_sec * sample_rate)
        silence = b"\x00\x00" * n_samples * channels
        return AudioCaptureModule._encode_wav(silence, sample_rate, channels, sampwidth=2)

    async def stream(self, sample_rate: int = 16000, channels: int = 1) -> AsyncIterator[bytes]:
        """无限 yield 静音块（调用方应自行 ``break``）."""
        silence_block = b"\x00\x00" * 1024 * channels
        while True:
            yield silence_block
