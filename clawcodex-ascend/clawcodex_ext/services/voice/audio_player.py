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

"""Cross-platform PCM audio player — F-64 P64-E8.

Plays mono PCM16 bytes through the system audio device. Three backends
with graceful fallback (mirrors :mod:`audio_recorder`'s PyAudio→SoX
strategy):

1. **PyAudio** (preferred) — opens a blocking stream and writes frames
   at the device's realtime pace. Lowest latency, but PyAudio install
   is the most fragile across distros.
2. **SoX** ``play`` — pipes PCM via stdin; available on most macOS/Linux
   installs without a Python build toolchain.
3. **ffplay** (fallback / 试听 path) — writes the PCM to a temp WAV and
   plays it once. Highest latency (process spawn + file I/O) but the
   most universally available (bundled with ffmpeg). Used by the
   ``/tts say`` 试听 path when neither PyAudio nor SoX is present.

The ``play_pcm(pcm, sample_rate=...)`` convenience function tries the
backends in order and raises ``RuntimeError`` if none are available —
the ``/tts`` command surfaces this as a clear "install pyaudio / sox /
ffmpeg" message.

Design
------
* :class:`AudioPlayer` — streaming player backed by :class:`AudioOutQueue`
  (provider pushes frames, player drains at device pace). Used by the
  agent-reply path (P64-E9 future).
* :func:`play_pcm` — one-shot batch player for the 试听 command. Writes
  the whole clip to the device; no queueing. Simpler and good enough
  for a few seconds of audio.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

from .audio_out_queue import AudioOutQueue
from .tts import TTSChunk

logger = logging.getLogger(__name__)

__all__ = [
    "AudioPlayer",
    "play_pcm",
    "has_pyaudio",
    "has_sox",
    "has_ffplay",
]


# ── backend probes ────────────────────────────────────────────────────────


def has_pyaudio() -> bool:
    """True if the ``pyaudio`` Python module is importable."""
    try:
        import pyaudio  # type: ignore[import-not-found]  # noqa: F401

        return True
    except Exception:
        return False


def has_sox() -> bool:
    """True if the SoX ``play`` binary is on PATH."""
    return shutil.which("play") is not None


def has_ffplay() -> bool:
    """True if the ``ffplay`` binary (from ffmpeg) is on PATH."""
    return shutil.which("ffplay") is not None


# ── one-shot batch player (试听 path) ─────────────────────────────────────


def _write_wav(path: Path, pcm: bytes, sample_rate: int, channels: int = 1) -> None:
    """Wrap raw PCM16 bytes in a WAV container (SoX/ffplay need a header)."""
    # pylint: disable=no-member
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # PCM16 = 2 bytes/sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    # pylint: enable=no-member


def _play_pyaudio(pcm: bytes, sample_rate: int, channels: int = 1) -> None:
    """Stream PCM16 to the default output device via PyAudio (blocking)."""
    import pyaudio  # type: ignore[import-not-found]

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True,
        )
        try:
            # Write in 1024-frame chunks so large clips don't allocate one
            # giant buffer on the device side.
            frame_bytes = channels * 2 * 1024
            for i in range(0, len(pcm), frame_bytes):
                stream.write(pcm[i : i + frame_bytes])
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        pa.terminate()


def _play_sox(pcm: bytes, sample_rate: int, channels: int = 1) -> None:
    """Pipe PCM16 to SoX ``play`` via stdin (no temp file)."""
    cmd = [
        "play",
        "-t",
        "raw",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        "-b",
        "16",
        "-e",
        "signed",
        "--endian",
        "little",
        "-",
    ]
    proc = subprocess.Popen(  # noqa: S603 — fixed arg list, no shell  # pylint: disable=consider-using-with
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    assert proc.stdin is not None
    try:
        proc.stdin.write(pcm)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait(timeout=60)


def _play_ffplay(pcm: bytes, sample_rate: int, channels: int = 1) -> None:
    """Write PCM to a temp WAV and play once via ffplay (highest latency)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _write_wav(tmp_path, pcm, sample_rate, channels)
        cmd = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            str(tmp_path),
        ]
        subprocess.run(  # noqa: S603 — fixed arg list, no shell
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60
        )
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def play_pcm(pcm: bytes, *, sample_rate: int = 24000, channels: int = 1) -> None:
    """Play a complete PCM16 clip through the first available backend.

    Tries PyAudio → SoX → ffplay in order. Raises ``RuntimeError`` if
    none are available. Used by the ``/tts say`` 试听 path.
    """
    if not pcm:
        return
    if has_pyaudio():
        _play_pyaudio(pcm, sample_rate, channels)
        return
    if has_sox():
        _play_sox(pcm, sample_rate, channels)
        return
    if has_ffplay():
        _play_ffplay(pcm, sample_rate, channels)
        return
    raise RuntimeError(
        "No audio player backend available. Install one of: pyaudio, "
        "sox (provides `play`), or ffmpeg (provides `ffplay`)."
    )


# ── streaming player (agent-reply path) ───────────────────────────────────


class AudioPlayer:
    """Streaming PCM player backed by an :class:`AudioOutQueue`.

    The provider task pushes :class:`TTSChunk` frames into the queue;
    :meth:`run` drains the queue and writes each frame to the device at
    realtime pace (PyAudio blocking write handles the clocking). When the
    queue closes (``is_final`` frame consumed) the player releases the
    device and returns.

    Single-use: one player per synthesis session. Construct, ``run``,
    discard. The controller (future P64-E9) owns the lifecycle.
    """

    def __init__(
        self,
        *,
        queue: Optional[AudioOutQueue] = None,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        self._queue = queue or AudioOutQueue()
        self._sample_rate = sample_rate
        self._channels = channels
        self._task: Optional[asyncio.Task] = None

    @property
    def queue(self) -> AudioOutQueue:
        return self._queue

    async def push(self, chunk: TTSChunk) -> None:
        await self._queue.push(chunk)

    def start(self) -> None:
        """Schedule the drain task on the running loop."""
        if self._task is not None:
            return
        self._task = asyncio.ensure_future(self.run())

    async def run(self) -> None:
        """Drain the queue and play each frame (PyAudio blocking in executor).

        PyAudio's ``stream.write`` blocks the calling thread for the
        frame's realtime duration — we offload it to the default
        executor so the event loop stays responsive. Backend selection
        mirrors :func:`play_pcm`: PyAudio preferred, SoX/ffplay fallback
        not implemented for the streaming path (SoX/ffplay are one-shot
        process spawns; streaming through them would re-spawn per frame).
        If PyAudio is unavailable the streaming player logs an error and
        drains the queue silently (no playback) rather than crashing —
        the agent reply still appears as text.
        """
        if not has_pyaudio():
            logger.error("AudioPlayer streaming path requires PyAudio; draining queue without playback.")
            async for _ in self._queue:
                pass
            return
        import pyaudio  # type: ignore[import-not-found]

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            output=True,
        )
        loop = asyncio.get_event_loop()
        try:
            async for chunk in self._queue:
                if not chunk.pcm:
                    continue
                # Blocking write in a worker thread so the loop stays free.
                await loop.run_in_executor(None, stream.write, chunk.pcm)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:  # nosec B110 - best-effort audio stream release during drain teardown
                pass
            pa.terminate()

    async def stop(self) -> None:
        """Cancel the drain task and release the queue.

        Original behaviour preserved: this is the path used by the
        TTS-driven agent reply flow at the end of a synthesis. Closes
        the queue (so any further ``push`` is a no-op) and cancels the
        drain task.

        For F-65 P65-C full-duplex barge-in we instead want
        :meth:`stop_nowait` — cancel without closing the queue so the
        same player can be reused for the next turn.
        """
        await self._queue.close()
        await self._cancel_task()

    async def stop_nowait(self) -> None:
        """Cancel the drain task immediately, keep the queue alive.

        Used by the F-65 P65-C interrupt path: the user barges in while
        the agent is still speaking. The provider will send more
        :class:`TTSChunk` frames on the next response turn, so we only
        stop *this* one without closing the underlying
        :class:`AudioOutQueue`. Callers that want to discard the in-flight
        frames too should also call :meth:`AudioOutQueue.clear`.
        """
        await self._cancel_task()

    async def stop_and_close(self) -> None:
        """Cancel the drain task, close the queue, and release the device.

        Equivalent to the old combined ``stop()`` for code paths that
        do want full teardown at session end (F-65 dialogue session
        ``close()``). :meth:`stop` keeps its original semantic so the
        F-64 P64-E8 agent-reply integration is unchanged.
        """
        await self._queue.close()
        await self._cancel_task()

    async def _cancel_task(self) -> None:
        """Cancel ``self._task`` if one is in flight. Idempotent."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
