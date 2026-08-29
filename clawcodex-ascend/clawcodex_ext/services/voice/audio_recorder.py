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

"""Cross-platform audio recorder.

Mirrors TS ``src/hooks/useVoice.ts`` audio-capture backend: prefer
PyAudio (low latency, cross-platform) and fall back to SoX (Sound
eXchange, ubiquitous CLI recorder) when PyAudio isn't installed or the
default input device can't be opened.

Design
------
* :class:`AudioRecorder` — abstract surface: ``start(on_chunk)`` /
  ``stop()``. ``on_chunk`` is a sync callback invoked from the recorder
  thread with each PCM frame (16 kHz mono ``pcm_s16le`` to match the
  STT providers' default :class:`STTConfig`).
* :class:`PyAudioRecorder` — primary backend. Opens a stream on the
  default input device and reads frames in a background thread. Falls
  back to :class:`SoXRecorder` if PyAudio is missing or ``open()``
  raises (no mic permission, device busy, …).
* :class:`SoXRecorder` — fallback. Shells out to ``rec`` (SoX) with
  PCM output on stdout, read in a background thread. SoX is a single
  binary available on Linux/macOS/Homebrew and most CI images; it's the
  lowest-common-denominator recorder when Python audio bindings are
  unavailable.
* :func:`make_recorder` — factory that picks the best available backend
  for the current platform. Tests inject a stub recorder directly.

Thread-safety
-------------
``start`` spawns a daemon thread that calls ``on_chunk`` per frame;
``stop`` signals the thread to exit and joins it. The ``on_chunk``
callback is responsible for thread-safe handoff (the push-to-talk
controller pushes into an :class:`AudioChunkQueue`, which is
thread-safe via ``call_soon_threadsafe``).

Both backends emit the same PCM format (16 kHz, 16-bit, mono, signed
little-endian) so the STT providers don't need to know which recorder
was used.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "AudioRecorder",
    "PyAudioRecorder",
    "SoXRecorder",
    "make_recorder",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHUNK_MS",
]

# 16 kHz mono pcm_s16le — matches STTConfig defaults. Both STT providers
# accept this format directly with no resampling.
DEFAULT_SAMPLE_RATE = 16000
# 20 ms frames → 640 bytes per chunk (16 kHz * 2 bytes * 0.02 s). Small
# enough for low-latency streaming, large enough that per-chunk overhead
# is negligible.
DEFAULT_CHUNK_MS = 20

ChunkCallback = Callable[[bytes], None]


class AudioRecorder(ABC):
    """Abstract audio recorder surface.

    Lifecycle: ``start(callback)`` → callback fires per PCM frame on a
    background thread → ``stop()`` joins the thread. Recorders are
    single-use: call ``start`` once, then ``stop``; to record again,
    construct a fresh instance.
    """

    @abstractmethod
    def start(self, on_chunk: ChunkCallback) -> None:
        """Begin recording; invoke ``on_chunk`` per PCM frame on a bg thread."""

    @abstractmethod
    def stop(self) -> None:
        """Stop recording and join the background thread."""

    @property
    @abstractmethod
    def is_recording(self) -> bool:
        """True while the background capture thread is running."""


class PyAudioRecorder(AudioRecorder):
    """Primary backend — PyAudio direct stream read.

    PyAudio is the standard Python audio binding (portaudio wrapper). We
    open the default input device at 16 kHz mono 16-bit and read
    ``frames_per_buffer`` samples per iteration in a background thread,
    forwarding each buffer to ``on_chunk``.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_ms: int = DEFAULT_CHUNK_MS,
    ) -> None:
        self._sample_rate = sample_rate
        self._frames_per_buffer = int(sample_rate * chunk_ms / 1000)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stream: Optional[object] = None  # pyaudio.Stream
        self._pyaudio: Optional[object] = None  # pyaudio.PyAudio
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(self, on_chunk: ChunkCallback) -> None:
        try:
            import pyaudio  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("PyAudio not installed; pip install pyaudio (or use the SoX fallback)") from exc

        self._pyaudio = pyaudio.PyAudio()
        try:
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=self._frames_per_buffer,
            )
        except Exception as exc:
            # Clean up the PyAudio instance before propagating so we
            # don't leak a portaudio handle. The caller (make_recorder)
            # will fall back to SoX.
            self._pyaudio.terminate()
            self._pyaudio = None
            raise RuntimeError(f"PyAudio failed to open input device: {exc}") from exc

        self._stop_event.clear()
        self._is_recording = True
        self._thread = threading.Thread(target=self._run, args=(on_chunk,), name="voice-pyaudio", daemon=True)
        self._thread.start()

    def _run(self, on_chunk: ChunkCallback) -> None:
        assert self._stream is not None
        try:
            while not self._stop_event.is_set():
                try:
                    data = self._stream.read(self._frames_per_buffer, exception_on_overflow=False)
                except OSError as exc:
                    logger.warning("PyAudio read error: %s", exc)
                    break
                if data:
                    on_chunk(data)
        except Exception:
            logger.exception("PyAudio recorder thread crashed")
        finally:
            self._is_recording = False

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:  # nosec B110 - best-effort audio stream release during stop
                pass  # Intentional best-effort path; the surrounding fallback remains valid.
            self._stream = None
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:  # nosec B110 - best-effort PyAudio release during stop
                pass  # Intentional best-effort path; the surrounding fallback remains valid.
            self._pyaudio = None
        self._is_recording = False


class SoXRecorder(AudioRecorder):
    """Fallback backend — shells out to SoX ``rec`` for PCM on stdout.

    Uses ``rec -q -r <rate> -e signed-integer -b 16 -c 1 -t raw -`` so
    the PCM stream comes out on stdout (no temp files). We read it in a
    background thread and forward fixed-size chunks to ``on_chunk``.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_ms: int = DEFAULT_CHUNK_MS,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_bytes = int(sample_rate * 2 * chunk_ms / 1000)
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(self, on_chunk: ChunkCallback) -> None:
        if shutil.which("rec") is None and shutil.which("sox") is None:
            raise RuntimeError(
                "Neither 'rec' nor 'sox' found on PATH; install SoX (brew install sox / apt install sox)"
            )
        # Prefer ``rec`` (SoX's recording alias); fall back to ``sox -d``
        # (the default audio device) for builds that ship only ``sox``.
        cmd = [
            "rec",
            "-q",
            "-r",
            str(self._sample_rate),
            "-e",
            "signed-integer",
            "-b",
            "16",
            "-c",
            "1",
            "-t",
            "raw",
            "-",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            cmd = [
                "sox",
                "-d",
                "-q",
                "-r",
                str(self._sample_rate),
                "-e",
                "signed-integer",
                "-b",
                "16",
                "-c",
                "1",
                "-t",
                "raw",
                "-",
            ]
            self._proc = subprocess.Popen(  # pylint: disable=consider-using-with
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        self._stop_event.clear()
        self._is_recording = True
        self._thread = threading.Thread(target=self._run, args=(on_chunk,), name="voice-sox", daemon=True)
        self._thread.start()

    def _run(self, on_chunk: ChunkCallback) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while not self._stop_event.is_set():
                chunk = self._proc.stdout.read(self._chunk_bytes)
                if not chunk:
                    break  # EOF — sox exited
                on_chunk(chunk)
        except Exception:
            logger.exception("SoX recorder thread crashed")
        finally:
            self._is_recording = False

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:  # nosec B110 - last-resort kill during recorder stop; nothing left to do
                    pass  # Intentional best-effort path; the surrounding fallback remains valid.
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._is_recording = False


def make_recorder(
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    prefer: str = "pyaudio",
) -> AudioRecorder:
    """Pick the best available recorder backend for this platform.

    Order: PyAudio (preferred) → SoX (fallback). If PyAudio's ``open()``
    fails at ``start`` time (no mic permission / device busy) the caller
    should catch :class:`RuntimeError` and retry with ``SoXRecorder``
    directly — we don't retry inside ``start`` because the failure mode
    is usually persistent within a session.

    Tests bypass this factory and inject a stub :class:`AudioRecorder`
    directly into :class:`PushToTalkController`.
    """
    if prefer == "pyaudio":
        try:
            import pyaudio  # noqa: F401  # type: ignore[import-untyped]
        except ImportError:
            logger.info("PyAudio unavailable, falling back to SoX recorder")
            return SoXRecorder(sample_rate=sample_rate, chunk_ms=chunk_ms)
        return PyAudioRecorder(sample_rate=sample_rate, chunk_ms=chunk_ms)
    return SoXRecorder(sample_rate=sample_rate, chunk_ms=chunk_ms)
