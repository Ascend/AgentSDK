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

"""Async audio output queue — F-64 P64-E8.

The mirror of :class:`AudioChunkQueue` (P64-C) for the playback direction.
A TTS provider's background task *pushes* decoded PCM frames as they
arrive from the backend; a player thread *pulls* frames at the device's
realtime pace. The queue bridges the two timing domains without buffering
the whole clip in memory or stalling the provider on a slow device.

Why a separate class from ``AudioChunkQueue``
---------------------------------------------
The STT queue's single-slot design assumes the consumer (the WebSocket
pump) drains frames faster than the recorder produces them — true for
16 kHz mono PCM over a fast network. TTS playback is *clocked* by the
audio device: the player thread blocks on ``pyaudio.write`` for the
frame's real duration (e.g. 20ms per 320-sample frame at 16 kHz), so
the provider can easily out-produce the device. We therefore keep a
small bounded buffer (default 50 frames ≈ 1s of audio at 20ms/frame)
and drop the oldest frame on overflow rather than blocking the provider
task — a dropped TTS frame is a brief glitch, far better than stalling
the agent-reply pipeline.

End-of-stream is signalled by pushing a :class:`TTSChunk` with
``is_final=True``; the player drains the buffer then releases the device.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .tts import TTSChunk

logger = logging.getLogger(__name__)

__all__ = ["AudioOutQueue"]

# Default bounded buffer depth. At 20ms/frame (a common PCM chunk size)
# this is ~1s of audio — enough to absorb provider jitter without
# introducing noticeable latency, and small enough that a stalled player
# frees memory quickly.
_DEFAULT_MAX_FRAMES = 50


class AudioOutQueue:
    """Bounded async queue of :class:`TTSChunk` frames for playback.

    * :meth:`push` — provider task (async) enqueues a decoded PCM frame.
      On overflow the *oldest* frame is dropped (log debug) so the
      provider never blocks on a slow device.
    * :meth:`push_nowait` — sync enqueue for non-async providers.
    * ``async for chunk in queue`` — player thread's async drain loop;
      terminates after the ``is_final`` frame is consumed.
    * :meth:`close` — release the queue (drop buffered frames, wake the
      consumer with a final sentinel).

    Thread-safety: the queue uses an :class:`asyncio.Queue` under the
    hood, so ``push`` must run on the same loop as the consumer. The
    player thread should be an asyncio task (or use ``run_in_executor``
    with a small sync→async hop for the PyAudio blocking write).
    """

    def __init__(self, *, max_frames: int = _DEFAULT_MAX_FRAMES) -> None:
        self._q: asyncio.Queue[Optional[TTSChunk]] = asyncio.Queue(maxsize=max_frames)
        self._closed = False

    async def push(self, chunk: TTSChunk) -> None:
        """Enqueue a PCM frame from the provider task.

        If the buffer is full the oldest frame is dropped (FIFO overflow)
        so the provider isn't blocked on a slow consumer. This trades a
        brief audio glitch for pipeline liveness — the right tradeoff for
        streaming TTS where the next frame is always imminent.
        """
        if self._closed:
            return
        try:
            self._q.put_nowait(chunk)
        except asyncio.QueueFull:
            # Drop the oldest frame to make room.
            try:
                dropped = self._q.get_nowait()
                logger.debug(
                    "AudioOutQueue overflow — dropped oldest frame (%d bytes)",
                    len(dropped.pcm) if dropped else 0,
                )
            except asyncio.QueueEmpty:
                pass  # raced — shouldn't happen given QueueFull, but be safe
            try:
                self._q.put_nowait(chunk)
            except asyncio.QueueFull:
                logger.debug("AudioOutQueue still full after drop — frame skipped")

    def push_nowait(self, chunk: TTSChunk) -> None:
        """Sync enqueue — for providers that aren't async on the push side.

        Same overflow policy as :meth:`push`. Must be called from the
        queue's loop thread (or via ``loop.call_soon_threadsafe``).
        """
        if self._closed:
            return
        try:
            self._q.put_nowait(chunk)
        except asyncio.QueueFull:
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._q.put_nowait(chunk)
            except asyncio.QueueFull:
                logger.debug("AudioOutQueue still full after drop — frame skipped")

    def clear(self) -> int:
        """Drop every buffered frame without changing the closed state.

        Used by the full-duplex dialogue interrupt path (F-65 P65-C): the
        user barges in mid-playback, we cancel the response on the
        server side AND discard whatever PCM frames haven't been written
        to the device yet — otherwise the speaker would play a 200ms
        tail of the interrupted reply and undercut the barge-in.

        Returns the number of frames dropped (for logging / metrics).
        Idempotent: clearing an empty or already-closed queue is safe
        and returns 0 — we deliberately don't reopen a closed queue
        because the producer (TTS task) is dead at that point.
        """
        dropped = 0
        while True:
            try:
                self._q.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        if dropped:
            logger.debug("AudioOutQueue.clear() dropped %d frames", dropped)
        return dropped

    async def close(self) -> None:
        """Signal end-of-stream and wake the consumer.

        Pushes ``None`` as a sentinel; the consumer's ``async for``
        terminates after draining remaining frames.
        """
        if self._closed:
            return
        self._closed = True
        await self._q.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self) -> TTSChunk:
        item = await self._q.get()
        if item is None:
            raise StopAsyncIteration
        return item
