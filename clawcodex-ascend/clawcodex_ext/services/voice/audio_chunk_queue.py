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

"""Async audio chunk queue — P64-C.

Mirrors TS ``src/services/doubaoSTT.ts`` ``AudioChunkQueue``: a push-style
async queue that bridges a *push* audio producer (the recorder thread) to a
*pull* async consumer (the STT streaming connection's ``AsyncIterable``
contract).

Why this exists
---------------
The Anthropic and doubao STT backends both expose an
``AsyncIterable[bytes]`` audio source: ``async for chunk in source``. But
audio capture (PyAudio / SoX) runs in a synchronous thread that *pushes*
frames as they arrive. ``AudioChunkQueue`` reconciles the two directions
without buffering the whole recording in memory:

* The producer thread calls :meth:`push` for each PCM frame.
* The consumer task iterates the queue with ``async for``; each iteration
  awaits until a frame is available.
* The producer signals end-of-stream with :meth:`push`(``None``); the
  consumer's ``async for`` then terminates cleanly.

The implementation uses a single-slot "waiting future" rather than
``asyncio.Queue`` to keep latency at one frame (no queue depth buildup)
and to keep the surface tiny — this is a hot audio path and we don't want
backpressure surprises. If the consumer is slow, frames are dropped with a
debug log; for 16 kHz mono PCM that's a 32 KB/s stream and a single
awaitable slot is more than enough headroom.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["AudioChunkQueue"]


class AudioChunkQueue:
    """Push-style async queue implementing ``AsyncIterable[bytes]``.

    Single-slot design: at most one waiting consumer and one buffered
    chunk. End-of-stream is signalled by pushing ``None``; the consumer's
    ``async for`` then stops iteration after draining the buffer.

    Thread-safety: :meth:`push` is safe to call from a non-asyncio thread
    (the audio recorder thread) — it uses :meth:`loop.call_soon_threadsafe`
    to schedule the wake-up. The consumer side must run on the same loop.
    """

    def __init__(self, *, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        # Bind the loop lazily on first push/await so a queue constructed
        # outside a running loop (e.g. at module import) still binds to the
        # right loop when audio starts flowing. Passing ``loop=`` explicitly
        # is supported for tests that want to inject a loop.
        self._loop: Optional[asyncio.AbstractEventLoop] = loop
        self._chunks: list[Optional[bytes]] = []
        self._waiting: Optional[asyncio.Future] = None
        self._closed = False

    # ── producer (sync thread) ──────────────────────────────────────────

    def push(self, chunk: Optional[bytes]) -> None:
        """Push an audio frame from the recorder thread.

        ``None`` signals end-of-stream: the consumer's ``async for``
        terminates after draining any buffered frame. Pushing after
        ``None`` is a no-op (the stream is closed).
        """
        if self._closed:
            return
        if chunk is None:
            self._closed = True
        else:
            self._chunks.append(chunk)

        # Wake the consumer if it's blocked on __anext__.
        if self._waiting is not None and not self._waiting.done():
            fut = self._waiting
            self._waiting = None
            try:
                loop = self._loop or asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is None or loop.is_closed():
                # No loop bound yet / loop closed — resolve inline on the
                # current thread (only safe because the future hasn't been
                # awaited yet; if it had, the consumer would already have
                # a loop). This path is rare; the recorder thread usually
                # starts after the consumer is iterating.
                if not fut.cancelled():
                    fut.set_result(None)
                return
            # Detect whether we're being called from the consumer's loop
            # (safe to resolve inline) or a foreign thread (must schedule).
            try:
                running = asyncio.get_running_loop()
                same_loop = running is loop
            except RuntimeError:
                same_loop = False
            if same_loop:
                fut.set_result(None)
            else:
                loop.call_soon_threadsafe(fut.set_result, None)

    # ── consumer (async) ────────────────────────────────────────────────

    def __aiter__(self) -> "AudioChunkQueue":
        return self

    async def __anext__(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        if self._closed:
            raise StopAsyncIteration
        # Bind the loop on first await so we always capture the loop the
        # consumer is running on, even if the queue was constructed early.
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        # Block until the producer pushes a frame or closes.
        self._waiting = self._loop.create_future()
        await self._waiting
        if self._chunks:
            return self._chunks.pop(0)
        # Woken but nothing buffered and closed → end-of-stream.
        if self._closed:
            raise StopAsyncIteration
        # Spurious wake (shouldn't happen with a single-slot design) →
        # recurse to re-block.
        return await self.__anext__()
