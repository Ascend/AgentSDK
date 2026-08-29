#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Generic text tail-follower for monitor task logs.

Mirrors ``tail -f`` behaviour for plain text files (not JSONL).  Used by the
Monitor TUI panel and by ``MonitorController.tail`` to stream the latest
output of a background monitor task without re-reading the whole file.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

# Poll interval when no new data is available.  500 ms is short enough for
# an interactive TUI panel while keeping idle CPU near zero.
_POLL_INTERVAL = 0.5


class TextTailFollower:
    """Async text-tail reader with an in-memory ring buffer.

    The follower reads newly appended bytes from a file on demand and keeps
    the most recent ``ring_size`` bytes in a deque.  It handles file
    truncation by resetting to the current end of file.
    """

    def __init__(self, path: str | Path, *, ring_size: int = 200_000) -> None:
        self._path = str(path)
        self._offset: int = 0
        self._ring_size = ring_size
        self._ring: deque[str] = deque()
        self._ring_bytes: int = 0
        self._stopping = False
        self._poll_interval = _POLL_INTERVAL

    # ---- public API -------------------------------------------------------

    async def start(self, from_offset: int | None = None) -> None:
        """Start following from ``from_offset`` bytes.

        If ``from_offset`` is omitted, follow from the current end of the
        file (``tail -f`` semantics).  If the file is shorter than the
        requested offset, reset to the current end of the file (truncation
        recovery).
        """
        self._stopping = False
        self._ring.clear()
        self._ring_bytes = 0
        stat = self._get_stat()
        if from_offset is None:
            self._offset = stat.st_size if stat is not None else 0
        else:
            self._offset = from_offset
            if stat is not None and stat.st_size < from_offset:
                self._offset = stat.st_size

    async def stop(self) -> None:
        """Signal the follower to stop after the next read cycle."""
        self._stopping = True

    async def read_chunk(self) -> str:
        """Return text appended since the last read, blocking if necessary.

        This method waits until at least one new byte is available or the
        follower has been stopped.  The returned text is also appended to
        the ring buffer.
        """
        while not self._stopping:
            chunk = self._read_available()
            if chunk:
                self._append_to_ring(chunk)
                return chunk
            await asyncio.sleep(self._poll_interval)
        return ""

    def read_available_now(self) -> str:
        """Return any currently available new text without blocking."""
        chunk = self._read_available()
        if chunk:
            self._append_to_ring(chunk)
        return chunk

    @property
    def current_tail(self) -> str:
        """Return the concatenated contents of the ring buffer."""
        return "".join(self._ring)

    @property
    def offset(self) -> int:
        """Current read offset in bytes."""
        return self._offset

    def set_fast_poll(self, enabled: bool) -> None:
        """Switch between default (500 ms) and fast (100 ms) polling."""
        self._poll_interval = 0.1 if enabled else _POLL_INTERVAL

    def __aiter__(self) -> TextTailFollower:
        return self

    async def __anext__(self) -> str:
        chunk = await self.read_chunk()
        if self._stopping and not chunk:
            raise StopAsyncIteration
        if not chunk:
            raise StopAsyncIteration
        return chunk

    # ---- internals --------------------------------------------------------

    def _append_to_ring(self, text: str) -> None:
        """Append ``text`` to the ring, keeping at most ``ring_size`` bytes."""
        if not text:
            return
        if self._ring_size <= 0:
            return
        if len(text) > self._ring_size:
            text = text[-self._ring_size :]
        while self._ring and self._ring_bytes + len(text) > self._ring_size:
            dropped = self._ring.popleft()
            self._ring_bytes -= len(dropped)
        self._ring.append(text)
        self._ring_bytes += len(text)

    def _read_available(self) -> str:
        try:
            stat = self._get_stat()
            if stat is None:
                return ""
            current_size = stat.st_size

            if current_size < self._offset:
                # File was truncated — jump to the new end.
                self._offset = current_size
                return ""

            if current_size == self._offset:
                return ""

            with open(self._path, "rb") as fh:
                fh.seek(self._offset)
                raw = fh.read()
                self._offset = current_size

            return raw.decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        except OSError as exc:
            logger.warning("error reading %s: %s", self._path, exc)
            return ""

    def _get_stat(self) -> os.stat_result | None:
        try:
            return os.stat(self._path)
        except FileNotFoundError:
            return None
        except OSError:
            return None


class TextTailBuffer:
    """Synchronous ring-buffer helper for non-async consumers.

    Holds the last ``maxlen`` characters and can be fed line-by-line or in
    chunks.  Used by the TUI panel to render the current tail without doing
    async I/O on every render frame.
    """

    def __init__(self, maxlen: int = 200_000) -> None:
        self._ring: deque[str] = deque()
        self._maxlen = maxlen
        self._bytes: int = 0

    def append(self, text: str) -> None:
        """Append text to the ring, dropping the oldest bytes if needed."""
        if not text:
            return
        if self._maxlen <= 0:
            return
        if len(text) > self._maxlen:
            text = text[-self._maxlen :]
        while self._ring and self._bytes + len(text) > self._maxlen:
            dropped = self._ring.popleft()
            self._bytes -= len(dropped)
        self._ring.append(text)
        self._bytes += len(text)

    def append_lines(self, lines: Iterator[str] | list[str]) -> None:
        for line in lines:
            self.append(line if line.endswith("\n") else line + "\n")

    @property
    def text(self) -> str:
        return "".join(self._ring)

    def clear(self) -> None:
        self._ring.clear()
        self._bytes = 0

    def __len__(self) -> int:
        return self._bytes

    def snapshot(self) -> list[str]:
        """Return a copy of the buffered chunks."""
        return list(self._ring)


__all__ = [
    "TextTailBuffer",
    "TextTailFollower",
]
