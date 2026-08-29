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

"""Recording wrapper for :class:`RecordingChromeController`.

A recording wrapper is *not* a standalone controller: it accepts
any :class:`ChromeController` and layers GIF capture on top of it.
Every :meth:`screenshot` call (explicit or implicit via the
recording loop) is intercepted, the frame is appended to an
in-memory buffer, and on :meth:`stop_recording` the buffer is
composited into an animated GIF using Pillow.

Why a wrapper?
--------------
The spec asks for a recording mode that captures the *entire*
agent session, not just explicit screenshots. By wrapping the
controller we capture every screenshot the agent takes
(``chrome_screenshot`` tool, recording loop, eval_js
introspection) without instrumenting each backend separately.

Pillow is an optional dependency. When it is not importable,
:meth:`start_recording` is a no-op and :meth:`stop_recording`
returns the empty string — the wrapper degrades silently so
the rest of the agent loop is unaffected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import ChromeActionResult

if TYPE_CHECKING:
    from .base import ChromeController

logger = logging.getLogger(__name__)


# Sentinel module for "Pillow not installed". Keeps the import
# boundary explicit so tests can monkey-patch the loader without
# the real PIL hanging off sys.modules.
def _try_import_pillow() -> Any:
    """Return the ``PIL.Image`` module or ``None`` if Pillow is missing."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    return Image


class RecordingChromeController:
    """Wraps a :class:`ChromeController` and records every screenshot
    into a single GIF.

    The wrapper implements the *public* surface of
    :class:`ChromeController` (the methods tools call) by
    delegating to the inner controller. The :meth:`start_recording`
    / :meth:`stop_recording` pair is enhanced; :meth:`screenshot`
    additionally stashes the frame for the GIF.
    """

    def __init__(self, inner: ChromeController) -> None:
        self._inner = inner
        self._pil = _try_import_pillow()
        self._frames: list[bytes] = []
        self._output_path: str = ""
        self._fps: int = 1
        self._lock = threading.RLock()
        self._capture_task: asyncio.Task[None] | None = None
        self._is_recording: bool = False

    # ------------------------------------------------------------------
    # ChromeController surface — delegated, with screenshot interception
    # ------------------------------------------------------------------

    async def start(self, headless: bool = True) -> None:
        await self._inner.start(headless=headless)

    async def stop(self) -> None:
        if self._is_recording:
            await self.stop_recording()
        await self._inner.stop()

    async def navigate(self, url: str) -> ChromeActionResult:
        return await self._inner.navigate(url)

    async def click(self, selector: str) -> ChromeActionResult:
        return await self._inner.click(selector)

    async def type_text(
        self,
        selector: str,
        text: str,
        *,
        clear_first: bool = True,
    ) -> ChromeActionResult:
        return await self._inner.type_text(selector, text, clear_first=clear_first)

    async def select_option(self, selector: str, value: str) -> ChromeActionResult:
        return await self._inner.select_option(selector, value)

    async def hover(self, selector: str) -> ChromeActionResult:
        return await self._inner.hover(selector)

    async def scroll(self, *, dx: int = 0, dy: int = 1) -> ChromeActionResult:
        return await self._inner.scroll(dx=dx, dy=dy)

    async def screenshot(
        self,
        selector: str | None = None,
        *,
        full_page: bool = True,
    ) -> ChromeActionResult:
        result = await self._inner.screenshot(selector, full_page=full_page)
        # Stash the frame for the GIF, if recording is on and the
        # result carries bytes. We accept the small race (recording
        # toggled between get and put) — the buffer is best-effort.
        if (
            self._is_recording
            and result.success
            and isinstance(result.data, (bytes, bytearray))
            and len(result.data) > 0
        ):
            with self._lock:
                self._frames.append(bytes(result.data))
        return result

    async def eval_js(self, script: str) -> ChromeActionResult:
        return await self._inner.eval_js(script)

    async def get_visible_text(self) -> ChromeActionResult:
        return await self._inner.get_visible_text()

    async def get_html(self) -> ChromeActionResult:
        return await self._inner.get_html()

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    async def start_recording(
        self,
        output_path: str,
        *,
        fps: int = 1,
    ) -> None:
        """Begin frame capture. Idempotent — a second call while a
        recording is in flight refreshes the output path.
        """
        if self._pil is None:
            logger.info(
                "Pillow not installed; chrome recording degrades to no-op. "
                "Install with `pip install pillow` to enable GIF capture."
            )
            return

        with self._lock:
            self._output_path = output_path
            self._fps = max(1, int(fps))
            self._frames.clear()
            self._is_recording = True
            # Spawn a frame-capture loop on the running event loop.
            loop = asyncio.get_event_loop()
            self._capture_task = loop.create_task(self._capture_loop())

    async def stop_recording(self) -> str:
        """Stop frame capture, finalize the GIF, return its path.

        Returns the empty string when Pillow is missing (the
        recording is a no-op in that case) or no frames were
        captured.
        """
        with self._lock:
            self._is_recording = False
            task = self._capture_task
            self._capture_task = None
            output_path = self._output_path
            frames = list(self._frames)
            self._frames.clear()

        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # The capture loop is best-effort; cancellation  # noqa: BLE001
                # silence is the contract.
                pass

        if self._pil is None or not frames or not output_path:
            return ""

        try:
            self._finalize_gif(frames, output_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GIF finalization failed: %s", exc)
            return ""
        return output_path

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def current_url(self) -> str:
        return self._inner.current_url

    @property
    def inner(self) -> ChromeController:
        """The wrapped controller. Exposed so tests and operators
        can introspect or stop it directly.
        """
        return self._inner

    def health(self) -> dict[str, Any]:
        h = self._inner.health() if hasattr(self._inner, "health") else {}
        h["is_recording"] = self._is_recording
        h["frames_captured"] = len(self._frames)
        h["output_path"] = self._output_path
        return h

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _capture_loop(self) -> None:
        """Take a screenshot every ``1000 // fps`` ms while recording."""
        interval = 1.0 / max(1, self._fps)
        while True:
            if not self._is_recording:
                return
            try:
                result = await self._inner.screenshot(full_page=True)
                if result.success and isinstance(result.data, (bytes, bytearray)) and len(result.data) > 0:
                    with self._lock:
                        self._frames.append(bytes(result.data))
            except Exception as exc:  # noqa: BLE001
                logger.debug("frame capture failed: %s", exc)
            await asyncio.sleep(interval)

    def _finalize_gif(self, frames: list[bytes], output_path: str) -> None:
        """Composite ``frames`` into an animated GIF at ``output_path``.

        Pillow needs a real image file to anchor the
        ``save_all=True`` call. We write the first frame to a
        temp file, append the rest as ``append_images``, and
        rename atomically. If anything fails the temp file is
        removed so we never leave a torn write.
        """
        if len(frames) < 1:
            return
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Materialize the first frame into a temp file and let
        # Pillow open + append the rest.
        fd, tmp_path = tempfile.mkstemp(dir=out.parent, prefix=".tmp_chrome_rec_", suffix=".png")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(frames[0])
            anchor = self._pil.open(tmp_path)
            anchor.load()  # eager decode so the file handle is free
            anchor_frames = []
            for raw in frames[1:]:
                fd2, tmp2 = tempfile.mkstemp(dir=out.parent, prefix=".tmp_chrome_rec_", suffix=".png")
                try:
                    with os.fdopen(fd2, "wb") as f:
                        f.write(raw)
                    anchor_frames.append(self._pil.open(tmp2))
                except Exception:  # noqa: BLE001
                    Path(tmp2).unlink(missing_ok=True)
                    raise
            duration_ms = max(1, int(1000 / max(1, self._fps)))
            anchor.save(
                out,
                format="GIF",
                save_all=True,
                append_images=anchor_frames or None,
                duration=duration_ms,
                loop=0,
            )
            # Close the per-frame PIL images to release the temp
            # file handles before we delete them.
            for img in anchor_frames:
                try:
                    img.close()
                except Exception:  # noqa: BLE001  # nosec B110
                    pass  # Cleanup is best-effort and must not replace the primary operation result.
            anchor.close()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            # PIL keeps the temp files open via the lazy-decode
            # path on some versions; sweep the dir.
            for leftover in out.parent.glob(".tmp_chrome_rec_*.png"):
                try:
                    leftover.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001  # nosec B110
                    pass  # Cleanup is best-effort and must not replace the primary operation result.

    # ------------------------------------------------------------------
    # Recording-metadata sidecar
    # ------------------------------------------------------------------

    def write_metadata_sidecar(
        self,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        """Write a JSON sidecar next to the GIF describing the
        recording session. Returns the sidecar path, or ``None``
        if no recording has been finalized yet.

        The sidecar is intentionally minimal — the goal is to
        give downstream tools (e.g. a GIF replayer) enough info
        to render a tooltip without needing the full controller
        state.
        """
        if not self._output_path:
            return None
        payload = {
            "output_path": self._output_path,
            "fps": self._fps,
            "frame_count": len(self._frames),
            "created_at": time.time(),
        }
        if extra:
            payload["extra"] = extra
        sidecar = Path(self._output_path).with_suffix(".json")
        sidecar.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(sidecar)


__all__ = ["RecordingChromeController"]


# Re-export the abstract base + a runtime check, so callers can
# ``isinstance(rec, ChromeController)`` without importing base.
from .base import ChromeController  # noqa: E402
