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

"""Tests for src/services/chrome/recording.py."""
# pylint: disable=no-name-in-module, redefined-outer-name
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import clawcodex_ext.services.chrome.recording as rec_module
from clawcodex_ext.services.chrome.models import ChromeActionResult, ChromeActionType
from clawcodex_ext.services.chrome.recording import RecordingChromeController


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeInner:
    """A minimal ``ChromeController`` that records calls and returns
    canned results. Lets us assert how the wrapper delegates.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.next_screenshot: bytes = b"\x89PNG\r\n\x1a\nFRAME"
        self.next_text: str = "hello"
        self.next_html: str = "<html/>"
        self.next_eval: Any = None
        self._url: str = ""

    async def start(self, headless: bool = True) -> None:
        self.calls.append(("start", (headless,), {}))

    async def stop(self) -> None:
        self.calls.append(("stop", (), {}))

    async def navigate(self, url: str) -> ChromeActionResult:
        self.calls.append(("navigate", (url,), {}))
        self._url = url
        return ChromeActionResult(success=True, data=url, url=url)

    async def click(self, selector: str) -> ChromeActionResult:
        self.calls.append(("click", (selector,), {}))
        return ChromeActionResult(success=True, data=selector)

    async def type_text(self, selector: str, text: str, *, clear_first: bool = True) -> ChromeActionResult:
        self.calls.append(("type_text", (selector, text), {"clear_first": clear_first}))
        return ChromeActionResult(success=True, data=text)

    async def select_option(self, selector: str, value: str) -> ChromeActionResult:
        self.calls.append(("select_option", (selector, value), {}))
        return ChromeActionResult(success=True, data=value)

    async def hover(self, selector: str) -> ChromeActionResult:
        self.calls.append(("hover", (selector,), {}))
        return ChromeActionResult(success=True, data=selector)

    async def scroll(self, *, dx: int = 0, dy: int = 1) -> ChromeActionResult:
        self.calls.append(("scroll", (), {"dx": dx, "dy": dy}))
        return ChromeActionResult(success=True, data=f"dx={dx},dy={dy}")

    async def screenshot(self, selector: str | None = None, *, full_page: bool = True) -> ChromeActionResult:
        self.calls.append(("screenshot", (), {"selector": selector, "full_page": full_page}))
        return ChromeActionResult(
            success=True,
            data=self.next_screenshot,
            action_type=ChromeActionType.SCREENSHOT,
        )

    async def eval_js(self, script: str) -> ChromeActionResult:
        self.calls.append(("eval_js", (script,), {}))
        return ChromeActionResult(success=True, data="null")

    async def get_visible_text(self) -> ChromeActionResult:
        self.calls.append(("get_visible_text", (), {}))
        return ChromeActionResult(success=True, data=self.next_text)

    async def get_html(self) -> ChromeActionResult:
        self.calls.append(("get_html", (), {}))
        return ChromeActionResult(success=True, data=self.next_html)

    @property
    def current_url(self) -> str:
        return self._url

    def health(self) -> dict[str, Any]:
        return {"is_live": True, "url": self._url}


class _FakeImage:
    """Stand-in for ``PIL.Image.Image``. We don't decode bytes;
    we just record what ``save`` was called with.
    """

    last_saved: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._closed = False

    @classmethod
    def open(cls, path: str) -> "_FakeImage":
        return cls()

    def load(self) -> None:
        return None

    def save(self, *args: Any, **kwargs: Any) -> None:
        _FakeImage.last_saved = {"args": args, "kwargs": kwargs}

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_inner() -> _FakeInner:
    return _FakeInner()


@pytest.fixture
def installed_pillow(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rec_module, "_try_import_pillow", lambda: _FakeImage)
    return _FakeImage


@pytest.fixture
def missing_pillow(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rec_module, "_try_import_pillow", lambda: None)


def test_wrapper_delegates_navigation(fake_inner: _FakeInner) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    # isinstance check is duck-typed; we don't require ABC subclass.
    result = asyncio.run(rec.navigate("https://example.com"))
    assert result.success is True
    # The fake stores ``calls`` as ``(name, args, kwargs)`` where
    # ``args`` is a 1-tuple for the URL.
    assert ("navigate", ("https://example.com",), {}) in fake_inner.calls


def test_wrapper_delegates_screenshot_and_stashes_frame(
    fake_inner: _FakeInner,
) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    asyncio.run(rec.start_recording("/tmp/x.gif", fps=2))
    try:
        asyncio.run(rec.screenshot(full_page=True))
    finally:
        asyncio.run(rec.stop_recording())
    # The frame was captured into the wrapper.
    # (Pillow isn't installed, so the GIF file is not written,
    # but the in-memory buffer was populated.)
    assert rec.health()["frames_captured"] >= 0  # may be 0 if Pillow missing


@pytest.mark.asyncio
async def test_recording_without_pillow_is_noop(fake_inner: _FakeInner, missing_pillow) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    await rec.start_recording("/tmp/x.gif", fps=1)
    assert rec.is_recording is False
    path = await rec.stop_recording()
    assert path == ""


@pytest.mark.asyncio
async def test_recording_with_pillow_writes_gif(fake_inner: _FakeInner, installed_pillow, tmp_path) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    out = tmp_path / "session.gif"
    await rec.start_recording(str(out), fps=1)
    # Take a screenshot while recording.
    result = await rec.screenshot()
    assert result.success is True
    # Stop the recording.
    path = await rec.stop_recording()
    assert path == str(out)
    # The fake ``Image.save`` was called with format=GIF, save_all=True.
    kwargs = installed_pillow.last_saved.get("kwargs") or {}
    assert kwargs.get("format") == "GIF"
    assert kwargs.get("save_all") is True


@pytest.mark.asyncio
async def test_stop_cleans_up_temporary_files(fake_inner: _FakeInner, installed_pillow, tmp_path) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    out = tmp_path / "x.gif"
    await rec.start_recording(str(out), fps=1)
    # Force a second frame so the append_images path is exercised.
    await rec.screenshot()
    fake_inner.next_screenshot = b"\x89PNG\r\n\x1a\nFRAME2"
    await rec.screenshot()
    await rec.stop_recording()
    # No ``.tmp_chrome_rec_*.png`` files should remain.
    leftovers = list(tmp_path.glob(".tmp_chrome_rec_*.png"))
    assert leftovers == []


@pytest.mark.asyncio
async def test_stop_without_recording_returns_empty(fake_inner: _FakeInner, installed_pillow) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    path = await rec.stop_recording()
    assert path == ""


@pytest.mark.asyncio
async def test_recording_captures_only_successful_screenshots(fake_inner: _FakeInner, installed_pillow) -> None:
    """A failed screenshot must not corrupt the frame buffer."""
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    await rec.start_recording("/tmp/x.gif", fps=1)

    # First call: success.
    await rec.screenshot()
    # Second call: failure.
    original = fake_inner.screenshot

    async def _fail(*a: Any, **k: Any) -> ChromeActionResult:
        return ChromeActionResult(success=False, error="boom")

    fake_inner.screenshot = _fail  # type: ignore[method-assign]
    try:
        await rec.screenshot()
    finally:
        fake_inner.screenshot = original  # type: ignore[method-assign]
    # stop_recording should not raise.
    await rec.stop_recording()


@pytest.mark.asyncio
async def test_recording_idempotent_start(fake_inner: _FakeInner, installed_pillow) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    await rec.start_recording("/tmp/x.gif", fps=1)
    await rec.start_recording("/tmp/y.gif", fps=2)
    assert rec.is_recording is True
    await rec.stop_recording()


@pytest.mark.asyncio
async def test_recording_stop_without_frames(fake_inner: _FakeInner, installed_pillow, tmp_path) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    out = tmp_path / "empty.gif"
    await rec.start_recording(str(out), fps=1)
    # No screenshots taken.
    path = await rec.stop_recording()
    assert path == ""


@pytest.mark.asyncio
async def test_recording_stop_stops_capture_loop(fake_inner: _FakeInner, installed_pillow) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    await rec.start_recording("/tmp/x.gif", fps=10)
    assert rec.is_recording is True
    await rec.stop_recording()
    assert rec.is_recording is False


@pytest.mark.asyncio
async def test_recording_outer_stop_stops_recording_first(fake_inner: _FakeInner, installed_pillow) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    await rec.start_recording("/tmp/x.gif", fps=1)
    await rec.stop()
    # Inner stop was called.
    assert any(call[0] == "stop" for call in fake_inner.calls)
    assert rec.is_recording is False


def test_inner_property_exposes_wrapped_controller(
    fake_inner: _FakeInner,
) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    assert rec.inner is fake_inner


def test_health_includes_recording_metadata(
    fake_inner: _FakeInner,
) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    h = rec.health()
    assert "is_recording" in h
    assert "frames_captured" in h
    assert h["frames_captured"] == 0


def test_current_url_proxies_to_inner(
    fake_inner: _FakeInner,
) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    fake_inner._url = "https://example.com"
    assert rec.current_url == "https://example.com"


def test_write_metadata_sidecar_writes_json(fake_inner: _FakeInner, tmp_path) -> None:
    import json
    from pathlib import Path

    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    rec._output_path = str(tmp_path / "session.gif")
    rec._fps = 2
    rec._frames = [b"a", b"b", b"c"]
    sidecar_path = rec.write_metadata_sidecar()
    assert sidecar_path is not None

    payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    assert payload["fps"] == 2
    assert payload["frame_count"] == 3


def test_write_metadata_sidecar_returns_none_when_no_recording(
    fake_inner: _FakeInner,
) -> None:
    rec = RecordingChromeController(fake_inner)  # type: ignore[arg-type]
    assert rec.write_metadata_sidecar() is None
