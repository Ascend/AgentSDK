#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Tests for src/services/chrome/playwright_impl.py.

Playwright is an *optional* dependency, so the tests patch
``_try_import_playwright`` (or the module's binding) rather than
importing it. The fake returned by the patch implements just
enough of the async API to drive the controller.
"""
# pylint: disable=no-name-in-module, redefined-outer-name

from __future__ import annotations

from typing import Any

import pytest

import clawcodex_ext.services.chrome.playwright_impl as pw_module
from clawcodex_ext.services.chrome.base import ChromeController
from clawcodex_ext.services.chrome.models import ChromeActionType
from clawcodex_ext.services.chrome.playwright_impl import PlaywrightChromeController


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeElement:
    def __init__(self) -> None:
        self.screenshot_calls: int = 0
        self._bytes: bytes = b"\x89PNG\r\n\x1a\nFAKE"

    async def screenshot(self) -> bytes:
        self.screenshot_calls += 1
        return self._bytes


class _FakePage:
    def __init__(self) -> None:
        self.url: str = ""
        self.goto_calls: list[str] = []
        self.click_calls: list[str] = []
        self.fill_calls: list[tuple[str, str]] = []
        self.type_calls: list[tuple[str, str]] = []
        self.select_calls: list[tuple[str, str]] = []
        self.hover_calls: list[str] = []
        self.evaluate_calls: list[str] = []
        self.screenshot_calls: int = 0
        self._elements: dict[str, _FakeElement] = {}
        self._next_text: str = ""
        self._next_html: str = ""
        self._next_eval: Any = None
        self._next_screenshot: bytes = b"\x89PNG\r\n\x1a\nFULL"

    def set_next_text(self, value: str) -> None:
        self._next_text = value

    def set_next_html(self, value: str) -> None:
        self._next_html = value

    def set_next_eval(self, value: Any) -> None:
        self._next_eval = value

    def set_next_screenshot(self, value: bytes) -> None:
        self._next_screenshot = value

    def add_element(self, selector: str, element: _FakeElement) -> None:
        self._elements[selector] = element

    async def goto(self, url: str, wait_until: str = "load") -> None:
        self.goto_calls.append(url)
        self.url = url

    async def click(self, selector: str) -> None:
        self.click_calls.append(selector)

    async def fill(self, selector: str, value: str) -> None:
        self.fill_calls.append((selector, value))

    async def type(self, selector: str, value: str) -> None:
        self.type_calls.append((selector, value))

    async def select_option(self, selector: str, value: str) -> None:
        self.select_calls.append((selector, value))

    async def hover(self, selector: str) -> None:
        self.hover_calls.append(selector)

    async def evaluate(self, script: str) -> Any:
        self.evaluate_calls.append(script)
        if "scrollBy" in script:
            return None
        if "document.body.innerText" in script:
            return self._next_text
        if "document.documentElement.outerHTML" in script:
            return self._next_html
        return self._next_eval

    async def query_selector(self, selector: str) -> _FakeElement | None:
        return self._elements.get(selector)

    async def screenshot(self, full_page: bool = True) -> bytes:
        self.screenshot_calls += 1
        return self._next_screenshot


class _FakeBrowser:
    def __init__(self) -> None:
        self.page = _FakePage()
        self.closed: bool = False

    async def new_page(self) -> _FakePage:
        return self.page

    async def close(self) -> None:
        self.closed = True


class _FakePlaywright:
    """Returned by the patched ``_try_import_playwright``.

    Mirrors the real ``async_playwright()`` API surface used by
    the controller: ``await factory().start()`` returns a
    playwright object whose ``.chromium.launch()`` yields a
    browser, and whose ``.stop()`` shuts the manager down.
    """

    def __init__(self) -> None:
        self.chromium = self
        self._browser = _FakeBrowser()
        self.stopped: bool = False

    async def start(self) -> "_FakePlaywright":
        # Real ``async_playwright().start()`` returns the
        # ``AsyncPlaywright`` instance — for the fake, that's
        # ``self``.
        return self

    async def launch(self, headless: bool = True) -> _FakeBrowser:
        return self._browser

    async def stop(self) -> None:
        self.stopped = True


def _fake_pw_factory() -> Any:
    """Returns a callable that produces a fresh ``_FakePlaywright``
    each time the controller's ``start()`` calls it.
    """

    def _factory() -> _FakePlaywright:
        return _FakePlaywright()

    return _factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def installed_playwright(monkeypatch: pytest.MonkeyPatch):
    """Patch ``_try_import_playwright`` to return a factory whose
    output the controller treats as a real Playwright instance.
    """
    factory = _fake_pw_factory()
    monkeypatch.setattr(pw_module, "_try_import_playwright", lambda: factory)
    monkeypatch.setattr(pw_module, "_warned_missing_dep", True)
    return factory


@pytest.fixture
def missing_playwright(monkeypatch: pytest.MonkeyPatch):
    """Patch ``_try_import_playwright`` to return ``None`` (SDK missing)."""
    monkeypatch.setattr(pw_module, "_try_import_playwright", lambda: None)
    monkeypatch.setattr(pw_module, "_warned_missing_dep", True)


# ---------------------------------------------------------------------------
# Tests — dep-not-installed
# ---------------------------------------------------------------------------


def test_unavailable_when_playwright_missing(missing_playwright) -> None:
    ctrl = PlaywrightChromeController()
    assert ctrl.is_live is False
    assert "playwright not installed" in ctrl._unavailable_reason


@pytest.mark.asyncio
async def test_navigate_unavailable_when_missing(missing_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()  # becomes a no-op
    result = await ctrl.navigate("https://example.com")
    assert result.success is False
    assert "playwright not installed" in (result.error or "")


# ---------------------------------------------------------------------------
# Tests — dep installed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_launches_browser(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    assert ctrl.is_live is True
    health = ctrl.health()
    assert health["is_live"] is True
    assert health["backend"] == "playwright"
    assert health["headless"] is True


@pytest.mark.asyncio
async def test_start_idempotent(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    first = ctrl._browser
    await ctrl.start()
    assert ctrl._browser is first


@pytest.mark.asyncio
async def test_stop_closes_browser(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    browser = ctrl._browser
    assert browser is not None
    await ctrl.stop()
    assert browser.closed is True
    assert ctrl.is_live is False


@pytest.mark.asyncio
async def test_stop_when_not_started_is_noop(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.stop()  # must not raise
    assert ctrl.is_live is False


@pytest.mark.asyncio
async def test_navigate_updates_url(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    result = await ctrl.navigate("https://example.com")
    assert result.success is True
    assert result.url == "https://example.com"
    assert result.action_type is ChromeActionType.NAVIGATE
    assert ctrl._page is not None
    assert "https://example.com" in ctrl._page.goto_calls


@pytest.mark.asyncio
async def test_navigate_swallows_playwright_errors(
    installed_playwright,
) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    assert ctrl._page is not None
    ctrl._page.goto = _raise  # type: ignore[method-assign]
    result = await ctrl.navigate("https://x")
    assert result.success is False
    assert "boom" in (result.error or "")


@pytest.mark.asyncio
async def test_click(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    result = await ctrl.click("button.submit")
    assert result.success is True
    assert result.data == "button.submit"
    assert ctrl._page is not None
    assert ctrl._page.click_calls == ["button.submit"]


@pytest.mark.asyncio
async def test_type_text_default_uses_fill(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    result = await ctrl.type_text("#q", "hello")
    assert result.success is True
    assert result.data == "hello"
    assert ctrl._page is not None
    assert ctrl._page.fill_calls == [("#q", "hello")]
    assert ctrl._page.type_calls == []


@pytest.mark.asyncio
async def test_type_text_clear_first_false_uses_type(
    installed_playwright,
) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    await ctrl.type_text("#q", "x", clear_first=False)
    assert ctrl._page is not None
    assert ctrl._page.type_calls == [("#q", "x")]


@pytest.mark.asyncio
async def test_select_option(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    result = await ctrl.select_option("select#x", "v")
    assert result.success is True
    assert result.data == "v"
    assert ctrl._page is not None
    assert ctrl._page.select_calls == [("select#x", "v")]


@pytest.mark.asyncio
async def test_hover(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    await ctrl.hover("a.link")
    assert ctrl._page is not None
    assert ctrl._page.hover_calls == ["a.link"]


@pytest.mark.asyncio
async def test_scroll_runs_scrollBy(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    result = await ctrl.scroll(dx=5, dy=10)
    assert result.success is True
    assert "dx=5,dy=10" in (result.data or "")
    assert ctrl._page is not None
    assert any("scrollBy" in c for c in ctrl._page.evaluate_calls)


@pytest.mark.asyncio
async def test_screenshot_full_page(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    assert ctrl._page is not None
    ctrl._page.set_next_screenshot(b"\x89PNG\r\n\x1a\nFULL_PAGE")
    result = await ctrl.screenshot()
    assert result.success is True
    assert result.data == b"\x89PNG\r\n\x1a\nFULL_PAGE"
    assert result.action_type is ChromeActionType.SCREENSHOT


@pytest.mark.asyncio
async def test_screenshot_element(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    assert ctrl._page is not None
    el = _FakeElement()
    ctrl._page.add_element("#target", el)
    result = await ctrl.screenshot(selector="#target", full_page=False)
    assert result.success is True
    assert el.screenshot_calls == 1
    assert result.data == el._bytes


@pytest.mark.asyncio
async def test_screenshot_element_not_found(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    result = await ctrl.screenshot(selector="#missing")
    assert result.success is False
    assert "element not found" in (result.error or "")


@pytest.mark.asyncio
async def test_eval_js_returns_string(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    assert ctrl._page is not None
    ctrl._page.set_next_eval({"answer": 42})
    result = await ctrl.eval_js("({answer: 42})")
    assert result.success is True
    # JSON-serialised, not str-coerced.
    assert '"answer": 42' in (result.data or "")


@pytest.mark.asyncio
async def test_eval_js_returns_string_literal(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    assert ctrl._page is not None
    ctrl._page.set_next_eval("plain")
    result = await ctrl.eval_js("'plain'")
    assert result.success is True
    assert result.data == "plain"


@pytest.mark.asyncio
async def test_eval_js_swallows_errors(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    assert ctrl._page is not None
    ctrl._page.evaluate = _raise  # type: ignore[method-assign]
    result = await ctrl.eval_js("throw")
    assert result.success is False


@pytest.mark.asyncio
async def test_get_visible_text(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    assert ctrl._page is not None
    ctrl._page.set_next_text("Hello world")
    result = await ctrl.get_visible_text()
    assert result.success is True
    assert result.data == "Hello world"


@pytest.mark.asyncio
async def test_get_html(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    assert ctrl._page is not None
    ctrl._page.set_next_html("<html><body>hi</body></html>")
    result = await ctrl.get_html()
    assert result.success is True
    assert "<body>hi</body>" in (result.data or "")


@pytest.mark.asyncio
async def test_start_recording_is_noop(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    await ctrl.start()
    await ctrl.start_recording("/tmp/x.gif", fps=2)
    assert ctrl.is_recording is False
    assert await ctrl.stop_recording() == ""


def test_current_url_reflects_page(installed_playwright) -> None:
    ctrl = PlaywrightChromeController()
    assert ctrl.current_url == ""
    fake_page = _FakePage()
    fake_page.url = "https://example.com"
    ctrl._page = fake_page
    assert ctrl.current_url == "https://example.com"


def test_playwright_controller_is_chrome_controller_subclass(
    installed_playwright,
) -> None:
    assert isinstance(PlaywrightChromeController(), ChromeController)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _raise(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("boom")
