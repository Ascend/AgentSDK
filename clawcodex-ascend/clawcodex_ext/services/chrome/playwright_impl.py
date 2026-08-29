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

"""P62-B/C — :class:`PlaywrightChromeController`.

The primary concrete controller. Wraps the ``playwright`` async
API, which is the de-facto Python wrapper around the Chrome
DevTools Protocol. The module-level :func:`_try_import_playwright`
helper makes the dep optional — the factory can call
``PlaywrightChromeController()`` even when Playwright is not
installed; the constructor sets a flag and every operation
returns a :class:`ChromeActionResult` with an install-hint error
matching the :class:`NullChromeController` contract.

Auto-install of the chromium binary (``playwright install
chromium``) is intentionally **not** performed — the binary is
~200 MB and the user is the right person to decide whether to
install it. The error message names the one-liner.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from .base import ChromeController
from .models import ChromeActionResult, ChromeActionType

if TYPE_CHECKING:
    from playwright.async_api import (  # type: ignore[import-not-found]
        Browser,
        Page,
        Playwright,
    )

logger = logging.getLogger(__name__)


# Module-level "have we tried importing yet?" guard mirrors the
# langfuse client's ``_warned_missing_dep`` pattern.
_warned_missing_dep: bool = False


def _try_import_playwright() -> Any:
    """Return the ``playwright.async_api`` module or ``None`` if
    Playwright is not installed.
    """
    global _warned_missing_dep
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except ImportError:
        if not _warned_missing_dep:
            logger.info(
                "playwright not installed; PlaywrightChromeController will "
                "degrade to NullChromeController behaviour. Install with "
                "`pip install clawcodex[chrome] && playwright install chromium` "
                "to enable local browser control."
            )
            _warned_missing_dep = True
        return None
    return async_playwright


class PlaywrightChromeController(ChromeController):
    """Real Chrome controller backed by Playwright.

    All operations are async; all return :class:`ChromeActionResult`
    rather than raising. Per-operation exceptions are caught and
    surfaced as ``success=False, error=str(exc)`` so the agent
    loop can recover / retry.
    """

    def __init__(self, *, headless: bool = True) -> None:
        self._headless = headless
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._lock = threading.RLock()
        self._started: bool = False
        self._async_playwright_factory = _try_import_playwright()
        self._unavailable_reason: str = (
            ""
            if self._async_playwright_factory is not None
            else (
                "playwright not installed; install with `pip install clawcodex[chrome] && playwright install chromium`"
            )
        )

    # ------------------------------------------------------------------
    # Properties — used by the factory + recording wrapper
    # ------------------------------------------------------------------

    @property
    def is_live(self) -> bool:
        return self._started and self._page is not None

    @property
    def current_url(self) -> str:
        # The page's URL is read from the in-memory handle. It is
        # not awaited — a sync property is the public surface.
        page = self._page
        if page is None:
            return ""
        # Playwright stores the URL as a sync attribute.
        try:
            return str(getattr(page, "url", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    @property
    def headless(self) -> bool:
        return self._headless

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, headless: bool = True) -> None:
        if self._started:
            return
        if self._async_playwright_factory is None:
            # Mirror NullController behaviour: start is a no-op,
            # every operation will return the install-hint error.
            self._started = True
            return

        self._headless = headless
        self._pw = await self._async_playwright_factory().start()
        self._browser = await self._pw.chromium.launch(headless=headless)
        self._page = await self._browser.new_page()
        self._started = True

    async def stop(self) -> None:
        with self._lock:
            started = self._started
            browser = self._browser
            pw = self._pw
            self._browser = None
            self._page = None
            self._pw = None
            self._started = False

        if not started:
            return
        if browser is not None:
            try:
                await browser.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("playwright browser close failed: %s", exc)
        if pw is not None:
            try:
                await pw.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("playwright stop failed: %s", exc)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("navigate")
        start = time.monotonic()
        try:
            await self._page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.NAVIGATE, str(exc), url=url)
        return self._ok(
            ChromeActionType.NAVIGATE,
            data=url,
            url=self.current_url or url,
            elapsed_ms=_elapsed_ms(start),
        )

    async def click(self, selector: str) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("click")
        start = time.monotonic()
        try:
            await self._page.click(selector)
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.CLICK, str(exc))
        return self._ok(
            ChromeActionType.CLICK,
            data=selector,
            url=self.current_url,
            elapsed_ms=_elapsed_ms(start),
        )

    async def type_text(
        self,
        selector: str,
        text: str,
        *,
        clear_first: bool = True,
    ) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("type_text")
        start = time.monotonic()
        try:
            if clear_first:
                # Playwright's ``fill`` clears + types in one step.
                await self._page.fill(selector, text)
            else:
                await self._page.type(selector, text)
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.TYPE, str(exc))
        return self._ok(
            ChromeActionType.TYPE,
            data=text,
            url=self.current_url,
            elapsed_ms=_elapsed_ms(start),
        )

    async def select_option(self, selector: str, value: str) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("select_option")
        start = time.monotonic()
        try:
            await self._page.select_option(selector, value)
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.SELECT, str(exc))
        return self._ok(
            ChromeActionType.SELECT,
            data=value,
            url=self.current_url,
            elapsed_ms=_elapsed_ms(start),
        )

    async def hover(self, selector: str) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("hover")
        start = time.monotonic()
        try:
            await self._page.hover(selector)
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.HOVER, str(exc))
        return self._ok(
            ChromeActionType.HOVER,
            data=selector,
            url=self.current_url,
            elapsed_ms=_elapsed_ms(start),
        )

    async def scroll(self, *, dx: int = 0, dy: int = 1) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("scroll")
        start = time.monotonic()
        try:
            await self._page.evaluate(f"window.scrollBy({int(dx)}, {int(dy)})")
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.SCROLL, str(exc))
        return self._ok(
            ChromeActionType.SCROLL,
            data=f"dx={dx},dy={dy}",
            url=self.current_url,
            elapsed_ms=_elapsed_ms(start),
        )

    async def screenshot(
        self,
        selector: str | None = None,
        *,
        full_page: bool = True,
    ) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("screenshot")
        start = time.monotonic()
        try:
            if selector:
                element = await self._page.query_selector(selector)
                if element is None:
                    return self._fail(
                        ChromeActionType.SCREENSHOT,
                        f"element not found: {selector}",
                    )
                data = await element.screenshot()
            else:
                data = await self._page.screenshot(full_page=full_page)
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.SCREENSHOT, str(exc))
        return ChromeActionResult(
            success=True,
            data=data,
            url=self.current_url,
            action_type=ChromeActionType.SCREENSHOT,
            elapsed_ms=_elapsed_ms(start),
        )

    async def eval_js(self, script: str) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("eval_js")
        start = time.monotonic()
        try:
            result = await self._page.evaluate(script)
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.EVAL_JS, str(exc))
        # ``evaluate`` returns JSON-serialisable values; serialise
        # the result so it is stable across types.
        serialised = _serialise_js_result(result)
        return self._ok(
            ChromeActionType.EVAL_JS,
            data=serialised,
            url=self.current_url,
            elapsed_ms=_elapsed_ms(start),
        )

    async def get_visible_text(self) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("get_visible_text")
        start = time.monotonic()
        try:
            text = await self._page.evaluate("document.body.innerText")
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.GET_TEXT, str(exc))
        return self._ok(
            ChromeActionType.GET_TEXT,
            data=str(text or ""),
            url=self.current_url,
            elapsed_ms=_elapsed_ms(start),
        )

    async def get_html(self) -> ChromeActionResult:
        if not self.is_live or self._page is None:
            return self._unavailable("get_html")
        start = time.monotonic()
        try:
            html = await self._page.evaluate("document.documentElement.outerHTML")
        except Exception as exc:  # noqa: BLE001
            return self._fail(ChromeActionType.GET_HTML, str(exc))
        return self._ok(
            ChromeActionType.GET_HTML,
            data=str(html or ""),
            url=self.current_url,
            elapsed_ms=_elapsed_ms(start),
        )

    # ------------------------------------------------------------------
    # Recording — the Playwright controller does not own the
    # recording loop. The :class:`RecordingChromeController` wrapper
    # handles that. Here we expose the minimum surface for the
    # ABC contract; both methods are no-ops.
    # ------------------------------------------------------------------

    async def start_recording(
        self,
        output_path: str,
        *,
        fps: int = 1,
    ) -> None:
        return None

    async def stop_recording(self) -> str:
        return ""

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "is_live": self.is_live,
            "is_recording": False,
            "url": self.current_url,
            "headless": self._headless,
            "backend": "playwright",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ok(
        self,
        action_type: ChromeActionType,
        *,
        data: str | bytes | None = None,
        url: str = "",
        elapsed_ms: float = 0.0,
    ) -> ChromeActionResult:
        return ChromeActionResult(
            success=True,
            data=data,
            url=url,
            action_type=action_type,
            elapsed_ms=elapsed_ms,
        )

    def _fail(
        self,
        action_type: ChromeActionType,
        error: str,
        *,
        url: str = "",
    ) -> ChromeActionResult:
        return ChromeActionResult(
            success=False,
            error=error,
            url=url,
            action_type=action_type,
        )

    def _unavailable(self, op: str) -> ChromeActionResult:
        return ChromeActionResult(
            success=False,
            error=self._unavailable_reason or f"chrome controller not started (called {op!r})",
            action_type=None,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000.0, 3)


def _serialise_js_result(result: Any) -> str:
    """Coerce a Playwright ``evaluate`` result into a stable string.

    Strings pass through unchanged; everything else is
    ``json.dumps``-ed. ``json.dumps``'s ``default=str`` is the
    safety net for unserialisable values (e.g. ``Map``, ``Set``).
    """
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        return f"<unserialisable: {type(result).__name__}: {exc}>"


__all__ = ["PlaywrightChromeController"]


# Late import so the abstract base is available for
# ``isinstance`` checks without an import cycle in
# factory / recording.
from .base import ChromeController  # noqa: E402  # pylint: disable=reimported
