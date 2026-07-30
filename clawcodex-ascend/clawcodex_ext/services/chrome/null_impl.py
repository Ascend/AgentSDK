"""F-62 P62-A — :class:`NullChromeController` (graceful degradation).

The null controller is the fallback when no real backend is
available — Playwright is not installed, and ``CHROME_MCP_URL`` is
not set. Every operation returns a :class:`ChromeActionResult`
with ``success=False`` and a single canonical error message that
points the user at the install path. ``start`` and ``stop`` are
no-ops, ``is_live`` is always ``False``.

Mirrors :class:`src.services.analytics.sink.NullSink` (lines
35-39) and :class:`src.services.channels.null_channel.NullChannel`.
The contract is: importing and using the chrome surface must
never crash, regardless of optional-dep availability.
"""

from __future__ import annotations

from .base import ChromeController
from .models import ChromeActionResult, ChromeActionType

# A single module-level message keeps the test assertions stable
# and makes log scrapes trivial: every NullController call site
# produces the same string.
_UNAVAILABLE_MSG: str = (
    "chrome controller not available; install playwright "
    "(`pip install clawcodex[chrome] && playwright install chromium`) "
    "or set CHROME_MCP_URL to point at a Chrome DevTools MCP server"
)


class NullChromeController(ChromeController):
    """No-op controller. All operations degrade to a structured
    failure with an actionable error message.
    """

    def __init__(self) -> None:
        self._started: bool = False

    # ---- lifecycle -----------------------------------------------------

    async def start(self, headless: bool = True) -> None:
        # Idempotent — calling start twice is a no-op so the factory's
        # lazy-init pattern doesn't surprise callers.
        self._started = True

    async def stop(self) -> None:
        self._started = False

    # ---- operations ----------------------------------------------------

    async def navigate(self, url: str) -> ChromeActionResult:
        return self._fail(ChromeActionType.NAVIGATE, url=url)

    async def click(self, selector: str) -> ChromeActionResult:
        return self._fail(ChromeActionType.CLICK)

    async def type_text(
        self,
        selector: str,
        text: str,
        *,
        clear_first: bool = True,
    ) -> ChromeActionResult:
        return self._fail(ChromeActionType.TYPE)

    async def select_option(self, selector: str, value: str) -> ChromeActionResult:
        return self._fail(ChromeActionType.SELECT)

    async def hover(self, selector: str) -> ChromeActionResult:
        return self._fail(ChromeActionType.HOVER)

    async def scroll(self, *, dx: int = 0, dy: int = 1) -> ChromeActionResult:
        return self._fail(ChromeActionType.SCROLL)

    async def screenshot(
        self,
        selector: str | None = None,
        *,
        full_page: bool = True,
    ) -> ChromeActionResult:
        return self._fail(ChromeActionType.SCREENSHOT)

    async def eval_js(self, script: str) -> ChromeActionResult:
        return self._fail(ChromeActionType.EVAL_JS)

    async def get_visible_text(self) -> ChromeActionResult:
        return self._fail(ChromeActionType.GET_TEXT)

    async def get_html(self) -> ChromeActionResult:
        return self._fail(ChromeActionType.GET_HTML)

    # ---- recording -----------------------------------------------------

    async def start_recording(
        self,
        output_path: str,
        *,
        fps: int = 1,
    ) -> None:
        # Recording is a no-op too; we deliberately do not raise so
        # callers can be agnostic about the backend.
        return None

    async def stop_recording(self) -> str:
        return ""

    # ---- introspection -------------------------------------------------

    @property
    def is_live(self) -> bool:
        return False

    @property
    def is_recording(self) -> bool:
        return False

    @property
    def current_url(self) -> str:
        return ""

    # ---- internal ------------------------------------------------------

    def _fail(
        self,
        action_type: ChromeActionType,
        *,
        url: str = "",
    ) -> ChromeActionResult:
        return ChromeActionResult(
            success=False,
            error=_UNAVAILABLE_MSG,
            url=url,
            action_type=action_type,
        )


__all__ = ["NullChromeController"]
