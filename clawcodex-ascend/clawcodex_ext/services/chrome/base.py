"""F-62 P62-A/B/C — :class:`ChromeController` abstract base.

Mirrors the layered pattern used by :mod:`src.services.computer_use`
and :mod:`src.services.channels`: an ABC declares the contract, and
each backend (Playwright, MCP, Null) is a separate concrete class
that the factory chooses between.

Every method is ``async`` and returns a :class:`ChromeActionResult`
rather than raising. This is deliberate — browser operations
against a live session can fail in many non-fatal ways (selector
not found, navigation timeout, JS error in the page). Surfacing
those as :class:`ChromeActionResult` lets the agent loop decide
whether to retry, screenshot, or report the failure, instead of
crashing the tool dispatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import ChromeActionResult


class ChromeError(RuntimeError):
    """Raised only for programmer errors (e.g. calling an operation
    before :meth:`start` finished). Per-operation failures are
    surfaced via :class:`ChromeActionResult.success=False`.
    """


class ChromeController(ABC):
    """Async browser-control surface.

    Lifecycle: :meth:`start` → operations → :meth:`stop`. Operations
    called before :meth:`start` completes (or after :meth:`stop`)
    should return a :class:`ChromeActionResult` with ``success=False``
    rather than raising — but :class:`ChromeError` is the escape
    hatch for bugs.
    """

    # ---- lifecycle -----------------------------------------------------

    @abstractmethod
    async def start(self, headless: bool = True) -> None:
        """Bring the browser online. Idempotent: a second call while
        already started is a no-op.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Tear the browser down. Idempotent: a second call while
        already stopped is a no-op.
        """

    # ---- navigation ----------------------------------------------------

    @abstractmethod
    async def navigate(self, url: str) -> ChromeActionResult:
        """Navigate the current page to ``url``."""

    # ---- element interaction -------------------------------------------

    @abstractmethod
    async def click(self, selector: str) -> ChromeActionResult:
        """Click the first element matching ``selector`` (CSS)."""

    @abstractmethod
    async def type_text(
        self,
        selector: str,
        text: str,
        *,
        clear_first: bool = True,
    ) -> ChromeActionResult:
        """Type ``text`` into the element matched by ``selector``.

        When ``clear_first`` is true (the default) the field is
        emptied first; otherwise the new text is appended to the
        existing value.
        """

    @abstractmethod
    async def select_option(self, selector: str, value: str) -> ChromeActionResult:
        """Select ``value`` in a ``<select>`` element matched by ``selector``."""

    @abstractmethod
    async def hover(self, selector: str) -> ChromeActionResult:
        """Move the virtual cursor over the element matched by ``selector``."""

    @abstractmethod
    async def scroll(self, *, dx: int = 0, dy: int = 1) -> ChromeActionResult:
        """Scroll the page by ``(dx, dy)`` increments."""

    # ---- capture + introspection --------------------------------------

    @abstractmethod
    async def screenshot(
        self,
        selector: str | None = None,
        *,
        full_page: bool = True,
    ) -> ChromeActionResult:
        """Capture a PNG. ``selector=None`` → full page. ``full_page``
        is ignored when a selector is provided.
        """

    @abstractmethod
    async def eval_js(self, script: str) -> ChromeActionResult:
        """Run ``script`` in the page and return its serialised result."""

    @abstractmethod
    async def get_visible_text(self) -> ChromeActionResult:
        """Return ``document.body.innerText`` — the agent's primary
        way to read what's on the page.
        """

    @abstractmethod
    async def get_html(self) -> ChromeActionResult:
        """Return ``document.documentElement.outerHTML``."""

    # ---- recording -----------------------------------------------------

    @abstractmethod
    async def start_recording(
        self,
        output_path: str,
        *,
        fps: int = 1,
    ) -> None:
        """Begin frame-by-frame capture to ``output_path`` (a ``.gif``).
        ``fps`` controls capture interval (``1000 // fps`` ms).
        """

    @abstractmethod
    async def stop_recording(self) -> str:
        """Stop recording and return the GIF file path.

        Raises :class:`ChromeError` if no recording is in flight.
        """

    # ---- concrete helpers (not abstract) ------------------------------

    @property
    def is_recording(self) -> bool:
        """True iff a recording session is currently in flight.

        Subclasses with a recording implementation override this
        property. The default returns ``False`` so that minimal
        controllers (Null, MCP) can stay simple.
        """
        return False

    @property
    def current_url(self) -> str:
        """Best-effort current URL; empty string when the controller
        is not started or does not track the URL.
        """
        return ""

    def health(self) -> dict[str, Any]:
        """Snapshot of internal state for diagnostics.

        The default implementation returns ``{"is_live": False,
        "is_recording": self.is_recording, "url": self.current_url}``.
        Backends with extra state (Playwright's headless flag,
        MCP's connected-server name) override and add keys.
        """
        return {
            "is_live": False,
            "is_recording": self.is_recording,
            "url": self.current_url,
        }


__all__ = ["ChromeController", "ChromeError"]
