"""F-62 P62-B/C/D — data models for the Chrome controller.

The action-type enum and result dataclass are the only pieces
of the chrome surface that are guaranteed to be importable
without optional dependencies. Everything else in the package
either pulls in Playwright (heavy) or the MCP client
(connected to a running server), and so falls behind lazy /
deferred import boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ChromeActionType(str, Enum):
    """The set of operations the agent can ask a browser to perform.

    ``str`` mixin so the enum serialises cleanly to JSON (MCP
    responses, recording metadata, analytics events) without an
    explicit ``.value`` call.
    """

    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    SCREENSHOT = "screenshot"
    EVAL_JS = "eval_js"
    GET_HTML = "get_html"
    GET_TEXT = "get_text"
    HOVER = "hover"
    SCROLL = "scroll"


@dataclass(frozen=True)
class ChromeActionResult:
    """Uniform return shape for every controller operation.

    Frozen so callers can hash / dedupe results when assembling
    transcripts. ``data`` carries the operation's payload — a
    URL string, screenshot bytes, JS result (JSON-encoded), or
    visible text — depending on ``action_type``. ``url`` is the
    page's last-known URL after the action completed; tools that
    don't change the URL (e.g. ``eval_js``) leave it empty.
    """

    success: bool
    data: str | bytes | None = None
    error: str | None = None
    url: str = ""
    screenshot_path: str | None = None
    elapsed_ms: float = 0.0
    action_type: ChromeActionType | None = None
    metadata: dict[str, Any] | None = None


__all__ = ["ChromeActionResult", "ChromeActionType"]
