"""F-62 Chrome browser automation control.

Public surface:

* :class:`ChromeController` — the abstract base every backend
  implements. All operations are async and return
  :class:`ChromeActionResult` (never raise on per-operation
  failure).
* :class:`ChromeActionType` / :class:`ChromeActionResult` —
  the data models.
* :class:`NullChromeController` — the graceful-degradation
  backend. Every operation returns a structured failure with
  an actionable error message.
* :class:`PlaywrightChromeController` — the primary backend
  (Playwright is an *optional* dependency; the controller
  imports it lazily).
* :class:`MCPChromeController` — opt-in backend that talks to
  a running Chrome DevTools MCP server via
  :class:`clawcodex_ext.services.mcp.MCPConnectionManager`.
* :class:`RecordingChromeController` — wrapper that adds GIF
  frame capture to any backend.
* :func:`build_chrome_controller` — resolves the best available
  backend for the current environment.
* :func:`build_chrome_tools` — returns the seven ``chrome_*``
  :class:`Tool` objects registered into ``EXTENSION_TOOLS``.

Backend selection
-----------------
:func:`build_chrome_controller` honours a ``prefer`` keyword:

* ``"auto"`` (default) — MCP if ``CHROME_MCP_URL`` /
  ``CHROME_MCP_COMMAND`` is set, otherwise Playwright if
  importable, otherwise :class:`NullChromeController`.
* ``"mcp"`` — force MCP; falls back to Null if unconfigured.
* ``"playwright"`` — force Playwright; falls back to Null.
* ``"null"`` — always :class:`NullChromeController` (tests).

The resolved controller is cached in a module-level singleton;
:func:`_reset_chrome_singleton` is the test escape hatch.
"""

from __future__ import annotations

from .base import ChromeController, ChromeError
from .factory import (
    _reset_chrome_singleton,
    build_chrome_controller,
    build_chrome_tools,
)
from .mcp_impl import MCPChromeController
from .models import ChromeActionResult, ChromeActionType
from .null_impl import NullChromeController
from .playwright_impl import PlaywrightChromeController
from .recording import RecordingChromeController

__all__ = [
    "ChromeActionResult",
    "ChromeActionType",
    "ChromeController",
    "ChromeError",
    "MCPChromeController",
    "NullChromeController",
    "PlaywrightChromeController",
    "RecordingChromeController",
    "_reset_chrome_singleton",
    "build_chrome_controller",
    "build_chrome_tools",
]
