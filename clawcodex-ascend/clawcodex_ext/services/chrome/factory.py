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

"""F-62 P62-A/B/C — :func:`build_chrome_controller` and :func:`build_chrome_tools`.

The factory resolves the best available :class:`ChromeController`
based on the runtime environment:

1. ``CHROME_MCP_URL`` / ``CHROME_MCP_COMMAND`` set →
   :class:`MCPChromeController` (reuses the existing
   ``MCPConnectionManager``).
2. ``playwright`` importable →
   :class:`PlaywrightChromeController` (standalone browser).
3. otherwise → :class:`NullChromeController` (graceful no-op).

The factory caches the resolved controller in a module-level
singleton so the seven :func:`build_chrome_tools` lambdas all
share the same backend. Tests can call :func:`_reset_chrome_singleton`
to drop the cache between cases.

Tools are built with :func:`src.tool_system.build_tool.build_tool`
to fill in the standard hook defaults — the raw ``Tool(...)``
constructor in the spec sketch would skip ``is_enabled=True`` and
friends. Each tool's ``call`` bridges the sync ``Tool`` interface
to the async controller using the same loop-detection pattern as
:mod:`clawcodex_ext.services.mcp.tool_wrapper` (lines 213-232).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import os
import threading
from typing import Any, Literal

from .base import ChromeController, ChromeError
from .models import ChromeActionResult, ChromeActionType
from .null_impl import NullChromeController

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton + reset (test escape hatch)
# ---------------------------------------------------------------------------


_controller_lock = threading.RLock()
_cached_controller: ChromeController | None = None


def _reset_chrome_singleton() -> None:
    """Drop the cached controller. Test-only escape hatch."""
    global _cached_controller
    with _controller_lock:
        _cached_controller = None


def _get_or_build_controller(*, prefer: str) -> ChromeController:
    """Return the cached controller, building it on first call.

    The ``prefer`` arg is forwarded to :func:`build_chrome_controller`
    the first time; subsequent calls reuse the cached instance
    regardless of ``prefer``. That keeps a single live controller
    in the process — operations on the same backend share state.
    """
    global _cached_controller
    with _controller_lock:
        if _cached_controller is None:
            _cached_controller = build_chrome_controller(prefer=prefer)
        return _cached_controller


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------


def build_chrome_controller(
    *,
    prefer: Literal["auto", "playwright", "mcp", "null"] = "auto",
) -> ChromeController:
    """Resolve the best available :class:`ChromeController`.

    ``prefer="auto"`` (default):
        1. ``CHROME_MCP_URL`` or ``CHROME_MCP_COMMAND`` set →
           :class:`MCPChromeController`.
        2. ``playwright`` importable →
           :class:`PlaywrightChromeController`.
        3. otherwise → :class:`NullChromeController`.

    ``prefer="playwright"``:
        Return a :class:`PlaywrightChromeController` regardless of
        env vars. If Playwright is not importable, fall back to
        :class:`NullChromeController` and log a warning.

    ``prefer="mcp"``:
        Return an :class:`MCPChromeController`. If neither env var
        is set, fall back to :class:`NullChromeController`.

    ``prefer="null"``:
        Always return a :class:`NullChromeController`. Useful for
        unit tests and offline environments.
    """
    if prefer == "null":
        return NullChromeController()

    if prefer == "mcp":
        return _build_mcp_controller()

    if prefer == "playwright":
        return _build_playwright_controller()

    # prefer == "auto"
    if _chrome_mcp_configured():
        ctrl = _build_mcp_controller()
        if not isinstance(ctrl, NullChromeController):
            return ctrl
    pw = _build_playwright_controller()
    if not isinstance(pw, NullChromeController):
        return pw
    return NullChromeController()


def _chrome_mcp_configured() -> bool:
    return bool(os.environ.get("CHROME_MCP_URL", "").strip()) or bool(os.environ.get("CHROME_MCP_COMMAND", "").strip())


def _build_mcp_controller() -> ChromeController:
    try:
        from .mcp_impl import MCPChromeController
    except ImportError as exc:  # pragma: no cover — defensive
        logger.warning("MCPChromeController import failed: %s", exc)
        return NullChromeController()
    return MCPChromeController()


def _build_playwright_controller() -> ChromeController:
    try:
        from .playwright_impl import PlaywrightChromeController
    except ImportError as exc:  # pragma: no cover — defensive
        logger.warning("PlaywrightChromeController import failed: %s", exc)
        return NullChromeController()
    return PlaywrightChromeController()


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def build_chrome_tools() -> list[Any]:
    """Return the seven ``chrome_*`` :class:`Tool` objects.

    Each tool's ``call`` bridges to the async controller using a
    loop-detection pattern (running loop → run in a worker
    thread, no loop → ``asyncio.run``). The chrome controller is
    a process-singleton resolved lazily on the first call.
    """
    # Late import — ``build_tool`` is a heavy module that pulls
    # in the tool-system package, and the chrome public API
    # should be importable from anywhere.
    from src.tool_system.build_tool import build_tool  # type: ignore[import-not-found]
    from src.tool_system.context import ToolContext  # type: ignore[import-not-found]
    from src.tool_system.protocol import ToolResult  # type: ignore[import-not-found]

    def _async_runner(coro: Any) -> Any:
        """Run ``coro`` to completion, returning the result.

        Mirrors :mod:`clawcodex_ext.services.mcp.tool_wrapper` lines
        213-232. The running loop branch is for callers that
        happen to be inside an event loop already.
        """
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False
        if running:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return asyncio.run(coro)

    def _wrap(operation: ChromeActionType) -> Any:
        """Build a sync ``Tool.call`` for an async chrome operation.

        ``operation_to_coro`` is set inside the closure so it
        binds to the controller's method at call time (the
        controller can be swapped via the singleton reset).
        """

        def _call(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            controller = _get_or_build_controller(prefer="auto")
            coro = _call_controller(controller, operation, args)
            try:
                result = _async_runner(coro)
            except ChromeError as exc:
                return ToolResult(
                    name=f"chrome_{operation.value}",
                    output=str(exc),
                    is_error=True,
                )
            except Exception as exc:
                return ToolResult(
                    name=f"chrome_{operation.value}",
                    output=f"unexpected error: {exc}",
                    is_error=True,
                )
            return _result_to_tool_result(operation, result)

        return _call

    return [
        _make_chrome_navigate(build_tool, _wrap),
        _make_chrome_click(build_tool, _wrap),
        _make_chrome_type(build_tool, _wrap),
        _make_chrome_select(build_tool, _wrap),
        _make_chrome_screenshot(build_tool, _wrap),
        _make_chrome_eval_js(build_tool, _wrap),
        _make_chrome_get_text(build_tool, _wrap),
    ]


# ---------------------------------------------------------------------------
# Per-tool builders
# ---------------------------------------------------------------------------


def _common_chrome_kwargs() -> dict[str, Any]:
    """Tool metadata shared by all seven chrome tools."""
    return {
        "search_hint": "browser",
        "is_read_only": lambda _input: False,
        "is_destructive": lambda _input: True,
        "is_concurrency_safe": lambda _input: False,
        "is_open_world": lambda _input: True,
    }


def _make_chrome_navigate(build_tool: Any, _wrap: Any) -> Any:
    return build_tool(
        name="chrome_navigate",
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute URL to navigate the current page to.",
                },
            },
            "required": ["url"],
        },
        call=_wrap(ChromeActionType.NAVIGATE),
        description=lambda _i: "Navigate the browser to a URL.",
        prompt=lambda: "Use chrome_navigate to open a URL in the browser.",
        get_tool_use_summary=lambda i: f"navigate → {i.get('url', '') if i else ''}",
        get_activity_description=lambda i: f"navigating to {i.get('url', '') if i else ''}",
        **_common_chrome_kwargs(),
    )


def _make_chrome_click(build_tool: Any, _wrap: Any) -> Any:
    return build_tool(
        name="chrome_click",
        input_schema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the element to click.",
                },
            },
            "required": ["selector"],
        },
        call=_wrap(ChromeActionType.CLICK),
        description=lambda _i: "Click a page element by CSS selector.",
        prompt=lambda: "Use chrome_click to interact with buttons / links.",
        get_tool_use_summary=lambda i: f"click → {i.get('selector', '') if i else ''}",
        get_activity_description=lambda i: f"clicking {i.get('selector', '') if i else ''}",
        **_common_chrome_kwargs(),
    )


def _make_chrome_type(build_tool: Any, _wrap: Any) -> Any:
    return build_tool(
        name="chrome_type",
        input_schema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the input element.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the element.",
                },
                "clear_first": {
                    "type": "boolean",
                    "default": True,
                    "description": "Clear the field before typing.",
                },
            },
            "required": ["selector", "text"],
        },
        call=_wrap(ChromeActionType.TYPE),
        description=lambda _i: "Type text into a form field.",
        prompt=lambda: "Use chrome_type to fill out form fields.",
        get_tool_use_summary=lambda i: f"type → {i.get('selector', '') if i else ''}",
        get_activity_description=lambda i: f"typing into {i.get('selector', '') if i else ''}",
        **_common_chrome_kwargs(),
    )


def _make_chrome_select(build_tool: Any, _wrap: Any) -> Any:
    return build_tool(
        name="chrome_select",
        input_schema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the <select> element.",
                },
                "value": {
                    "type": "string",
                    "description": "Option value or visible label.",
                },
            },
            "required": ["selector", "value"],
        },
        call=_wrap(ChromeActionType.SELECT),
        description=lambda _i: "Select an option in a <select> element.",
        prompt=lambda: "Use chrome_select to pick an option from a dropdown.",
        get_tool_use_summary=lambda i: f"select → {i.get('selector', '') if i else ''}",
        get_activity_description=lambda i: f"selecting in {i.get('selector', '') if i else ''}",
        **_common_chrome_kwargs(),
    )


def _make_chrome_screenshot(build_tool: Any, _wrap: Any) -> Any:
    return build_tool(
        name="chrome_screenshot",
        input_schema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to capture a single element.",
                },
                "full_page": {
                    "type": "boolean",
                    "default": True,
                    "description": "Capture the full scrollable page.",
                },
            },
        },
        call=_wrap(ChromeActionType.SCREENSHOT),
        description=lambda _i: "Capture a screenshot (full page or element).",
        prompt=lambda: "Use chrome_screenshot to inspect page state visually.",
        get_tool_use_summary=lambda i: "screenshot",
        get_activity_description=lambda i: "taking a screenshot",
        **_common_chrome_kwargs(),
    )


def _make_chrome_eval_js(build_tool: Any, _wrap: Any) -> Any:
    return build_tool(
        name="chrome_eval_js",
        input_schema={
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "JavaScript to evaluate in the page context.",
                },
            },
            "required": ["script"],
        },
        call=_wrap(ChromeActionType.EVAL_JS),
        description=lambda _i: "Evaluate JavaScript in the current page.",
        prompt=lambda: "Use chrome_eval_js for page introspection or custom actions.",
        get_tool_use_summary=lambda i: "eval_js",
        get_activity_description=lambda i: "evaluating JavaScript",
        **_common_chrome_kwargs(),
    )


def _make_chrome_get_text(build_tool: Any, _wrap: Any) -> Any:
    return build_tool(
        name="chrome_get_text",
        input_schema={"type": "object", "properties": {}},
        call=_wrap(ChromeActionType.GET_TEXT),
        description=lambda _i: "Read the visible text of the current page.",
        prompt=lambda: "Use chrome_get_text to extract page content.",
        get_tool_use_summary=lambda _i: "get_text",
        get_activity_description=lambda _i: "reading page text",
        **_common_chrome_kwargs(),
    )


# ---------------------------------------------------------------------------
# Controller-call dispatch
# ---------------------------------------------------------------------------


async def _call_controller(
    controller: ChromeController,
    operation: ChromeActionType,
    args: dict[str, Any],
) -> ChromeActionResult:
    """Translate ``(operation, args)`` into the right
    controller call and await it.
    """
    if operation is ChromeActionType.NAVIGATE:
        return await controller.navigate(str(args.get("url", "")))
    if operation is ChromeActionType.CLICK:
        return await controller.click(str(args.get("selector", "")))
    if operation is ChromeActionType.TYPE:
        return await controller.type_text(
            str(args.get("selector", "")),
            str(args.get("text", "")),
            clear_first=bool(args.get("clear_first", True)),
        )
    if operation is ChromeActionType.SELECT:
        return await controller.select_option(str(args.get("selector", "")), str(args.get("value", "")))
    if operation is ChromeActionType.SCREENSHOT:
        return await controller.screenshot(
            selector=args.get("selector"),
            full_page=bool(args.get("full_page", True)),
        )
    if operation is ChromeActionType.EVAL_JS:
        return await controller.eval_js(str(args.get("script", "")))
    if operation is ChromeActionType.GET_TEXT:
        return await controller.get_visible_text()
    if operation is ChromeActionType.GET_HTML:
        return await controller.get_html()
    if operation is ChromeActionType.HOVER:
        return await controller.hover(str(args.get("selector", "")))
    if operation is ChromeActionType.SCROLL:
        return await controller.scroll(dx=int(args.get("dx", 0)), dy=int(args.get("dy", 1)))
    return ChromeActionResult(
        success=False,
        error=f"unsupported chrome operation: {operation.value!r}",
    )


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------


def _result_to_tool_result(
    operation: ChromeActionType,
    result: ChromeActionResult,
) -> Any:
    """Translate a :class:`ChromeActionResult` into a
    :class:`ToolResult` (sync).
    """
    # Local import — keeps the top of this module importable in
    # environments where the tool-system is not yet initialised.
    from src.tool_system.protocol import ToolResult  # type: ignore[import-not-found]

    name = f"chrome_{operation.value}"
    if not result.success:
        return ToolResult(name=name, output=result.error or "unknown error", is_error=True)

    data = result.data
    if data is None:
        return ToolResult(
            name=name,
            output=json.dumps(
                {
                    "url": result.url,
                    "elapsed_ms": result.elapsed_ms,
                },
                ensure_ascii=False,
            ),
        )
    if isinstance(data, (bytes, bytearray)):
        # The agent loop cannot embed raw bytes in text output.
        # Persist a sidecar via the recording metadata helper
        # (this also keeps the size predictable). We emit a
        # text marker pointing the agent at the path; the actual
        # image bytes are saved below.
        path = _persist_screenshot_bytes(bytes(data))
        result = dataclasses.replace(result, screenshot_path=path)
        return ToolResult(
            name=name,
            output=json.dumps(
                {
                    "screenshot_path": path,
                    "url": result.url,
                    "size_bytes": len(data),
                    "elapsed_ms": result.elapsed_ms,
                },
                ensure_ascii=False,
            ),
        )
    if isinstance(data, (dict, list)):
        payload = {
            "data": data,
            "url": result.url,
            "elapsed_ms": result.elapsed_ms,
        }
        return ToolResult(name=name, output=json.dumps(payload, ensure_ascii=False))
    # String / anything else.
    return ToolResult(
        name=name,
        output=str(data),
        content_type="text",
    )


def _persist_screenshot_bytes(data: bytes) -> str:
    """Write raw screenshot bytes to a temp file and return the path.

    Kept module-private; the recording wrapper has its own
    per-session finaliser. This is the path returned to the
    agent when ``chrome_screenshot`` runs without a recording
    in progress.
    """
    import tempfile
    from pathlib import Path

    out_dir = Path(
        os.environ.get(
            "CLAWCODEX_CHROME_SCREENSHOT_DIR",
            tempfile.gettempdir(),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=out_dir, prefix="chrome_screenshot_", suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise
    return path


__all__ = [
    "_reset_chrome_singleton",
    "build_chrome_controller",
    "build_chrome_tools",
]
