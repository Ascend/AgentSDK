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

"""P62-A — :class:`MCPChromeController`.

Bridges the abstract :class:`ChromeController` surface to a
running Chrome DevTools MCP server. The Chrome DevTools MCP
server is a separate JS application (e.g. Anthropic's
``chrome-devtools-mcp``) that exposes the same browser
primitives — ``navigate``, ``click``, ``screenshot`` etc. — as
MCP tools. clawcodex never ships the server; it merely connects
to one the user has running.

Configuration
-------------
Two env vars are read at construction time:

* ``CHROME_MCP_URL``  — the existing transport hint, e.g.
  ``http://localhost:1234/mcp`` for an HTTP-transport server or
  ``ws://localhost:1234/ws`` for the WebSocket variant. When
  unset, :meth:`start` short-circuits to a no-op and every
  operation returns a :class:`ChromeActionResult` with
  ``success=False`` and an actionable error message.
* ``CHROME_MCP_COMMAND`` / ``CHROME_MCP_ARGS`` — the
  command-line of a stdio-transport server. When both are
  present, the controller spawns the process itself; otherwise
  the URL mode is assumed.

Why a thin shim?
----------------
The Chrome DevTools MCP server's tool names already match the
claude-code convention (``chrome_navigate`` etc.). :meth:`_call`
just translates the clawcodex operation into the
``call_tool`` invocation, and renders the response into a
:class:`ChromeActionResult`. No new auth, transport, or
protocol code is added — we reuse the existing
:class:`MCPConnectionManager` end-to-end.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import threading
from typing import Any

from .base import ChromeController, ChromeError
from .models import ChromeActionResult, ChromeActionType

logger = logging.getLogger(__name__)


class ChromeConfigError(ChromeError):
    """Raised when the chrome controller can't read its config.

    The two causes are: (1) the upstream ``mcp`` SDK isn't
    installed, so the McpServerConfig types can't be imported;
    (2) the env vars contradict each other in a way that
    precludes a valid config. Both leave the controller in a
    "degraded" state — ``is_live`` is False, every operation
    returns ``success=False`` with the error message.
    """


# Map (clawcodex operation) → (MCP tool name on the server).
# The server's tool names are the public contract; we map 1:1
# rather than re-naming, so a user who already has the Chrome
# DevTools MCP server running needs zero extra configuration.
_MCP_TOOL_NAMES: dict[ChromeActionType, str] = {
    ChromeActionType.NAVIGATE: "chrome_navigate",
    ChromeActionType.CLICK: "chrome_click",
    ChromeActionType.TYPE: "chrome_type",
    ChromeActionType.SELECT: "chrome_select",
    ChromeActionType.SCREENSHOT: "chrome_screenshot",
    ChromeActionType.EVAL_JS: "chrome_eval_js",
    ChromeActionType.GET_TEXT: "chrome_get_text",
    ChromeActionType.GET_HTML: "chrome_get_html",
    ChromeActionType.HOVER: "chrome_hover",
    ChromeActionType.SCROLL: "chrome_scroll",
}


class MCPChromeController(ChromeController):
    """Browser controller backed by a Chrome DevTools MCP server.

    Activated only when ``CHROME_MCP_URL`` or
    ``CHROME_MCP_COMMAND`` is set. Without either, the controller
    stays in a "not configured" state and every operation
    degrades to :class:`ChromeActionResult` with
    ``success=False``.
    """

    def __init__(
        self,
        *,
        manager: Any | None = None,
        server_name: str = "chrome",
    ) -> None:
        self._manager = manager  # injected for tests; resolved lazily otherwise
        self._server_name = server_name
        self._lock = threading.RLock()
        self._started: bool = False
        self._available_tools: set[str] = set()
        self._current_url: str = ""
        # When True, the lazy ``start`` was skipped because no
        # configuration was present.
        self._unconfigured: bool = not self._is_configured()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _is_configured(self) -> bool:
        url = os.environ.get("CHROME_MCP_URL", "").strip()
        cmd = os.environ.get("CHROME_MCP_COMMAND", "").strip()
        return bool(url) or bool(cmd)

    def _config_error(self) -> str:
        return (
            "CHROME_MCP_URL or CHROME_MCP_COMMAND not set; "
            "either set one of these env vars to point at a running "
            "Chrome DevTools MCP server, or install Playwright "
            "(`pip install clawcodex[chrome] && playwright install chromium`)"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_live(self) -> bool:
        if self._unconfigured:
            return False
        if not self._started:
            # ``stop()`` flips ``_started`` back to False; the
            # manager's connection state may still report
            # "connected" but the controller is no longer using it.
            return False
        # Defer to the manager's view of the connection.
        manager = self._manager
        if manager is None:
            return False
        try:
            state = manager.get_state(self._server_name)
        except Exception:
            return False
        return state is not None and getattr(state, "type", None) == "connected"

    @property
    def current_url(self) -> str:
        return self._current_url

    @property
    def server_name(self) -> str:
        return self._server_name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, headless: bool = True) -> None:
        """Connect to the Chrome DevTools MCP server.

        The connection is mediated by the existing
        :class:`MCPConnectionManager` — we inject a dynamic
        server config and let the manager own the lifecycle.
        """
        if self._started:
            return
        if self._unconfigured:
            # Mirror NullController: start is a no-op, every
            # operation will return the config error.
            self._started = True
            return

        manager = self._manager
        if manager is None:
            from clawcodex_ext.services.mcp.connection_manager import (  # type: ignore[import-not-found]  # pylint: disable=no-name-in-module
                MCPConnectionManager,
            )

            manager = MCPConnectionManager()
            self._manager = manager

        try:
            config = self._build_server_config()
        except ChromeConfigError as exc:
            logger.warning("MCPChromeController start failed: %s", exc)
            self._unconfigured = True
            self._started = True
            return
        try:
            # ``inject_dynamic_config`` is a coroutine on the
            # real manager; we await it through ``asyncio.run``
            # if there is no running loop.
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(manager.inject_dynamic_config(self._server_name, config, auto_connect=True))
            else:
                # Already inside a loop — schedule the inject.
                # The caller is expected to ``await start()``,
                # so we await directly.
                await manager.inject_dynamic_config(self._server_name, config, auto_connect=True)
        except Exception as exc:
            logger.warning("MCPChromeController start failed: %s", exc)
            self._unconfigured = True
            self._started = True
            return

        # Snapshot the server's tool list so we can give precise
        # errors when an operation isn't supported.
        try:
            tools = manager.get_tools(self._server_name)
            self._available_tools = {t.name for t in tools}
        except Exception:
            self._available_tools = set()

        self._started = True

    async def stop(self) -> None:
        with self._lock:
            self._started = False
            self._available_tools = set()

    # ------------------------------------------------------------------
    # Operations — all translate to ``client.call_tool`` via the manager
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> ChromeActionResult:
        result = await self._call(ChromeActionType.NAVIGATE, {"url": url})
        if result.success and not self._current_url:
            # ``_render`` may have already populated
            # ``_current_url`` from a JSON ``{"url": ...}`` payload
            # in the response — only fall back to the request
            # URL when the server didn't tell us where it landed.
            self._current_url = url
        return result

    async def click(self, selector: str) -> ChromeActionResult:
        return await self._call(ChromeActionType.CLICK, {"selector": selector})

    async def type_text(
        self,
        selector: str,
        text: str,
        *,
        clear_first: bool = True,
    ) -> ChromeActionResult:
        return await self._call(
            ChromeActionType.TYPE,
            {
                "selector": selector,
                "text": text,
                "clear_first": clear_first,
            },
        )

    async def select_option(self, selector: str, value: str) -> ChromeActionResult:
        return await self._call(
            ChromeActionType.SELECT,
            {"selector": selector, "value": value},
        )

    async def hover(self, selector: str) -> ChromeActionResult:
        return await self._call(ChromeActionType.HOVER, {"selector": selector})

    async def scroll(self, *, dx: int = 0, dy: int = 1) -> ChromeActionResult:
        return await self._call(ChromeActionType.SCROLL, {"dx": int(dx), "dy": int(dy)})

    async def screenshot(
        self,
        selector: str | None = None,
        *,
        full_page: bool = True,
    ) -> ChromeActionResult:
        args: dict[str, Any] = {"full_page": full_page}
        if selector is not None:
            args["selector"] = selector
        return await self._call(ChromeActionType.SCREENSHOT, args)

    async def eval_js(self, script: str) -> ChromeActionResult:
        return await self._call(ChromeActionType.EVAL_JS, {"script": script})

    async def get_visible_text(self) -> ChromeActionResult:
        return await self._call(ChromeActionType.GET_TEXT, {})

    async def get_html(self) -> ChromeActionResult:
        return await self._call(ChromeActionType.GET_HTML, {})

    # ------------------------------------------------------------------
    # Recording — the MCP controller does not own the recording loop.
    # Mirrors PlaywrightChromeController.
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
            "backend": "mcp",
            "server_name": self._server_name,
            "available_tools": sorted(self._available_tools),
        }

    # ------------------------------------------------------------------
    # Internal — call_tool bridge
    # ------------------------------------------------------------------

    async def _call(
        self,
        action_type: ChromeActionType,
        args: dict[str, Any],
    ) -> ChromeActionResult:
        if self._unconfigured:
            return self._unconfigured_result(action_type)
        if not self.is_live:
            return ChromeActionResult(
                success=False,
                error="MCP chrome server is not connected",
                action_type=action_type,
            )

        tool_name = _MCP_TOOL_NAMES.get(action_type)
        if tool_name is None:
            return ChromeActionResult(
                success=False,
                error=f"no MCP tool mapping for action {action_type.value!r}",
                action_type=action_type,
            )
        if self._available_tools and tool_name not in self._available_tools:
            return ChromeActionResult(
                success=False,
                error=(
                    f"chrome MCP server does not expose {tool_name!r}; available tools: {sorted(self._available_tools)}"
                ),
                action_type=action_type,
            )

        manager = self._manager
        if manager is None:
            return self._unconfigured_result(action_type)
        try:
            state = manager.get_state(self._server_name)
        except Exception as exc:
            return ChromeActionResult(
                success=False,
                error=f"failed to look up chrome MCP server: {exc}",
                action_type=action_type,
            )
        if state is None or getattr(state, "client", None) is None:
            return ChromeActionResult(
                success=False,
                error="chrome MCP client not available",
                action_type=action_type,
            )
        client = state.client
        try:
            raw = await client.call_tool(tool_name, args)
        except Exception as exc:
            return ChromeActionResult(
                success=False,
                error=f"chrome MCP {tool_name} failed: {exc}",
                action_type=action_type,
            )
        return self._render(action_type, raw)

    def _render(
        self,
        action_type: ChromeActionType,
        raw: Any,
    ) -> ChromeActionResult:
        """Translate an MCP ``call_tool`` response into a
        :class:`ChromeActionResult`.

        The MCP client returns objects with ``.content`` (list of
        content blocks) and ``.is_error``. We flatten ``content``
        to text / bytes using the same conventions the existing
        MCP tool wrapper uses.
        """
        is_error = bool(getattr(raw, "is_error", False))
        content_blocks = list(getattr(raw, "content", None) or [])
        if is_error:
            text = self._flatten_text(content_blocks)
            return ChromeActionResult(
                success=False,
                error=text or f"{action_type.value} failed",
                action_type=action_type,
            )

        data = self._flatten(content_blocks)
        # The navigate action conventionally returns the new URL
        # in ``data["url"]``; mirror that into the result's ``url``.
        if action_type is ChromeActionType.NAVIGATE and isinstance(data, dict) and isinstance(data.get("url"), str):
            self._current_url = data["url"]
        return ChromeActionResult(
            success=True,
            data=data,
            url=self._current_url,
            action_type=action_type,
        )

    @staticmethod
    def _flatten(content_blocks: list[Any]) -> Any:
        """Concatenate content blocks into a single payload.

        Mirrors the helpers in
        :mod:`clawcodex_ext.services.mcp.tool_wrapper` — text blocks are
        joined; an image block becomes raw bytes; a resource
        block is rendered as a path reference.

        If the joined text is a valid JSON document, it is
        decoded into a structured value (dict / list / scalar)
        so callers can read response fields like ``{"url": ...}``.
        The Chrome DevTools MCP server returns JSON payloads for
        navigation; decoding here lets the navigate handler pick
        the new URL out of the response.
        """
        texts: list[str] = []
        for block in content_blocks:
            btype = getattr(block, "type", None)
            if btype == "text":
                texts.append(str(getattr(block, "text", "")))
            elif btype == "image":
                # ``data`` is base64 in the MCP wire protocol; we
                # don't decode here because the agent tool will
                # handle persistence. Return a small marker.
                texts.append("<image>")
            elif btype == "resource":
                texts.append(str(getattr(block, "uri", "<resource>")))
        if not texts:
            return None
        joined = "\n".join(texts)
        stripped = joined.strip()
        if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
            try:
                return json.loads(stripped)
            except (ValueError, TypeError):
                # Not a JSON document after all — fall through
                # to the raw text below.
                pass
        return joined

    @staticmethod
    def _flatten_text(content_blocks: list[Any]) -> str:
        texts: list[str] = []
        for block in content_blocks:
            if getattr(block, "type", None) == "text":
                texts.append(str(getattr(block, "text", "")))
        return "\n".join(texts)

    def _unconfigured_result(self, action_type: ChromeActionType) -> ChromeActionResult:
        return ChromeActionResult(
            success=False,
            error=self._config_error(),
            action_type=action_type,
        )

    # ------------------------------------------------------------------
    # Server config builders — translate env vars to McpServerConfig
    # ------------------------------------------------------------------

    def _build_server_config(self) -> Any:
        """Return the right ``McpServerConfig`` for the env-var mode.

        The two modes are stdio (spawn a process) and HTTP/WS
        (connect to a running endpoint). The mode is decided by
        which env var is set; both are supported so the user can
        pick whichever the Chrome DevTools MCP server ships.
        """
        try:
            from clawcodex_ext.services.mcp.types import (  # type: ignore[import-not-found]
                McpHTTPServerConfig,
                McpSSEServerConfig,
                McpStdioServerConfig,
                McpWebSocketServerConfig,
            )
        except ImportError as exc:
            # The MCP service module is optional — if it can't
            # be imported (e.g. the upstream ``mcp`` SDK isn't
            # installed), the chrome controller degrades to an
            # unconfigured state. ``start()`` reports the error.
            raise ChromeConfigError(f"mcp types unavailable: {exc}") from exc

        url = os.environ.get("CHROME_MCP_URL", "").strip()
        cmd = os.environ.get("CHROME_MCP_COMMAND", "").strip()
        if cmd:
            args_list = (
                shlex.split(os.environ.get("CHROME_MCP_ARGS", ""))
                if os.environ.get("CHROME_MCP_ARGS", "").strip()
                else []
            )
            env_str = os.environ.get("CHROME_MCP_ENV", "").strip()
            env_map: dict[str, str] | None = None
            if env_str:
                env_map = {}
                for piece in env_str.split(";"):
                    if "=" in piece:
                        k, v = piece.split("=", 1)
                        env_map[k.strip()] = v.strip()
            return McpStdioServerConfig(command=cmd, args=args_list, env=env_map)
        # Fall back to the URL mode. Pick the discriminator from
        # the scheme so the MCPConnectionManager wires the right
        # transport.
        if url.startswith(("http://", "https://")):
            return McpHTTPServerConfig(url=url)
        if url.startswith(("ws://", "wss://")):
            return McpWebSocketServerConfig(url=url)
        if url.startswith("sse://"):
            return McpSSEServerConfig(url=url.replace("sse://", "http://"))
        # Default to HTTP for a bare ``localhost:1234``-style URL.
        if "://" not in url and url:
            return McpHTTPServerConfig(url=f"http://{url}")
        # Unreachable when ``_is_configured`` was True at the
        # top of ``start``; defensive fallback.
        return McpHTTPServerConfig(url="http://localhost:0")


__all__ = ["MCPChromeController"]
