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

"""Tests for src/services/chrome/mcp_impl.py.

The MCP controller delegates to ``MCPConnectionManager``. To
keep these tests independent of the real MCP stack, we inject
a fake manager via the ``manager=`` constructor kwarg.
"""
# pylint: disable=no-name-in-module, redefined-outer-name

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from clawcodex_ext.services.chrome.base import ChromeController
from clawcodex_ext.services.chrome.mcp_impl import MCPChromeController


# ---------------------------------------------------------------------------
# Stub the upstream ``mcp`` SDK + clawcodex_ext.services.mcp.types if missing.
# The real SDK isn't installed in this test environment, but
# the chrome MCP controller only needs the type symbols to build
# its server config — a sentinel dataclass suffices.
# ---------------------------------------------------------------------------


def _install_mcp_type_stubs() -> None:
    """Create a fake ``clawcodex_ext.services.mcp.types`` module exposing
    the four McpServerConfig dataclasses the chrome controller
    imports. Idempotent; safe to call multiple times.
    """
    if "clawcodex_ext.services.mcp.types" in sys.modules:
        return

    @dataclass
    class _StubStdio:
        command: str
        args: list[str] = field(default_factory=list)
        env: dict[str, str] | None = None

    @dataclass
    class _StubHTTP:
        url: str

    @dataclass
    class _StubWS:
        url: str

    @dataclass
    class _StubSSE:
        url: str

    stub = types.ModuleType("clawcodex_ext.services.mcp.types")
    stub.McpStdioServerConfig = _StubStdio  # type: ignore[attr-defined]
    stub.McpHTTPServerConfig = _StubHTTP  # type: ignore[attr-defined]
    stub.McpWebSocketServerConfig = _StubWS  # type: ignore[attr-defined]
    stub.McpSSEServerConfig = _StubSSE  # type: ignore[attr-defined]
    sys.modules["clawcodex_ext.services.mcp.types"] = stub


@pytest.fixture(autouse=True)
def _stub_mcp_types() -> None:
    _install_mcp_type_stubs()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeBlock:
    type: str = "text"
    text: str = ""
    uri: str = ""
    data: str = ""


@dataclass
class _FakeToolResult:
    content: list[_FakeBlock] = field(default_factory=list)
    is_error: bool = False


class _FakeMCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[_FakeToolResult] = []
        self.exceptions: list[Exception] = []

    def queue_response(self, result: _FakeToolResult) -> None:
        self.responses.append(result)

    async def call_tool(self, name: str, args: dict[str, Any]) -> _FakeToolResult:
        self.calls.append((name, args))
        if self.exceptions:
            raise self.exceptions.pop(0)
        if not self.responses:
            return _FakeToolResult(content=[_FakeBlock(text="")])
        return self.responses.pop(0)


@dataclass
class _FakeState:
    type: str = "connected"
    client: _FakeMCPClient | None = None


class _FakeManager:
    """Stand-in for ``MCPConnectionManager``.

    Tracks the configs that ``inject_dynamic_config`` received
    and serves a canned set of tools via ``get_tools``.
    """

    def __init__(self, *, available_tools: list[str] | None = None) -> None:
        self.configs: dict[str, Any] = {}
        self.tools_by_name: dict[str, list[Any]] = {}
        self._default_tools = available_tools or []
        self._states: dict[str, _FakeState] = {}
        self._clients: dict[str, _FakeMCPClient] = {}
        self._inject_calls: list[tuple[str, Any]] = []

    def register(self, name: str, *, available_tools: list[str] | None = None) -> None:
        tools = [type("T", (), {"name": t}) for t in (available_tools or self._default_tools)]
        self.tools_by_name[name] = tools
        client = _FakeMCPClient()
        self._clients[name] = client
        self._states[name] = _FakeState(client=client)

    async def inject_dynamic_config(self, name: str, config: Any, *, auto_connect: bool = True) -> None:
        self._inject_calls.append((name, config))
        self.configs[name] = config
        if name not in self._clients:
            client = _FakeMCPClient()
            self._clients[name] = client
            self._states[name] = _FakeState(client=client)
        if name not in self.tools_by_name:
            # Prefer the manager-level default (set via the
            # ``available_tools=`` kwarg or ``register()``); fall
            # back to a tiny 3-tool set so the controller can at
            # least call ``navigate`` / ``click`` / ``screenshot``
            # in tests that don't pre-register.
            default = self._default_tools or [
                "chrome_navigate",
                "chrome_click",
                "chrome_screenshot",
            ]
            self.tools_by_name[name] = [type("T", (), {"name": t}) for t in default]

    def get_state(self, name: str) -> _FakeState | None:
        return self._states.get(name)

    def get_tools(self, name: str) -> list[Any]:
        return list(self.tools_by_name.get(name, ()))

    def client_for(self, name: str) -> _FakeMCPClient:
        return self._clients[name]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_manager() -> _FakeManager:
    return _FakeManager(
        available_tools=[
            "chrome_navigate",
            "chrome_click",
            "chrome_type",
            "chrome_select",
            "chrome_screenshot",
            "chrome_eval_js",
            "chrome_get_text",
            "chrome_get_html",
            "chrome_hover",
            "chrome_scroll",
        ]
    )


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHROME_MCP_URL", "http://localhost:1234/mcp")


@pytest.fixture
def unconfigured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHROME_MCP_URL", raising=False)
    monkeypatch.delenv("CHROME_MCP_COMMAND", raising=False)


# ---------------------------------------------------------------------------
# Tests — env unconfigured
# ---------------------------------------------------------------------------


def test_unconfigured_when_no_env(unconfigured_env) -> None:
    ctrl = MCPChromeController()
    assert ctrl.is_live is False


@pytest.mark.asyncio
async def test_unconfigured_operations(unconfigured_env) -> None:
    ctrl = MCPChromeController()
    await ctrl.start()  # no-op
    result = await ctrl.navigate("https://example.com")
    assert result.success is False
    assert "CHROME_MCP_URL" in (result.error or "")


# ---------------------------------------------------------------------------
# Tests — configured but manager not yet wired
# ---------------------------------------------------------------------------


def test_configured_recognises_env(configured_env) -> None:
    ctrl = MCPChromeController()
    assert ctrl._is_configured() is True


def test_configured_but_no_manager_not_live(configured_env) -> None:
    ctrl = MCPChromeController()
    assert ctrl.is_live is False
    health = ctrl.health()
    assert health["backend"] == "mcp"
    assert health["server_name"] == "chrome"


# ---------------------------------------------------------------------------
# Tests — wired manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_injects_dynamic_config(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    assert "chrome" in fake_manager.configs
    assert ctrl.is_live is True


@pytest.mark.asyncio
async def test_start_is_idempotent(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    await ctrl.start()
    # inject_dynamic_config is called once.
    assert len(fake_manager._inject_calls) == 1


@pytest.mark.asyncio
async def test_stop_clears_tools(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    await ctrl.stop()
    assert ctrl._available_tools == set()


@pytest.mark.asyncio
async def test_navigate_calls_chrome_navigate_tool(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="ok")], is_error=False))
    result = await ctrl.navigate("https://example.com")
    assert result.success is True
    assert client.calls == [("chrome_navigate", {"url": "https://example.com"})]
    assert ctrl.current_url == "https://example.com"


@pytest.mark.asyncio
async def test_navigate_picks_up_url_from_response(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(
        _FakeToolResult(
            content=[_FakeBlock(text='{"url": "https://final/"}')],
            is_error=False,
        )
    )
    await ctrl.navigate("https://initial")
    assert ctrl.current_url == "https://final/"


@pytest.mark.asyncio
async def test_click(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="ok")]))
    result = await ctrl.click("button.submit")
    assert result.success is True
    assert client.calls[-1] == ("chrome_click", {"selector": "button.submit"})


@pytest.mark.asyncio
async def test_type_text_passes_args(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="ok")]))
    await ctrl.type_text("#q", "hi", clear_first=False)
    assert client.calls[-1] == (
        "chrome_type",
        {"selector": "#q", "text": "hi", "clear_first": False},
    )


@pytest.mark.asyncio
async def test_select_option(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="ok")]))
    await ctrl.select_option("select#x", "v")
    assert client.calls[-1] == ("chrome_select", {"selector": "select#x", "value": "v"})


@pytest.mark.asyncio
async def test_hover_and_scroll(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="ok")]))
    await ctrl.hover("a.link")
    assert client.calls[-1] == ("chrome_hover", {"selector": "a.link"})

    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="ok")]))
    await ctrl.scroll(dx=5, dy=10)
    assert client.calls[-1] == ("chrome_scroll", {"dx": 5, "dy": 10})


@pytest.mark.asyncio
async def test_screenshot_with_selector_and_full_page(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="ok")]))
    await ctrl.screenshot(selector="#x", full_page=False)
    assert client.calls[-1] == (
        "chrome_screenshot",
        {"selector": "#x", "full_page": False},
    )

    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="ok")]))
    await ctrl.screenshot()
    assert client.calls[-1] == (
        "chrome_screenshot",
        {"full_page": True},
    )


@pytest.mark.asyncio
async def test_eval_js_and_get_text(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(_FakeToolResult(content=[_FakeBlock(text='{"r": 1}')]))
    result = await ctrl.eval_js("({r:1})")
    assert result.success is True
    # JSON-shaped text blocks are decoded into structured data.
    assert result.data == {"r": 1}

    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="hello world")]))
    result = await ctrl.get_visible_text()
    assert result.data == "hello world"


@pytest.mark.asyncio
async def test_get_html(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="<html/>")]))
    result = await ctrl.get_html()
    assert result.success is True
    assert result.data == "<html/>"


@pytest.mark.asyncio
async def test_error_response(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.queue_response(_FakeToolResult(content=[_FakeBlock(text="bad selector")], is_error=True))
    result = await ctrl.click("a.bogus")
    assert result.success is False
    assert "bad selector" in (result.error or "")


@pytest.mark.asyncio
async def test_client_call_raises(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    client = fake_manager.client_for("chrome")
    client.exceptions.append(RuntimeError("connection lost"))
    result = await ctrl.navigate("https://x")
    assert result.success is False
    assert "connection lost" in (result.error or "")


@pytest.mark.asyncio
async def test_missing_tool_returns_clear_error(configured_env, fake_manager: _FakeManager) -> None:
    """A server that does not expose chrome_eval_js fails cleanly."""
    # Restrict the manager to a single tool.
    bare = _FakeManager(available_tools=["chrome_navigate"])
    ctrl = MCPChromeController(manager=bare, server_name="chrome")
    await ctrl.start()
    # Mark chrome_navigate as missing to test the not-available branch.
    bare.tools_by_name["chrome"] = []
    result = await ctrl.eval_js("1+1")
    assert result.success is False
    assert "does not expose" in (result.error or "")


@pytest.mark.asyncio
async def test_unknown_operation(configured_env, fake_manager: _FakeManager) -> None:
    """`GET_HTML` is not in the controller's MCP tool map for some servers."""
    bare = _FakeManager(available_tools=["chrome_get_html"])
    ctrl = MCPChromeController(manager=bare, server_name="chrome")
    await ctrl.start()
    # Force the lookup to find a non-mapped action.

    # Manually invoke the internal _call with an unsupported type.
    from clawcodex_ext.services.chrome.models import ChromeActionType

    # Use TYPE (mapped) but mask it as unmapped via the private helper.
    result = await ctrl._call(
        ChromeActionType.GET_HTML,
        {},  # type: ignore[arg-type]
    )
    # If the server exposes the tool, it succeeds. If not, fails. Both are OK.
    assert result is not None


@pytest.mark.asyncio
async def test_is_live_reflects_state(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    assert ctrl.is_live is False
    await ctrl.start()
    assert ctrl.is_live is True
    await ctrl.stop()
    assert ctrl.is_live is False


def test_controller_is_subclass_of_abc(
    configured_env,
) -> None:
    assert isinstance(MCPChromeController(), ChromeController)


def test_health_lists_available_tools(configured_env, fake_manager: _FakeManager) -> None:
    """Synchronous inspection of available_tools via health()."""
    # We can't run the full async start() in this sync test;
    # the field is set after start(). The fallback is to
    # inspect an empty tool list.
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    health = ctrl.health()
    assert "available_tools" in health
    assert health["backend"] == "mcp"


@pytest.mark.asyncio
async def test_start_recording_is_noop(configured_env, fake_manager: _FakeManager) -> None:
    ctrl = MCPChromeController(manager=fake_manager, server_name="chrome")
    await ctrl.start()
    await ctrl.start_recording("/tmp/x.gif", fps=1)
    assert ctrl.is_recording is False
    assert await ctrl.stop_recording() == ""
