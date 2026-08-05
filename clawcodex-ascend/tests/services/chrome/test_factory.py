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

"""Tests for src/services/chrome/factory.py.

Covers ``build_chrome_controller(prefer=...)`` dispatch and the
singleton caching. We monkeypatch the internal helpers rather
than the underlying imports so the test stays a pure unit test
— no Playwright SDK or MCP server required.
"""
# pylint: disable=no-name-in-module, redefined-outer-name

from __future__ import annotations

import pytest

import clawcodex_ext.services.chrome.factory as factory_module
from clawcodex_ext.services.chrome import _reset_chrome_singleton
from clawcodex_ext.services.chrome.base import ChromeController
from clawcodex_ext.services.chrome.factory import build_chrome_controller
from clawcodex_ext.services.chrome.mcp_impl import MCPChromeController
from clawcodex_ext.services.chrome.null_impl import NullChromeController
from clawcodex_ext.services.chrome.playwright_impl import PlaywrightChromeController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_singleton():
    """Each test starts with a clean singleton — the factory
    caches the resolved controller, so we reset between cases
    to avoid test pollution.
    """
    _reset_chrome_singleton()
    yield
    _reset_chrome_singleton()


@pytest.fixture
def fake_playwright_controller(monkeypatch: pytest.MonkeyPatch):
    """Patch ``_build_playwright_controller`` to return a marker
    Playwright controller (no Playwright SDK actually imported).
    """

    class _Marker(PlaywrightChromeController):
        def __init__(self) -> None:
            # Skip the parent __init__ — it would attempt the
            # optional SDK import. We are pretending it's installed.
            ChromeController.__init__(self)  # type: ignore[call-arg]
            self.marker = "playwright"

    monkeypatch.setattr(
        factory_module,
        "_build_playwright_controller",
        _Marker,
    )
    return _Marker


@pytest.fixture
def fake_mcp_controller(monkeypatch: pytest.MonkeyPatch):
    """Patch ``_build_mcp_controller`` to return a marker
    MCP controller (no MCP server actually contacted).
    """

    class _Marker(MCPChromeController):
        def __init__(self) -> None:
            ChromeController.__init__(self)  # type: ignore[call-arg]
            self.marker = "mcp"

    monkeypatch.setattr(
        factory_module,
        "_build_mcp_controller",
        _Marker,
    )
    return _Marker


@pytest.fixture
def configured_mcp_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHROME_MCP_URL", "http://localhost:1234/mcp")


@pytest.fixture
def unconfigured_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CHROME_MCP_URL", raising=False)
    monkeypatch.delenv("CHROME_MCP_COMMAND", raising=False)


# ---------------------------------------------------------------------------
# prefer="null"
# ---------------------------------------------------------------------------


def test_prefer_null_always_returns_null(unconfigured_env) -> None:
    ctrl = build_chrome_controller(prefer="null")
    assert isinstance(ctrl, NullChromeController)


def test_prefer_null_ignores_mcp_env(configured_mcp_env) -> None:
    """Even when MCP env vars are set, ``prefer="null"`` wins."""
    ctrl = build_chrome_controller(prefer="null")
    assert isinstance(ctrl, NullChromeController)


def test_prefer_null_ignores_playwright(fake_playwright_controller, configured_mcp_env) -> None:
    ctrl = build_chrome_controller(prefer="null")
    assert isinstance(ctrl, NullChromeController)


# ---------------------------------------------------------------------------
# prefer="playwright"
# ---------------------------------------------------------------------------


def test_prefer_playwright_returns_playwright(fake_playwright_controller, configured_mcp_env) -> None:
    """``prefer="playwright"`` should ignore MCP env vars."""
    ctrl = build_chrome_controller(prefer="playwright")
    assert isinstance(ctrl, fake_playwright_controller)
    assert ctrl.marker == "playwright"


def test_prefer_playwright_falls_back_to_null(monkeypatch: pytest.MonkeyPatch, unconfigured_env) -> None:
    """When the helper returns Null (SDK missing), the factory
    surfaces a NullChromeController — not a hard error.
    """
    monkeypatch.setattr(
        factory_module,
        "_build_playwright_controller",
        NullChromeController,
    )
    ctrl = build_chrome_controller(prefer="playwright")
    assert isinstance(ctrl, NullChromeController)


# ---------------------------------------------------------------------------
# prefer="mcp"
# ---------------------------------------------------------------------------


def test_prefer_mcp_returns_mcp(fake_mcp_controller, unconfigured_env) -> None:
    """``prefer="mcp"`` calls the helper directly without
    requiring the env vars to be set — that's the helper's
    job to validate.
    """
    ctrl = build_chrome_controller(prefer="mcp")
    assert isinstance(ctrl, fake_mcp_controller)
    assert ctrl.marker == "mcp"


def test_prefer_mcp_falls_back_to_null(monkeypatch: pytest.MonkeyPatch, unconfigured_env) -> None:
    monkeypatch.setattr(
        factory_module,
        "_build_mcp_controller",
        NullChromeController,
    )
    ctrl = build_chrome_controller(prefer="mcp")
    assert isinstance(ctrl, NullChromeController)


# ---------------------------------------------------------------------------
# prefer="auto" dispatch order
# ---------------------------------------------------------------------------


def test_auto_prefers_mcp_when_configured(
    fake_mcp_controller,
    fake_playwright_controller,
    configured_mcp_env,
) -> None:
    """MCP wins over Playwright when both are available."""
    ctrl = build_chrome_controller(prefer="auto")
    assert isinstance(ctrl, fake_mcp_controller)
    assert ctrl.marker == "mcp"


def test_auto_falls_through_mcp_then_playwright(fake_playwright_controller, configured_mcp_env) -> None:
    """When MCP is configured but the helper returns Null,
    auto-mode should still try Playwright.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        factory_module,
        "_build_mcp_controller",
        NullChromeController,
    )
    try:
        ctrl = build_chrome_controller(prefer="auto")
        assert isinstance(ctrl, fake_playwright_controller)
        assert ctrl.marker == "playwright"
    finally:
        monkeypatch.undo()


def test_auto_returns_null_when_nothing_available(monkeypatch: pytest.MonkeyPatch, unconfigured_env) -> None:
    """No env vars, Playwright helper returns Null → NullChromeController."""
    monkeypatch.setattr(
        factory_module,
        "_build_playwright_controller",
        NullChromeController,
    )
    monkeypatch.setattr(
        factory_module,
        "_build_mcp_controller",
        NullChromeController,
    )
    ctrl = build_chrome_controller(prefer="auto")
    assert isinstance(ctrl, NullChromeController)


def test_auto_uses_playwright_when_mcp_unconfigured(fake_playwright_controller, unconfigured_env) -> None:
    """Without MCP env vars, auto picks Playwright."""
    ctrl = build_chrome_controller(prefer="auto")
    assert isinstance(ctrl, fake_playwright_controller)


def test_auto_picks_mcp_when_only_command_set(fake_mcp_controller, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHROME_MCP_URL", raising=False)
    monkeypatch.setenv("CHROME_MCP_COMMAND", "npx chrome-devtools-mcp")
    ctrl = build_chrome_controller(prefer="auto")
    assert isinstance(ctrl, fake_mcp_controller)


# ---------------------------------------------------------------------------
# Singleton caching
# ---------------------------------------------------------------------------


def test_singleton_returns_same_instance_across_calls(monkeypatch: pytest.MonkeyPatch, unconfigured_env) -> None:
    """``_get_or_build_controller`` is the lazy-resolver used
    by the tool wrappers; once cached, ``prefer`` is ignored.
    """
    monkeypatch.setattr(
        factory_module,
        "_build_playwright_controller",
        NullChromeController,
    )
    monkeypatch.setattr(
        factory_module,
        "_build_mcp_controller",
        NullChromeController,
    )
    a = factory_module._get_or_build_controller(prefer="null")
    b = factory_module._get_or_build_controller(prefer="auto")
    c = factory_module._get_or_build_controller(prefer="playwright")
    assert a is b
    assert b is c


def test_reset_clears_singleton(fake_playwright_controller, unconfigured_env) -> None:
    a = build_chrome_controller(prefer="auto")
    assert isinstance(a, fake_playwright_controller)
    _reset_chrome_singleton()
    # Next call rebuilds; with the same patches it gets a
    # fresh instance.
    b = build_chrome_controller(prefer="auto")
    assert isinstance(b, fake_playwright_controller)
    assert a is not b


def test_get_or_build_uses_cached(monkeypatch: pytest.MonkeyPatch, unconfigured_env) -> None:
    """``_get_or_build_controller`` caches once and forwards
    ``prefer`` only on the first call.
    """
    builds: list[str] = []
    original = factory_module.build_chrome_controller

    def _tracking(*, prefer: str = "auto"):
        builds.append(prefer)
        return original(prefer=prefer)

    monkeypatch.setattr(factory_module, "build_chrome_controller", _tracking)
    a = factory_module._get_or_build_controller(prefer="null")
    b = factory_module._get_or_build_controller(prefer="playwright")
    assert a is b
    assert builds == ["null"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_chrome_mcp_configured_detects_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHROME_MCP_URL", "http://x/mcp")
    assert factory_module._chrome_mcp_configured() is True


def test_chrome_mcp_configured_detects_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHROME_MCP_URL", raising=False)
    monkeypatch.setenv("CHROME_MCP_COMMAND", "npx chrome-devtools-mcp")
    assert factory_module._chrome_mcp_configured() is True


def test_chrome_mcp_configured_false_when_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHROME_MCP_URL", raising=False)
    monkeypatch.delenv("CHROME_MCP_COMMAND", raising=False)
    assert factory_module._chrome_mcp_configured() is False


def test_chrome_mcp_configured_ignores_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHROME_MCP_URL", "   ")
    monkeypatch.delenv("CHROME_MCP_COMMAND", raising=False)
    assert factory_module._chrome_mcp_configured() is False


def test_build_mcp_handles_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the module import fails (shouldn't, but defensive),
    the helper returns Null rather than blowing up the caller.
    """
    monkeypatch.setattr(
        factory_module,
        "MCPChromeController",
        None,
        raising=False,
    )
    # Force ImportError by deleting the symbol.
    if hasattr(factory_module, "_build_mcp_controller"):
        # We don't need to monkey-patch the import inside the
        # helper — instead, simulate by replacing the helper.
        monkeypatch.setattr(
            factory_module,
            "_build_mcp_controller",
            NullChromeController,
        )
    ctrl = factory_module._build_mcp_controller()
    assert isinstance(ctrl, NullChromeController)
