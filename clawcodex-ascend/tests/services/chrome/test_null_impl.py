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

"""Tests for src/services/chrome/null_impl.py."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import pytest

from clawcodex_ext.services.chrome.base import ChromeController
from clawcodex_ext.services.chrome.models import ChromeActionType
from clawcodex_ext.services.chrome.null_impl import NullChromeController


def test_null_controller_is_a_chrome_controller() -> None:
    assert isinstance(NullChromeController(), ChromeController)


def test_null_controller_is_not_live() -> None:
    ctrl = NullChromeController()
    assert ctrl.is_live is False
    assert ctrl.is_recording is False
    assert ctrl.current_url == ""


@pytest.mark.asyncio
async def test_start_stop_are_idempotent() -> None:
    ctrl = NullChromeController()
    await ctrl.start()
    await ctrl.start()  # idempotent
    await ctrl.stop()
    await ctrl.stop()  # idempotent
    assert ctrl.is_live is False


@pytest.mark.asyncio
async def test_navigate_returns_unavailable_result() -> None:
    ctrl = NullChromeController()
    result = await ctrl.navigate("https://example.com")
    assert result.success is False
    assert "chrome controller not available" in (result.error or "")
    assert "pip install clawcodex[chrome]" in (result.error or "")
    assert result.action_type is ChromeActionType.NAVIGATE
    assert result.url == "https://example.com"


@pytest.mark.asyncio
async def test_click_returns_unavailable_result() -> None:
    result = await NullChromeController().click("button.submit")
    assert result.success is False
    assert result.action_type is ChromeActionType.CLICK
    assert "chrome controller not available" in (result.error or "")


@pytest.mark.asyncio
async def test_type_text_returns_unavailable_result() -> None:
    result = await NullChromeController().type_text("#q", "hello", clear_first=False)
    assert result.success is False
    assert result.action_type is ChromeActionType.TYPE


@pytest.mark.asyncio
async def test_select_option_returns_unavailable_result() -> None:
    result = await NullChromeController().select_option("select#x", "v")
    assert result.success is False
    assert result.action_type is ChromeActionType.SELECT


@pytest.mark.asyncio
async def test_hover_returns_unavailable_result() -> None:
    result = await NullChromeController().hover("a.link")
    assert result.success is False
    assert result.action_type is ChromeActionType.HOVER


@pytest.mark.asyncio
async def test_scroll_returns_unavailable_result() -> None:
    result = await NullChromeController().scroll(dx=10, dy=20)
    assert result.success is False
    assert result.action_type is ChromeActionType.SCROLL


@pytest.mark.asyncio
async def test_screenshot_returns_unavailable_result() -> None:
    result = await NullChromeController().screenshot(selector="div", full_page=False)
    assert result.success is False
    assert result.action_type is ChromeActionType.SCREENSHOT


@pytest.mark.asyncio
async def test_eval_js_returns_unavailable_result() -> None:
    result = await NullChromeController().eval_js("1 + 1")
    assert result.success is False
    assert result.action_type is ChromeActionType.EVAL_JS


@pytest.mark.asyncio
async def test_get_visible_text_returns_unavailable_result() -> None:
    result = await NullChromeController().get_visible_text()
    assert result.success is False
    assert result.action_type is ChromeActionType.GET_TEXT


@pytest.mark.asyncio
async def test_get_html_returns_unavailable_result() -> None:
    result = await NullChromeController().get_html()
    assert result.success is False
    assert result.action_type is ChromeActionType.GET_HTML


@pytest.mark.asyncio
async def test_start_recording_is_noop() -> None:
    ctrl = NullChromeController()
    await ctrl.start_recording("/tmp/x.gif", fps=2)
    assert ctrl.is_recording is False
    path = await ctrl.stop_recording()
    assert path == ""


def test_health_snapshot() -> None:
    ctrl = NullChromeController()
    health = ctrl.health()
    assert health == {"is_live": False, "is_recording": False, "url": ""}


def test_error_message_mentions_mcp_fallback() -> None:
    """The error should mention CHROME_MCP_URL as a fallback path."""
    ctrl = NullChromeController()
    # Force a failure to inspect the error string.
    import asyncio

    result = asyncio.run(ctrl.navigate("https://x"))
    assert "CHROME_MCP_URL" in (result.error or "")
