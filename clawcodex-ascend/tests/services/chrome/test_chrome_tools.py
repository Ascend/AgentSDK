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

"""Tests for the ``chrome_*`` tools produced by
:func:`src.services.chrome.build_chrome_tools`.

The tools are :class:`Tool` objects with a sync ``call``
that bridges to the async controller. We test:

* the seven tool names match the spec,
* each tool's ``call`` invokes the right controller method,
* a ``NullChromeController``-backed tool returns a
  ``ToolResult(is_error=True, ...)`` with the install-hint error,
* result rendering handles the four ``ChromeActionResult.data``
  shapes (None / bytes / dict / str).
"""
# pylint: disable=no-name-in-module, redefined-outer-name, reimported

from __future__ import annotations

import json
from typing import Any

import pytest

import clawcodex_ext.services.chrome.factory as factory_module
from clawcodex_ext.services.chrome import _reset_chrome_singleton, build_chrome_tools
from clawcodex_ext.services.chrome.models import ChromeActionResult
from clawcodex_ext.services.chrome.null_impl import NullChromeController
from src.tool_system.build_tool import Tool
from src.tool_system.context import ToolContext
from src.tool_system.protocol import ToolResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_singleton():
    _reset_chrome_singleton()
    yield
    _reset_chrome_singleton()


@pytest.fixture
def tools():
    return build_chrome_tools()


@pytest.fixture
def fake_controller(monkeypatch: pytest.MonkeyPatch):
    """Install a controller that records calls and returns canned
    results. Bypasses the singleton so each test sees a fresh
    instance.
    """

    class _RecordingController(NullChromeController):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
            self._next: ChromeActionResult = ChromeActionResult(success=True, data="ok")

        def set_next(self, result: ChromeActionResult) -> None:
            self._next = result

        async def navigate(self, url: str) -> ChromeActionResult:
            self.calls.append(("navigate", (url,), {}))
            return self._next

        async def click(self, selector: str) -> ChromeActionResult:
            self.calls.append(("click", (selector,), {}))
            return self._next

        async def type_text(self, selector: str, text: str, *, clear_first: bool = True) -> ChromeActionResult:
            self.calls.append(("type_text", (selector, text), {"clear_first": clear_first}))
            return self._next

        async def select_option(self, selector: str, value: str) -> ChromeActionResult:
            self.calls.append(("select_option", (selector, value), {}))
            return self._next

        async def hover(self, selector: str) -> ChromeActionResult:
            self.calls.append(("hover", (selector,), {}))
            return self._next

        async def scroll(self, *, dx: int = 0, dy: int = 1) -> ChromeActionResult:
            self.calls.append(("scroll", (), {"dx": dx, "dy": dy}))
            return self._next

        async def screenshot(self, selector: str | None = None, *, full_page: bool = True) -> ChromeActionResult:
            self.calls.append(("screenshot", (), {"selector": selector, "full_page": full_page}))
            return self._next

        async def eval_js(self, script: str) -> ChromeActionResult:
            self.calls.append(("eval_js", (script,), {}))
            return self._next

        async def get_visible_text(self) -> ChromeActionResult:
            self.calls.append(("get_visible_text", (), {}))
            return self._next

        async def get_html(self) -> ChromeActionResult:
            self.calls.append(("get_html", (), {}))
            return self._next

    ctrl = _RecordingController()
    # Bypass the singleton: monkey-patch the lazy-resolver so
    # the tools use our recorder.
    monkeypatch.setattr(factory_module, "_cached_controller", ctrl, raising=False)
    _reset_chrome_singleton()
    # Patch the getter that tools actually call.
    monkeypatch.setattr(
        factory_module,
        "_get_or_build_controller",
        lambda *, prefer="auto": ctrl,
    )
    return ctrl


def _ctx() -> ToolContext:
    """A minimal ToolContext for unit tests."""
    from pathlib import Path

    return ToolContext(
        workspace_root=Path("."),
        cwd=Path("."),
    )


# ---------------------------------------------------------------------------
# Tool factory shape
# ---------------------------------------------------------------------------


def test_build_chrome_tools_returns_seven(tools: list[Tool]) -> None:
    assert len(tools) == 7


def test_tool_names_match_spec(tools: list[Tool]) -> None:
    expected = {
        "chrome_navigate",
        "chrome_click",
        "chrome_type",
        "chrome_select",
        "chrome_screenshot",
        "chrome_eval_js",
        "chrome_get_text",
    }
    actual = {t.name for t in tools}
    assert actual == expected


def test_each_tool_is_a_tool_instance(tools: list[Tool]) -> None:
    for t in tools:
        assert isinstance(t, Tool)


def test_each_tool_has_input_schema(tools: list[Tool]) -> None:
    for t in tools:
        assert isinstance(t.input_schema, dict)
        assert t.input_schema.get("type") == "object"
        assert "properties" in t.input_schema


def test_each_tool_has_call_hook(tools: list[Tool]) -> None:
    for t in tools:
        assert callable(t.call)


def test_each_tool_is_marked_destructive(tools: list[Tool]) -> None:
    """Browser ops can mutate state; the agent loop should
    treat them as such.
    """
    for t in tools:
        assert t.is_destructive({}) is True


def test_each_tool_is_marked_not_concurrency_safe(tools: list[Tool]) -> None:
    """Browser state is shared; tools must run serially per session."""
    for t in tools:
        assert t.is_concurrency_safe({}) is False


def test_each_tool_is_marked_open_world(tools: list[Tool]) -> None:
    for t in tools:
        assert t.is_open_world({}) is True


def test_each_tool_is_enabled(tools: list[Tool]) -> None:
    for t in tools:
        assert t.is_enabled() is True


def test_navigate_requires_url(tools: list[Tool]) -> None:
    nav = next(t for t in tools if t.name == "chrome_navigate")
    assert "url" in nav.input_schema["required"]


def test_type_requires_selector_and_text(tools: list[Tool]) -> None:
    type_tool = next(t for t in tools if t.name == "chrome_type")
    assert set(type_tool.input_schema["required"]) == {"selector", "text"}


def test_select_requires_selector_and_value(tools: list[Tool]) -> None:
    sel = next(t for t in tools if t.name == "chrome_select")
    assert set(sel.input_schema["required"]) == {"selector", "value"}


def test_eval_js_requires_script(tools: list[Tool]) -> None:
    ev = next(t for t in tools if t.name == "chrome_eval_js")
    assert "script" in ev.input_schema["required"]


def test_screenshot_arguments_are_optional(tools: list[Tool]) -> None:
    """screenshot's selector and full_page both default."""
    shot = next(t for t in tools if t.name == "chrome_screenshot")
    assert "required" not in shot.input_schema or not shot.input_schema["required"]


def test_get_text_takes_no_args(tools: list[Tool]) -> None:
    get_text = next(t for t in tools if t.name == "chrome_get_text")
    assert get_text.input_schema.get("properties", {}) == {}
    assert not get_text.input_schema.get("required", [])


# ---------------------------------------------------------------------------
# Per-tool dispatch
# ---------------------------------------------------------------------------


def test_navigate_call_dispatches(tools: list[Tool], fake_controller) -> None:
    nav = next(t for t in tools if t.name == "chrome_navigate")
    fake_controller.set_next(ChromeActionResult(success=True, data="https://example.com", url="https://example.com"))
    result = nav.call({"url": "https://example.com"}, _ctx())
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert fake_controller.calls[0][0] == "navigate"


def test_click_call_dispatches(tools: list[Tool], fake_controller) -> None:
    click = next(t for t in tools if t.name == "chrome_click")
    fake_controller.set_next(ChromeActionResult(success=True, data="button.submit"))
    result = click.call({"selector": "button.submit"}, _ctx())
    assert result.is_error is False
    assert fake_controller.calls[0][0] == "click"
    assert fake_controller.calls[0][1] == ("button.submit",)


def test_type_call_dispatches(tools: list[Tool], fake_controller) -> None:
    type_tool = next(t for t in tools if t.name == "chrome_type")
    fake_controller.set_next(ChromeActionResult(success=True, data="hi"))
    result = type_tool.call({"selector": "#q", "text": "hi", "clear_first": False}, _ctx())
    assert result.is_error is False
    op, args, kwargs = fake_controller.calls[0]
    assert op == "type_text"
    assert args == ("#q", "hi")
    assert kwargs == {"clear_first": False}


def test_type_default_clears_first(tools: list[Tool], fake_controller) -> None:
    """Omitting ``clear_first`` should default to True."""
    type_tool = next(t for t in tools if t.name == "chrome_type")
    type_tool.call({"selector": "#q", "text": "hi"}, _ctx())
    _, _, kwargs = fake_controller.calls[0]
    assert kwargs["clear_first"] is True


def test_select_call_dispatches(tools: list[Tool], fake_controller) -> None:
    sel = next(t for t in tools if t.name == "chrome_select")
    fake_controller.set_next(ChromeActionResult(success=True, data="v"))
    sel.call({"selector": "select#x", "value": "v"}, _ctx())
    op, args, _ = fake_controller.calls[0]
    assert op == "select_option"
    assert args == ("select#x", "v")


def test_screenshot_call_dispatches_full_page(tools: list[Tool], fake_controller) -> None:
    shot = next(t for t in tools if t.name == "chrome_screenshot")
    fake_controller.set_next(ChromeActionResult(success=True, data=b"\x89PNG\r\n\x1a\nX"))
    # Provide a tempdir so screenshot bytes get persisted.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAWCODEX_CHROME_SCREENSHOT_DIR", "/tmp")
        shot.call({}, _ctx())
    op, _, kwargs = fake_controller.calls[0]
    assert op == "screenshot"
    assert kwargs["full_page"] is True
    assert kwargs["selector"] is None


def test_screenshot_with_selector(tools: list[Tool], fake_controller) -> None:
    shot = next(t for t in tools if t.name == "chrome_screenshot")
    fake_controller.set_next(ChromeActionResult(success=True, data=b"\x89PNG\r\n\x1a\nY"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAWCODEX_CHROME_SCREENSHOT_DIR", "/tmp")
        shot.call({"selector": "#x", "full_page": False}, _ctx())
    _, _, kwargs = fake_controller.calls[0]
    assert kwargs["selector"] == "#x"
    assert kwargs["full_page"] is False


def test_eval_js_call_dispatches(tools: list[Tool], fake_controller) -> None:
    ev = next(t for t in tools if t.name == "chrome_eval_js")
    fake_controller.set_next(ChromeActionResult(success=True, data='{"r": 1}'))
    ev.call({"script": "({r:1})"}, _ctx())
    op, args, _ = fake_controller.calls[0]
    assert op == "eval_js"
    assert args == ("({r:1})",)


def test_get_text_call_dispatches(tools: list[Tool], fake_controller) -> None:
    gt = next(t for t in tools if t.name == "chrome_get_text")
    fake_controller.set_next(ChromeActionResult(success=True, data="hello"))
    gt.call({}, _ctx())
    assert fake_controller.calls[0][0] == "get_visible_text"


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------


def test_string_result_renders_as_text(tools: list[Tool], fake_controller) -> None:
    gt = next(t for t in tools if t.name == "chrome_get_text")
    fake_controller.set_next(ChromeActionResult(success=True, data="hello world"))
    result = gt.call({}, _ctx())
    assert result.output == "hello world"
    assert result.is_error is False


def test_dict_result_renders_as_json(tools: list[Tool], fake_controller) -> None:
    ev = next(t for t in tools if t.name == "chrome_eval_js")
    fake_controller.set_next(ChromeActionResult(success=True, data={"answer": 42}, url="https://x"))
    result = ev.call({"script": "({answer:42})"}, _ctx())
    payload = json.loads(result.output)
    assert payload["data"] == {"answer": 42}
    assert payload["url"] == "https://x"


def test_none_result_renders_metadata(tools: list[Tool], fake_controller) -> None:
    nav = next(t for t in tools if t.name == "chrome_navigate")
    fake_controller.set_next(ChromeActionResult(success=True, data=None, url="https://example.com"))
    result = nav.call({"url": "https://example.com"}, _ctx())
    payload = json.loads(result.output)
    assert payload["url"] == "https://example.com"


def test_bytes_result_persists_to_disk(tools: list[Tool], fake_controller, tmp_path) -> None:
    """Screenshot bytes should be persisted to a temp file."""
    shot = next(t for t in tools if t.name == "chrome_screenshot")
    payload = b"\x89PNG\r\n\x1a\nFAKE_BYTES"
    fake_controller.set_next(ChromeActionResult(success=True, data=payload, url="https://x"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLAWCODEX_CHROME_SCREENSHOT_DIR", str(tmp_path))
        result = shot.call({}, _ctx())
    payload_out = json.loads(result.output)
    assert "screenshot_path" in payload_out
    with open(payload_out["screenshot_path"], "rb") as fh:
        written = fh.read()
    assert written == payload
    assert payload_out["size_bytes"] == len(payload)


def test_failed_result_renders_as_error(tools: list[Tool], fake_controller) -> None:
    nav = next(t for t in tools if t.name == "chrome_navigate")
    fake_controller.set_next(ChromeActionResult(success=False, error="boom"))
    result = nav.call({"url": "https://x"}, _ctx())
    assert result.is_error is True
    assert "boom" in str(result.output)


# ---------------------------------------------------------------------------
# Null backend integration
# ---------------------------------------------------------------------------


def test_null_backend_returns_install_hint_error() -> None:
    """When no backend is configured, chrome_navigate must surface
    a clear install message so the agent knows what to do.
    """
    _reset_chrome_singleton()
    # Force Null by patching both helpers to return Null.
    import clawcodex_ext.services.chrome.factory as fm

    original_pw = fm._build_playwright_controller
    original_mcp = fm._build_mcp_controller

    def _pw() -> Any:
        return NullChromeController()

    def _mcp() -> Any:
        return NullChromeController()

    fm._build_playwright_controller = _pw  # type: ignore[assignment]
    fm._build_mcp_controller = _mcp  # type: ignore[assignment]
    try:
        tools = build_chrome_tools()
        nav = next(t for t in tools if t.name == "chrome_navigate")
        result = nav.call({"url": "https://example.com"}, _ctx())
        assert result.is_error is True
        assert "chrome controller not available" in str(result.output)
    finally:
        fm._build_playwright_controller = original_pw  # type: ignore[assignment]
        fm._build_mcp_controller = original_mcp  # type: ignore[assignment]


def test_build_chrome_tools_does_not_raise_when_no_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_chrome_tools`` should always succeed — the actual
    backend resolution is lazy.
    """
    monkeypatch.delenv("CHROME_MCP_URL", raising=False)
    monkeypatch.delenv("CHROME_MCP_COMMAND", raising=False)
    tools = build_chrome_tools()
    assert len(tools) == 7


# ---------------------------------------------------------------------------
# Map-and-describe helpers
# ---------------------------------------------------------------------------


def test_each_tool_provides_description(tools: list[Tool]) -> None:
    for t in tools:
        desc = t.description({})
        assert isinstance(desc, str)
        assert desc  # non-empty


def test_each_tool_provides_prompt(tools: list[Tool]) -> None:
    for t in tools:
        prompt = t.prompt()
        assert isinstance(prompt, str)
        assert prompt


def test_each_tool_provides_user_facing_name(tools: list[Tool]) -> None:
    for t in tools:
        name = t.user_facing_name({})
        assert isinstance(name, str)
        assert name


def test_each_tool_provides_tool_use_summary(tools: list[Tool]) -> None:
    """``get_tool_use_summary`` may be None for some tools, but
    chrome tools should each provide a short summary string.
    """
    for t in tools:
        summary = t.get_tool_use_summary
        assert summary is not None
        # Smoke: summary returns a string for an empty input.
        assert isinstance(summary({}), str)


def test_each_tool_provides_activity_description(tools: list[Tool]) -> None:
    for t in tools:
        activity = t.get_activity_description
        assert activity is not None
        assert isinstance(activity({}), str)
