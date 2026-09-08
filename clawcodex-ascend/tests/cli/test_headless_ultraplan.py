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

"""Headless (non-interactive) /ultraplan and mcp__* tool dispatch tests.

Regression coverage for two defects in the headless path:
1. ``/ultraplan`` (and the ``/ultra`` / ``/up`` aliases) executed through the
   sync command dispatcher used to crash with "'coroutine' object has no
   attribute 'value'" because the async body leaked into the sync slot.
2. Tools registered by ``RuntimeContext.build`` — notably the MCP-wrapped
   ``mcp__<server>__<tool>`` tools — were dropped when headless rebuilt a
   default-only registry, making them undispatchable in non-interactive
   sessions. Headless now reuses a per-run copy of the runtime registry.
"""

from __future__ import annotations

import io

import pytest
from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.tool_system.build_tool import McpInfo, ToolResult, build_tool
from src.entrypoints import HeadlessOptions, run_headless


class _FakeProvider:
    """Minimal stand-in for an LLM provider (same shape as test_headless_cli)."""

    def __init__(self, api_key: str, base_url=None, model=None, *, responses=None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or "fake-model"
        self._responses = list(responses or [])

    def chat(self, messages, tools=None, **kwargs):
        if not self._responses:
            raise AssertionError("FakeProvider ran out of scripted responses")
        return self._responses.pop(0)

    async def chat_async(self, messages, tools=None, **kwargs):
        return self.chat(messages, tools=tools, **kwargs)

    def chat_stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError


class _FakeRegistry:
    def list_tools(self):
        return []


@pytest.fixture
def fake_wiring(monkeypatch):
    """Patch provider/tool wiring with fakes that require no API key."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    scripted_responses: list[ChatResponse] = []

    def _fake_get_provider_class(_name):
        def _factory(api_key, base_url=None, model=None, **_kwargs):
            return _FakeProvider(api_key, base_url=base_url, model=model, responses=list(scripted_responses))

        return _factory

    monkeypatch.setattr(
        ext_headless,
        "get_provider_class",
        _fake_get_provider_class,
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "get_provider_config",
        lambda _name: {
            "api_key": "test-key",
            "base_url": None,
            "default_model": "fake-model",
        },
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "get_default_provider",
        lambda: "anthropic",
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "build_default_registry",
        lambda provider=None: _FakeRegistry(),
        raising=False,
    )

    return scripted_responses


def _text_response(text: str) -> ChatResponse:
    return ChatResponse(
        content=text,
        model="fake-model",
        usage={"input_tokens": 5, "output_tokens": len(text.split())},
        finish_reason="end_turn",
        tool_uses=None,
    )


# ---------------------------------------------------------------------------
# /ultraplan through the headless non-interactive dispatcher (defect 1)
# ---------------------------------------------------------------------------


def test_headless_ultraplan_ls_succeeds(fake_wiring, tmp_path, monkeypatch):
    """``/ultraplan ls`` in headless mode exits 0 and prints its result.

    Regression: this used to crash with ``AttributeError`` ('coroutine'
    object has no attribute 'value') and exit 1.
    """
    monkeypatch.setenv("CLAWCODEX_ULTRAPLAN_DIR", str(tmp_path / "ultraplan"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="/ultraplan ls",
            output_format="text",
            stdout=stdout,
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code == 0, f"stderr: {stderr.getvalue()}"
    assert "No ultraplan plans found." in stdout.getvalue()
    assert "AttributeError" not in stderr.getvalue()


@pytest.mark.parametrize("alias", ["ultra", "up"])
def test_headless_ultraplan_alias_succeeds(fake_wiring, tmp_path, monkeypatch, alias):
    """``/ultra ls`` and ``/up ls`` route to the ultraplan command."""
    monkeypatch.setenv("CLAWCODEX_ULTRAPLAN_DIR", str(tmp_path / "ultraplan"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt=f"/{alias} ls",
            output_format="text",
            stdout=stdout,
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code == 0, f"stderr: {stderr.getvalue()}"
    assert "No ultraplan plans found." in stdout.getvalue()


def test_headless_ultraplan_create_gate_off_returns_friendly_message(
    fake_wiring,
    tmp_path,
    monkeypatch,
):
    """A disabled LLM planner yields a friendly message, not a raw crash."""
    monkeypatch.setenv("ULTRAPLAN_LLM_PLANNER", "off")
    monkeypatch.setenv("CLAWCODEX_ULTRAPLAN_DIR", str(tmp_path / "ultraplan"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="/ultraplan create build a rocket",
            output_format="text",
            stdout=stdout,
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code == 0, f"stderr: {stderr.getvalue()}"
    assert "ULTRAPLAN_LLM_PLANNER is disabled" in stdout.getvalue()
    assert "Traceback" not in stdout.getvalue()
    assert "Traceback" not in stderr.getvalue()


# ---------------------------------------------------------------------------
# mcp__* tool dispatch in a headless session (defect 2)
# ---------------------------------------------------------------------------


def test_headless_dispatches_mcp_tool_from_forwarded_registry(
    fake_wiring,
    tmp_path,
    monkeypatch,
):
    """A tool named ``mcp__<server>__<tool>`` present in the forwarded
    RuntimeContext registry is dispatchable in a non-interactive session.
    """
    # Per-server MCP wrappers are "deferred" tools: under tool-search mode
    # they stay out of the request tool list until discovered. Standard mode
    # (ENABLE_TOOL_SEARCH=false) loads them inline — the deployment shape
    # where direct ``mcp__*`` invocation is expected to work.
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "false")
    from src.tool_system.registry import ToolRegistry

    captured: dict[str, object] = {}

    def _ping(input_: dict, context) -> ToolResult:
        del context
        captured["input"] = input_
        return ToolResult(
            name="mcp__demo__ping",
            output={"reply": "pong", "echo": input_.get("message", "")},
        )

    ping_tool = build_tool(
        name="mcp__demo__ping",
        description="Ping the demo MCP server.",
        input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        call=_ping,
        is_mcp=True,
        mcp_info=McpInfo(server_name="demo", tool_name="ping"),
    )
    registry = ToolRegistry([ping_tool])
    fake_wiring.extend(
        [
            ChatResponse(
                content="",
                model="fake-model",
                usage={"input_tokens": 3, "output_tokens": 1},
                finish_reason="tool_use",
                tool_uses=[
                    {
                        "id": "call_ping",
                        "name": "mcp__demo__ping",
                        "input": {"message": "ping"},
                    }
                ],
            ),
            _text_response("The ping tool replied: pong."),
        ]
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="call the ping tool and report what it says",
            output_format="text",
            stdout=stdout,
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
            skip_permissions=True,
            tool_registry=registry,
        )
    )

    assert code == 0, f"stderr: {stderr.getvalue()}"
    assert "The ping tool replied: pong." in stdout.getvalue()
    assert captured.get("input") == {"message": "ping"}
