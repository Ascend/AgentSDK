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

from __future__ import annotations


# pylint: disable=C0412,E0611
import asyncio
from pathlib import Path
from typing import Any

from clawcodex_ext.permissions.types import ToolPermissionContext
from clawcodex_ext.providers.base import ChatResponse
from src.query.agent_loop_compat import run_query_as_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.renderers import ToolEvent
from clawcodex_ext.types.messages import UserMessage


class _TextOnlyProvider:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        return ChatResponse(
            content="agent smoke text ok",
            model="fake-agent-smoke",
            usage={"input_tokens": 3, "output_tokens": 4},
            finish_reason="stop",
            tool_uses=None,
        )

    def chat_stream(self, messages, tools=None, **kwargs):
        return iter(())

    def chat_stream_response(self, messages, tools=None, **kwargs):
        raise NotImplementedError


class _WriteOnceProvider:
    def __init__(self, target: Path, content: str = "agent-smoke-ok\n") -> None:
        self.target = target
        self.content = content
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(
                content="",
                model="fake-agent-smoke",
                usage={"input_tokens": 8, "output_tokens": 6},
                finish_reason="tool_use",
                tool_uses=[
                    {
                        "id": "call_write_smoke",
                        "name": "Write",
                        "input": {
                            "file_path": str(self.target),
                            "content": self.content,
                        },
                    }
                ],
            )
        return ChatResponse(
            content="agent smoke write done",
            model="fake-agent-smoke",
            usage={"input_tokens": 5, "output_tokens": 4},
            finish_reason="stop",
            tool_uses=None,
        )

    def chat_stream(self, messages, tools=None, **kwargs):
        return iter(())

    def chat_stream_response(self, messages, tools=None, **kwargs):
        raise NotImplementedError


def _run_smoke(provider, context: ToolContext, events: list[ToolEvent] | None = None):
    registry = build_default_registry(provider=provider, load_agent_tools=False)
    return asyncio.run(
        run_query_as_agent_loop(
            initial_messages=[UserMessage(content="run the deterministic smoke task")],
            provider=provider,
            tool_registry=registry,
            tool_context=context,
            system_prompt="You are a deterministic no-live-key CI smoke agent.",
            max_turns=3,
            on_event=events.append if events is not None else None,
        )
    )


def test_agent_smoke_text_response_without_provider_keys(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GLM_API_KEY", "ZHIPUAI_API_KEY"):
        monkeypatch.setenv(key, "")

    provider = _TextOnlyProvider()
    context = ToolContext(workspace_root=tmp_path)

    result = _run_smoke(provider, context)

    assert result.response_text == "agent smoke text ok"
    assert result.terminal is not None
    assert result.terminal.reason == "completed"
    assert len(provider.calls) == 1


def test_agent_smoke_executes_mocked_write_tool(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GLM_API_KEY", "ZHIPUAI_API_KEY"):
        monkeypatch.setenv(key, "")

    target = tmp_path / "agent_smoke.txt"
    provider = _WriteOnceProvider(target)
    context = ToolContext(workspace_root=tmp_path)
    events: list[ToolEvent] = []

    result = _run_smoke(provider, context, events)

    assert result.response_text == "agent smoke write done"
    assert target.read_text(encoding="utf-8") == "agent-smoke-ok\n"
    assert provider.calls == 2
    assert any(event.kind == "tool_use" and event.tool_name == "Write" for event in events)
    assert any(event.kind == "tool_result" and not event.is_error for event in events)


def test_agent_smoke_permission_denial_does_not_write(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GLM_API_KEY", "ZHIPUAI_API_KEY"):
        monkeypatch.setenv(key, "")

    target = tmp_path / "blocked.txt"
    provider = _WriteOnceProvider(target, content="should-not-write\n")
    context = ToolContext(
        workspace_root=tmp_path,
        permission_context=ToolPermissionContext(mode="dontAsk"),
    )
    events: list[ToolEvent] = []

    result = _run_smoke(provider, context, events)

    assert result.response_text == "agent smoke write done"
    assert not target.exists()
    denied = [event for event in events if event.kind == "tool_result" and event.is_error]
    assert denied
    assert "Permission denied" in (denied[0].error or denied[0].tool_output or "")
