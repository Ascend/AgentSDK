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

"""Focused tests for the Remote API query-loop runner."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from extensions.remote_api.errors import RemoteAPIError
from extensions.remote_api.runner import (
    RemoteAgentRunner,
    RemoteRunComplete,
    RemoteRunConfig,
    RemoteTextDelta,
    RemoteToolCall,
    RemoteToolResult,
    _session_id_for_run,
)
from clawcodex_ext.types.messages import AssistantMessage, UserMessage


class _AbortController:
    def __init__(self) -> None:
        self.signal = SimpleNamespace(aborted=False, reason=None)

    def abort(self, reason: str) -> None:
        self.signal.aborted = True
        self.signal.reason = reason


class _SdkContext:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _SessionId(str):
    pass


def _install_query_modules(
    monkeypatch: pytest.MonkeyPatch,
    agent_loop: Any,
) -> tuple[dict[str, Any], _AbortController]:
    """Install only the reconstructed-runtime shapes consumed by the runner."""

    abort = _AbortController()
    tool_context = SimpleNamespace(
        session_id=None,
        output_style_name=None,
        output_style_dir=None,
    )
    runtime = {
        "provider": object(),
        "tool_registry": object(),
        "tool_context": tool_context,
        "abort_controller": abort,
    }

    src = types.ModuleType("src")
    src.__path__ = []  # type: ignore[attr-defined]
    output_styles = types.ModuleType("src.outputStyles")
    output_styles.resolve_output_style = lambda *_args: SimpleNamespace(prompt="base prompt")
    query = types.ModuleType("src.query")
    query.__path__ = []  # type: ignore[attr-defined]
    compat = types.ModuleType("src.query.agent_loop_compat")
    compat.build_effective_system_prompt = lambda prompt, _context: prompt
    compat.run_query_as_agent_loop = agent_loop

    bootstrap_state = types.ModuleType("clawcodex_ext.bootstrap.state")
    bootstrap_state.SdkContext = _SdkContext
    bootstrap_state.SessionId = _SessionId

    @contextmanager
    def run_with_sdk_context(_context: _SdkContext):
        yield

    bootstrap_state.run_with_sdk_context = run_with_sdk_context

    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.outputStyles", output_styles)
    monkeypatch.setitem(sys.modules, "src.query", query)
    monkeypatch.setitem(sys.modules, "src.query.agent_loop_compat", compat)
    monkeypatch.setitem(sys.modules, "clawcodex_ext.bootstrap.state", bootstrap_state)
    monkeypatch.setattr("extensions.remote_api.runner._build_runtime", lambda _cfg: runtime)
    return runtime, abort


@pytest.mark.asyncio
async def test_run_aggregates_stream_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = RemoteAgentRunner(
        RemoteRunConfig(workspace=tmp_path),
        messages=[UserMessage(content="hello")],
        run_id="run_1",
    )

    async def fake_stream() -> AsyncIterator[Any]:
        yield RemoteTextDelta("first ")
        yield RemoteTextDelta("second")
        yield RemoteRunComplete(
            reason="success",
            response_text="",
            usage={"input_tokens": 1},
            messages=[
                UserMessage(content="hello"),
                AssistantMessage(content="first second"),
            ],
        )

    monkeypatch.setattr(runner, "stream", fake_stream)
    result = await runner.run()

    assert result.text == "first second"
    assert result.reason == "success"
    assert result.usage == {"input_tokens": 1}
    assert len(result.events) == 3


@pytest.mark.asyncio
async def test_stream_projects_text_tool_and_message_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_agent_loop(**kwargs: Any) -> Any:
        kwargs["on_text_chunk"]("hello")
        kwargs["on_event"](
            SimpleNamespace(
                kind="tool_use",
                tool_name="Read",
                tool_input={"path": "README.md"},
                tool_use_id="call_1",
            )
        )
        kwargs["on_event"](
            SimpleNamespace(
                kind="tool_result",
                tool_name="Read",
                tool_output="contents",
                tool_use_id="call_1",
                is_error=False,
                error=None,
            )
        )
        kwargs["on_message"](AssistantMessage(content="hello"))
        return SimpleNamespace(response_text="hello", usage={"output_tokens": 1})

    runtime, _abort = _install_query_modules(monkeypatch, fake_agent_loop)
    runner = RemoteAgentRunner(
        RemoteRunConfig(workspace=tmp_path, session_id="conversation/unsafe"),
        messages=[UserMessage(content="run")],
        instructions="extra rule",
        run_id="resp_1",
    )

    events = [event async for event in runner.stream()]

    assert isinstance(events[0], RemoteTextDelta)
    assert events[0].content == "hello"
    assert events[1] == RemoteToolCall("Read", {"path": "README.md"}, "call_1")
    assert events[2] == RemoteToolResult(
        "Read",
        {"output": "contents", "is_error": False},
        "call_1",
    )
    assert isinstance(events[-1], RemoteRunComplete)
    assert isinstance(events[-1].messages[-1], AssistantMessage)
    assert events[-1].messages[-1].content == "hello"
    assert runtime["tool_context"].session_id == "conversation_unsafe"


@pytest.mark.asyncio
async def test_runner_backpressure_preserves_more_chunks_than_queue_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunk_count = 300

    async def fake_agent_loop(**kwargs: Any) -> Any:
        for index in range(chunk_count):
            kwargs["on_text_chunk"](f"{index},")
        return SimpleNamespace(response_text="", usage={})

    _install_query_modules(monkeypatch, fake_agent_loop)
    runner = RemoteAgentRunner(
        RemoteRunConfig(workspace=tmp_path),
        messages=[UserMessage(content="run")],
        run_id="many_chunks",
    )

    events = [event async for event in runner.stream()]
    chunks = [event.content for event in events if isinstance(event, RemoteTextDelta)]

    assert len(chunks) == chunk_count
    assert chunks[0] == "0,"
    assert chunks[-1] == "299,"


@pytest.mark.asyncio
async def test_closing_stream_aborts_inflight_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker_started = threading.Event()
    runtime_ref: dict[str, Any] = {}

    async def fake_agent_loop(**kwargs: Any) -> Any:
        worker_started.set()
        kwargs["on_text_chunk"]("started")
        while not runtime_ref["abort"].signal.aborted:
            await asyncio.sleep(0.01)
        return SimpleNamespace(response_text="", usage={})

    _runtime, abort = _install_query_modules(monkeypatch, fake_agent_loop)
    runtime_ref["abort"] = abort
    runner = RemoteAgentRunner(
        RemoteRunConfig(workspace=tmp_path),
        messages=[UserMessage(content="run")],
        run_id="cancel_me",
    )

    stream = runner.stream()
    first = await stream.__anext__()
    assert first == RemoteTextDelta("started")
    assert worker_started.is_set()

    await stream.aclose()

    assert abort.signal.aborted is True
    assert abort.signal.reason == "remote_api_stream_closed"


@pytest.mark.asyncio
async def test_system_exit_is_exposed_as_remote_api_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_agent_loop(**_kwargs: Any) -> Any:
        raise SystemExit(7)

    _install_query_modules(monkeypatch, fake_agent_loop)
    runner = RemoteAgentRunner(
        RemoteRunConfig(workspace=tmp_path),
        messages=[UserMessage(content="run")],
    )

    with pytest.raises(RemoteAPIError, match="exit_code=7"):
        _ = [event async for event in runner.stream()]


@pytest.mark.asyncio
async def test_unexpected_agent_failure_is_logged_without_client_detail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_agent_loop(**_kwargs: Any) -> Any:
        raise RuntimeError("provider-secret-token")

    _install_query_modules(monkeypatch, fake_agent_loop)
    runner = RemoteAgentRunner(
        RemoteRunConfig(workspace=tmp_path),
        messages=[UserMessage(content="run")],
    )

    with (
        caplog.at_level(logging.ERROR, logger="extensions.remote_api.runner"),
        pytest.raises(RemoteAPIError) as raised,
    ):
        _ = [event async for event in runner.stream()]

    assert raised.value.detail == "agent run failed"
    assert "provider-secret-token" in caplog.text


def test_session_id_is_sanitized_and_bounded() -> None:
    assert _session_id_for_run("conversation/with spaces") == "conversation_with_spaces"
    assert len(_session_id_for_run("x" * 200)) == 128
