#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Focused tests for Chat completion and Remote API service orchestration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from extensions.remote_api import response_payloads
from extensions.remote_api.errors import RemoteAPIError
from extensions.remote_api.runner import (
    RemoteRunComplete,
    RemoteRunResult,
    RemoteTextDelta,
)
from extensions.remote_api.core import (
    API_MODEL_NAME,
    RemoteAPIConfig,
    RemoteAPIService,
    _response_metadata,
)
from clawcodex_ext.types.content_blocks import ToolResultBlock, ToolUseBlock
from clawcodex_ext.types.messages import AssistantMessage, UserMessage


def _decode_sse(frame: str) -> Any:
    data = "\n".join(line.removeprefix("data: ") for line in frame.splitlines() if line.startswith("data: "))
    return data if data == "[DONE]" else json.loads(data)


class _RunnerStub:
    calls: list["_RunnerStub"] = []
    run_result = RemoteRunResult(
        text="assistant reply",
        reason="success",
        usage={"input_tokens": 2, "output_tokens": 3},
        messages=[AssistantMessage(content="assistant reply")],
        events=[],
    )
    stream_events: list[Any] = [
        RemoteTextDelta("assistant "),
        RemoteTextDelta("reply"),
        RemoteRunComplete(
            reason="success",
            response_text="assistant reply",
            usage={"input_tokens": 2, "output_tokens": 3},
            messages=[AssistantMessage(content="assistant reply")],
        ),
    ]
    run_delay = 0.0
    stream_closed = False

    def __init__(
        self,
        config: Any,
        *,
        messages: list[Any],
        instructions: str = "",
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.messages = messages
        self.instructions = instructions
        self.run_id = run_id
        type(self).calls.append(self)

    async def run(self) -> RemoteRunResult:
        if type(self).run_delay:
            await asyncio.sleep(type(self).run_delay)
        return type(self).run_result

    async def stream(self) -> AsyncIterator[Any]:
        try:
            for event in type(self).stream_events:
                if isinstance(event, BaseException):
                    raise event
                yield event
        finally:
            type(self).stream_closed = True


@pytest.fixture(autouse=True)
def _reset_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    _RunnerStub.calls = []
    _RunnerStub.run_delay = 0.0
    _RunnerStub.stream_closed = False
    _RunnerStub.run_result = RemoteRunResult(
        text="assistant reply",
        reason="success",
        usage={"input_tokens": 2, "output_tokens": 3},
        messages=[AssistantMessage(content="assistant reply")],
        events=[],
    )
    _RunnerStub.stream_events = [
        RemoteTextDelta("assistant "),
        RemoteTextDelta("reply"),
        RemoteRunComplete(
            reason="success",
            response_text="assistant reply",
            usage={"input_tokens": 2, "output_tokens": 3},
            messages=[AssistantMessage(content="assistant reply")],
        ),
    ]
    monkeypatch.setattr("extensions.remote_api.core.RemoteAgentRunner", _RunnerStub)


def test_health_models_capabilities_and_auth(tmp_path: Path) -> None:
    service = RemoteAPIService(RemoteAPIConfig(tmp_path, api_key="secret", model="service-model", state_limit=4))

    assert service.health() == {
        "status": "ok",
        "version": "unknown",
        "workspace": str(tmp_path),
        "model": "service-model",
        "provider": "default",
    }
    detailed = service.detailed_health()
    assert detailed["auth"] == {"type": "bearer", "required": True}
    assert detailed["state_limit"] == 4
    assert service.models()["data"][0]["id"] == "service-model"
    assert service.capabilities()["features"]["responses_streaming"] is True

    service.require_auth("Bearer secret")
    with pytest.raises(RemoteAPIError, match="invalid bearer token"):
        service.require_auth("Bearer other")


def test_config_defaults_are_secure_and_non_interactive(tmp_path: Path) -> None:
    config = RemoteAPIConfig(tmp_path)

    assert config.host == "127.0.0.1"
    assert config.port == 8642
    assert config.permission_mode == "bypassPermissions"
    assert config.timeout_seconds == 600.0
    assert config.state_limit == 128


def test_core_reexports_response_metadata_for_response_payloads() -> None:
    assert _response_metadata is response_payloads._response_metadata


@pytest.mark.asyncio
async def test_chat_completion_normalizes_history_and_runner_config(
    tmp_path: Path,
) -> None:
    service = RemoteAPIService(
        RemoteAPIConfig(
            tmp_path,
            provider="provider-a",
            max_turns=7,
            permission_mode="dontAsk",
        )
    )

    payload = await service.chat_completion(
        {
            "model": "request-model",
            "messages": [
                {"role": "system", "content": "system rule"},
                {"role": "developer", "content": "developer rule"},
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "Read",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
            ],
        }
    )

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "request-model"
    assert payload["choices"][0]["message"]["content"] == "assistant reply"
    assert payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    runner = _RunnerStub.calls[0]
    assert runner.config.provider == "provider-a"
    assert runner.config.model == "request-model"
    assert runner.config.max_turns == 7
    assert runner.config.permission_mode == "dontAsk"
    assert runner.instructions == "system rule\n\ndeveloper rule"
    assert isinstance(runner.messages[0], UserMessage)
    assert isinstance(runner.messages[1].content[1], ToolUseBlock)
    assert isinstance(runner.messages[2].content[0], ToolResultBlock)
    assert service.active_runs == 0


@pytest.mark.asyncio
async def test_advertised_agent_model_defers_to_provider_default(
    tmp_path: Path,
) -> None:
    service = RemoteAPIService(RemoteAPIConfig(tmp_path))

    await service.chat_completion({"model": API_MODEL_NAME, "messages": [{"role": "user", "content": "hello"}]})

    assert _RunnerStub.calls[0].config.model is None
    assert _RunnerStub.calls[0].config.provider is None


@pytest.mark.asyncio
async def test_chat_stream_emits_openai_chunks_usage_and_done(tmp_path: Path) -> None:
    service = RemoteAPIService(RemoteAPIConfig(tmp_path, model="service-model"))

    frames = await service.chat_completion_sse(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    payloads = [_decode_sse(frame) for frame in frames]

    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert payloads[1]["choices"][0]["delta"] == {"content": "assistant "}
    assert payloads[2]["choices"][0]["delta"] == {"content": "reply"}
    assert payloads[-3]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-2]["choices"] == []
    assert payloads[-2]["usage"]["total_tokens"] == 5
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_stream_error_is_encoded_before_done(tmp_path: Path) -> None:
    _RunnerStub.stream_events = [RemoteAPIError(500, "provider failed")]
    service = RemoteAPIService(RemoteAPIConfig(tmp_path))

    frames = await service.chat_completion_sse({"messages": [{"role": "user", "content": "hello"}], "stream": True})

    assert frames[-2].startswith("event: error")
    assert _decode_sse(frames[-2])["error"]["message"] == "provider failed"
    assert _decode_sse(frames[-1]) == "[DONE]"


@pytest.mark.asyncio
async def test_timeout_and_failed_reason_cleanup_active_count(tmp_path: Path) -> None:
    _RunnerStub.run_delay = 0.05
    service = RemoteAPIService(RemoteAPIConfig(tmp_path, timeout_seconds=0.001))

    with pytest.raises(RemoteAPIError, match="timed out") as timeout:
        await service.chat_completion({"messages": [{"role": "user", "content": "hello"}]})
    assert timeout.value.status_code == 504
    assert service.active_runs == 0

    _RunnerStub.run_delay = 0.0
    _RunnerStub.run_result = RemoteRunResult("", "failed", {}, [], [])
    with pytest.raises(RemoteAPIError, match="agent run failed: failed"):
        await service.chat_completion({"messages": [{"role": "user", "content": "hello"}]})
    assert service.active_runs == 0


@pytest.mark.asyncio
async def test_same_conversation_serializes_and_releases_lock(tmp_path: Path) -> None:
    service = RemoteAPIService(RemoteAPIConfig(tmp_path))
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        async with service._conversation_scope("alpha"):
            order.append("first-enter")
            first_entered.set()
            await release_first.wait()
            order.append("first-exit")

    async def second() -> None:
        await first_entered.wait()
        async with service._conversation_scope("alpha"):
            order.append("second-enter")

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await first_entered.wait()
    await asyncio.sleep(0.02)
    assert order == ["first-enter"]

    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert order == ["first-enter", "first-exit", "second-enter"]
    entry = service._conversation_locks["alpha"]
    assert entry.refs == 0
    assert entry.lock.locked() is False


@pytest.mark.asyncio
async def test_independent_conversations_do_not_block_each_other(
    tmp_path: Path,
) -> None:
    service = RemoteAPIService(RemoteAPIConfig(tmp_path))
    both_entered = asyncio.Event()
    count = 0
    guard = asyncio.Lock()

    async def enter(conversation: str) -> None:
        nonlocal count
        async with service._conversation_scope(conversation):
            async with guard:
                count += 1
                if count == 2:
                    both_entered.set()
            await both_entered.wait()

    await asyncio.wait_for(asyncio.gather(enter("alpha"), enter("beta")), timeout=1)

    assert count == 2
    assert all(entry.refs == 0 for entry in service._conversation_locks.values())
    assert all(entry.lock.locked() is False for entry in service._conversation_locks.values())


@pytest.mark.asyncio
async def test_background_chat_requests_do_not_block_or_share_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ConcurrentRunner(_RunnerStub):
        calls: list[_RunnerStub] = []
        active = 0
        max_active = 0

        async def run(self) -> RemoteRunResult:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            try:
                await asyncio.sleep(0.03)
                prompt = self.messages[-1].content
                text = f"answer:{prompt}"
                return RemoteRunResult(
                    text=text,
                    reason="success",
                    usage={"input_tokens": 1, "output_tokens": 1},
                    messages=[*self.messages, AssistantMessage(content=text)],
                    events=[RemoteTextDelta(text)],
                )
            finally:
                type(self).active -= 1

    monkeypatch.setattr("extensions.remote_api.core.RemoteAgentRunner", ConcurrentRunner)
    service = RemoteAPIService(RemoteAPIConfig(tmp_path, api_key=""))
    prompts = ["main answer", "generate a title", "follow-up suggestions", "next turn"]

    results = await asyncio.gather(
        *(service.chat_completion({"messages": [{"role": "user", "content": prompt}]}) for prompt in prompts)
    )

    assert ConcurrentRunner.max_active == len(prompts)
    assert [result["choices"][0]["message"]["content"] for result in results] == [
        f"answer:{prompt}" for prompt in prompts
    ]
    assert [call.messages[-1].content for call in ConcurrentRunner.calls] == prompts
    assert service.active_runs == 0
    assert service.detailed_health()["stored_responses"] == 0
    assert service.detailed_health()["conversations"] == 0


@pytest.mark.asyncio
async def test_concurrent_chat_streams_keep_chunks_isolated_and_ordered(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class StreamingRunner(_RunnerStub):
        calls: list[_RunnerStub] = []
        active = 0
        max_active = 0

        async def stream(self) -> AsyncIterator[Any]:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            prompt = self.messages[-1].content
            try:
                yield RemoteTextDelta(f"{prompt}:a")
                await asyncio.sleep(0.02)
                yield RemoteTextDelta(":b")
                yield RemoteRunComplete(
                    reason="success",
                    response_text=f"{prompt}:a:b",
                    usage={"input_tokens": 1, "output_tokens": 1},
                    messages=[
                        *self.messages,
                        AssistantMessage(content=f"{prompt}:a:b"),
                    ],
                )
            finally:
                type(self).active -= 1

    async def collect(service: RemoteAPIService, index: int) -> str:
        frames = [
            _decode_sse(frame)
            async for frame in service.chat_completion_sse_events(
                {
                    "stream": True,
                    "messages": [{"role": "user", "content": f"stream-{index}"}],
                }
            )
        ]
        return "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in frames
            if isinstance(payload, dict) and payload["choices"]
        )

    monkeypatch.setattr("extensions.remote_api.core.RemoteAgentRunner", StreamingRunner)
    service = RemoteAPIService(RemoteAPIConfig(tmp_path, api_key=""))
    texts = await asyncio.gather(*(collect(service, index) for index in range(12)))

    assert StreamingRunner.max_active == 12
    assert texts == [f"stream-{index}:a:b" for index in range(12)]
    assert service.active_runs == 0


@pytest.mark.asyncio
async def test_same_conversation_serializes_complete_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class SlowEchoRunner(_RunnerStub):
        calls: list[_RunnerStub] = []

        async def run(self) -> RemoteRunResult:
            prompt = self.messages[-1].content
            if prompt == "first":
                await asyncio.sleep(0.05)
            text = f"answer:{prompt}"
            return RemoteRunResult(
                text=text,
                reason="success",
                usage={"input_tokens": 1, "output_tokens": 1},
                messages=[*self.messages, AssistantMessage(content=text)],
                events=[RemoteTextDelta(text)],
            )

    monkeypatch.setattr("extensions.remote_api.core.RemoteAgentRunner", SlowEchoRunner)
    service = RemoteAPIService(RemoteAPIConfig(tmp_path, api_key=""))
    first = asyncio.create_task(service.responses({"input": "first", "conversation": "shared"}))
    await asyncio.sleep(0.005)
    second = asyncio.create_task(service.responses({"input": "second", "conversation": "shared"}))

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result["output_text"] == "answer:first"
    assert second_result["output_text"] == "answer:second"
    assert [message.content for message in SlowEchoRunner.calls[1].messages] == [
        "first",
        "answer:first",
        "second",
    ]


@pytest.mark.asyncio
async def test_idle_conversation_locks_remain_bounded(tmp_path: Path) -> None:
    service = RemoteAPIService(RemoteAPIConfig(tmp_path, state_limit=1))

    for index in range(40):
        async with service._conversation_scope(f"conversation-{index}"):
            pass

    assert len(service._conversation_locks) <= 16
    assert all(entry.refs == 0 for entry in service._conversation_locks.values())
    assert all(entry.lock.locked() is False for entry in service._conversation_locks.values())


@pytest.mark.asyncio
async def test_closing_service_stream_closes_runner_and_decrements_active(
    tmp_path: Path,
) -> None:
    service = RemoteAPIService(RemoteAPIConfig(tmp_path))
    stream = service._stream_agent(
        messages=[UserMessage(content="hello")],
        instructions="",
        model=None,
        run_id="chat_1",
    )

    first = await stream.__anext__()
    assert first == RemoteTextDelta("assistant ")
    assert service.active_runs == 1

    await stream.aclose()

    assert _RunnerStub.stream_closed is True
    assert service.active_runs == 0
