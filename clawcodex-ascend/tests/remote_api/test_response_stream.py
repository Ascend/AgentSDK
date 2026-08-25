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

"""Focused tests for Responses API server-sent events."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from extensions.remote_api.errors import RemoteAPIError
from extensions.remote_api.response_stream import response_sse_events
from extensions.remote_api.runner import (
    RemoteRunComplete,
    RemoteTextDelta,
    RemoteToolCall,
    RemoteToolResult,
)
from extensions.remote_api.state import ResponseStore
from clawcodex_ext.types.messages import AssistantMessage, UserMessage


def _decode_frame(frame: str) -> dict[str, Any]:
    data = "\n".join(line.removeprefix("data: ") for line in frame.splitlines() if line.startswith("data: "))
    return json.loads(data)


def _event_name(frame: str) -> str | None:
    for line in frame.splitlines():
        if line.startswith("event: "):
            return line.removeprefix("event: ")
    return None


class _StreamingHarness:
    def __init__(self, tmp_path: Path, events: list[Any]) -> None:
        self.workspace = tmp_path
        self.events = list(events)
        self._responses = ResponseStore()
        self.scope_entries: list[str | None] = []

    @asynccontextmanager
    async def _conversation_scope(self, conversation: str | None) -> AsyncIterator[None]:
        self.scope_entries.append(conversation)
        yield

    def _prepare_responses_run(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("invalid"):
            raise RemoteAPIError(400, "invalid request")
        return {
            "response_id": "resp_stream",
            "created_at": 10,
            "response_model": "model",
            "query_model": None,
            "previous_response_id": None,
            "conversation": body.get("conversation"),
            "session_id": "session_stream",
            "messages": [UserMessage(content="hello")],
            "input_items": [
                {
                    "id": "msg_stream_input_0",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            "instructions": "",
            "should_store": body.get("store", True),
        }

    async def _stream_agent(self, **_kwargs: Any) -> AsyncIterator[Any]:
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield event

    def get_stored(self, response_id: str) -> dict[str, Any]:
        stored = self._responses.get(response_id)
        assert stored is not None
        return stored.response


@pytest.mark.asyncio
async def test_responses_stream_emits_lifecycle_and_stores_final_response(
    tmp_path: Path,
) -> None:
    service = _StreamingHarness(
        tmp_path,
        [
            RemoteTextDelta("hel"),
            RemoteTextDelta("lo"),
            RemoteRunComplete(
                reason="success",
                response_text="hello final",
                usage={"input_tokens": 1, "output_tokens": 2},
                messages=[
                    UserMessage(content="hello"),
                    AssistantMessage(content="hello final"),
                ],
            ),
        ],
    )

    frames = [frame async for frame in response_sse_events(service, {})]
    names = [_event_name(frame) for frame in frames]
    payloads = [_decode_frame(frame) for frame in frames]

    assert names[:2] == ["response.created", "response.in_progress"]
    assert names.count("response.output_text.delta") == 2
    assert names[-1] == "response.completed"
    assert [payload["sequence_number"] for payload in payloads] == list(range(1, len(payloads) + 1))
    completed = payloads[-1]["response"]
    assert completed["output_text"] == "hello final"
    assert completed["usage"]["total_tokens"] == 3
    assert service.get_stored("resp_stream")["output_text"] == "hello final"


@pytest.mark.asyncio
async def test_stream_emits_required_function_call_argument_frames(
    tmp_path: Path,
) -> None:
    service = _StreamingHarness(
        tmp_path,
        [
            RemoteToolCall("Read", {"path": "README.md"}, "call_1"),
            RemoteToolResult("Read", {"output": "contents"}, "call_1"),
            RemoteRunComplete(reason="success", response_text="done", messages=[]),
        ],
    )

    frames = [frame async for frame in response_sse_events(service, {})]
    names = [_event_name(frame) for frame in frames]
    payloads = [_decode_frame(frame) for frame in frames]

    assert "response.function_call_arguments.delta" in names
    assert "response.function_call_arguments.done" in names
    added = [payload for payload in payloads if payload["type"] == "response.output_item.added"]
    assert added[0]["item"]["type"] == "function_call"
    assert added[1]["item"]["type"] == "function_call_output"
    assert added[1]["item"]["call_id"] == "call_1"


@pytest.mark.asyncio
async def test_mixed_stream_events_keep_contiguous_sequence_numbers(
    tmp_path: Path,
) -> None:
    service = _StreamingHarness(
        tmp_path,
        [
            RemoteTextDelta("before "),
            RemoteToolCall("Read", {"path": "README.md"}, "call_1"),
            RemoteToolResult("Read", {"output": "contents"}, "call_1"),
            RemoteTextDelta("after"),
            RemoteRunComplete(reason="success", response_text="before after", messages=[]),
        ],
    )

    frames = [frame async for frame in response_sse_events(service, {})]
    payloads = [_decode_frame(frame) for frame in frames]

    assert [payload["sequence_number"] for payload in payloads] == list(range(1, len(payloads) + 1))
    assert payloads[-1]["type"] == "response.completed"
    assert payloads[-1]["response"]["output_text"] == "before after"


@pytest.mark.asyncio
async def test_stream_failure_preserves_partial_text_and_error_shape(
    tmp_path: Path,
) -> None:
    service = _StreamingHarness(
        tmp_path,
        [RemoteTextDelta("partial"), RemoteAPIError(504, "agent run timed out")],
    )

    frames = [frame async for frame in response_sse_events(service, {})]
    payload = _decode_frame(frames[-1])

    assert _event_name(frames[-1]) == "response.failed"
    assert payload["response"]["status"] == "failed"
    assert payload["response"]["output_text"] == "partial"
    assert payload["response"]["error"] == {
        "message": "agent run timed out",
        "type": "invalid_request_error",
        "code": "timeout",
    }


@pytest.mark.asyncio
async def test_unexpected_stream_failure_is_logged_and_projected(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = _StreamingHarness(
        tmp_path,
        [RemoteTextDelta("partial"), RuntimeError("provider-secret-token")],
    )

    with caplog.at_level(
        logging.ERROR,
        logger="extensions.remote_api.response_stream",
    ):
        frames = [frame async for frame in response_sse_events(service, {})]
    payload = _decode_frame(frames[-1])

    assert _event_name(frames[-1]) == "response.failed"
    assert payload["response"]["output_text"] == "partial"
    assert payload["response"]["error"] == {
        "message": "agent stream failed",
        "type": "invalid_request_error",
        "code": "internal_error",
    }
    assert "provider-secret-token" in caplog.text


@pytest.mark.asyncio
async def test_invalid_conversation_or_request_yields_error_event(
    tmp_path: Path,
) -> None:
    service = _StreamingHarness(tmp_path, [])

    invalid_conversation = [frame async for frame in response_sse_events(service, {"conversation": {"id": 4}})]
    invalid_request = [frame async for frame in response_sse_events(service, {"invalid": True})]

    assert len(invalid_conversation) == 1
    assert _event_name(invalid_conversation[0]) == "error"
    assert _decode_frame(invalid_conversation[0])["error"]["code"] == "invalid_request"
    assert len(invalid_request) == 1
    assert _event_name(invalid_request[0]) == "error"


@pytest.mark.asyncio
async def test_first_delta_is_available_before_run_completes(tmp_path: Path) -> None:
    allow_completion = asyncio.Event()

    class _DelayedHarness(_StreamingHarness):
        async def _stream_agent(self, **_kwargs: Any) -> AsyncIterator[Any]:
            yield RemoteTextDelta("early")
            await allow_completion.wait()
            yield RemoteRunComplete(reason="success", response_text="early done", messages=[])

    service = _DelayedHarness(tmp_path, [])
    stream = response_sse_events(service, {})

    observed: list[dict[str, Any]] = []
    while True:
        frame = await stream.__anext__()
        payload = _decode_frame(frame)
        observed.append(payload)
        if payload["type"] == "response.output_text.delta":
            break

    assert observed[-1]["delta"] == "early"
    allow_completion.set()
    remaining = [frame async for frame in stream]
    assert _event_name(remaining[-1]) == "response.completed"


@pytest.mark.asyncio
async def test_store_false_does_not_persist_streamed_response(tmp_path: Path) -> None:
    service = _StreamingHarness(
        tmp_path,
        [RemoteRunComplete(reason="success", response_text="ephemeral", messages=[])],
    )

    frames = [frame async for frame in response_sse_events(service, {"store": False})]

    assert _decode_frame(frames[-1])["response"]["store"] is False
    assert service._responses.get("resp_stream") is None
