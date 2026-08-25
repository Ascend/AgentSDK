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

"""Focused tests for non-streaming Responses API state and payloads."""

from __future__ import annotations

import asyncio
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from extensions.remote_api.errors import RemoteAPIError
from extensions.remote_api.response_payloads import (
    API_MODEL_NAME,
    _chat_include_usage,
    _openai_usage,
    _resolve_query_model,
    _response_input_items,
    _response_metadata,
    _responses_payload,
)
from extensions.remote_api.responses import ResponsesMixin
from extensions.remote_api.runner import (
    RemoteRunResult,
    RemoteToolCall,
    RemoteToolResult,
)
from extensions.remote_api.state import ResponseStore
from clawcodex_ext.types.content_blocks import ImageBlock, TextBlock
from clawcodex_ext.types.messages import AssistantMessage, UserMessage


def _result(
    text: str,
    *,
    events: list[Any] | None = None,
    messages: list[Any] | None = None,
) -> RemoteRunResult:
    return RemoteRunResult(
        text=text,
        reason="success",
        usage={"input_tokens": 2, "output_tokens": 3, "cached_tokens": 1},
        messages=list(messages or [AssistantMessage(content=text)]),
        events=list(events or []),
    )


class _ResponsesHarness(ResponsesMixin):
    def __init__(self, tmp_path: Path, *, state_limit: int = 128) -> None:
        self.config = SimpleNamespace(model=None, workspace=tmp_path)
        self._responses = ResponseStore(state_limit)
        self.results: list[RemoteRunResult] = []
        self.calls: list[dict[str, Any]] = []

    def advertised_model(self) -> str:
        return API_MODEL_NAME

    @asynccontextmanager
    async def _conversation_scope(self, _conversation: str | None) -> AsyncIterator[None]:
        yield

    async def _run_agent(self, **kwargs: Any) -> RemoteRunResult:
        self.calls.append(kwargs)
        if self.results:
            return self.results.pop(0)
        return _result("ok", messages=[*kwargs["messages"], AssistantMessage(content="ok")])


def test_response_metadata_reports_automation_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = types.ModuleType("extensions.remote_api.state_reporter")
    reporter.current_automation_state = lambda: {"phase": "active", "focus": "minimal"}
    monkeypatch.setitem(sys.modules, reporter.__name__, reporter)

    metadata = _response_metadata()

    assert metadata["automation_state"]["phase"] == "active"
    assert metadata["automation_state"]["focus"] == "minimal"


@pytest.mark.asyncio
async def test_responses_store_get_input_items_and_delete(tmp_path: Path) -> None:
    service = _ResponsesHarness(tmp_path)

    response = await service.responses({"model": "model-a", "input": "hello", "instructions": "be concise"})

    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["output_text"] == "ok"
    assert response["model"] == "model-a"
    assert response["instructions"] == "be concise"
    assert response["usage"]["total_tokens"] == 5
    assert service.get_response(response["id"]) == response

    items = service.get_response_input_items(response["id"])
    assert items["object"] == "list"
    assert items["data"][0]["role"] == "user"
    assert items["data"][0]["content"] == [{"type": "input_text", "text": "hello"}]

    assert service.delete_response(response["id"])["deleted"] is True
    with pytest.raises(RemoteAPIError, match="response not found"):
        service.get_response(response["id"])


@pytest.mark.asyncio
async def test_previous_response_reuses_history_and_session(tmp_path: Path) -> None:
    service = _ResponsesHarness(tmp_path)

    first = await service.responses({"input": "first"})
    first_session = service.calls[0]["session_id"]
    second = await service.responses({"input": "second", "previous_response_id": first["id"]})

    assert second["previous_response_id"] == first["id"]
    assert service.calls[1]["session_id"] == first_session
    contents = [message.content for message in service.calls[1]["messages"]]
    assert contents == ["first", "ok", "second"]


@pytest.mark.asyncio
async def test_store_false_is_honored_when_continuing_response(tmp_path: Path) -> None:
    service = _ResponsesHarness(tmp_path)
    first = await service.responses({"input": "first"})

    second = await service.responses(
        {
            "input": "second",
            "previous_response_id": first["id"],
            "store": False,
        }
    )

    assert second["store"] is False
    with pytest.raises(RemoteAPIError, match="response not found"):
        service.get_response(second["id"])


@pytest.mark.asyncio
async def test_store_false_does_not_replace_named_conversation_head(tmp_path: Path) -> None:
    service = _ResponsesHarness(tmp_path)
    first = await service.responses({"input": "first", "conversation": "alpha"})

    second = await service.responses(
        {
            "input": "second",
            "conversation": "alpha",
            "store": False,
        }
    )
    third = await service.responses({"input": "third", "conversation": "alpha"})

    assert second["store"] is False
    assert third["previous_response_id"] is None
    assert [message.content for message in service.calls[2]["messages"]] == [
        "first",
        "ok",
        "third",
    ]
    assert service.get_response(first["id"])["id"] == first["id"]


@pytest.mark.asyncio
async def test_named_conversation_reuses_latest_history(tmp_path: Path) -> None:
    service = _ResponsesHarness(tmp_path)

    first = await service.responses({"input": "first", "conversation": {"id": "alpha"}})
    second = await service.responses({"input": "second", "conversation": "alpha"})

    assert first["conversation"] == {"id": "alpha"}
    assert second["conversation"] == {"id": "alpha"}
    assert service.calls[1]["session_id"] == service.calls[0]["session_id"]
    assert [message.content for message in service.calls[1]["messages"]] == [
        "first",
        "ok",
        "second",
    ]


@pytest.mark.asyncio
async def test_concurrent_independent_conversations_keep_history_isolated(
    tmp_path: Path,
) -> None:
    class ConcurrentHarness(_ResponsesHarness):
        def __init__(self, path: Path) -> None:
            super().__init__(path, state_limit=16)
            self.active = 0
            self.max_active = 0

        async def _run_agent(self, **kwargs: Any) -> RemoteRunResult:
            self.calls.append(kwargs)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.01)
                prompt = kwargs["messages"][-1].content
                text = f"answer:{prompt}"
                return _result(
                    text,
                    messages=[*kwargs["messages"], AssistantMessage(content=text)],
                )
            finally:
                self.active -= 1

    service = ConcurrentHarness(tmp_path)
    results = await asyncio.gather(
        *(service.responses({"input": f"turn-{index}", "conversation": f"thread-{index}"}) for index in range(10))
    )

    assert service.max_active == 10
    assert [result["output_text"] for result in results] == [f"answer:turn-{index}" for index in range(10)]
    assert [call["messages"][-1].content for call in service.calls] == [f"turn-{index}" for index in range(10)]
    assert service._responses.counts() == {
        "responses": 10,
        "conversations": 10,
        "limit": 16,
    }


@pytest.mark.asyncio
async def test_store_false_is_not_retrievable_without_chain(tmp_path: Path) -> None:
    service = _ResponsesHarness(tmp_path)

    response = await service.responses({"input": "ephemeral", "store": False})

    assert response["store"] is False
    with pytest.raises(RemoteAPIError, match="response not found"):
        service.get_response(response["id"])


@pytest.mark.asyncio
async def test_state_limit_evicts_oldest_response_and_alias(tmp_path: Path) -> None:
    service = _ResponsesHarness(tmp_path, state_limit=1)

    first = await service.responses({"input": "one", "conversation": "old"})
    second = await service.responses({"input": "two", "conversation": "new"})

    with pytest.raises(RemoteAPIError, match=first["id"]):
        service.get_response(first["id"])
    assert service.get_response(second["id"])["id"] == second["id"]
    assert service._responses.latest_for_conversation("old") is None


def test_responses_validation_rejects_conflicts_and_missing_history(
    tmp_path: Path,
) -> None:
    service = _ResponsesHarness(tmp_path)

    with pytest.raises(RemoteAPIError, match="cannot both be set"):
        service.validate_responses_request(
            {
                "input": "hello",
                "previous_response_id": "resp_missing",
                "conversation": "alpha",
            }
        )
    with pytest.raises(RemoteAPIError, match="response not found"):
        service.validate_responses_request({"input": "hello", "previous_response_id": "resp_missing"})
    with pytest.raises(RemoteAPIError, match="model must be"):
        service.validate_responses_request({"input": "hello", "model": 42})
    with pytest.raises(RemoteAPIError, match="stream must be a boolean"):
        service.validate_responses_request({"input": "hello", "stream": "yes"})


def test_response_payload_pairs_tool_results_by_id_or_name() -> None:
    events = [
        RemoteToolCall("Read", {"path": "a"}, None),
        RemoteToolResult("Read", {"output": {"text": "A"}}, None),
        RemoteToolCall("Write", {"path": "b"}, "call_exact"),
        RemoteToolResult("Write", "done", "call_exact"),
    ]

    payload = _responses_payload(
        response_id="resp_123",
        model="model",
        result=_result("finished", events=events),
        previous_response_id=None,
        conversation=None,
        store=True,
    )

    assert payload["output"][0]["call_id"] == "call_123_0"
    assert payload["output"][1]["call_id"] == "call_123_0"
    assert payload["output"][1]["output"] == [{"type": "input_text", "text": '{"text": "A"}'}]
    assert payload["output"][2]["call_id"] == "call_exact"
    assert payload["output"][3]["call_id"] == "call_exact"
    assert payload["output"][-1]["content"][0]["text"] == "finished"


def test_response_input_items_preserve_text_and_data_images() -> None:
    messages = [
        UserMessage(
            content=[
                TextBlock(text="inspect"),
                ImageBlock(
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "aGVsbG8=",
                    }
                ),
            ]
        )
    ]

    items = _response_input_items("resp_1", messages)

    assert items[0]["content"] == [
        {"type": "input_text", "text": "inspect"},
        {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
    ]


def test_model_usage_and_stream_options_helpers() -> None:
    assert _resolve_query_model(API_MODEL_NAME, None) is None
    assert _resolve_query_model("requested", None) == "requested"
    assert _resolve_query_model(API_MODEL_NAME, "configured") == "configured"
    assert _openai_usage({"input_tokens": "2", "output_tokens": -1}) == {
        "prompt_tokens": 2,
        "completion_tokens": 0,
        "total_tokens": 2,
    }
    assert _chat_include_usage({"stream": True, "stream_options": {"include_usage": True}})
    with pytest.raises(RemoteAPIError, match="only supported when stream is true"):
        _chat_include_usage({"stream_options": {"include_usage": True}})
