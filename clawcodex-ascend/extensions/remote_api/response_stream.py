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

"""Streaming Responses API event projection."""

from __future__ import annotations

# Relative imports keep streaming behavior bound to extension-local contracts.
# pylint: disable=relative-beyond-top-level

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .errors import RemoteAPIError
from .response_payloads import (
    _optional_conversation_id,
    _remember_pending_call,
    _response_message_item,
    _response_tool_call_item,
    _response_tool_result_item,
    _responses_base_payload,
    _responses_payload,
    _stream_error_payload,
    _take_pending_call,
)
from .runner import (
    RemoteRunComplete,
    RemoteRunResult,
    RemoteTextDelta,
    RemoteToolCall,
    RemoteToolResult,
)
from .sse import encode_sse


logger = logging.getLogger(__name__)


@dataclass
class _ResponseStreamState:
    """Mutable projection state shared by the focused stream helpers."""

    message_item: dict[str, Any]
    sequence_number: int = 1
    text_parts: list[str] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
    output_items: list[dict[str, Any]] = field(default_factory=list)
    pending_call_ids: dict[str, list[str]] = field(default_factory=dict)
    next_output_index: int = 0
    message_output_index: int | None = None
    complete: RemoteRunComplete | None = None


async def response_sse_events(service: Any, body: dict[str, Any]) -> AsyncIterator[str]:
    """Project one Responses API run as an ordered SSE event stream."""

    try:
        conversation = _optional_conversation_id(body.get("conversation"))
    except RemoteAPIError as exc:
        yield encode_sse(_stream_error_payload(exc), event="error")
        return

    async with service._conversation_scope(conversation):
        try:
            prepared = service._prepare_responses_run(body)
        except RemoteAPIError as exc:
            yield encode_sse(_stream_error_payload(exc), event="error")
            return

        state = _ResponseStreamState(
            message_item=_response_message_item(prepared["response_id"], ""),
        )
        for frame in _response_started_frames(prepared, state):
            yield frame

        try:
            async for frame in _agent_event_frames(service, prepared, state):
                yield frame
        except RemoteAPIError as exc:
            yield _response_failed_frame(prepared, state, exc)
            return
        except Exception:
            logger.exception("Remote API response stream failed")
            yield _response_failed_frame(
                prepared,
                state,
                RemoteAPIError(500, "agent stream failed"),
            )
            return

        async for frame in _response_completed_frames(service, prepared, state):
            yield frame


def _sequenced_frame(
    state: _ResponseStreamState,
    event: str,
    payload: dict[str, Any],
) -> str:
    payload["sequence_number"] = state.sequence_number
    state.sequence_number += 1
    return encode_sse(payload, event=event)


def _response_started_frames(
    prepared: dict[str, Any],
    state: _ResponseStreamState,
) -> list[str]:
    response_created = _responses_base_payload(
        response_id=prepared["response_id"],
        model=prepared["response_model"],
        previous_response_id=prepared["previous_response_id"],
        conversation=prepared["conversation"],
        store=prepared["should_store"],
        status="in_progress",
        completed_at=None,
        output=[],
        output_text="",
        usage=None,
        instructions=prepared["instructions"] or None,
        created_at=prepared["created_at"],
    )
    return [
        _sequenced_frame(
            state,
            "response.created",
            {"type": "response.created", "response": response_created},
        ),
        _sequenced_frame(
            state,
            "response.in_progress",
            {"type": "response.in_progress", "response": response_created},
        ),
    ]


async def _agent_event_frames(
    service: Any,
    prepared: dict[str, Any],
    state: _ResponseStreamState,
) -> AsyncIterator[str]:
    async for event in service._stream_agent(
        messages=prepared["messages"],
        instructions=prepared["instructions"],
        model=prepared["query_model"],
        run_id=prepared["response_id"],
        session_id=prepared["session_id"],
    ):
        state.events.append(event)
        if isinstance(event, RemoteTextDelta) and event.content:
            for frame in _text_delta_frames(state, event):
                yield frame
        elif isinstance(event, RemoteToolCall):
            item = _response_tool_call_item(
                event,
                prepared["response_id"],
                state.next_output_index,
            )
            state.output_items.append(item)
            _remember_pending_call(state.pending_call_ids, event, item["call_id"])
            async for frame in _response_output_item_frames(
                state=state,
                item=item,
                output_index=state.next_output_index,
            ):
                yield frame
            state.next_output_index += 1
        elif isinstance(event, RemoteToolResult):
            fallback_call_id = _take_pending_call(state.pending_call_ids, event)
            item = _response_tool_result_item(
                event,
                prepared["response_id"],
                state.next_output_index,
                fallback_call_id=fallback_call_id,
            )
            state.output_items.append(item)
            async for frame in _response_output_item_frames(
                state=state,
                item=item,
                output_index=state.next_output_index,
            ):
                yield frame
            state.next_output_index += 1
        elif isinstance(event, RemoteRunComplete):
            state.complete = event


def _message_output_started_frames(state: _ResponseStreamState) -> list[str]:
    if state.message_output_index is not None:
        return []

    state.message_output_index = state.next_output_index
    state.next_output_index += 1
    state.output_items.append(state.message_item)
    output_index = state.message_output_index
    return [
        _sequenced_frame(
            state,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    **state.message_item,
                    "status": "in_progress",
                    "content": [],
                },
            },
        ),
        _sequenced_frame(
            state,
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": state.message_item["id"],
                "output_index": output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                },
            },
        ),
    ]


def _text_delta_frames(
    state: _ResponseStreamState,
    event: RemoteTextDelta,
) -> list[str]:
    frames = _message_output_started_frames(state)
    state.text_parts.append(event.content)
    frames.append(
        _sequenced_frame(
            state,
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": state.message_item["id"],
                "output_index": state.message_output_index,
                "content_index": 0,
                "delta": event.content,
            },
        )
    )
    return frames


async def _response_completed_frames(
    service: Any,
    prepared: dict[str, Any],
    state: _ResponseStreamState,
) -> AsyncIterator[str]:
    complete = state.complete
    text = complete.response_text if complete is not None and complete.response_text else "".join(state.text_parts)
    usage = dict(complete.usage if complete is not None else {})
    messages = complete.messages if complete is not None else list(prepared["messages"])
    state.message_item = _response_message_item(prepared["response_id"], text)
    for frame in _message_output_started_frames(state):
        yield frame

    state.output_items = [
        state.message_item if item["id"] == state.message_item["id"] else item for item in state.output_items
    ]
    result = RemoteRunResult(
        text=text,
        reason=complete.reason if complete is not None else "success",
        usage=usage,
        messages=messages,
        events=state.events,
    )
    response = _responses_payload(
        response_id=prepared["response_id"],
        model=prepared["response_model"],
        result=result,
        previous_response_id=prepared["previous_response_id"],
        conversation=prepared["conversation"],
        store=prepared["should_store"],
        instructions=prepared["instructions"] or None,
        output=state.output_items,
        created_at=prepared["created_at"],
    )
    if prepared["should_store"]:
        service._responses.put(
            prepared["response_id"],
            response,
            messages,
            prepared["input_items"],
            conversation=prepared["conversation"],
            session_id=prepared["session_id"],
        )

    for frame in _response_done_frames(state, result, response):
        yield frame


def _response_done_frames(
    state: _ResponseStreamState,
    result: RemoteRunResult,
    response: dict[str, Any],
) -> list[str]:
    output_index = state.message_output_index
    return [
        _sequenced_frame(
            state,
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": state.message_item["id"],
                "output_index": output_index,
                "content_index": 0,
                "text": result.text,
            },
        ),
        _sequenced_frame(
            state,
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": state.message_item["id"],
                "output_index": output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": result.text,
                    "annotations": [],
                },
            },
        ),
        _sequenced_frame(
            state,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": state.message_item,
            },
        ),
        _sequenced_frame(
            state,
            "response.completed",
            {"type": "response.completed", "response": response},
        ),
    ]


def _response_failed_frame(
    prepared: dict[str, Any],
    state: _ResponseStreamState,
    error: RemoteAPIError,
) -> str:
    """Project a stream failure without leaking unexpected exception details."""

    failed_message = _response_message_item(
        prepared["response_id"],
        "".join(state.text_parts),
    )
    failed_output = [failed_message if item["id"] == state.message_item["id"] else item for item in state.output_items]
    if state.message_output_index is not None:
        failed_message["status"] = "incomplete"
    failed_response = _responses_base_payload(
        response_id=prepared["response_id"],
        model=prepared["response_model"],
        previous_response_id=prepared["previous_response_id"],
        conversation=prepared["conversation"],
        store=prepared["should_store"],
        status="failed",
        completed_at=int(time.time()),
        output=failed_output,
        output_text="".join(state.text_parts),
        usage=None,
        instructions=prepared["instructions"] or None,
        created_at=prepared["created_at"],
        error={
            "message": error.detail,
            "type": error.error_type,
            "code": error.code,
        },
    )
    return _sequenced_frame(
        state,
        "response.failed",
        {
            "type": "response.failed",
            "response": failed_response,
        },
    )


async def _response_output_item_frames(
    *,
    state: _ResponseStreamState,
    item: dict[str, Any],
    output_index: int,
) -> AsyncIterator[str]:
    """Emit SSE frames for one output item (added → … → done).

    For ``function_call`` items this includes the REQUIRED intermediate
    ``function_call_arguments.delta`` / ``function_call_arguments.done``
    events that the OpenAI Responses API expects.  Skipping those
    events causes clients (including Open WebUI) to lose track of the
    call and ignore its paired ``function_call_output``.
    """
    item_type = item.get("type", "")
    item_id = item["id"]

    yield _sequenced_frame(
        state,
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {**item, "status": "in_progress"},
        },
    )

    if item_type == "function_call":
        arguments = item.get("arguments", "")
        yield _sequenced_frame(
            state,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item_id,
                "output_index": output_index,
                "delta": arguments,
            },
        )
        yield _sequenced_frame(
            state,
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": item_id,
                "output_index": output_index,
                "arguments": arguments,
            },
        )
        yield _sequenced_frame(
            state,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {**item, "status": "completed"},
            },
        )
    else:
        yield _sequenced_frame(
            state,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {**item, "status": "completed"},
            },
        )
