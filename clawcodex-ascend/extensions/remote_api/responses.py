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

"""Non-streaming Responses API behavior shared by the service host."""

from __future__ import annotations

# Relative imports keep Responses behavior bound to extension-local contracts.
# pylint: disable=relative-beyond-top-level

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any

from .errors import RemoteAPIError
from .normalization import (
    merge_instructions,
    normalize_responses_input,
    reject_workspace_override,
)
from .response_payloads import (
    _optional_conversation_id,
    _optional_string_field,
    _resolve_query_model,
    _response_input_items,
    _responses_payload,
    _validate_common_request_fields,
)
from .runner import RemoteRunResult
from .state import ResponseStore, StoredResponse


class ResponsesMixin:
    """Responses methods mixed into :class:`RemoteAPIService`.

    The concrete service owns runner lifecycle and conversation locking. Keeping
    those seams abstract avoids duplicating the Chat service implementation.
    """

    config: Any
    _responses: ResponseStore

    def advertised_model(self) -> str:
        raise NotImplementedError

    def _conversation_scope(
        self,
        conversation: str | None,
    ) -> AbstractAsyncContextManager[None]:
        raise NotImplementedError

    async def _run_agent(
        self,
        *,
        messages: list[Any],
        instructions: str,
        model: str | None,
        run_id: str,
        session_id: str | None = None,
    ) -> RemoteRunResult:
        raise NotImplementedError

    async def _stream_agent(
        self,
        *,
        messages: list[Any],
        instructions: str,
        model: str | None,
        run_id: str,
        session_id: str | None = None,
    ) -> AsyncIterator[Any]:
        raise NotImplementedError
        yield  # pylint: disable=unreachable  # pragma: no cover - async generator contract.

    def validate_responses_request(self, body: dict[str, Any]) -> None:
        """Reject malformed Responses requests before sending SSE headers."""

        reject_workspace_override(body)
        _validate_common_request_fields(body)
        previous_response_id = _optional_string_field(body, "previous_response_id")
        conversation = _optional_conversation_id(body.get("conversation"))
        if previous_response_id and conversation:
            raise RemoteAPIError(400, "previous_response_id and conversation cannot both be set")
        _optional_string_field(body, "instructions")
        normalize_responses_input(body.get("input"))
        if previous_response_id and self._responses.get(previous_response_id) is None:
            raise RemoteAPIError(404, f"response not found: {previous_response_id}")

    async def responses(self, body: dict[str, Any]) -> dict[str, Any]:
        conversation = _optional_conversation_id(body.get("conversation"))
        async with self._conversation_scope(conversation):
            response, _result = await self._responses_impl(body)
            return response

    async def responses_sse(self, body: dict[str, Any]) -> list[str]:
        return [frame async for frame in self.responses_sse_events(body)]

    async def responses_sse_events(self, body: dict[str, Any]) -> AsyncIterator[str]:
        # Load the streaming encoder only when a streaming response is requested.
        from .response_stream import response_sse_events

        async for frame in response_sse_events(self, body):
            yield frame

    def get_response(self, response_id: str) -> dict[str, Any]:
        stored = self._responses.get(response_id)
        if stored is None:
            raise RemoteAPIError(404, f"response not found: {response_id}")
        return stored.response

    def get_response_input_items(self, response_id: str) -> dict[str, Any]:
        stored = self._responses.get(response_id)
        if stored is None:
            raise RemoteAPIError(404, f"response not found: {response_id}")
        data = list(stored.input_items)
        return {
            "object": "list",
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": False,
        }

    def delete_response(self, response_id: str) -> dict[str, Any]:
        if not self._responses.delete(response_id):
            raise RemoteAPIError(404, f"response not found: {response_id}")
        return {
            "id": response_id,
            "object": "response.deleted",
            "deleted": True,
        }

    def _prepare_responses_run(self, body: dict[str, Any]) -> dict[str, Any]:
        reject_workspace_override(body)
        _validate_common_request_fields(body)
        previous_response_id = _optional_string_field(body, "previous_response_id")
        conversation = _optional_conversation_id(body.get("conversation"))
        if previous_response_id and conversation:
            raise RemoteAPIError(
                400,
                "previous_response_id and conversation cannot both be set",
            )
        instructions = _optional_string_field(body, "instructions") or ""
        response_model = str(body.get("model") or self.advertised_model())
        query_model = _resolve_query_model(body.get("model"), self.config.model)
        response_id = f"resp_{uuid.uuid4().hex}"
        created_at = int(time.time())

        base: StoredResponse | None = None
        if previous_response_id:
            base = self._responses.get(previous_response_id)
            if base is None:
                raise RemoteAPIError(404, f"response not found: {previous_response_id}")
        elif conversation:
            base = self._responses.latest_for_conversation(conversation)

        normalized = normalize_responses_input(body.get("input"))
        input_items = _response_input_items(response_id, normalized.messages)
        messages = [*(base.messages if base is not None else []), *normalized.messages]
        merged_instructions = merge_instructions(
            instructions,
            normalized.instructions,
        )
        should_store = body.get("store", True) is not False
        return {
            "response_id": response_id,
            "created_at": created_at,
            "response_model": response_model,
            "query_model": query_model,
            "previous_response_id": previous_response_id,
            "conversation": conversation,
            "session_id": (base.session_id if base is not None and base.session_id else response_id),
            "messages": messages,
            "input_items": input_items,
            "instructions": merged_instructions,
            "should_store": should_store,
        }

    async def _responses_impl(self, body: dict[str, Any]) -> tuple[dict[str, Any], RemoteRunResult]:
        prepared = self._prepare_responses_run(body)
        result = await self._run_agent(
            messages=prepared["messages"],
            instructions=prepared["instructions"],
            model=prepared["query_model"],
            run_id=prepared["response_id"],
            session_id=prepared["session_id"],
        )
        response = _responses_payload(
            response_id=prepared["response_id"],
            model=prepared["response_model"],
            result=result,
            previous_response_id=prepared["previous_response_id"],
            conversation=prepared["conversation"],
            store=prepared["should_store"],
            instructions=prepared["instructions"] or None,
            created_at=prepared["created_at"],
        )
        if prepared["should_store"]:
            self._responses.put(
                prepared["response_id"],
                response,
                result.messages,
                prepared["input_items"],
                conversation=prepared["conversation"],
                session_id=prepared["session_id"],
            )
        return response, result
