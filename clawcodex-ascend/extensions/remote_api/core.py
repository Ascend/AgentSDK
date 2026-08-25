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

"""Chat completion and runtime orchestration for the Remote API."""

from __future__ import annotations

# Relative imports keep the service bound to its extension-local contracts.
# pylint: disable=relative-beyond-top-level

import asyncio
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .auth import require_bearer_auth, resolve_api_key
from .errors import RemoteAPIError
from .normalization import normalize_chat_messages, reject_workspace_override
from .response_payloads import (
    API_MODEL_NAME,
    _chat_completion_payload,
    _chat_include_usage,
    _openai_usage,
    _response_metadata as _response_metadata,  # pylint: disable=unused-import
    _resolve_query_model,
    _stream_error_payload,
    _validate_common_request_fields,
)
from .responses import ResponsesMixin
from .runner import (
    RemoteAgentRunner,
    RemotePermissionMode,
    RemoteRunComplete,
    RemoteRunConfig,
    RemoteRunResult,
    RemoteTextDelta,
)
from .sse import chat_chunk, chat_usage_chunk, encode_done, encode_sse
from .state import ResponseStore


class _ConversationLock:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.refs = 0


@dataclass(frozen=True)
class RemoteAPIConfig:
    """Runtime configuration for the remote API server."""

    workspace: Path
    host: str = "127.0.0.1"
    port: int = 8642
    provider: str | None = None
    model: str | None = None
    max_turns: int = 20
    permission_mode: RemotePermissionMode = "bypassPermissions"
    timeout_seconds: float = 600.0
    state_limit: int = 128
    api_key: str | None = None


class RemoteAPIService(ResponsesMixin):
    """Hermes-compatible API service with process-local state."""

    def __init__(self, config: RemoteAPIConfig) -> None:
        self.config = config
        self._active_lock = threading.Lock()
        self._active_runs = 0
        self._started_at = time.time()
        self._api_key = resolve_api_key(config.api_key)
        self._responses = ResponseStore(config.state_limit)
        self._conversation_locks_guard = threading.Lock()
        self._conversation_locks: dict[str, _ConversationLock] = {}

    @property
    def auth_required(self) -> bool:
        return bool(self._api_key)

    def require_auth(self, authorization: str | None) -> None:
        require_bearer_auth(self._api_key, authorization)

    def health(self) -> dict[str, Any]:
        version = "unknown"
        try:
            from src import __version__

            version = __version__
        except Exception:
            version = "unknown"
        return {
            "status": "ok",
            "version": version,
            "workspace": str(self.config.workspace),
            "model": self.advertised_model(),
            "provider": self.config.provider or "default",
        }

    def detailed_health(self) -> dict[str, Any]:
        counts = self._responses.counts()
        return {
            **self.health(),
            "uptime_seconds": max(0, int(time.time() - self._started_at)),
            "auth": {
                "type": "bearer",
                "required": self.auth_required,
            },
            "active_runs": self.active_runs,
            "stored_responses": counts["responses"],
            "conversations": counts["conversations"],
            "state_limit": counts["limit"],
        }

    @property
    def active_runs(self) -> int:
        with self._active_lock:
            return self._active_runs

    def advertised_model(self) -> str:
        return self.config.model or API_MODEL_NAME

    def models(self) -> dict[str, Any]:
        model = self.advertised_model()
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": int(self._started_at),
                    "owned_by": "clawcodex",
                }
            ],
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "object": "hermes.api_server.capabilities",
            "platform": "clawcodex",
            "model": self.advertised_model(),
            "auth": {
                "type": "bearer",
                "required": self.auth_required,
            },
            "features": {
                "chat_completions": True,
                "responses_api": True,
                "chat_streaming": True,
                "responses_streaming": True,
                "data_image_input": True,
                "remote_image_input": False,
                "run_submission": False,
                "run_status": False,
                "run_events_sse": False,
                "run_stop": False,
                "sessions_api": False,
                "cron_jobs": False,
                "automation_state": True,
            },
        }

    async def chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        prepared = self.prepare_chat_completion(body)

        result = await self._run_agent(
            messages=prepared["messages"],
            instructions=prepared["instructions"],
            model=prepared["query_model"],
            run_id=prepared["run_id"],
        )
        return _chat_completion_payload(
            prepared["run_id"],
            prepared["response_model"],
            result,
        )

    def prepare_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize a chat request before HTTP streaming starts."""

        reject_workspace_override(body)
        _validate_common_request_fields(body)
        normalized = normalize_chat_messages(body.get("messages"))
        return {
            "messages": normalized.messages,
            "instructions": normalized.instructions,
            "response_model": str(body.get("model") or self.advertised_model()),
            "query_model": _resolve_query_model(body.get("model"), self.config.model),
            "run_id": f"chatcmpl_{uuid.uuid4().hex}",
            "include_usage": _chat_include_usage(body),
        }

    async def chat_completion_sse(self, body: dict[str, Any]) -> list[str]:
        return [frame async for frame in self.chat_completion_sse_events(body)]

    async def chat_completion_sse_events(
        self,
        body: dict[str, Any],
        *,
        prepared: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        prepared = prepared or self.prepare_chat_completion(body)
        response_model = prepared["response_model"]
        run_id = prepared["run_id"]
        include_usage = prepared["include_usage"]
        created = int(time.time())

        failed = False
        complete: RemoteRunComplete | None = None
        yield encode_sse(
            chat_chunk(
                chunk_id=run_id,
                created=created,
                model=response_model,
                delta={"role": "assistant", "content": ""},
                include_usage=include_usage,
            )
        )
        try:
            async for event in self._stream_agent(
                messages=prepared["messages"],
                instructions=prepared["instructions"],
                model=prepared["query_model"],
                run_id=run_id,
            ):
                if isinstance(event, RemoteTextDelta) and event.content:
                    yield encode_sse(
                        chat_chunk(
                            chunk_id=run_id,
                            created=created,
                            model=response_model,
                            delta={"content": event.content},
                            include_usage=include_usage,
                        )
                    )
                elif isinstance(event, RemoteRunComplete):
                    complete = event
        except RemoteAPIError as exc:
            failed = True
            yield encode_sse(_stream_error_payload(exc), event="error")
        if not failed:
            yield encode_sse(
                chat_chunk(
                    chunk_id=run_id,
                    created=created,
                    model=response_model,
                    delta={},
                    finish_reason="stop",
                    include_usage=include_usage,
                )
            )
            if include_usage:
                yield encode_sse(
                    chat_usage_chunk(
                        chunk_id=run_id,
                        created=created,
                        model=response_model,
                        usage=_openai_usage(complete.usage if complete else {}),
                    )
                )
        yield encode_done()

    @asynccontextmanager
    async def _conversation_scope(  # pylint: disable=invalid-overridden-method
        self, conversation: str | None
    ) -> AsyncIterator[None]:
        if not conversation:
            yield
            return
        entry = self._claim_conversation_lock(conversation)
        acquired = False
        try:
            while not acquired:
                acquired = entry.lock.acquire(blocking=False)
                if not acquired:
                    await asyncio.sleep(0.01)
            yield
        finally:
            if acquired:
                entry.lock.release()
            self._release_conversation_lock(conversation, entry)

    def _claim_conversation_lock(self, conversation: str) -> _ConversationLock:
        with self._conversation_locks_guard:
            entry = self._conversation_locks.get(conversation)
            if entry is None:
                entry = _ConversationLock()
                self._conversation_locks[conversation] = entry
            entry.refs += 1
            return entry

    def _release_conversation_lock(
        self,
        conversation: str,
        entry: _ConversationLock,
    ) -> None:
        with self._conversation_locks_guard:
            current = self._conversation_locks.get(conversation)
            if current is entry:
                entry.refs = max(0, entry.refs - 1)
            self._prune_conversation_locks_locked()

    def _prune_conversation_locks_locked(self) -> None:
        max_locks = max(16, self.config.state_limit * 2)
        if len(self._conversation_locks) <= max_locks:
            return
        for name, entry in list(self._conversation_locks.items()):
            if entry.refs == 0 and not entry.lock.locked():
                self._conversation_locks.pop(name, None)
                if len(self._conversation_locks) <= max_locks:
                    break

    async def _run_agent(
        self,
        *,
        messages: list[Any],
        instructions: str,
        model: str | None,
        run_id: str,
        session_id: str | None = None,
    ) -> RemoteRunResult:
        with self._active_lock:
            self._active_runs += 1
        try:
            runner = RemoteAgentRunner(
                RemoteRunConfig(
                    workspace=self.config.workspace,
                    provider=self.config.provider,
                    model=model,
                    max_turns=self.config.max_turns,
                    permission_mode=self.config.permission_mode,
                    session_id=session_id,
                ),
                messages=messages,
                instructions=instructions,
                run_id=run_id,
            )
            try:
                result = await asyncio.wait_for(
                    runner.run(),
                    timeout=self.config.timeout_seconds,
                )
            except TimeoutError as exc:
                raise RemoteAPIError(504, "agent run timed out") from exc
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                raise RemoteAPIError(500, f"agent run failed: exit_code={code}") from exc
            if result.reason != "success":
                raise RemoteAPIError(500, f"agent run failed: {result.reason}")
            return result
        finally:
            with self._active_lock:
                self._active_runs = max(0, self._active_runs - 1)

    async def _stream_agent(
        self,
        *,
        messages: list[Any],
        instructions: str,
        model: str | None,
        run_id: str,
        session_id: str | None = None,
    ) -> AsyncIterator[Any]:
        with self._active_lock:
            self._active_runs += 1
        try:
            runner = RemoteAgentRunner(
                RemoteRunConfig(
                    workspace=self.config.workspace,
                    provider=self.config.provider,
                    model=model,
                    max_turns=self.config.max_turns,
                    permission_mode=self.config.permission_mode,
                    session_id=session_id,
                ),
                messages=messages,
                instructions=instructions,
                run_id=run_id,
            )
            deadline = time.monotonic() + self.config.timeout_seconds
            stream = runner.stream()
            stream_iter = aiter(stream)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RemoteAPIError(504, "agent run timed out")
                try:
                    event = await asyncio.wait_for(anext(stream_iter), timeout=remaining)
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    raise RemoteAPIError(504, "agent run timed out") from exc
                if isinstance(event, RemoteRunComplete) and event.reason != "success":
                    raise RemoteAPIError(500, f"agent run failed: {event.reason}")
                yield event
        finally:
            if "stream" in locals():
                await stream.aclose()
            with self._active_lock:
                self._active_runs = max(0, self._active_runs - 1)


__all__ = [
    "API_MODEL_NAME",
    "RemoteAPIConfig",
    "RemoteAPIError",
    "RemoteAPIService",
]
