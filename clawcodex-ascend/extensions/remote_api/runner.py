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

"""Request-time bridge from remote API messages to the ClawCodex query loop."""

from __future__ import annotations

# Relative imports bind local contracts; target subpackages are exposed dynamically.
# pylint: disable=relative-beyond-top-level,no-name-in-module

import asyncio
import logging
import threading
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

from clawcodex_ext.types.messages import Message

from .errors import RemoteAPIError


logger = logging.getLogger(__name__)

RemotePermissionMode = Literal["bypassPermissions", "dontAsk"]


@dataclass(frozen=True)
class RemoteRunConfig:
    workspace: Path
    provider: str | None = None
    model: str | None = None
    max_turns: int = 20
    # Remote runs cannot answer interactive permission prompts. Bypass is the
    # useful default for an operator-controlled agent server; dontAsk remains
    # available when callers intentionally want unapproved tools denied.
    permission_mode: RemotePermissionMode = "bypassPermissions"
    session_id: str | None = None


@dataclass
class RemoteTextDelta:
    content: str


@dataclass
class RemoteToolCall:
    tool_name: str
    params: dict[str, Any]
    tool_use_id: str | None = None


@dataclass
class RemoteToolResult:
    tool_name: str
    result: Any
    tool_use_id: str | None = None


@dataclass
class RemoteRunComplete:
    reason: str
    response_text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)


RemoteRunEvent = RemoteTextDelta | RemoteToolCall | RemoteToolResult | RemoteRunComplete


@dataclass
class RemoteRunResult:
    text: str
    reason: str
    usage: dict[str, Any]
    messages: list[Message]
    events: list[RemoteRunEvent]


class RemoteAgentRunner:
    """Run one ClawCodex query from already-normalized messages."""

    def __init__(
        self,
        config: RemoteRunConfig,
        *,
        messages: list[Message],
        instructions: str = "",
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.messages = list(messages)
        self.instructions = instructions
        self.run_id = run_id

    async def run(self) -> RemoteRunResult:
        events: list[RemoteRunEvent] = []
        text_parts: list[str] = []
        complete: RemoteRunComplete | None = None
        async for event in self.stream():
            events.append(event)
            if isinstance(event, RemoteTextDelta):
                text_parts.append(event.content)
            elif isinstance(event, RemoteRunComplete):
                complete = event
        if complete is None:
            complete = RemoteRunComplete(reason="unknown", messages=list(self.messages))
        text = complete.response_text or "".join(text_parts)
        return RemoteRunResult(
            text=text,
            reason=complete.reason,
            usage=complete.usage,
            messages=complete.messages,
            events=events,
        )

    async def stream(self) -> AsyncIterator[RemoteRunEvent]:
        runtime = _build_runtime(self.config)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[RemoteRunEvent | BaseException | object] = asyncio.Queue(maxsize=256)
        done = object()
        closed = threading.Event()
        worker_finished = threading.Event()
        emitted_text = False
        persisted_messages: list[Message] = []

        def emit(event: RemoteRunEvent | BaseException | object) -> None:
            if closed.is_set():
                return
            try:
                future = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            except RuntimeError:
                # The request loop may already be gone during shutdown or
                # client disconnect cleanup.
                return
            while not closed.is_set():
                try:
                    future.result(timeout=0.1)
                    return
                except FutureTimeoutError:
                    continue
                except (FutureCancelledError, RuntimeError):
                    return
            if not future.done():
                future.cancel()

        def on_text(chunk: str) -> None:
            nonlocal emitted_text
            if chunk:
                emitted_text = True
                emit(RemoteTextDelta(content=chunk))

        def on_event(event: Any) -> None:
            kind = getattr(event, "kind", "")
            tool_name = str(getattr(event, "tool_name", "") or "")
            tool_use_id = getattr(event, "tool_use_id", None)
            if kind == "tool_use":
                raw_input = getattr(event, "tool_input", None)
                params = raw_input if isinstance(raw_input, dict) else {}
                emit(
                    RemoteToolCall(
                        tool_name=tool_name,
                        params=params,
                        tool_use_id=tool_use_id,
                    )
                )
            elif kind in {"tool_result", "tool_error"}:
                result: dict[str, Any] = {
                    "output": getattr(event, "tool_output", None),
                    "is_error": bool(getattr(event, "is_error", False)) or kind == "tool_error",
                }
                error = getattr(event, "error", None)
                if error is not None:
                    result["error"] = error
                emit(
                    RemoteToolResult(
                        tool_name=tool_name,
                        result=result,
                        tool_use_id=tool_use_id,
                    )
                )

        def on_message(message: Message) -> None:
            persisted_messages.append(message)

        async def run_loop() -> None:
            try:
                from clawcodex_ext.bootstrap.state import (  # pylint: disable=no-name-in-module
                    SdkContext,
                    SessionId,
                    run_with_sdk_context,
                )

                # `src.outputStyles` + `src.query.agent_loop_compat` are public
                # upstream APIs exercised by tests/remote_api/test_remote_api.py
                # via ``unittest.mock.patch("src.…")`` (see :1006, :1013).
                # Layer 2 may import public ``src.*`` APIs directly per CLAUDE.md
                # (the prohibition is on ``src._internals``).
                from src.outputStyles import resolve_output_style
                from src.query.agent_loop_compat import (
                    build_effective_system_prompt,
                    run_query_as_agent_loop,
                )

                session_id = _session_id_for_run(self.config.session_id or self.run_id)
                runtime["tool_context"].session_id = session_id
                style_prompt = resolve_output_style(
                    getattr(runtime["tool_context"], "output_style_name", None),
                    getattr(runtime["tool_context"], "output_style_dir", None),
                ).prompt
                system_prompt = build_effective_system_prompt(style_prompt, runtime["tool_context"])
                if self.instructions:
                    system_prompt = f"{system_prompt}\n\n{self.instructions}" if system_prompt else self.instructions

                sdk_context = SdkContext(
                    session_id=SessionId(session_id),
                    session_project_dir=str(self.config.workspace),
                    cwd=str(self.config.workspace),
                    original_cwd=str(self.config.workspace),
                )
                with run_with_sdk_context(sdk_context):
                    result = await run_query_as_agent_loop(
                        initial_messages=list(self.messages),
                        provider=runtime["provider"],
                        tool_registry=runtime["tool_registry"],
                        tool_context=runtime["tool_context"],
                        system_prompt=system_prompt,
                        max_turns=self.config.max_turns,
                        on_event=on_event,
                        on_text_chunk=on_text,
                        on_message=on_message,
                        abort_controller=runtime["abort_controller"],
                    )
                if not emitted_text and result.response_text:
                    emit(RemoteTextDelta(content=result.response_text))
                emit(
                    RemoteRunComplete(
                        reason="success",
                        response_text=result.response_text,
                        usage=dict(result.usage or {}),
                        messages=[*self.messages, *persisted_messages],
                    )
                )
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                emit(RemoteAPIError(500, f"agent run failed: exit_code={code}"))
            except RemoteAPIError as exc:
                emit(exc)
            except Exception:
                logger.exception("Remote API agent run failed")
                emit(RemoteAPIError(500, "agent run failed"))

        def run_worker() -> None:
            try:
                asyncio.run(run_loop())
            except BaseException as exc:  # noqa: BLE001 - surface worker failures.
                emit(exc)
            finally:
                worker_finished.set()
                emit(done)

        worker = threading.Thread(
            target=run_worker,
            daemon=True,
            name=f"remote-api-agent-{self.run_id or id(self)}",
        )
        worker.start()
        try:
            while True:
                event = await queue.get()
                if event is done:
                    break
                if isinstance(event, BaseException):
                    raise event
                yield event
        finally:
            closed.set()
            if not worker_finished.is_set():
                abort_controller = runtime["abort_controller"]
                signal = getattr(abort_controller, "signal", None)
                if not bool(getattr(signal, "aborted", False)):
                    abort_controller.abort("remote_api_stream_closed")
                for _ in range(20):
                    if worker_finished.is_set():
                        break
                    await asyncio.sleep(0.05)
                if not worker_finished.is_set():
                    logger.warning(
                        "remote API agent worker did not stop promptly: run_id=%s",
                        self.run_id,
                    )


def _build_runtime(config: RemoteRunConfig) -> dict[str, Any]:
    from src.config import get_default_provider, get_provider_config
    from clawcodex_ext.permissions.types import ToolPermissionContext
    from clawcodex_ext.repl.core import get_provider_class
    from clawcodex_ext.tool_system.context import ToolContext
    from clawcodex_ext.tool_system.defaults import build_default_registry
    from clawcodex_ext.utils.abort_controller import AbortController

    provider_name = config.provider or get_default_provider()
    try:
        provider_cfg = get_provider_config(provider_name)
    except Exception as exc:
        raise RemoteAPIError(500, f"unable to load provider config: {exc}") from exc
    if not provider_cfg.get("api_key"):
        raise RemoteAPIError(
            500,
            f"API key for provider '{provider_name}' is not configured",
            code="provider_not_configured",
        )
    provider_cls = get_provider_class(provider_name)
    model = config.model or provider_cfg.get("default_model")
    provider = provider_cls(
        api_key=provider_cfg["api_key"],
        base_url=provider_cfg.get("base_url"),
        model=model,
    )
    tool_registry = build_default_registry(provider=provider)
    abort_controller = AbortController()
    tool_context = ToolContext(
        workspace_root=config.workspace,
        cwd=config.workspace,
        permission_context=ToolPermissionContext(
            mode=config.permission_mode,  # type: ignore[arg-type]
            is_bypass_permissions_mode_available=True,
            should_avoid_permission_prompts=True,
        ),
        abort_controller=abort_controller,
    )
    tool_context.options.is_non_interactive_session = True
    tool_context.allow_docs = True
    tool_context.permission_handler = None
    tool_context.ask_user = lambda _questions: {}
    tool_context.tool_registry = tool_registry
    return {
        "provider": provider,
        "tool_registry": tool_registry,
        "tool_context": tool_context,
        "abort_controller": abort_controller,
    }


def _session_id_for_run(run_id: str | None) -> str:
    raw = run_id or f"remote_{uuid4().hex}"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return safe[:128] or f"remote_{uuid4().hex}"
