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

# pylint: disable=arguments-differ

"""A transparent :class:`BaseProvider` wrapper for multi-model turns."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional

from clawcodex_ext.capabilities.multimodel_protocol import (  # pylint: disable=no-name-in-module
    AggregatedOutput,
    AggregatorProtocol,
    MultiModelResult,
    MultiModelStrategy,
)
from clawcodex_ext.providers.base import BaseProvider, ChatResponse, MessageInput

from .aggregators.base import no_successful_results_error, valid_results
from .slots import ProviderSlot


@dataclass(frozen=True)
class RouterConfig:
    """Execution limits shared by all router strategies."""

    max_concurrent: int = 5

    def __post_init__(self) -> None:
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero")


class MultiModelRouter(BaseProvider):
    """Expose an ensemble as one ordinary provider to the query engine."""

    def __init__(
        self,
        slots: list[ProviderSlot],
        strategy: MultiModelStrategy,
        aggregator: AggregatorProtocol | None = None,
        *,
        config: RouterConfig | None = None,
        session_bridge: Any | None = None,
    ) -> None:
        if not slots:
            raise ValueError("MultiModelRouter needs at least one provider slot")
        if len({slot.name for slot in slots}) != len(slots):
            raise ValueError("provider slot names must be unique")
        # Compose a display-friendly label from slot names so the REPL
        # startup header and other UI callers see something meaningful
        # instead of None.  Slot names are guaranteed non-empty and unique
        # (validated above), so no deduplication is needed.  The label is
        # purely cosmetic; each slot carries its own model for actual API
        # calls (see _call_slot).
        model_label = " | ".join(s.name for s in slots)
        super().__init__(api_key="", model=model_label)
        self.slots = list(slots)
        self.strategy = strategy
        self._aggregator = aggregator
        self.config = config or RouterConfig()
        self.session_bridge = session_bridge
        self._last_result: list[MultiModelResult] | None = None
        self._last_aggregated: AggregatedOutput | None = None
        # Sinks are deliberately provider-local: frontends can subscribe
        # without changing the query loop or sharing mutable global state.
        self._event_listeners: list[Callable[[str, dict[str, Any]], None]] = []

    def add_event_listener(self, listener: Callable[[str, dict[str, Any]], None]) -> None:
        if listener not in self._event_listeners:
            self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[str, dict[str, Any]], None]) -> None:
        try:
            self._event_listeners.remove(listener)
        except ValueError:
            pass

    def _emit(self, event: str, **payload: Any) -> None:
        for listener in tuple(self._event_listeners):
            try:
                listener(event, payload)
            except Exception:
                # Rendering/audit listeners must never fail a model request.
                continue

    @property
    def last_result(self) -> list[MultiModelResult] | None:
        return self._last_result

    @property
    def last_aggregated(self) -> AggregatedOutput | None:
        return self._last_aggregated

    async def _call_slot(self, slot: ProviderSlot, messages: list[MessageInput], **kwargs: Any) -> MultiModelResult:
        started = time.monotonic()
        call_kwargs = dict(kwargs)
        # A slot's explicit model is a group policy and deliberately wins over
        # the model selected by the enclosing single-provider runtime.
        if slot.model:
            call_kwargs["model"] = slot.model
        try:
            response = await asyncio.wait_for(
                self._stream_slot(slot, messages, **call_kwargs),
                timeout=slot.timeout_ms / 1000,
            )
            duration_ms = int((time.monotonic() - started) * 1000)
            usage = response.usage if isinstance(response.usage, dict) else {}
            result = MultiModelResult(
                slot.name,
                response,
                duration_ms,
                self._tokens(usage),
                completed_at=time.monotonic(),
            )
            self._emit("complete", result=result)
            return result
        except asyncio.CancelledError:
            duration_ms = int((time.monotonic() - started) * 1000)
            result = MultiModelResult(
                slot.name,
                self._empty_response(slot),
                duration_ms,
                {},
                cancelled=True,
                completed_at=time.monotonic(),
            )
            self._emit("complete", result=result)
            raise
        except asyncio.TimeoutError:
            result = MultiModelResult(
                slot.name,
                self._empty_response(slot),
                slot.timeout_ms,
                {},
                error=f"Timeout after {slot.timeout_ms}ms",
                completed_at=time.monotonic(),
            )
            self._emit("complete", result=result)
            return result
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            result = MultiModelResult(
                slot.name,
                self._empty_response(slot),
                duration_ms,
                {},
                error=str(exc),
                completed_at=time.monotonic(),
            )
            self._emit("complete", result=result)
            return result

    async def _stream_slot(self, slot: ProviderSlot, messages: list[MessageInput], **kwargs: Any) -> ChatResponse:
        """Prefer structured streaming, with a compatible async fallback."""

        accept_events = True

        def text(chunk: str) -> None:
            if accept_events:
                self._emit("progress", slot=slot.name, chunk=chunk, status="streaming")

        def thinking(chunk: str) -> None:
            if accept_events:
                self._emit("thinking", slot=slot.name, chunk=chunk, status="streaming")

        try:
            try:
                if not callable(getattr(slot.provider, "chat_stream_response", None)):
                    raise NotImplementedError
                return await asyncio.to_thread(
                    slot.provider.chat_stream_response,
                    messages,
                    on_text_chunk=text,
                    on_thinking_chunk=thinking,
                    **kwargs,
                )
            except NotImplementedError:
                response = await slot.provider.chat_async(messages, **kwargs)
                if response.reasoning_content:
                    thinking(response.reasoning_content)
                if response.content:
                    text(response.content)
                return response
        finally:
            # ``asyncio.to_thread`` cannot stop a blocking provider, but after
            # cancellation/timeout its late chunks must not revive UI progress.
            accept_events = False

    @staticmethod
    def _empty_response(slot: ProviderSlot) -> ChatResponse:
        return ChatResponse("", slot.model or slot.provider.model or "", {}, "error")

    @staticmethod
    def _tokens(usage: dict[str, Any]) -> dict[str, int]:
        """Normalise provider-specific usage payloads to the public contract."""

        def integer(value: Any) -> int:
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

        return {
            "input": integer(usage.get("input", usage.get("input_tokens", 0))),
            "output": integer(usage.get("output", usage.get("output_tokens", 0))),
        }

    async def _execute(self, messages: list[MessageInput], **kwargs: Any) -> ChatResponse:
        results = await self.strategy.execute(self, messages, **kwargs)
        self._last_result = results
        if not results:
            raise RuntimeError("No enabled providers in multi-model router")
        successful = valid_results(results)
        if not successful:
            error = no_successful_results_error(results)
            self._last_aggregated = None
            if self.session_bridge is not None:
                self.session_bridge.record(results, None)
            self._emit("failed", results=results, error=str(error))
            raise error
        aggregator = self._aggregator or getattr(self.strategy, "aggregator", None)
        if aggregator is not None:
            aggregated = await aggregator.aggregate(results, self._aggregation_context(messages))
            self._last_aggregated = aggregated
            chosen = aggregated.chosen
        else:
            chosen = successful[0].response
            self._last_aggregated = None
        if self.session_bridge is not None:
            self.session_bridge.record(results, self._last_aggregated)
        self._emit("aggregated", results=results, output=self._last_aggregated)
        return chosen

    def _aggregation_context(self, messages: list[MessageInput]) -> dict[str, Any]:
        """Supply aggregators with the actual task as well as slot metadata."""
        user_request = ""
        for message in reversed(messages):
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            if role != "user":
                continue
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
            if isinstance(content, str):
                user_request = content
            elif content is not None:
                user_request = str(content)
            break
        return {
            "slot_weights": {slot.name: slot.weight for slot in self.slots},
            "user_request": user_request,
        }

    def chat_stream_response(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk: Any = None,
        on_thinking_chunk: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        response = self._run_sync(self._execute(messages, tools=tools, **kwargs))
        # The child calls are intentionally non-streaming so all candidates can
        # be scheduled uniformly. Preserve BaseProvider's final-response
        # streaming compatibility for callers that supplied callbacks.
        if on_thinking_chunk is not None and response.reasoning_content:
            on_thinking_chunk(response.reasoning_content)
        if on_text_chunk is not None and response.content:
            on_text_chunk(response.content)
        return response

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        return self._run_sync(self._execute(messages, tools=tools, **kwargs))

    def chat_stream(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        response = self.chat(messages, tools=tools, **kwargs)
        if response.content:
            yield response.content

    def get_available_models(self) -> list[str]:
        models: list[str] = []
        for slot in self.slots:
            if slot.model:
                models.append(slot.model)
            else:
                models.extend(slot.provider.get_available_models())
        return list(dict.fromkeys(models))

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        """Run a coroutine even if an embedding UI already owns an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        outcome: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def runner() -> None:
            try:
                outcome["value"] = asyncio.run(coro)
            except BaseException as exc:  # propagate exact provider exception
                error["error"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error:
            raise error["error"]
        return outcome["value"]
