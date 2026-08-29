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

"""Behavioural coverage for the transparent provider router."""

from __future__ import annotations

import asyncio

import pytest

from clawcodex_ext.capabilities.multimodel_protocol import AggregatedOutput
from clawcodex_ext.multimodel import (
    FallbackStrategy,
    MultiModelRouter,
    ParallelStrategy,
    ProviderSlot,
    RouterConfig,
    RoutingRule,
    RoutingStrategy,
    SessionBridge,
)
from clawcodex_ext.multimodel.aggregators import PassThroughAggregator
from clawcodex_ext.multimodel.aggregators import FirstSuccessAggregator
from clawcodex_ext.multimodel.aggregators import MajorityVoteAggregator
from clawcodex_ext.providers.base import BaseProvider, ChatResponse


class Provider(BaseProvider):
    def __init__(self, answer: str = "answer", *, delay: float = 0, error: Exception | None = None):
        super().__init__("key", model="default")
        self.answer, self.delay, self.error = answer, delay, error
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, **kwargs):  # pragma: no cover - async path is under test
        raise NotImplementedError

    def chat_stream(self, messages, tools=None, **kwargs):
        yield self.answer

    def get_available_models(self):
        return ["default"]

    async def chat_async(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return ChatResponse(self.answer, kwargs.get("model", self.model), {"input": 2, "output": 3}, "stop")


def test_parallel_router_invokes_slots_and_preserves_audit_data() -> None:
    first, second = Provider("first"), Provider("second")
    bridge = SessionBridge()
    router = MultiModelRouter(
        [ProviderSlot("one", first, "one-model"), ProviderSlot("two", second)],
        ParallelStrategy(),
        PassThroughAggregator(),
        session_bridge=bridge,
    )
    emitted: list[str] = []
    response = router.chat_stream_response([{"role": "user", "content": "hello"}], on_text_chunk=emitted.append)

    assert response.content == "first"
    assert first.calls[0]["model"] == "one-model"
    assert len(router.last_result or []) == 2
    assert router.last_aggregated is not None
    assert bridge.total_tokens == {"input": 4, "output": 6}
    assert emitted == ["first"]


def test_fallback_moves_on_after_error_and_timeout() -> None:
    bad = Provider(error=RuntimeError("rate limited"))
    slow = Provider(delay=0.1)
    good = Provider("recovered")
    router = MultiModelRouter(
        [ProviderSlot("bad", bad), ProviderSlot("slow", slow, timeout_ms=1), ProviderSlot("good", good)],
        FallbackStrategy(),
    )
    assert router.chat([{"role": "user", "content": "hello"}]).content == "recovered"
    assert [item.slot_name for item in router.last_result or []] == ["bad", "slow", "good"]
    assert "Timeout" in (router.last_result or [])[1].error


def test_first_success_selects_the_fastest_parallel_slot() -> None:
    slow, fast = Provider("slow", delay=0.05), Provider("fast", delay=0.001)
    router = MultiModelRouter(
        [ProviderSlot("slow", slow), ProviderSlot("fast", fast)],
        ParallelStrategy(),
        FirstSuccessAggregator(),
    )

    assert router.chat([{"role": "user", "content": "hello"}]).content == "fast"


def test_routing_selects_rule_target_and_sync_api_works_inside_event_loop() -> None:
    cheap, strong = Provider("cheap"), Provider("strong")
    router = MultiModelRouter(
        [ProviderSlot("cheap", cheap), ProviderSlot("strong", strong)],
        RoutingStrategy([RoutingRule(lambda _messages, _ctx: "strong")]),
        config=RouterConfig(1),
    )

    async def call_router() -> str:
        return router.chat([{"role": "user", "content": "hello"}]).content

    assert asyncio.run(call_router()) == "strong"
    assert not cheap.calls and len(strong.calls) == 1


def test_router_requires_an_enabled_slot() -> None:
    router = MultiModelRouter([ProviderSlot("off", Provider(), enabled=False)], ParallelStrategy())
    with pytest.raises(RuntimeError, match="No enabled"):
        router.chat([])


def test_router_emits_slot_progress_completion_and_weighted_vote() -> None:
    first, second = Provider("same answer"), Provider("same answer")
    events: list[tuple[str, dict]] = []
    router = MultiModelRouter(
        [ProviderSlot("one", first, weight=1), ProviderSlot("two", second, weight=3)],
        ParallelStrategy(),
        MajorityVoteAggregator(),
    )
    router.add_event_listener(lambda kind, payload: events.append((kind, payload)))
    assert router.chat([{"role": "user", "content": "hello"}]).content == "same answer"
    assert [kind for kind, _ in events].count("progress") == 2
    assert [kind for kind, _ in events].count("complete") == 2
    assert router.last_aggregated.vote_summary["majority_weight"] == 4


def test_router_passes_the_latest_user_request_to_aggregators() -> None:
    class RecordingAggregator:
        def __init__(self) -> None:
            self.context: dict = {}

        async def aggregate(self, results, context):
            self.context = context
            return AggregatedOutput(results[0].response, [], list(results))

    aggregator = RecordingAggregator()
    router = MultiModelRouter([ProviderSlot("one", Provider())], ParallelStrategy(), aggregator)

    router.chat(
        [
            {"role": "user", "content": "First request"},
            {"role": "assistant", "content": "Prior response"},
            {"role": "user", "content": "Current request"},
        ]
    )

    assert aggregator.context["user_request"] == "Current request"
