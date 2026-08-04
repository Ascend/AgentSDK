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

"""Turn persistent model-group settings into a runtime provider."""

from __future__ import annotations

from typing import Any, Callable

from .aggregators.base import parse_score_json
from .session_bridge import SessionBridge
from .feature import require_multimodel_enabled

from .aggregators import (
    FirstSuccessAggregator,
    FusionAggregator,
    MajorityVoteAggregator,
    PassThroughAggregator,
    RankAggregator,
    ScoringAggregator,
)
from .config import GroupConfig
from .router import MultiModelRouter, RouterConfig
from .slots import ProviderSlot
from .strategies import FallbackStrategy, ParallelStrategy, RoutingRule, RoutingStrategy, VotingStrategy


def build_router(
    group: GroupConfig, provider_builder: Callable[[str, str], object], *, audit_path=None
) -> MultiModelRouter:
    """Build a router from a validated group without importing runtime modules.

    ``provider_builder`` is injected to keep configuration portable and make
    this construction path straightforward to test without credentials.
    """
    require_multimodel_enabled()
    slots = [
        ProviderSlot(
            name=item.name,
            provider=provider_builder(item.provider, item.model),  # type: ignore[arg-type]
            model=item.model,
            weight=item.weight,
            timeout_ms=item.timeout_ms,
            enabled=item.enabled,
        )
        for item in group.slots
    ]
    aggregator = _aggregator(group, provider_builder, slots)
    if group.strategy == "parallel":
        strategy = ParallelStrategy()
    elif group.strategy == "voting":
        strategy = VotingStrategy(aggregator=aggregator or MajorityVoteAggregator(group.min_votes or 2))
        aggregator = None
    elif group.strategy == "fallback":
        strategy = FallbackStrategy()
    elif group.strategy == "routing":

        def text_of(messages):
            return "\n".join(
                str(item.get("content", "")) if isinstance(item, dict) else str(getattr(item, "content", ""))
                for item in messages
            ).lower()

        rules = [
            RoutingRule(
                lambda messages, _context, route=route: route.slot
                if route.pattern.lower() in text_of(messages)
                else "",
                description=f"keyword {route.pattern!r} -> {route.slot}",
            )
            for route in group.routes
        ]
        strategy = RoutingStrategy(rules=rules, fallback_slot=slots[0].name)
    else:  # validate_group keeps this defensive branch unreachable
        raise ValueError(f"unknown multi-model strategy: {group.strategy}")
    return MultiModelRouter(
        slots,
        strategy,
        aggregator,
        config=RouterConfig(group.max_concurrent),
        session_bridge=SessionBridge(audit_path=audit_path),
    )


def _aggregator(group: GroupConfig, provider_builder: Callable[[str, str], object], slots: list[ProviderSlot]):
    if group.aggregator is None:
        return None
    if group.aggregator == "passthrough":
        return PassThroughAggregator()
    if group.aggregator == "first_success":
        return FirstSuccessAggregator()
    if group.aggregator == "majority":
        return MajorityVoteAggregator(group.min_votes or 2)
    if group.aggregator == "rank":
        providers = {slot.name: slot.provider for slot in slots}

        async def ranker(evaluator, candidates, _context):
            provider: Any = providers[evaluator.slot_name]
            prompt = (
                "Rank every candidate below from 1 to 10. Return only a JSON object mapping slot names to scores.\n\n"
                + "\n\n".join(f"[{candidate.slot_name}]\n{candidate.response.content}" for candidate in candidates)
            )
            response = await provider.chat_async([{"role": "user", "content": prompt}])
            return parse_score_json(response.content)

        return RankAggregator(ranker=ranker)
    if group.aggregator == "scoring":
        return ScoringAggregator(
            scorer_model=group.scorer_model,
            scorer_provider=provider_builder(group.scorer_provider, group.scorer_model),  # type: ignore[arg-type]
        )
    if group.aggregator == "fusion":
        return FusionAggregator(
            fusion_model=group.scorer_model,
            fusion_provider=provider_builder(group.scorer_provider, group.scorer_model),  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown multi-model aggregator: {group.aggregator}")
