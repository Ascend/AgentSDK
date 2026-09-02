#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Contracts shared by multi-model schedulers and aggregators.

The protocol module deliberately contains only the neutral data exchanged
between a router, a strategy and an aggregator.  Concrete aggregation
implementations live in :mod:`clawcodex_ext.multimodel.aggregators`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from clawcodex_ext.providers.base import ChatResponse, MessageInput


@dataclass
class MultiModelResult:
    """The complete outcome of one provider slot invocation."""

    slot_name: str
    response: ChatResponse
    duration_ms: int
    tokens: dict[str, int]
    error: str | None = None
    cancelled: bool = False
    # Absolute completion time lets aggregators distinguish actual arrival
    # order from configured slot order when calls run in parallel.
    completed_at: float = 0.0


@dataclass
class AggregatedOutput:
    """The response selected by an aggregator plus its audit trail."""

    chosen: ChatResponse
    runners_up: list[MultiModelResult]
    provenance: list[MultiModelResult]
    vote_summary: dict[str, Any] | None = None
    summary_text: str | None = None


@runtime_checkable
class MultiModelStrategy(Protocol):
    """Contract implemented by a multi-model scheduling strategy."""

    name: str

    async def execute(self, router: Any, messages: list[MessageInput], **kwargs: Any) -> list[MultiModelResult]: ...


@runtime_checkable
class AggregatorProtocol(Protocol):
    """Select one response from results produced by a model strategy."""

    async def aggregate(self, results: list[MultiModelResult], context: dict[str, Any]) -> AggregatedOutput: ...


__all__ = [
    "AggregatedOutput",
    "AggregatorProtocol",
    "MultiModelResult",
    "MultiModelStrategy",
]
