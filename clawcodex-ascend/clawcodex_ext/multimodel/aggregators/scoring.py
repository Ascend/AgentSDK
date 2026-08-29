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

"""Evaluator-backed scoring aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from clawcodex_ext.capabilities.multimodel_protocol import AggregatedOutput, MultiModelResult  # pylint: disable=no-name-in-module

from .base import (
    ScoreEvaluator,
    fallback_output,
    normalise_score,
    parse_score_json,
    require_results,
    resolve,
    score_prompt,
    valid_results,
)

if TYPE_CHECKING:
    from clawcodex_ext.providers.base import BaseProvider


@dataclass
class ScoringAggregator:
    """Select the response with the highest score from an independent judge.

    Supply ``scorer`` for a local or custom judge.  Alternatively a
    ``scorer_provider`` can be supplied and is asked to return the documented
    JSON schema.  Keeping this dependency injected prevents an aggregator
    from silently constructing a network client or requiring credentials.
    """

    scorer_model: str = "gpt-4o"
    criteria: list[str] = field(default_factory=lambda: ["correctness", "clarity", "completeness"])
    scorer: ScoreEvaluator | None = None
    scorer_provider: BaseProvider | None = None

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("criteria must contain at least one item")
        if self.scorer is not None and self.scorer_provider is not None:
            raise ValueError("provide either scorer or scorer_provider, not both")

    async def aggregate(self, results: list[MultiModelResult], context: dict[str, Any]) -> AggregatedOutput:
        require_results(results)
        valid = valid_results(results)
        if len(valid) <= 1:
            return fallback_output(results)
        # Keep the private hook's one-argument signature aligned with the
        # public design, so custom aggregators can override it directly.
        del context
        scores = [await self._score_one(result) for result in valid]
        best_index = max(range(len(scores)), key=lambda index: scores[index]["total"])
        chosen = valid[best_index]
        return AggregatedOutput(
            chosen=chosen.response,
            runners_up=[result for result in results if result is not chosen],
            provenance=list(results),
            vote_summary={
                "scores": {result.slot_name: score for result, score in zip(valid, scores)},
                "criteria": list(self.criteria),
            },
        )

    async def _score_one(self, result: MultiModelResult) -> dict[str, float]:
        if self.scorer is not None:
            raw = await resolve(self.scorer(result))
        elif self.scorer_provider is not None:
            response = await self.scorer_provider.chat_async(
                [{"role": "user", "content": score_prompt(result, self.criteria)}],
                model=self.scorer_model,
            )
            raw = parse_score_json(response.content)
        else:
            raise RuntimeError("ScoringAggregator needs a scorer or scorer_provider")
        return normalise_score(raw, self.criteria)
