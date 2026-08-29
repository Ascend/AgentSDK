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

"""Pairwise, evaluator-backed ranking aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clawcodex_ext.capabilities.multimodel_protocol import AggregatedOutput, MultiModelResult  # pylint: disable=no-name-in-module

from .base import RankEvaluator, fallback_output, require_results, resolve, valid_results


@dataclass
class RankAggregator:
    """Aggregate each candidate's peer ranking into a stable final ordering.

    ``ranker`` receives one evaluator result, all candidates, and context.  It
    may return a numeric score, or a mapping of peer slot names to scores.  A
    mapping is averaged, allowing each model (or an adapter representing it)
    to score every other response without coupling this package to a provider.
    """

    ranker: RankEvaluator | None = None

    async def aggregate(self, results: list[MultiModelResult], context: dict[str, Any]) -> AggregatedOutput:
        require_results(results)
        valid = valid_results(results)
        if len(valid) <= 1:
            return fallback_output(results)
        if self.ranker is None:
            raise RuntimeError("RankAggregator needs a ranker")

        totals = {result.slot_name: 0.0 for result in valid}
        counts = {result.slot_name: 0 for result in valid}
        for evaluator in valid:
            raw = await resolve(self.ranker(evaluator, valid, context))
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                totals[evaluator.slot_name] += float(raw)
                counts[evaluator.slot_name] += 1
            elif isinstance(raw, dict):
                for slot_name, value in raw.items():
                    if slot_name in totals and isinstance(value, (int, float)) and not isinstance(value, bool):
                        totals[slot_name] += float(value)
                        counts[slot_name] += 1
            else:
                raise TypeError("ranker must return a number or a mapping of slot scores")

        averages = {
            slot_name: totals[slot_name] / counts[slot_name] if counts[slot_name] else 0.0 for slot_name in totals
        }
        chosen = max(valid, key=lambda result: averages[result.slot_name])
        ordered = sorted(valid, key=lambda result: averages[result.slot_name], reverse=True)
        return AggregatedOutput(
            chosen=chosen.response,
            runners_up=[result for result in results if result is not chosen],
            provenance=list(results),
            vote_summary={
                "scores": averages,
                "ranking": [result.slot_name for result in ordered],
                "evaluators": len(valid),
            },
        )
