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

"""Text-similarity majority voting aggregator."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from clawcodex_ext.capabilities.multimodel_protocol import AggregatedOutput, MultiModelResult  # pylint: disable=no-name-in-module

from .base import fallback_output, require_results, valid_results


@dataclass
class MajorityVoteAggregator:
    """Cluster similar successful texts and select the largest cluster.

    Ties are intentionally resolved by original provider order, which makes
    the choice stable and lets callers express a preference through slot order.
    """

    min_votes: int = 2
    tolerance: float = 0.3

    def __post_init__(self) -> None:
        if self.min_votes < 1:
            raise ValueError("min_votes must be at least 1")
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError("tolerance must be between 0 and 1")

    async def aggregate(self, results: list[MultiModelResult], context: dict[str, Any]) -> AggregatedOutput:
        require_results(results)
        valid = valid_results(results)
        if len(valid) < self.min_votes:
            return fallback_output(results)

        clusters = self._cluster_by_similarity(valid)
        weights = context.get("slot_weights", {})

        def score(cluster: list[MultiModelResult]) -> tuple[float, int]:
            return (sum(float(weights.get(item.slot_name, 1.0)) for item in cluster), len(cluster))

        majority = max(clusters, key=score)
        chosen = majority[0]
        summary = {
            "total_votes": len(valid),
            "majority": len(majority),
            "clusters": {index: len(cluster) for index, cluster in enumerate(clusters)},
            "winning_slot": chosen.slot_name,
        }
        if any(float(weights.get(item.slot_name, 1.0)) != 1.0 for item in valid):
            summary["majority_weight"] = score(majority)[0]
        return AggregatedOutput(
            chosen=chosen.response,
            runners_up=[result for result in results if result is not chosen],
            provenance=list(results),
            vote_summary=summary,
        )

    def _cluster_by_similarity(self, results: list[MultiModelResult]) -> list[list[MultiModelResult]]:
        """Return connected components whose pairwise similarity clears tolerance."""

        parents = list(range(len(results)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left, result in enumerate(results):
            for right in range(left + 1, len(results)):
                other = results[right]
                similarity = SequenceMatcher(None, result.response.content, other.response.content).ratio()
                if similarity > self.tolerance:
                    union(left, right)

        grouped: dict[int, list[MultiModelResult]] = {}
        for index, result in enumerate(results):
            grouped.setdefault(find(index), []).append(result)
        return list(grouped.values())
