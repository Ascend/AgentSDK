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

"""Shared helpers and evaluator contracts for multi-model aggregators."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias

from clawcodex_ext.capabilities.multimodel_protocol import (  # pylint: disable=no-name-in-module
    AggregatedOutput,
    MultiModelResult,
)

Score: TypeAlias = Mapping[str, float | int]
ScoreEvaluator: TypeAlias = Callable[[MultiModelResult], Awaitable[Score] | Score]
RankEvaluator: TypeAlias = Callable[
    [MultiModelResult, list[MultiModelResult], dict[str, Any]],
    Awaitable[Mapping[str, float | int] | float | int] | Mapping[str, float | int] | float | int,
]


def valid_results(results: list[MultiModelResult]) -> list[MultiModelResult]:
    """Return completed, successful results in their original order."""

    return [result for result in results if result.error is None and not result.cancelled]


def require_results(results: list[MultiModelResult]) -> None:
    if not results:
        raise ValueError("cannot aggregate an empty result set")


def no_successful_results_error(results: list[MultiModelResult]) -> RuntimeError:
    """Build a diagnostic error for an exhausted provider group."""

    require_results(results)
    failures = "; ".join(
        f"{result.slot_name}: {result.error or ('cancelled' if result.cancelled else 'no usable response')}"
        for result in results
    )
    return RuntimeError(f"No successful provider responses ({failures})")


def fallback_output(results: list[MultiModelResult]) -> AggregatedOutput:
    """Choose the first usable result without turning failures into success."""

    require_results(results)
    valid = valid_results(results)
    if not valid:
        raise no_successful_results_error(results)
    chosen_result = valid[0]
    return AggregatedOutput(
        chosen=chosen_result.response,
        runners_up=[result for result in results if result is not chosen_result],
        provenance=list(results),
    )


async def resolve(value: Any) -> Any:
    """Await evaluator output only when the supplied evaluator is async."""

    return await value if inspect.isawaitable(value) else value


def normalise_score(raw: Mapping[str, float | int], criteria: list[str]) -> dict[str, float]:
    """Validate evaluator output and derive a total when it omits one."""

    if not isinstance(raw, Mapping):
        raise TypeError("the scoring evaluator must return a mapping")
    scores: dict[str, float] = {}
    for criterion in criteria:
        value = raw.get(criterion)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"scoring evaluator did not return a numeric {criterion!r} score")
        scores[criterion] = float(value)
    total = raw.get("total")
    if total is None:
        total = sum(scores.values()) / len(scores) if scores else 0.0
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        raise ValueError("scoring evaluator returned a non-numeric total")
    scores["total"] = float(total)
    return scores


def score_prompt(result: MultiModelResult, criteria: list[str]) -> str:
    """Build the documented judge prompt for provider-backed evaluators."""

    return (
        "Rate the following response on a scale of 1-10 for each criterion: "
        f"{', '.join(criteria)}.\n\nResponse to evaluate:\n"
        f"{result.response.content}\n\n"
        'Return JSON: {"<criterion>": <score>, "total": <average>}'
    )


def parse_score_json(content: str) -> Mapping[str, float | int]:
    """Parse a judge response, accepting a JSON object inside a code fence."""

    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("scoring provider returned invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("scoring provider JSON must be an object")
    return value
