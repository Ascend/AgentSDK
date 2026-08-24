# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
"""Strict-but-fail-open response parsing for F-124."""

from __future__ import annotations

import json
import re
from typing import Any

from .models import ClarifyQuestion, ClarifyResult


def parse_clarify_response(
    raw: str,
    *,
    min_confidence: float = 0.7,
    max_questions: int = 3,
) -> ClarifyResult:
    data = _loads_json(raw)
    if not isinstance(data, dict):
        return _degraded_clear("provider returned non-JSON output")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if confidence < min_confidence:
        return ClarifyResult(
            is_clear=True,
            confidence=confidence,
            reason="clarifier confidence below blocking threshold",
            degraded=True,
        )

    rows = data.get("ambiguities")
    if not isinstance(rows, list):
        rows = []
    questions: list[ClarifyQuestion] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = ClarifyQuestion.from_dict(row)
        if question.question:
            questions.append(question)
        if len(questions) >= max(1, int(max_questions)):
            break

    raw_is_clear = data.get("is_clear")
    if not isinstance(raw_is_clear, bool):
        return _degraded_clear("provider returned non-boolean is_clear", confidence)
    is_clear = raw_is_clear
    if is_clear:
        questions = []
    elif not questions:
        return _degraded_clear("unclear response contained no actionable questions", confidence)

    return ClarifyResult(
        is_clear=is_clear,
        ambiguities=tuple(questions),
        confidence=confidence,
        reason="provider analysis",
    )


def _degraded_clear(reason: str, confidence: float = 0.0) -> ClarifyResult:
    return ClarifyResult(
        is_clear=True,
        confidence=confidence,
        reason=reason,
        degraded=True,
    )


def _loads_json(raw: str) -> Any:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


__all__ = ["parse_clarify_response"]
