#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=no-name-in-module

"""LLM-as-Judge validator.

Uses an LLM to score stage output and judge pass/fail against a threshold.
Used for automatic scoring in threshold GATE mode.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from . import ValidationResult

logger = logging.getLogger(__name__)


# -- LLM Judge configuration --──────────────────────────────────────────────────


class LLMJudgeConfig:
    """LLM Judge configuration."""

    def __init__(
        self,
        threshold: float = 0.7,
        rubric: str = "",
        model: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.threshold = threshold
        self.rubric = rubric
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "LLMJudgeConfig":
        """Build configuration from a validator spec."""

        def _as_float(key: str, default: float) -> float:
            try:
                return float(spec.get(key, default))
            except (TypeError, ValueError):
                logger.warning("LLM judge spec: invalid %s=%r, using default %s", key, spec.get(key), default)
                return default

        def _as_int(key: str, default: int) -> int:
            try:
                return int(spec.get(key, default))
            except (TypeError, ValueError):
                logger.warning("LLM judge spec: invalid %s=%r, using default %s", key, spec.get(key), default)
                return default

        return cls(
            threshold=_as_float("threshold", 0.7),
            rubric=str(spec.get("rubric", "")),
            model=str(spec.get("model", "")),
            max_tokens=_as_int("max_tokens", 256),
            temperature=_as_float("temperature", 0.0),
        )


# -- LLM Judge validator --────────────────────────────────────────────────


async def validate_llm_judge(
    spec: dict[str, Any],
    llm_client: Any = None,
) -> ValidationResult:
    """Score stage output using an LLM.

    spec format:
    {
        "type": "llm_judge",
        "path": "output.md",          # File to evaluate
        "threshold": 0.7,             # Pass threshold
        "rubric": "...",             # Evaluation rubric
        "model": "gpt-4",             # Optional: LLM model
        "max_tokens": 256,            # Optional
        "temperature": 0.0,           # Optional
    }

    Args:
        spec: validator spec dict
        llm_client: LLM client (optional; returns a default low score if absent)

    Returns:
        ValidationResult: validation result
    """
    config = LLMJudgeConfig.from_spec(spec)
    path = spec.get("path", "")

    if not path:
        return ValidationResult(
            passed=False,
            message="llm_judge: no path specified",
            validator_type="llm_judge",
        )

    file_path = Path(path)
    if not file_path.exists():
        return ValidationResult(
            passed=False,
            message=f"llm_judge: file not found: {path}",
            validator_type="llm_judge",
        )

    # Read file content
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        return ValidationResult(
            passed=False,
            message=f"llm_judge: failed to read file: {exc}",
            validator_type="llm_judge",
        )

    if not content.strip():
        return ValidationResult(
            passed=False,
            message="llm_judge: file is empty",
            validator_type="llm_judge",
            score=0.0,
        )

    # Call LLM for evaluation
    score: float | None = None
    reasoning = ""

    if llm_client is not None:
        try:
            score, reasoning = await _call_llm_judge(llm_client, content, config)
        except Exception as exc:
            logger.warning("LLM judge call failed: %s, using fallback scoring", exc)
            score = None

    if score is None:
        score = _fallback_scoring(content, config)
        reasoning = "Fallback heuristic scoring (no LLM client available)"

    passed = score >= config.threshold

    return ValidationResult(
        passed=passed,
        message=f"llm_judge: score={score:.2f} {'>=' if passed else '<'} threshold={config.threshold}",
        validator_type="llm_judge",
        score=score,
        detail={"reasoning": reasoning, "threshold": config.threshold},
    )


async def _call_llm_judge(
    llm_client: Any,
    content: str,
    config: LLMJudgeConfig,
) -> tuple[float, str]:
    """Call the LLM to produce a score.

    Args:
        llm_client: LLM client (must support chat/completion interface)
        content: content to evaluate
        config: judge configuration

    Returns:
        (score, reasoning)
    """
    rubric = config.rubric or (
        "Evaluate the following output on a scale of 0.0 to 1.0. "
        "Consider completeness, correctness, clarity, and adherence to requirements. "
        'Respond with a JSON object: {"score": <float>, "reasoning": "<explanation>"}'
    )

    if len(content) > 3000:
        logger.warning("LLM judge: content truncated from %d to 3000 chars", len(content))

    prompt = f"""{rubric}

Output to evaluate:
---
{content[:3000]}
---

Respond ONLY with valid JSON: {{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    # Try multiple LLM client interfaces
    raw_response = None

    if hasattr(llm_client, "complete"):
        raw_response = await llm_client.complete(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    elif hasattr(llm_client, "chat"):
        raw_response = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    elif hasattr(llm_client, "generate"):
        raw_response = await llm_client.generate(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    else:
        logger.warning("LLM client does not support known interfaces")
        return 0.0, "LLM client interface not supported"

    # Parse response
    response_text = ""
    if isinstance(raw_response, str):
        response_text = raw_response
    elif hasattr(raw_response, "text"):
        response_text = raw_response.text
    elif hasattr(raw_response, "content"):
        response_text = raw_response.content
    elif isinstance(raw_response, dict):
        response_text = raw_response.get("text", "") or raw_response.get("content", "")

    return _parse_llm_response(response_text)


def _parse_llm_response(response: str) -> tuple[float, str]:
    """Parse the LLM response to extract score and reasoning."""
    # Try JSON parsing (robust to nested braces in reasoning)
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            score = float(data.get("score", 0.0))
            reasoning = str(data.get("reasoning", ""))
            return max(0.0, min(1.0, score)), reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fall back to extracting the first `{` to the last `}` block
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(response[start : end + 1])
            if isinstance(data, dict):
                score = float(data.get("score", 0.0))
                reasoning = str(data.get("reasoning", ""))
                return max(0.0, min(1.0, score)), reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try regex score extraction
    score_match = re.search(r"(?:score)[:\s]*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
            if 0.0 <= score <= 1.0:
                return score, response
            # Possibly a 0-100 scale
            if 0 <= score <= 100:
                return score / 100.0, response
        except ValueError:
            pass

    return 0.0, response


def _fallback_scoring(content: str, config: LLMJudgeConfig) -> float:
    """Fallback scoring: basic heuristics based on content length and structure.

    Used when the LLM is unavailable.
    """
    score = 0.0

    # Content is non-empty
    if content.strip():
        score += 0.3

    # Content length is reasonable
    if len(content) > 100:
        score += 0.2

    # Contains structured markers
    if re.search(r"#{1,3}\s", content):  # Markdown headings
        score += 0.1
    if re.search(r"```", content):  # Code blocks
        score += 0.1
    if re.search(r"[-*]\s", content):  # Lists
        score += 0.1

    # Contains keywords (common in technical docs)
    tech_keywords = [
        "implementation",
        "design",
        "test",
        "result",
        "output",
        "summary",
        "conclusion",
        "analysis",
        "data",
        "config",
    ]
    found = sum(1 for kw in tech_keywords if kw.lower() in content.lower())
    if found > 0:
        score += min(found * 0.05, 0.2)

    return min(score, 1.0)
