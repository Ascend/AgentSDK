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

"""Issue → mode routing strategies.

The ``Router`` Protocol lets ``ModeSelector`` consult an interchangeable
backend when it has to decide between several non-default modes. Two
implementations ship in-tree:

* ``HeuristicRouter`` — keyword matching on the issue title/description.
  Cheap, deterministic, and good enough to demonstrate routing without
  spending tokens on every poll. Used by default.
* ``LLMRouter`` — calls an OpenAI-compatible chat completions endpoint
  (deepseek by default; any OpenAI-protocol provider works) and asks
  the model to return a strict JSON decision. Falls back to single
  mode (low confidence) on any error so the daemon stays running.

A router never raises out of ``choose`` — it either returns a confident
decision or a low-confidence one with a reason; ``ModeSelector`` decides
how to handle low confidence (today: fall back to ``default_mode``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    from .tracker import Issue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterResult:
    """One router's pick. Returned by ``Router.choose``."""

    mode: str
    reason: str
    confidence: float = 0.5


@runtime_checkable
class Router(Protocol):
    """Backend ``ModeSelector`` consults when no explicit label is set."""

    def choose(self, issue: "Issue") -> RouterResult: ...


# ---------------------------------------------------------------------------
# HeuristicRouter — keyword-based, no external calls
# ---------------------------------------------------------------------------


# Ordered by specificity: the FIRST matching bucket wins. This is
# intentional — broader categories like "implement" would otherwise
# steal "implement a debate about X" → debate. Tighter buckets first.
_DEBATE_KEYWORDS: frozenset[str] = frozenset(
    {
        "design",
        "investigate",
        "research",
        "compare",
        "choose between",
        "evaluate",
        "decide",
        "tradeoff",
        "trade-off",
        "proposal",
    }
)
_COORDINATOR_KEYWORDS: frozenset[str] = frozenset(
    {
        "refactor",
        "cross-module",
        "multi-file",
        "multi-module",
        "rewrite",
        "split into",
        "extract",
        "consolidate",
        "migrate",
    }
)
_SWARM_KEYWORDS: frozenset[str] = frozenset(
    {
        "swarm",
        "decompose",
        "multi-step",
        "independent tasks",
        "parallel tasks",
        "end-to-end migration",
        "complex bug",
    }
)
_PIPELINE_KEYWORDS: frozenset[str] = frozenset(
    {
        "implement",
        "add feature",
        "build",
        "create endpoint",
        "create route",
        "introduce",
        "scaffold",
        "wire up",
    }
)


class HeuristicRouter:
    """Pick a mode based on simple keyword matches in title/description.

    Decision flow:

    1. Swarm / decomposition language → ``swarm``
    2. Design / investigation language → ``debate``
    3. Refactor / cross-module language → ``coordinator``
    4. Implementation language → ``pipeline``
    5. Otherwise → ``single`` (low confidence; selector falls back)

    Confidence is set high (0.7) for explicit keyword hits and low (0.2)
    for the catch-all, so a future selector tuning step can threshold on
    it without changing this code.
    """

    def __init__(
        self,
        *,
        confidence_match: float = 0.7,
        confidence_default: float = 0.2,
    ) -> None:
        self._confidence_match = confidence_match
        self._confidence_default = confidence_default

    def choose(self, issue: "Issue") -> RouterResult:
        try:
            text = self._extract_text(issue)
        except Exception:  # pragma: no cover — defensive
            logger.exception("HeuristicRouter: failed to read issue text")
            return RouterResult(
                mode="single",
                reason="router: could not read issue text",
                confidence=self._confidence_default,
            )

        # Order matters; see module-level keyword block.
        for label, keywords in (
            ("swarm", _SWARM_KEYWORDS),
            ("debate", _DEBATE_KEYWORDS),
            ("coordinator", _COORDINATOR_KEYWORDS),
            ("pipeline", _PIPELINE_KEYWORDS),
        ):
            hit = self._first_keyword_hit(text, keywords)
            if hit:
                return RouterResult(
                    mode=label,
                    reason=f"router: matched keyword {hit!r}",
                    confidence=self._confidence_match,
                )

        return RouterResult(
            mode="single",
            reason="router: no keyword match",
            confidence=self._confidence_default,
        )

    @staticmethod
    def _extract_text(issue: "Issue") -> str:
        title = getattr(issue, "title", "") or ""
        body = getattr(issue, "description", "") or ""
        return (title + " " + body).lower()

    @staticmethod
    def _first_keyword_hit(text: str, keywords: frozenset[str]) -> str | None:
        for k in keywords:
            if k in text:
                return k
        return None


# ---------------------------------------------------------------------------
# LLMRouter — Phase-3 hook
# ---------------------------------------------------------------------------


_LLM_ROUTER_VALID_MODES: frozenset[str] = frozenset({"single", "pipeline", "coordinator", "debate", "swarm"})


_LLM_ROUTER_SYSTEM_PROMPT: str = (
    "You are a routing decision agent for an automation orchestrator.\n\n"
    "Given an issue (title + description), choose which collaboration\n"
    "mode best fits the work:\n\n"
    "- single:      one agent handles the whole issue (simple bug fix,\n"
    "               docs, single-file changes — anything that doesn't\n"
    "               benefit from multi-stage decomposition).\n"
    "- pipeline:    three sequential stages (analyzer → implementer →\n"
    "               tester). Best for features that need a clear\n"
    "               design step before implementation, and explicit\n"
    "               testing after.\n"
    "- coordinator: one coordinator agent delegates to multiple workers.\n"
    "               Best for refactors that span many files, or work\n"
    "               that decomposes into independent sub-tasks.\n"
    "- debate:      two independent proposers + one judge. Best for\n"
    "               design questions where multiple valid approaches\n"
    "               exist and a comparison adds value.\n\n"
    "- swarm:       dynamically decomposed dependency waves executed by\n"
    "               coordinator workers. Best for complex multi-step work\n"
    "               with several bounded sub-tasks.\n\n"
    "Output ONLY a JSON object on a single line. Schema:\n"
    '  {"mode": "<one of single|pipeline|coordinator|debate|swarm>",\n'
    '   "reason": "<one sentence, under 25 words>",\n'
    '   "confidence": <float between 0.0 and 1.0>}\n\n'
    "Do not include any text outside the JSON. No code fences. No prose."
)


_LLM_ROUTER_USER_TEMPLATE: str = (
    "Issue title:\n{title}\n\nIssue description:\n{description}\n\nPick the mode for this issue and return the JSON."
)


class LLMRouter:
    """Routes via an OpenAI-compatible chat completions endpoint.

    Used when ``workflow.modes.router.kind == "llm"``. Sends one
    short chat completion call per issue (~200 input tokens, ~50
    output) asking the model to pick a mode + give a reason +
    self-rate confidence. Strict JSON output, low temperature.

    Defaults match the deepseek environment the orchestrator already
    uses for issue runs (workflow.md ``agent.provider == "deepseek"``),
    but any OpenAI-protocol endpoint works — pass a different
    ``endpoint`` / ``api_key_env_var`` to swap providers.

    Failure modes (all → low-confidence ``single`` fallback so the
    ``ModeSelector`` reverts to its default mode):

    * Network error / non-200 response.
    * JSON decode failure.
    * Model returns a mode not in ``_LLM_ROUTER_VALID_MODES``.
    * Missing API key (env var unset).
    * Timeout (default 15 s).
    """

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-flash",
        endpoint: str = "https://api.deepseek.com/chat/completions",
        api_key_env_var: str = "DEEPSEEK_API_KEY",
        timeout_seconds: float = 15.0,
        # Injectable HTTP layer for testing. When ``None`` we lazily
        # create a fresh ``httpx.Client`` per call (no shared session
        # because LLMRouter.choose is sync + called from sync code).
        http_client: Any | None = None,
    ) -> None:
        self._model = model
        self._endpoint = endpoint
        self._api_key_env_var = api_key_env_var
        self._timeout = timeout_seconds
        self._http_client = http_client

    # ------------------------------------------------------------------

    def choose(self, issue: "Issue") -> RouterResult:
        api_key = os.environ.get(self._api_key_env_var, "").strip()
        if not api_key:
            return RouterResult(
                mode="single",
                reason=(f"LLMRouter: env var {self._api_key_env_var} is unset; cannot call provider"),
                confidence=0.1,
            )

        try:
            response_text = self._post(api_key, issue)
        except Exception as exc:
            logger.warning("LLMRouter: provider call failed (%s); falling back", exc)
            return RouterResult(
                mode="single",
                reason=f"LLMRouter: provider call raised — {type(exc).__name__}",
                confidence=0.1,
            )

        return self._parse(response_text)

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    def _post(self, api_key: str, issue: "Issue") -> str:
        body = self._build_chat_body(issue)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self._http_client is None:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._endpoint, json=body, headers=headers)
        else:
            resp = self._http_client.post(self._endpoint, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible shape: choices[0].message.content
        content = data["choices"][0]["message"]["content"]
        return str(content).strip()

    def _build_chat_body(self, issue: "Issue") -> dict[str, Any]:
        title = getattr(issue, "title", "") or "(no title)"
        description = (getattr(issue, "description", "") or "(no description)")[
            :4000  # cap so we don't blow context for huge issues
        ]
        user_msg = _LLM_ROUTER_USER_TEMPLATE.format(title=title, description=description)
        return {
            "model": self._model,
            "temperature": 0.0,
            "max_tokens": 200,
            "messages": [
                {"role": "system", "content": _LLM_ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        }

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse(self, response_text: str) -> RouterResult:
        # Models occasionally wrap JSON in ```json ... ``` fences even
        # when told not to. Strip the fence before parsing.
        cleaned = _strip_json_fences(response_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("LLMRouter: model returned non-JSON output: %r", response_text[:200])
            return RouterResult(
                mode="single",
                reason="LLMRouter: model returned non-JSON output",
                confidence=0.1,
            )

        mode = str(data.get("mode", "")).strip().lower()
        if mode not in _LLM_ROUTER_VALID_MODES:
            logger.warning("LLMRouter: model picked unknown mode=%r; falling back", mode)
            return RouterResult(
                mode="single",
                reason=f"LLMRouter: model picked unknown mode={mode!r}",
                confidence=0.1,
            )

        reason = str(data.get("reason", "")).strip() or "LLMRouter: no reason given"
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        return RouterResult(
            mode=mode,
            reason=f"LLMRouter: {reason}",
            confidence=confidence,
        )


_JSON_FENCE_RE: re.Pattern[str] = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_json_fences(text: str) -> str:
    """Remove ``` or ```json fences around a JSON payload, if present."""
    text = text.strip()
    m = _JSON_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


__all__ = ["HeuristicRouter", "LLMRouter", "Router", "RouterResult"]
