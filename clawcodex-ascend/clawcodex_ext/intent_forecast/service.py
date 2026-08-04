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

"""Intent Forecast generation service."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from clawcodex_ext.intent_forecast.config import IntentForecastConfig
from clawcodex_ext.intent_forecast.context import ForecastContext, IntentForecastContextBuilder
from clawcodex_ext.intent_forecast.fallback import fallback_suggestions
from clawcodex_ext.intent_forecast.focus import compute_workspace_focuses, suggestion_matches_focus
from clawcodex_ext.intent_forecast.learning import build_feedback_features, feedback_weight
from clawcodex_ext.intent_forecast.messages import ForecastResult, ForecastSuggestion
from clawcodex_ext.intent_forecast.prompt import build_forecast_messages


class IntentForecastService:
    def __init__(
        self,
        *,
        conversation: Any | None,
        provider: Any | None,
        model: str | None,
        workspace_root: Path,
        config: IntentForecastConfig | None = None,
        context: ForecastContext | None = None,
    ) -> None:
        self.conversation = conversation
        self.provider = provider
        self.model = model
        self.workspace_root = Path(workspace_root)
        self.config = config or IntentForecastConfig()
        self.context = context

    def generate(self, *, trigger: str, force: bool = False) -> ForecastResult:
        if not force and not self.config.enabled:
            return ForecastResult(generated=False, suggestions=[], reason="Intent Forecast is disabled.")
        context = (
            self.context
            or IntentForecastContextBuilder(
                conversation=self.conversation,
                workspace_root=self.workspace_root,
                config=self.config,
            ).build()
        )
        suggestions: list[ForecastSuggestion] = []
        if self.provider is not None:
            messages = build_forecast_messages(context, max_input_tokens=self.config.max_input_tokens)
            try:
                response = self.provider.chat(
                    messages=messages,
                    tools=None,
                    model=self.model,
                    max_tokens=self.config.max_output_tokens,
                )
            except TypeError:
                response = self.provider.chat(messages=messages, tools=None, model=self.model)
            raw = str(getattr(response, "content", "") or "")
            suggestions = parse_forecast_response(raw, min_confidence=self.config.min_confidence)
            suggestions = filter_suggestions_for_context(suggestions, context)

        if not suggestions:
            suggestions = fallback_suggestions(context, min_confidence=self.config.min_confidence)

        suggestions = no_suggestion_gate(
            suggestions,
            context=context,
            trigger=trigger,
            min_confidence=self.config.min_confidence,
        )
        if not suggestions:
            return ForecastResult(
                generated=False,
                suggestions=[],
                reason="No confident next-step suggestions are available.",
                fingerprint=context.fingerprint,
            )
        suggestions = rank_suggestions_with_feedback(
            suggestions,
            cwd=context.cwd,
            fingerprint=context.fingerprint,
            context=context,
            trigger=trigger,
        )
        return ForecastResult(generated=True, suggestions=suggestions[:3], fingerprint=context.fingerprint)


def parse_forecast_response(raw: str, *, min_confidence: float) -> list[ForecastSuggestion]:
    data = _loads_json(raw)
    if not isinstance(data, dict):
        return []
    raw_suggestions = data.get("suggestions")
    if not isinstance(raw_suggestions, list):
        return []
    suggestions: list[ForecastSuggestion] = []
    for item in raw_suggestions:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not title or not prompt:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            continue
        refs = item.get("source_refs")
        if not isinstance(refs, list):
            refs = []
        suggestions.append(
            ForecastSuggestion(
                id=str(item.get("id") or f"forecast-{uuid.uuid4().hex[:10]}"),
                title=title[:160],
                prompt=prompt,
                reason=str(item.get("reason") or "").strip()[:300],
                confidence=max(0.0, min(1.0, confidence)),
                source_refs=[str(ref)[:120] for ref in refs[:8]],
            )
        )
    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    return suggestions


def no_suggestion_gate(
    suggestions: list[ForecastSuggestion],
    *,
    context: ForecastContext,
    trigger: str,
    min_confidence: float,
) -> list[ForecastSuggestion]:
    """Suppress weak automatic suggestions when local evidence is thin."""

    if not suggestions:
        return []
    manual = trigger in {"slash", "cli", "manual", "test"}
    task_state = context.task_state or {}
    if str(context.intent_stage or "") == "pause" and not manual:
        return []
    threshold = min_confidence if manual else min(0.95, max(min_confidence, min_confidence + 0.08))
    strong_signal = bool(
        context.current_messages
        or context.workspace.get("git_status")
        or context.workspace.get("recent_commits")
        or task_state.get("blocked_reason")
        or task_state.get("pending_tests")
        or any(float(session.get("relevance_score") or 0) >= 0.5 for session in context.sessions[:3])
    )
    if not strong_signal and not manual:
        return []
    return [suggestion for suggestion in suggestions if suggestion.confidence >= threshold]


def filter_suggestions_for_context(
    suggestions: list[ForecastSuggestion],
    context: ForecastContext,
) -> list[ForecastSuggestion]:
    filtered: list[ForecastSuggestion] = []
    require_chinese = context.response_language.lower().startswith("chinese")
    permission_change_allowed = _permission_change_allowed(context)
    focuses = _workspace_focuses(context)
    for suggestion in suggestions:
        text = " ".join([suggestion.title, suggestion.prompt, suggestion.reason]).lower()
        if _is_permission_mode_suggestion(text) and not permission_change_allowed:
            continue
        if require_chinese and not _contains_cjk(text):
            continue
        if not _passes_workspace_focus(suggestion, text, focuses):
            continue
        filtered.append(suggestion)
    return filtered


def rank_suggestions_with_feedback(
    suggestions: list[ForecastSuggestion],
    *,
    cwd: str,
    fingerprint: str,
    context: ForecastContext | None = None,
    trigger: str = "",
) -> list[ForecastSuggestion]:
    ranked: list[ForecastSuggestion] = []
    for suggestion in suggestions:
        features = build_feedback_features(suggestion=suggestion, context=context, trigger=trigger)
        adjusted = max(
            0.0,
            min(
                1.0,
                suggestion.confidence
                + feedback_weight(
                    suggestion.title,
                    cwd=cwd,
                    fingerprint=fingerprint,
                    features=features,
                ),
            ),
        )
        ranked.append(
            ForecastSuggestion(
                id=suggestion.id,
                title=suggestion.title,
                prompt=suggestion.prompt,
                reason=suggestion.reason,
                confidence=adjusted,
                source_refs=suggestion.refs(),
            )
        )
    ranked.sort(key=lambda s: s.confidence, reverse=True)
    return ranked


def _is_permission_mode_suggestion(text: str) -> bool:
    return (
        "permission mode" in text
        or "bypasspermissions" in text
        or "dontask" in text
        or "switch permission" in text
        or "\u6743\u9650\u6a21\u5f0f" in text
    )


def _permission_change_allowed(context: ForecastContext) -> bool:
    texts: list[str] = []
    for msg in context.current_messages[-8:]:
        texts.append(str(msg.get("content") or ""))
    for session in context.sessions[:4]:
        tail = session.get("transcript_tail")
        if isinstance(tail, list):
            for item in tail[-4:]:
                if isinstance(item, dict):
                    texts.append(str(item.get("content") or ""))
    haystack = "\n".join(texts).lower()
    blocked_markers = (
        "permission denied",
        "blocked by permission",
        "permissions blocked",
        "not permitted",
        "user denied",
        "\u6743\u9650\u963b\u6b62",
        "\u6743\u9650\u62d2\u7edd",
        "\u88ab\u6743\u9650\u963b\u6b62",
    )
    return any(marker in haystack for marker in blocked_markers)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _primary_workspace_focus(context: ForecastContext) -> dict[str, Any] | None:
    focuses = _workspace_focuses(context)
    return focuses[0] if focuses else None


def _workspace_focuses(context: ForecastContext) -> list[dict[str, Any]]:
    existing = context.workspace.get("focuses")
    if isinstance(existing, list):
        return [item for item in existing if isinstance(item, dict)]
    return compute_workspace_focuses(
        changed_files=[str(path) for path in context.workspace.get("changed_files") or []],
        recent_messages=context.current_messages,
    )


def _passes_workspace_focus(
    suggestion: ForecastSuggestion,
    text: str,
    focuses: list[dict[str, Any]],
) -> bool:
    if not focuses:
        return True
    top_confidence = float(focuses[0].get("confidence") or 0.0)
    if top_confidence < 0.55 and len(focuses) < 2:
        return True
    refs_text = " ".join(suggestion.refs()).lower()
    combined = f"{text} {refs_text}"
    if any(suggestion_matches_focus(combined, suggestion.refs(), focus) for focus in focuses):
        return True
    generic_current_work = (
        "current workspace" in combined
        or "current changes" in combined
        or "git changes" in combined
        or "changed files" in combined
        or "\u5f53\u524d\u5de5\u4f5c\u533a" in combined
        or "\u5f53\u524d\u53d8\u66f4" in combined
    )
    if generic_current_work:
        return True
    return False


def _loads_json(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
