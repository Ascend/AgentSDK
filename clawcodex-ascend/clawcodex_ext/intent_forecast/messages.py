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

"""Models and display helpers for Intent Forecast."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ForecastSuggestion:
    id: str
    title: str
    prompt: str
    reason: str = ""
    confidence: float = 0.0
    source_refs: list[str] | None = None

    def refs(self) -> list[str]:
        return list(self.source_refs or [])


@dataclass(frozen=True)
class ForecastResult:
    generated: bool
    suggestions: list[ForecastSuggestion]
    reason: str = ""
    fingerprint: str = ""


def format_forecast_for_display(result: ForecastResult) -> str:
    if not result.generated or not result.suggestions:
        return "Forecast: no suggestions right now."
    lines = ["Forecast"]
    for idx, suggestion in enumerate(result.suggestions, 1):
        lines.append("")
        lines.append(f"{idx}. {suggestion.title}")
        if suggestion.reason:
            lines.append(f"   {suggestion.reason}")
    lines.append("")
    lines.append("Use /forecast accept <number> to submit a suggestion, or /forecast dismiss.")
    return "\n".join(lines).strip()


def parse_selection(raw: str, suggestions: list[ForecastSuggestion]) -> ForecastSuggestion | None:
    token = raw.strip()
    if not token:
        # Bandit B105 is a false positive: this is the default UI selection.
        token = "1"  # nosec B105
    try:
        idx = int(token)
        if 1 <= idx <= len(suggestions):
            return suggestions[idx - 1]
    except ValueError:
        pass  # Invalid candidate; continue with the surrounding fallback.
    for suggestion in suggestions:
        if suggestion.id == token:
            return suggestion
    return None


def result_to_dict(result: ForecastResult) -> dict[str, Any]:
    return {
        "generated": result.generated,
        "reason": result.reason,
        "fingerprint": result.fingerprint,
        "suggestions": [
            {
                "id": s.id,
                "title": s.title,
                "prompt": s.prompt,
                "reason": s.reason,
                "confidence": s.confidence,
                "source_refs": s.refs(),
            }
            for s in result.suggestions
        ],
    }


def create_forecast_system_message(
    result: ForecastResult,
    *,
    trigger: str = "auto",
) -> Any:
    """Create a SystemMessage with subtype='intent_forecast' for persistence.

    Mirrors ``create_away_summary_message`` so forecast content survives
    REPL→TUI transitions and ``--resume`` replay.
    """
    from clawcodex_ext.types.messages import SystemMessage

    text = format_forecast_for_display(result)
    msg = SystemMessage(
        content=text,
        timestamp=datetime.now().isoformat(),
        subtype="intent_forecast",
        level="info",
        isMeta=False,
    )
    msg._forecast_meta = {
        "trigger": trigger,
        "fingerprint": result.fingerprint,
        "suggestion_count": len(result.suggestions),
        "generated": result.generated,
        "created_at": msg.timestamp,
    }
    return msg
