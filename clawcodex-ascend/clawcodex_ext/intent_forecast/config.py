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

"""Configuration for Intent Forecast."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IntentForecastConfig:
    enabled: bool = True
    idle_seconds: int = 120
    max_sessions: int = 12
    max_transcript_tail_messages: int = 12
    max_input_tokens: int = 16_000
    max_output_tokens: int = 800
    min_confidence: float = 0.45
    auto_display: bool = True
    feedback_enabled: bool = True
    summary_lazy_generate: bool = True
    response_language: str = "auto"
    intent_strategy: str = "user"

    @classmethod
    def from_mapping(cls, raw: Any) -> "IntentForecastConfig":
        data = raw if isinstance(raw, dict) else {}
        return cls(
            enabled=_bool_value(data.get("enabled"), True),
            idle_seconds=max(1, _int_value(data.get("idle_seconds"), 120)),
            max_sessions=max(0, _int_value(data.get("max_sessions"), 12)),
            max_transcript_tail_messages=max(
                0,
                _int_value(data.get("max_transcript_tail_messages"), 12),
            ),
            max_input_tokens=max(512, _int_value(data.get("max_input_tokens"), 16_000)),
            max_output_tokens=max(64, _int_value(data.get("max_output_tokens"), 800)),
            min_confidence=max(0.0, min(1.0, _float_value(data.get("min_confidence"), 0.45))),
            auto_display=_bool_value(data.get("auto_display"), True),
            feedback_enabled=_bool_value(data.get("feedback_enabled"), True),
            summary_lazy_generate=_bool_value(data.get("summary_lazy_generate"), True),
            response_language=_language_value(data.get("response_language"), "auto"),
            intent_strategy=_strategy_value(data.get("intent_strategy") or data.get("strategy"), "user"),
        )


def load_intent_forecast_config(
    *,
    cwd: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> IntentForecastConfig:
    if overrides is not None:
        return IntentForecastConfig.from_mapping(overrides)
    try:
        from clawcodex_ext.settings.settings import load_settings  # pylint: disable=no-name-in-module

        settings = load_settings(cwd=cwd)
        raw = getattr(settings, "extra", {}).get("intent_forecast", {})
        return IntentForecastConfig.from_mapping(raw)
    except Exception:
        return IntentForecastConfig()


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _language_value(value: Any, default: str) -> str:
    text = str(value or default).strip().lower()
    if text in {"zh", "zh-cn", "chinese", "中文", "simplified chinese"}:
        return "Chinese"
    if text in {"en", "en-us", "english"}:
        return "English"
    return "auto"


def _strategy_value(value: Any, default: str) -> str:
    text = str(value or default).strip().lower().replace("-", "_")
    aliases = {
        "user": "user",
        "user_first": "user",
        "user_priority": "user",
        "workspace": "workspace",
        "workspace_first": "workspace",
        "workspace_priority": "workspace",
        "project": "workspace",
        "project_first": "workspace",
        "history": "history",
        "history_first": "history",
        "history_priority": "history",
        "session": "history",
    }
    return aliases.get(text, default)
