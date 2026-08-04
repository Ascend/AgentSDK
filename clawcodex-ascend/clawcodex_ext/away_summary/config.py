#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Configuration for the Away Summary extension."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AwaySummaryConfig:
    """Typed view of ``settings.away_summary``."""

    enabled: bool = True
    idle_seconds: int = 300
    recap_command_enabled: bool = True
    min_turns: int = 1
    max_input_tokens: int = 12_000
    max_output_tokens: int = 500
    persist_last_recap: bool = True
    response_language: str = "auto"
    # When True, the automatic recap prepends a short session-memory
    # block (read from the session-summary sidecar) so the model can
    # see broader context the recent transcript may have lost to
    # truncation. Off by default to keep the auto recap byte-identical
    # to its prior behaviour unless the user opts in.
    include_session_memory: bool = False
    # When True, /recap shares the parent query loop's prompt-cache
    # prefix via ``run_forked_agent`` rather than issuing an
    # independent provider call. Defaults ON; disable only if a fork
    # regression needs to be isolated.
    enable_recap_cache: bool = True

    @classmethod
    def from_mapping(cls, raw: Any) -> "AwaySummaryConfig":
        data = raw if isinstance(raw, dict) else {}
        idle_seconds = _int_value(data.get("idle_seconds"), 300)
        if "idle_minutes" in data and "idle_seconds" not in data:
            idle_seconds = _int_value(data.get("idle_minutes"), 5) * 60

        return cls(
            enabled=_bool_value(data.get("enabled"), True),
            idle_seconds=max(1, idle_seconds),
            recap_command_enabled=_bool_value(
                data.get("recap_command_enabled"),
                True,
            ),
            min_turns=max(0, _int_value(data.get("min_turns"), 1)),
            max_input_tokens=max(256, _int_value(data.get("max_input_tokens"), 12_000)),
            max_output_tokens=max(64, _int_value(data.get("max_output_tokens"), 500)),
            persist_last_recap=_bool_value(data.get("persist_last_recap"), True),
            response_language=_language_value(data.get("response_language"), "auto"),
            include_session_memory=_bool_value(
                data.get("include_session_memory"),
                False,
            ),
            enable_recap_cache=_bool_value(
                data.get("enable_recap_cache"),
                True,
            ),
        )


def load_away_summary_config(
    *,
    cwd: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AwaySummaryConfig:
    """Load Away Summary config from the existing settings hierarchy.

    The upstream-shaped ``SettingsSchema`` keeps unknown top-level settings in
    ``extra``. Reading ``away_summary`` from there lets this downstream feature
    support ``settings.away_summary`` without changing ``src/settings``.
    """

    if overrides is not None:
        return AwaySummaryConfig.from_mapping(overrides)

    try:
        from src.settings.settings import load_settings

        settings = load_settings(cwd=cwd)
        raw = getattr(settings, "extra", {}).get("away_summary", {})
        return AwaySummaryConfig.from_mapping(raw)
    except Exception:
        logger.debug(
            "Away Summary settings unavailable; using defaults",
            exc_info=True,
        )
        return AwaySummaryConfig()


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


def _language_value(value: Any, default: str) -> str:
    text = str(value or default).strip().lower()
    if text in {
        "zh",
        "zh-cn",
        "zh-hans",
        "chinese",
        "中文",
        "simplified chinese",
        "chinese simplified",
        "zhongwen",
        "zhong wen",
    }:
        return "Chinese"
    if text in {"en", "en-us", "english"}:
        return "English"
    return "auto"
