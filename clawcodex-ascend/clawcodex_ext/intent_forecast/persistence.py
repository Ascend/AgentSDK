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

"""Persistent, context-isolated forecast history."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from clawcodex_ext.intent_forecast.messages import (
    ForecastResult,
    ForecastSuggestion,
    result_to_dict,
)


def forecast_history_path(base_dir: Path | None = None) -> Path:
    root = base_dir or (Path.home() / ".clawcodex")
    return root / "intent_forecast" / "history.jsonl"


def save_forecast_result(
    result: ForecastResult,
    *,
    trigger: str,
    cwd: str | Path | None = None,
    model: str | None = None,
    stale: bool = False,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Append a forecast result to history without adding it to model context."""

    row: dict[str, Any] = {
        "record_id": f"forecast-record-{uuid.uuid4().hex[:12]}",
        "created_at": time.time(),
        "trigger": trigger,
        "cwd": str(cwd or ""),
        "model": model or "",
        "stale": bool(stale),
        "result": result_to_dict(result),
    }
    path = forecast_history_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def read_forecast_history(
    *,
    limit: int = 50,
    cwd: str | Path | None = None,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    path = forecast_history_path(base_dir)
    if not path.exists():
        return []
    cwd_text = str(cwd) if cwd is not None else ""
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines[-max(1, limit * 4) :]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if cwd_text and str(row.get("cwd") or "") != cwd_text:
            continue
        rows.append(row)
    return rows[-limit:]


def load_latest_forecast(
    *,
    cwd: str | Path | None = None,
    include_stale: bool = False,
    base_dir: Path | None = None,
) -> ForecastResult | None:
    for row in reversed(read_forecast_history(limit=200, cwd=cwd, base_dir=base_dir)):
        if row.get("stale") and not include_stale:
            continue
        result = _result_from_mapping(row.get("result"))
        if result is not None and result.suggestions:
            return result
    return None


def _result_from_mapping(raw: Any) -> ForecastResult | None:
    if not isinstance(raw, dict):
        return None
    raw_suggestions = raw.get("suggestions")
    if not isinstance(raw_suggestions, list):
        return None
    suggestions: list[ForecastSuggestion] = []
    for item in raw_suggestions:
        if not isinstance(item, dict):
            continue
        suggestions.append(
            ForecastSuggestion(
                id=str(item.get("id") or ""),
                title=str(item.get("title") or ""),
                prompt=str(item.get("prompt") or ""),
                reason=str(item.get("reason") or ""),
                confidence=float(item.get("confidence") or 0.0),
                source_refs=[str(ref) for ref in item.get("source_refs") or []],
            )
        )
    return ForecastResult(
        generated=bool(raw.get("generated", bool(suggestions))),
        suggestions=suggestions,
        reason=str(raw.get("reason") or ""),
        fingerprint=str(raw.get("fingerprint") or ""),
    )


__all__ = [
    "forecast_history_path",
    "load_latest_forecast",
    "read_forecast_history",
    "save_forecast_result",
]
