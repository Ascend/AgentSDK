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

"""Feedback persistence for Intent Forecast."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from clawcodex_ext.intent_forecast.messages import ForecastSuggestion


def feedback_path(base_dir: Path | None = None) -> Path:
    root = base_dir or (Path.home() / ".clawcodex")
    return root / "intent_forecast" / "feedback.jsonl"


def record_feedback(
    event: str,
    *,
    suggestion: ForecastSuggestion | None = None,
    cwd: str | Path | None = None,
    fingerprint: str = "",
    features: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> None:
    path = feedback_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "event": event,
        "cwd": str(cwd or ""),
        "fingerprint": fingerprint,
        "features": features or {},
        "created_at": time.time(),
    }
    if suggestion is not None:
        payload.update(
            {
                "suggestion_id": suggestion.id,
                "title": suggestion.title,
                "prompt": suggestion.prompt,
                "confidence": suggestion.confidence,
            }
        )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_recent_feedback(*, limit: int = 50, base_dir: Path | None = None) -> list[dict[str, Any]]:
    path = feedback_path(base_dir)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    except OSError:
        return []
    return rows


def feedback_weight(
    suggestion_title: str,
    *,
    cwd: str | Path | None = None,
    fingerprint: str = "",
    features: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> float:
    """Return a small ranking adjustment from recent feedback."""

    title = suggestion_title.strip().lower()
    active_features = features or {}
    if not title and not active_features:
        return 0.0
    weight = 0.0
    cwd_text = str(cwd or "")
    for row in read_recent_feedback(limit=200, base_dir=base_dir):
        row_title = str(row.get("title") or "").strip().lower()
        same_cwd = not cwd_text or str(row.get("cwd") or "") == cwd_text
        if not same_cwd:
            continue
        title_similar = bool(row_title and title and (title in row_title or row_title in title))
        feature_similarity = feedback_feature_similarity(active_features, row.get("features"))
        if not title_similar and feature_similarity < 0.45:
            continue
        event = str(row.get("event") or "")
        multiplier = 1.0 if title_similar else feature_similarity
        if event == "accepted_completed":
            weight += 0.1 * multiplier
        elif event in {"accepted_followup", "accepted"}:
            weight += 0.07 * multiplier
        elif event == "accepted_started":
            weight += 0.03 * multiplier
        elif event in {"dismissed", "rejected"}:
            weight -= 0.08 * multiplier
        elif event in {"accepted_aborted", "accepted_corrected"}:
            weight -= 0.1 * multiplier
        if fingerprint and row.get("fingerprint") == fingerprint and event == "dismissed":
            weight -= 0.12
    return max(-0.25, min(0.25, weight))


def build_feedback_features(
    *,
    suggestion: ForecastSuggestion | None = None,
    context: Any | None = None,
    trigger: str = "",
) -> dict[str, Any]:
    """Build stable feature keys for feedback learning."""

    workspace = getattr(context, "workspace", {}) if context is not None else {}
    task_state = getattr(context, "task_state", {}) if context is not None else {}
    focuses = workspace.get("focuses") if isinstance(workspace, dict) else None
    changed_files = [
        str(path) for path in (workspace.get("changed_files") if isinstance(workspace, dict) else []) or []
    ]
    features: dict[str, Any] = {
        "stage": str(getattr(context, "intent_stage", "") or ""),
        "focus_ids": [str(item.get("id")) for item in focuses or [] if isinstance(item, dict) and item.get("id")],
        "changed_file_globs": changed_file_globs(changed_files),
        "suggestion_kind": classify_suggestion_kind(suggestion),
        "has_dirty_worktree": bool(
            (workspace.get("git_status") if isinstance(workspace, dict) else "") or changed_files
        ),
        "had_recent_failure": bool(task_state.get("blocked_reason") if isinstance(task_state, dict) else False),
        "language": str(getattr(context, "response_language", "") or ""),
        "trigger": trigger,
    }
    return {key: value for key, value in features.items() if value not in ("", [], None)}


def classify_suggestion_kind(suggestion: ForecastSuggestion | None) -> str:
    if suggestion is None:
        return ""
    text = " ".join([suggestion.title, suggestion.prompt, suggestion.reason]).lower()
    if _contains_any(text, ("failed", "failure", "fix failing", "traceback", "失败", "报错", "修复最近失败")):
        return "fix_failure"
    if _contains_any(text, ("pytest", "test", "测试", "验证")):
        return "run_tests"
    if _contains_any(text, ("review", "diff", "inspect", "检查", "审查")):
        return "review_changes"
    if _contains_any(text, ("doc", "readme", "文档", "说明")):
        return "write_docs"
    if _contains_any(text, ("commit", "pr", "提交")):
        return "prepare_commit"
    if _contains_any(text, ("question", "clarif", "确认", "问题")):
        return "answer_question"
    return "continue_impl"


def changed_file_globs(paths: list[str]) -> list[str]:
    globs: list[str] = []
    for path in paths[:20]:
        normalized = path.replace("\\", "/")
        parts = normalized.split("/")
        if len(parts) >= 3:
            glob = "/".join(parts[:2]) + "/*"
        elif len(parts) == 2:
            glob = parts[0] + "/*"
        else:
            glob = normalized
        if glob and glob not in globs:
            globs.append(glob)
    return globs[:10]


def feedback_feature_similarity(left: dict[str, Any], right: Any) -> float:
    if not isinstance(right, dict) or not left:
        return 0.0
    score = 0.0
    total = 0.0
    for key, weight in (
        ("stage", 1.0),
        ("suggestion_kind", 1.4),
        ("language", 0.5),
        ("trigger", 0.4),
    ):
        total += weight
        if left.get(key) and left.get(key) == right.get(key):
            score += weight
    for key, weight in (("focus_ids", 1.2), ("changed_file_globs", 1.0)):
        left_set = {str(item) for item in left.get(key) or []}
        right_set = {str(item) for item in right.get(key) or []}
        if left_set and right_set:
            total += weight
            score += weight * (len(left_set & right_set) / len(left_set | right_set))
    for key, weight in (("has_dirty_worktree", 0.5), ("had_recent_failure", 0.8)):
        total += weight
        if bool(left.get(key)) == bool(right.get(key)):
            score += weight
    return score / total if total else 0.0


def looks_like_correction(text: str) -> bool:
    lowered = text.lower()
    return _contains_any(
        lowered,
        (
            "not that",
            "wrong direction",
            "instead",
            "不是",
            "不对",
            "方向错",
            "改成",
            "应该是",
        ),
    )


def looks_like_followup(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(continue|also|next|then)\b", lowered)) or _contains_any(
        lowered,
        ("继续", "接着", "下一步", "顺便", "再补"),
    )


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)
