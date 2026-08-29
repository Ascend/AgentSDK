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

"""Workspace focus detection for Intent Forecast."""

from __future__ import annotations

from typing import Any


def compute_workspace_focuses(
    *,
    changed_files: list[str],
    recent_messages: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    changed = [str(path).lower().replace("\\", "/") for path in changed_files]
    recent = " ".join(str(msg.get("content") or "").lower() for msg in (recent_messages or [])[-6:])
    if not changed and not recent:
        return []

    focus_defs = focus_definitions()
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for path in changed:
        for focus_id, meta in focus_defs.items():
            score = _path_alias_score(path, meta)
            if score:
                scores[focus_id] = scores.get(focus_id, 0.0) + score
                evidence.setdefault(focus_id, []).append(path)
    for focus_id, meta in focus_defs.items():
        score = _text_alias_score(recent, meta)
        if score:
            scores[focus_id] = scores.get(focus_id, 0.0) + score
            evidence.setdefault(focus_id, []).append("conversation:recent-user")

    if not scores:
        return []
    total_changed = max(1, len(changed))
    focuses: list[dict[str, Any]] = []
    for focus_id, score in scores.items():
        confidence = min(1.0, score / total_changed)
        meta = focus_defs[focus_id]
        focuses.append(
            {
                "id": focus_id,
                "label": meta["label"],
                "confidence": round(confidence, 4),
                "evidence": evidence.get(focus_id, [])[:8],
            }
        )
    focuses.sort(key=lambda item: item["confidence"], reverse=True)
    top = focuses[0]["confidence"]
    return [item for item in focuses if item["confidence"] >= max(0.34, top - 0.2)]


def suggestion_matches_focus(text: str, refs: list[str], focus: dict[str, Any]) -> bool:
    focus_id = str(focus.get("id") or "")
    meta = focus_definitions().get(focus_id)
    if not meta:
        return False
    combined = f"{text} {' '.join(refs)}".lower().replace("\\", "/")
    aliases = tuple(meta["strong_path_aliases"]) + tuple(meta["module_aliases"]) + tuple(meta["weak_text_aliases"])
    return any(alias in combined for alias in aliases)


def focus_definitions() -> dict[str, dict[str, Any]]:
    return {
        "intent_forecast": {
            "label": "Intent Forecast",
            "strong_path_aliases": ("clawcodex_ext/intent_forecast/", "tests/intent_forecast/"),
            "module_aliases": ("intent_forecast", "intent-forecast", "intent forecast"),
            "weak_text_aliases": ("意图预测", "forecast"),
        },
        "session_intelligence": {
            "label": "Session Intelligence",
            "strong_path_aliases": (
                "clawcodex_ext/session_intelligence/",
                "tests/session_intelligence/",
            ),
            "module_aliases": ("session_intelligence", "session summary", "summary.json"),
            "weak_text_aliases": ("summary",),
        },
        "away_summary": {
            "label": "Away Summary",
            "strong_path_aliases": ("away_summary/", "away_summary\\"),
            "module_aliases": ("away_summary", "away summary"),
            "weak_text_aliases": ("recap",),
        },
        "tui": {
            "label": "TUI",
            "strong_path_aliases": ("clawcodex_ext/tui/", "src/tui/", "tests/tui/"),
            "module_aliases": ("tui", "textual", "prompt_input"),
            "weak_text_aliases": ("repl screen",),
        },
        "repl": {
            "label": "REPL",
            "strong_path_aliases": ("clawcodex_ext/repl/", "src/repl/"),
            "module_aliases": ("repl", "slash command"),
            "weak_text_aliases": ("interactive shell",),
        },
        "command_system": {
            "label": "Command System",
            "strong_path_aliases": ("clawcodex_ext/command_system/", "src/command_system/"),
            "module_aliases": ("command_system", "localcommand"),
            "weak_text_aliases": ("commands", "slash command"),
        },
        "orchestrator": {
            "label": "Orchestrator",
            "strong_path_aliases": ("extensions/orchestrator/", "tests/orchestrator/"),
            "module_aliases": ("orchestrator",),
            "weak_text_aliases": (),
        },
        "permissions": {
            "label": "Permissions",
            "strong_path_aliases": ("permissions/", "permission_"),
            "module_aliases": ("permissions", "permission mode", "bypasspermissions", "dontask"),
            "weak_text_aliases": ("permission",),
        },
        "context": {
            "label": "Context",
            "strong_path_aliases": ("context/", "prompt_assembly"),
            "module_aliases": ("context", "system_prompt"),
            "weak_text_aliases": (),
        },
        "tests": {
            "label": "Tests",
            "strong_path_aliases": ("tests/",),
            "module_aliases": ("pytest", "test_"),
            "weak_text_aliases": ("tests",),
        },
    }


def _path_alias_score(path: str, meta: dict[str, Any]) -> float:
    if any(alias in path for alias in meta["strong_path_aliases"]):
        return 1.0
    if any(alias in path for alias in meta["module_aliases"]):
        return 0.75
    return 0.0


def _text_alias_score(text: str, meta: dict[str, Any]) -> float:
    if not text:
        return 0.0
    if any(alias in text for alias in meta["module_aliases"]):
        return 0.65
    # Weak aliases need another module/path signal; text alone is too broad.
    return 0.0
