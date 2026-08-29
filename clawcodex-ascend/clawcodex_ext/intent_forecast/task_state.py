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

"""Task-state extraction for Intent Forecast."""

from __future__ import annotations

import re
from typing import Any


INTENT_STAGES = {
    "explore",
    "plan",
    "implement",
    "test",
    "debug",
    "review",
    "document",
    "commit",
    "pause",
}


def build_task_state(
    *,
    current_messages: list[dict[str, str]],
    sessions: list[dict[str, Any]],
    workspace: dict[str, Any],
    user_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a small structured view of the active task."""

    user_intent = user_intent or {}
    recent_user = str(user_intent.get("latest_user_input") or _latest_role_text(current_messages, "user"))
    initial_user = str(user_intent.get("initial_user_input") or "")
    recent_assistant = _latest_role_text(current_messages, "assistant")
    recent_text = "\n".join(msg.get("content", "") for msg in current_messages[-8:])
    last_failure = _last_failure(recent_text, workspace)
    pending_tests = _pending_tests(workspace)
    open_questions = _open_questions(current_messages)
    active_goal = _active_goal(recent_user or initial_user, sessions, workspace)
    last_completed_step = _last_completed_step(recent_assistant, recent_text)
    next_unfinished_step = _next_unfinished_step(
        active_goal=active_goal,
        pending_tests=pending_tests,
        blocked_reason=last_failure,
        sessions=sessions,
        workspace=workspace,
    )
    recent_decisions = _recent_decisions(current_messages)
    return {
        "active_goal": active_goal,
        "last_completed_step": last_completed_step,
        "next_unfinished_step": next_unfinished_step,
        "blocked_reason": last_failure,
        "pending_tests": pending_tests,
        "open_questions": open_questions,
        "recent_decisions": recent_decisions,
    }


def classify_intent_stage(
    *,
    current_messages: list[dict[str, str]],
    task_state: dict[str, Any],
    workspace: dict[str, Any],
    user_intent: dict[str, Any] | None = None,
) -> str:
    """Classify the user's current work stage with lightweight rules."""

    user_intent = user_intent or {}
    recent_user = str(user_intent.get("latest_user_input") or _latest_role_text(current_messages, "user")).lower()
    initial_user = str(user_intent.get("initial_user_input") or "").lower()
    user_text = "\n".join(item for item in (initial_user, recent_user) if item)
    recent_all = "\n".join(msg.get("content", "") for msg in current_messages[-6:]).lower()
    if _contains_any(user_text, ("pause", "later", "hold", "stop", "先暂停", "等下", "稍后", "暂停")):
        return "pause"
    if _contains_any(user_text, ("commit", "pr", "diff", "提交", "拉取请求", "变更摘要")):
        return "commit"
    if _contains_any(user_text, ("doc", "readme", "文档", "说明", "计划", "feature_plan")):
        return "document"
    if _contains_any(user_text, ("review", "审查", "检查风险", "代码审查")):
        return "review"
    if task_state.get("blocked_reason") or _contains_any(recent_all, ("traceback", "failed", "error", "失败", "报错")):
        return "debug"
    if task_state.get("pending_tests") or _contains_any(user_text, ("test", "pytest", "测试", "验证")):
        return "test"
    if _contains_any(user_text, ("implement", "build", "fix", "add", "补全", "实现", "修复", "接入")):
        return "implement"
    if _contains_any(user_text, ("plan", "design", "方案", "规划", "拆分")):
        return "plan"
    if _contains_any(user_text, ("inspect", "analyze", "read", "查看", "分析", "梳理")):
        return "explore"
    if workspace.get("git_status"):
        return "implement"
    return "explore"


def _latest_role_text(messages: list[dict[str, str]], role: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == role:
            return str(msg.get("content") or "").strip()
    return ""


def _active_goal(
    recent_user: str,
    sessions: list[dict[str, Any]],
    workspace: dict[str, Any],
) -> str:
    if recent_user:
        return _first_line(recent_user, limit=240)
    for session in sessions[:3]:
        summary = session.get("summary") if isinstance(session, dict) else None
        if isinstance(summary, dict):
            for key in ("open_threads", "goals", "next_action_candidates"):
                values = summary.get(key)
                if isinstance(values, list) and values:
                    return str(values[0])[:240]
                if isinstance(values, str) and values:
                    return values[:240]
    changed = workspace.get("changed_files") or []
    if changed:
        return f"Continue work around {', '.join(str(item) for item in changed[:3])}"
    return ""


def _last_completed_step(recent_assistant: str, recent_text: str) -> str:
    haystack = recent_assistant or recent_text
    patterns = (
        r"(?:completed|implemented|fixed|verified|passed)[:\s]+(.{1,180})",
        r"(?:已完成|已实现|已修复|验证通过)[:：\s]*(.{1,180})",
    )
    for pattern in patterns:
        match = re.search(pattern, haystack, flags=re.IGNORECASE)
        if match:
            return _first_line(match.group(1), limit=180)
    return ""


def _next_unfinished_step(
    *,
    active_goal: str,
    pending_tests: list[str],
    blocked_reason: str,
    sessions: list[dict[str, Any]],
    workspace: dict[str, Any],
) -> str:
    if blocked_reason:
        return "Fix the recent failure before continuing broader implementation."
    if pending_tests:
        return f"Run focused tests: {', '.join(pending_tests[:3])}."
    for session in sessions[:3]:
        summary = session.get("summary") if isinstance(session, dict) else None
        candidates = summary.get("next_action_candidates") if isinstance(summary, dict) else None
        if isinstance(candidates, list) and candidates:
            return str(candidates[0])[:240]
    if workspace.get("git_status"):
        return "Review current changes and finish or verify the modified paths."
    return active_goal[:240]


def _last_failure(recent_text: str, workspace: dict[str, Any]) -> str:
    failures = workspace.get("last_test_failures") or []
    if failures:
        return str(failures[0])[:240]
    lower = recent_text.lower()
    markers = (
        "traceback",
        "assertionerror",
        "failed",
        "error:",
        "permission denied",
        "失败",
        "报错",
    )
    if not any(marker in lower for marker in markers):
        return ""
    for line in reversed(recent_text.splitlines()):
        if any(marker in line.lower() for marker in markers):
            return line.strip()[:240]
    return "Recent output indicates a failure."


def _pending_tests(workspace: dict[str, Any]) -> list[str]:
    if workspace.get("last_command_exit") == 0 and _looks_like_test_command(str(workspace.get("last_command") or "")):
        return []
    mapping = workspace.get("changed_test_mapping")
    if isinstance(mapping, list) and mapping:
        return [str(item) for item in mapping[:6]]
    changed = [str(path) for path in workspace.get("changed_files") or []]
    tests = [path for path in changed if _is_test_path(path)]
    if tests:
        return tests[:6]
    sources = [path for path in changed if path.endswith(".py") and not _is_test_path(path)]
    return [_source_to_test_hint(path) for path in sources[:4]]


def _open_questions(messages: list[dict[str, str]]) -> list[str]:
    questions: list[str] = []
    for msg in reversed(messages[-6:]):
        if msg.get("role") != "assistant":
            continue
        text = str(msg.get("content") or "").strip()
        if text.endswith("?") or text.endswith("？"):
            questions.append(_first_line(text, limit=180))
            break
    return questions


def _recent_decisions(messages: list[dict[str, str]]) -> list[str]:
    decisions: list[str] = []
    markers = ("prefer", "use ", "don't", "do not", "keep", "不要", "使用", "保持", "改成", "优先")
    for msg in messages[-8:]:
        if msg.get("role") != "user":
            continue
        text = str(msg.get("content") or "").strip()
        if _contains_any(text.lower(), markers):
            decisions.append(_first_line(text, limit=180))
    return decisions[-4:]


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return "/tests/" in f"/{normalized}" or normalized.startswith("tests/") or "test_" in normalized


def _source_to_test_hint(path: str) -> str:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if normalized.startswith("clawcodex_ext/intent_forecast/"):
        return "tests/intent_forecast"
    if normalized.startswith("clawcodex_ext/tui/"):
        return "tests/tui"
    if normalized.startswith("extensions/orchestrator/"):
        return "tests/orchestrator"
    return f"tests matching {name}"


def _looks_like_test_command(command: str) -> bool:
    text = command.lower()
    return "pytest" in text or "test" in text


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _first_line(text: str, *, limit: int) -> str:
    return text.strip().splitlines()[0][:limit]
