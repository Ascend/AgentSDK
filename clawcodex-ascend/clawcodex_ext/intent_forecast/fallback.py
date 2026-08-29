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

"""Rule-based fallback strategy library for Intent Forecast."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

from clawcodex_ext.intent_forecast.context import ForecastContext
from clawcodex_ext.intent_forecast.focus import compute_workspace_focuses
from clawcodex_ext.intent_forecast.messages import ForecastSuggestion


@dataclass(frozen=True)
class FallbackRule:
    name: str
    apply: Callable[[ForecastContext, float], ForecastSuggestion | None]


def fallback_suggestions(context: ForecastContext, *, min_confidence: float) -> list[ForecastSuggestion]:
    if context.intent_stage == "pause":
        return []
    suggestions: list[ForecastSuggestion] = []
    for rule in fallback_rules():
        suggestion = rule.apply(context, min_confidence)
        if suggestion is not None:
            suggestions.append(suggestion)
    suggestions.sort(key=lambda item: _strategy_score(item, context), reverse=True)
    return suggestions[:3]


def fallback_rules() -> list[FallbackRule]:
    return [
        FallbackRule("open_question", _open_question),
        FallbackRule("recent_failure", _recent_failure),
        FallbackRule("focused_tests", _focused_tests),
        FallbackRule("document", _document),
        FallbackRule("commit", _commit),
        FallbackRule("recent_commits", _recent_commits),
        FallbackRule("intent_forecast_focus", _intent_forecast_focus),
        FallbackRule("intent_forecast_tests", _intent_forecast_tests),
        FallbackRule("intent_forecast_history", _intent_forecast_history),
        FallbackRule("workspace_review", _workspace_review),
        FallbackRule("recent_user", _recent_user),
        FallbackRule("session_next_action", _session_next_action),
    ]


def _open_question(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    questions = [str(item) for item in (context.task_state or {}).get("open_questions") or [] if str(item)]
    if not questions:
        return None
    zh = _zh(context)
    question = questions[0]
    return _suggestion(
        title="先回答待确认问题" if zh else "Resolve the open question",
        prompt=f"请先围绕这个待确认问题继续：{question}"
        if zh
        else f"Continue by resolving this open question first: {question}",
        reason="最近 assistant 刚提出澄清问题，继续实现前应先补齐决策信息。"
        if zh
        else "The latest assistant turn asked for clarification, so resolve that decision before implementation.",
        confidence=max(min_confidence, 0.72),
        refs=["task_state:open_questions"],
    )


def _recent_failure(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    blocked = str((context.task_state or {}).get("blocked_reason") or "")
    if not blocked:
        return None
    zh = _zh(context)
    return _suggestion(
        title="修复最近失败" if zh else "Fix the recent failure",
        prompt=f"请优先定位并修复最近的失败：{blocked}"
        if zh
        else f"Prioritize diagnosing and fixing the recent failure: {blocked}",
        reason="任务状态显示最近存在阻塞或测试失败。" if zh else "Task state shows a recent blocker or test failure.",
        confidence=max(min_confidence, 0.76),
        refs=["task_state:blocked_reason"],
    )


def _focused_tests(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    stage = str(context.intent_stage or "explore")
    tests = [str(item) for item in (context.task_state or {}).get("pending_tests") or [] if str(item)]
    if not tests or stage not in {"test", "debug", "implement"}:
        return None
    zh = _zh(context)
    tests_text = ", ".join(tests[:3])
    return _suggestion(
        title="运行相关测试" if zh else "Run focused tests",
        prompt=f"请运行相关测试并根据结果继续处理：{tests_text}"
        if zh
        else f"Run the focused tests and continue from the results: {tests_text}",
        reason="当前变更已经映射到可验证的测试范围。"
        if zh
        else "The current changes map to a focused verification set.",
        confidence=max(min_confidence, 0.7),
        refs=["task_state:pending_tests"],
    )


def _document(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    if context.intent_stage != "document":
        return None
    zh = _zh(context)
    return _suggestion(
        title="继续完善文档" if zh else "Continue documentation work",
        prompt="请基于当前变更继续完善相关文档，并检查文档内容是否与已实现行为一致。"
        if zh
        else "Continue updating the relevant documentation from the current changes and check that it matches the implemented behavior.",
        reason="最近用户请求或变更指向文档阶段。"
        if zh
        else "The recent user request or changed files point to documentation work.",
        confidence=max(min_confidence, 0.66),
        refs=["intent_stage:document"],
    )


def _commit(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    if context.intent_stage != "commit":
        return None
    zh = _zh(context)
    return _suggestion(
        title="整理当前变更" if zh else "Prepare the current changes",
        prompt="请查看当前 diff，整理变更摘要和剩余风险，并确认是否适合提交。"
        if zh
        else "Review the current diff, summarize the changes and remaining risks, and confirm whether it is ready to commit.",
        reason="当前意图阶段更接近提交前整理。"
        if zh
        else "The current intent stage is closest to pre-commit preparation.",
        confidence=max(min_confidence, 0.65),
        refs=["intent_stage:commit"],
    )


def _recent_commits(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    """Continue from the most recent commit when no other strong signal exists."""

    commits = context.workspace.get("recent_commits") or []
    if not commits:
        return None
    last = commits[0] if isinstance(commits[0], dict) else {}
    subject = str(last.get("subject") or "").strip()
    if not subject:
        return None
    short_hash = str(last.get("short_hash") or last.get("hash") or "")[:10]
    zh = _zh(context)
    anchor = f"`{short_hash} {subject}`" if short_hash else f"`{subject}`"
    return _suggestion(
        title="基于最近提交继续" if zh else "Continue from recent commits",
        prompt=(
            f"请基于最近提交 {anchor} 继续推进，并预测合理的下一步计划。\n\n最近提交主题：{subject}"
            if zh
            else f"Continue from the most recent commit {anchor} and predict the next step.\n\nMost recent commit subject: {subject}"
        ),
        reason=(
            "工作区最近的提交是最强的本地信号；没有更新的对话或未提交改动时，应基于 commit 主题预测下一步。"
            if zh
            else "Recent commits are the strongest local signal — when no newer dialog or uncommitted changes exist, predict the next step from the most recent commit subject."
        ),
        confidence=max(min_confidence, 0.58),
        refs=["git:recent-commits"],
    )


def _intent_forecast_focus(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    if not _has_focus(context, "intent_forecast"):
        return None
    zh = _zh(context)
    return _suggestion(
        title="验证 Intent Forecast 修复" if zh else "Verify Intent Forecast fixes",
        prompt="请检查当前 Intent Forecast 相关变更，运行 tests/intent_forecast 回归测试，并确认自动触发、/forecast 和 CLI 的推荐都与当前功能相关。"
        if zh
        else "Review the current Intent Forecast changes, run the tests/intent_forecast regression suite, and confirm auto, /forecast, and CLI suggestions stay relevant to the active feature.",
        reason="当前工作区变更集中在 intent_forecast 功能上。"
        if zh
        else "Current workspace changes are concentrated in the intent_forecast feature.",
        confidence=max(min_confidence, 0.68),
        refs=["git:changed-files", "focus:intent_forecast"],
    )


def _intent_forecast_tests(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    if not _has_focus(context, "intent_forecast"):
        return None
    zh = _zh(context)
    return _suggestion(
        title="运行 Intent Forecast 回归测试" if zh else "Run Intent Forecast regression tests",
        prompt="请运行 tests/intent_forecast，确认 CLI、/forecast 和自动触发的过滤与保存逻辑正常。"
        if zh
        else "Run tests/intent_forecast to verify CLI, /forecast, and automatic forecast filtering and persistence.",
        reason="当前修改涉及 forecast 生成、过滤和持久化，适合先做针对性回归。"
        if zh
        else "Current changes touch forecast generation, filtering, and persistence, so focused regression tests are the safest next step.",
        confidence=max(min_confidence, 0.64),
        refs=["tests:intent_forecast"],
    )


def _intent_forecast_history(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    if not _has_focus(context, "intent_forecast"):
        return None
    zh = _zh(context)
    return _suggestion(
        title="检查最新 Forecast 历史记录" if zh else "Inspect latest Forecast history",
        prompt="请查看 ~/.clawcodex/intent_forecast/history.jsonl 的最新记录，确认建议不再包含无关模块或权限模式误判。"
        if zh
        else "Inspect the latest ~/.clawcodex/intent_forecast/history.jsonl records and confirm suggestions no longer include unrelated module or permission-mode claims.",
        reason="统一 history sidecar 会保留每次 forecast 结果，可直接验证过滤效果。"
        if zh
        else "The unified history sidecar keeps every forecast result, making it direct evidence for filtering quality.",
        confidence=max(min_confidence, 0.6),
        refs=["history:intent_forecast"],
    )


def _has_focus(context: ForecastContext, focus_id: str) -> bool:
    focuses = context.workspace.get("focuses") or compute_workspace_focuses(
        changed_files=[str(path) for path in context.workspace.get("changed_files") or []],
        recent_messages=context.current_messages,
    )
    return any(isinstance(item, dict) and item.get("id") == focus_id for item in focuses)


def _workspace_review(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    if not str(context.workspace.get("git_status") or "").strip():
        return None
    zh = _zh(context)
    return _suggestion(
        title="检查当前工作区变更" if zh else "Review current workspace changes",
        prompt="请检查当前 git 变更，识别未完成的工作，并提出下一步实现计划。"
        if zh
        else "Review the current git changes, identify unfinished work, and propose the next implementation step.",
        reason="工作区存在未提交变更。" if zh else "The workspace has uncommitted changes.",
        confidence=max(min_confidence, 0.5),
        refs=["git:status"],
    )


def _recent_user(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    recent_user = ""
    for msg in reversed(context.current_messages):
        if msg.get("role") == "user":
            recent_user = str(msg.get("content") or "").strip()
            break
    if not recent_user:
        return None
    zh = _zh(context)
    return _suggestion(
        title="继续最近的任务" if zh else "Continue the recent task",
        prompt=f"请基于最新用户请求和当前工作区状态继续推进：\n\n{recent_user}"
        if zh
        else f"Continue from the latest user request and current workspace state:\n\n{recent_user}",
        reason="最近的用户消息是当前最强的本地信号。"
        if zh
        else "The most recent user message is the strongest local signal.",
        confidence=max(min_confidence, 0.55),
        refs=["conversation:recent-user"],
    )


def _session_next_action(context: ForecastContext, min_confidence: float) -> ForecastSuggestion | None:
    for session in context.sessions:
        summary = session.get("summary") if isinstance(session, dict) else None
        candidates = summary.get("next_action_candidates") if isinstance(summary, dict) else None
        if isinstance(candidates, list) and candidates:
            zh = _zh(context)
            title = str(candidates[0])[:160]
            return _suggestion(
                title=title,
                prompt=f"请继续推进最近会话中的下一步行动：{title}"
                if zh
                else f"Continue this next action from the recent session: {title}",
                reason="最近会话摘要将它列为下一步行动。"
                if zh
                else "A recent session summary listed this as a next action.",
                confidence=max(min_confidence, 0.48),
                refs=[f"session:{session.get('session_id', '')}"],
            )
    return None


def _suggestion(
    *,
    title: str,
    prompt: str,
    reason: str,
    confidence: float,
    refs: list[str],
) -> ForecastSuggestion:
    return ForecastSuggestion(
        id=f"forecast-{uuid.uuid4().hex[:10]}",
        title=title,
        prompt=prompt,
        reason=reason,
        confidence=confidence,
        source_refs=refs,
    )


def _zh(context: ForecastContext) -> bool:
    return context.response_language.lower().startswith("chinese")


def _strategy_score(suggestion: ForecastSuggestion, context: ForecastContext) -> float:
    strategy = getattr(context, "intent_strategy", "user") or "user"
    refs = " ".join(suggestion.refs()).lower()
    score = suggestion.confidence
    if strategy == "workspace":
        if any(marker in refs for marker in ("git:", "focus:", "tests:", "intent_stage:")):
            score += 0.18
        if "conversation:" in refs:
            score -= 0.08
        if "session:" in refs:
            score -= 0.04
    elif strategy == "history":
        if any(marker in refs for marker in ("session:", "history:")):
            score += 0.2
        if "conversation:" in refs:
            score -= 0.06
        if "git:" in refs:
            score -= 0.04
    else:
        if "conversation:" in refs or "task_state:" in refs:
            score += 0.18
        if "session:" in refs:
            score -= 0.08
    return score
