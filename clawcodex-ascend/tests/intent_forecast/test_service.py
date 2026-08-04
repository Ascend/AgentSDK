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

from __future__ import annotations

from clawcodex_ext.intent_forecast.context import ForecastContext
from clawcodex_ext.intent_forecast.messages import ForecastSuggestion
from clawcodex_ext.intent_forecast.prompt import build_forecast_messages
from clawcodex_ext.intent_forecast.service import (
    IntentForecastService,
    filter_suggestions_for_context,
    no_suggestion_gate,
    parse_forecast_response,
)


def test_parse_forecast_response_filters_low_confidence() -> None:
    raw = '{"suggestions":[{"title":"A","prompt":"do a","confidence":0.8},{"title":"B","prompt":"do b","confidence":0.1}]}'
    suggestions = parse_forecast_response(raw, min_confidence=0.45)
    assert [s.title for s in suggestions] == ["A"]


def test_service_uses_fallback_without_provider(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        current_messages=[{"role": "user", "content": "finish tests"}],
        workspace={"git_status": " M file.py"},
        fingerprint="abc",
    )
    result = IntentForecastService(
        conversation=None,
        provider=None,
        model=None,
        workspace_root=tmp_path,
        context=context,
    ).generate(trigger="test", force=True)
    assert result.generated is True
    assert result.suggestions[0].prompt
    assert result.fingerprint == "abc"


def test_prompt_carries_response_language(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        current_messages=[{"role": "user", "content": "\u7ee7\u7eed\u5b9e\u73b0\u529f\u80fd"}],
        response_language="Chinese",
    )

    messages = build_forecast_messages(context, max_input_tokens=4000)

    assert '"response_language": "Chinese"' in messages[0]["content"]
    assert '"user_intent"' in messages[0]["content"]
    assert "MUST use the context field `response_language`" in messages[0]["content"]
    assert "Treat `user_intent.initial_user_input`" in messages[0]["content"]
    assert "Obey exactly one `intent_strategy`" in messages[0]["content"]
    assert "Do not suggest changing permission mode" in messages[0]["content"]
    assert "Treat `dontAsk` as permissive" in messages[0]["content"]


def test_fallback_uses_chinese_when_context_language_is_chinese(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        current_messages=[{"role": "user", "content": "\u7ee7\u7eed\u8865\u9f50\u6d4b\u8bd5"}],
        response_language="Chinese",
        fingerprint="abc",
    )

    result = IntentForecastService(
        conversation=None,
        provider=None,
        model=None,
        workspace_root=tmp_path,
        context=context,
    ).generate(trigger="test", force=True)

    assert result.suggestions[0].title == "\u7ee7\u7eed\u6700\u8fd1\u7684\u4efb\u52a1"
    assert result.suggestions[0].prompt.startswith("\u8bf7\u57fa\u4e8e\u6700\u65b0\u7528\u6237\u8bf7\u6c42")


def test_filter_drops_english_suggestions_when_chinese_required(tmp_path) -> None:
    context = ForecastContext(cwd=str(tmp_path), response_language="Chinese")
    suggestions = [
        ForecastSuggestion(
            id="s1",
            title="Run tests",
            prompt="Run the stability tests.",
            reason="Likely next step.",
            confidence=0.8,
        ),
        ForecastSuggestion(
            id="s2",
            title="\u8fd0\u884c\u6d4b\u8bd5",
            prompt="\u8bf7\u8fd0\u884c\u7a33\u5b9a\u6027\u6d4b\u8bd5\u3002",
            reason="\u8fd9\u662f\u5408\u7406\u7684\u4e0b\u4e00\u6b65\u3002",
            confidence=0.7,
        ),
    ]

    filtered = filter_suggestions_for_context(suggestions, context)

    assert [s.id for s in filtered] == ["s2"]


def test_filter_drops_permission_mode_suggestion_without_current_block(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        current_messages=[{"role": "user", "content": "continue forecast work"}],
        response_language="English",
    )
    suggestions = [
        ForecastSuggestion(
            id="s1",
            title="Verify Permission Mode Configuration",
            prompt="Switch permission mode to ask.",
            reason="Previous sessions mentioned dontAsk.",
            confidence=0.8,
        ),
        ForecastSuggestion(
            id="s2",
            title="Run focused tests",
            prompt="Run the intent forecast tests.",
            confidence=0.7,
        ),
    ]

    filtered = filter_suggestions_for_context(suggestions, context)

    assert [s.id for s in filtered] == ["s2"]


def test_filter_drops_unrelated_history_when_workspace_focus_is_intent_forecast(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        workspace={"changed_files": ["clawcodex_ext/intent_forecast/service.py"]},
        response_language="English",
    )
    suggestions = [
        ForecastSuggestion(
            id="s1",
            title="Run Orchestrator Unit Tests",
            prompt="Run orchestrator unit tests.",
            reason="Previous sessions mentioned orchestrator stability.",
            confidence=0.8,
        ),
        ForecastSuggestion(
            id="s2",
            title="Run Intent Forecast tests",
            prompt="Run tests/intent_forecast.",
            reason="The active changes are in intent_forecast.",
            confidence=0.7,
        ),
    ]

    filtered = filter_suggestions_for_context(suggestions, context)

    assert [s.id for s in filtered] == ["s2"]


def test_filter_allows_generic_current_changes_for_focused_workspace(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        workspace={"changed_files": ["clawcodex_ext/intent_forecast/service.py"]},
        response_language="English",
    )
    suggestions = [
        ForecastSuggestion(
            id="s1",
            title="Review current workspace changes",
            prompt="Review current changes and identify unfinished work.",
            confidence=0.7,
        )
    ]

    filtered = filter_suggestions_for_context(suggestions, context)

    assert [s.id for s in filtered] == ["s1"]


def test_filter_keeps_multiple_focuses_for_cross_module_changes(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        workspace={
            "changed_files": [
                "clawcodex_ext/intent_forecast/service.py",
                "clawcodex_ext/tui/app.py",
            ]
        },
        response_language="English",
    )
    suggestions = [
        ForecastSuggestion(
            id="s1",
            title="Verify Intent Forecast filtering",
            prompt="Run tests/intent_forecast.",
            confidence=0.8,
        ),
        ForecastSuggestion(
            id="s2",
            title="Check TUI forecast picker",
            prompt="Inspect TUI app wiring.",
            confidence=0.7,
        ),
        ForecastSuggestion(
            id="s3",
            title="Run Orchestrator Unit Tests",
            prompt="Run orchestrator unit tests.",
            confidence=0.6,
        ),
    ]

    filtered = filter_suggestions_for_context(suggestions, context)

    assert [s.id for s in filtered] == ["s1", "s2"]


def test_fallback_prioritizes_intent_forecast_workspace_focus(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        workspace={
            "git_status": " M clawcodex_ext/intent_forecast/service.py",
            "changed_files": ["clawcodex_ext/intent_forecast/service.py"],
        },
        response_language="English",
    )

    result = IntentForecastService(
        conversation=None,
        provider=None,
        model=None,
        workspace_root=tmp_path,
        context=context,
    ).generate(trigger="test", force=True)

    assert result.suggestions[0].title == "Verify Intent Forecast fixes"
    assert [s.title for s in result.suggestions] == [
        "Verify Intent Forecast fixes",
        "Run Intent Forecast regression tests",
        "Inspect latest Forecast history",
    ]


def test_fallback_prioritizes_recent_failure(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        task_state={"blocked_reason": "tests/intent_forecast/test_service.py failed"},
        intent_stage="debug",
        response_language="English",
    )

    result = IntentForecastService(
        conversation=None,
        provider=None,
        model=None,
        workspace_root=tmp_path,
        context=context,
    ).generate(trigger="test", force=True)

    assert result.suggestions[0].title == "Fix the recent failure"
    assert "tests/intent_forecast/test_service.py failed" in result.suggestions[0].prompt


def test_fallback_uses_document_stage_before_dirty_worktree(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        workspace={"git_status": " M clawcodex_ext/intent_forecast/service.py"},
        task_state={},
        intent_stage="document",
        response_language="English",
    )

    result = IntentForecastService(
        conversation=None,
        provider=None,
        model=None,
        workspace_root=tmp_path,
        context=context,
    ).generate(trigger="test", force=True)

    assert result.suggestions[0].title == "Continue documentation work"


def test_auto_forecast_suppresses_weak_empty_context(tmp_path) -> None:
    context = ForecastContext(cwd=str(tmp_path), response_language="English", fingerprint="empty")

    result = IntentForecastService(
        conversation=None,
        provider=None,
        model=None,
        workspace_root=tmp_path,
        context=context,
    ).generate(trigger="auto", force=True)

    assert result.generated is False
    assert result.reason == "No confident next-step suggestions are available."


def test_no_suggestion_gate_keeps_suggestions_when_recent_commits_exist() -> None:
    """recent_commits counts as a strong signal — auto forecast should not suppress."""

    context = ForecastContext(
        cwd="/tmp/repo",
        response_language="English",
        workspace={
            "recent_commits": [
                {
                    "hash": "abc1234",
                    "short_hash": "abc1234",
                    "subject": "fix: tweak recent_commits signal",
                    "author": "alice",
                    "timestamp": "1718000000",
                }
            ]
        },
    )
    suggestions = [ForecastSuggestion(id="s1", title="Continue forecast work", prompt="continue", confidence=0.55)]

    kept = no_suggestion_gate(suggestions, context=context, trigger="auto", min_confidence=0.45)

    assert [s.id for s in kept] == ["s1"]


def test_no_suggestion_gate_still_suppresses_weak_auto_without_commits() -> None:
    """Without any strong signal, auto trigger still suppresses — back-compat guard."""

    context = ForecastContext(cwd="/tmp/repo", response_language="English", workspace={})
    suggestions = [ForecastSuggestion(id="s1", title="Continue forecast work", prompt="continue", confidence=0.55)]

    kept = no_suggestion_gate(suggestions, context=context, trigger="auto", min_confidence=0.45)

    assert kept == []


def test_auto_forecast_generates_with_only_recent_commits_signal(tmp_path) -> None:
    """End-to-end: with only recent_commits and no provider, the fallback rule fires."""

    context = ForecastContext(
        cwd=str(tmp_path),
        response_language="English",
        workspace={
            "recent_commits": [
                {
                    "hash": "abc1234",
                    "short_hash": "abc1234",
                    "subject": "feat: add recent_commits signal",
                    "author": "alice",
                    "timestamp": "1718000000",
                }
            ]
        },
        fingerprint="fp",
    )

    result = IntentForecastService(
        conversation=None,
        provider=None,
        model=None,
        workspace_root=tmp_path,
        context=context,
    ).generate(trigger="auto", force=True)

    assert result.generated is True
    assert result.suggestions
    assert result.suggestions[0].title == "Continue from recent commits"
    assert "feat: add recent_commits signal" in result.suggestions[0].prompt
