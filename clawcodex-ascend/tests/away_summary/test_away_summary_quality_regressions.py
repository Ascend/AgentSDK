#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

# pylint: disable=no-name-in-module

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from clawcodex_ext.away_summary.service import (
    _extract_summary,
    _fallback_summary,
    _find_file_path,
    _generate_via_chat,
    _generate_via_fork,
    _normalize_summary_output,
)
from clawcodex_ext.providers.base import ChatResponse


@pytest.mark.parametrize(
    "failure",
    [TypeError("provider internal type failure"), RuntimeError("permanent failure")],
)
def test_deterministic_errors_do_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    calls = 0

    class Provider:
        def chat(self, **_kwargs: Any) -> ChatResponse:
            nonlocal calls
            calls += 1
            raise failure

    monkeypatch.setattr("clawcodex_ext.away_summary.service.time.sleep", lambda _seconds: None)

    with pytest.raises(type(failure), match=str(failure)):
        _generate_via_chat(
            Provider(),
            [{"role": "user", "content": "recap"}],
            model="test",
            max_output_tokens=128,
        )

    assert calls == 1


def test_unsupported_max_tokens_falls_back_without_discarding_response() -> None:
    calls: list[dict[str, Any]] = []

    class Provider:
        def chat(self, **kwargs: Any) -> ChatResponse:
            calls.append(kwargs)
            if "max_tokens" in kwargs:
                raise TypeError("chat() got an unexpected keyword argument 'max_tokens'")
            return ChatResponse(
                content="recap",
                model="test",
                usage={},
                finish_reason="stop",
            )

    response = _generate_via_chat(
        Provider(),
        [{"role": "user", "content": "recap"}],
        model="test",
        max_output_tokens=128,
    )

    assert response.content == "recap"
    assert len(calls) == 2
    assert "max_tokens" in calls[0]
    assert "max_tokens" not in calls[1]


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_transient_http_status_retries_once(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    calls = 0

    class TransientHTTPError(Exception):
        def __init__(self) -> None:
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    class Provider:
        def chat(self, **_kwargs: Any) -> ChatResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TransientHTTPError()
            return ChatResponse(
                content="recap",
                model="test",
                usage={},
                finish_reason="stop",
            )

    monkeypatch.setattr("clawcodex_ext.away_summary.service.time.sleep", lambda _seconds: None)

    response = _generate_via_chat(
        Provider(),
        [{"role": "user", "content": "recap"}],
        model="test",
        max_output_tokens=128,
    )

    assert response.content == "recap"
    assert calls == 2


def test_single_hallmark_does_not_discard_valid_recap() -> None:
    recap = "We fixed the input transcript: parser and tests now pass."
    cot = "Draft Recap (Mental Refinement in Simplified Chinese): expose internal steps"

    assert _extract_summary(SimpleNamespace(content=recap)) == recap
    assert _extract_summary(SimpleNamespace(content=cot)) == ""


def test_fallback_excludes_reasoning_but_keeps_text_and_tools() -> None:
    conversation = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", content="Fix parser"),
            SimpleNamespace(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "SECRET internal reasoning"},
                    {"type": "reasoning", "reasoning": "SECRET model reasoning"},
                    SimpleNamespace(type="thinking", thinking="SECRET object reasoning"),
                    SimpleNamespace(type="text", text="Updated the parser."),
                    SimpleNamespace(
                        type="tool_use",
                        name="Read",
                        input={"file_path": "parser.py"},
                    ),
                ],
            ),
        ]
    )

    summary = _fallback_summary(conversation)

    assert "SECRET internal reasoning" not in summary
    assert "SECRET model reasoning" not in summary
    assert "SECRET object reasoning" not in summary
    assert "Updated the parser" in summary
    assert "Read(parser.py)" in summary


def test_markdown_italic_cleanup_preserves_underscored_filename() -> None:
    text = "Updated my_config_file.py and _documented_ the behavior."

    assert _normalize_summary_output(text) == ("Updated my_config_file.py and documented the behavior.")


def test_file_path_fallback_ignores_plain_sentence_with_period() -> None:
    assert _find_file_path({"text": "The implementation is complete."}) == ""
    assert _find_file_path({"text": "Inspect src/parser.py next"}) == "src/parser.py"


def test_fork_forwards_max_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Any] = []

    async def fake_run(params: Any) -> SimpleNamespace:
        from clawcodex_ext.types.messages import AssistantMessage

        captured.append(params)
        return SimpleNamespace(messages=[AssistantMessage(content="recap")])

    monkeypatch.setattr("clawcodex_ext.agent.forked_agent.run_forked_agent", fake_run)
    context = SimpleNamespace(_active_provider=object())
    _generate_via_fork(
        SimpleNamespace(tool_use_context=context),
        [{"role": "user", "content": "recap"}],
        model="test",
        max_output_tokens=321,
    )

    assert captured[0].max_output_tokens == 321


def test_two_chapter_recap_is_not_treated_as_cot() -> None:
    recap = "1. Completed changes:\nParser tests pass.\n2. Next steps:\nReview results."
    summary = _extract_summary(SimpleNamespace(content=recap))

    assert "Completed changes" in summary
    assert "Next steps" in summary
