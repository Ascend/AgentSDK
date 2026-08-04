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

"""Prompt construction for Intent Forecast."""

from __future__ import annotations

import json

from clawcodex_ext.intent_forecast.context import ForecastContext


FORECAST_INSTRUCTIONS = """You predict the user's likely next step in an interactive coding session.

Return strict JSON only, shaped as:
{"suggestions":[{"title":"...","prompt":"...","reason":"...","confidence":0.0,"source_refs":["..."]}]}

Rules:
- Suggest at most 3 concrete next actions.
- The prompt must be ready to submit to the coding agent.
- MUST use the context field `response_language` for every suggestion title, prompt, and reason.
- Do not start work yourself.
- If `response_language` is Chinese, write natural Simplified Chinese.
- If `response_language` is English, write natural English.
- Prefer current workspace signals and changed_files over older session summaries.
- If changed_files point at a specific feature/module, suggestions must stay on that feature/module unless the current user messages say otherwise.
- Do not suggest changing permission mode unless current context explicitly shows a tool/test was blocked by permissions.
- Treat `dontAsk` as permissive/logging mode, not as evidence that tools are blocked.
- Use confidence between 0 and 1.
- Avoid repeating suggestions that feedback says were dismissed.
- Treat `user_intent.initial_user_input`, `user_intent.latest_user_input`, and `user_intent.previous_user_inputs` as the primary evidence of intent.
- Assistant/system messages are secondary evidence only: use them to infer completed work, failures, blockers, or unanswered questions, not to invent a new user goal.
- If assistant output is verbose but user input is vague or only a greeting, avoid over-interpreting the assistant output.
- Obey exactly one `intent_strategy`: `user`, `workspace`, or `history`.
- If `intent_strategy` is `user`, first decide from `user_intent`, then `task_state` and `intent_stage`; use workspace/history only as supporting evidence.
- If `intent_strategy` is `workspace`, first decide from changed files, workspace focus, tests, and git state; use user input/history only as supporting evidence.
- If `intent_strategy` is `history`, first decide from relevant session summaries and feedback; use current user/workspace only to filter stale or off-topic history.
- If `task_state.blocked_reason` is present, prioritize fixing that failure.
- If `task_state.open_questions` is non-empty, do not suggest autonomous implementation; suggest answering or resolving the question.
- If `intent_stage` is `test` or `debug`, keep suggestions focused on verification or failure repair.
- Treat `workspace.recent_commits` as the strongest evidence of where the user has actually been working: predict the next step that continues, verifies, or extends the most recent commit subject — do not invent unrelated directions.
- If `workspace.recent_commits` is non-empty but other signals (user messages, git status, blocked_reason, pending_tests) are empty or stale, lean on `recent_commits` rather than on older session summaries."""


def build_forecast_messages(
    context: ForecastContext,
    *,
    max_input_tokens: int,
) -> list[dict[str, str]]:
    payload = json.dumps(context.to_prompt_dict(), ensure_ascii=False, default=str, indent=2)
    payload = _truncate(payload, max_input_tokens=max_input_tokens)
    return [
        {
            "role": "user",
            "content": f"{FORECAST_INSTRUCTIONS}\n\nContext:\n{payload}\n\nReturn JSON only.",
        }
    ]


def _truncate(text: str, *, max_input_tokens: int) -> str:
    budget = max_input_tokens * 4
    if len(text) <= budget:
        return text
    head = max(1000, budget // 4)
    tail = max(1000, budget - head - 120)
    return text[:head].rstrip() + "\n\n[... omitted for forecast budget ...]\n\n" + text[-tail:].lstrip()
