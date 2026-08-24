# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Focused tests for F-124 models, parsing, and prompt construction."""

import json

from types import SimpleNamespace

from extensions.orchestrator.issue_clarifier.models import ClarifyResult
from extensions.orchestrator.issue_clarifier.parser import parse_clarify_response
from extensions.orchestrator.issue_clarifier.prompt import build_clarify_messages


def test_parse_unclear_response_and_round_trip() -> None:
    raw = """```json
    {"is_clear": false, "confidence": 0.95, "reason": "missing target",
     "ambiguities": [{"question": "Which package?", "ambiguity_type": "missing",
                       "evidence": "target is unspecified"}]}
    ```"""

    result = parse_clarify_response(raw, min_confidence=0.7, max_questions=3)

    assert not result.is_clear
    assert result.questions == ["Which package?"]
    assert ClarifyResult.from_dict(result.to_dict()) == result


def test_low_confidence_response_fails_open() -> None:
    result = parse_clarify_response(
        '{"is_clear": false, "confidence": 0.2, "ambiguities": []}',
        min_confidence=0.7,
        max_questions=3,
    )

    assert result.is_clear
    assert result.degraded


def test_prompt_contains_issue_and_prior_reply() -> None:
    issue = SimpleNamespace(
        id="124",
        identifier="F-124",
        title="Clarify migration",
        description="The destination is unspecified.",
        labels=["feature"],
    )

    messages = build_clarify_messages(
        issue,
        prior_replies=("Use AgentSDK",),
        max_questions=2,
        max_input_tokens=2000,
    )

    rendered = "\n".join(str(message) for message in messages)
    assert "Clarify migration" in rendered
    assert "Use AgentSDK" in rendered


def test_null_suggested_option_is_discarded() -> None:
    result = parse_clarify_response(
        '{"is_clear": false, "confidence": 0.95, "ambiguities": '
        '[{"question": "Which package?", "ambiguity_type": "missing", '
        '"suggested_options": [null, "package-a"]}]}',
        min_confidence=0.7,
        max_questions=3,
    )

    assert result.ambiguities[0].suggested_options == ("package-a",)


def test_large_workspace_focus_is_truncated_without_losing_field() -> None:
    issue = SimpleNamespace(title="Clarify migration", description="", labels=[])

    messages = build_clarify_messages(
        issue,
        max_input_tokens=1,
        workspace_focuses=[{"path": "src/app.py", "content": "x" * 5000}],
    )

    payload = json.loads(messages[1]["content"])
    assert len(messages[1]["content"]) <= 1000
    assert payload["_truncated"] is True
    assert "workspace_focuses" in payload
