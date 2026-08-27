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

"""Focused tests for the issue clarification dispatch gate boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from extensions.orchestrator.issue_clarifier.gate import IssueClarificationGate


def _gate(*, enabled: bool) -> IssueClarificationGate:
    return IssueClarificationGate(
        service=MagicMock(),
        resolver=MagicMock(),
        registry=MagicMock(),
        config=SimpleNamespace(enabled=enabled, max_analyses_per_poll=1),
    )


def _waiting_record(**overrides):
    values = {
        "clarification_replies": [],
        "clarification_status": "awaiting_author",
        "clarifier_fingerprint": "old-fingerprint",
        "clarification_round": 2,
        "open_questions": ["Old question"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_disabled_gate_allows_dispatch_without_dependencies() -> None:
    gate = _gate(enabled=False)
    issue = SimpleNamespace(id="124")

    assert await gate.should_dispatch(issue)
    gate.registry.get.assert_not_called()


@pytest.mark.asyncio
async def test_missing_registry_record_allows_dispatch() -> None:
    gate = _gate(enabled=True)
    gate.registry.get.return_value = None

    assert await gate.should_dispatch(SimpleNamespace(id="124"))


def test_begin_poll_resets_analysis_budget() -> None:
    gate = _gate(enabled=True)
    gate._analyses_this_poll = 1

    gate.begin_poll()

    assert gate._analyses_this_poll == 0


@pytest.mark.asyncio
async def test_issue_edit_retires_stale_question_without_resetting_round() -> None:
    tracker = MagicMock()
    gate = IssueClarificationGate(
        service=MagicMock(),
        resolver=MagicMock(),
        registry=MagicMock(),
        config=SimpleNamespace(enabled=True, max_analyses_per_poll=1, remote_label="needs-clarification"),
        tracker=tracker,
    )
    record = _waiting_record()
    gate.registry.get.return_value = record
    gate.service.fingerprint.return_value = "new-fingerprint"
    gate.resolver.get_answer.return_value = None
    gate._analyses_this_poll = 1

    assert not await gate.should_dispatch(SimpleNamespace(id="124"))

    assert record.clarification_round == 2
    gate.registry.update_clarification.assert_called_once_with("124", open_questions=[])
    tracker.remove_label.assert_called_once_with("124", "needs-clarification")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "should_dispatch"),
    [("manual_resolved", True), ("manual_required", False)],
)
async def test_terminal_clarification_status_removes_remote_label(status, should_dispatch) -> None:
    tracker = MagicMock()
    gate = IssueClarificationGate(
        service=MagicMock(),
        resolver=MagicMock(),
        registry=MagicMock(),
        config=SimpleNamespace(enabled=True, max_analyses_per_poll=1, remote_label="needs-clarification"),
        tracker=tracker,
    )
    gate.registry.get.return_value = _waiting_record(
        clarification_status=status,
        clarifier_fingerprint="current-fingerprint",
    )
    gate.service.fingerprint.return_value = "current-fingerprint"

    assert await gate.should_dispatch(SimpleNamespace(id="124")) is should_dispatch
    tracker.remove_label.assert_called_once_with("124", "needs-clarification")
