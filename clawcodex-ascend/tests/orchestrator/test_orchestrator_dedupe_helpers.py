#!/usr/bin/env python3

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

"""Focused tests for the dedupe helpers extracted during the #160 audit.

The audit pass found three repeated sub-blocks inside the orchestrator
mixins and converged them into private helpers:

- ``OrchestratorRebaseMixin._fetch_rebase_issue`` (3 identical
  fetch-or-synthesize sites in the rebase poll / PR-conflict-scan /
  control paths)
- ``OrchestratorSessionMixin._mark_issue_intent`` (4 identical
  ``registry.mark_intent`` scaffold blocks in ``_poll_and_dispatch``)
- ``OrchestratorOpsMixin._process_escalated_issues`` (3 duplicated policy
  branches driven by ``_ESCALATION_ACTIONS``)

These tests drive each helper with stub collaborators to pin the behaviour
that previously lived inline at every call site.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from extensions.orchestrator.events import EventLevel
from extensions.orchestrator.issue import Issue
from extensions.orchestrator.orchestrator_ops import (
    _ESCALATION_ACTIONS,
    OrchestratorOpsMixin,
)
from extensions.orchestrator.orchestrator_rebase import OrchestratorRebaseMixin
from extensions.orchestrator.orchestrator_session import OrchestratorSessionMixin

# ---------------------------------------------------------------------------
# _fetch_rebase_issue
# ---------------------------------------------------------------------------


class _FakeTracker:
    def __init__(self, fetched) -> None:
        self.fetched = fetched

    async def fetch_issue_states_by_ids(self, issue_ids):
        return self.fetched


def _record() -> SimpleNamespace:
    return SimpleNamespace(issue_identifier="rebase-issue-1", branch_name="main")


async def test_fetch_rebase_issue_returns_tracker_issue() -> None:
    runner = OrchestratorRebaseMixin()
    tracker_issue = Issue(id="issue-1", identifier="rebase-issue-1", title="tracker title")
    runner.tracker = _FakeTracker({"issue-1": tracker_issue})
    result = await runner._fetch_rebase_issue("issue-1", _record())
    assert result is tracker_issue


async def test_fetch_rebase_issue_synthesizes_when_tracker_unknown() -> None:
    runner = OrchestratorRebaseMixin()
    # Tracker returns an empty mapping (offline / not-yet-created issue).
    runner.tracker = _FakeTracker({})
    issue = await runner._fetch_rebase_issue("issue-1", _record())
    assert isinstance(issue, Issue)
    assert issue.id == "issue-1"
    assert issue.identifier == "rebase-issue-1"
    assert issue.title == "(unknown)"
    assert issue.branch_name == "main"


async def test_fetch_rebase_issue_synthesizes_when_tracker_none() -> None:
    runner = OrchestratorRebaseMixin()
    runner.tracker = _FakeTracker(None)
    issue = await runner._fetch_rebase_issue("issue-1", _record())
    assert isinstance(issue, Issue)
    assert issue.id == "issue-1"


# ---------------------------------------------------------------------------
# _mark_issue_intent
# ---------------------------------------------------------------------------


class _IntentRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def mark_intent(self, issue_id, intent, *, source=None, command=None) -> None:
        self.calls.append((issue_id, intent, source, command))


def _make_session_runner() -> tuple[OrchestratorSessionMixin, _IntentRegistry]:
    runner = OrchestratorSessionMixin()
    registry = _IntentRegistry()
    runner._registry = registry
    return runner, registry


def test_mark_issue_intent_passes_source_and_command() -> None:
    runner, registry = _make_session_runner()
    issue = SimpleNamespace(id="issue-1")
    command = SimpleNamespace(value="retry")
    runner._mark_issue_intent(issue, "RETRY", "comment", command)
    assert registry.calls == [("issue-1", "RETRY", "comment", "/agent retry")]


def test_mark_issue_intent_falls_back_to_command_source() -> None:
    runner, registry = _make_session_runner()
    issue = SimpleNamespace(id="issue-1")
    command = SimpleNamespace(value="review")
    runner._mark_issue_intent(issue, "REVIEW", None, command)
    assert registry.calls == [("issue-1", "REVIEW", "command", "/agent review")]


def test_mark_issue_intent_falls_back_to_label_source() -> None:
    runner, registry = _make_session_runner()
    issue = SimpleNamespace(id="issue-1")
    runner._mark_issue_intent(issue, "RETRY", None, None)
    assert registry.calls == [("issue-1", "RETRY", "label", None)]


# ---------------------------------------------------------------------------
# _process_escalated_issues
# ---------------------------------------------------------------------------


class _EscalationRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def mark_failed(self, issue_id: str) -> None:
        self.calls.append(("mark_failed", issue_id))

    def mark_abandoned(self, issue_id: str) -> None:
        self.calls.append(("mark_abandoned", issue_id))


async def _drive(tmp_path, policy: str, issue_ids: list[str], completed=(), claimed=()):
    sentinel = tmp_path / ".escalated_issues.json"
    sentinel.write_text(json.dumps({i: {"rounds": 3} for i in issue_ids}))
    runner = OrchestratorOpsMixin()
    registry = _EscalationRegistry()
    synced: list[tuple[str, str]] = []
    emitted: list[tuple] = []
    runner._workspace_root = tmp_path
    runner._registry = registry
    runner._clarification_resolver = SimpleNamespace(_config=SimpleNamespace(escalation=policy))
    runner._state = SimpleNamespace(completed=set(completed), claimed=set(claimed))
    sync_calls = synced

    async def _sync(issue_id: str, status: str) -> None:
        synced.append((issue_id, status))

    runner._sync_tracker_issue_state = _sync
    runner._emit_im_event = lambda *args, **kwargs: emitted.append((args, kwargs))
    await runner._process_escalated_issues()
    return registry, sync_calls, emitted, sentinel


async def test_escalation_mark_failed(tmp_path) -> None:
    registry, synced, emitted, _sentinel = await _drive(tmp_path, "mark_failed", ["issue-1"])
    assert registry.calls == [("mark_failed", "issue-1")]
    assert synced == [("issue-1", "failed")]
    assert emitted and emitted[0][0][2] is EventLevel.WARN


async def test_escalation_notify_uses_error_level(tmp_path) -> None:
    registry, synced, emitted, _sentinel = await _drive(tmp_path, "notify", ["issue-1"])
    assert registry.calls == [("mark_failed", "issue-1")]
    assert synced == [("issue-1", "failed")]
    assert emitted and emitted[0][0][2] is EventLevel.ERROR


async def test_escalation_skip_abandons(tmp_path) -> None:
    registry, synced, emitted, _sentinel = await _drive(tmp_path, "skip", ["issue-1"])
    assert registry.calls == [("mark_abandoned", "issue-1")]
    assert synced == [("issue-1", "abandoned")]
    assert emitted and emitted[0][0][2] is EventLevel.WARN


async def test_escalation_unknown_policy_falls_back_to_skip(tmp_path) -> None:
    registry, synced, _emitted, _sentinel = await _drive(tmp_path, "unknown", ["issue-1"])
    assert registry.calls == [("mark_abandoned", "issue-1")]
    assert synced == [("issue-1", "abandoned")]


async def test_escalation_skips_completed_and_prunes_sentinel(tmp_path) -> None:
    registry, _synced, _emitted, sentinel = await _drive(
        tmp_path, "skip", ["done-1", "pending-1"], completed=["done-1"]
    )
    # done-1 is pruned without a registry transition; pending-1 is processed.
    assert registry.calls == [("mark_abandoned", "pending-1")]
    remaining = json.loads(sentinel.read_text())
    assert remaining == {}


async def test_escalation_missing_sentinel_is_noop(tmp_path) -> None:
    registry, synced, _emitted, _sentinel = await _drive(tmp_path, "notify", [])
    assert registry.calls == []
    assert synced == []


def test_escalation_action_table_shape() -> None:
    # Every policy record has exactly (registry, tracker, log level, log
    # fmt, event level) and maps to existing registry/log members.
    for _mark, tracker, log_level, log_fmt, level in _ESCALATION_ACTIONS.values():
        assert tracker in ("failed", "abandoned")
        assert log_level in ("", "info", "warning")
        assert log_fmt == "" or log_fmt.endswith("%s")
        assert level in (EventLevel.WARN, EventLevel.ERROR)
