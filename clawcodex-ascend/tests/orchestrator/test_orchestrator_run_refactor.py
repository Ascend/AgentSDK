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

"""Focused tests for the ``_run_issue`` split helpers in orchestrator_run.

``_run_issue`` was split into small private helpers (agent execution, session
completion, exception recording, viz journal and terminal-status routing).
These tests drive the extracted helpers directly with stub collaborators to
pin the routing / finalization behaviour that previously lived in the giant
method body.
"""

from __future__ import annotations

from types import SimpleNamespace

from extensions.orchestrator.orchestrator_run import OrchestratorRunMixin


class _Recorder:
    """Records every call made against a fake collaborator."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _rec(self, name, *args, **kwargs) -> None:
        self.calls.append((name, args, kwargs))


class _RegistryRecorder(_Recorder):
    def get(self, issue_id: str):
        return None

    def flush(self) -> None:
        self._rec("flush")

    def mark_completed(self, issue_id: str) -> None:
        self._rec("mark_completed", issue_id)

    def mark_failed(self, issue_id: str) -> None:
        self._rec("mark_failed", issue_id)

    def mark_failed_with_reason(self, issue_id: str, reason: str) -> None:
        self._rec("mark_failed_with_reason", issue_id, reason)

    def mark_verification_failed(self, issue_id: str, output=None, hook_error=None) -> None:
        self._rec("mark_verification_failed", issue_id, output, hook_error)

    def update_report(self, issue_id: str, **kwargs) -> None:
        self._rec("update_report", issue_id, kwargs)


class _DashboardRecorder(_Recorder):
    def on_session_start(self, session_status) -> None:
        self._rec("on_session_start", session_status)

    def on_session_complete(self, issue_id: str) -> None:
        self._rec("on_session_complete", issue_id)

    def on_session_failed(self, issue_id: str, status: str) -> None:
        self._rec("on_session_failed", issue_id, status)


def _make_runner(**workflow_agent) -> tuple[OrchestratorRunMixin, _RegistryRecorder, _DashboardRecorder]:
    runner = OrchestratorRunMixin()
    state = SimpleNamespace(
        running={},
        completed=set(),
        failed=set(),
        pending_review=set(),
        claimed=set(),
    )
    registry = _RegistryRecorder()
    dashboard = _DashboardRecorder()
    sync_calls: list[tuple[str, str]] = []
    retry_calls: list[tuple[object, dict]] = []
    emitted: list[tuple] = []

    async def _sync(issue_id: str, status: str) -> None:
        sync_calls.append((issue_id, status))

    async def _retry(session, **kwargs) -> None:
        retry_calls.append((session, kwargs))

    runner._state = state
    runner._registry = registry
    runner.status_dashboard = dashboard
    runner._sync_tracker_issue_state = _sync
    runner._schedule_retry = _retry
    runner._emit_im_event = lambda *args, **kwargs: emitted.append((args, kwargs))
    runner._session_payload = lambda session, **kwargs: {}
    agent = SimpleNamespace(rate_limit_max_backoff_ms=60_000, max_turns_retry_delay_ms=30_000)
    agent.__dict__.update(workflow_agent)
    runner.workflow = SimpleNamespace(agent=agent)
    return runner, registry, dashboard, sync_calls, retry_calls, emitted


def _session(status: str, issue_id: str = "issue-1", **extra) -> SimpleNamespace:
    data = {
        "workspace": SimpleNamespace(path="/ws"),
        "status": status,
        "session_end_reason": None,
        "session_end_summary": "",
        "verification_status": None,
        "verification_output": None,
        "last_hook_error": None,
        "turn_count": 0,
        "tool_count": 0,
        "run_id": None,
    }
    data.update(extra)
    return SimpleNamespace(issue=SimpleNamespace(id=issue_id), **data)


def _names(recorder: _Recorder) -> list[str]:
    return [name for name, _args, _kwargs in recorder.calls]


async def test_route_terminal_status_completed() -> None:
    runner, registry, dashboard, sync_calls, _retry, emitted = _make_runner()
    await runner._route_session_terminal_status(_session("completed"))
    assert "issue-1" in runner._state.completed
    assert _names(registry) == ["mark_completed"]
    assert _names(dashboard) == ["on_session_complete"]
    assert sync_calls == [("issue-1", "completed")]
    event_types = [args[1] for args, _kwargs in emitted]
    assert "issue.completed" in event_types


async def test_route_terminal_status_pending_review_short_circuits() -> None:
    runner, registry, dashboard, _sync, _retry, _emitted = _make_runner()
    runner._state.pending_review.add("issue-1")
    await runner._route_session_terminal_status(_session("completed"))
    assert "issue-1" not in runner._state.completed
    assert _names(registry) == []
    assert _names(dashboard) == []


async def test_route_terminal_status_verification_failed_retries() -> None:
    runner, registry, _dashboard, _sync, retry_calls, _emitted = _make_runner()
    await runner._route_session_terminal_status(_session("verification_failed"))
    assert [n for n, _a, _k in registry.calls] == ["mark_verification_failed"]
    assert len(retry_calls) == 1


async def test_route_terminal_status_rate_limit_uses_max_backoff() -> None:
    runner, registry, _dashboard, _sync, retry_calls, _emitted = _make_runner(rate_limit_max_backoff_ms=123_456)
    await runner._route_session_terminal_status(_session("rate_limit_circuit_open"))
    assert [n for n, _a, _k in registry.calls] == ["mark_failed"]
    assert retry_calls and retry_calls[0][1] == {"delay_base_ms": 123_456}


async def test_route_terminal_status_cancelled_does_not_retry() -> None:
    runner, registry, _dashboard, _sync, retry_calls, _emitted = _make_runner()
    await runner._route_session_terminal_status(_session("cancelled"))
    assert [n for n, _a, _k in registry.calls] == ["mark_failed"]
    assert retry_calls == []


async def test_route_terminal_status_generic_failure_uses_operator_failure_detail() -> None:
    runner, registry, _dashboard, _sync, retry_calls, _emitted = _make_runner()
    session = _session("failed", operator_failure_detail="request_failed status=500: boom")
    await runner._route_session_terminal_status(session)
    assert _names(registry) == ["mark_failed_with_reason", "update_report"]
    assert registry.calls[0][1][1] == "request_failed status=500: boom"
    assert len(retry_calls) == 1


async def test_route_terminal_status_stagnation_marks_failed_no_retry() -> None:
    runner, registry, _dashboard, _sync, retry_calls, _emitted = _make_runner()
    await runner._route_session_terminal_status(_session("stagnation"))
    assert [n for n, _a, _k in registry.calls] == ["mark_failed"]
    assert retry_calls == []


async def test_handle_run_failure_before_run_status() -> None:
    runner = OrchestratorRunMixin()
    session = _session("completed")
    await runner._handle_run_failure(session, RuntimeError("boom"), ran_agent=False)
    assert session.status == "before_run_failed"
    assert session.session_end_summary == "boom"
    assert session.operator_failure_detail == "boom"
    assert session.verification_status == "failed"


async def test_handle_run_failure_extracts_request_detail() -> None:
    runner = OrchestratorRunMixin()
    session = _session("completed")
    exc = Exception('request_failed status=500 body={"error_message":"server exploded"}')
    await runner._handle_run_failure(session, exc, ran_agent=True)
    assert session.status == "failed"
    assert session.operator_failure_detail == "request_failed status=500: server exploded"


async def test_handle_run_timeout_returns_workspace_dirty(monkeypatch) -> None:
    import extensions.orchestrator.orchestrator_run as run_mod

    monkeypatch.setattr(run_mod, "get_file_status", lambda _path: ["modified.txt"])
    emitted: list[tuple] = []
    runner = OrchestratorRunMixin()
    runner._emit_im_event = lambda *args, **kwargs: emitted.append(args)
    runner._session_payload = lambda session, **kwargs: {}
    runner.workflow = SimpleNamespace(agent=SimpleNamespace(run_timeout_ms=10_000))
    session = _session("completed")
    dirty = await runner._handle_run_timeout(session)
    assert dirty is True
    assert session.status == "agent_timeout"
    assert session.verification_status == "failed"
    assert "Agent run exceeded configured timeout" in session.verification_output
    assert emitted and emitted[0][1] == "issue.failed"


async def test_mark_verification_failed_sets_status_trio() -> None:
    runner = OrchestratorRunMixin()
    session = _session("completed")
    runner._mark_verification_failed(session, "boom")
    assert session.status == "verification_failed"
    assert session.verification_status == "failed"
    assert session.verification_output == "boom"


async def test_start_issue_run_syncs_dashboard_and_tracks_task() -> None:
    import asyncio

    runner = OrchestratorRunMixin()
    runner._tasks = set()
    runner._issue_tasks = {}
    runner.agent_runner = SimpleNamespace(max_turns=7)
    runner._sync_gitignore_to_workspace = lambda ws: None
    dashboard = _DashboardRecorder()
    runner.status_dashboard = dashboard
    started: list[SimpleNamespace] = []
    runner._run_issue = _make_run_issue(started)
    session = _session("running", issue_id="issue-9")
    session.issue.identifier = "ISSUE-9"

    runner._start_issue_run(session)

    # The task is registered for cancellation/cleanup while it runs.
    assert "issue-9" in runner._issue_tasks
    assert len(runner._tasks) == 1
    assert len(dashboard.calls) == 1
    assert _names(dashboard) == ["on_session_start"]
    status = dashboard.calls[0][1][0]
    assert status.issue_id == "issue-9"
    assert status.issue_identifier == "ISSUE-9"
    assert status.max_turns == 7
    assert status.workspace_path == "/ws"

    await asyncio.sleep(0)
    assert started == [session]
    # Done-callbacks are scheduled after task completion; let the loop
    # run them before asserting unregistration.
    await asyncio.sleep(0)
    assert "issue-9" not in runner._issue_tasks
    assert not runner._tasks


def _make_run_issue(started):
    async def _run_issue(session) -> None:
        started.append(session)

    return _run_issue
