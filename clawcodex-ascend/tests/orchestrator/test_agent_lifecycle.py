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
"""Focused tests for Agent Runtime lifecycle helpers."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace

import pytest

from extensions.orchestrator.agent_lifecycle import AgentLifecycleMixin


def _runner(**config) -> AgentLifecycleMixin:
    runner = AgentLifecycleMixin()
    runner.agent_config = SimpleNamespace(**config)
    return runner


def test_build_run_id_formats_regular_and_followup(monkeypatch) -> None:
    class _FixedDateTime:
        @staticmethod
        def now(_timezone):
            return datetime(2026, 8, 17, 10, 30, tzinfo=timezone.utc)

    monkeypatch.setattr("extensions.orchestrator.agent_lifecycle.datetime", _FixedDateTime)
    runner = _runner()
    regular = SimpleNamespace(run_kind="agent", attempt=3)
    followup = SimpleNamespace(
        run_kind="review_followup",
        attempt=3,
        issue_attempt=2,
        followup_attempt=4,
    )
    assert runner._build_run_id(regular) == "run-03-20260817T103000Z"
    assert runner._build_run_id(followup) == ("run-2-followup-4-20260817T103000Z")


@pytest.mark.asyncio
async def test_summary_placeholder_remembers_comment_id() -> None:
    class _Tracker:
        @staticmethod
        async def create_comment(issue_id: str, body: str):
            assert issue_id == "I-137"
            assert "Run in progress" in body
            return SimpleNamespace(id="comment-1")

    session = SimpleNamespace(issue=SimpleNamespace(id="I-137"), summary_comment_id=None)
    await _runner()._post_summary_placeholder(session, _Tracker())
    assert session.summary_comment_id == "comment-1"


@pytest.mark.asyncio
async def test_state_cache_can_skip_tracker_poll() -> None:
    class _Cache:
        @staticmethod
        def has_recent_inactive(_issue_id, _turn):
            return False

        @staticmethod
        def should_skip_poll(_issue_id, _turn):
            return True

    class _Tracker:
        @staticmethod
        async def fetch_issue_states_by_ids(_ids):
            raise AssertionError("tracker must be skipped")

    issue = SimpleNamespace(id="I-137", state="open")
    session = SimpleNamespace(state_cache=_Cache(), turn_count=2, user_interrupted=False)
    should_continue, returned = await _runner()._should_continue(issue, _Tracker(), session)
    assert should_continue is True
    assert returned is issue


@pytest.mark.asyncio
async def test_inactive_tracker_state_stops_run() -> None:
    refreshed = SimpleNamespace(id="I-137", state="closed")

    class _Tracker:
        active_states = ["open", "in progress"]

        @staticmethod
        async def fetch_issue_states_by_ids(_ids):
            return {"I-137": refreshed}

    active, returned = await _runner()._should_continue(SimpleNamespace(id="I-137", state="open"), _Tracker())
    assert active is False
    assert returned is refreshed


def test_subprocess_env_expands_path(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "existing-path")
    environment = _runner(env={"PATH": "tools;$PATH", "MODE": "test"})._build_subprocess_env()
    assert environment["PATH"] == "tools;existing-path"
    assert environment["MODE"] == "test"


def test_subprocess_env_returns_none_without_overrides() -> None:
    assert _runner(env={})._build_subprocess_env() is None


@pytest.mark.asyncio
async def test_empty_recent_commits_are_detected(tmp_path) -> None:
    runner = _runner(env={})
    runner._git_capture = lambda *_args: SimpleNamespace(returncode=0, stdout="")
    session = SimpleNamespace(workspace=SimpleNamespace(path=tmp_path))
    assert await runner._recent_commits_are_empty(session) is True


@pytest.mark.asyncio
async def test_workspace_state_failures_are_logged(caplog, monkeypatch, tmp_path) -> None:
    compat = ModuleType("extensions.orchestrator_runtime.adapters.clawcodex_compat")
    compat.get_file_status = lambda _path: []
    monkeypatch.setitem(
        sys.modules,
        "extensions.orchestrator_runtime.adapters.clawcodex_compat",
        compat,
    )
    runner = _runner(env={})
    runner._git_capture = lambda *_args: (_ for _ in ()).throw(RuntimeError("git unavailable"))
    session = SimpleNamespace(
        issue=SimpleNamespace(id="I-137"),
        workspace=SimpleNamespace(path=tmp_path),
        start_commit_sha="start",
    )

    with caplog.at_level(logging.WARNING):
        assert await runner._workspace_completion_state(session) is None
        assert await runner._recent_commits_are_empty(session) is False

    assert "Workspace completion state check failed issue_id=I-137" in caplog.text
    assert "Recent commit check failed issue_id=I-137" in caplog.text
    assert "git unavailable" in caplog.text


def test_export_events_copies_run_artifacts(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    reports = workspace / ".reports"
    reports.mkdir(parents=True)
    (reports / "run-1.events.ndjson").write_text("event\n", encoding="utf-8")
    (reports / "agent_spawns.ndjson").write_text("spawn\n", encoding="utf-8")
    monkeypatch.setattr("extensions.orchestrator.agent_lifecycle.Path.home", lambda: home)
    session = SimpleNamespace(run_id="run-1", workspace=SimpleNamespace(path=workspace))

    _runner()._export_events_for_viz(session)

    destination = home / ".clawcodex/sessions/run-1"
    assert (destination / "events.ndjson").read_text() == "event\n"
    assert (destination / "agent_spawns.ndjson").read_text() == "spawn\n"


@pytest.mark.asyncio
async def test_verification_skips_when_no_command() -> None:
    assert await _runner(test_command=None)._run_verification(SimpleNamespace()) is True


@pytest.mark.asyncio
async def test_verification_requires_workspace() -> None:
    session = SimpleNamespace(workspace=None, issue=SimpleNamespace(id="I-137"))
    assert await _runner(test_command="pytest")._run_verification(session) is False


@pytest.mark.asyncio
async def test_verification_timeout_kills_and_reaps_process(monkeypatch, tmp_path) -> None:
    session_module = ModuleType("extensions.orchestrator.agent_session")
    session_module._set_pdeathsig = None
    monkeypatch.setitem(
        sys.modules,
        "extensions.orchestrator.agent_session",
        session_module,
    )

    class _Process:
        def __init__(self) -> None:
            self.returncode = None
            self.killed = False

        async def communicate(self):
            if not self.killed:
                await asyncio.Event().wait()
            return b"", b""

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = _Process()

    async def _create_subprocess_shell(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _create_subprocess_shell)
    runner = _runner(
        test_command="pytest",
        verification=SimpleNamespace(timeout_ms=1),
        env={},
    )
    session = SimpleNamespace(
        workspace=SimpleNamespace(path=tmp_path),
        issue=SimpleNamespace(id="I-137"),
    )

    assert await runner._run_verification(session) is False
    assert process.killed is True
