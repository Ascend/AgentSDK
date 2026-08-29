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

from __future__ import annotations

import json

from clawcodex_ext.intent_forecast.config import IntentForecastConfig
from clawcodex_ext.intent_forecast.context import IntentForecastContextBuilder


def test_workspace_signals_read_last_command_failures_and_test_mapping(tmp_path) -> None:
    claw = tmp_path / ".clawcodex"
    claw.mkdir()
    (claw / "last_command.json").write_text(
        json.dumps(
            {
                "command": "pytest tests/intent_forecast",
                "exit_code": 1,
                "output": "FAILED tests/intent_forecast/test_service.py::test_x",
            }
        ),
        encoding="utf-8",
    )

    context = IntentForecastContextBuilder(
        conversation=None,
        workspace_root=tmp_path,
        config=IntentForecastConfig(),
        sessions_dir=tmp_path / "sessions",
        feedback_base_dir=tmp_path,
    ).build()

    assert context.workspace["last_command"] == "pytest tests/intent_forecast"
    assert context.workspace["last_command_exit"] == 1
    assert context.workspace["last_test_failures"] == ["FAILED tests/intent_forecast/test_service.py::test_x"]
    assert context.intent_stage == "debug"


def test_workspace_signals_collects_recent_commits(monkeypatch, tmp_path) -> None:
    """recent_commits should be parsed from `git log` into structured rows."""

    field_outputs = {
        "%H": "abc1234567890def\ndef2345678901abc",
        "%h": "abc1234\ndef2345",
        "%s": "fix(intent_forecast): no_suggestion_gate threshold\nfeat: add recent_commits signal",
        "%an": "alice\nbob",
        "%at": "1718000000\n1717900000",
    }

    def _fake_run_git(cwd, args):
        if not args or args[0] != "log":
            return ""
        fmt = ""
        for arg in args:
            if arg.startswith("--format="):
                fmt = arg[len("--format=") :]
                break
        return field_outputs.get(fmt, "")

    monkeypatch.setattr("clawcodex_ext.intent_forecast.context._run_git", _fake_run_git)

    context = IntentForecastContextBuilder(
        conversation=None,
        workspace_root=tmp_path,
        config=IntentForecastConfig(),
        sessions_dir=tmp_path / "sessions",
        feedback_base_dir=tmp_path,
    ).build()

    commits = context.workspace["recent_commits"]
    assert len(commits) == 2
    assert commits[0]["short_hash"] == "abc1234"
    assert commits[0]["subject"].startswith("fix(intent_forecast):")
    assert commits[0]["author"] == "alice"
    assert commits[0]["timestamp"] == "1718000000"
    assert commits[1]["short_hash"] == "def2345"


def test_workspace_signals_recent_commits_unaffected_by_pipe_in_subject(monkeypatch, tmp_path) -> None:
    """Subjects or authors containing `|` must not corrupt the other fields.

    Regression: the earlier single-call implementation used
    ``%H|%h|%s|%an|%at`` + ``split("|", 4)``. A ``|`` inside a subject or
    author name would shift the author/timestamp into the subject field,
    leaking garbage to the LLM and degrading next-step prediction quality.
    """

    field_outputs = {
        "%H": "abc1234567890def\ndef2345678901abc",
        "%h": "abc1234\ndef2345",
        "%s": "fix: handle a|b|c in subject\nfeat: another | pipe",
        "%an": "alice|with-pipe\nbob",
        "%at": "1718000000\n1717900000",
    }

    def _fake_run_git(cwd, args):
        if not args or args[0] != "log":
            return ""
        fmt = ""
        for arg in args:
            if arg.startswith("--format="):
                fmt = arg[len("--format=") :]
                break
        return field_outputs.get(fmt, "")

    monkeypatch.setattr("clawcodex_ext.intent_forecast.context._run_git", _fake_run_git)

    context = IntentForecastContextBuilder(
        conversation=None,
        workspace_root=tmp_path,
        config=IntentForecastConfig(),
        sessions_dir=tmp_path / "sessions",
        feedback_base_dir=tmp_path,
    ).build()

    commits = context.workspace["recent_commits"]
    assert len(commits) == 2
    assert commits[0]["subject"] == "fix: handle a|b|c in subject"
    assert commits[0]["author"] == "alice|with-pipe"
    assert commits[0]["timestamp"] == "1718000000"
    assert commits[1]["subject"] == "feat: another | pipe"
    assert commits[1]["author"] == "bob"
    assert commits[1]["timestamp"] == "1717900000"


def test_workspace_signals_recent_commits_empty_when_not_a_repo(monkeypatch, tmp_path) -> None:
    """Non-git workspace should yield an empty recent_commits list, not crash."""

    monkeypatch.setattr("clawcodex_ext.intent_forecast.context._run_git", lambda cwd, args: "")

    context = IntentForecastContextBuilder(
        conversation=None,
        workspace_root=tmp_path,
        config=IntentForecastConfig(),
        sessions_dir=tmp_path / "sessions",
        feedback_base_dir=tmp_path,
    ).build()

    assert context.workspace["recent_commits"] == []
