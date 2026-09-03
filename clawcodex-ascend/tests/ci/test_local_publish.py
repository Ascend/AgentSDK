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

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "scripts" / "ci" / "local_publish.py").is_file(),
    reason="standalone ClawCodex release helpers are not part of the AgentSDK runtime migration",
)


def _load_module(monkeypatch):
    sys.modules.pop("local_publish", None)
    monkeypatch.syspath_prepend("scripts/ci")
    return importlib.import_module("local_publish")


class _FakeStdout:
    def __init__(self, is_tty: bool) -> None:
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def _args(**overrides):
    values = {
        "release_target": None,
        "tag": "v0.5.0",
        "skip_tests": False,
        "skip_gitcode_release": False,
        "check_credentials": False,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_required_tokens_follow_target_and_gitcode_flag(monkeypatch):
    local_publish = _load_module(monkeypatch)

    assert local_publish._required_tokens("testpypi", False) == [
        "TEST_PYPI_TOKEN",
        "GITCODE_TOKEN",
    ]
    assert local_publish._required_tokens("pypi", True) == ["PYPI_TOKEN"]
    assert local_publish._required_tokens(None, True) == []
    assert local_publish._required_tokens(None, False) == ["GITCODE_TOKEN"]


def test_dry_run_plan_skips_credentials_but_keeps_publish_flow(monkeypatch):
    local_publish = _load_module(monkeypatch)

    steps = local_publish._build_step_plan(_args(dry_run=True, skip_gitcode_release=True))
    by_name = {step.name: step for step in steps}

    assert by_name["release / credentials"].skip_reason == "dry-run does not require publish tokens"
    assert by_name["ci / package-smoke"].skip_reason is None
    assert by_name["publish / package"].skip_reason is None
    assert by_name["publish / GitCode Release"].skip_reason == "--skip-gitcode-release was requested"


def test_check_credentials_mode_skips_build_and_publish(monkeypatch):
    local_publish = _load_module(monkeypatch)

    steps = local_publish._build_step_plan(_args(check_credentials=True))
    by_name = {step.name: step for step in steps}

    assert by_name["release / credentials"].skip_reason is None
    assert by_name["release / clean-tree"].skip_reason == "credential check mode"
    assert by_name["ci / package-smoke"].skip_reason == "credential check mode"
    assert by_name["publish / package"].skip_reason == "credential check mode"


def test_typecheck_step_is_blocking(monkeypatch):
    local_publish = _load_module(monkeypatch)

    steps = local_publish._build_step_plan(_args())
    by_name = {step.name: step for step in steps}

    assert "ci / typecheck-advisory" not in by_name
    assert by_name["ci / typecheck"].advisory is False
    assert by_name["ci / typecheck"].skip_reason is None


def test_local_publish_reports_release_artifact_locations(monkeypatch):
    local_publish = _load_module(monkeypatch)

    paths = {path for path, _description in local_publish._artifact_locations()}

    assert ".release-smoke/" in paths
    assert "dist/" in paths
    assert local_publish.SDIST_STAGING_DIR_GLOB in paths


def test_rmtree_recovers_from_unscannable_reparse_point(tmp_path, monkeypatch):
    local_publish = _load_module(monkeypatch)
    target = tmp_path / "tree"
    failed = target / "lib64"
    removed: list[str] = []

    def fake_rmtree(path, *, onerror):
        onerror(os.scandir, str(failed), (OSError, OSError("cannot scan"), None))

    monkeypatch.setattr(local_publish.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(local_publish.os.path, "lexists", lambda path: True)
    monkeypatch.setattr(local_publish.os, "rmdir", lambda path: removed.append(path))

    local_publish._rmtree(target)

    assert removed == [str(failed)]


def test_flow_table_marks_initial_skip_status(monkeypatch, capsys):
    local_publish = _load_module(monkeypatch)
    monkeypatch.setattr(local_publish, "RICH_AVAILABLE", False)
    steps = local_publish._build_step_plan(_args(dry_run=True))

    local_publish._print_flow_table(steps, title="Sample Flow")

    output = capsys.readouterr().out
    assert "[SKIP]" in output
    assert "dry-run does not require" in output


def test_resolve_tag_creates_missing_explicit_tag(monkeypatch):
    local_publish = _load_module(monkeypatch)
    commands: list[list[str]] = []

    def fake_run(args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    def fake_command(args, *, show_output=False):
        commands.append(args)
        return local_publish.CommandResult(args, 0, "", 0.0)

    def fake_git(args, *, show_output=False):
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        if args == ["rev-list", "-n", "1", "v0.5.0"]:
            return "abc123"
        raise AssertionError(args)

    monkeypatch.setattr(local_publish.subprocess, "run", fake_run)
    monkeypatch.setattr(local_publish, "_run_command", fake_command)
    monkeypatch.setattr(local_publish, "_git", fake_git)

    assert local_publish._resolve_tag("v0.5.0", create_missing=True) == (
        "v0.5.0",
        "abc123",
    )
    assert ["git", "tag", "v0.5.0", "abc123"] in commands


def test_resolve_tag_dry_run_does_not_create_missing_tag(monkeypatch):
    local_publish = _load_module(monkeypatch)
    commands: list[list[str]] = []

    def fake_run(args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    def fake_command(args, *, show_output=False):
        commands.append(args)
        return local_publish.CommandResult(args, 0, "", 0.0)

    def fake_git(args, *, show_output=False):
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        raise AssertionError(args)

    monkeypatch.setattr(local_publish.subprocess, "run", fake_run)
    monkeypatch.setattr(local_publish, "_run_command", fake_command)
    monkeypatch.setattr(local_publish, "_git", fake_git)

    assert local_publish._resolve_tag(
        "v0.5.0",
        create_missing=True,
        dry_run=True,
    ) == ("v0.5.0", "abc123")
    assert ["git", "tag", "v0.5.0", "abc123"] not in commands


def test_auto_ui_enables_rich_live_for_interactive_stdout(monkeypatch):
    local_publish = _load_module(monkeypatch)
    if not local_publish.RICH_AVAILABLE:
        return

    ui_mode, reason = local_publish._resolve_ui(
        "auto",
        {},
        _FakeStdout(is_tty=True),
    )

    assert ui_mode == "live"
    assert reason == "stdout is interactive"


def test_auto_ui_uses_static_rich_for_powershell_wrapped_stdout(monkeypatch):
    local_publish = _load_module(monkeypatch)
    if not local_publish.RICH_AVAILABLE:
        return

    ui_mode, reason = local_publish._resolve_ui(
        "auto",
        {"PSModulePath": "C:\\Program Files\\PowerShell\\Modules"},
        _FakeStdout(is_tty=False),
    )

    assert ui_mode == "static"
    assert "PowerShell" in reason


def test_flow_runner_notifies_renderer_on_status_changes(monkeypatch):
    local_publish = _load_module(monkeypatch)
    step = local_publish.FlowStep("release / sample", "sample release step")
    seen: list[str] = []
    runner = local_publish.FlowRunner(
        [step],
        failure_lines=5,
        on_update=lambda: seen.append(step.status),
    )

    assert runner.run(step, lambda: None) is True

    assert seen == ["RUN", "PASS"]


def test_flow_runner_notifies_renderer_for_skipped_steps(monkeypatch):
    local_publish = _load_module(monkeypatch)
    step = local_publish.FlowStep(
        "release / sample",
        "sample release step",
        skip_reason="not needed",
    )
    seen: list[str] = []
    runner = local_publish.FlowRunner(
        [step],
        failure_lines=5,
        on_update=lambda: seen.append(step.status),
    )

    assert runner.run(step, lambda: None) is True

    assert seen == ["SKIP"]


def test_flow_runner_marks_dynamic_package_upload_skip(monkeypatch):
    local_publish = _load_module(monkeypatch)
    step = local_publish.FlowStep("publish / package", "upload package")
    seen: list[str] = []
    runner = local_publish.FlowRunner(
        [step],
        failure_lines=5,
        on_update=lambda: seen.append(step.status),
    )

    assert runner.run(step, lambda: (_ for _ in ()).throw(local_publish.StepSkipped("token missing")))

    assert step.status == "SKIP"
    assert step.result == "token missing"
    assert seen == ["RUN", "SKIP"]
