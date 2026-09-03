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

"""Regression coverage for orchestrator asciicast recording."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from extensions.api.query import PhaseComplete, SessionComplete, TurnComplete
from extensions.capabilities.recorder import AsciicastHeader
from extensions.orchestrator.asciicast_sink import AsciicastSink, format_phase_label
from extensions.recording.asciicast_writer import AsciicastWriter


class _FakeSession:
    """Minimal stand-in for the AgentSession interface."""


def _open_writer(tmp_path: Path) -> AsciicastWriter:
    writer = AsciicastWriter(
        tmp_path / "demo.cast",
        AsciicastHeader(width=120, height=36),
    )
    writer.open()
    return writer


def _frames(path: Path) -> list[list[Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[1:]]


def test_phase_and_session_events_are_recorded(tmp_path: Path) -> None:
    writer = _open_writer(tmp_path)
    sink = AsciicastSink(writer.capture, task_id="issue-7", phases_total=5)

    sink.on_phase_complete(PhaseComplete(phase=3, turn_count=12), _FakeSession())
    sink.on_session_complete(SessionComplete(reason="exit_code=0"), _FakeSession())
    writer.close()

    frames = _frames(tmp_path / "demo.cast")
    markers = [frame[2] for frame in frames if frame[1] == "m"]
    assert markers == ["[phase 3/5]", "session:exit_code=0"]
    assert any("issue-7" in frame[2] for frame in frames if frame[1] == "o")


def test_turn_events_are_not_written_to_capture(tmp_path: Path) -> None:
    writer = _open_writer(tmp_path)
    sink = AsciicastSink(writer.capture, task_id="issue-1")

    sink.on_turn_complete(TurnComplete(turn=7), _FakeSession())
    writer.close()

    lines = (tmp_path / "demo.cast").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_closed_capture_never_breaks_orchestrator(tmp_path: Path) -> None:
    writer = _open_writer(tmp_path)
    sink = AsciicastSink(writer.capture, task_id="issue-1", phases_total=3)
    writer.close()

    sink.on_phase_complete(PhaseComplete(phase=1, turn_count=4), _FakeSession())
    sink.on_session_complete(SessionComplete(reason="exit_code=1"), _FakeSession())


def test_phase_label_format() -> None:
    assert format_phase_label(3, 7) == "[phase 3/7]"
    assert format_phase_label(3, None) == "[phase 3]"


def test_record_cli_opens_orchestrator_source(tmp_path: Path) -> None:
    out_path = tmp_path / "orchestrator.cast"
    script = (
        "from clawcodex_ext.cli.subcommand_registry import get_subcommand; "
        "raise SystemExit(get_subcommand('record')(["
        "'--sources','orchestrator','--out',"
        f"{str(out_path)!r},'--duration','0.01s','--validate']))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    markers = [
        json.loads(line)[2]
        for line in out_path.read_text(encoding="utf-8").splitlines()[1:]
        if json.loads(line)[1] == "m"
    ]
    assert markers == [
        "orchestrator:recording_started",
        "orchestrator:recording_closed",
    ]


def test_record_cli_auto_runs_without_repository_tests(tmp_path: Path) -> None:
    out_path = tmp_path / "auto.cast"
    script = (
        "from clawcodex_ext.cli.subcommand_registry import get_subcommand; "
        "raise SystemExit(get_subcommand('record')(["
        "'--auto','--out',"
        f"{str(out_path)!r},'--auto-duration-s','0.05',"
        "'--auto-issue-count','1','--auto-frame-delay-s','0.01']))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "validation: OK" in result.stdout
    assert out_path.is_file()
