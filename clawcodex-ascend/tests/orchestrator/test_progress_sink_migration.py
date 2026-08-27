#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# ruff: noqa: UP009

"""Focused migration tests for orchestrator progress fan-out."""

from __future__ import annotations

from types import SimpleNamespace

from clawcodex_ext.tool_system.context import ToolContext
from extensions.api.query import PhaseComplete, SessionComplete, TurnComplete
from extensions.orchestrator.progress_reporter import ProgressReporter
from extensions.orchestrator.progress_sink import (
    CompositeProgressSink,
    ProgressSink,
    ToolContextProgressSink,
)


class _RecordingSink:
    task_id = "recording"

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_phase_complete(self, _event, _session) -> None:
        self.events.append("phase")

    def on_turn_complete(self, _event, _session) -> None:
        self.events.append("turn")

    def on_session_complete(self, _event, _session) -> None:
        self.events.append("session")


class _FailingSink(_RecordingSink):
    def on_phase_complete(self, _event, _session) -> None:
        raise RuntimeError("private-provider-token")


def _session():
    return SimpleNamespace(
        issue=SimpleNamespace(identifier="F-40"),
        status="running",
        turn_count=2,
    )


def _context(*task_ids: str) -> ToolContext:
    context = ToolContext(workspace_root="/tmp")
    for task_id in task_ids:
        context.tasks[task_id] = {"id": task_id, "metadata": {}}
    return context


def test_composite_fans_out_and_satisfies_protocol() -> None:
    first = _RecordingSink()
    second = _RecordingSink()
    sink = CompositeProgressSink([first, second])

    assert isinstance(sink, ProgressSink)
    sink.on_phase_complete(PhaseComplete(phase=1, turn_count=1), _session())
    sink.on_turn_complete(TurnComplete(turn=1), _session())
    sink.on_session_complete(SessionComplete(reason="success"), _session())

    assert first.events == ["phase", "turn", "session"]
    assert second.events == first.events


def test_composite_isolates_failure_and_redacts_exception(caplog) -> None:
    good = _RecordingSink()
    sink = CompositeProgressSink([_FailingSink(), good])

    with caplog.at_level("WARNING", logger="extensions.orchestrator.progress_sink"):
        sink.on_phase_complete(PhaseComplete(phase=1, turn_count=1), _session())

    assert good.events == ["phase"]
    assert "RuntimeError" in caplog.text
    assert "private-provider-token" not in caplog.text


def test_duplicate_phase_names_use_positional_progress() -> None:
    context = _context("f-40")
    sink = ToolContextProgressSink(
        task_id="f-40",
        context=context,
        workflow_phases=["review", "review", "ship"],
    )

    sink.on_phase_complete(PhaseComplete(phase=1, turn_count=1), _session())
    sink.on_phase_complete(PhaseComplete(phase=2, turn_count=2), _session())

    stages = context.tasks["f-40"]["metadata"]["progress_stages"]
    assert [stage["stage"] for stage in stages] == ["review", "review"]
    assert [stage["progress"] for stage in stages] == [33, 66]


def test_configured_progress_is_bounded() -> None:
    sink = ToolContextProgressSink(
        task_id="f-40",
        context=_context("f-40"),
        workflow_phases=["analysis", "ship"],
    )

    assert sink._phase_progress(-1) == 0
    assert sink._phase_progress(0) == 0
    assert sink._phase_progress(3) == 100


def test_only_success_reports_terminal_100_percent() -> None:
    context = _context("success", "failure")
    success = ToolContextProgressSink(task_id="success", context=context)
    failure = ToolContextProgressSink(task_id="failure", context=context)

    success.on_session_complete(SessionComplete(reason="success"), _session())
    failure.on_session_complete(SessionComplete(reason="stagnation"), _session())

    success_stage = context.tasks["success"]["metadata"]["progress_stages"][0]
    failure_stage = context.tasks["failure"]["metadata"]["progress_stages"][0]
    assert success_stage["progress"] == 100
    assert failure_stage.get("progress") is None


def test_progress_reporter_shim_keeps_legacy_event_api() -> None:
    context = _context("f-40")
    reporter = ProgressReporter(context)
    reporter.set_task_id("f-40")

    reporter.on_event(PhaseComplete(phase=1, turn_count=1), _session())
    reporter.on_event(SessionComplete(reason="success"), _session())

    stages = context.tasks["f-40"]["metadata"]["progress_stages"]
    assert [stage["stage"] for stage in stages] == ["phase_1", "session_success"]
