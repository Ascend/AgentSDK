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

"""Spec-3 tests for the upstream-compatible `/goal` slash command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clawcodex_ext.command_system.engine import CommandEngine
from clawcodex_ext.command_system.registry import CommandRegistry
from clawcodex_ext.command_system.types import CommandContext, NullUIHost, UIOption
from clawcodex_ext.agent.conversation import Conversation
from clawcodex_ext.feature_gate import get_registry, reset_registry
from clawcodex_ext.goal.command import GOAL_COMMAND
from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
from clawcodex_ext.goal.protocol import GoalEventLog, ThreadGoalProtocol
from clawcodex_ext.goal.files import MAX_THREAD_GOAL_OBJECTIVE_CHARS
from clawcodex_ext.goal.service import GoalService, GoalServiceError
from clawcodex_ext.goal.store import GoalStore, goals_db_filename
from src.command_system.safe_commands import is_bridge_safe_command
from src.tui.state import AppState


def _run(coro):
    return asyncio.run(coro)


class ScriptedUIHost(NullUIHost):
    def __init__(
        self,
        *,
        selects: list[str | None] | None = None,
        prompts: list[str | None] | None = None,
    ) -> None:
        self.selects = list(selects or [])
        self.prompts = list(prompts or [])
        self.seen_selects: list[tuple[str, list[UIOption]]] = []
        self.seen_prompts: list[tuple[str, str]] = []

    async def select(
        self,
        title: str,
        options: list[UIOption],
        *,
        current: str | None = None,
    ) -> str | None:
        del current
        self.seen_selects.append((title, list(options)))
        return self.selects.pop(0) if self.selects else None

    async def prompt_text(
        self,
        title: str,
        *,
        default: str = "",
        placeholder: str | None = None,
    ) -> str | None:
        del placeholder
        self.seen_prompts.append((title, default))
        return self.prompts.pop(0) if self.prompts else None


def _context(tmp_path: Path, ui: ScriptedUIHost | None = None) -> CommandContext:
    service = GoalService(
        store=GoalStore(tmp_path / goals_db_filename()),
        codex_home=tmp_path / "codex-home",
    )
    events = GoalEventLog()
    api = ThreadGoalProtocol(service=service, events=events)
    context = CommandContext(
        workspace_root=Path("/tmp"),
        cwd=Path("/tmp"),
        conversation=Conversation(),
        tool_context=SimpleNamespace(session_id="spec-1"),
        ui=ui or NullUIHost(),
    )
    context.goal_service = service  # type: ignore[attr-defined]
    context.goal_api = api  # type: ignore[attr-defined]
    context.goal_events = events  # type: ignore[attr-defined]
    return context


@pytest.fixture(autouse=True)
def _fresh_feature_registry():
    reg = reset_registry()
    yield reg
    reset_registry()


def test_goal_command_is_feature_gated_by_goals_flag():
    reg = get_registry()

    assert GOAL_COMMAND.is_enabled() is True
    assert reg.get_flag("goals") is not None

    reg.set_override("goals", False)

    assert GOAL_COMMAND.is_enabled() is False


def test_goal_command_remains_interactive_not_bridge_safe():
    assert is_bridge_safe_command(GOAL_COMMAND) is False


def test_goal_command_user_visible_arguments_only_include_upstream_set():
    assert GOAL_COMMAND.argument_hint == "[<condition>|clear]"
    assert GOAL_COMMAND.aliases == []


def test_bare_goal_shows_usage_when_no_goal_is_set(tmp_path: Path):
    outcome = _run(GOAL_COMMAND.run("", _context(tmp_path)))

    assert outcome.display == "system"
    assert outcome.should_query is False
    assert "/goal <condition> to set one" in outcome.message
    assert "No goal set" in outcome.message


def test_goal_objective_creates_active_goal(tmp_path: Path):
    context = _context(tmp_path)

    outcome = _run(GOAL_COMMAND.run("ship spec 3", context))

    assert outcome.display == "system"
    assert outcome.should_query is True
    assert outcome.message == "Goal set: ship spec 3"
    assert context.goal_service.get_goal("spec-1").objective == "ship spec 3"  # type: ignore[attr-defined]
    assert (  # type: ignore[attr-defined]
        context.goal_service.get_goal("spec-1").completion_mode is GoalCompletionMode.EVALUATOR
    )


def test_goal_command_uses_tool_context_goal_service_for_runtime_continuation(
    tmp_path: Path,
):
    from clawcodex_ext.goal.runtime import (
        EVALUATOR_START_MARKER,
        goal_runtime_for_context,
    )

    service = GoalService(
        store=GoalStore(tmp_path / goals_db_filename()),
        codex_home=tmp_path / "codex-home",
    )
    tool_context = SimpleNamespace(
        session_id="spec-1",
        goal_service=service,
    )
    context = CommandContext(
        workspace_root=Path("/tmp"),
        cwd=Path("/tmp"),
        tool_context=tool_context,
        ui=NullUIHost(),
    )

    _run(GOAL_COMMAND.run("continue after slash command", context))

    runtime = goal_runtime_for_context(tool_context)
    request = runtime.continue_if_idle() if runtime is not None else None

    assert service.get_goal("spec-1").objective == "continue after slash command"
    assert request is not None
    assert EVALUATOR_START_MARKER in str(request.messages[0].content)


def test_goal_condition_rejects_more_than_4000_characters(tmp_path: Path):
    context = _context(tmp_path)
    objective = "x" * (MAX_THREAD_GOAL_OBJECTIVE_CHARS + 1)
    registry = CommandRegistry()
    registry.register(GOAL_COMMAND)
    engine = CommandEngine(
        registry=registry,
        workspace_root=tmp_path,
        context=context,
    )

    result = _run(engine.execute(f"/goal {objective}"))

    assert result.success is False
    assert "4,000 characters" in result.error
    assert context.goal_service.get_goal("spec-1") is None  # type: ignore[attr-defined]


def test_goal_objective_replaces_unfinished_goal_without_confirmation(
    tmp_path: Path,
):
    ui = ScriptedUIHost(selects=["cancel"])
    context = _context(tmp_path, ui)
    old = context.goal_service.set_goal("spec-1", "current")  # type: ignore[attr-defined]

    outcome = _run(GOAL_COMMAND.run("replacement", context))
    current = context.goal_service.get_goal("spec-1")  # type: ignore[attr-defined]

    assert outcome.message == "Goal set: replacement"
    assert ui.seen_selects == []
    assert current.objective == "replacement"
    assert current.goal_id != old.goal_id
    assert [message.method for message in context.goal_events.messages] == [  # type: ignore[attr-defined]
        "thread/goal/set",
        "thread/goal/updated",
    ]


def test_goal_replace_failure_is_command_failure_and_preserves_current_goal(
    tmp_path: Path,
):
    context = _context(tmp_path)
    current = context.goal_service.set_goal("spec-1", "keep me")  # type: ignore[attr-defined]
    context.goal_events.clear()  # type: ignore[attr-defined]
    context.goal_service.replace_goal = Mock(  # type: ignore[attr-defined,method-assign]
        side_effect=GoalServiceError("replace transaction failed")
    )
    registry = CommandRegistry()
    registry.register(GOAL_COMMAND)
    engine = CommandEngine(
        registry=registry,
        workspace_root=tmp_path,
        context=context,
    )

    result = _run(engine.execute("/goal replacement"))

    assert result.success is False
    assert result.error == "replace transaction failed"
    assert context.goal_service.store.get_thread_goal("spec-1") == current  # type: ignore[attr-defined]
    assert context.goal_events.messages == []  # type: ignore[attr-defined]


def test_goal_objective_replaces_complete_goal_without_confirmation(tmp_path: Path):
    ui = ScriptedUIHost()
    context = _context(tmp_path, ui)
    old = context.goal_service.set_goal("spec-1", "complete me")  # type: ignore[attr-defined]
    context.goal_service.update_goal(  # type: ignore[attr-defined]
        "spec-1",
        ThreadGoalStatus.COMPLETE,
        expected_goal_id=old.goal_id,
    )

    outcome = _run(GOAL_COMMAND.run("fresh", context))

    assert outcome.message == "Goal set: fresh"
    assert ui.seen_selects == []
    assert context.goal_service.get_goal("spec-1").objective == "fresh"  # type: ignore[attr-defined]


def test_goal_clear_deletes_current_goal(tmp_path: Path):
    context = _context(tmp_path)
    context.goal_service.set_goal("spec-1", "delete")  # type: ignore[attr-defined]

    outcome = _run(GOAL_COMMAND.run("clear", context))

    assert outcome.message == "Goal cleared: delete"
    assert context.goal_service.get_goal("spec-1") is None  # type: ignore[attr-defined]


def test_goal_set_and_clear_are_durable_transcript_events(tmp_path: Path):
    context = _context(tmp_path)

    _run(GOAL_COMMAND.run("durable condition", context))
    _run(GOAL_COMMAND.run("clear", context))

    notices = [
        message
        for message in context.conversation.messages
        if getattr(message, "subtype", None) in {"goal_set", "goal_cleared"}
    ]
    assert [notice.subtype for notice in notices] == ["goal_set", "goal_cleared"]
    assert notices[0].data["condition"] == "durable condition"
    assert notices[0].data["state"] == "active"
    assert notices[0].data["goalId"]
    assert notices[1].data["state"] == "cleared"
    assert notices[1].data["goalId"] == notices[0].data["goalId"]


@pytest.mark.parametrize("clear_alias", ["clear", "stop", "off", "reset", "none", "cancel"])
def test_goal_clear_aliases_remove_active_goal(tmp_path: Path, clear_alias: str):
    context = _context(tmp_path)
    context.goal_service.set_goal("spec-1", "stop this condition")  # type: ignore[attr-defined]

    outcome = _run(GOAL_COMMAND.run(clear_alias, context))

    assert outcome.message == "Goal cleared: stop this condition"
    assert context.goal_service.get_goal("spec-1") is None  # type: ignore[attr-defined]


def test_goal_clear_without_active_goal_matches_claude_message(tmp_path: Path):
    outcome = _run(GOAL_COMMAND.run("clear", _context(tmp_path)))

    assert outcome.message == "No goal set"


def test_goal_command_updates_app_state_goal_status(tmp_path: Path):
    context = _context(tmp_path)
    context.app_state = AppState(model="test", provider="test")  # type: ignore[attr-defined]

    _run(GOAL_COMMAND.run("sync status", context))

    assert context.app_state.goal_status["status"] == "active"  # type: ignore[attr-defined]
    assert context.app_state.goal_status["objective"] == "sync status"  # type: ignore[attr-defined]

    _run(GOAL_COMMAND.run("clear", context))

    assert context.app_state.goal_status is None  # type: ignore[attr-defined]


def test_bare_goal_summary_updates_app_state_goal_status(tmp_path: Path):
    context = _context(tmp_path)
    context.app_state = AppState(model="test", provider="test")  # type: ignore[attr-defined]
    context.goal_service.set_goal("spec-1", "sync on summary")  # type: ignore[attr-defined]

    outcome = _run(GOAL_COMMAND.run("", context))

    assert "Goal: sync on summary" in outcome.message
    assert "Goal active" in outcome.message
    assert outcome.transient is True
    assert context.app_state.goal_status["status"] == "active"  # type: ignore[attr-defined]
    assert context.app_state.goal_status["objective"] == "sync on summary"  # type: ignore[attr-defined]


def test_bare_goal_summary_accounts_live_idle_elapsed_time(tmp_path: Path):
    from clawcodex_ext.goal.accounting import GoalAccountingState
    from clawcodex_ext.goal.runtime import GoalRuntime

    now = [1_000.0]
    context = _context(tmp_path)
    service = context.goal_service  # type: ignore[attr-defined]
    runtime = GoalRuntime(
        thread_id="spec-1",
        service=service,
        accounting_state=GoalAccountingState(clock=lambda: now[0]),
    )
    service.register_runtime(runtime)
    context.tool_context.goal_service = service
    context.tool_context.goal_runtime = runtime
    service.replace_goal("spec-1", "show current elapsed time")
    now[0] += 7

    outcome = _run(GOAL_COMMAND.run("", context))

    assert "running 7s" in outcome.message
    assert service.get_goal("spec-1").time_used_seconds == 7


def test_bare_goal_shows_achieved_summary(tmp_path: Path):
    context = _context(tmp_path)
    goal = context.goal_service.set_goal("spec-1", "finished condition")  # type: ignore[attr-defined]
    context.goal_service.update_goal(  # type: ignore[attr-defined]
        "spec-1",
        ThreadGoalStatus.COMPLETE,
        expected_goal_id=goal.goal_id,
    )

    outcome = _run(GOAL_COMMAND.run("", context))

    assert "Goal achieved" in outcome.message
    assert "Goal: finished condition" in outcome.message
    assert outcome.transient is True


def test_bare_goal_shows_evaluated_turns_and_last_check(tmp_path: Path):
    from clawcodex_ext.goal.evaluator import GoalEvaluation

    context = _context(tmp_path)
    goal = context.goal_service.set_goal(  # type: ignore[attr-defined]
        "spec-1", "pass tests", completion_mode=GoalCompletionMode.EVALUATOR
    )
    context.goal_service.record_evaluation(  # type: ignore[attr-defined]
        "spec-1",
        GoalEvaluation(met=False, reason="one test remains", usage={}),
        expected_goal_id=goal.goal_id,
        expected_evaluation_count=0,
    )

    outcome = _run(GOAL_COMMAND.run("", context))

    assert "1 turn" in outcome.message
    assert "Last check: one test remains" in outcome.message


def test_bare_goal_hides_turns_and_last_check_before_first_evaluation(tmp_path: Path):
    context = _context(tmp_path)
    context.goal_service.set_goal("spec-1", "pass tests")  # type: ignore[attr-defined]

    outcome = _run(GOAL_COMMAND.run("", context))

    assert " turn" not in outcome.message
    assert "Last check:" not in outcome.message


@pytest.mark.parametrize(
    "condition",
    ["pause", "resume", "edit", "status", "continue", "complete"],
)
def test_non_clear_goal_words_are_conditions(tmp_path: Path, condition: str):
    context = _context(tmp_path)

    outcome = _run(GOAL_COMMAND.run(condition, context))

    assert outcome.display == "system"
    assert outcome.should_query is True
    assert outcome.message == f"Goal set: {condition}"
    assert context.goal_service.get_goal("spec-1").objective == condition  # type: ignore[attr-defined]


def test_engine_reports_goal_disabled_when_gate_is_off():
    registry = CommandRegistry()
    registry.register(GOAL_COMMAND)
    engine = CommandEngine(
        registry=registry,
        workspace_root=Path("/tmp"),
        context=_context(tmp_path=Path("/tmp")),
    )
    get_registry().set_override("goals", False)

    result = _run(engine.execute("/goal"))

    assert result.success is False
    assert "disabled" in result.error.lower()
