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

"""Focused tests for Agent Runtime per-turn preparation."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from extensions.orchestrator.agent_turn import AgentTurnMixin, RunState, TurnState


class _Runner(AgentTurnMixin):
    def __init__(self) -> None:
        self.max_turns = 6
        self.workspace_cfg = SimpleNamespace()
        self.agent_config = SimpleNamespace(
            provider="deepseek",
            model="deepseek-chat",
            permission_mode="never",
            audit_log="full",
            run_timeout_ms=60_000,
            stall_timeout_ms=10_000,
            stall_warn_ms=1_000,
            env={"MODE": "test"},
            max_no_op_turns=0,
            loop_detection_window=1,
            loop_detection_threshold=1,
        )
        self._coordinator = SimpleNamespace(is_active=lambda: True)

    @staticmethod
    def _make_control_drain_fn(_session):
        return lambda: None

    @staticmethod
    def _make_pause_wait_fn(_session):
        async def _wait() -> None:
            return None

        return _wait


def _session(*, prompt_override=None):
    return SimpleNamespace(
        issue=SimpleNamespace(
            id="I-137",
            identifier="A2",
            python_executable="",
        ),
        workspace=SimpleNamespace(path="workspace"),
        prompt_override=prompt_override,
        previous_run_ids=[],
        conflict_files=[],
        run_id="run-1",
        debug_log_path="debug.ndjson",
        _runtime_tasks=None,
        _pause_gate=object(),
    )


def _install_prompt_builder(monkeypatch) -> type:
    module = ModuleType("extensions.orchestrator.prompt_builder")

    class _PromptBuilder:
        @staticmethod
        def render_parts(_issue, **kwargs):
            assert kwargs["python_executable"] == "python"
            return "system background", "issue prompt"

        @staticmethod
        def build_continuation_prompt(**kwargs):
            return f"continuation-{kwargs['turn_number']}"

        @staticmethod
        def build_clarification_context(**kwargs):
            return f"clarify:{kwargs.get('clarification_answer') or kwargs.get('options')}"

    module.PromptBuilder = _PromptBuilder
    module.resolve_python_executable = lambda **_kwargs: "python"
    monkeypatch.setitem(sys.modules, "extensions.orchestrator.prompt_builder", module)
    return _PromptBuilder


def test_run_state_clamps_guard_thresholds() -> None:
    state = _Runner()._create_run_state()
    assert state.max_no_op_turns == 3
    assert state.loop_window == 2
    assert state.loop_threshold == 2


def test_run_state_updates_session_and_callback() -> None:
    seen: list[int] = []
    state = RunState(tool_count=4, diagnostics_callback=lambda session: seen.append(session.tool_count))
    session = SimpleNamespace(issue=SimpleNamespace(id="I-137"), tool_count=0)
    state.update_diagnostics(session)
    assert session.tool_count == 4
    assert seen == [4]


def test_turn_state_uses_independent_mutable_defaults() -> None:
    first = TurnState()
    second = TurnState()
    first.tool_names.append("Read")
    assert second.tool_names == []


def test_prompt_override_becomes_issue_context(monkeypatch) -> None:
    _install_prompt_builder(monkeypatch)
    session = _session(prompt_override="custom stage prompt")
    prompt = _Runner()._build_turn_prompt(session, 0)
    assert prompt == "custom stage prompt"
    assert session._issue_context == prompt
    assert session._system_prompt_append == ""


def test_initial_prompt_is_split_into_system_and_user(monkeypatch) -> None:
    _install_prompt_builder(monkeypatch)
    session = _session()
    prompt = _Runner()._build_turn_prompt(session, 0)
    assert prompt == "issue prompt"
    assert session._system_prompt_append == "system background"
    assert session._issue_context == "issue prompt"


def test_continuation_prompt_uses_current_turn(monkeypatch) -> None:
    _install_prompt_builder(monkeypatch)
    session = _session()
    session._issue_context = "issue prompt"
    assert _Runner()._build_turn_prompt(session, 3) == "continuation-3"


def test_session_clarification_answer_has_priority(monkeypatch) -> None:
    _install_prompt_builder(monkeypatch)
    session = _session()
    session.clarification_answer = "use option B"
    session.clarification_question = "which option?"
    session.clarification_source = "comment"
    context, question, options = _Runner._clarification_context(session, None)
    assert context == "clarify:use option B"
    assert question == "which option?"
    assert options is None


def test_query_runner_receives_runtime_callbacks(monkeypatch) -> None:
    query_module = ModuleType("extensions.api.query")

    class _QueryConfig:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)

    class _QueryRunner:
        def __init__(self, config) -> None:
            self.config = config

    query_module.QueryConfig = _QueryConfig
    query_module.QueryRunner = _QueryRunner
    api_module = ModuleType("extensions.api")
    api_module.query = query_module
    monkeypatch.setitem(sys.modules, "extensions.api", api_module)
    monkeypatch.setitem(sys.modules, "extensions.api.query", query_module)

    result = _Runner()._build_query_runner(_session(), "do the work")
    assert result.config.prompt == "do the work"
    assert result.config.model == "deepseek-chat"
    assert result.config.env["CLAUDE_CODE_COORDINATOR_MODE"] == "1"
    assert callable(result.config.control_drain_fn)
    assert callable(result.config.pause_wait_fn)
