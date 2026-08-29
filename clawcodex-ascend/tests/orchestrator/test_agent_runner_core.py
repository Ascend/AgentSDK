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
"""Focused tests for the composed AgentRunner entrypoint."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from contextlib import nullcontext
from types import MethodType, ModuleType, SimpleNamespace

import pytest


def _install_dependency_stubs() -> None:
    modules = {
        "agent_completion": "AgentCompletionMixin",
        "agent_control": "AgentControlMixin",
        "agent_events": "AgentEventMixin",
        "agent_lifecycle": "AgentLifecycleMixin",
        "agent_stream": "AgentStreamMixin",
    }
    for module_name, class_name in modules.items():
        qualified_name = f"extensions.orchestrator.{module_name}"
        if find_spec(qualified_name) is not None:
            continue
        module = ModuleType(qualified_name)
        setattr(module, class_name, type(class_name, (), {}))
        sys.modules[module.__name__] = module

    if find_spec("extensions.orchestrator.agent_session") is None:
        session_module = ModuleType("extensions.orchestrator.agent_session")
        session_module.AgentSession = type("AgentSession", (), {})
        session_module.RetryItem = type("RetryItem", (), {})
        sys.modules[session_module.__name__] = session_module

    if find_spec("extensions.orchestrator.agent_turn") is not None:
        return
    turn_module = ModuleType("extensions.orchestrator.agent_turn")
    turn_module.AgentTurnMixin = type("AgentTurnMixin", (), {})

    class _TurnState:
        def __init__(self) -> None:
            self.output = ""
            self.has_tool_calls = False
            self.cap_reached = False

    turn_module.TurnState = _TurnState
    sys.modules[turn_module.__name__] = turn_module


_install_dependency_stubs()

from extensions.orchestrator.agent_runner import AgentRunner  # noqa: E402


class _Event:
    def __init__(self, **values) -> None:
        self.__dict__.update(values)


@pytest.fixture
def query_module(monkeypatch):
    module = ModuleType("extensions.api.query")
    for name in ("SessionComplete", "TextDelta", "ToolCallEvent", "ToolResultEvent"):
        setattr(module, name, type(name, (_Event,), {}))
    api_module = ModuleType("extensions.api")
    api_module.query = module
    monkeypatch.setitem(sys.modules, "extensions.api", api_module)
    monkeypatch.setitem(sys.modules, "extensions.api.query", module)
    return module


def _runner() -> AgentRunner:
    runner = object.__new__(AgentRunner)
    runner.max_turns = 3
    runner.agent_config = SimpleNamespace(coordinator_mode=False)
    return runner


@pytest.mark.asyncio
async def test_run_enters_requested_coordinator_mode() -> None:
    runner = _runner()
    entered: list[bool] = []

    class _Coordinator:
        @staticmethod
        def enter(enabled: bool):
            entered.append(enabled)
            return nullcontext()

    runner._coordinator = _Coordinator()
    runner._resolve_protocols = lambda: None
    calls: list[object] = []

    async def _run_impl(self, session, workflow, **_kwargs) -> None:
        calls.append((session, workflow))

    runner._run_impl = MethodType(_run_impl, runner)
    session = SimpleNamespace(coordinator_mode=True, run_kind="single")
    workflow = SimpleNamespace(
        agent=SimpleNamespace(
            provider="deepseek",
            model="default-model",
            stage_overrides={},
        )
    )
    await runner.run(session, workflow)
    assert entered == [True]
    assert calls == [(session, workflow)]


@pytest.mark.asyncio
async def test_dispatch_routes_text_delta(query_module) -> None:
    runner = _runner()
    seen: list[object] = []

    async def _handle(self, session, event, turn, run, dashboard):
        seen.append((session, event, turn, run, dashboard))

    runner._handle_text_delta_event = MethodType(_handle, runner)
    session = object()
    event = query_module.TextDelta(content="hello")
    action = await runner._dispatch_stream_event(
        session,
        event,
        "run",
        "turn",
        {},
        tracker=None,
        status_dashboard="dashboard",
        stream_iter=None,
    )
    assert action == "continue"
    assert seen == [(session, event, "turn", "run", "dashboard")]


@pytest.mark.asyncio
async def test_session_continue_becomes_turn_complete(query_module) -> None:
    runner = _runner()

    async def _complete(self, *_args, **_kwargs):
        return "continue"

    runner._handle_session_complete_event = MethodType(_complete, runner)
    action = await runner._dispatch_stream_event(
        object(),
        query_module.SessionComplete(reason="success"),
        object(),
        object(),
        {},
        tracker=object(),
        status_dashboard=None,
        stream_iter=None,
    )
    assert action == "turn_complete"


@pytest.mark.asyncio
async def test_forced_boundary_advances_and_flushes() -> None:
    runner = _runner()
    flushed: list[str] = []
    runner._flush_turn_transcript = lambda _session: flushed.append("buffer")

    class _Storage:
        @staticmethod
        def flush() -> None:
            flushed.append("storage")

    session = SimpleNamespace(turn_count=0, _transcript_storage=_Storage())
    state = SimpleNamespace(turn_number=1)
    await runner._finish_forced_turn_boundary(session, state)
    assert state.turn_number == 2
    assert session.turn_count == 2
    assert flushed == ["buffer", "storage"]


@pytest.mark.asyncio
async def test_budget_exhaustion_sets_terminal_metadata() -> None:
    runner = _runner()
    finalized: list[str] = []

    async def _finalize(self, _session, _state, reason) -> None:
        finalized.append(reason)

    runner._finalize_session = MethodType(_finalize, runner)
    session = SimpleNamespace(
        status="running",
        session_end_reason=None,
        session_end_summary="",
        tool_count=0,
    )
    state = SimpleNamespace(turn_number=3, tool_count=8)
    await runner._finish_budget_exhausted(session, state)
    assert session.status == "max_turns_exceeded"
    assert session.session_end_reason == "budget_exhausted"
    assert session.tool_count == 8
    assert finalized == ["budget_exhausted"]


@pytest.mark.asyncio
async def test_run_impl_always_finalizes_artifacts() -> None:
    runner = _runner()
    session = SimpleNamespace(session_end_reason=None)
    state = SimpleNamespace(turn_number=0, tool_count=0)
    finalized: list[object] = []

    async def _initialize(self, *_args, **_kwargs):
        return state, {}

    runner._initialize_run = MethodType(_initialize, runner)
    runner._build_turn_prompt = lambda *_args: "prompt"

    async def _services(*_args):
        return None

    runner._ensure_turn_services = _services
    runner._append_turn_start_debug = lambda *_args: None

    class _Stream:
        async def __anext__(self):
            return SimpleNamespace()

    runner._build_query_runner = lambda *_args: SimpleNamespace(stream=lambda: _Stream())
    runner._record_stream_event = lambda *_args: None

    async def _dispatch(*_args, **_kwargs):
        return "return"

    runner._dispatch_stream_event = _dispatch
    runner._finalize_run_artifacts = lambda current: finalized.append(current)

    await runner._run_impl(session, object())
    assert finalized == [session]
