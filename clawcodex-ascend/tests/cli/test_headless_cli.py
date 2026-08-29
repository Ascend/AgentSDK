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

"""Integration tests for the headless (``--print``) CLI path.

These tests bypass the real provider and tool registry by monkey-patching the
wiring inside ``src.entrypoints.headless`` so we can exercise the stdout
contract without any network IO.
"""

from __future__ import annotations

import io
import json

import pytest
from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.query.agent_loop_compat import AgentLoopRunResult
from clawcodex_ext.query.transitions import Terminal
from clawcodex_ext.tool_system.renderers import AgentLoopResult
from clawcodex_ext.utils.resume_hint import reset_resume_hint_for_test_only
from src.entrypoints import HeadlessOptions, run_headless


class _FakeProvider:
    """Minimal stand-in for an LLM provider.

    ``responses`` is a list of ``ChatResponse`` to return in order. If tool
    calls are requested, they must match the shape
    ``{"id": str, "name": str, "input": dict}``.
    """

    def __init__(self, api_key: str, base_url=None, model=None, *, responses=None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or "fake-model"
        self._responses = list(responses or [])

    def chat(self, messages, tools=None, **kwargs):
        if not self._responses:
            raise AssertionError("FakeProvider ran out of scripted responses")
        return self._responses.pop(0)

    async def chat_async(self, messages, tools=None, **kwargs):
        return self.chat(messages, tools=tools, **kwargs)

    def chat_stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError


class _FakeRegistry:
    def list_tools(self):
        return []


@pytest.fixture
def fake_wiring(monkeypatch):
    """Patch provider/tool wiring with fakes that require no API key."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    scripted_responses: list[ChatResponse] = []

    def _fake_get_provider_class(_name):
        def _factory(api_key, base_url=None, model=None, **_kwargs):
            return _FakeProvider(api_key, base_url=base_url, model=model, responses=list(scripted_responses))

        return _factory

    monkeypatch.setattr(
        ext_headless,
        "get_provider_class",
        _fake_get_provider_class,
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "get_provider_config",
        lambda _name: {
            "api_key": "test-key",
            "base_url": None,
            "default_model": "fake-model",
        },
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "get_default_provider",
        lambda: "anthropic",
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "build_default_registry",
        lambda provider=None: _FakeRegistry(),
        raising=False,
    )

    return scripted_responses


def _text_response(text: str) -> ChatResponse:
    return ChatResponse(
        content=text,
        model="fake-model",
        usage={"input_tokens": 5, "output_tokens": len(text.split())},
        finish_reason="end_turn",
        tool_uses=None,
    )


# ---------------------------------------------------------------------------
# text output


def test_headless_text_output_prints_assistant_reply(fake_wiring, tmp_path):
    fake_wiring.append(_text_response("Hello, human!"))

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="hi",
            output_format="text",
            stdout=stdout,
            stderr=stderr,
            workspace_root=tmp_path,
        )
    )

    assert code == 0
    assert stdout.getvalue().strip() == "Hello, human!"


def test_headless_text_reads_prompt_from_stdin_when_dash(fake_wiring, tmp_path):
    fake_wiring.append(_text_response("from-stdin"))

    code = run_headless(
        HeadlessOptions(
            prompt="-",
            output_format="text",
            stdin=io.StringIO("piped prompt"),
            stdout=(out := io.StringIO()),
            stderr=io.StringIO(),
            workspace_root=tmp_path,
        )
    )

    assert code == 0
    assert "from-stdin" in out.getvalue()


def test_headless_bundled_skill_slash_expands_before_agent_loop(
    fake_wiring,
    tmp_path,
    monkeypatch,
):
    """Prompt skills use the workspace catalogue and the bound ToolContext."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    monkeypatch.delenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SIMPLE", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_REMOTE", raising=False)
    captured: dict[str, object] = {}

    async def _capture_expanded_prompt(*args, **kwargs):
        del args
        messages = kwargs["initial_messages"]
        captured["content"] = messages[-1].content
        captured["tool_context"] = kwargs["tool_context"]
        return AgentLoopResult(
            response_text="remember reviewed",
            usage={"input_tokens": 1, "output_tokens": 1},
            num_turns=1,
        )

    monkeypatch.setattr(
        ext_headless,
        "run_query_as_agent_loop",
        _capture_expanded_prompt,
    )

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="/remember focus on testing preferences",
            output_format="text",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code == 0
    assert stdout.getvalue().strip() == "remember reviewed"
    assert captured["tool_context"] is not None
    content = captured["content"]
    assert isinstance(content, str)
    assert "# Memory Review" in content
    assert "## Additional context from user" in content
    assert "focus on testing preferences" in content
    assert not content.startswith("/remember")


def test_headless_goal_summary_runs_without_provider_config(monkeypatch, tmp_path):
    """A provider is only needed once a slash command invokes the model."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    def _provider_must_not_be_touched(*_args, **_kwargs):
        raise AssertionError("provider wiring should not be touched for /goal summary")

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "clawcodex-home"))
    monkeypatch.setattr(
        ext_headless,
        "get_provider_config",
        _provider_must_not_be_touched,
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "get_provider_class",
        _provider_must_not_be_touched,
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "provider_requires_api_key",
        _provider_must_not_be_touched,
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "build_default_registry",
        _provider_must_not_be_touched,
        raising=False,
    )

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="/goal",
            output_format="text",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code == 0
    rendered = stdout.getvalue()
    assert "/goal <condition> to set one" in rendered
    assert "No goal set" in rendered


@pytest.mark.parametrize("output_format", ["json", "stream-json"])
def test_headless_goal_summary_structured_output_skips_provider(
    monkeypatch,
    tmp_path,
    output_format,
):
    import clawcodex_ext.entrypoints.headless as ext_headless

    def _provider_must_not_be_touched(*_args, **_kwargs):
        raise AssertionError("provider wiring should not be touched for /goal summary")

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "clawcodex-home"))
    monkeypatch.setattr(
        ext_headless,
        "get_provider_config",
        _provider_must_not_be_touched,
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "get_provider_class",
        _provider_must_not_be_touched,
        raising=False,
    )

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="/goal",
            output_format=output_format,
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code == 0
    payloads = [json.loads(line) for line in stdout.getvalue().splitlines()]
    result = payloads[-1]
    assert result["type"] == "result"
    assert result["subtype"] == "success"
    assert result["num_turns"] == 0
    assert "No goal set" in result["result"]
    if output_format == "stream-json":
        assert [payload["type"] for payload in payloads] == [
            "system",
            "assistant",
            "result",
        ]


@pytest.mark.parametrize("output_format", ["text", "json", "stream-json"])
def test_headless_goal_clear_runs_without_provider_config(
    monkeypatch,
    tmp_path,
    output_format,
):
    """Clearing a local goal must not initialize a chat provider."""
    import clawcodex_ext.entrypoints.headless as ext_headless
    from clawcodex_ext.goal.service import GoalService
    from clawcodex_ext.goal.store import GoalStore
    from src.agent import Conversation

    goal_home = tmp_path / "clawcodex-home"
    session_id = "provider-free-clear-goal"
    monkeypatch.setenv("CLAWCODEX_HOME", str(goal_home))
    service = GoalService(store=GoalStore(goal_home / "goals_1.sqlite"))
    service.replace_goal(session_id, "clear without provider")
    service.store.close()

    class LocalSession:
        provider = "anthropic"
        model = "fake-model"
        conversation = Conversation()

        def __init__(self):
            self.session_id = session_id

    def _provider_must_not_be_touched(*_args, **_kwargs):
        raise AssertionError("provider wiring should not be touched for /goal clear")

    monkeypatch.setattr(
        ext_headless,
        "get_provider_config",
        _provider_must_not_be_touched,
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "get_provider_class",
        _provider_must_not_be_touched,
        raising=False,
    )

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="/goal cancel",
            output_format=output_format,
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            external_session=LocalSession(),
            persist_on_exit=False,
        )
    )

    assert code == 0
    if output_format == "text":
        rendered_result = stdout.getvalue()
    else:
        payloads = [json.loads(line) for line in stdout.getvalue().splitlines()]
        rendered_result = payloads[-1]["result"]
    assert "Goal cleared: clear without provider" in rendered_result
    check = GoalService(store=GoalStore(goal_home / "goals_1.sqlite"))
    assert check.get_goal(session_id) is None
    check.store.close()


def test_headless_goal_summary_resume_uses_target_and_resets_metrics(
    monkeypatch,
    tmp_path,
):
    """Provider-free status uses the resumed session and its fresh baseline."""
    import clawcodex_ext.entrypoints.headless as ext_headless
    from clawcodex_ext.goal.evaluator import GoalEvaluation
    from clawcodex_ext.goal.service import GoalService
    from clawcodex_ext.goal.store import GoalStore
    from src.agent import Conversation

    session_id = "provider-free-resume-goal"
    goal_home = tmp_path / "goal-home"
    monkeypatch.setenv("CLAWCODEX_HOME", str(goal_home))
    service = GoalService(store=GoalStore(goal_home / "goals_1.sqlite"))
    active = service.replace_goal(session_id, "resume condition")
    service.account_usage(
        session_id,
        expected_goal_id=active.goal_id,
        token_delta=17,
        elapsed_seconds=6,
    )
    service.record_evaluation(
        session_id,
        GoalEvaluation(met=False, reason="old evaluation", usage={}),
        expected_goal_id=active.goal_id,
        expected_evaluation_count=0,
    )
    service.store.close()

    class ResumedSession:
        provider = "anthropic"
        model = "fake-model"
        conversation = Conversation()

        def __init__(self, resumed_session_id: str):
            self.session_id = resumed_session_id

        def save(self):
            return None

    resume_calls: list[str] = []

    def _resume(_cls, resumed_session_id: str):
        resume_calls.append(resumed_session_id)
        return ResumedSession(resumed_session_id)

    def _provider_must_not_be_touched(*_args, **_kwargs):
        raise AssertionError("provider wiring should not be touched for /goal summary")

    monkeypatch.setattr(ext_headless.Session, "resume", classmethod(_resume))
    monkeypatch.setattr(
        ext_headless,
        "get_provider_config",
        _provider_must_not_be_touched,
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "get_provider_class",
        _provider_must_not_be_touched,
        raising=False,
    )
    monkeypatch.setattr(
        ext_headless,
        "provider_requires_api_key",
        _provider_must_not_be_touched,
        raising=False,
    )

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="/goal",
            output_format="text",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            resume_session_id=session_id,
            persist_on_exit=False,
        )
    )

    restored_service = GoalService(store=GoalStore(goal_home / "goals_1.sqlite"))
    restored = restored_service.get_goal(session_id)
    restored_service.store.close()
    assert code == 0
    assert resume_calls == [session_id]
    assert restored is not None
    assert restored.objective == "resume condition"
    assert restored.tokens_used == 0
    assert restored.time_used_seconds == 0
    assert restored.evaluation_count == 0
    assert restored.last_evaluation_reason is None
    assert "◎ Goal active" in stdout.getvalue()
    assert "Goal: resume condition" in stdout.getvalue()
    assert "0 tokens" in stdout.getvalue()
    assert "Last check:" not in stdout.getvalue()


def test_headless_goal_too_long_returns_nonzero_without_model_call(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """A typed /goal validation error is a command failure, not a prompt."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "goal-home"))
    model_calls: list[str] = []

    async def _capture_model_call(*args, **kwargs):
        del args
        model_calls.append(kwargs["initial_messages"][-1].content)
        return AgentLoopResult(response_text="unexpected", usage={}, num_turns=1)

    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _capture_model_call)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt=f"/goal {'x' * 4001}",
            output_format="text",
            stdout=stdout,
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code != 0
    assert model_calls == []
    assert "4,000 characters or fewer" in stderr.getvalue()
    assert "unexpected" not in stdout.getvalue()


def test_headless_goal_engine_exception_never_reaches_model(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """Unexpected command-engine failures cannot leak literal /goal text."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "goal-home"))
    model_calls: list[str] = []

    async def _raise_goal_error(*_args, **_kwargs):
        raise RuntimeError("goal engine exploded")

    async def _capture_model_call(*args, **kwargs):
        del args
        model_calls.append(kwargs["initial_messages"][-1].content)
        return AgentLoopResult(response_text="unexpected", usage={}, num_turns=1)

    monkeypatch.setattr(ext_headless.CommandEngine, "execute", _raise_goal_error)
    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _capture_model_call)
    stderr = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="/goal never send this literal command",
            output_format="text",
            stdout=io.StringIO(),
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code != 0
    assert model_calls == []
    assert "goal engine exploded" in stderr.getvalue()


def test_headless_goal_condition_starts_agent_with_condition(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """Print mode executes /goal asynchronously and uses its condition as directive."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "goal-home"))
    captured: dict[str, object] = {}

    async def _capture_goal_run(*args, **kwargs):
        del args
        captured["messages"] = kwargs["initial_messages"]
        captured["tool_context"] = kwargs["tool_context"]
        captured["max_turns"] = kwargs["max_turns"]
        return AgentLoopResult(
            response_text="goal run complete",
            usage={"input_tokens": 1, "output_tokens": 1},
            num_turns=1,
        )

    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _capture_goal_run)
    stdout = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="/goal produce deterministic proof",
            output_format="text",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code == 0
    assert "Goal set: produce deterministic proof" in stdout.getvalue()
    assert "goal run complete" in stdout.getvalue()
    assert captured["max_turns"] == 0
    messages = captured["messages"]
    assert messages[-1].content == "produce deterministic proof"
    goal = captured["tool_context"].goal_service.get_goal(captured["tool_context"].session_id)
    assert goal is not None
    assert goal.objective == "produce deterministic proof"


def test_headless_goal_explicit_max_turns_is_a_failure(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """An explicit safety cap must not turn an unmet goal into success."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "goal-home"))
    captured: dict[str, object] = {}

    async def _hit_explicit_cap(*args, **kwargs):
        del args
        captured["max_turns"] = kwargs["max_turns"]
        return AgentLoopRunResult(
            response_text="[Max tool turns reached]",
            usage={"input_tokens": 7, "output_tokens": 2},
            num_turns=2,
            terminal=Terminal(reason="max_turns"),
        )

    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _hit_explicit_cap)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="/goal finish before the explicit cap",
            output_format="json",
            max_turns=2,
            max_turns_explicit=True,
            stdout=stdout,
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    payload = json.loads(stdout.getvalue())
    assert code == 1
    assert captured["max_turns"] == 2
    assert payload["subtype"] == "error"
    assert payload["num_turns"] == 2
    assert payload["usage"] == {"input_tokens": 7, "output_tokens": 2}
    assert "Goal not achieved before --max-turns=2" in stderr.getvalue()


def test_headless_goal_explicit_max_turns_stops_real_evaluator_loop(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """The real text/evaluator loop reaches the headless nonzero cap path."""
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "goal-home"))
    fake_wiring.extend(
        [
            _text_response("The condition is not met yet."),
            _text_response('{"met": false, "reason": "More work is required."}'),
            _text_response("The condition remains unmet."),
            _text_response('{"met": false, "reason": "The explicit cap has been reached."}'),
        ]
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="/goal finish before the explicit cap",
            output_format="json",
            max_turns=2,
            max_turns_explicit=True,
            stdout=stdout,
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    payload = json.loads(stdout.getvalue())
    assert code == 1
    assert payload["subtype"] == "error"
    assert payload["is_error"] is True
    assert payload["num_turns"] == 2
    assert "Goal not achieved before --max-turns=2" in stderr.getvalue()


def test_headless_goal_allows_literal_max_turns_text_when_evaluator_succeeds(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """Model text resembling the legacy sentinel is not a terminal reason."""
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "goal-home"))
    fake_wiring.extend(
        [
            _text_response("[Max tool turns reached]"),
            _text_response('{"met": true, "reason": "The condition is satisfied."}'),
        ]
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="/goal return the requested literal text",
            output_format="json",
            max_turns=1,
            max_turns_explicit=True,
            stdout=stdout,
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["subtype"] == "success"
    assert payload["result"].endswith("[Max tool turns reached]")
    assert payload["num_turns"] == 1
    assert stderr.getvalue() == ""


def test_headless_goal_uses_independent_evaluator_until_condition_is_met(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """The real print-mode loop continues after an unmet evaluator decision."""
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "goal-home"))
    fake_wiring.extend(
        [
            _text_response("The deterministic proof is not complete yet."),
            _text_response('{"met": false, "reason": "The required proof is still missing."}'),
            _text_response("The deterministic proof is now complete."),
            _text_response('{"met": true, "reason": "The required proof is present."}'),
        ]
    )
    stdout = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="/goal produce deterministic proof",
            output_format="text",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    assert code == 0
    assert "Goal set: produce deterministic proof" in stdout.getvalue()
    assert "The deterministic proof is now complete." in stdout.getvalue()


def test_headless_goal_evaluator_error_keeps_aggregate_usage(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "goal-home"))
    fake_wiring.extend(
        [
            ChatResponse(
                content="Main goal turn finished.",
                model="fake-model",
                usage={"input_tokens": 2, "output_tokens": 2},
                finish_reason="end_turn",
                tool_uses=None,
            ),
            ChatResponse(
                content="not json",
                model="fake-evaluator",
                usage={"input_tokens": 3, "output_tokens": 1},
                finish_reason="end_turn",
                tool_uses=None,
            ),
        ]
    )
    stdout = io.StringIO()

    code = run_headless(
        HeadlessOptions(
            prompt="/goal keep active after evaluator failure",
            output_format="json",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )

    payload = json.loads(stdout.getvalue())
    assert code == 1
    assert payload["subtype"] == "error"
    assert payload["num_turns"] == 1
    assert payload["usage"] == {"input_tokens": 5, "output_tokens": 3}


def test_headless_persists_complete_goal_lifecycle_notices(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """Headless transcripts retain structured goal notices for resume/replay."""
    import clawcodex_ext.entrypoints.headless as ext_headless
    from clawcodex_ext.types.messages import SystemMessage
    from src.agent import Conversation

    session = type(
        "HeadlessSession",
        (),
        {
            "session_id": "headless-goal-transcript",
            "provider": "anthropic",
            "model": "fake-model",
            "conversation": Conversation(),
        },
    )()
    notices = [
        SystemMessage(
            content="Goal check: not achieved",
            subtype="goal_evaluation",
            level="info",
            data={"goalId": "goal-1", "state": "active", "met": False},
            usage={"input_tokens": 3, "output_tokens": 1},
        ),
        SystemMessage(
            content="Goal achieved",
            subtype="goal_achieved",
            level="info",
            data={"goalId": "goal-1", "state": "achieved", "met": True},
            usage={"input_tokens": 2, "output_tokens": 1},
        ),
        SystemMessage(
            content="Goal evaluator failed: invalid JSON",
            subtype="goal_evaluator_error",
            level="warning",
            data={"goalId": "goal-2", "state": "active", "met": None},
            usage={"input_tokens": 4, "output_tokens": 2},
        ),
    ]

    async def _emit_goal_notices(*args, **kwargs):
        del args
        for notice in notices:
            kwargs["on_message"](notice)
        return AgentLoopResult(
            response_text="goal lifecycle recorded",
            usage={"input_tokens": 9, "output_tokens": 4},
            num_turns=1,
        )

    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _emit_goal_notices)

    code = run_headless(
        HeadlessOptions(
            prompt="continue",
            output_format="text",
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            external_session=session,
            persist_on_exit=False,
        )
    )

    assert code == 0
    persisted = [
        message
        for message in session.conversation.messages
        if getattr(message, "subtype", None)
        in {
            "goal_evaluation",
            "goal_achieved",
            "goal_evaluator_error",
        }
    ]
    assert persisted == notices
    for actual, expected in zip(persisted, notices, strict=True):
        assert actual is expected
        assert actual.data == expected.data
        assert actual.usage == expected.usage


def test_headless_goal_transcript_round_trip_preserves_order_and_chain(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """Disk transcript and Session.load retain one ordered goal lifecycle."""
    from clawcodex_ext.agent.session import Session

    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("CLAWCODEX_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "clawcodex-home"))
    session = Session(
        session_id="headless-goal-round-trip",
        provider="anthropic",
        model="fake-model",
    )
    fake_wiring.extend(
        [
            ChatResponse(
                content="The deterministic proof is complete.",
                model="fake-model",
                usage={"input_tokens": 2, "output_tokens": 1},
                finish_reason="end_turn",
                tool_uses=None,
            ),
            ChatResponse(
                content='{"met": true, "reason": "The proof is present."}',
                model="fake-evaluator",
                usage={"input_tokens": 3, "output_tokens": 1},
                finish_reason="end_turn",
                tool_uses=None,
            ),
        ]
    )

    code = run_headless(
        HeadlessOptions(
            prompt="/goal produce deterministic proof",
            output_format="text",
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            external_session=session,
        )
    )

    assert code == 0
    transcript_path = sessions_dir / session.session_id / "transcript.jsonl"
    entries = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines()]
    disk_messages = [entry for entry in entries if entry.get("uuid")]

    def _lifecycle_marker(message):
        return message.get("subtype") or message.get("role")

    assert [_lifecycle_marker(message) for message in disk_messages] == [
        "goal_set",
        "user",
        "assistant",
        "goal_achieved",
    ]
    assert len({message["uuid"] for message in disk_messages}) == 4
    expected_parent = None
    for message in disk_messages:
        assert "parentUuid" in message
        assert message["parentUuid"] == expected_parent
        expected_parent = message["uuid"]

    loaded = Session.load(session.session_id)
    assert loaded is not None
    loaded_messages = loaded.conversation.messages
    assert [getattr(message, "subtype", None) or message.role for message in loaded_messages] == [
        "goal_set",
        "user",
        "assistant",
        "goal_achieved",
    ]
    assert [message.uuid for message in loaded_messages] == [message["uuid"] for message in disk_messages]
    achieved = loaded_messages[-1]
    assert achieved.data["state"] == "achieved"
    assert achieved.data["met"] is True
    assert achieved.usage == {"input_tokens": 3, "output_tokens": 1}


def test_headless_resume_resets_active_goal_metrics_before_agent_run(
    fake_wiring,
    monkeypatch,
    tmp_path,
):
    """A real --resume preserves the condition but restarts its live counters."""
    import clawcodex_ext.entrypoints.headless as ext_headless
    from clawcodex_ext.goal.service import GoalService
    from clawcodex_ext.goal.store import GoalStore
    from src.agent import Conversation

    goal_home = tmp_path / "goal-home"
    monkeypatch.setenv("CLAWCODEX_HOME", str(goal_home))
    service = GoalService(store=GoalStore(goal_home / "goals_1.sqlite"))
    active = service.replace_goal("resume-goal-session", "resume condition")
    service.account_usage(
        "resume-goal-session",
        expected_goal_id=active.goal_id,
        token_delta=17,
        elapsed_seconds=6,
    )
    service.store.close()

    class ResumedSession:
        session_id = "resume-goal-session"
        provider = "anthropic"
        model = "fake-model"
        conversation = Conversation()

        def save(self):
            return None

    captured: dict[str, object] = {}

    async def _capture_resumed_goal(*args, **kwargs):
        del args
        context = kwargs["tool_context"]
        captured["goal"] = context.goal_service.get_goal(context.session_id)
        return AgentLoopResult(
            response_text="resumed",
            usage={"input_tokens": 1, "output_tokens": 1},
            num_turns=1,
        )

    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _capture_resumed_goal)
    code = run_headless(
        HeadlessOptions(
            prompt="continue",
            output_format="text",
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            resume_session_id="resume-goal-session",
            external_session=ResumedSession(),
            persist_on_exit=False,
        )
    )

    restored = captured["goal"]
    assert code == 0
    assert restored.objective == "resume condition"
    assert restored.tokens_used == 0
    assert restored.time_used_seconds == 0


# ---------------------------------------------------------------------------
# json output


def test_headless_json_output_emits_single_object(fake_wiring, tmp_path):
    fake_wiring.append(_text_response("json reply"))

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="hi",
            output_format="json",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
        )
    )

    assert code == 0
    payload = json.loads(stdout.getvalue().strip())
    assert payload["type"] == "result"
    assert payload["subtype"] == "success"
    assert payload["result"] == "json reply"
    assert payload["provider"] == "anthropic"
    assert payload["num_turns"] == 1
    assert payload["usage"]["input_tokens"] == 5


def test_headless_json_groups_physical_turns_under_goal_operation_id(
    fake_wiring,
    tmp_path,
):
    import clawcodex_ext.entrypoints.headless as ext_headless

    captured: dict = {}
    original = ext_headless.run_query_as_agent_loop

    async def _logical_goal_run(*args, **kwargs):
        captured["tool_context"] = kwargs["tool_context"]
        return AgentLoopResult(
            response_text="logical goal result",
            usage={"input_tokens": 9, "output_tokens": 4},
            num_turns=3,
        )

    ext_headless.run_query_as_agent_loop = _logical_goal_run  # type: ignore[assignment]
    try:
        stdout = io.StringIO()
        code = run_headless(
            HeadlessOptions(
                prompt="run a goal",
                output_format="json",
                stdout=stdout,
                stderr=io.StringIO(),
                workspace_root=tmp_path,
            )
        )
    finally:
        ext_headless.run_query_as_agent_loop = original  # type: ignore[assignment]

    payload = json.loads(stdout.getvalue().strip())
    assert code == 0
    assert payload["result"] == "logical goal result"
    assert payload["num_turns"] == 3
    assert payload["goal_operation_id"] == payload["session_id"]
    assert captured["tool_context"].session_id == payload["session_id"]


# ---------------------------------------------------------------------------
# stream-json output


def test_headless_stream_json_emits_system_assistant_result(fake_wiring, tmp_path):
    fake_wiring.append(_text_response("stream reply"))

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="hi",
            output_format="stream-json",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
        )
    )

    assert code == 0
    lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    parsed = [json.loads(line) for line in lines]
    types = [ev["type"] for ev in parsed]
    assert types[0] == "system"
    assert "assistant" in types
    assert types[-1] == "result"
    assistant = next(ev for ev in parsed if ev["type"] == "assistant")
    assert assistant["text"] == "stream reply"
    result = parsed[-1]
    assert result["result"] == "stream reply"
    assert result["num_turns"] == 1
    assert result["subtype"] == "success"
    assert result["goal_operation_id"] == result["session_id"]


def test_headless_stream_json_input_requires_matching_output(fake_wiring, tmp_path):
    stderr = io.StringIO()
    with pytest.raises(SystemExit) as excinfo:
        run_headless(
            HeadlessOptions(
                prompt="hi",
                input_format="stream-json",
                output_format="text",
                stdout=io.StringIO(),
                stderr=stderr,
                workspace_root=tmp_path,
            )
        )
    assert excinfo.value.code == 2


def test_headless_stream_json_multi_turn_from_stdin(fake_wiring, tmp_path):
    fake_wiring.append(_text_response("A"))
    fake_wiring.append(_text_response("B"))

    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "one"}}),
                json.dumps({"type": "user", "message": {"content": "two"}}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            output_format="stream-json",
            input_format="stream-json",
            stdin=stdin,
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
        )
    )

    assert code == 0
    parsed = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assistants = [ev for ev in parsed if ev["type"] == "assistant"]
    assert [ev["text"] for ev in assistants] == ["A", "B"]
    result = parsed[-1]
    assert result["num_turns"] == 2
    assert "A" in result["result"] and "B" in result["result"]


# ---------------------------------------------------------------------------
# permission handling in headless mode


def test_headless_coordinator_injects_role_and_worker_context(fake_wiring, tmp_path):
    fake_wiring.append(_text_response("ok"))
    import clawcodex_ext.entrypoints.headless as ext_headless
    from clawcodex_ext.coordinator.mode import (
        coordinator_mode_context,
        get_coordinator_user_context,
    )
    from clawcodex_ext.coordinator.prompt import get_coordinator_system_prompt

    captured: dict = {}
    original = ext_headless.run_query_as_agent_loop

    async def _capture(*args, **kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        return await original(*args, **kwargs)

    ext_headless.run_query_as_agent_loop = _capture  # type: ignore[assignment]
    try:
        with coordinator_mode_context(True):
            worker_context = get_coordinator_user_context()["workerToolsContext"]
            code = run_headless(
                HeadlessOptions(
                    prompt="delegate this task",
                    append_system_prompt="CALLER CONSTRAINT",
                    output_format="text",
                    stdout=io.StringIO(),
                    stderr=io.StringIO(),
                    workspace_root=tmp_path,
                )
            )
    finally:
        ext_headless.run_query_as_agent_loop = original  # type: ignore[assignment]

    assert code == 0
    prompt = captured["system_prompt"]
    assert prompt.startswith(get_coordinator_system_prompt())
    assert worker_context in prompt
    assert prompt.endswith("CALLER CONSTRAINT")


def test_headless_without_skip_permissions_installs_auto_deny_handler(fake_wiring, tmp_path):
    fake_wiring.append(_text_response("ok"))
    import clawcodex_ext.entrypoints.headless as ext_headless

    captured: dict = {}
    original = ext_headless.run_query_as_agent_loop

    async def _capture(*args, **kwargs):
        captured["tool_context"] = kwargs["tool_context"]
        return await original(*args, **kwargs)

    ext_headless.run_query_as_agent_loop = _capture  # type: ignore[assignment]
    try:
        code = run_headless(
            HeadlessOptions(
                prompt="hi",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
            )
        )
    finally:
        ext_headless.run_query_as_agent_loop = original  # type: ignore[assignment]
    assert code == 0
    ctx = captured["tool_context"]
    assert ctx.options.is_non_interactive_session is True
    # Non-interactive mode installs an auto-deny handler.
    from src.permissions.types import PermissionAskRequest

    reply = ctx.permission_handler(PermissionAskRequest(tool_name="Bash", message="needs approval"))
    assert reply.behavior == "deny"


def test_headless_with_skip_permissions_clears_handler(fake_wiring, tmp_path):
    fake_wiring.append(_text_response("ok"))
    import clawcodex_ext.entrypoints.headless as ext_headless

    captured: dict = {}
    original = ext_headless.run_query_as_agent_loop

    async def _capture(*args, **kwargs):
        captured["tool_context"] = kwargs["tool_context"]
        return await original(*args, **kwargs)

    ext_headless.run_query_as_agent_loop = _capture  # type: ignore[assignment]
    try:
        run_headless(
            HeadlessOptions(
                prompt="hi",
                output_format="text",
                skip_permissions=True,
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
            )
        )
    finally:
        ext_headless.run_query_as_agent_loop = original  # type: ignore[assignment]

    ctx = captured["tool_context"]
    assert ctx.permission_handler is None
    assert ctx.allow_docs is True
    assert ctx.options.is_non_interactive_session is True


# ---------------------------------------------------------------------------
# flag validation


def test_headless_invalid_output_format_exits_2(fake_wiring, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        run_headless(
            HeadlessOptions(
                prompt="hi",
                output_format="bogus",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
            )
        )
    assert excinfo.value.code == 2


def test_headless_empty_prompt_exits_2(fake_wiring, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        run_headless(
            HeadlessOptions(
                prompt="",
                output_format="text",
                stdin=io.StringIO(""),
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
            )
        )
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# S-R1: resume hint on text / json / stream-json output
#
# The headless entrypoint should print the standard
# ``Resume this session with: clawcodex --resume <sid>`` line to a TTY in
# text mode. JSON and stream-json consumers already receive the session id
# in their structured payload, so the hint must not pollute those streams.


class _FakeTTYStdout:
    """Minimal stdout that pretends to be a TTY.

    The resume-hint helper gates on ``stream.isatty()``; this stand-in
    lets the text-mode test exercise the gate being open.
    """

    def __init__(self) -> None:
        self._buf = io.StringIO()

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        return self._buf.write(s)

    def flush(self) -> None:
        self._buf.flush()

    def getvalue(self) -> str:
        return self._buf.getvalue()


def _wire_tty_wiring(monkeypatch, scripted_responses):
    """Self-contained fixture for the S-R1 headless tests.

    Patches the *real* ext module (not the src.entrypoints proxy) so
    ``run_headless`` can run with no API key, no real provider, and no
    real tool registry.
    """
    import clawcodex_ext.entrypoints.headless as ext_headless

    def _fake_get_provider_class(_name):
        def _factory(api_key, base_url=None, model=None, **_kwargs):
            return _FakeProvider(api_key, model=model, responses=list(scripted_responses))

        return _factory

    def _fake_get_provider_config(_name):
        return {
            "api_key": "test-key",
            "base_url": None,
            "default_model": "fake-model",
        }

    monkeypatch.setattr(ext_headless, "get_provider_class", _fake_get_provider_class, raising=False)
    monkeypatch.setattr(ext_headless, "get_provider_config", _fake_get_provider_config, raising=False)
    monkeypatch.setattr(ext_headless, "get_default_provider", lambda: "anthropic", raising=False)
    monkeypatch.setattr(
        ext_headless,
        "build_default_registry",
        lambda provider=None: _FakeRegistry(),
        raising=False,
    )


@pytest.fixture
def tty_fake_wiring(monkeypatch):
    """Per-test scripted responses + TTY-friendly provider wiring."""
    reset_resume_hint_for_test_only()
    scripted: list[ChatResponse] = []
    _wire_tty_wiring(monkeypatch, scripted)
    return scripted


def test_headless_text_output_prints_resume_hint_on_tty(tty_fake_wiring, tmp_path):
    """S-R1: text mode on a TTY must append the resume hint after the reply."""
    tty_fake_wiring.append(_text_response("Hello, human!"))

    stdout = _FakeTTYStdout()
    stderr = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="hi",
            output_format="text",
            stdout=stdout,  # type: ignore[arg-type]
            stderr=stderr,
            workspace_root=tmp_path,
        )
    )

    assert code == 0
    rendered = stdout.getvalue()
    # The reply is still emitted first.
    assert "Hello, human!" in rendered
    # The hint follows, matching the CCB ``printResumeHint()`` format.
    assert "Resume this session with: clawcodex --resume " in rendered
    # Pull the recoverable session id from the hint.
    after = rendered.split("Resume this session with: clawcodex --resume ", 1)[1]
    sid = after.strip().splitlines()[0].strip()
    assert sid


def test_headless_json_output_omits_resume_hint(tty_fake_wiring, tmp_path):
    """S-R1: JSON mode must not append the hint — the structured
    ``session_id`` field is the canonical channel for machine consumers.
    """
    tty_fake_wiring.append(_text_response("json reply"))

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="hi",
            output_format="json",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
        )
    )

    assert code == 0
    rendered = stdout.getvalue()
    # No human-readable hint text on the JSON stream.
    assert "Resume this session with:" not in rendered
    assert "clawcodex --resume" not in rendered
    # But the structured payload still carries the session id.
    payload = json.loads(rendered.strip())
    assert payload["session_id"]
    assert payload["goal_operation_id"] == payload["session_id"]


def test_headless_stream_json_output_omits_resume_hint(tty_fake_wiring, tmp_path):
    """S-R1: stream-json mode must not append the hint either —
    the session id is already in the SystemEvent and ResultEvent frames.
    """
    tty_fake_wiring.append(_text_response("stream reply"))

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="hi",
            output_format="stream-json",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
        )
    )

    assert code == 0
    rendered = stdout.getvalue()
    # The structured frames contain the session id; the plain-text hint
    # must not appear anywhere on the stream.
    assert "Resume this session with:" not in rendered
    assert "clawcodex --resume" not in rendered
    # Verify the session id is still in the structured frames.
    frames = [json.loads(line) for line in rendered.splitlines() if line.strip()]
    assert any("session_id" in f and f["session_id"] for f in frames)
