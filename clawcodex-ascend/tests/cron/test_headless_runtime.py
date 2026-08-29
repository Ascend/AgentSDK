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

"""Headless entrypoint cron runtime integration tests.

These tests verify that ``clawcodex_ext.entrypoints.headless.run_headless``
wires cron tools, attaches the scheduler with the busy gate, drains cron
prompts from ``tool_context.outbox``, and finalizes cron runs through the
claim/completed lifecycle.
"""

# pylint: disable=redefined-outer-name,no-name-in-module

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from src.entrypoints import HeadlessOptions, run_headless
from src.tool_system.defaults import build_default_registry as real_build_default_registry
from clawcodex_ext.cron_system.runs import (
    CreateCronRunParams,
    create_queued_run,
    read_cron_runs,
)
from clawcodex_ext.cron_system.tools import CronCreateTool, CronRunTool
from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.query.outbox_types import CronPromptEvent


class _FakeProvider:
    def __init__(self, api_key: str, base_url=None, model=None, *, responses=None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or "fake-model"
        self._responses = list(responses or [])

    def chat(self, messages, tools=None, **kwargs):
        if not self._responses:
            raise AssertionError("FakeProvider ran out of scripted responses")
        return self._responses.pop(0)

    def chat_stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError


def _text_response(text: str) -> ChatResponse:
    return ChatResponse(
        content=text,
        model="fake-model",
        usage={"input_tokens": 5, "output_tokens": len(text.split())},
        finish_reason="end_turn",
        tool_uses=None,
    )


@pytest.fixture
def headless_cron_wiring(monkeypatch):
    """Patch headless provider/registry wiring with fakes and a real registry.

    Unlike ``tests/cli/test_headless_cli.py::fake_wiring`` which uses a stub
    registry, this fixture restores the real ``build_default_registry`` so
    cron tool replacement can be observed end-to-end.
    """
    import clawcodex_ext.entrypoints.headless as ext_headless

    scripted_responses: list[ChatResponse] = []

    def _fake_get_provider_class(_name):
        def _factory(api_key, base_url=None, model=None, **_kwargs):
            return _FakeProvider(
                api_key,
                base_url=base_url,
                model=model,
                responses=list(scripted_responses),
            )

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
        lambda provider=None: real_build_default_registry(provider=provider),
        raising=False,
    )

    return scripted_responses


def test_headless_replaces_cron_tools(headless_cron_wiring, tmp_path):
    """Cron tools from clawcodex_ext replace the upstream fallback tools."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    headless_cron_wiring.append(_text_response("ok"))
    captured: dict = {}
    original_replace = ext_headless.replace_cron_tools

    def _record_and_replace(registry):
        captured["registry"] = registry
        return original_replace(registry)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ext_headless, "replace_cron_tools", _record_and_replace)
    try:
        code = run_headless(
            HeadlessOptions(
                prompt="hi",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                persist_on_exit=False,
            )
        )
    finally:
        monkeypatch.undo()

    assert code == 0
    registry = captured["registry"]
    assert registry is not None
    assert registry.get("CronCreate") is CronCreateTool
    assert registry.get("CronRun") is CronRunTool


def test_headless_attaches_cron_scheduler(headless_cron_wiring, tmp_path):
    """A CronScheduler is attached to tool_context and started."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    headless_cron_wiring.append(_text_response("ok"))
    captured: dict = {}

    async def _capture_tool_context(*args, **kwargs):
        captured["tool_context"] = kwargs["tool_context"]
        return SimpleNamespace(response_text="ok", usage={}, num_turns=1)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _capture_tool_context, raising=False)
    try:
        code = run_headless(
            HeadlessOptions(
                prompt="hi",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                persist_on_exit=False,
            )
        )
    finally:
        monkeypatch.undo()

    assert code == 0
    tool_context = captured["tool_context"]
    scheduler = getattr(tool_context, "cron_scheduler", None)
    assert scheduler is not None
    # Scheduler should have been stopped by the finally block.
    assert not scheduler._thread.is_alive()


def test_headless_scheduler_is_loading_follows_agent_loop(headless_cron_wiring, tmp_path):
    """The scheduler's busy gate is True while the agent loop is in flight."""
    import clawcodex_ext.entrypoints.headless as ext_headless

    headless_cron_wiring.append(_text_response("ok"))
    observed: list[bool] = []

    async def _observe_is_loading(*args, **kwargs):
        tool_context = kwargs["tool_context"]
        scheduler = getattr(tool_context, "cron_scheduler", None)
        if scheduler is not None:
            observed.append(scheduler.is_loading())
        return SimpleNamespace(response_text="ok", usage={}, num_turns=1)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _observe_is_loading, raising=False)
    try:
        run_headless(
            HeadlessOptions(
                prompt="hi",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                persist_on_exit=False,
            )
        )
    finally:
        monkeypatch.undo()

    assert observed, "run_query_as_agent_loop was never reached"
    assert all(observed), "is_loading should be True throughout agent loop execution"


def test_headless_drains_cron_prompt_and_finalizes_run(headless_cron_wiring, tmp_path):
    """A cron_prompt in tool_context.outbox is executed and finalized."""
    import clawcodex_ext.entrypoints.headless as ext_headless
    from clawcodex_ext.cron_system.scheduler import now_ms

    headless_cron_wiring.append(_text_response("cron ok"))

    # Inject a CronPromptEvent into the outbox right after the runtime is wired.
    original_attach = ext_headless.attach_cron_runtime

    def _attach_and_inject(ctx, *args, **kwargs):
        result = original_attach(ctx, *args, **kwargs)
        run = create_queued_run(
            tmp_path,
            CreateCronRunParams(
                task_id="injected-task",
                prompt="injected cron prompt",
                queued_at=now_ms(),
            ),
        )
        assert run is not None
        ctx.outbox.append(
            CronPromptEvent(
                prompt="injected cron prompt",
                task_id="injected-task",
                run_id=run.id,
            )
        )
        return result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ext_headless, "attach_cron_runtime", _attach_and_inject, raising=False)

    async def _fake_loop(*args, **kwargs):
        return SimpleNamespace(response_text="cron ok", usage={}, num_turns=1)

    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _fake_loop, raising=False)

    try:
        code = run_headless(
            HeadlessOptions(
                prompt="user prompt",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                persist_on_exit=False,
            )
        )
    finally:
        monkeypatch.undo()

    assert code == 0
    runs = read_cron_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].task_id == "injected-task"


def test_headless_accumulation_guard_cancels_duplicate_active_run(headless_cron_wiring, tmp_path):
    """Two cron_prompt events for the same task cancel the second run."""
    import clawcodex_ext.entrypoints.headless as ext_headless
    from clawcodex_ext.cron_system.scheduler import now_ms

    headless_cron_wiring.append(_text_response("cron ok"))

    original_attach = ext_headless.attach_cron_runtime

    def _attach_and_inject_duplicates(ctx, *args, **kwargs):
        result = original_attach(ctx, *args, **kwargs)
        # The run store blocks duplicate active runs for the same task,
        # so create two runs for different tasks but emit outbox events
        # sharing a task_id to exercise the headless accumulation guard.
        first = create_queued_run(
            tmp_path,
            CreateCronRunParams(
                task_id="shared-task",
                prompt="first",
                queued_at=now_ms(),
            ),
        )
        second = create_queued_run(
            tmp_path,
            CreateCronRunParams(
                task_id="other-task",
                prompt="second",
                queued_at=now_ms(),
            ),
        )
        assert first is not None and second is not None
        ctx.outbox.append(CronPromptEvent(prompt="first", task_id="shared-task", run_id=first.id))
        ctx.outbox.append(CronPromptEvent(prompt="second", task_id="shared-task", run_id=second.id))
        return result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ext_headless,
        "attach_cron_runtime",
        _attach_and_inject_duplicates,
        raising=False,
    )

    async def _fake_loop(*args, **kwargs):
        return SimpleNamespace(response_text="cron ok", usage={}, num_turns=1)

    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _fake_loop, raising=False)

    try:
        code = run_headless(
            HeadlessOptions(
                prompt="user prompt",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                persist_on_exit=False,
            )
        )
    finally:
        monkeypatch.undo()

    assert code == 0
    runs = {run.id: run for run in read_cron_runs(tmp_path)}
    statuses = {run.status for run in runs.values()}
    assert "completed" in statuses
    assert "cancelled" in statuses
    assert len(runs) == 2


def test_headless_cron_run_failure_finalizes_run_failed(headless_cron_wiring, tmp_path):
    """If the agent loop raises for a cron prompt, the run is marked failed."""
    import clawcodex_ext.entrypoints.headless as ext_headless
    from clawcodex_ext.cron_system.scheduler import now_ms

    headless_cron_wiring.append(_text_response("unused"))

    original_attach = ext_headless.attach_cron_runtime

    def _attach_and_inject(ctx, *args, **kwargs):
        result = original_attach(ctx, *args, **kwargs)
        run = create_queued_run(
            tmp_path,
            CreateCronRunParams(
                task_id="failing-task",
                prompt="boom",
                queued_at=now_ms(),
            ),
        )
        assert run is not None
        ctx.outbox.append(CronPromptEvent(prompt="boom", task_id="failing-task", run_id=run.id))
        return result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ext_headless, "attach_cron_runtime", _attach_and_inject, raising=False)

    call_count = 0

    async def _fail_once(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Fail the cron prompt (first call) but succeed the user prompt.
        if call_count == 1:
            raise RuntimeError("simulated cron failure")
        return SimpleNamespace(response_text="user ok", usage={}, num_turns=1)

    monkeypatch.setattr(ext_headless, "run_query_as_agent_loop", _fail_once, raising=False)

    try:
        code = run_headless(
            HeadlessOptions(
                prompt="user prompt",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                persist_on_exit=False,
            )
        )
    finally:
        monkeypatch.undo()

    # The cron failure returns False from _run_cron_prompt; the user prompt
    # still runs, so the overall exit code stays 0.
    assert code == 0
    runs = read_cron_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert "simulated cron failure" in (runs[0].error or "")
