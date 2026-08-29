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


# pylint: disable=C0412,E0611,W0101,W0125
import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from clawcodex_ext.utils.abort_controller import AbortError
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.protocol import ToolCall
from clawcodex_ext.types.content_blocks import TextBlock
from src.types.messages import AssistantMessage


def _wait_for_task_status(context: ToolContext, task_id: str, timeout_s: float = 2.0) -> str:
    """Poll the chapter-10 runtime_tasks registry for terminal status.

    Pre-Chunk-B this checked ``context.tasks[task_id].status`` (the dict
    that hosted async-agent state via the ``metadata._internal=True``
    workaround). Post-WI-1.5 the source of truth is
    ``context.runtime_tasks``; the legacy dict no longer holds agent state.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = context.runtime_tasks.get(task_id)
        if state is not None:
            status = str(state.status)
            # ``running`` is the chapter-10 in-flight value. ``completed``
            # / ``failed`` / ``killed`` are terminal — return the moment
            # we see one.
            if status and status != "running" and status != "pending":
                return status
        time.sleep(0.05)
    state = context.runtime_tasks.get(task_id)
    return str(state.status) if state is not None else ""


def _task_output_text(context: ToolContext, task_id: str) -> str:
    """Helper: return the agent's final text output post-Chunk-B."""
    from src.tasks.local_agent import LocalAgentTaskState

    state = context.runtime_tasks.get(task_id)
    if isinstance(state, LocalAgentTaskState):
        return state.result_text
    return ""


def test_async_agent_launch_persists_completed_output(tmp_path: Path) -> None:
    registry = build_default_registry(provider=object())
    context = ToolContext(workspace_root=tmp_path)

    async def _fake_run_agent(_params):
        yield AssistantMessage(content=[TextBlock(text="async done")])

    with patch("src.tool_system.tools.agent.run_agent", _fake_run_agent):
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={
                    "description": "background test",
                    "prompt": "do background work",
                    "run_in_background": True,
                },
            ),
            context,
        )

        assert result.is_error is False
        assert isinstance(result.output, dict)
        assert result.output.get("status") == "async_launched"

        task_id = str(result.output["agent_id"])
        assert result.output.get("task_output_key") == task_id
        # Chapter-10 / Chunk B / WI-1.5: the runtime task lives on
        # ``context.runtime_tasks`` (typed), no longer on ``context.tasks``.
        assert task_id in context.runtime_tasks
        assert task_id not in context.tasks  # explicit migration assertion

        final_status = _wait_for_task_status(context, task_id)
        assert final_status == "completed"
        assert "async done" in _task_output_text(context, task_id)


def test_async_agent_launch_marks_failed_output(tmp_path: Path) -> None:
    registry = build_default_registry(provider=object())
    context = ToolContext(workspace_root=tmp_path)

    async def _failing_run_agent(_params):
        if False:
            yield AssistantMessage(content=[TextBlock(text="unused")])
        raise RuntimeError("boom")

    with patch("src.tool_system.tools.agent.run_agent", _failing_run_agent):
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={
                    "description": "background failure",
                    "prompt": "fail",
                    "run_in_background": True,
                },
            ),
            context,
        )

        task_id = str(result.output["agent_id"])
        final_status = _wait_for_task_status(context, task_id)
        assert final_status == "failed"
        assert "boom" in _task_output_text(context, task_id)


@pytest.mark.asyncio
async def test_async_agent_is_detached_and_session_shutdown_aborts_it(tmp_path: Path) -> None:
    """Background agents must not become pending tasks on the parent loop.

    A session-level shutdown still reaches the child's independent abort
    controller, marks the runtime task killed, and joins the managed worker.
    """

    registry = build_default_registry(provider=object())
    context = ToolContext(workspace_root=tmp_path)
    started = threading.Event()
    aborted = threading.Event()
    child_params = {}

    async def _blocking_run_agent(params):
        child_params["value"] = params
        started.set()
        while not params.abort_controller.signal.aborted:
            await asyncio.sleep(0.01)
        aborted.set()
        raise AbortError("task_stopped")
        if False:  # pragma: no cover - keeps this an async generator
            yield AssistantMessage(content=[TextBlock(text="unused")])

    with patch("src.tool_system.tools.agent.run_agent", _blocking_run_agent):
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={
                    "description": "detached background test",
                    "prompt": "wait until session shutdown",
                    "run_in_background": True,
                },
            ),
            context,
        )

        task_id = str(result.output["agent_id"])
        assert started.wait(1.0)
        # Regression guard: the old in-loop branch skipped TaskManager here,
        # causing asyncio.run() to wait for this child during parent cancel.
        assert context.task_manager.list()
        context.abort_controller.abort("user_interrupt")
        await asyncio.sleep(0.05)
        assert child_params["value"].abort_controller.signal.aborted is False
        assert context.runtime_tasks.get(task_id).status == "running"

        assert context.task_manager.shutdown(timeout=2.0) is True
        assert aborted.wait(1.0)
        state = context.runtime_tasks.get(task_id)
        assert state is not None
        assert state.status == "killed"
