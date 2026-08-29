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

"""Regression: the sync subagent path must register its
``LocalAgentTaskState`` on ``context.runtime_tasks`` via ``upsert`` — never
via bracket assignment.

The original bug at ``src/tool_system/tools/agent.py:446`` was
``context.runtime_tasks[agent_id] = sync_state``, which raised
``TypeError: 'RuntimeTaskRegistry' object does not support item assignment``
because the typed registry (see ``src/task_registry.py:RuntimeTaskRegistry``)
exposes ``upsert`` / ``remove`` / ``update`` only — there is no
``__setitem__``.

This test pins the corrected contract: a single sync ``Agent`` invocation
produces exactly one ``LocalAgentTaskState`` registered under its
generated ``agent_id``, and the reference captured at registration time
is the same object identity that a follow-up ``runtime_tasks.get`` would
read back. That identity guarantee matters because
``_run_sync_agent`` later mutates the state in place (status flip,
``result_text``, ``completed_at``) — readers that captured the original
reference must see those updates.
"""

from __future__ import annotations


# pylint: disable=C0412,C1803,E0611
from pathlib import Path
from unittest.mock import patch


from clawcodex_ext.agent.agent_definitions import AgentDefinition
from src.tasks.local_agent import LocalAgentTaskState
from src.tool_system.context import ToolContext, ToolUseOptions
from src.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.protocol import ToolCall
from clawcodex_ext.types.content_blocks import TextBlock
from src.types.messages import AssistantMessage


def _stub_general_purpose_agent() -> AgentDefinition:
    """Minimal in-memory agent def so the test does not depend on
    filesystem-based agent discovery finding a built-in.
    """
    return AgentDefinition(
        agent_type="general-purpose",
        when_to_use="regression test stub",
        max_turns=1,
    )


def _make_context(tmp_path: Path) -> ToolContext:
    ctx = ToolContext(workspace_root=tmp_path)
    # Pre-populate the active-agents override so ``_get_agent_definitions``
    # short-circuits before touching the filesystem.
    ctx.options = ToolUseOptions(
        agent_definitions={"active_agents": [_stub_general_purpose_agent()]},
    )
    return ctx


def test_sync_agent_registers_on_runtime_tasks(tmp_path: Path) -> None:
    """Sync ``Agent`` invocation registers a single ``LocalAgentTaskState``
    on ``context.runtime_tasks`` whose identity survives the in-place
    completion mutations the path performs after streaming.
    """
    registry = build_default_registry(provider=object())
    context = _make_context(tmp_path)

    # Spy on ``upsert`` so we can capture the exact reference the sync
    # path hands to the registry. Identity preservation is the contract
    # under test — not just "something got registered".
    captured: list[LocalAgentTaskState] = []
    original_upsert = context.runtime_tasks.upsert

    def spy_upsert(state: LocalAgentTaskState) -> None:
        captured.append(state)
        return original_upsert(state)

    context.runtime_tasks.upsert = spy_upsert  # type: ignore[method-assign]

    # ``get_agent_transcript_path`` would otherwise try to open a
    # transcript file under ``.clawcodex/transcripts/<sid>/<id>.jsonl``;
    # returning None makes the path skip the writer entirely.
    with patch(
        "src.agent.transcript.get_agent_transcript_path",
        return_value=None,
    ):

        async def fake_run_agent(_params):
            # One empty assistant message so ``finalize_agent_tool`` has
            # something to format; the streaming loop then exits cleanly.
            yield AssistantMessage(content=[TextBlock(text="")])

        with patch(
            "src.tool_system.tools.agent.run_agent",
            fake_run_agent,
        ):
            result = registry.dispatch(
                ToolCall(
                    name="Agent",
                    input={
                        "description": "regression test",
                        "prompt": "noop",
                        "subagent_type": "general-purpose",
                    },
                ),
                context,
            )

    # The sync path must have completed without error.
    assert result.is_error is False, f"sync Agent dispatch failed: {getattr(result, 'output', None)!r}"

    # The heart of the regression: exactly one state was registered on
    # the runtime registry, and it is the same object identity that
    # ``runtime_tasks.get(agent_id)`` reads back. This is what the
    # original ``registry[agent_id] = state`` bug would have broken
    # (TypeError at registration) and what the in-place mutation
    # contract depends on.
    assert len(captured) == 1, "sync subagent path must register exactly one task state on runtime_tasks"
    sync_state = captured[0]
    assert isinstance(sync_state, LocalAgentTaskState)
    assert sync_state.id == sync_state.agent_id  # chapter-fidelity
    assert sync_state.description == "regression test"

    registered = context.runtime_tasks.get(sync_state.id)
    assert registered is sync_state, (
        "registered object identity lost — the path likely replaced the "
        "entry instead of mutating it in place, breaking any reader that "
        "captured the original reference (the post-streaming status flip, "
        "result_text, and completed_at would no longer reach them)"
    )
    # And the in-place mutation actually happened — status flipped to
    # completed on the very same object.
    assert registered.status == "completed"


def test_sync_agent_does_not_use_bracket_assignment(tmp_path: Path) -> None:
    """Belt-and-braces: assert that the sync path does NOT touch
    ``runtime_tasks`` via ``__setitem__``. ``RuntimeTaskRegistry`` has no
    ``__setitem__`` — any such call would raise ``TypeError`` at
    registration time, which is the very failure mode this whole
    contract guards against.
    """
    registry = build_default_registry(provider=object())
    context = _make_context(tmp_path)

    setitem_calls: list[tuple[str, object]] = []

    def trap_setitem(self, key, value):  # noqa: ANN001
        setitem_calls.append((key, value))
        # Mirror Mapping's behaviour so the test would fail loudly if
        # the call site were ever reintroduced — the bug surfaces here
        # as a TypeError on the production code path.
        raise TypeError("'RuntimeTaskRegistry' object does not support item assignment")

    with patch(
        "src.agent.transcript.get_agent_transcript_path",
        return_value=None,
    ):

        async def fake_run_agent(_params):
            yield AssistantMessage(content=[TextBlock(text="")])

        with (
            patch(
                "src.tool_system.tools.agent.run_agent",
                fake_run_agent,
            ),
            patch.object(
                type(context.runtime_tasks),
                "__setitem__",
                trap_setitem,
                create=True,
            ),
        ):
            result = registry.dispatch(
                ToolCall(
                    name="Agent",
                    input={
                        "description": "guard test",
                        "prompt": "noop",
                        "subagent_type": "general-purpose",
                    },
                ),
                context,
            )

    assert result.is_error is False
    assert setitem_calls == [], (
        f"sync path reached into RuntimeTaskRegistry via __setitem__; "
        f"this is the exact failure mode the upsert-based contract exists "
        f"to prevent. calls={setitem_calls!r}"
    )
