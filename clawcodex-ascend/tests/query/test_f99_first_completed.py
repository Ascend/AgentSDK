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

"""Tests for ``_run_tools_partitioned`` cancellation with FIRST_COMPLETED.

The fix replaces ``asyncio.gather`` with task creation +
``asyncio.wait(FIRST_COMPLETED)`` so a user abort short-circuits
remaining concurrent tool dispatches as soon as one tool finishes
(or the abort signal trips), instead of waiting for the slowest
tool in the batch.

These tests construct real ``Tool`` objects so
``_partition_tool_calls`` produces the expected concurrent-safe
batch. ``_dispatch_single_tool`` is patched out so we don't have
to wire up a real registry.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tool_system.build_tool import Tool
from src.tool_system.context import ToolContext
from src.types.content_blocks import ToolResultBlock, ToolUseBlock
from src.types.messages import UserMessage
from src.utils.abort_controller import AbortController

from src.query.query import _run_tools_partitioned


# ---------------------------------------------------------------------------
# Fixtures / helpers


def _make_block(tool_use_id: str, name: str = "Read", input: dict | None = None) -> ToolUseBlock:
    return ToolUseBlock(
        type="tool_use",
        id=tool_use_id,
        name=name,
        input=input or {"file_path": f"/tmp/fake_{tool_use_id}"},
    )


def _make_tool_result_msg(tool_use_id: str, content: str = "ok") -> UserMessage:
    return UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id=tool_use_id,
                content=content,
                is_error=False,
            )
        ],
    )


def _make_concurrency_safe_tool(name: str = "Read") -> Tool:
    """A minimal Tool instance whose ``is_concurrency_safe`` returns True.

    Real production tools (Read, Grep, Glob) declare this; tests
    need to bypass the default ``is_concurrency_safe=lambda _input:
    False`` so ``_partition_tool_calls`` groups multiple ``Read``
    blocks into one concurrent-safe batch. Without this, every
    block lands in its own exclusive batch and the FIRST_COMPLETED
    path is never exercised.
    """
    return Tool(
        name=name,
        input_schema={"type": "object", "properties": {}},
        call=lambda _input, _ctx: None,
        prompt=lambda: "",
        description=lambda _input: "",
        map_result_to_api=lambda output, _id: {"content": str(output)},
        check_permissions=lambda _input, _ctx: None,
        is_enabled=lambda: True,
        is_concurrency_safe=lambda _input: True,
        is_read_only=lambda _input: True,
        is_destructive=lambda _input: False,
        user_facing_name=lambda _input=None: name,
        to_auto_classifier_input=lambda _input: "",
    )


@pytest.fixture
def tool_context():
    """ToolContext with abort controller wired so we can trip mid-batch."""
    tmp = tempfile.TemporaryDirectory()
    workspace = Path(tmp.name)
    controller = AbortController()
    ctx = ToolContext(workspace_root=workspace, abort_controller=controller)
    yield ctx, controller
    tmp.cleanup()


@pytest.fixture
def fake_registry():
    """A MagicMock registry — we patch ``_dispatch_single_tool`` directly."""
    return MagicMock()


@pytest.fixture
def concurrent_tools():
    """A ``Tools`` list containing one concurrency-safe Read tool.

    ``_partition_tool_calls`` iterates this list to determine each
    block's ``is_concurrent_safe``; the default Tool factory returns
    ``False`` for that predicate, which routes every block through
    the sequential exclusive-batch path and never exercises the
    FIRST_COMPLETED machinery. These tests
    must supply a tool whose predicate returns True.
    """
    return [_make_concurrency_safe_tool("Read")]


# ---------------------------------------------------------------------------
# FIRST_COMPLETED short-circuit on abort


@pytest.mark.asyncio
async def test_first_completed_short_circuits_on_abort(
    tool_context,
    fake_registry,
    concurrent_tools,
) -> None:
    """An abort mid-batch unblocks the agent loop before the slowest tool finishes.

    With three concurrent tools taking 0.3s/0.3s/1.0s and an abort at
    0.4s, the agent loop must return well before 1.0s. The expected
    bound is one tool's runtime + the abort poll interval — the slow
    tool's 1.0s runtime must NOT gate the return.
    """
    ctx, controller = tool_context
    blocks = [_make_block("t1"), _make_block("t2"), _make_block("t3")]

    tool_durations = {"t1": 0.3, "t2": 0.3, "t3": 1.0}

    def _fake_dispatch(block, *args, **kwargs):
        duration = tool_durations.get(block.id, 0.1)
        time.sleep(duration)
        return _make_tool_result_msg(block.id, f"done-{block.id}"), []

    async def _trip_abort_after_one():
        await asyncio.sleep(0.4)  # after t1 + t2 complete, before t3
        controller.abort("user_interrupt")

    start = time.monotonic()
    with patch("src.query.query._dispatch_single_tool", side_effect=_fake_dispatch):
        trip_task = asyncio.create_task(_trip_abort_after_one())
        result = await _run_tools_partitioned(
            blocks,
            fake_registry,
            ctx,
            concurrent_tools,
        )
        await trip_task
    elapsed = time.monotonic() - start

    # The slow tool (t3) is 1.0s. The fix must bound cancel latency
    # well below that — we expect ≤ 0.9s (abort poll interval + one
    # tool runtime + scheduling margin).
    assert elapsed < 0.9, (
        f"F-99 fix should cancel within ~one tool runtime; got {elapsed:.3f}s (slow tool alone takes 1.0s)"
    )
    # tool_use/tool_result pairing must hold — three blocks in, three
    # results out (some real, some synthesised cancelled).
    assert len(result) == 3, f"expected 3 tool_results, got {len(result)}"


@pytest.mark.asyncio
async def test_first_completed_no_abort_waits_for_all(
    tool_context,
    fake_registry,
    concurrent_tools,
) -> None:
    """Without an abort, behaviour matches the previous ``asyncio.gather`` path.

    Regression guard: the FIRST_COMPLETED rewrite must preserve the
    happy-path contract — all concurrent tools' results land in the
    output list. We don't want a behaviour change for non-aborting
    flows.
    """
    ctx, controller = tool_context
    blocks = [_make_block(f"t{i}") for i in range(3)]

    def _fake_dispatch(block, *args, **kwargs):
        time.sleep(0.05)
        return _make_tool_result_msg(block.id, f"done-{block.id}"), []

    with patch("src.query.query._dispatch_single_tool", side_effect=_fake_dispatch):
        result = await _run_tools_partitioned(
            blocks,
            fake_registry,
            ctx,
            concurrent_tools,
        )

    assert len(result) == 3
    result_ids = {b.tool_use_id for msg in result for b in msg.content if isinstance(b, ToolResultBlock)}
    assert result_ids == {"t0", "t1", "t2"}


@pytest.mark.asyncio
async def test_first_completed_preserves_pairing_on_abort(
    tool_context,
    fake_registry,
    concurrent_tools,
) -> None:
    """Every ``tool_use`` gets a paired ``tool_result``, even on abort.

    The agent loop relies on the tool_use/tool_result pairing
    invariant — an orphan tool_use on the next API call causes a
    400. The fix must emit a synthetic cancelled tool_result for
    any cancelled task so pairing stays intact on the abort path.
    """
    ctx, controller = tool_context
    blocks = [_make_block(f"t{i}") for i in range(3)]

    def _fake_dispatch(block, *args, **kwargs):
        # Slow tools so the abort trips before any complete.
        time.sleep(1.0)
        return _make_tool_result_msg(block.id, f"done-{block.id}"), []

    async def _trip_abort_immediately():
        await asyncio.sleep(0.05)
        controller.abort("user_interrupt")

    with patch("src.query.query._dispatch_single_tool", side_effect=_fake_dispatch):
        trip_task = asyncio.create_task(_trip_abort_immediately())
        result = await _run_tools_partitioned(
            blocks,
            fake_registry,
            ctx,
            concurrent_tools,
        )
        await trip_task

    # Every tool_use_id from the input must have a paired tool_result
    # in the output (real or synthesised).
    paired_ids = {b.tool_use_id for msg in result for b in msg.content if isinstance(b, ToolResultBlock)}
    assert paired_ids == {"t0", "t1", "t2"}


@pytest.mark.asyncio
async def test_first_completed_single_tool_unchanged(
    tool_context,
    fake_registry,
    concurrent_tools,
) -> None:
    """A single-tool batch uses the simple path without FIRST_COMPLETED overhead.

    The single-tool case is common (most batches have one tool) and
    the fix keeps it on the original ``asyncio.to_thread`` direct
    path to avoid unnecessary asyncio task creation overhead.
    """
    ctx, controller = tool_context
    blocks = [_make_block("t0")]

    def _fake_dispatch(block, *args, **kwargs):
        return _make_tool_result_msg(block.id, "done-t0"), []

    with patch("src.query.query._dispatch_single_tool", side_effect=_fake_dispatch):
        result = await _run_tools_partitioned(
            blocks,
            fake_registry,
            ctx,
            concurrent_tools,
        )

    assert len(result) == 1
    paired_ids = {b.tool_use_id for msg in result for b in msg.content if isinstance(b, ToolResultBlock)}
    assert paired_ids == {"t0"}
