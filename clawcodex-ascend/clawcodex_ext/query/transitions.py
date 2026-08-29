#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from clawcodex_ext.types.messages import Message
from clawcodex_ext.tool_system.context import ToolContext


ContinueReason = Literal[
    "next_turn",
    "max_output_tokens_recovery",
    "max_output_tokens_escalate",
    "reactive_compact_retry",
    "collapse_drain_retry",
    "stop_hook_blocking",
    "token_budget_continuation",
    "continuation_nudge",
]


TerminalReason = Literal[
    "blocking_limit",
    "image_error",
    "model_error",
    "aborted_streaming",
    "prompt_too_long",
    "completed",
    "stop_hook_prevented",
    "aborted_tools",
    "hook_stopped",
    "max_turns",
    "tool_failure_loop",
    "goal_evaluator_error",
]


@dataclass(frozen=True)
class Transition:
    reason: ContinueReason
    attempt: int | None = None
    committed: int | None = None


@dataclass(frozen=True)
class Terminal:
    reason: TerminalReason
    error: Exception | None = None
    turn_count: int | None = None


class TerminalHolder:
    """Mutable container an async generator appends its Terminal to
    just before its bare ``return``. Callers read ``.value`` after
    consuming the generator.

    Required because Python's async generator protocol does not
    expose the return value of a ``return`` statement, and Python
    forbids ``return value`` inside ``async def`` generators
    (SyntaxError on 3.10-3.14; PEP 828 still Draft).
    """

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: Terminal | None = None


def set_terminal(
    holder: TerminalHolder,
    flag: list[bool],
    terminal: Terminal,
) -> None:
    """Set holder.value and mark the exit as natural.

    Use at every ``return`` site inside the query loop's inner
    generator. Call as the last action before ``return``.

    The ``flag`` argument is a single-element list[bool] used as a
    mutable container that the outer wrapper can inspect after
    iteration to distinguish natural termination from
    ``.aclose()`` / exception unwinds. Phase A introduces the helper
    and the parameter; a future phase wires the outer wrapper that
    reads the flag — until then ``flag`` is write-only and the
    parameter is preserved for forward compatibility.
    """
    holder.value = terminal
    flag[0] = True


ToolUseContext = ToolContext


@dataclass
class QueryState:
    messages: list[Message]
    tool_use_context: ToolUseContext
    auto_compact_tracking: Any | None = None
    max_output_tokens_recovery_count: int = 0
    has_attempted_reactive_compact: bool = False
    max_output_tokens_override: int | None = None
    stop_hook_active: bool | None = None
    turn_count: int = 1
    # Promise from previous turn's Haiku summary, resolved during current streaming.
    # Mirrors TS State.pendingToolUseSummary at query.ts:212. Currently unused in
    # Python (tool-use summary is out-of-scope for ch5) but reserved so callers
    # can plumb a future Haiku promise through state without a struct change.
    pending_tool_use_summary: Any | None = None
    # Count of consecutive continuation nudges within the current turn.
    # Capped at MAX_CONTINUATION_NUDGES to prevent infinite nudge loops
    # when the model keeps matching continuation signals without tool calls.
    # Mirrors TS State.continuationNudgeCount at query.ts:218.
    continuation_nudge_count: int = 0
    # Session-chain latch for the one-time exhaustive-result audit. Unlike
    # continuation_nudge_count, this survives successful tool rounds.
    exhaustive_audit_performed: bool = False
    transition: Transition | None = None
    # P102-E: External consumers such as Budget Mode can register
    # callbacks at turn boundaries without modifying ``query()``.
    on_turn_start_callbacks: list[Callable[["QueryState"], None]] = field(default_factory=list)
    on_turn_end_callbacks: list[Callable[["QueryState"], None]] = field(default_factory=list)
