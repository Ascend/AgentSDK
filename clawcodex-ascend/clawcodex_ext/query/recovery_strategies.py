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

# pylint: disable=relative-beyond-top-level
"""P102-B Agent-loop recovery strategy registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from clawcodex_ext.types.messages import AssistantMessage, Message

if TYPE_CHECKING:
    from .config import FrozenQueryConfig, QueryConfig
    from .query import QueryParams
    from .transitions import QueryState

logger = logging.getLogger(__name__)


@dataclass
class RecoveryContext:
    """Context passed to a recovery strategy."""

    state: QueryState
    last_message: Message | None
    config: QueryConfig | FrozenQueryConfig
    params: QueryParams
    messages: list[Message]
    assistant_messages: list[AssistantMessage]
    error_type: str


RecoveryStrategyFn = Callable[[RecoveryContext], tuple["QueryState | None", list[Message]] | None]


@dataclass
class RecoveryStrategy:
    """Metadata and handler for one recovery strategy."""

    name: str
    fn: RecoveryStrategyFn
    priority: int = 0


# Global strategy list ordered by priority.
_STRATEGIES: list[RecoveryStrategy] = []

# Stable built-in names support testing and unregistration.
MAX_OUTPUT_TOKENS_ESCALATE = "max_output_tokens_escalate"
MAX_OUTPUT_TOKENS_RECOVERY = "max_output_tokens_recovery"
MAX_OUTPUT_TOKENS_EXHAUSTED = "max_output_tokens_exhausted"
COLLAPSE_ENGINE_RECOVERY = "collapse_engine_recovery"
REACTIVE_COMPACT_RECOVERY = "reactive_compact_recovery"
MEDIA_SIZE_FALLBACK = "media_size_fallback"
PROMPT_TOO_LONG_FALLBACK = "prompt_too_long_fallback"


# -- Public API -----------------------------------------------------------


def register_recovery_strategy(
    name: str,
    fn: RecoveryStrategyFn,
    priority: int = 0,
) -> None:
    """Register an agent-loop recovery strategy."""
    unregister_recovery_strategy(name)
    _STRATEGIES.append(RecoveryStrategy(name=name, fn=fn, priority=priority))
    _STRATEGIES.sort(key=lambda s: s.priority)
    logger.debug("Registered recovery strategy %r (priority=%d)", name, priority)


def unregister_recovery_strategy(name: str) -> None:
    """Unregister an agent-loop recovery strategy."""
    before = len(_STRATEGIES)
    _STRATEGIES[:] = [s for s in _STRATEGIES if s.name != name]
    if len(_STRATEGIES) < before:
        logger.debug("Unregistered recovery strategy %r", name)


def find_recovery_strategies(
    error_type: str,
    state: QueryState,  # noqa: ARG001
) -> list[RecoveryStrategy]:
    """Return recovery strategies that support the failure."""
    return list(_STRATEGIES)


def clear_recovery_strategies() -> None:
    """Clear recovery strategies for test isolation."""
    _STRATEGIES.clear()


# -- Built-in strategies -------------------------------------------------


def _max_output_tokens_escalate(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """Increase the output-token limit for one retry."""
    if ctx.error_type != "max_output_tokens":
        return None
    from .transitions import QueryState, Transition

    s = ctx.state
    if s.max_output_tokens_override is not None or s.max_output_tokens_recovery_count != 0:
        return None

    from .query import ESCALATED_MAX_TOKENS

    new_state = QueryState(
        messages=s.messages,
        tool_use_context=s.tool_use_context,
        auto_compact_tracking=s.auto_compact_tracking,
        max_output_tokens_recovery_count=s.max_output_tokens_recovery_count,
        has_attempted_reactive_compact=s.has_attempted_reactive_compact,
        max_output_tokens_override=ESCALATED_MAX_TOKENS,
        stop_hook_active=None,
        turn_count=s.turn_count,
        pending_tool_use_summary=s.pending_tool_use_summary,
        continuation_nudge_count=s.continuation_nudge_count,
        transition=Transition(reason="max_output_tokens_escalate"),
    )
    return (new_state, [])


def _max_output_tokens_recovery(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """Retry after an output-token limit failure."""
    if ctx.error_type != "max_output_tokens":
        return None
    from .transitions import QueryState, Transition

    s = ctx.state
    if s.max_output_tokens_recovery_count >= 3:  # MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
        return None

    from .query import _create_user_message

    recovery_message = _create_user_message(
        "Output token limit hit. Resume directly — no apology, no recap of what you were doing. "
        "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.",
        is_meta=True,
    )
    new_state = QueryState(
        messages=[*s.messages, *ctx.assistant_messages, recovery_message],
        tool_use_context=s.tool_use_context,
        auto_compact_tracking=s.auto_compact_tracking,
        max_output_tokens_recovery_count=s.max_output_tokens_recovery_count + 1,
        has_attempted_reactive_compact=s.has_attempted_reactive_compact,
        max_output_tokens_override=None,
        stop_hook_active=None,
        turn_count=s.turn_count,
        pending_tool_use_summary=s.pending_tool_use_summary,
        continuation_nudge_count=s.continuation_nudge_count,
        transition=Transition(
            reason="max_output_tokens_recovery",
            attempt=s.max_output_tokens_recovery_count + 1,
        ),
    )
    return (new_state, [])


def _max_output_tokens_exhausted(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """Stop recovery after output-token retries are exhausted."""
    if ctx.error_type != "max_output_tokens":
        return None
    s = ctx.state
    if s.max_output_tokens_recovery_count < 3:  # MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
        return None
    return (None, [ctx.last_message] if ctx.last_message is not None else [])


def _collapse_engine_recovery(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """Recover through the context-collapse engine."""
    if ctx.error_type != "prompt_too_long":
        return None
    s = ctx.state
    if s.has_attempted_reactive_compact or ctx.params.collapse_engine is None:
        return None

    from clawcodex_ext.services.api.errors import PromptTooLongError

    synthetic_err = PromptTooLongError("synthetic 413: prompt is too long, recovering via CollapseEngine")
    try:
        recovery = ctx.params.collapse_engine.recover_from_413(messages=s.messages, error=synthetic_err)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "CollapseEngine.recover_from_413 raised %r; falling back to reactive_compact",
            exc,
        )
        return None
    if recovery is None or not getattr(recovery, "applied", False):
        return None

    projected = ctx.params.collapse_engine.store.project_view(list(s.messages))
    yield_msgs: list[Message] = list(projected)

    if ctx.params.pipeline_config is not None and ctx.params.pipeline_config.autocompact_tracking is not None:
        ctx.params.pipeline_config.autocompact_tracking.consecutive_failures = 0

    from .transitions import QueryState, Transition

    new_state = QueryState(
        messages=projected,
        tool_use_context=s.tool_use_context,
        auto_compact_tracking=(
            ctx.params.pipeline_config.autocompact_tracking if ctx.params.pipeline_config is not None else None
        ),
        max_output_tokens_recovery_count=s.max_output_tokens_recovery_count,
        has_attempted_reactive_compact=True,
        max_output_tokens_override=None,
        stop_hook_active=s.stop_hook_active,
        turn_count=s.turn_count,
        pending_tool_use_summary=s.pending_tool_use_summary,
        continuation_nudge_count=s.continuation_nudge_count,
        transition=Transition(reason="collapse_engine_retry"),
    )
    return (new_state, yield_msgs)


async def _reactive_compact_recovery(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """Recover by compacting the active context."""
    if ctx.error_type not in ("prompt_too_long", "media_size"):
        return None
    s = ctx.state
    if s.has_attempted_reactive_compact:
        return None
    from .config import FrozenQueryConfig, QueryConfig

    assert isinstance(ctx.config, (QueryConfig, FrozenQueryConfig))
    if not ctx.config.reactive_compact_enabled:
        return None

    from clawcodex_ext.services.api.errors import PromptTooLongError
    from src.services.compact.reactive_compact import reactive_compact

    synthetic_err = PromptTooLongError("withheld during streaming, recovering")
    result = await reactive_compact(
        messages=s.messages,
        error=synthetic_err,
        provider=ctx.params.provider,
        model=ctx.config.model,
    )
    if not result.compacted:
        return (None, [ctx.last_message] if ctx.last_message is not None else [])

    post_compact_messages: list[Message] = result.messages
    yield_msgs: list[Message] = list(post_compact_messages)

    if ctx.params.pipeline_config is not None and ctx.params.pipeline_config.autocompact_tracking is not None:
        ctx.params.pipeline_config.autocompact_tracking.consecutive_failures = 0

    from .transitions import QueryState, Transition

    new_state = QueryState(
        messages=post_compact_messages,
        tool_use_context=s.tool_use_context,
        auto_compact_tracking=(
            ctx.params.pipeline_config.autocompact_tracking if ctx.params.pipeline_config is not None else None
        ),
        max_output_tokens_recovery_count=s.max_output_tokens_recovery_count,
        has_attempted_reactive_compact=True,
        max_output_tokens_override=None,
        stop_hook_active=s.stop_hook_active,
        turn_count=s.turn_count,
        pending_tool_use_summary=s.pending_tool_use_summary,
        continuation_nudge_count=s.continuation_nudge_count,
        transition=Transition(reason="reactive_compact_retry"),
    )
    return (new_state, yield_msgs)


def _media_size_fallback(ctx: RecoveryContext) -> tuple[QueryState | None, list[Message]] | None:
    """Retry media generation with a supported size."""
    if ctx.error_type != "media_size":
        return None
    if not ctx.state.has_attempted_reactive_compact:
        return None
        # A ``None`` state and non-empty output terminate recovery.
    return (None, [ctx.last_message] if ctx.last_message is not None else [])


def _prompt_too_long_fallback(
    ctx: RecoveryContext,
) -> tuple[QueryState | None, list[Message]] | None:
    """Recover from an overlong prompt."""
    if ctx.error_type != "prompt_too_long":
        return None
    if not ctx.state.has_attempted_reactive_compact:
        return None
    return (None, [ctx.last_message] if ctx.last_message is not None else [])


# Register built-ins; lower values run first.
def _register_builtin_strategies() -> None:
    register_recovery_strategy(MAX_OUTPUT_TOKENS_ESCALATE, _max_output_tokens_escalate, priority=10)
    register_recovery_strategy(MAX_OUTPUT_TOKENS_RECOVERY, _max_output_tokens_recovery, priority=20)
    register_recovery_strategy(MAX_OUTPUT_TOKENS_EXHAUSTED, _max_output_tokens_exhausted, priority=45)
    register_recovery_strategy(COLLAPSE_ENGINE_RECOVERY, _collapse_engine_recovery, priority=30)
    register_recovery_strategy(REACTIVE_COMPACT_RECOVERY, _reactive_compact_recovery, priority=40)
    register_recovery_strategy(MEDIA_SIZE_FALLBACK, _media_size_fallback, priority=100)
    register_recovery_strategy(PROMPT_TOO_LONG_FALLBACK, _prompt_too_long_fallback, priority=100)


_register_builtin_strategies()
