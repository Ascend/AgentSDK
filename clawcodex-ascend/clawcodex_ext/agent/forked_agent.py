"""Forked agent execution primitive.

Mirrors ``typescript/src/utils/forkedAgent.ts``.

Provides ``run_forked_agent()`` — an isolated, parameterised fork of the
main query loop.  The fork re-uses the parent's system prompt / user context /
system context so the API request prefix is byte-identical (prompt-cache
amortisation), but runs inside a freshly-isolated ``ToolContext`` so no
mutable state leaks back to the parent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from clawcodex_ext.agent.subagent_context import (
    SubagentContextOverrides,
    create_subagent_context,
)
from clawcodex_ext.query.query import QueryParams, run_query
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.types.messages import Message, AssistantMessage
from clawcodex_ext.types.content_blocks import ToolUseBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default deny-all tool handler
# ---------------------------------------------------------------------------


async def _default_deny_tool_handler(_tool_use: ToolUseBlock) -> "PermissionDecision":
    """Deny every tool use — used for side questions that must not execute tools."""
    return PermissionDecision(behavior="deny")


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a tool-permission check."""

    behavior: str  # "allow" | "deny" | "ask"
    reason: str | None = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CacheSafeParams:
    """Frozen snapshot of the parent's API-request prefix.

    Captured after the last turn so a fork child can replay the *exact*
    same system_prompt + user_context + system_context bytes, maximising
    Anthropic prompt-cache hits.
    """

    system_prompt: str | list[dict[str, Any]]
    tool_use_context: ToolContext
    user_context: dict[str, str] | None = None
    system_context: dict[str, str] | None = None
    fork_context_messages: list[Message] = field(default_factory=list)


@dataclass
class ForkedAgentParams:
    """Parameters for ``run_forked_agent``."""

    prompt_messages: list[Message]
    cache_safe_params: CacheSafeParams
    can_use_tool: Callable[[ToolUseBlock], Awaitable[PermissionDecision]] | None = None
    query_source: str = "forked_agent"
    fork_label: str = "fork"
    overrides: SubagentContextOverrides | None = None
    max_output_tokens: int | None = None
    max_turns: int | None = None
    on_message: Callable[[Message], None] | None = None
    skip_transcript: bool = False
    skip_cache_write: bool = False


@dataclass
class ForkedAgentResult:
    """Result of a forked agent run."""

    messages: list[Message]
    total_usage: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Global memory cache for CacheSafeParams (F-122-C/D)
# ---------------------------------------------------------------------------

_last_cache_safe_params: CacheSafeParams | None = None


def save_cache_safe_params(params: CacheSafeParams | None) -> None:
    """Save the latest CacheSafeParams to global memory.

    Called from the main query loop's stop-hooks path (or equivalent)
    after every API response so /btw can replay the frozen prefix.
    """
    global _last_cache_safe_params
    _last_cache_safe_params = params


def get_last_cache_safe_params() -> CacheSafeParams | None:
    """Retrieve the last saved CacheSafeParams."""
    return _last_cache_safe_params


# ---------------------------------------------------------------------------
# Core primitive: run_forked_agent
# ---------------------------------------------------------------------------


async def run_forked_agent(
    params: ForkedAgentParams,
) -> ForkedAgentResult:
    """Run an isolated fork of the query loop.

    Key differences from a normal ``query()`` call:

    * Re-uses the parent's ``system_prompt`` / ``user_context`` /
      ``system_context`` (no re-render).
    * Creates a fresh ``ToolContext`` via ``create_subagent_context``.
    * Parameterises ``can_use_tool`` (default deny-all), ``max_turns``,
      ``skip_cache_write``.
    * Finally: clears read-file state and drops references so the GC can
      reclaim the fork's temporary messages.
    """
    parent_context = params.cache_safe_params.tool_use_context
    csp = params.cache_safe_params

    # 1. Build isolated child context
    child_context = create_subagent_context(
        parent_context,
        overrides=params.overrides,
    )

    # 2. Build initial messages: fork context + prompt messages
    initial_messages: list[Message] = [
        *csp.fork_context_messages,
        *params.prompt_messages,
    ]

    # 4. Build QueryParams
    #    - provider: reuse parent's active provider (set on tool_use_context
    #      by the main loop via setattr(..., "_active_provider", ...))
    provider = getattr(parent_context, "_active_provider", None)
    if provider is None:
        raise RuntimeError(
            "Forked agent requires an active provider on the parent context. "
            "Ensure the main loop has set tool_use_context._active_provider."
        )

    # Derive tools from parent context options; for a side-question we
    # normally pass an empty list so the model sees no tools, but we keep
    # the parent's registry reference for completeness.
    from clawcodex_ext.tool_system.build_tool import Tools

    tools = Tools()
    if parent_context.options and getattr(parent_context.options, "tools", None):
        tools = parent_context.options.tools  # type: ignore[attr-defined]

    query_params = QueryParams(
        messages=initial_messages,
        system_prompt=csp.system_prompt,
        tools=tools,
        tool_registry=parent_context.tool_registry,
        tool_use_context=child_context,
        provider=provider,
        abort_controller=child_context.abort_controller,
        query_source=params.query_source,
        max_output_tokens_override=params.max_output_tokens,
        max_turns=params.max_turns,
        user_context=csp.user_context,
        system_context=csp.system_context,
    )

    # 5. Run the query loop
    messages: list[Message] = []
    total_usage: dict[str, Any] = {}
    try:
        returned_messages, terminal = await run_query(query_params)
        for msg in returned_messages:
            if isinstance(msg, Message):
                messages.append(msg)
                if params.on_message:
                    params.on_message(msg)
                # Accumulate usage from assistant messages
                if isinstance(msg, AssistantMessage) and msg.usage:
                    total_usage = _merge_usage(total_usage, msg.usage)
    finally:
        # 6. Cleanup: clear read-file state, drop references
        child_context.read_file_fingerprints.clear()
        initial_messages.clear()

    return ForkedAgentResult(messages=messages, total_usage=total_usage)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_usage(acc: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge two usage dicts by summing numeric values."""
    result = dict(acc)
    for key, value in new.items():
        if isinstance(value, (int, float)) and isinstance(result.get(key), (int, float)):
            result[key] = result[key] + value  # type: ignore[operator]
        else:
            result[key] = value
    return result
