"""/btw side-question command implementation.

Mirrors ``typescript/src/commands/btw/btw.tsx``.

Provides a context-zero-pollution side channel for users to ask quick
questions without interrupting the main agent work stream.
"""

from __future__ import annotations

import logging
from typing import Any

from clawcodex_ext.agent.side_question import run_side_question
from clawcodex_ext.agent.forked_agent import CacheSafeParams, get_last_cache_safe_params
from clawcodex_ext.command_system.btw_stats import increment_btw_use_count
from clawcodex_ext.command_system.types import (
    CommandContext,
    InteractiveCommand,
    InteractiveOutcome,
)

__all__ = ["BTW_COMMAND", "btw_command_run", "BtwCommand"]

logger = logging.getLogger(__name__)


# F-122-F: when the answer body (excluding the 💡 prefix) exceeds this many
# lines we mark the InteractiveOutcome as scrollable so the REPL enters its
# keyboard-scrolled view. Below the threshold, a flat print is friendlier —
# the spinner-then-scrollable-modal would feel heavy for one-line replies.
_SCROLLABLE_LINE_THRESHOLD = 8


def _should_render_scrollable(message: str) -> bool:
    """Decide whether the answer should be rendered in scrollable mode.

    Conservative heuristic on the rendered body (prefix and trailing blanks
    excluded): if it spills past the threshold line count, mark scrollable.
    The REPL still re-checks against the live terminal height, so this is
    just the *suggestion* — a tiny terminal degrades to flat print.
    """
    if not message:
        return False
    # Strip the 💡 prefix and surrounding blanks so the threshold reflects
    # body length, not the decoration.
    body = message.lstrip()
    if body.startswith("💡"):
        body = body[1:].lstrip()
    body = body.rstrip()
    line_count = body.count("\n") + 1 if body else 0
    return line_count > _SCROLLABLE_LINE_THRESHOLD


# ---------------------------------------------------------------------------
# Command implementation
# ---------------------------------------------------------------------------


async def btw_command_run(args: str, context: CommandContext) -> InteractiveOutcome:
    """Handle /btw command.

    Args:
        args: User's question text (everything after ``/btw``).
        context: Command execution context.

    Returns:
        InteractiveOutcome with the answer text or usage help. Long
        answers (more than :data:`_SCROLLABLE_LINE_THRESHOLD` lines) carry
        ``scrollable=True`` so the REPL enters a keyboard-scrolled view
        instead of dumping a wall of text (F-122-F).
    """
    question = args.strip()
    if not question:
        return InteractiveOutcome(
            message="Usage: /btw <your question> — Ask a quick side question without interrupting your main session.",
            display="user",
        )

    # F-122-I: record this /btw invocation in the persistent use-count
    # before any further work. Counting happens at the command layer so
    # every UI path (REPL/TUI/headless) flows through the same gate; the
    # counter increments regardless of whether the side question itself
    # succeeds — a failed fork is still a real user attempt to use /btw.
    increment_btw_use_count(question=question)

    # Build cache-safe params (cached or fallback)
    params = await _build_cache_safe_params(context)
    if params is None:
        return InteractiveOutcome(
            message="⚠️ Cannot build side-question context. Please ask directly in the main session.",
            display="user",
        )

    # Execute side question
    try:
        result = await run_side_question(question, params)
    except Exception as e:
        logger.exception("Side question failed")
        return InteractiveOutcome(
            message=f"⚠️ Side question failed: {e}",
            display="user",
        )

    if result.response:
        message = f"💡 {result.response}"
        return InteractiveOutcome(
            message=message,
            display="user",
            scrollable=_should_render_scrollable(message),
        )
    return InteractiveOutcome(
        message="⚠️ Side question returned no answer. Please ask directly in the main session.",
        display="user",
    )


# ---------------------------------------------------------------------------
# CacheSafeParams builder
# ---------------------------------------------------------------------------


async def _build_cache_safe_params(
    context: CommandContext,
) -> CacheSafeParams | None:
    """Build CacheSafeParams for the side question.

    Preferred path: read from the global cache populated by the main loop
    after each turn (prompt-cache hit).

    Fallback path: rebuild from scratch (prompt-cache miss, more expensive
    but functionally complete).
    """
    saved = get_last_cache_safe_params()
    if saved is not None:
        return saved

    # Fallback: rebuild from scratch
    tool_context = getattr(context, "tool_context", None)
    if tool_context is None:
        logger.warning("btw: no tool_context available for fallback rebuild")
        return None

    # Rebuild system prompt
    system_prompt = ""
    try:
        from clawcodex_ext.context_system.prompt_assembly import build_full_system_prompt  # pylint: disable=no-name-in-module

        system_prompt = build_full_system_prompt(
            cwd=str(context.cwd),
            tools=getattr(tool_context.options, "tools", None) if tool_context.options else None,
        )
    except Exception:
        logger.exception("btw: failed to rebuild system prompt")

    # Rebuild user / system context
    user_context: dict[str, str] | None = None
    system_context: dict[str, str] | None = None
    try:
        from clawcodex_ext.context_system.prompt_assembly import (  # pylint: disable=no-name-in-module
            get_system_context,
            get_user_context,
        )

        user_context = await get_user_context(cwd=str(context.cwd))
        system_context = await get_system_context(cwd=str(context.cwd))
    except Exception:
        logger.exception("btw: failed to rebuild context")

    # 确保 _active_provider 已设置（回退路径时可能尚未通过 query 循环设置）
    active_provider = getattr(tool_context, "_active_provider", None)
    if active_provider is None:
        provider = getattr(context, "provider", None)
        if provider is not None:
            try:
                setattr(tool_context, "_active_provider", provider)
            except Exception:
                logger.exception("btw: failed to set _active_provider on tool_context")
        else:
            logger.warning("btw: no provider available on context or tool_context — forked agent may fail")

    # Fork context messages: use conversation messages if available
    fork_context_messages: list[Any] = []
    conversation = getattr(context, "conversation", None)
    if conversation is not None and hasattr(conversation, "messages"):
        fork_context_messages = list(conversation.messages)

    return CacheSafeParams(
        system_prompt=system_prompt,
        user_context=user_context,
        system_context=system_context,
        tool_use_context=tool_context,
        fork_context_messages=fork_context_messages,
    )


# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------


class BtwCommand(InteractiveCommand):
    """/btw — side question without polluting the main conversation."""

    async def run(self, args: str, context: CommandContext) -> InteractiveOutcome:
        return await btw_command_run(args, context)


BTW_COMMAND = BtwCommand(
    name="btw",
    description="Ask a quick side question without interrupting your main session",
    argument_hint="<your question>",
)
