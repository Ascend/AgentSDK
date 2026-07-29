"""Test helper: sync ``run_agent_loop`` for legacy tests.

Pre-consolidation tests inherited from the ``run_agent_loop`` era
call through this sync wrapper so they don't need to thread their
own ``asyncio.run`` + system-prompt resolution + message persistence
at every call site.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.outputStyles import resolve_output_style
from src.query.agent_loop_compat import build_effective_system_prompt, run_query_as_agent_loop
from src.tool_system.renderers import AgentLoopResult


def run_agent_loop(
    conversation: Any,
    provider: Any,
    tool_registry: Any,
    tool_context: Any,
    max_turns: int = 20,
    stream: bool = True,  # kept for signature compat; adapter always streams
    verbose: bool = False,  # kept for signature compat; ignored
    on_event: Any = None,
    on_text_chunk: Any = None,
    cancel_signal: Any = None,
) -> AgentLoopResult:
    """Sync wrapper around :func:`run_query_as_agent_loop` matching the
    legacy ``run_agent_loop`` signature.

    Mirrors the departed ``clawcodex_ext.query.agent_loop_compat.run_query_as_agent_loop_sync``:
    resolves output style, builds the effective system prompt, persists
    messages back into the conversation, and returns a legacy
    ``AgentLoopResult``.
    """
    style_prompt = resolve_output_style(
        getattr(tool_context, "output_style_name", None),
        getattr(tool_context, "output_style_dir", None),
    ).prompt
    effective_system_prompt = build_effective_system_prompt(
        style_prompt,
        tool_context,
    )

    def _persist(msg: Any) -> None:
        try:
            conversation.add_existing_message(msg)
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Failed to persist message into conversation: role=%s",
                getattr(msg, "role", "?"),
            )
            raise

    compat_result = asyncio.run(
        run_query_as_agent_loop(
            initial_messages=list(conversation.messages),
            provider=provider,
            tool_registry=tool_registry,
            tool_context=tool_context,
            system_prompt=effective_system_prompt,
            max_turns=max_turns,
            on_event=on_event,
            on_text_chunk=on_text_chunk,
            on_message=_persist,
            cancel_signal=cancel_signal,
        )
    )
    return AgentLoopResult(
        response_text=compat_result.response_text,
        usage=(compat_result.usage if compat_result.num_turns > 0 else None),
        num_turns=compat_result.num_turns,
    )
