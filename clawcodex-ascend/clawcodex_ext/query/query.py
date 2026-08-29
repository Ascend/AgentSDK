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

# pylint: disable=too-many-lines,ungrouped-imports
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable

from clawcodex_ext.types.messages import (
    AssistantMessage,
    AttachmentMessage,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
    Message,
    REJECT_MESSAGE,
    SystemMessage,
    UserMessage,
    normalize_messages_for_api,
)
from clawcodex_ext.types.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from clawcodex_ext.tool_system.build_tool import Tools, find_tool_by_name
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.protocol import ToolCall
from clawcodex_ext.tool_system.registry import ToolRegistry
from clawcodex_ext.tool_system.tool_search import filter_tools_for_request
from clawcodex_ext.utils.abort_controller import AbortController, AbortError  # pylint: disable=no-name-in-module
from clawcodex_ext.utils.image_validation import ImageSizeError  # pylint: disable=no-name-in-module
from clawcodex_ext.providers.base import BaseProvider, ChatResponse
from clawcodex_ext.providers.anthropic_provider import AnthropicProvider
from clawcodex_ext.providers.minimax_provider import MinimaxProvider
from clawcodex_ext.context_system.cache_boundary import SYSTEM_PROMPT_DYNAMIC_BOUNDARY  # pylint: disable=no-name-in-module
from src.models.model import canonical_model_name
from src.settings.settings import get_settings
from src.utils.advisor import (
    ADVISOR_BETA_HEADER,
    ADVISOR_MODE_CLIENT_SIDE,
    ADVISOR_MODE_INACTIVE,
    ADVISOR_MODE_SERVER_SIDE,
    ADVISOR_TOOL_INSTRUCTIONS,
    build_advisor_tool_schema,
    build_client_advisor_tool_schema,
    decide_advisor_mode,
    strip_advisor_blocks,
)

from clawcodex_ext.query.config import build_query_config
from src.query.continuation_nudge import (
    MAX_CONTINUATION_NUDGES,
    NUDGE_MESSAGE,
    detect_continuation_signal,
)
from clawcodex_ext.query.hook_registry import call_hooks, LoopHookPhase
from clawcodex_ext.query.stop_hooks import StopHookResult, handle_stop_hooks_streaming
from clawcodex_ext.query.token_budget import ContinueDecision, check_token_budget, create_budget_tracker
from src.query.tool_failure_loop_guard import (
    create_tool_failure_loop_guard_state,
    update_tool_failure_loop_guard,
)

# Late-binding imports — moved from the original mid-file location (after
# _call_hooks_if_enabled) to satisfy ruff E402 (import not at top of file).
from clawcodex_ext.query.recovery_strategies import RecoveryContext, find_recovery_strategies
from clawcodex_ext.query.transitions import (
    QueryState,
    Terminal,
    TerminalHolder,
    Transition,
    set_terminal,
)
from clawcodex_ext.services.compact.pipeline import (  # noqa: E402
    PipelineConfig,
    run_compression_pipeline,
)
from clawcodex_ext.services.api.retry import CannotRetryError  # noqa: E402
from clawcodex_ext.utils.token_estimation import rough_token_count_estimation_for_messages  # noqa: E402  # pylint: disable=no-name-in-module

# ---------------------------------------------------------------------------
# Feature-gate helper for hook phases.
# Wraps hook invocation so that HOOK_PRE_LLM / HOOK_POST_LLM flags can
# disable the entire hook pipeline without changing the call sites.
# ---------------------------------------------------------------------------


def _call_hooks_if_enabled(phase: LoopHookPhase, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Call hooks only when the corresponding feature gate is enabled.

    The ``HOOK_PRE_LLM`` flag gates ``pre_llm`` hooks,
    ``HOOK_POST_LLM`` gates ``post_llm`` hooks.  All other phases
    remain unconditionally enabled.
    """
    from clawcodex_ext.feature_gate import guarded_is_enabled  # pylint: disable=no-name-in-module

    gate_map = {
        "pre_llm": "HOOK_PRE_LLM",
        "post_llm": "HOOK_POST_LLM",
    }
    gate_name = gate_map.get(phase)
    if gate_name and not guarded_is_enabled(gate_name):
        # Feature gate is disabled — return args unchanged.
        return args
    return call_hooks(phase, *args, **kwargs)


logger = logging.getLogger(__name__)

ESCALATED_MAX_TOKENS = 64_000

# ch04 round-3 G3 — 529/overloaded retry lane. TS withRetry.ts: general
# budget DEFAULT_MAX_RETRIES=10 with a 529-specific MAX_529_RETRIES=3
# counter that triggers model-fallback or the external-user bail
# (:346-385); the port adopts the bail posture (3 retries then the
# existing model_error path). Gated to foreground sources only
# (withRetry.ts:62-90) -- background lanes (compact, session_memory)
# bail immediately to avoid gateway amplification.
MAX_529_RETRIES = 3
# ch04 round-4 GAP B — general retryable budget (429/5xx/connection/
# timeout), TS DEFAULT_MAX_RETRIES (withRetry.ts:52).
DEFAULT_MAX_RETRIES = 10
RETRY_BASE_DELAY_SECONDS = 0.5
# Narrower than TS's FOREGROUND_529_RETRY_SOURCES (agents/compact/
# side_question etc., withRetry.ts:62-90) -- minimal posture; widen to
# agent sources when subagent traffic matters. 'sdk' added in ch05
# round-4 alongside the headless query_source relabel ('repl_main_thread'
# -> 'sdk'): TS's set includes 'sdk' (withRetry.ts:67), so headless keeps
# the yield-based retry lane after the relabel.
FOREGROUND_529_RETRY_SOURCES = frozenset({"repl_main_thread", "sdk"})
MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3
# The downstream empty-end-turn recovery is deliberately tighter than the
# upstream action-intent continuation lane: two retries avoid an extra model
# call when a provider repeatedly emits empty end_turn responses.
EMPTY_END_TURN_NUDGE_LIMIT = 2
CONTINUATION_NUDGE_MESSAGE = (
    "Your previous response ended without any visible text or a tool call. "
    "The task may still be in progress. Continue from the latest tool result. "
    "If the user's request is complete, briefly summarize the completed work."
)
PROMPT_TOO_LONG_ERROR_MESSAGE = (
    "Your conversation is too long. Please use /compact to reduce context size, or start a new conversation."
)


def _mark_proactive_context_blocked_on_error(error: BaseException) -> None:
    """Block proactive ticks after provider/API errors.

    Proactive ticks are generated without new user intent, so a provider
    error can otherwise become a tight tick -> error -> tick loop. This
    hook is best-effort and must never change normal query error handling.
    """
    try:
        from clawcodex_ext.services.proactive import get_default_controller

        ctrl = get_default_controller()
        if ctrl.is_active():
            ctrl.set_context_blocked(True)
    except Exception:
        logger.debug("failed to mark proactive context blocked", exc_info=True)


@dataclass
class QueryParams:
    messages: list[Message]
    # WI-1.1: ``system_prompt`` accepts either the legacy ``str`` shape
    # (joined sections, no cache_control markers) OR the block-list shape
    # ``list[dict]`` produced by ``build_full_system_prompt_blocks``. The
    # block-list shape is what engages Anthropic's prompt cache via
    # ``cache_control: {type: 'ephemeral'}`` markers; the str shape is
    # retained for backward compat with callers that pass a custom prompt.
    system_prompt: str | list[dict[str, Any]]
    tools: Tools
    tool_registry: ToolRegistry
    tool_use_context: ToolContext
    provider: BaseProvider
    abort_controller: AbortController
    # Per-request model override. When set, this model is used instead of the
    # provider's default model. Passed through to ``call_kwargs`` so providers
    # that support per-call model switching (e.g. ``_get_model(**kwargs)``) can
    # pick it up.
    model: str | None = None
    query_source: str = "repl_main_thread"
    # Optional per-turn output budget used by the upstream continuation lane.
    token_budget: int | None = None
    max_output_tokens_override: int | None = None
    max_turns: int | None = None
    # ch04 round-4 GAP B — model to switch to after MAX_529_RETRIES
    # consecutive overloaded errors (TS QueryParams.fallbackModel,
    # query.ts:276; --fallback-model in headless). Session-sticky switch
    # via provider.model; never persisted to settings.
    fallback_model: str | None = None
    # ch05 round-4 note — N/A-by-architecture on the LIVE paths: the
    # adapter's build_effective_system_prompt already injects CLAWCODEX.md
    # ("## Project Instructions") and the date via the SYSTEM prompt, so
    # wiring TS's prependUserContext there would double-inject. Only the
    # test-only engine path uses the message-based mechanism
    # (prepend_user_context at engine.py:221). One mechanism per surface.
    user_context: dict[str, str] | None = None
    system_context: dict[str, str] | None = None
    pipeline_config: PipelineConfig | None = None
    # P84-F: when set, the compression pipeline uses this engine for
    # layer 4 (context collapse) instead of the legacy read-only
    # ``ContextCollapseStore.project_view`` path. The engine owns its
    # own store and records new commits when the trigger fires. Typed
    # as ``Any`` to avoid an import cycle with the engine module.
    collapse_engine: Any | None = None
    # Ch5/F-followup: live streaming text callback. When set, the
    # provider's chat_stream_response receives this callback so each
    # SSE text-delta drives the UI in real time. Critical for the
    # TUI/headless live-stream UX after the F.2/F.3 migration to this
    # loop — without it, callers see the entire response materialize
    # at once after the model turn completes. The callback can also
    # raise AbortError from inside the SDK's stream context to tear
    # down the HTTP socket on ESC.
    on_text_chunk: Callable[[str], None] | None = None
    # Live thinking deltas (separate channel from on_text_chunk) for the TUI's
    # streaming thinking view. None → thinking isn't surfaced live.
    on_thinking_chunk: Callable[[str], None] | None = None

    # Extended thinking ("adaptive" mode) — opt the model into a private
    # reasoning scratchpad before producing its visible answer. Defaults
    # to ``None`` which auto-enables on Anthropic Claude 4.x models
    # (the only family the API supports it on) and stays off elsewhere.
    # Pass ``False`` to force-disable (e.g. for determinism in tests).
    # Mirrors the TS reference which always passes
    # ``thinking: {type: "adaptive"}`` on these models.
    extended_thinking: bool | None = None
    # Output-effort hint forwarded as ``output_config.effort`` on models
    # that support it. ``None`` = auto: the persisted ``settings.effort``
    # when set, else the parameter is OMITTED and the API applies its model
    # default (TS parity). An explicit value ("low" | "medium" | "high" |
    # "xhigh" | "max") wins over settings — precedence mirrors TS
    # ``parseEffortValue(options.effort) ?? getInitialEffortSetting()``
    # (main.tsx:2631). "xhigh" degrades to "high" on models that reject it
    # (see :func:`resolve_thinking_effort`). Only sent when extended
    # thinking is active.
    thinking_effort: str | None = None
    # Persist-only callback for meta attachments injected by the canonical
    # loop. Live surfaces wire this to their Conversation; unlike
    # ``on_message`` it does not render the reminder as user-authored text.
    on_attachment: Callable[[Message], None] | None = None
    # False for notification/goal maintenance turns that do not represent a
    # fresh user interaction. Such turns must not consume reminder cadence.
    task_reminder_enabled: bool = True


@dataclass
class StreamEvent:
    type: str
    data: Any = None


def _is_prompt_too_long_message(msg: Message) -> bool:
    if not isinstance(msg, AssistantMessage):
        return False
    if not hasattr(msg, "_api_error"):
        return False
    return getattr(msg, "_api_error", None) == "prompt_too_long"


def _create_user_message(content: str, *, is_meta: bool = False) -> UserMessage:
    return UserMessage(
        content=content,
        isMeta=is_meta,
    )


def _assistant_response_is_empty(messages: list[AssistantMessage]) -> bool:
    """Return whether a model turn contains no visible output or tool block."""
    if not messages:
        return True
    for message in messages:
        content = message.content
        if isinstance(content, str):
            if content.strip():
                return False
        elif content:
            return False
    return True


def _create_assistant_api_error_message(
    content: str,
    *,
    error: str | None = None,
) -> AssistantMessage:
    msg = AssistantMessage(content=content, isApiErrorMessage=True)
    msg._api_error = error  # type: ignore[attr-defined]
    return msg


def _terminal_error_for_api_message(
    message: Message | None,
    *,
    error_type: str | None = None,
) -> RuntimeError:
    """Turn a terminal API-error message back into a propagated error."""

    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, list):
        details = " ".join(block.text for block in content if isinstance(block, TextBlock)).strip()
    else:
        details = str(content).strip()
    if error_type == "max_output_tokens":
        prefix = "output token recovery exhausted"
    else:
        prefix = error_type or getattr(message, "_api_error", None) or "model API error"
    return RuntimeError(f"{prefix}: {details}" if details else str(prefix))


def _create_user_interruption_message(*, tool_use: bool = False) -> UserMessage:
    content = INTERRUPT_MESSAGE_FOR_TOOL_USE if tool_use else INTERRUPT_MESSAGE
    return UserMessage(content=content, isMeta=True)


def _create_max_turns_attachment(max_turns: int, turn_count: int) -> SystemMessage:
    return SystemMessage(
        content=f"Reached maximum number of turns ({max_turns})",
        subtype="max_turns_reached",
    )


async def _fire_post_sampling_hooks(
    assistant_messages: list[AssistantMessage],
    provider: Any,
    tool_use_context: Any,
) -> None:
    """Run configured ``PostSampling`` hooks for a completed model stream.

    ch01 round-4 WI-2 — restores the dependency-graph edge "Query Loop
    fires Hooks" for the one event TS fires from query.ts itself
    (``executePostSamplingHooks``, query.ts:1079-1089). Reads the global
    ``AsyncHookRegistry`` (populated at startup by
    ``bootstrap_hook_config_manager``); with no ``PostSampling`` hooks
    configured this returns after one in-memory lookup.

    Deviation from TS, deliberate: awaited inline instead of
    fire-and-forget. The agent-server drives each turn with
    ``asyncio.run(...)``, so a task created on the loop's final iteration
    would be cancelled at teardown — losing the hook on exactly the
    turn-final response. Inline await trades stream/hook overlap for a
    completion guarantee.

    Hook failures are logged and swallowed — a hook must never kill the
    turn (TS parity: logError + continue). ``additional_contexts`` from
    hook results are debug-logged and dropped: TS post-sampling hooks
    return void, so there are no injection semantics to mirror; a uniform
    injection lane across events is the ch12 round-4 subject.
    """
    if not assistant_messages:
        return
    try:
        from clawcodex_ext.hooks.post_sampling_hooks import run_post_sampling_hooks  # pylint: disable=no-name-in-module
        from clawcodex_ext.hooks.trust_gate import should_skip_hook_due_to_trust  # pylint: disable=no-name-in-module

        last = assistant_messages[-1]
        results = await run_post_sampling_hooks(
            model=(getattr(last, "model", None) or getattr(provider, "model", None) or ""),
            usage=getattr(last, "usage", None) or {},
            stop_reason=getattr(last, "stop_reason", None),
            # Same trust rule as the tool-hook lane: untrusted workspace →
            # policy hooks only (trust_gate WI-0.2).
            untrusted_workspace=should_skip_hook_due_to_trust(tool_use_context),
        )
        for entry in results:
            injected = entry.get("injected_messages")
            if injected:
                logger.debug(
                    "PostSampling hook additional_contexts dropped (no injection lane yet — ch12): %d block(s)",
                    len(injected),
                )
    except (asyncio.CancelledError, AbortError):  # pylint: disable=try-except-raise
        # User intent wins — AbortError subclasses Exception, so without the
        # explicit re-raise the blanket handler below would swallow it.
        raise
    except Exception:  # noqa: BLE001 — hook failure must never kill the turn
        logger.error("PostSampling hook execution failed", exc_info=True)


def _drain_pending_user_messages(tool_use_context: Any) -> list[UserMessage]:
    """Drain the running agent's ``pending_messages`` inbox, if any.

    Chapter-10 / Chunk D / WI-3.3 hook. The TS implementation drains at
    the tool-round boundary inside the agent's run loop; the Python
    equivalent is here, between `tool_results` and the next API call,
    where the chapter's "messages arrive between tool rounds, not
    mid-execution" contract holds.

    No-op when:
    * The context has no ``agent_id`` (top-level / non-runtime-task agents).
    * The context has no ``runtime_tasks`` registry (test fixtures
      that didn't construct a real ToolContext).
    * The agent's entry isn't a ``LocalAgentTaskState`` (defensive —
      a future task type that runs through the same query loop).
    * The inbox is empty.

    Returns the drained messages as a list of fresh ``UserMessage``
    objects, which the caller appends to the next turn's prompt.
    """
    agent_id = getattr(tool_use_context, "agent_id", None)
    runtime = getattr(tool_use_context, "runtime_tasks", None)
    if not agent_id or runtime is None:
        return []
    # Local import to avoid pulling the tasks package into the query
    # module's import graph at startup; this hook only fires when an
    # agent_id is set, so the tasks module will already be loaded.
    try:
        from src.tasks.local_agent import (
            LocalAgentTaskState,
            drain_pending_messages,
        )
    except ImportError:
        return []
    state = runtime.get(agent_id)
    if not isinstance(state, LocalAgentTaskState):
        return []
    drained = drain_pending_messages(agent_id, runtime)
    if not drained:
        return []
    return [UserMessage(content=text) for text in drained]


def _yield_missing_tool_result_blocks(
    assistant_messages: list[AssistantMessage],
    error_message: str,
) -> list[UserMessage]:
    results: list[UserMessage] = []
    for assistant_msg in assistant_messages:
        content = assistant_msg.content
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, ToolUseBlock):
                results.append(
                    UserMessage(
                        content=[
                            ToolResultBlock(
                                tool_use_id=block.id,
                                content=error_message,
                                is_error=True,
                            )
                        ],
                    )
                )
    return results


def _get_context_window(provider: BaseProvider) -> int:
    """Ch5/B.4 — return the model's context-window size in tokens.

    Defaults to 200_000 (the Claude family default) when the provider
    doesn't expose a real integer ``context_window`` attribute.

    The ``isinstance(_, int)`` guard handles MagicMock test fixtures:
    a MagicMock returns another mock for unset attributes, so a naive
    ``getattr(..., None) or default`` would short-circuit past the
    default. Be explicit about the type so the loop is robust to test
    doubles.
    """
    cw = getattr(provider, "context_window", None)
    if isinstance(cw, int) and cw > 0:
        return cw
    model = getattr(provider, "model", None)
    if isinstance(model, str) and model:
        try:
            from src.models.context import get_context_window_for_model

            return get_context_window_for_model(model, base_url=getattr(provider, "base_url", None))
        except Exception:
            logger.debug("model context-window resolution failed", exc_info=True)
    return 200_000


def _is_withheld_max_output_tokens(msg: Message | None) -> bool:
    if msg is None:
        return False
    if not isinstance(msg, AssistantMessage):
        return False
    return getattr(msg, "_api_error", None) == "max_output_tokens"


def _is_withheld_prompt_too_long(msg: Message | None) -> bool:
    """Ch5/B.1 — mirrors TS ``isWithheldPromptTooLong`` at query.ts:877.

    Returns True for an assistant message that carries an
    ``_api_error == "prompt_too_long"`` tag. The query loop checks this
    after streaming to suppress the message from the yield stream until
    PTL recovery (reactive_compact) has been attempted (B.2).
    """
    if msg is None:
        return False
    if not isinstance(msg, AssistantMessage):
        return False
    return getattr(msg, "_api_error", None) == "prompt_too_long"


def _is_withheld_media_size(msg: Message | None) -> bool:
    """Ch5/B.1 — mirrors TS ``isWithheldMediaSizeError`` at query.ts:892.

    Returns True for an assistant message that carries an
    ``_api_error == "media_size"`` tag.
    """
    if msg is None:
        return False
    if not isinstance(msg, AssistantMessage):
        return False
    return getattr(msg, "_api_error", None) == "media_size"


def _is_hook_stopped_continuation(msg: Message | None) -> bool:
    """Ch5/round2 — mirrors TS attachment scan at query.ts:1540-1545.

    Returns True for an ``AttachmentMessage`` whose ``attachments`` list
    contains an attachment with ``type == 'hook_stopped_continuation'``.
    The query loop checks this after the tool batch completes; if any
    tool_result carries the marker, the loop exits with
    ``Terminal(reason='hook_stopped')`` instead of advancing to the
    next turn — mirroring TS at query.ts:1698-1701.

    The attachment is produced by ``PreToolUse`` / ``PostToolUse`` hooks
    that set ``prevent_continuation`` — see
    ``src/services/tool_execution/tool_execution.py:362-372`` and
    ``src/services/tool_execution/tool_hooks.py:185-195``. The current
    production dispatch path (``_run_tools_partitioned`` → registry
    dispatch) does NOT route through those producers — that wiring is
    the C7 architectural-unification gap, deferred. Landing the terminal
    mapping here means the contract is in place the moment any future
    change wires hook-aware dispatch into the loop.

    Local import of ``AttachmentMessage`` matches the pattern at
    ``_drain_pending_user_messages`` — keeps the top-level import list
    lean and avoids a circular-import risk with future ``types.messages``
    refactors.
    """
    if msg is None:
        return False
    if not isinstance(msg, AttachmentMessage):
        return False
    attachments = getattr(msg, "attachments", None) or []
    for att in attachments:
        if isinstance(att, dict) and att.get("type") == "hook_stopped_continuation":
            return True
    return False


_THINKING_ELIGIBLE_MODEL_PATTERN = re.compile(
    r"claude-(?:sonnet|opus|haiku|fable)-(?:4-\d+|[5-9]\b|\d{2,})",
    re.IGNORECASE,
)


def _model_supports_extended_thinking(model: str | None) -> bool:
    """True iff the model supports extended thinking at all (any type).

    Thinking was introduced with the Claude 4 series — the Anthropic API
    rejects any ``thinking`` param on 3.x and earlier. On the first-party
    endpoint every Claude 4+ model (Sonnet, Opus, AND Haiku 4.5) supports
    thinking (TS ``modelSupportsThinking``, thinking.ts:117-148, firstParty
    branch). Detection is by name pattern so unreleased snapshots
    (e.g. ``claude-opus-4-7-20260201``) opt in automatically.

    NOTE: supporting thinking does NOT imply supporting the *adaptive*
    thinking type — that's the narrower :func:`_model_supports_adaptive_thinking`
    allowlist. Models here that aren't adaptive-capable take a token budget
    instead (see the caller in ``_call_model_sync``).
    """
    if not model:
        return False
    return bool(_THINKING_ELIGIBLE_MODEL_PATTERN.search(model))


def _model_supports_adaptive_thinking(model: str | None) -> bool:
    """True iff the model supports the *adaptive* thinking type.

    Only a subset of Claude 4 models accept ``thinking={"type": "adaptive"}``;
    the rest support thinking only with an explicit ``budget_tokens``. Sending
    adaptive to a non-adaptive model is rejected with HTTP 400 "adaptive
    thinking is not supported on this model". Allowlist ported from TS
    ``modelSupportsAdaptiveThinking`` (thinking.ts:152-169): Opus 4.6/4.7
    and Sonnet 4.6; extended with Opus 4.8 and Fable 5, where adaptive is
    the ONLY accepted thinking config — the budget fallback below would be
    a hard 400 on them (``budget_tokens`` is removed on 4.7+; Fable 5 also
    rejects ``{"type": "disabled"}``, and accepts adaptive or an omitted
    param). Substring match mirrors the reference's ``.includes()`` so
    dated snapshots (``claude-sonnet-4-6-20250929``) match.
    """
    if not model:
        return False
    m = model.lower()
    return "fable-5" in m or "opus-4-8" in m or "opus-4-7" in m or "opus-4-6" in m or "sonnet-4-6" in m


def _model_supports_effort(model: str | None) -> bool:
    """True iff the model accepts ``output_config={"effort": ...}``.

    Narrower than thinking support — Opus 4.6/4.8, Sonnet 4.6, and Fable 5
    (TS ``modelSupportsEffort``, effort.ts:32-51, plus the 4.8/Fable
    additions where effort is GA). Sending effort to a model that doesn't
    support it is rejected, so the caller gates on this independently of
    the thinking type.
    """
    if not model:
        return False
    m = model.lower()
    return "fable-5" in m or "opus-4-8" in m or "opus-4-6" in m or "sonnet-4-6" in m


def _model_supports_xhigh_effort(model: str | None) -> bool:
    """True iff the model accepts ``output_config={"effort": "xhigh"}``.

    The Claude effort ladder is low|medium|high|xhigh|max, but ``xhigh``
    acceptance is model-dependent. Wire-probed on the first-party API
    2026-07-18: opus-4-8 accepts it; sonnet-4-6 and opus-4-6 reject it with
    400 "This model does not support effort level 'xhigh'. Supported
    levels: high, low, max, medium" — note the same error confirms ``max``
    is broadly accepted, so only ``xhigh`` needs gating (the vendored TS
    snapshot's opus-4-6-only ``modelSupportsMaxEffort`` predates this).
    Fable 5 is included by analogy with opus-4-8 (Claude Code exposes the
    full ladder on it; unprobed — subscription plans don't carry it).
    """
    if not model:
        return False
    m = model.lower()
    return "opus-4-8" in m or "fable-5" in m


# Kept in sync with settings.constants.VALID_EFFORT_VALUES (minus the empty
# "auto" sentinel) — a dedicated test asserts the two ladders match.
VALID_THINKING_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def resolve_thinking_effort(explicit: str | None, model: str | None) -> str | None:
    """Effective ``output_config.effort`` value for one request, or ``None``
    to omit the parameter entirely.

    Precedence mirrors TS main.tsx:2631 ``parseEffortValue(options.effort)
    ?? getInitialEffortSetting()``: an explicit per-session value (the
    ``--effort`` flag via ``QueryParams.thinking_effort``) wins over the
    persisted ``settings.effort``. When NEITHER is set the parameter is
    omitted, matching TS ``configureEffortParams`` (services/api/claude.ts:
    449-463 — ``effortValue === undefined`` sends no ``outputConfig``) and
    letting the API apply its own model default (per TS
    ``getDisplayedEffortLevel`` that default is "high"; the pre-wiring
    Python hardcode of "medium" was a divergence, not the wire default).
    ``"xhigh"`` degrades to ``"high"`` on models outside
    :func:`_model_supports_xhigh_effort`'s allowlist rather than 400ing the
    request; ``"max"`` passes through everywhere effort-capable (see the
    probe notes on the allowlist helper).
    """
    value = (explicit or "").strip().lower()
    if value not in VALID_THINKING_EFFORT_LEVELS:
        # Local import: settings is read at the wire boundary only, keeping
        # QueryParams construction sites free of hidden global reads.
        try:
            value = (get_settings().effort or "").strip().lower()
        except Exception:  # noqa: BLE001 — settings must never break a request
            value = ""
    if value not in VALID_THINKING_EFFORT_LEVELS:
        return None
    if value == "xhigh" and not _model_supports_xhigh_effort(model):
        logger.debug("effort xhigh not supported on %s; sending high instead", model)
        return "high"
    return value


def _is_overloaded_error(e: Exception) -> bool:
    """Anthropic 529 / overloaded_error classification (duck-typed so
    test fakes and other providers' shapes participate).
    """
    status = getattr(e, "status_code", None)
    if status == 529:
        return True
    text = str(e).lower()
    return "overloaded_error" in text or "overloaded" in text


def _retry_after_seconds(e: Exception, default: float) -> float:
    """Honor a Retry-After header when the SDK exposes response headers."""
    response = getattr(e, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after")
        if raw:
            try:
                value = float(raw)
                if 0 < value <= 60:
                    return value
            except (TypeError, ValueError):
                pass  # The candidate value is invalid; continue with the existing fallback.
    return default


async def _fire_stop_failure_hooks(last_message: Any, tool_use_context: Any) -> None:
    """Dispatch StopFailure hooks at the error-exit paths (ch05 G1).

    TS fires fire-and-forget at query.ts:1256/:1263/:1347; the port
    awaits (terminal paths; latency bounded by the hook timeout) and
    never raises.
    """
    try:
        from clawcodex_ext.hooks.hook_executor import execute_stop_failure_hooks  # pylint: disable=no-name-in-module

        async for _result in execute_stop_failure_hooks(last_message, tool_use_context):
            pass
    except Exception:
        logger.debug("StopFailure hook dispatch failed", exc_info=True)


# ch04 round-4 GAP C.2 — required whenever a system block carries
# cache_control.scope == "global" (TS constants/betas.ts:17-18).
PROMPT_CACHING_SCOPE_BETA_HEADER = "prompt-caching-scope-2026-01-05"


def _strip_block_metadata(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copies of ``blocks`` ready for the 1P Anthropic wire: the
    internal ``_cache_scope`` key removed AND the dynamic-boundary marker
    block dropped.

    The Anthropic provider forwards system blocks verbatim to its SDK
    (``call_kwargs["system"] = system_prompt``), so the inert ``_cache_scope``
    tag emitted by the prompt assembler must be stripped before it lands on a
    1P request. ch04 round-4 GAP C: the boundary is a SPLIT SIGNAL, never
    wire content — TS's splitSysPromptPrefix skips it (utils/api.ts:388,424);
    before this fix the literal ``__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`` text
    block went out on every 1P request. Non-Anthropic providers flatten via
    ``_split_system_prompt_blocks``, which already drops it.
    """
    cleaned: list[dict[str, Any]] = []
    for blk in blocks:
        if isinstance(blk, dict) and blk.get("type") == "text" and blk.get("text") == SYSTEM_PROMPT_DYNAMIC_BOUNDARY:
            continue
        if isinstance(blk, dict) and "_cache_scope" in blk:
            blk = {k: v for k, v in blk.items() if k != "_cache_scope"}
        cleaned.append(blk)
    return cleaned


def _split_system_prompt_blocks(blocks: list[dict[str, Any]], *, relocate_request_scope: bool) -> tuple[str, str]:
    """Flatten system-prompt blocks for an OpenAI-compatible provider.

    Returns ``(system_text, volatile_tail_text)``.

    The ``__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`` marker is always dropped (it is
    an Anthropic cache-only signal that would be unintelligible prose to other
    models).

    When ``relocate_request_scope`` is True (DeepSeek only), blocks tagged
    ``_cache_scope == "request"`` — the env section, the auto-memory section
    (which embeds the mutable ``MEMORY.md`` body), plan-mode / non-interactive
    / tool-restriction sections — are routed into ``volatile_tail_text`` so the
    caller can place them AFTER the conversation history. That keeps the
    ``system + tools + history`` prefix byte-stable across turns, so DeepSeek's
    automatic prefix cache covers it even when memory or the environment
    changes mid-session.

    When False (every other provider), the tail is empty and all non-boundary
    text is concatenated into ``system_text`` — byte-for-byte the prior
    behaviour.
    """
    stable: list[str] = []
    volatile: list[str] = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        text = blk.get("text")
        if not text or text == SYSTEM_PROMPT_DYNAMIC_BOUNDARY:
            continue
        if relocate_request_scope and blk.get("_cache_scope") == "request":
            volatile.append(str(text))
        else:
            stable.append(str(text))
    return "\n\n".join(stable), "\n\n".join(volatile)


def _append_session_context_tail(api_messages: list[dict[str, Any]], tail_text: str) -> list[dict[str, Any]]:
    """Place the DeepSeek relocated-volatile sections AFTER the conversation.

    Wrapped as ambient ``<system-reminder>`` context. Merged into the trailing
    user message when that is a plain user turn (string content, or a
    content-block list with no ``tool_result``) so the wire keeps strict
    user/assistant alternation. Otherwise — e.g. the turn ends in a tool result
    (which converts to ``role:tool`` on the wire) — appended as a standalone
    trailing user message, which lands correctly after the tool messages.
    Returns a new list; ``api_messages`` is not mutated.
    """
    return {}


def _skill_effort_kwargs(
    provider: BaseProvider,
    effort: str | int | None,
) -> dict[str, Any]:
    """Translate a skill effort override only for supporting providers."""
    if effort is None or effort == "":
        return {}
    module_name = type(provider).__module__.lower()
    class_name = type(provider).__name__.lower()
    if "anthropic_provider" in module_name and "minimax" not in class_name:
        return {"output_config": {"effort": effort}}
    if "kimi_provider" in module_name:
        return {"reasoning_config": {"effort": effort}}
    if any(marker in module_name for marker in ("openai", "litellm", "grok")):
        return {"reasoning_effort": effort}
    mapper = getattr(provider, "skill_effort_kwargs", None)
    if callable(mapper):
        try:
            return dict(mapper(effort) or {})
        except Exception:
            logger.warning("provider skill effort mapping failed", exc_info=True)
            return {}
    logger.warning(
        "Provider %s does not support skill effort override %r; continuing without it",
        type(provider).__name__,
        effort,
    )
    return {}


async def _call_model_sync(
    *,
    provider: BaseProvider,
    messages: list[Message],
    system_prompt: str,
    tools: Tools,
    model: str | None = None,
    effort_override: str | int | None = None,
    max_output_tokens_override: int | None = None,
    abort_signal: Any = None,
    on_text_chunk: Callable[[str], None] | None = None,
    on_thinking_chunk: Callable[[str], None] | None = None,
    extended_thinking: bool | None = None,
    thinking_effort: str | None = None,
    sdk_max_retries: int | None = None,
) -> tuple[list[AssistantMessage], list[ToolUseBlock]]:
    # pylint: disable=too-many-nested-blocks
    # Advisor activation decision. Three outcomes:
    #
    # * SERVER_SIDE: 1P Anthropic provider + opus-4-6/sonnet-4-6 main
    #   loop + valid server-side advisor target. Carries the beta
    #   header and the ``advisor_20260301`` schema; the API runs the
    #   reviewer model server-side. Optimal path — single roundtrip,
    #   cache-friendly.
    # * CLIENT_SIDE: any provider, any tool-calling main loop, any
    #   advisor model that routes to a known provider. Registers
    #   ``advisor`` as a regular client-side tool; the dispatcher
    #   (``src/tool_system/tools/advisor.py``) makes a separate API
    #   call to the configured advisor model. Two roundtrips but
    #   provider-agnostic.
    # * INACTIVE: no advisor on this request (env-disabled, no
    #   advisor_model set, or no path can be resolved).
    #
    # The full decision table lives in ``decide_advisor_mode``. Any
    # exception during the predicate degrades to INACTIVE rather than
    # killing the turn (critic M1 from the original advisor PR).
    main_loop_model = getattr(provider, "model", "") or ""
    advisor_mode = ADVISOR_MODE_INACTIVE
    advisor_model_normalized: str | None = None
    try:
        settings = get_settings()
        configured = (getattr(settings, "advisor_model", "") or "").strip()
        configured_provider = (getattr(settings, "advisor_provider", "") or "").strip()
        force_client = bool(getattr(settings, "advisor_client_mode", False))
        if configured:
            candidate = canonical_model_name(configured)
            advisor_mode = decide_advisor_mode(
                provider,
                main_loop_model,
                candidate,
                force_client_mode=force_client,
                advisor_provider=configured_provider,
            )
            if advisor_mode != ADVISOR_MODE_INACTIVE:
                advisor_model_normalized = candidate
    except Exception:
        logger.exception("Advisor activation check failed; treating advisor as inactive")
        advisor_mode = ADVISOR_MODE_INACTIVE
        advisor_model_normalized = None

    api_messages = normalize_messages_for_api(messages)

    # Server-side advisor blocks (``server_tool_use(name=advisor)`` and
    # ``advisor_tool_result``) require the beta header on every request
    # that carries them — the API 400s otherwise. Strip from history on
    # any request that won't send the header.
    #
    # In CLIENT_SIDE mode the advisor surfaces as regular
    # ``tool_use``/``tool_result`` blocks, which pass through normal
    # message handling untouched. Only the SERVER_SIDE shape is gated
    # by the header, so stripping is keyed off "current request carries
    # the beta" — which is exactly SERVER_SIDE-and-only-SERVER_SIDE.
    if advisor_mode != ADVISOR_MODE_SERVER_SIDE:
        api_messages = strip_advisor_blocks(api_messages)

    # --- Diagnostic tracing ---
    _diag = os.environ.get("CLAWCODEX_DEBUG", "").lower() in ("1", "true", "yes")
    if _diag:
        _total_chars = sum(
            len(m.get("content", ""))
            if isinstance(m.get("content"), str)
            else sum(len(str(b)) for b in m.get("content", []))
            for m in api_messages
        )
        if isinstance(system_prompt, str):
            sys_desc = f"{len(system_prompt)} chars"
        else:
            sys_total_chars = sum(len(blk.get("text", "")) for blk in system_prompt if isinstance(blk, dict))
            sys_desc = f"{len(system_prompt)} blocks, {sys_total_chars} chars"
        logger.warning(
            "[DIAG] _call_model_sync: %d api_messages, ~%d chars, system_prompt=%s, %d tools",
            len(api_messages),
            _total_chars,
            sys_desc,
            len(list(tools)),
        )
        for i, m in enumerate(api_messages):
            role = m.get("role", "?")
            c = m.get("content", "")
            if isinstance(c, str):
                clen = len(c)
                logger.warning("[DIAG]   msg[%d] role=%s content_len=%d", i, role, clen)
            else:
                block_types = []
                for b in c:
                    if isinstance(b, dict):
                        bt = b.get("type", "?")
                        if bt == "tool_use":
                            block_types.append("tool_use")
                        elif bt == "tool_result":
                            block_types.append("tool_result")
                        else:
                            block_types.append(bt)
                    else:
                        block_types.append(str(type(b).__name__))
                logger.warning("[DIAG]   msg[%d] role=%s  blocks=%s", i, role, block_types)
    _t0 = time.monotonic()
    request_model = model or getattr(provider, "model", None) or ""
    request_tools = filter_tools_for_request(tools, request_model, api_messages)
    tool_schemas = []
    for tool in request_tools:
        # Filter out internal/hidden tools (is_enabled=False) so they
        # don't leak into the API tools[] alongside the advisor schema
        # we append below. Some callers pass an unfiltered tool list
        # from ``registry.list_tools()``; this guard keeps the API
        # from receiving duplicate names. ``getattr`` with default
        # True keeps test fakes that don't implement is_enabled working.
        is_enabled_fn = getattr(tool, "is_enabled", None)
        if callable(is_enabled_fn) and not is_enabled_fn():
            continue
        tool_schemas.append(
            {
                "name": tool.name,
                "description": tool.prompt(),
                "input_schema": dict(tool.input_schema),
            }
        )

    # Append the advisor schema AFTER the regular tools so the
    # ``cache_control`` marker (which conventionally lives on the last
    # cached tool — the final entry in ``tool_schemas`` before this
    # append) stays in place. If we prepended or interleaved, toggling
    # /advisor would shift the marker and bust the prompt cache. Mirrors
    # TS claude.ts:1411-1421 explicitly.
    #
    # The schema shape differs by mode: server-side carries the dated
    # ``advisor_20260301`` discriminator + model field; client-side is
    # a regular tool_use schema with no params, routed through the
    # tool registry's AdvisorTool.
    if advisor_mode == ADVISOR_MODE_SERVER_SIDE:
        tool_schemas.append(build_advisor_tool_schema(advisor_model_normalized))
    elif advisor_mode == ADVISOR_MODE_CLIENT_SIDE:
        tool_schemas.append(build_client_advisor_tool_schema())

    call_kwargs: dict[str, Any] = {"tools": tool_schemas}

    # Per-request model override. When ``model`` is set, pass it through
    # so the provider's ``_get_model(**kwargs)`` picks it up instead of the
    # provider's default model.
    if model:
        call_kwargs["model"] = model
    call_kwargs.update(_skill_effort_kwargs(provider, effort_override))

    if advisor_mode == ADVISOR_MODE_SERVER_SIDE:
        # Opt into the server-side advisor tool. ``betas`` lives outside
        # ``extra_headers`` because the SDK auto-converts it into the
        # ``anthropic-beta`` header AND filters out 3P-incompatible
        # entries on Bedrock/Vertex transports. setdefault-append so it
        # composes with the global-cache-scope beta (GAP C.2) below.
        call_kwargs.setdefault("betas", []).append(ADVISOR_BETA_HEADER)
        # CLIENT_SIDE deliberately does NOT set betas — 3P endpoints
        # reject the advisor beta, and 1P-with-force-client doesn't
        # need it because the advisor schema is a regular tool here.

    is_anthropic = isinstance(provider, (AnthropicProvider, MinimaxProvider))
    advisor_instructions_active = advisor_mode != ADVISOR_MODE_INACTIVE
    if is_anthropic:
        # Forward whatever shape the engine produced — str or list[dict].
        # The SDK's ``system`` param accepts ``Union[str, Iterable[TextBlockParam]]``;
        # cache_control markers on blocks engage server-side prompt caching.
        #
        # When the advisor is active (server OR client side), append
        # ``ADVISOR_TOOL_INSTRUCTIONS`` AFTER the existing system prompt
        # blocks. Mirrors TS claude.ts:1395 — the advisor instructions
        # come AFTER the cached system blocks, so they land in the
        # request-scope partition and toggling /advisor doesn't churn
        # the cached prefix. The instruction text is provider-agnostic
        # (tells the model "use the advisor tool"), so it works for
        # both the server-side ``server_tool_use`` invocation and the
        # client-side regular ``tool_use`` invocation.
        if advisor_instructions_active:
            if isinstance(system_prompt, list):
                system_prompt = list(system_prompt) + [{"type": "text", "text": ADVISOR_TOOL_INSTRUCTIONS}]
            elif isinstance(system_prompt, str):
                system_prompt = (
                    f"{system_prompt}\n\n{ADVISOR_TOOL_INSTRUCTIONS}" if system_prompt else ADVISOR_TOOL_INSTRUCTIONS
                )
            else:
                # Defensive: the upstream contract is
                # ``str | list[dict[str, Any]]``. A future caller that
                # passes something else (e.g. None, a TextBlock object)
                # silently loses the instructions if we don't warn.
                logger.warning(
                    "Advisor active but system_prompt has unexpected type %s — ADVISOR_TOOL_INSTRUCTIONS NOT injected",
                    type(system_prompt).__name__,
                )
        # Strip the inert ``_cache_scope`` metadata + the dynamic-boundary
        # marker block; Anthropic forwards the system list verbatim to its
        # SDK, so neither must reach the 1P wire (GAP C).
        if isinstance(system_prompt, list):
            system_prompt = _strip_block_metadata(system_prompt)
            # ch04 round-4 GAP C.2 — scope:'global' requires the
            # prompt-caching-scope beta (TS claude.ts:1231-1236 pushes
            # PROMPT_CACHING_SCOPE_BETA_HEADER whenever global cache is
            # active). The scope only appears when the operator enabled
            # CLAUDE_CODE_ENABLE_GLOBAL_CACHE_SCOPE; without the header the
            # API rejects/ignores the field.
            if any(
                isinstance(blk, dict)
                and isinstance(blk.get("cache_control"), dict)
                and blk["cache_control"].get("scope") == "global"
                for blk in system_prompt
            ):
                call_kwargs.setdefault("betas", []).append(PROMPT_CACHING_SCOPE_BETA_HEADER)
        call_kwargs["system"] = system_prompt
    else:
        # Non-Anthropic providers (OpenAI-compat, GLM, etc.) consume the
        # system prompt as a single string injected as a ``system`` message.
        # Flatten the block-list shape to a string by concatenating block text;
        # cache_control markers don't apply to these providers anyway.
        #
        # Critically, FILTER OUT the dynamic-boundary marker block. The
        # literal ``__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`` is a cache-only
        # signal for the Anthropic backend; emitting it as raw text into
        # a non-Anthropic system prompt embeds an unintelligible token in
        # the prose that may confuse those models.
        #
        # DeepSeek-only: route the per-request-volatile (REQUEST-scope)
        # sections to a trailing tail so the system prefix stays byte-stable
        # for DeepSeek's automatic prefix cache. ``relocate_request_scope`` is
        # False for every other provider, so ``flattened`` keeps every
        # non-boundary block (byte-for-byte the prior behaviour) and
        # ``volatile_tail`` is "".
        # ``is True`` (not ``bool(...)``): every real provider sets the flag to a
        # literal ``True``/``False`` (see ``BaseProvider.is_deepseek``), so this
        # is identical in production — but it also makes a bare test double (e.g.
        # ``MagicMock()``, whose auto-attributes are truthy) fall through to the
        # non-relocating path instead of silently exercising DeepSeek relocation.
        if isinstance(system_prompt, list):
            flattened = "\n\n".join(
                str(blk.get("text", ""))
                for blk in system_prompt
                if isinstance(blk, dict) and blk.get("text") and blk.get("text") != SYSTEM_PROMPT_DYNAMIC_BOUNDARY
            )
        else:
            flattened = system_prompt
        # CLIENT_SIDE on a 3P provider: append the advisor instructions
        # to the flattened system prompt so the model knows how + when
        # to invoke the ``advisor`` tool. (Server-side instructions
        # only land on 1P, handled by the is_anthropic branch above.)
        if advisor_instructions_active and advisor_mode == ADVISOR_MODE_CLIENT_SIDE:
            if flattened:
                flattened = f"{flattened}\n\n{ADVISOR_TOOL_INSTRUCTIONS}"
            else:
                flattened = ADVISOR_TOOL_INSTRUCTIONS
        api_messages = [{"role": "system", "content": flattened}, *api_messages]

    if is_anthropic:
        # ch04 round-3 G0: resolve max_tokens on EVERY Anthropic request
        # (override → CLAUDE_CODE_MAX_OUTPUT_TOKENS env → per-model
        # table). Previously only the override branch set it and normal
        # requests silently went out at the provider-default 4096.
        # Non-Anthropic providers keep their override-only behavior —
        # they send NO max_tokens today (the provider-API default
        # applies) and capping them at the table default would be a
        # silent behavior change outside this gap's evidence.
        from src.models.context import resolve_max_output_tokens

        call_kwargs["max_tokens"] = resolve_max_output_tokens(
            max_output_tokens_override,
            getattr(provider, "model", None),
            base_url=getattr(provider, "base_url", None),
        )
    elif max_output_tokens_override is not None:
        call_kwargs["max_tokens"] = max_output_tokens_override

    # Extended thinking (Claude 4.x family). Forwarded straight through
    # the provider's kwargs pass-through to client.messages.stream(
    # thinking=..., output_config=...). Off-API on older Claude versions
    # and on non-Anthropic providers, so guarded by both. ``None`` =
    # auto-enable; ``True`` / ``False`` = caller override.
    #
    # The adaptive-vs-budget selection mirrors TS claude.ts:1612-1640:
    # models that support thinking AND the adaptive type get
    # ``{type: "adaptive"}``; models that support thinking but NOT adaptive
    # (Sonnet 4.5, Haiku 4.5, older Opus 4.x, …) get an explicit
    # ``{type: "enabled", budget_tokens: N}`` instead — sending adaptive to
    # them is a hard 400 ("adaptive thinking is not supported on this
    # model"). ``output_config.effort`` is gated separately and even more
    # narrowly (Opus 4.6 / Sonnet 4.6 only). Getting either gate wrong
    # breaks every request on the affected model — including the whole
    # subscription/OAuth path, which is how this surfaced.
    if extended_thinking is not False and is_anthropic:
        provider_model = getattr(provider, "model", None) or call_kwargs.get("model")
        if extended_thinking is True or _model_supports_extended_thinking(provider_model):
            if _model_supports_adaptive_thinking(provider_model):
                call_kwargs["thinking"] = {"type": "adaptive"}
            else:
                # Non-adaptive (incl. an explicit ``extended_thinking=True``
                # on an unknown alias): budget thinking. This is the safe
                # direction — budget works on any thinking-capable model,
                # whereas adaptive 400s where unsupported; TS defaults an
                # unknown first-party alias to adaptive (thinking.ts:178-183),
                # we deliberately prefer budget. budget_tokens must be ≥ 1024
                # (API minimum) and strictly less than max_tokens, which is
                # already resolved above for every Anthropic request. TS uses
                # maxOutputTokens-1 (claude.ts:1628-1635); we clamp identically.
                _max_tok = int(call_kwargs.get("max_tokens") or 0)
                if _max_tok > 1024:
                    call_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": max(1024, _max_tok - 1),
                    }
                else:
                    # No budget in [1024, max_tokens) fits — omit thinking
                    # rather than send an invalid request (max_tokens this
                    # small only happens via an explicit tiny override).
                    logger.debug(
                        "thinking omitted: max_tokens=%s too small for a valid budget on non-adaptive model %s",
                        _max_tok,
                        provider_model,
                    )
            if _model_supports_effort(provider_model):
                resolved_effort = resolve_thinking_effort(thinking_effort, provider_model)
                # None = nothing requested anywhere — omit the parameter so
                # the API applies its own model default (TS parity; see
                # resolve_thinking_effort).
                if resolved_effort is not None:
                    call_kwargs["output_config"] = {"effort": resolved_effort}

    # TS callModel() uses SSE streaming for faster first-byte latency and
    # progressive text display.  Use chat_stream_response() which streams
    # internally and reassembles the full ChatResponse.  Fall back to the
    # synchronous chat() if the provider doesn't support structured streaming.
    if _diag:
        logger.warning("[DIAG] _call_model_sync: calling provider (streaming)...")

    def _do_provider_call() -> ChatResponse:
        # Keep the downstream request throttle in the same blocking lane as
        # the provider call. Background agents therefore wait off-loop while
        # the middleware's process-wide lock preserves the configured spacing.
        try:
            from clawcodex_ext.query.query_middleware import enforce_request_delay

            enforce_request_delay()
        except ImportError:  # nosec B110
            pass  # The optional integration is unavailable; continue with the built-in path.

        # ``abort_signal`` reaches the provider so a tripped controller can
        # close the streaming response immediately. Preserve both downstream
        # live callbacks and the compatibility retries for older providers.
        try:
            if on_text_chunk is not None:
                try:
                    stream_response = provider.chat_stream_response(
                        api_messages,
                        on_text_chunk=on_text_chunk,
                        on_thinking_chunk=on_thinking_chunk,
                        abort_signal=abort_signal,
                        **call_kwargs,
                    )
                except TypeError:
                    # Provider doesn't accept on_thinking_chunk — retry text-only,
                    # then kwargless, so text streaming never regresses.
                    try:
                        stream_response = provider.chat_stream_response(
                            api_messages,
                            on_text_chunk=on_text_chunk,
                            abort_signal=abort_signal,
                            **call_kwargs,
                        )
                    except TypeError:
                        stream_response = provider.chat_stream_response(
                            api_messages,
                            abort_signal=abort_signal,
                            **call_kwargs,
                        )
            else:
                stream_response = provider.chat_stream_response(
                    api_messages,
                    abort_signal=abort_signal,
                    **call_kwargs,
                )
            if not isinstance(stream_response, ChatResponse):
                raise NotImplementedError("provider returned no structured stream response")
            return stream_response
        except (NotImplementedError, AttributeError):
            if _diag:
                logger.warning("[DIAG] _call_model_sync: streaming not supported, falling back to chat()")
            fallback_response = provider.chat(api_messages, **call_kwargs)
            # Emulate streaming chunks from the final text when the
            # caller asked for live streaming (on_text_chunk set) but
            # the provider only supports plain chat(). Legacy
            # ``agent_loop._emit_text_chunks`` did this so the headless
            # ``--include-partial-messages`` flag and CLI live-text UX
            # work uniformly across providers — without it, a
            # ``stream=True`` caller paired with a non-streaming
            # provider would see the entire response materialize at
            # once after the model finishes.
            if on_text_chunk is not None and fallback_response.content:
                from clawcodex_ext.tool_system.renderers import emit_text_chunks

                emit_text_chunks(on_text_chunk, fallback_response.content)
            return fallback_response

    try:
        # Provider streaming is synchronous. Run background/workflow calls off
        # the event loop so parallel agents do not serialize. Keep either live
        # callback on the event-loop thread because UI consumers are not
        # required to be thread-safe.
        if on_text_chunk is None and on_thinking_chunk is None:
            response = await asyncio.to_thread(_do_provider_call)
        else:
            response = _do_provider_call()
    except AbortError:
        # User-initiated cancel — propagate so the query loop's
        # ``except AbortError: pass`` boundary unwinds to the
        # post-API abort-check block. We do NOT route this through
        # the error-message classification below: a future addition
        # to those substring checks could accidentally match an abort
        # reason and convert the cancel into a model-error reply.
        raise
    except ImageSizeError as e:
        # Client-side pre-API validation tripped (BaseProvider._prepare_messages).
        # Surface as a media_size error with the same classification the
        # server-side guard uses, so the reactive-compact recovery path
        # (Ch5/B.2) treats them identically.
        err_msg = _create_assistant_api_error_message(
            f"Media too large: {e}",
            error="media_size",
        )
        err_msg._api_error = "media_size"  # type: ignore[attr-defined]
        return [err_msg], []
    except Exception as e:
        if _diag:
            logger.warning("[DIAG] _call_model_sync: EXCEPTION after %.1fs: %s", time.monotonic() - _t0, e)
        error_str = str(e)
        if "prompt is too long" in error_str.lower() or "prompt_too_long" in error_str.lower():
            err_msg = _create_assistant_api_error_message(
                PROMPT_TOO_LONG_ERROR_MESSAGE,
                error="prompt_too_long",
            )
            err_msg._api_error = "prompt_too_long"  # type: ignore[attr-defined]
            return [err_msg], []

        # Rate-limit handling — delegated to extensions so the upstream
        # query loop stays free of orchestrator-specific error policies.
        try:
            from clawcodex_ext.query.query_middleware import handle_rate_limit_error

            rate_limit_msg = handle_rate_limit_error(error_str)
            if rate_limit_msg is not None:
                return [rate_limit_msg], []
        except ImportError:
            # Fallback: inline the check if the extension is not available.
            if "429" in error_str or "rate_limit" in error_str.lower():
                err_msg = _create_assistant_api_error_message(
                    "Rate limit exceeded. Please wait and retry.",
                    error="rate_limit",
                )
                err_msg._api_error = "rate_limit"  # type: ignore[attr-defined]
                return [err_msg], []

        if "max_tokens" in error_str.lower() or "max_output_tokens" in error_str.lower():
            err_msg = _create_assistant_api_error_message(
                "Output token limit reached.",
                error="max_output_tokens",
            )
            err_msg._api_error = "max_output_tokens"  # type: ignore[attr-defined]
            return [err_msg], []

        # Ch5/B.1 — tag media-size errors so the loop can withhold them
        # and (in B.2) route through reactive-compact recovery. Mirrors
        # TS `isWithheldMediaSizeError` at query.ts:892. `is_media_size_error`
        # expects a str (substring match), so pass error_str explicitly.
        from clawcodex_ext.services.api.errors import is_media_size_error

        if is_media_size_error(error_str):
            err_msg = _create_assistant_api_error_message(
                f"Media too large: {error_str}",
                error="media_size",
            )
            err_msg._api_error = "media_size"  # type: ignore[attr-defined]
            return [err_msg], []

        # Model-capability rejection: the selected model has zero image
        # support, so the request can never succeed as long as the image
        # stays in conversation context. Tag the error so the engine
        # strips images from history (see QueryEngine.submit_message)
        # and surfaces a clear user-facing message instead of the raw
        # provider 404. The user-facing wording shape follows TS's
        # friendly error messages (e.g. getPdfInvalidErrorMessage at
        # typescript/src/services/api/errors.ts) but the case itself
        # is Python-new — TS has no dedicated handler for this
        # capability rejection, so don't grep TS for an analog branch.
        # See IMAGE_UNSUPPORTED_ERROR_MESSAGE in services/api/errors.py
        # for the longer rationale.
        from clawcodex_ext.services.api.errors import (
            IMAGE_UNSUPPORTED_ERROR_MESSAGE,
            is_image_unsupported_error,
        )

        if is_image_unsupported_error(error_str):
            err_msg = _create_assistant_api_error_message(
                IMAGE_UNSUPPORTED_ERROR_MESSAGE,
                error="image_unsupported",
            )
            err_msg._api_error = "image_unsupported"  # type: ignore[attr-defined]
            # Preserve raw provider wording so a future bug report
            # ("the fix didn't work") has the actual 404 payload to
            # debug against, instead of just the friendly message.
            # Mirrors TS's ``errorDetails: error.message`` at
            # typescript/src/services/api/errors.ts:752.
            err_msg.errorDetails = error_str
            return [err_msg], []

        raise

    assistant_blocks: list[Any] = []
    tool_use_blocks: list[ToolUseBlock] = []

    # Signed Anthropic thinking must precede the visible text/tool blocks in
    # exactly the order returned. Keeping it in history preserves reasoning
    # state across tool turns and exposes it to stream-json/session logging.
    if response.thinking_blocks:
        from clawcodex_ext.types.content_blocks import content_block_from_dict

        for raw in response.thinking_blocks:
            assistant_blocks.append(content_block_from_dict(raw))

    if response.content:
        assistant_blocks.append(TextBlock(text=response.content))

    if response.tool_uses:
        for tu in response.tool_uses:
            block = ToolUseBlock(
                id=tu["id"],
                name=tu["name"],
                input=tu["input"],
            )
            assistant_blocks.append(block)
            tool_use_blocks.append(block)

    # Preserve advisor server-tool blocks as passthrough dicts so the
    # next turn can replay them to the API as a matched use/result pair.
    # ``normalize_messages_for_api`` round-trips dict blocks unchanged
    # (via ``content_block_to_dict``), and ``ensure_tool_result_pairing``
    # treats the advisor pair as a self-contained server-side
    # use/result on the assistant message (already paired in-message).
    # Stripping happens centrally in this function when ``advisor_active``
    # is False on a future turn.
    if response.raw_content_blocks:
        for raw in response.raw_content_blocks:
            assistant_blocks.append(dict(raw))

    stop_reason = response.finish_reason or "end_turn"

    # Capture the real LLM inference duration outside the _diag gate
    # so the value always lands on the AssistantMessage (and from
    # there on the JSONL transcript). The downstream session-analysis
    # viewer reads this to render the real LLM bar width; without
    # it, the viewer would have to fall back to a heuristic.
    _inference_ms = int((time.monotonic() - _t0) * 1000)

    if _diag:
        _text_len = len(response.content) if response.content else 0
        _tool_count = len(response.tool_uses) if response.tool_uses else 0
        logger.warning(
            "[DIAG] _call_model_sync: response in %.1fs  text=%d chars  tools=%d  finish=%s  usage=%s",
            _inference_ms / 1000.0,
            _text_len,
            _tool_count,
            stop_reason,
            response.usage,
        )

    # ch04 round-3 G1: the cost-accumulation head (TS addToTotalSessionCost,
    # claude.ts:2270-2275). Streaming and the watchdog chat() fallback
    # converge here, so every main-loop response is counted exactly once.
    # Empty usage (a stream whose final-message read failed) records zeros.
    try:
        from clawcodex_ext.bootstrap.state import add_to_total_duration_state  # pylint: disable=no-name-in-module
        from src.cost_tracker import record_api_usage

        record_api_usage(
            getattr(response, "model", None) or getattr(provider, "model", "unknown"),
            response.usage,
        )
        # The original records per-request API duration alongside cost
        # (addToTotalDuration beside addToTotalSessionCost); this feeds
        # /cost's "Total duration (API)". This layer can't split retries:
        # provider-internal ones are inside the span (so both counters get
        # the same value), while a raise-then-retry by our caller records
        # nothing for the failed attempt (the original's including-retries
        # counter would) — a slight undercount, accepted.
        _api_ms = int((time.monotonic() - _t0) * 1000)
        add_to_total_duration_state(_api_ms, _api_ms)
    except Exception:
        logger.debug("cost recording failed", exc_info=True)

    assistant_msg = AssistantMessage(
        content=assistant_blocks if assistant_blocks else "",
        stop_reason=stop_reason,
        # TS assistant messages carry the responding model (query.ts message
        # assembly); consumers like the PostSampling hook payload and cost
        # attribution read it. Same fallback chain as record_api_usage above.
        model=getattr(response, "model", None) or getattr(provider, "model", None),
        usage=response.usage,
        duration_ms=_inference_ms,
    )
    if response.reasoning_content:
        # Preserve provider thinking metadata for follow-up turns.
        assistant_msg.reasoning_content = response.reasoning_content  # type: ignore[attr-defined]

    if stop_reason == "max_tokens":
        assistant_msg._api_error = "max_output_tokens"  # type: ignore[attr-defined]
        assistant_msg.isApiErrorMessage = False

    return [assistant_msg], tool_use_blocks


# Max tools to run in parallel (TS default: 10, configurable via env var).
# Reads CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY (the name documented in
# chapter 7 and used in the TS reference). Falls back to the legacy
# CLAWCODEX_MAX_TOOL_USE_CONCURRENCY with a DeprecationWarning so
# users learn to migrate.
def _resolve_max_tool_use_concurrency() -> int:
    canonical = os.environ.get("CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY")
    if canonical is not None:
        return int(canonical)
    legacy = os.environ.get("CLAWCODEX_MAX_TOOL_USE_CONCURRENCY")
    if legacy is not None:
        import warnings

        warnings.warn(
            "CLAWCODEX_MAX_TOOL_USE_CONCURRENCY is deprecated; use CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(legacy)
    return 10


MAX_TOOL_USE_CONCURRENCY = _resolve_max_tool_use_concurrency()


@dataclass
class _ToolBatch:
    """A batch of tool_use blocks with the same concurrency classification."""

    is_concurrent_safe: bool
    blocks: list[ToolUseBlock]


def _partition_tool_calls(
    tool_use_blocks: list[ToolUseBlock],
    tools: Tools,
) -> list[_ToolBatch]:
    """Partition tool calls into batches per TS partitionToolCalls().

    Consecutive ConcurrencySafe tools are grouped for parallel execution.
    Non-safe tools each get their own exclusive batch.

    Mirrors TS: evaluates isConcurrencySafe per-call with actual tool input
    (not a static lookup), so e.g. read-only Bash commands can be parallel.
    """
    batches: list[_ToolBatch] = []
    for block in tool_use_blocks:
        tool = find_tool_by_name(tools, block.name)
        try:
            is_safe = bool(tool.is_concurrency_safe(block.input)) if tool else False
        except Exception:
            is_safe = False
        if batches and is_safe and batches[-1].is_concurrent_safe:
            batches[-1].blocks.append(block)
        else:
            batches.append(_ToolBatch(is_concurrent_safe=is_safe, blocks=[block]))
    return batches


def _is_user_cancelled_abort(tool_use_context: ToolContext) -> bool:
    """True iff the abort signal fired with a user-initiated reason.

    ``sibling_error`` is the streaming-executor's parallel-tool cascade
    and is NOT a user-rejected signal — surfacing REJECT_MESSAGE for it
    would mask the real underlying failure. Every other abort reason in
    the Python runtime (``user_interrupt`` from ESC, ``interrupt`` held
    in reserve for TS parity) is collapsed into the user-cancelled
    bucket here.

    Divergence vs TS: ``StreamingToolExecutor.ts:219-229`` treats
    ``'interrupt'`` (user typed mid-stream) and ``'user_interrupted'``
    (ESC) differently — for ``'interrupt'`` it only synthesizes
    REJECT_MESSAGE on tools whose ``interruptBehavior() === 'cancel'``.
    Python today emits neither ``'interrupt'`` nor any per-tool
    ``interrupt_behavior`` override on the production path, so the
    collapsed check is sound. If a future change wires up
    ``'interrupt'`` as a real reason, the per-tool gate must land first.
    """
    ctrl = tool_use_context.abort_controller
    if not ctrl.signal.aborted:
        return False
    return ctrl.signal.reason != "sibling_error"


def _build_user_cancelled_result(tool_use_id: str) -> UserMessage:
    """Synthetic tool_result returned when the user aborts mid-run.

    The bash tool's interrupted path emits
    ``<error>Command was aborted before completion</error>``, which the
    model reads as a generic command failure — on the next turn it tends
    to retry the command rather than honour the user's cancel. Replacing
    the tool_result with REJECT_MESSAGE makes the cancellation
    unambiguous. Mirrors
    ``typescript/src/services/tools/StreamingToolExecutor.ts:153-205``
    (``createSyntheticErrorMessage`` for ``user_interrupted``).
    """
    return UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id=tool_use_id,
                content=REJECT_MESSAGE,
                is_error=True,
            )
        ],
    )


def _build_unavailable_tool_result(tool_use_id: str, tool_name: str) -> UserMessage:
    """Reject a tool call that was not exposed to the current agent.

    Agent runs receive a filtered ``tools`` list but share the host registry.
    The registry is intentionally broader because other contexts reuse it;
    therefore the execution boundary must enforce the per-agent view before
    dispatch instead of trusting the model to call only advertised tools.
    """

    return UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id=tool_use_id,
                content=f"Error: Tool {tool_name!r} is not available in this agent context.",
                is_error=True,
            )
        ],
    )


def _dispatch_single_tool(
    block: ToolUseBlock,
    tool_registry: ToolRegistry,
    tool_use_context: ToolContext,
    tools: Tools | None = None,
) -> tuple[UserMessage, list[UserMessage]]:
    """Dispatch a single tool and return ``(primary, extras)``.

    ``primary`` is always the tool_result UserMessage. ``extras`` is any
    supplemental ``new_messages`` the tool returned (e.g. the image
    dimensions metadata user message). Callers MUST emit all primaries
    before any extras across a multi-tool batch — otherwise tool_result
    pairing in ``ensure_tool_result_pairing`` mis-attributes the missing
    pair and injects a synthetic error placeholder.

    Routes through ``process_tool_result_block`` (mirrors TS Step 11 of
    the execution pipeline at ``processToolResultBlock``) so the per-tool
    persistence threshold AND the WI-5.1 per-message aggregate budget
    both engage on the production path. The running aggregate is held on
    ``tool_use_context.tool_result_chars_so_far`` (reset at the top of
    each per-turn loop in :func:`query`).
    """
    # Pre-tool gate: ESC may trip after the model picked this tool but
    # before we entered dispatch (e.g. between the post-streaming abort
    # check and the head of the partition loop). Hand back the
    # synthetic result instead of running the tool. Mirrors the initial-
    # abort branch in ``StreamingToolExecutor.collectResults``
    # (typescript/src/services/tools/StreamingToolExecutor.ts:278-292).
    if _is_user_cancelled_abort(tool_use_context):
        return _build_user_cancelled_result(block.id), []

    # ``tools=None`` is the legacy unrestricted path. An explicit list,
    # including an empty list, is the authoritative capability boundary for
    # this query/agent and must be checked before the shared registry runs.
    selected_tool = find_tool_by_name(tools, block.name) if tools is not None else None
    if tools is not None and selected_tool is None:
        return _build_unavailable_tool_result(block.id, block.name), []

    try:
        call = ToolCall(
            name=block.name,
            input=block.input,
            tool_use_id=block.id,
        )
        _t0 = time.monotonic()
        result = tool_registry.dispatch(call, tool_use_context)
        _dur_ms = (time.monotonic() - _t0) * 1000

        # Record tool and skill usage without blocking the hot path.
        try:
            from clawcodex_ext.tool_stats import record_tool, record_skill  # pylint: disable=no-name-in-module

            _ok = not getattr(result, "is_error", False)
            _err = None
            if not _ok:
                _err = str(getattr(result, "output", {}).get("error", "")) or None

            if block.name == "Skill":
                _skill_name = block.input.get("skill", "") or "unknown"
                record_skill(_skill_name, dur_ms=_dur_ms, ok=_ok, error=_err)
            else:
                record_tool(block.name, dur_ms=_dur_ms, ok=_ok, error=_err)
        except Exception:
            logger.debug("Failed to record tool stats", exc_info=True)

        # Post-tool override: bash's interrupted payload reads as a
        # generic failure; replace it so the resume turn sees an
        # unambiguous "user rejected" signal. Mirrors TS at
        # ``StreamingToolExecutor.ts:332-345``.
        if _is_user_cancelled_abort(tool_use_context):
            return _build_user_cancelled_result(block.id), []

        tool = selected_tool
        metadata: dict[str, Any] = {}
        if isinstance(result.output, dict):
            metadata["tool_output"] = result.output

        if tool is not None:
            # WI-5.1: route through ``process_tool_result_block`` so the
            # 200K per-message aggregate cap is enforced. Without this
            # call the production REPL ran ``map_result_to_api`` directly
            # and never engaged the gate (critic B2).
            #
            # The read-decide-write on ``tool_result_chars_so_far`` is
            # serialized via ``_aggregate_lock`` because ``_run_tools_partitioned``
            # dispatches concurrency-safe tools (Read/Grep/Glob) via
            # ``asyncio.to_thread`` (critic B6). The decision MUST be
            # made on a fresh snapshot of the counter — otherwise N
            # threads racing the read all see 0, all decide "under cap"
            # and the cap is silently bypassed. ``process_tool_result_block``
            # is called inside the critical section: for small blocks
            # under threshold it just returns the block (no I/O); the
            # rare persist-to-disk path runs while serialized but those
            # are at most O(1) per turn (typically <5%).
            from clawcodex_ext.services.tool_execution.tool_result_persistence import (
                compute_block_chars,
                process_tool_result_block,
                resolve_tool_results_dir,
            )

            tool_results_dir = resolve_tool_results_dir(tool_use_context)
            with tool_use_context._aggregate_lock:
                aggregate_so_far = tool_use_context.tool_result_chars_so_far
                api_block = process_tool_result_block(
                    tool,
                    result.output,
                    block.id,
                    tool_results_dir=tool_results_dir,
                    aggregate_chars_so_far=aggregate_so_far,
                )
                # Update the running aggregate AFTER the block is
                # finalized (post-persistence, so the wrapper message
                # size is what counts toward the budget — not the
                # original 200K output).
                tool_use_context.tool_result_chars_so_far += compute_block_chars(api_block)
            raw_content = api_block.get("content", "")
            # Preserve list-of-content-blocks shape so multimodal tool_results
            # (e.g. Read's image content blocks, bash data:image/... captures)
            # reach the API as proper image/document blocks instead of being
            # JSON-stringified into text. The Anthropic API rejects images
            # delivered as text and the model just sees JSON gibberish in the
            # tool_result content otherwise. ``ToolResultBlock.content`` and
            # ``content_block_to_dict`` already handle list shape end-to-end
            # (see content_blocks.py:159-173).
            if isinstance(raw_content, (str, list)):
                result_content: str | list[Any] = raw_content
            else:
                result_content = str(raw_content)
        elif isinstance(result.output, str):
            result_content = result.output
        elif isinstance(result.output, dict):
            result_content = json.dumps(result.output, ensure_ascii=False)
        else:
            result_content = str(result.output)

        result_msg = UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id=block.id,
                    content=result_content,
                    is_error=result.is_error,
                    metadata=metadata,
                )
            ],
        )
        # Collect any supplemental messages the tool produced (e.g. the
        # FileReadTool image-dimensions metadata user message and PDF
        # page-image blocks). Mirrors TS messages flow where the Read
        # tool's createUserMessage({isMeta:true}) returns are pushed
        # into the conversation alongside the tool_result. Returned
        # separately so callers can emit all primaries before any extras
        # (see docstring).
        extras: list[UserMessage] = []
        if result.new_messages:
            for msg in result.new_messages:
                if isinstance(msg, UserMessage):
                    extras.append(msg)
                elif isinstance(msg, dict):
                    # Defensive: accept the raw-dict form too.
                    extras.append(
                        UserMessage(
                            content=msg.get("content", ""),
                            isMeta=bool(msg.get("isMeta", False)),
                        )
                    )
        return result_msg, extras
    except AbortError as abort_err:
        # Two contracts to satisfy at once:
        #
        # 1. **tool_use/tool_result pairing must stay intact** — every
        #    emitted tool_use needs a paired tool_result or the next
        #    API call 400s on the orphan. Returning a tool_result
        #    (not raising) preserves the pair. Pinned by
        #    ``tests/test_esc_reject_message_dispatch.py``.
        # 2. **No follow-up API turn after AbortError** — the loop
        #    must NOT issue another model call. Pinned by
        #    ``test_agent_loop_does_not_swallow_abort_error_as_tool_error``.
        #
        # Reconcile both: when AbortError surfaces from a tool and
        # the user-cancel signal isn't already tripped, trip it. The
        # post-tool gate downstream sees the signal aborted, sets
        # terminal=aborted_tools, and the adapter raises AbortError
        # to the caller. The conversation ends well-formed (tool_use
        # has its tool_result) AND the loop exits without another
        # API turn. Critic-flagged on Stage 4 review.
        if _is_user_cancelled_abort(tool_use_context):
            return _build_user_cancelled_result(block.id), []
        ctrl = getattr(tool_use_context, "abort_controller", None)
        if ctrl is not None:
            try:
                ctrl.abort("tool_raised_abort_error")
            except Exception as exc:  # nosec B110
                logger.warning("Failed to notify the abort controller (%s)", type(exc).__name__)
        return UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id=block.id,
                    content=f"Error: Tool execution aborted ({abort_err})",
                    is_error=True,
                )
            ],
        ), []
    except Exception as e:
        # A late abort can still race the post-tool gate above (the
        # signal trips between the post-tool check and the exception).
        # Honour it here so a tool that raises an unrelated error AFTER
        # ESC landed doesn't get reported as a tool bug when the user
        # actually pressed ESC.
        if _is_user_cancelled_abort(tool_use_context):
            return _build_user_cancelled_result(block.id), []
        error_str = f"Error: {e}"
        return UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id=block.id,
                    content=error_str,
                    is_error=True,
                )
            ],
        ), []


async def _run_tools_partitioned(
    tool_use_blocks: list[ToolUseBlock],
    tool_registry: ToolRegistry,
    tool_use_context: ToolContext,
    tools: Tools,
) -> list[UserMessage]:
    """Run tools with TS-matching concurrency: safe tools parallel, unsafe exclusive.

    Mirrors typescript/src/tools/partitionToolCalls + runTools (Mode 2).
    ConcurrencySafe tools (Read, Grep, Glob, etc.) run in parallel up to
    MAX_TOOL_USE_CONCURRENCY. Non-safe tools (Bash, Edit, Write) run
    exclusively one at a time.

    Dispatch concurrent tools as ``asyncio`` tasks (rather
    than ``asyncio.gather`` on raw coroutines) and poll with
    ``asyncio.wait(FIRST_COMPLETED)`` so a user abort can cancel the
    in-flight tool tasks as soon as one completes, instead of
    waiting for the slowest tool in the batch. ``asyncio.to_thread``
    runs the tool synchronously on a worker thread, so cancelling the
    wrapping task does NOT preempt the underlying tool — the tool
    finishes naturally and its result is discarded by the gate. This
    still wins for UX: ``Ctrl+C`` mid-batch unblocks the agent loop
    within ``max(remaining_tool_time, ~0s)`` instead of waiting for
    every tool to finish, and the next turn starts as soon as the
    abort controller's ``AbortError`` propagates out of
    :func:`_is_user_cancelled_abort`.
    """
    batches = _partition_tool_calls(tool_use_blocks, tools)
    # Emit all primary tool_result messages first, then all supplemental
    # (isMeta) messages. Interleaving (e.g. [a_result, a_meta, b_result])
    # breaks ensure_tool_result_pairing because the merge guard refuses to
    # combine tool_result-bearing user messages with text-only ones, so
    # b_result becomes orphaned and the pairing logic injects a synthetic
    # "[Tool result missing due to internal error]" placeholder.
    primaries: list[UserMessage] = []
    extras: list[UserMessage] = []

    def _accumulate(pair: tuple[UserMessage, list[UserMessage]]) -> None:
        primaries.append(pair[0])
        extras.extend(pair[1])

    # Dispatch a batch of tool-use blocks concurrently
    # with FIRST_COMPLETED polling. Aborts short-circuit remaining tasks
    # so the agent loop unwinds the moment the abort signal trips,
    # without waiting for stragglers. Returns when every task is done
    # OR when the abort signal trips (whichever first); on abort,
    # pending tasks are cancelled (the underlying worker thread is
    # NOT preemptable — Python has no thread cancellation — but its
    # result is discarded by the caller-supplied gate at task-done
    # time).
    async def _run_concurrent_batch(
        blocks: list[ToolUseBlock],
    ) -> None:
        if not blocks:
            return
        if len(blocks) == 1:
            # Single tool: no benefit from FIRST_COMPLETED polling,
            # keep the original gather-like shape so the caller path
            # is unchanged for the common case.
            pair = await asyncio.to_thread(
                _dispatch_single_tool,
                blocks[0],
                tool_registry,
                tool_use_context,
                tools,
            )
            _accumulate(pair)
            return

        tasks: dict[asyncio.Task[tuple[UserMessage, list[UserMessage]]], ToolUseBlock] = {}
        for block in blocks:
            task = asyncio.create_task(
                asyncio.to_thread(
                    _dispatch_single_tool,
                    block,
                    tool_registry,
                    tool_use_context,
                    tools,
                ),
                name=f"tool-dispatch-{block.name}",
            )
            tasks[task] = block
        try:
            pending: set[asyncio.Task[tuple[UserMessage, list[UserMessage]]]] = set(tasks)
            # Abort polling: ``asyncio.wait`` only returns when
            # a task completes, so without a timeout it would block
            # until the slowest remaining tool finishes — the very
            # behaviour we're fixing. By passing ``timeout=0.1`` we
            # re-check the abort signal every 100ms, matching the
            # cancel-latency bound the streaming drain helper
            # (clawcodex_ext.providers._stream_drain) uses. The 100ms
            # interval is invisible to the user (well below human
            # reaction time) and adds no measurable overhead when no
            # abort fires.
            _ABORT_POLL_INTERVAL_S = 0.1
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=_ABORT_POLL_INTERVAL_S,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    # ``asyncio.to_thread`` does not raise CancelledError
                    # in the worker; cancellation surfaces as the task
                    # being cancelled (with no result). We discard the
                    # cancellation path here and rely on the post-loop
                    # abort check below.
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        # Surface unexpected exceptions so they reach the
                        # agent-loop's outer try/except — never swallow
                        # a tool failure silently. AbortError is the
                        # one exception we DO want to translate into a
                        # synthetic cancelled tool_result below; it
                        # arrives via the in-tool AbortError branch in
                        # _dispatch_single_tool, which has already
                        # tripped the controller and returned a
                        # synthetic (non-raising) result, so this path
                        # should not see AbortError in practice.
                        raise exc
                    _accumulate(task.result())
                # Poll for aborts between batches. Even if all
                # remaining tasks finish naturally, an abort that
                # fired mid-batch must short-circuit subsequent work
                # so we don't burn another tool round-trip the user
                # already cancelled. With ``timeout`` set above this
                # also catches aborts that fire while no task is
                # completing — the loop wakes up every 100ms to
                # check ``aborted`` even when ``done`` is empty.
                if pending and _is_user_cancelled_abort(tool_use_context):
                    break
        finally:
            # Cancel anything still in flight and await the
            # cancellations so the event loop doesn't accumulate
            # dangling task references. ``return_exceptions=True``
            # keeps CancelledError from being raised into our caller
            # (which would mask the user's actual AbortError).
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            # Synthesize cancelled tool_results for any block whose
            # task did not produce one. Mirrors the
            # ``_is_user_cancelled_abort`` branch at the head of
            # _dispatch_single_tool, applied retroactively so the
            # tool_use/tool_result pairing invariant is preserved
            # even on the abort path.
            if _is_user_cancelled_abort(tool_use_context):
                for task, block in tasks.items():
                    if not task.done() or task.cancelled():
                        primaries.append(_build_user_cancelled_result(block.id))

    for batch in batches:
        if batch.is_concurrent_safe and len(batch.blocks) > 1:
            # Split at the concurrency cap to match the upstream
            # behaviour: at most MAX_TOOL_USE_CONCURRENCY tools in
            # flight per window.
            window = batch.blocks[:MAX_TOOL_USE_CONCURRENCY]
            await _run_concurrent_batch(window)
            if _is_user_cancelled_abort(tool_use_context):
                # Don't drain the overflow window once the user has
                # cancelled — emit synthetic results so pairing stays
                # intact, then exit the batch loop early.
                for block in batch.blocks[MAX_TOOL_USE_CONCURRENCY:]:
                    primaries.append(_build_user_cancelled_result(block.id))
                break
            if len(batch.blocks) > MAX_TOOL_USE_CONCURRENCY:
                overflow = batch.blocks[MAX_TOOL_USE_CONCURRENCY:]
                await _run_concurrent_batch(overflow)
                if _is_user_cancelled_abort(tool_use_context):
                    # _run_concurrent_batch already appended synthetic
                    # results for any cancelled-in-flight tool; nothing
                    # more to do here.
                    break
        else:
            for block in batch.blocks:
                pair = await asyncio.to_thread(
                    _dispatch_single_tool,
                    block,
                    tool_registry,
                    tool_use_context,
                    tools,
                )
                _accumulate(pair)
                if _is_user_cancelled_abort(tool_use_context):
                    # Mark the remaining exclusive-batch tools as
                    # cancelled too so we don't issue them and so the
                    # pairing invariant holds.
                    remaining = batch.blocks[batch.blocks.index(block) + 1 :]
                    for tail_block in remaining:
                        primaries.append(_build_user_cancelled_result(tail_block.id))
                    break
            else:
                continue
            # Broke out of the exclusive-batch for-loop due to abort.
            break

    return [*primaries, *extras]


def _run_tools_sync(
    tool_use_blocks: list[ToolUseBlock],
    tool_registry: ToolRegistry,
    tool_use_context: ToolContext,
) -> list[UserMessage]:
    """Legacy synchronous tool execution (no partitioning).

    Same primaries-first invariant as :func:`_run_tools_partitioned`.
    """
    primaries: list[UserMessage] = []
    extras: list[UserMessage] = []
    for block in tool_use_blocks:
        primary, extras_for_block = _dispatch_single_tool(block, tool_registry, tool_use_context)
        primaries.append(primary)
        extras.extend(extras_for_block)
    return [*primaries, *extras]


def _resolve_effective_tools(
    params: QueryParams,
    tool_use_context: ToolContext,
    messages: list[Message],
) -> Tools:
    """Resolve the tool list sent to the LLM for this turn."""
    if not tool_use_context.options.tools:
        tool_use_context.options.tools = list(params.tools)

    base_tools = tool_use_context.options.tools or params.tools
    try:
        from extensions.sop_converter.runtime.macros.session import iter_effective_tools

        base_tools = iter_effective_tools(tool_use_context, list(base_tools or []))
    except ImportError:  # nosec B110
        pass  # The optional integration is unavailable; continue with the built-in path.
    model = tool_use_context.options.main_loop_model or getattr(params.provider, "model", "") or ""

    filtered = filter_tools_for_request(base_tools, model, messages)
    try:
        from clawcodex_ext.goal.tools import filter_goal_model_tools_for_context  # pylint: disable=no-name-in-module

        filtered = filter_goal_model_tools_for_context(filtered, tool_use_context)
    except Exception:
        logger.debug("Failed to filter goal model tools for context", exc_info=True)
    # Re-merge after filters so session overlay tools are not dropped.
    try:
        from extensions.sop_converter.runtime.macros.session import iter_effective_tools

        filtered = iter_effective_tools(tool_use_context, list(filtered or []))
    except ImportError:  # nosec B110
        pass  # The optional integration is unavailable; continue with the built-in path.
    return filtered


async def _query_impl(  # pylint: disable=too-many-nested-blocks
    params: QueryParams,
    *,
    terminal_holder: TerminalHolder | None = None,
) -> AsyncGenerator[Message | StreamEvent, None]:
    """Canonical agent loop (chapter 5, Phase A foundation).

    The async generator yields messages and stream events to the consumer.
    The final ``Terminal`` is written to ``terminal_holder.value`` just
    before the generator returns (Python async generators cannot return
    values: PEP 525). Callers who care about the terminal pass their own
    ``TerminalHolder`` and read its ``.value`` after iteration.

    See :func:`run_query` for a convenience helper that consumes the
    generator and returns ``(messages, terminal)``.

    Recovery integration, stop hooks, token budget, the continuation
    nudge (ch05 rounds 2-3), and the retry lane with model fallback
    (ch04 round-4) are all wired: after ``MAX_529_RETRIES`` consecutive
    overloaded errors with ``params.fallback_model`` configured, the lane
    switches ``provider.model`` session-sticky (never persisted) and
    keeps going. No tombstones by design — the lane only retries when
    nothing streamed, so no partial assistant message ever needs
    retraction (TS retries wrap partial streams and must tombstone).
    """

    _diag = os.environ.get("CLAWCODEX_DEBUG", "").lower() in ("1", "true", "yes")
    holder = terminal_holder or TerminalHolder()
    # Inner-only flag for the future outer two-layer wrapper (Phase G).
    # Until that lands, the flag is local; set_terminal still writes it
    # so every exit site uses the canonical helper.
    natural_termination: list[bool] = [False]
    state = QueryState(
        messages=list(params.messages),
        tool_use_context=params.tool_use_context,
        max_output_tokens_override=params.max_output_tokens_override,
    )
    config = build_query_config()
    # Keep one guard state for the entire query so repeated tool failures can
    # be detected across turns; successful results reset its counters.
    tool_failure_guard_state = create_tool_failure_loop_guard_state()
    from clawcodex_ext.bootstrap.state import snapshot_output_tokens_for_turn  # pylint: disable=no-name-in-module

    if getattr(params.tool_use_context, "agent_id", None) is None and params.query_source not in (
        "compact",
        "session_memory",
    ):
        snapshot_output_tokens_for_turn(params.token_budget)
    budget_tracker = create_budget_tracker()
    params.tool_use_context.options.tools = list(params.tools)
    goal_runtime = None
    goal_turn_id: str | None = None
    goal_turn_start_id: str | None = None
    goal_evidence_boundary_id: str | None = None
    goal_evidence_boundary_content: str | None = None

    def _goal_start_turn(tool_use_context: ToolContext) -> None:
        nonlocal goal_runtime, goal_turn_id, goal_turn_start_id
        from clawcodex_ext.goal.runtime import goal_runtime_for_context  # pylint: disable=no-name-in-module

        has_bound_goal_state = bool(
            getattr(tool_use_context, "goal_runtime", None) is not None
            or getattr(tool_use_context, "goal_service", None) is not None
        )
        try:
            goal_runtime = goal_runtime_for_context(tool_use_context)
        except Exception:
            if has_bound_goal_state:
                raise
            goal_runtime = None
        if goal_runtime is None:
            goal_turn_id = None
            goal_turn_start_id = None
            return
        goal_turn_id = goal_runtime.on_turn_start(plan_mode=bool(getattr(tool_use_context, "plan_mode", False)))
        goal_turn_start_id = goal_runtime.goal_id_at_turn_start(goal_turn_id)

    def _active_evaluator_goal() -> Any | None:
        if goal_runtime is None:
            return None
        from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus  # pylint: disable=no-name-in-module

        candidate = goal_runtime.service.get_goal(goal_runtime.thread_id)
        if (
            candidate is not None
            and candidate.status is ThreadGoalStatus.ACTIVE
            and candidate.completion_mode is GoalCompletionMode.EVALUATOR
        ):
            return candidate
        return None

    def _goal_notice_data(
        goal: Any,
        *,
        state: str,
        met: bool | None,
        reason: str | None,
    ) -> dict[str, Any]:
        return {
            "goalId": goal.goal_id,
            "condition": goal.objective,
            "state": state,
            "met": met,
            "reason": reason,
            "turns": goal.evaluation_count,
            "tokens": goal.tokens_used,
            "durationSeconds": goal.time_used_seconds,
        }

    def _goal_record_usage(assistant_messages: list[AssistantMessage]) -> None:
        if goal_runtime is None or goal_turn_id is None:
            return
        candidate = goal_runtime.service.get_goal(goal_runtime.thread_id)
        if candidate is not None and candidate.goal_id != goal_turn_start_id:
            return
        for assistant in assistant_messages:
            goal_runtime.on_token_usage(goal_turn_id, getattr(assistant, "usage", None) or {})

    def _goal_evaluation_messages(
        evaluator_goal: Any,
        messages: list[Message],
        assistant_messages: list[AssistantMessage],
    ) -> list[Message]:
        transcript = [*messages, *assistant_messages]
        if evaluator_goal.goal_id != goal_evidence_boundary_id:
            return transcript
        if goal_evidence_boundary_content is None:
            return list(assistant_messages)
        for index in range(len(messages) - 1, -1, -1):
            if str(getattr(messages[index], "content", "")) == goal_evidence_boundary_content:
                return [*messages[index:], *assistant_messages]
        # A context pipeline may discard the meta boundary.  In that case,
        # use only the current goal-bound output rather than leak evidence
        # produced for the replaced goal into the evaluator.
        return list(assistant_messages)

    def _goal_finish_tools(
        tool_use_blocks: list[ToolUseBlock],
        *,
        handler_executed: bool,
    ) -> list[UserMessage]:
        if goal_runtime is None or goal_turn_id is None:
            return []
        steering: list[UserMessage] = []
        for block in tool_use_blocks:
            steering.extend(
                goal_runtime.on_tool_finish(
                    goal_turn_id,
                    tool_name=block.name,
                    call_id=block.id,
                    handler_executed=handler_executed,
                )
            )
        return steering

    def _goal_finish_turn(kind: str, error: BaseException | None = None) -> None:
        nonlocal goal_turn_id, goal_turn_start_id
        if goal_runtime is None or goal_turn_id is None:
            return
        turn_id = goal_turn_id
        goal_turn_id = None
        goal_turn_start_id = None
        if kind == "error" and error is not None:
            goal_runtime.on_turn_error(turn_id, error)
        elif kind == "abort":
            goal_runtime.on_turn_abort(turn_id)
        else:
            goal_runtime.on_turn_stop(turn_id)

    # ch09 round-4 WI-1 — capture the parent's rendered system prompt onto
    # its context so a fork spawned DURING this turn threads the parent's
    # EXACT prompt (byte-identical), letting the fork child chain onto the
    # parent's warm [system+tools+history] cache. Without this the field
    # was dead scaffolding (permanently None) and fork children fell back
    # to DEFAULT_AGENT_PROMPT — diverging at byte 0 and reprocessing the
    # whole history at full price (fork's entire economic point, inert).
    # TS: toolUseContext.renderedSystemPrompt (AgentTool.tsx:496). The
    # captured value is the INPUT to _call_model_sync's assembly; the fork
    # child threads the same input and goes through the same assembly, so
    # the wire bytes match. Set unconditionally (cheap; only fork reads it;
    # a fork-of-subagent is guarded regardless).
    if params.system_prompt:
        try:
            params.tool_use_context.rendered_system_prompt = params.system_prompt
        except Exception:  # noqa: BLE001 — read-only stub context
            logger.debug("Failed to set rendered_system_prompt on tool_use_context", exc_info=True)

    def _exceeded_max_turns(next_turn_count: int) -> int | None:
        max_turns = int(params.max_turns or 0)
        if max_turns and next_turn_count > max_turns:
            return max_turns
        return None

    def _finish_at_max_turns(next_turn_count: int) -> None:
        error = RuntimeError("max turns reached")
        set_terminal(
            holder,
            natural_termination,
            Terminal(reason="max_turns", turn_count=next_turn_count),
        )
        _goal_finish_turn("error", error)
        # Evaluator decisions are recorded after the work turn is closed.
        # When the explicit cap ends the run at that boundary, keep the
        # still-active condition on an idle clock for `/goal` status/retry.
        if goal_runtime is not None and _active_evaluator_goal() is not None:
            goal_runtime.restore_after_resume()

    while True:
        messages = state.messages
        # Invoke per-turn callbacks at the start of each turn.
        for cb in state.on_turn_start_callbacks:
            cb(state)
        if _diag:
            logger.warning(
                "[DIAG] query loop: turn=%d  messages=%d  transition=%s",
                state.turn_count,
                len(messages),
                state.transition.reason if state.transition else "initial",
            )
        tool_use_context = state.tool_use_context
        _goal_start_turn(tool_use_context)
        # WI-5.1: reset the per-message aggregate counter at each turn
        # boundary. The 200K cap is PER USER MESSAGE (the next batch of
        # tool_result blocks the model will see), not per session. Without
        # this reset the counter grows monotonically and every tool result
        # eventually gets persisted regardless of size. Mirrors TS
        # ``toolResultStorage.ts:collectCandidatesByMessage`` which
        # partitions evaluation by message.
        tool_use_context.tool_result_chars_so_far = 0
        has_attempted_reactive_compact = state.has_attempted_reactive_compact
        max_output_tokens_override = state.max_output_tokens_override
        turn_count = state.turn_count

        yield StreamEvent(type="stream_request_start")

        # --- Phase 0: Compression Pipeline ---
        # Mirrors TS query loop Phase 0: toolResultBudget → snip → microcompact → collapse → autocompact
        if params.pipeline_config is not None:
            try:
                # Estimate input tokens so layer 5 (autocompact) can decide
                # whether to fire. Without this the MIN_INPUT_TOKENS_FOR_AUTOCOMPACT
                # guard short-circuits and auto-compact never triggers.
                est_input_tokens = rough_token_count_estimation_for_messages(messages)
                pipeline_result = await run_compression_pipeline(
                    messages,
                    input_token_count=est_input_tokens,
                    config=params.pipeline_config,
                )
                if pipeline_result.tokens_saved > 0:
                    messages = pipeline_result.messages
                    if _diag:
                        logger.warning(
                            "[DIAG] Compression pipeline saved %d tokens (layers: %s)",
                            pipeline_result.tokens_saved,
                            ", ".join(pipeline_result.layers_applied),
                        )
            except Exception:
                logger.warning("Compression pipeline failed, continuing with original messages", exc_info=True)

        # The general pre-LLM hook lets external policies, such as Budget Mode,
        # modify messages or the system prompt before ``_call_model_sync``.
        current_system_prompt = params.system_prompt
        hook_result = _call_hooks_if_enabled("pre_llm", messages, current_system_prompt, state=state, params=params)
        messages = hook_result[0]
        current_system_prompt = hook_result[1]

        # Ch5/B.4 + B.5 — pre-emption guards before the API call.
        # Two distinct guards:
        #   B.4: hard blocking limit (auto-compact off OR no other
        #        recovery available) — saves the 500 the API would
        #        return anyway.
        #   B.5: autocompact circuit-breaker tripped (3 consecutive
        #        failures) AND we're still over the autocompact
        #        threshold — gives the user an actionable message
        #        instead of an opaque 500.
        # Skip when this iteration is a recovery retry (the messages
        # were already validated under the limit), or when this is a
        # compact/session_memory forked query (those need to run to
        # REDUCE the token count, blocking would deadlock).
        from clawcodex_ext.services.compact.autocompact import (
            MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
            calculate_token_warning_state,
            is_auto_compact_enabled,
        )

        skip_blocking_guards = params.query_source in ("compact", "session_memory") or (
            state.transition is not None
            and state.transition.reason in ("collapse_drain_retry", "reactive_compact_retry")
        )

        if not skip_blocking_guards:
            context_window = _get_context_window(params.provider)
            # NB: ``messages`` here is already the post-pipeline list (we
            # reassigned ``messages = pipeline_result.messages`` above when
            # the pipeline freed tokens). Guard math uses the post-pipeline
            # count directly.
            token_usage = rough_token_count_estimation_for_messages(messages)
            warning = calculate_token_warning_state(token_usage, context_window)

            # B.5 (checked first — gives the more actionable message
            # when the breaker has tripped). Mirrors TS query.ts:705-725.
            tracking = state.auto_compact_tracking or (
                params.pipeline_config.autocompact_tracking if params.pipeline_config is not None else None
            )
            consec = getattr(tracking, "consecutive_failures", 0) if tracking is not None else 0
            if (
                consec >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES
                and is_auto_compact_enabled()
                and warning["is_above_auto_compact_threshold"]
            ):
                yield _create_assistant_api_error_message(
                    content=(
                        "The conversation has exceeded the context limit "
                        "and automatic compaction has failed. Press esc twice "
                        "to go up a few messages and try again, or start a "
                        "new session with /new."
                    ),
                    error="invalid_request",
                )
                set_terminal(
                    holder,
                    natural_termination,
                    Terminal(reason="blocking_limit"),
                )
                _goal_finish_turn("error", RuntimeError("context blocking limit"))
                return

            # B.4 (hard blocking limit). Mirrors TS query.ts:683-696.
            # Only fires when reactive-compact recovery is NOT available
            # — otherwise we let the API 413 and the B.2 recovery path
            # handle it. This matches TS: the guard exists to short-
            # circuit only when no recovery would catch the 500 anyway.
            elif warning["is_at_blocking_limit"] and not (
                config.reactive_compact_enabled and is_auto_compact_enabled()
            ):
                yield _create_assistant_api_error_message(
                    content=PROMPT_TOO_LONG_ERROR_MESSAGE,
                    error="invalid_request",
                )
                set_terminal(
                    holder,
                    natural_termination,
                    Terminal(reason="blocking_limit"),
                )
                _goal_finish_turn("error", RuntimeError("context blocking limit"))
                return

        assistant_messages: list[AssistantMessage] = []
        tool_results: list[UserMessage] = []
        tool_use_blocks: list[ToolUseBlock] = []
        needs_follow_up = False

        effective_tools = _resolve_effective_tools(params, tool_use_context, messages)
        # LKB/Task-v2 periodic reminder. Evaluate at every model-call boundary
        # so a read-only TaskList round can become the tenth idle assistant
        # turn and trigger immediately before the follow-up call. Counting
        # uses the uncompressed QueryState history; the reminder itself joins
        # the working message stream and is persisted through the attachment
        # callback for cross-request cadence de-duplication.
        if params.task_reminder_enabled:
            try:
                from clawcodex_ext.query.task_reminder import build_task_reminder

                task_reminder = build_task_reminder(
                    state.messages,
                    tool_use_context,
                    effective_tools,
                    query_source=params.query_source,
                )
                if task_reminder is not None:
                    messages = [*messages, task_reminder]
                    if params.on_attachment is not None:
                        try:
                            params.on_attachment(task_reminder)
                        except Exception:
                            logger.debug(
                                "task-reminder attachment persistence failed",
                                exc_info=True,
                            )
                    yield task_reminder
            except Exception:
                logger.debug("task-reminder wiring failed", exc_info=True)

        # ch04 round-3 G3, widened by round-4 GAP B: the retry lane.
        # Yield-based like TS withRetry (status surfaces in the message
        # stream, not a side channel). Constraints preserved from round-3:
        # foreground sources only; never after partial output (a mid-stream
        # failure with rendered text follows the normal error path — no
        # duplicate text); SDK auto-retry disabled underneath via
        # sdk_max_retries=0 so attempt counts tell the truth.
        #
        # Round-4 additions (TS withRetry.ts parity):
        #   * general retryable classes — 429 / 5xx / connection / timeout —
        #     retry under DEFAULT_MAX_RETRIES with exponential backoff +
        #     0-25% jitter (TS getRetryDelay); quota-exhausted and
        #     non-retryable errors bail immediately;
        #   * model fallback — after MAX_529_RETRIES consecutive 529s with
        #     params.fallback_model configured, switch provider.model
        #     (session-sticky like TS's mainLoopModel switch; NOT persisted
        #     to settings), announce it, reset the 529 counter, keep going.
        #     No tombstones needed (deliberate divergence): this lane only
        #     retries when NOTHING streamed; TS's retry wraps partial
        #     streams and must tombstone the orphans (query.ts:795-824).
        _is_foreground = params.query_source in FOREGROUND_529_RETRY_SOURCES
        _streamed_any = [False]
        _outer_chunk_cb = params.on_text_chunk

        def _marking_chunk_cb(text: str) -> None:
            _streamed_any[0] = True
            if _outer_chunk_cb is not None:
                _outer_chunk_cb(text)

        try:
            _general_attempts = 0
            _consecutive_529s = 0
            while True:
                _streamed_any[0] = False
                try:
                    returned_assistants, returned_tool_blocks = await _call_model_sync(
                        provider=params.provider,
                        messages=messages,
                        system_prompt=current_system_prompt,
                        tools=effective_tools,
                        model=tool_use_context.skill_model_override or params.model,
                        effort_override=tool_use_context.skill_effort_override,
                        max_output_tokens_override=max_output_tokens_override,
                        abort_signal=params.abort_controller.signal,
                        on_text_chunk=(_marking_chunk_cb if _outer_chunk_cb is not None else None),
                        on_thinking_chunk=params.on_thinking_chunk,
                        extended_thinking=params.extended_thinking,
                        thinking_effort=params.thinking_effort,
                        sdk_max_retries=0 if _is_foreground else None,
                    )
                    break
                except AbortError:
                    raise
                except Exception as retry_exc:
                    if not _is_foreground or _streamed_any[0] or params.abort_controller.signal.aborted:
                        raise

                    from clawcodex_ext.services.api.errors import (
                        categorize_retryable_api_error,
                        is_quota_exhausted,
                    )

                    is_529 = _is_overloaded_error(retry_exc)
                    if not is_529:
                        classification = categorize_retryable_api_error(retry_exc)
                        if not classification.retryable or is_quota_exhausted(retry_exc):
                            raise

                    _general_attempts += 1
                    if is_529:
                        _consecutive_529s += 1
                    else:
                        _consecutive_529s = 0

                    # Model fallback — TS withRetry.ts:345-369 →
                    # query.ts:977-1032. Fires once; a 529 storm on the
                    # fallback model then follows the normal exhaustion path
                    # (the model equality check keeps it single-shot).
                    if (
                        is_529
                        and _consecutive_529s >= MAX_529_RETRIES
                        and params.fallback_model
                        and params.fallback_model != getattr(params.provider, "model", None)
                    ):
                        _original_model = getattr(params.provider, "model", "?")
                        try:
                            params.provider.model = params.fallback_model
                        except Exception:  # noqa: BLE001 — read-only provider stub
                            raise retry_exc
                        logger.warning(
                            "model fallback: %s -> %s after %d consecutive overloaded errors",
                            _original_model,
                            params.fallback_model,
                            _consecutive_529s,
                        )
                        yield SystemMessage(
                            content=(f"Switched to {params.fallback_model} due to high demand for {_original_model}"),
                            level="warning",
                            subtype="model_fallback",
                        )
                        _consecutive_529s = 0
                        continue

                    if is_529 and _consecutive_529s > MAX_529_RETRIES:
                        raise
                    if _general_attempts > DEFAULT_MAX_RETRIES:
                        raise

                    _base = RETRY_BASE_DELAY_SECONDS * (2 ** (_general_attempts - 1))
                    # 0-25% jitter (TS getRetryDelay, withRetry.ts:561-566);
                    # capped so a long exponential tail can't exceed the
                    # Retry-After clamp either way.
                    _base = min(_base, 60.0) * (1.0 + random.random() * 0.25)  # nosec B311 — jitter, not security
                    delay = _retry_after_seconds(retry_exc, _base)
                    if is_529:
                        _status = (
                            f"Server overloaded — retrying in {delay:.1f}s "
                            f"(attempt {_consecutive_529s}/{MAX_529_RETRIES})"
                        )
                    else:
                        _status = (
                            f"API error ({classification.error_type}) — "
                            f"retrying in {delay:.1f}s "
                            f"(attempt {_general_attempts}/{DEFAULT_MAX_RETRIES})"
                        )
                    yield SystemMessage(
                        content=_status,
                        level="warning",
                        subtype="api_retry",
                    )
                    # Abort-aware backoff (critic): TS sleeps WITH the
                    # signal (withRetry sleep(delay, signal)); a
                    # Retry-After can direct up to 60s — ESC must not
                    # be stuck behind it. Sliced sleep, checked every
                    # 250ms; an abort falls through to the existing
                    # aborted_streaming handling on the next attempt.
                    _remaining = delay
                    while _remaining > 0:
                        if params.abort_controller.signal.aborted:
                            break
                        _tick = min(0.25, _remaining)
                        await asyncio.sleep(_tick)
                        _remaining -= _tick
                    if params.abort_controller.signal.aborted:
                        # Land in the REAL abort lane (the outer
                        # except AbortError + post-call abort block),
                        # so the user sees the interruption message
                        # rather than a retry-lane model_error.
                        raise AbortError("interrupted during retry backoff") from retry_exc
                    continue
            assistant_messages = returned_assistants
            tool_use_blocks = returned_tool_blocks
            needs_follow_up = len(tool_use_blocks) > 0
            _goal_record_usage(assistant_messages)

            # Invoke post_llm after the response and before tool execution.
            hook_result = _call_hooks_if_enabled(
                "post_llm", assistant_messages, tool_use_blocks, state=state, params=params
            )
            assistant_messages = hook_result[0]
            tool_use_blocks = hook_result[1]

            for msg in assistant_messages:
                # Ch5/B.1 — three-source withholding pattern.
                # Mirrors TS query.ts:866-903. Recoverable errors are
                # suppressed from the yield stream so SDK consumers
                # (Cowork, the desktop app) don't disconnect mid-recovery.
                # If recovery exhausts, the message is surfaced later by
                # the dispatch code in the no-follow-up branch (B.2).
                withheld = (
                    _is_withheld_max_output_tokens(msg)
                    or _is_withheld_prompt_too_long(msg)
                    or _is_withheld_media_size(msg)
                )
                if not withheld:
                    yield msg

        except AbortError:
            # The provider's abort listener closed the streaming HTTP
            # response mid-flight (ESC pressed while the model was still
            # generating). The signal is already tripped, so let the
            # ``if params.abort_controller.signal.aborted`` block right
            # below us do the cancellation processing in exactly one
            # place — anything we did here would duplicate that work.
            pass

        except CannotRetryError as e:
            original_error = e.original_error
            logger.error("Query error: %s", original_error)
            _mark_proactive_context_blocked_on_error(original_error)
            error_message = str(original_error)

            for err_msg in _yield_missing_tool_result_blocks(assistant_messages, error_message):
                yield err_msg

            yield _create_assistant_api_error_message(content=error_message)
            set_terminal(
                holder,
                natural_termination,
                Terminal(reason="model_error", error=original_error),
            )
            _goal_finish_turn("error", original_error)
            return

        except Exception as e:
            logger.error("Query error: %s", e)
            _mark_proactive_context_blocked_on_error(e)
            error_message = str(e)

            for err_msg in _yield_missing_tool_result_blocks(assistant_messages, error_message):
                yield err_msg

            yield _create_assistant_api_error_message(content=error_message)
            set_terminal(holder, natural_termination, Terminal(reason="model_error", error=e))
            _goal_finish_turn("error", e)
            return

        if params.abort_controller.signal.aborted:
            for err_msg in _yield_missing_tool_result_blocks(assistant_messages, "Interrupted by user"):
                yield err_msg

            if params.abort_controller.signal.reason != "interrupt":
                yield _create_user_interruption_message(tool_use=False)
            set_terminal(holder, natural_termination, Terminal(reason="aborted_streaming"))
            _goal_finish_turn("abort")
            return

        # ch01 round-4 WI-2 — PostSampling hook wire (TS query.ts:1079-1089).
        # Placed after the abort check: TS fires before it, but its call is
        # non-blocking (`void …`) so pre-abort is free there; an inline await
        # before the abort return would delay ESC responsiveness by the hook
        # runtime. Consequence: hooks do not fire for user-aborted streams.
        if assistant_messages:
            await _fire_post_sampling_hooks(
                assistant_messages,
                params.provider,
                tool_use_context,
            )

        if not needs_follow_up:
            last_message = assistant_messages[-1] if assistant_messages else None

            # Some providers occasionally emit an empty end_turn directly
            # after a successful tool result. Do not misclassify that as
            # task completion: ask the model to resume, with a strict cap.
            if (
                state.transition is not None
                and state.transition.reason in {"next_turn", "continuation_nudge"}
                and _assistant_response_is_empty(assistant_messages)
                and state.continuation_nudge_count < EMPTY_END_TURN_NUDGE_LIMIT
            ):
                nudge = _create_user_message(CONTINUATION_NUDGE_MESSAGE, is_meta=True)
                yield nudge
                _goal_finish_turn("stop")
                state = QueryState(
                    messages=[*messages, *assistant_messages, nudge],
                    tool_use_context=tool_use_context,
                    auto_compact_tracking=state.auto_compact_tracking,
                    max_output_tokens_recovery_count=0,
                    has_attempted_reactive_compact=False,
                    max_output_tokens_override=None,
                    stop_hook_active=state.stop_hook_active,
                    continuation_nudge_count=state.continuation_nudge_count + 1,
                    pending_tool_use_summary=state.pending_tool_use_summary,
                    transition=Transition(reason="continuation_nudge"),
                )
                continue

            # Use the recovery-strategy registry for withheld-response errors.
            error_type = None
            if _is_withheld_max_output_tokens(last_message):
                error_type = "max_output_tokens"
            elif _is_withheld_prompt_too_long(last_message):
                error_type = "prompt_too_long"
            elif _is_withheld_media_size(last_message):
                error_type = "media_size"

            if error_type is not None:
                recovery_ctx = RecoveryContext(
                    state=state,
                    last_message=last_message,
                    config=config,
                    params=params,
                    messages=messages,
                    assistant_messages=assistant_messages,
                    error_type=error_type,
                )
                strategy_applied = False
                recovered_state: QueryState | None = None
                for strategy in find_recovery_strategies(error_type, state):
                    try:
                        raw_result = strategy.fn(recovery_ctx)
                        if asyncio.iscoroutine(raw_result):
                            result = await raw_result
                        else:
                            result = raw_result
                    except Exception:
                        logger.exception("Recovery strategy %r failed", strategy.name)
                        continue
                    if result is not None:
                        strategy_applied = True
                        new_state, yield_messages = result
                        for msg in yield_messages:
                            yield msg
                        if new_state is not None:
                            recovered_state = new_state
                            break
                        # new_state is None: terminate
                        if error_type == "media_size":
                            set_terminal(
                                holder,
                                natural_termination,
                                Terminal(reason="image_error"),
                            )
                            _goal_finish_turn("error", RuntimeError("image error"))
                        else:
                            if error_type == "prompt_too_long":
                                terminal_error = RuntimeError("prompt too long")
                                set_terminal(
                                    holder,
                                    natural_termination,
                                    Terminal(
                                        reason="prompt_too_long",
                                        error=terminal_error,
                                    ),
                                )
                                _goal_finish_turn("error", terminal_error)
                            elif _active_evaluator_goal() is not None:
                                terminal_error = _terminal_error_for_api_message(
                                    last_message,
                                    error_type=error_type,
                                )
                                set_terminal(
                                    holder,
                                    natural_termination,
                                    Terminal(reason="model_error", error=terminal_error),
                                )
                                _goal_finish_turn("error", terminal_error)
                            else:
                                set_terminal(
                                    holder,
                                    natural_termination,
                                    Terminal(reason="completed"),
                                )
                                _goal_finish_turn("stop")
                        await _fire_stop_failure_hooks(
                            last_message,
                            tool_use_context,
                        )
                        return
                # If no strategy applies, yield the last message and fall through.
                if recovered_state is not None:
                    state = recovered_state
                    _goal_finish_turn("stop")
                    continue
                if not strategy_applied and last_message is not None:
                    yield last_message

            if last_message and getattr(last_message, "isApiErrorMessage", False):
                await _fire_stop_failure_hooks(last_message, tool_use_context)
                if _active_evaluator_goal() is not None:
                    terminal_error = _terminal_error_for_api_message(last_message)
                    set_terminal(
                        holder,
                        natural_termination,
                        Terminal(reason="model_error", error=terminal_error),
                    )
                    _goal_finish_turn("error", terminal_error)
                else:
                    set_terminal(holder, natural_termination, Terminal(reason="completed"))
                    _goal_finish_turn("stop")
                return

            evaluator_goal = _active_evaluator_goal()
            if evaluator_goal is not None and evaluator_goal.goal_id != goal_turn_start_id:
                # The response was generated from the previous goal (or from
                # no goal at all). Do not let it satisfy a replacement goal.
                from clawcodex_ext.goal.steering import evaluator_start_message  # pylint: disable=no-name-in-module

                steering_messages = goal_runtime.consume_pending_steering_messages() if goal_runtime is not None else []
                if not steering_messages:
                    steering_messages = [evaluator_start_message(evaluator_goal)]
                boundary_message = steering_messages[-1]
                goal_evidence_boundary_id = evaluator_goal.goal_id
                goal_evidence_boundary_content = str(boundary_message.content)
                for steering_message in steering_messages:
                    yield steering_message
                _goal_finish_turn("stop")
                next_turn_count = turn_count + 1
                exceeded_max_turns = _exceeded_max_turns(next_turn_count)
                if exceeded_max_turns is not None:
                    yield _create_max_turns_attachment(
                        exceeded_max_turns,
                        next_turn_count,
                    )
                    _finish_at_max_turns(next_turn_count)
                    return
                state = QueryState(
                    messages=[
                        *messages,
                        *assistant_messages,
                        *steering_messages,
                    ],
                    tool_use_context=tool_use_context,
                    auto_compact_tracking=state.auto_compact_tracking,
                    turn_count=next_turn_count,
                    max_output_tokens_recovery_count=0,
                    has_attempted_reactive_compact=False,
                    max_output_tokens_override=None,
                    stop_hook_active=state.stop_hook_active,
                    continuation_nudge_count=0,
                    pending_tool_use_summary=state.pending_tool_use_summary,
                    transition=Transition(reason="stop_hook_blocking"),
                )
                continue

            stop_result = StopHookResult()
            try:
                async for item in handle_stop_hooks_streaming(
                    messages,
                    assistant_messages,
                    params.system_prompt if isinstance(params.system_prompt, str) else "",
                    tool_use_context,
                    params.query_source,
                    state.stop_hook_active,
                ):
                    if isinstance(item, StopHookResult):
                        stop_result = item
                    else:
                        yield item
            except Exception:
                logger.exception("stop hooks failed; continuing to exit")

            if stop_result.prevent_continuation:
                set_terminal(
                    holder,
                    natural_termination,
                    Terminal(reason="stop_hook_prevented"),
                )
                _goal_finish_turn("stop")
                return

            if stop_result.blocking_errors:
                _goal_finish_turn("stop")
                state = QueryState(
                    messages=[
                        *messages,
                        *assistant_messages,
                        *stop_result.blocking_errors,
                    ],
                    tool_use_context=tool_use_context,
                    auto_compact_tracking=state.auto_compact_tracking,
                    max_output_tokens_recovery_count=0,
                    has_attempted_reactive_compact=has_attempted_reactive_compact,
                    max_output_tokens_override=None,
                    stop_hook_active=True,
                    turn_count=turn_count,
                    pending_tool_use_summary=None,
                    continuation_nudge_count=state.continuation_nudge_count,
                    transition=Transition(reason="stop_hook_blocking"),
                )
                continue

            from clawcodex_ext.bootstrap.state import (  # pylint: disable=no-name-in-module
                get_current_turn_token_budget,
                get_turn_output_tokens,
                increment_budget_continuation_count,
            )

            budget_decision = check_token_budget(
                budget_tracker,
                getattr(tool_use_context, "agent_id", None),
                get_current_turn_token_budget(),
                get_turn_output_tokens(),
            )
            if isinstance(budget_decision, ContinueDecision):
                increment_budget_continuation_count()
                _goal_finish_turn("stop")
                state = QueryState(
                    messages=[
                        *messages,
                        *assistant_messages,
                        UserMessage(content=budget_decision.nudge_message, isMeta=True),
                    ],
                    tool_use_context=tool_use_context,
                    auto_compact_tracking=state.auto_compact_tracking,
                    max_output_tokens_recovery_count=0,
                    has_attempted_reactive_compact=False,
                    max_output_tokens_override=None,
                    stop_hook_active=None,
                    turn_count=turn_count,
                    pending_tool_use_summary=None,
                    continuation_nudge_count=state.continuation_nudge_count,
                    transition=Transition(reason="token_budget_continuation"),
                )
                continue

            # The model said it would continue acting but emitted no tool
            # call. Mirror the upstream continuation lane and cap the
            # automatic nudges per turn-chain.
            if (
                assistant_messages
                and (params.max_turns is None or turn_count < params.max_turns)
                and state.continuation_nudge_count < MAX_CONTINUATION_NUDGES
            ):
                last_assistant = assistant_messages[-1]
                content = getattr(last_assistant, "content", "")
                if isinstance(content, str):
                    last_text = content
                else:
                    last_text = " ".join(
                        getattr(block, "text", "") for block in content if getattr(block, "type", None) == "text"
                    )
                # An apparent completion is not sufficient for a request that
                # explicitly asks for every/multiple result. Require one
                # bounded audit pass before accepting a clean exit; the latch
                # survives tool rounds so this cannot become an infinite loop.
                from src.query.continuation_nudge import (
                    EXHAUSTIVE_AUDIT_NUDGE,
                    requests_exhaustive_results,
                )

                exhaustive_request = any(
                    not getattr(msg, "isMeta", False)
                    and getattr(msg, "role", None) == "user"
                    and requests_exhaustive_results(
                        getattr(msg, "content", "") if isinstance(getattr(msg, "content", ""), str) else ""
                    )
                    for msg in messages
                )
                if exhaustive_request and not state.exhaustive_audit_performed:
                    logger.debug("Exhaustive-result audit nudge triggered")
                    _goal_finish_turn("stop")
                    state = QueryState(
                        messages=[
                            *messages,
                            *assistant_messages,
                            UserMessage(content=EXHAUSTIVE_AUDIT_NUDGE, isMeta=True),
                        ],
                        tool_use_context=tool_use_context,
                        auto_compact_tracking=state.auto_compact_tracking,
                        max_output_tokens_recovery_count=0,
                        has_attempted_reactive_compact=False,
                        max_output_tokens_override=None,
                        stop_hook_active=None,
                        turn_count=turn_count,
                        pending_tool_use_summary=None,
                        continuation_nudge_count=state.continuation_nudge_count,
                        exhaustive_audit_performed=True,
                        transition=Transition(reason="continuation_nudge"),
                    )
                    continue
                if last_text and detect_continuation_signal(last_text):
                    _goal_finish_turn("stop")
                    state = QueryState(
                        messages=[
                            *messages,
                            *assistant_messages,
                            UserMessage(content=NUDGE_MESSAGE, isMeta=True),
                        ],
                        tool_use_context=tool_use_context,
                        auto_compact_tracking=state.auto_compact_tracking,
                        max_output_tokens_recovery_count=0,
                        has_attempted_reactive_compact=False,
                        max_output_tokens_override=None,
                        stop_hook_active=None,
                        turn_count=turn_count,
                        pending_tool_use_summary=None,
                        continuation_nudge_count=state.continuation_nudge_count + 1,
                        transition=Transition(reason="continuation_nudge"),
                    )
                    continue

            if evaluator_goal is not None:
                # _active_evaluator_goal() only returns a goal when a runtime
                # is bound. Keep that invariant explicit for type checking and
                # for future changes to the evaluator path.
                assert goal_runtime is not None
                from clawcodex_ext.goal.evaluator import (  # pylint: disable=no-name-in-module
                    GoalEvaluationError,
                    evaluate_goal,
                )
                from clawcodex_ext.goal.steering import (  # pylint: disable=no-name-in-module
                    evaluator_continuation_message,
                )

                try:
                    evaluation = await evaluate_goal(
                        params.provider,
                        evaluator_goal,
                        _goal_evaluation_messages(
                            evaluator_goal,
                            messages,
                            assistant_messages,
                        ),
                        abort_signal=params.abort_controller.signal,
                    )
                except asyncio.CancelledError:
                    _goal_finish_turn("abort")
                    raise
                except GoalEvaluationError as exc:
                    # The main work turn itself succeeded. Keep the goal active
                    # and stop this run explicitly instead of retry-spinning.
                    if goal_runtime is not None and goal_turn_id is not None:
                        goal_runtime.on_token_usage(goal_turn_id, exc.usage)
                    _goal_finish_turn("error", exc)
                    failed_goal = (
                        goal_runtime.service.get_goal(goal_runtime.thread_id)
                        if goal_runtime is not None
                        else evaluator_goal
                    )
                    notice = SystemMessage(
                        content=f"Goal evaluator failed: {exc}",
                        subtype="goal_evaluator_error",
                        level="warning",
                        data=_goal_notice_data(
                            failed_goal or evaluator_goal,
                            state="active",
                            met=None,
                            reason=str(exc),
                        ),
                        usage=exc.usage,
                    )
                    yield notice
                    set_terminal(
                        holder,
                        natural_termination,
                        Terminal(reason="goal_evaluator_error", error=exc),
                    )
                    return

                if goal_runtime is not None and goal_turn_id is not None:
                    goal_runtime.on_token_usage(goal_turn_id, evaluation.usage)
                _goal_finish_turn("stop")
                if params.abort_controller.signal.aborted:
                    raise asyncio.CancelledError(params.abort_controller.signal.reason or "user_interrupt")
                evaluated_goal = goal_runtime.service.record_evaluation(
                    goal_runtime.thread_id,
                    evaluation,
                    expected_goal_id=evaluator_goal.goal_id,
                    expected_evaluation_count=evaluator_goal.evaluation_count,
                )
                if evaluated_goal is None:
                    set_terminal(holder, natural_termination, Terminal(reason="completed"))
                    return

                if evaluation.met:
                    turn_label = "turn" if evaluated_goal.evaluation_count == 1 else "turns"
                    notice = SystemMessage(
                        content=(
                            "✓ Goal achieved\n\n"
                            f"Goal: {evaluated_goal.objective}\n\n"
                            f"running {evaluated_goal.time_used_seconds}s · "
                            f"{evaluated_goal.evaluation_count} {turn_label} · "
                            f"{evaluated_goal.tokens_used} tokens\n\n"
                            f"Last check: {evaluation.reason}"
                        ),
                        subtype="goal_achieved",
                        level="info",
                        data=_goal_notice_data(
                            evaluated_goal,
                            state="achieved",
                            met=True,
                            reason=evaluation.reason,
                        ),
                        usage=evaluation.usage,
                    )
                    yield notice
                    set_terminal(holder, natural_termination, Terminal(reason="completed"))
                    return

                notice = SystemMessage(
                    content=f"Goal check: not achieved\n\n{evaluation.reason}",
                    subtype="goal_evaluation",
                    level="info",
                    data=_goal_notice_data(
                        evaluated_goal,
                        state="active",
                        met=False,
                        reason=evaluation.reason,
                    ),
                    usage=evaluation.usage,
                )
                yield notice
                next_turn_count = turn_count + 1
                exceeded_max_turns = _exceeded_max_turns(next_turn_count)
                if exceeded_max_turns is not None:
                    yield _create_max_turns_attachment(
                        exceeded_max_turns,
                        next_turn_count,
                    )
                    _finish_at_max_turns(next_turn_count)
                    return
                guidance = evaluator_continuation_message(
                    evaluated_goal,
                    evaluation.reason,
                )
                yield guidance
                state = QueryState(
                    messages=[*messages, *assistant_messages, guidance],
                    tool_use_context=tool_use_context,
                    auto_compact_tracking=state.auto_compact_tracking,
                    turn_count=next_turn_count,
                    max_output_tokens_recovery_count=0,
                    has_attempted_reactive_compact=False,
                    max_output_tokens_override=None,
                    stop_hook_active=state.stop_hook_active,
                    continuation_nudge_count=0,
                    pending_tool_use_summary=state.pending_tool_use_summary,
                    transition=Transition(reason="stop_hook_blocking"),
                )
                continue

            set_terminal(holder, natural_termination, Terminal(reason="completed"))
            _goal_finish_turn("stop")
            return

        for block in tool_use_blocks:
            yield SystemMessage(
                content=f"Running tool: {block.name}",
                subtype="tool_use_progress",
            )

        if _diag:
            _tools_t0 = time.monotonic()
            _batches = _partition_tool_calls(tool_use_blocks, effective_tools)
            _batch_desc = ", ".join(
                f"[{'parallel' if b.is_concurrent_safe else 'exclusive'}: {[bl.name for bl in b.blocks]}]"
                for b in _batches
            )
            logger.warning(
                "[DIAG] query loop: running %d tools in %d batches: %s",
                len(tool_use_blocks),
                len(_batches),
                _batch_desc,
            )

        # Snapshot the current conversation onto the ToolContext so
        # tools that need history (currently: the client-side advisor)
        # can read it via ``ctx.messages``. The list is the post-pipeline
        # message stream up to and including the assistant message that
        # just emitted the tool_use blocks we're about to dispatch. The
        # advisor strips its own prior blocks via
        # ``build_advisor_forwarded_messages`` before forwarding. Other
        # tools ignore the field (the existing default factory was an
        # empty list, so behavior is unchanged for them).
        tool_use_context.messages = list(messages)
        # Also snapshot the active provider via a dynamic attribute so
        # the client-side advisor can reuse it (and its config) when
        # the user is on a proxy that probably proxies the advisor
        # model too. Set as a plain attribute (not a dataclass field)
        # to avoid touching ToolContext's public surface.
        setattr(tool_use_context, "_active_provider", params.provider)

        # Save CacheSafeParams for ``/btw`` side questions.
        try:
            from clawcodex_ext.agent.forked_agent import (
                CacheSafeParams,
                save_cache_safe_params,
            )

            save_cache_safe_params(
                CacheSafeParams(
                    system_prompt=params.system_prompt,
                    tool_use_context=tool_use_context,
                    user_context=params.user_context,
                    system_context=params.system_context,
                )
            )
        except Exception:
            logger.warning("Failed to save CacheSafeParams", exc_info=True)

        # Allow external policies to modify tool-use blocks before execution.
        hook_result = _call_hooks_if_enabled("pre_tool", tool_use_blocks, state=state, params=params)
        tool_use_blocks = hook_result[0]

        tool_results = await _run_tools_partitioned(
            tool_use_blocks,
            params.tool_registry,
            tool_use_context,
            effective_tools,
        )

        # Allow external policies to modify tool results after execution.
        hook_result = _call_hooks_if_enabled("post_tool", tool_results, state=state, params=params)
        tool_results = hook_result[0]
        goal_steering_messages = _goal_finish_tools(
            tool_use_blocks,
            handler_executed=not params.abort_controller.signal.aborted,
        )
        if goal_runtime is not None:
            try:
                goal_steering_messages = [
                    *goal_runtime.consume_pending_steering_messages(),
                    *goal_steering_messages,
                ]
            except Exception:
                logger.debug("Failed to consume steering messages", exc_info=True)

        if _diag:
            logger.warning(
                "[DIAG] query loop: tools finished in %.1fs, %d results",
                time.monotonic() - _tools_t0,
                len(tool_results),
            )
            for tr in tool_results:
                if isinstance(tr.content, list):
                    for b in tr.content:
                        if hasattr(b, "content"):
                            clen = len(b.content) if isinstance(b.content, str) else len(str(b.content))
                            logger.warning(
                                "[DIAG]   result: tool_use_id=%s  is_error=%s  content_len=%d",
                                getattr(b, "tool_use_id", "?"),
                                getattr(b, "is_error", False),
                                clen,
                            )

        for result_msg in tool_results:
            yield result_msg

        if params.abort_controller.signal.aborted:
            if params.abort_controller.signal.reason != "interrupt":
                yield _create_user_interruption_message(tool_use=True)
            # ch05 round-4 GAP C — TS query.ts:1621-1629: an abort that
            # lands ON the max-turns boundary also announces the limit
            # before the aborted_tools return (the terminal reason stays
            # aborted_tools either way).
            next_turn_count_on_abort = turn_count + 1
            if params.max_turns and next_turn_count_on_abort > params.max_turns:
                yield _create_max_turns_attachment(
                    params.max_turns,
                    next_turn_count_on_abort,
                )
            set_terminal(holder, natural_termination, Terminal(reason="aborted_tools"))
            _goal_finish_turn("abort")
            return

        # Ch5/round2 — hook_stopped terminal mapping. Mirrors TS at
        # query.ts:1698-1701. After tool execution, scan tool_results
        # for any AttachmentMessage carrying a
        # ``hook_stopped_continuation`` marker. If found, exit cleanly
        # with Terminal(reason='hook_stopped') rather than advancing to
        # the next turn.
        #
        # Order matters:
        #   * AFTER abort check (above) — user-driven abort wins,
        #     matching TS at query.ts:1665.
        #   * BEFORE max_turns check (below) — a hook-stopped exit must
        #     NOT also yield ``max_turns_reached``; the loop is exiting
        #     for a different reason. Matches TS where the hook_stopped
        #     return at :1701 precedes the max_turns check at :1885.
        #   * BEFORE state reconstruction — no next iteration follows.
        if any(_is_hook_stopped_continuation(msg) for msg in tool_results):
            set_terminal(
                holder,
                natural_termination,
                Terminal(reason="hook_stopped"),
            )
            _goal_finish_turn("stop")
            return

        # Run after abort/hook-stop handling and before max-turns, matching
        # upstream's terminal precedence while retaining the downstream tool
        # execution lane above.
        guard_decision = update_tool_failure_loop_guard(
            state=tool_failure_guard_state,
            tool_use_blocks=tool_use_blocks,
            tool_results=tool_results,
        )
        if guard_decision.tripped:
            logger.debug(
                "Tool failure loop guard tripped: kind=%s threshold=%s tool_name=%s error_category=%s path=%s",
                guard_decision.kind,
                guard_decision.threshold,
                guard_decision.tool_name,
                guard_decision.error_category,
                guard_decision.path,
            )
            yield _create_assistant_api_error_message(guard_decision.message or "")
            set_terminal(
                holder,
                natural_termination,
                Terminal(reason="tool_failure_loop"),
            )
            _goal_finish_turn("error", RuntimeError("tool failure loop"))
            return

        advisory_messages = [UserMessage(content=advisory) for advisory in guard_decision.advisories]

        next_turn_count = turn_count + 1

        exceeded_max_turns = _exceeded_max_turns(next_turn_count)
        if exceeded_max_turns is not None:
            yield _create_max_turns_attachment(exceeded_max_turns, next_turn_count)
            _finish_at_max_turns(next_turn_count)
            return

        # Chapter-10 / Chunk D / WI-3.3 — pending-messages drain at the
        # tool-round boundary. Per chapter §"Background: Three Channels":
        # *"messages arrive between tool rounds, not mid-execution. The
        # agent finishes its current thought, then receives the new
        # information."* We drain AFTER tool_results have been appended
        # but BEFORE the next API call, so the model sees the queued
        # messages on the next turn alongside the tool results.
        injected_messages = [
            *goal_steering_messages,
            *_drain_pending_user_messages(tool_use_context),
        ]
        for inj in injected_messages:
            yield inj

        # Invoke per-turn callbacks at the end of each turn.
        for cb in state.on_turn_end_callbacks:
            cb(state)
        # P102-D: on_turn_end hook
        call_hooks("on_turn_end", state, params=params)
        _goal_finish_turn("stop")

        state = QueryState(
            messages=[
                *messages,
                *assistant_messages,
                *tool_results,
                *advisory_messages,
                *injected_messages,
            ],
            tool_use_context=tool_use_context,
            auto_compact_tracking=state.auto_compact_tracking,
            turn_count=next_turn_count,
            max_output_tokens_recovery_count=0,
            has_attempted_reactive_compact=False,
            max_output_tokens_override=None,
            stop_hook_active=state.stop_hook_active,
            # Phase A: reset per-turn counter; carry pending summary forward.
            continuation_nudge_count=0,
            exhaustive_audit_performed=state.exhaustive_audit_performed,
            pending_tool_use_summary=state.pending_tool_use_summary,
            transition=Transition(reason="next_turn"),
        )


def _restore_skill_runtime_scope(context: ToolContext) -> None:
    """Restore request-private Skill overrides without touching session hooks."""
    if context._skill_permission_base is not None:
        context.permission_context = context._skill_permission_base
    if context._skill_options_base is not None:
        context.options = context._skill_options_base
    if context._skill_resource_roots_base is not None:
        context.skill_resource_roots = context._skill_resource_roots_base
    context._skill_permission_base = None
    context._skill_options_base = None
    context._skill_resource_roots_base = None
    context.skill_model_override = None
    context.skill_effort_override = None
    context.skill_scope_pending = False
    context.skill_scope_active = False
    context.active_skill_names = ()


def _begin_skill_runtime_scope(context: ToolContext) -> None:
    if context.skill_scope_pending:
        context.skill_scope_pending = False
    else:
        _restore_skill_runtime_scope(context)
    context.skill_scope_active = True


async def query(
    params: QueryParams,
    *,
    terminal_holder: TerminalHolder | None = None,
) -> AsyncGenerator[Message | StreamEvent, None]:
    context = params.tool_use_context
    _begin_skill_runtime_scope(context)
    try:
        async for item in _query_impl(params, terminal_holder=terminal_holder):
            yield item
    finally:
        _restore_skill_runtime_scope(context)


async def run_query(
    params: QueryParams,
) -> tuple[list[Message | StreamEvent], Terminal]:
    """Ch5/A.4: Convenience helper for callers that want both the
    yielded messages and the Terminal in one call.

    Drives the canonical :func:`query` async generator, collects all
    yielded messages into a list, and returns ``(messages, terminal)``.
    The terminal's reason discriminates why the loop stopped (11
    distinct reasons, matching :mod:`query.transitions`).

    Tests and convenience entry points should use this helper.
    Streaming consumers (REPL, TUI) should keep using ``async for``
    with their own ``TerminalHolder``.
    """
    holder = TerminalHolder()
    messages: list[Message | StreamEvent] = []
    async for msg in query(params, terminal_holder=holder):
        messages.append(msg)
    if holder.value is None:
        # Contract violation — the loop returned without setting
        # the terminal. Fall back to a model_error so callers don't
        # see ``None`` and crash.
        holder.value = Terminal(
            reason="model_error",
            error=RuntimeError("query() returned without setting Terminal"),
        )
    return messages, holder.value
