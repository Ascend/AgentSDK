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

# pylint: disable=too-many-nested-blocks
"""Agent tool filtering and resolution utilities.

Mirrors typescript/src/tools/AgentTool/agentToolUtils.ts.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from clawcodex_ext.permissions.types import PermissionMode
from clawcodex_ext.tool_system.build_tool import Tool, Tools, tool_matches_name
from clawcodex_ext.types.content_blocks import ToolUseBlock
from clawcodex_ext.types.messages import AssistantMessage, Message

from clawcodex_ext.agent.constants import (
    AGENT_TOOL_NAME,
    ALL_AGENT_DISALLOWED_TOOLS,
    ASYNC_AGENT_ALLOWED_TOOLS,
    CUSTOM_AGENT_DISALLOWED_TOOLS,
)

logger = logging.getLogger(__name__)


def get_query_source_for_agent(agent_type: str, is_built_in: bool) -> str:
    """ch08 round-4 WI-2 — the subagent query_source label.

    Port of TS ``getQuerySourceForAgent`` (``utils/promptCategory.ts``):
    ``agent:builtin:<type>`` for built-in agents, ``agent:custom`` for
    user/project agents. (The fork path threads its own
    ``agent:builtin:fork`` explicitly.) The ``agent:`` prefix keeps these
    outside the ``("compact","session_memory")`` skip-set and inside the
    foreground-retry family, matching TS.
    """
    if is_built_in:
        return f"agent:builtin:{agent_type}"
    return "agent:custom"


@dataclass
class ResolvedAgentTools:
    """Result of resolving agent tools against available tools."""

    has_wildcard: bool
    valid_tools: list[str]
    invalid_tools: list[str]
    resolved_tools: Tools
    allowed_agent_types: list[str] | None = None


def filter_tools_for_agent(
    *,
    tools: Tools,
    is_built_in: bool,
    is_async: bool = False,
    permission_mode: PermissionMode | None = None,
) -> Tools:
    """Filter available tools based on agent type and mode.

    Mirrors filterToolsForAgent() from typescript/src/tools/AgentTool/agentToolUtils.ts.

    - MCP tools are always allowed for all agents.
    - ExitPlanMode is allowed for agents in plan mode.
    - ALL_AGENT_DISALLOWED_TOOLS are always blocked.
    - CUSTOM_AGENT_DISALLOWED_TOOLS are blocked for non-built-in agents.
    - Async agents are restricted to ASYNC_AGENT_ALLOWED_TOOLS whitelist.
    """
    result: Tools = []
    for tool in tools:
        # MCP tools always allowed
        if tool.name.startswith("mcp__") or tool.is_mcp:
            result.append(tool)
            continue

        # Allow ExitPlanMode for agents in plan mode
        if tool_matches_name(tool, "ExitPlanMode") and permission_mode == "plan":
            result.append(tool)
            continue

        # Block ALL_AGENT_DISALLOWED_TOOLS
        if tool.name in ALL_AGENT_DISALLOWED_TOOLS:
            continue

        # Block CUSTOM_AGENT_DISALLOWED_TOOLS for non-built-in agents
        if not is_built_in and tool.name in CUSTOM_AGENT_DISALLOWED_TOOLS:
            continue

        # Async agents: only whitelisted tools
        if is_async and tool.name not in ASYNC_AGENT_ALLOWED_TOOLS:
            continue

        result.append(tool)

    return result


def filter_tools_for_startup_agent(
    tools: Tools,
    startup_agent: Any | None,
) -> Tools:
    """Apply a bundle overview agent's tool allowlist to the main-loop tool pool."""
    if startup_agent is None:
        return list(tools)

    resolved = resolve_agent_tools(startup_agent, tools)
    if resolved.invalid_tools:
        logger.warning(
            "Startup agent %r has invalid tools: %s",
            getattr(startup_agent, "agent_type", startup_agent),
            resolved.invalid_tools,
        )

    filtered = list(resolved.resolved_tools)
    present = {tool.name for tool in filtered}
    requested = getattr(startup_agent, "tools", None) or []
    if AGENT_TOOL_NAME in requested and AGENT_TOOL_NAME not in present:
        for tool in tools:
            if tool.name == AGENT_TOOL_NAME:
                filtered.append(tool)
                break
    filtered = _preserve_goal_infrastructure_tools(filtered, tools, startup_agent)
    return _apply_bundle_tool_filter(filtered)


def _preserve_goal_infrastructure_tools(
    filtered: Tools,
    available_tools: Tools,
    startup_agent: Any,
) -> Tools:
    """Keep goal model tools after startup-agent filtering when already visible."""
    try:
        from clawcodex_ext.goal.tools import GOAL_MODEL_TOOL_NAMES
    except ImportError:
        return filtered

    disallowed = {_extract_tool_name(spec) for spec in (getattr(startup_agent, "disallowed_tools", None) or [])}
    present = {tool.name for tool in filtered}
    result = list(filtered)
    for tool in available_tools:
        if tool.name not in GOAL_MODEL_TOOL_NAMES:
            continue
        if tool.name in present or tool.name in disallowed:
            continue
        result.append(tool)
        present.add(tool.name)
    return result


def _apply_bundle_tool_filter(tools: Tools) -> Tools:
    try:
        from extensions.sop_converter.bundle_context import filter_tools_for_bundle

        return filter_tools_for_bundle(tools)
    except ImportError:
        return tools


def resolve_agent_tools(
    agent_definition: Any,
    available_tools: Tools,
    is_async: bool = False,
) -> ResolvedAgentTools:
    """Resolve and validate agent tools against available tools.

    Mirrors resolveAgentTools() from typescript/src/tools/AgentTool/agentToolUtils.ts.

    Handles wildcard expansion, validation, and disallowed tool filtering.
    """

    agent_tools = agent_definition.tools
    disallowed_tools = agent_definition.disallowed_tools
    source = agent_definition.source
    permission_mode = agent_definition.permission_mode

    # Apply base filtering
    filtered_available_tools = filter_tools_for_agent(
        tools=available_tools,
        is_built_in=(source == "built-in"),
        is_async=is_async,
        permission_mode=permission_mode,
    )

    # Build disallowed set
    disallowed_set: set[str] = set()
    if disallowed_tools:
        for tool_spec in disallowed_tools:
            # Extract tool name from spec (may include pattern like "Tool(arg)")
            tool_name = _extract_tool_name(tool_spec)
            disallowed_set.add(tool_name)

    # Filter by disallowed list
    allowed_available_tools = [t for t in filtered_available_tools if t.name not in disallowed_set]

    # If tools is None or ['*'], allow all tools (after filtering)
    has_wildcard = agent_tools is None or (len(agent_tools) == 1 and agent_tools[0] == "*")
    if has_wildcard:
        return ResolvedAgentTools(
            has_wildcard=True,
            valid_tools=[],
            invalid_tools=[],
            resolved_tools=allowed_available_tools,
        )

    # Build map of available tools (name + aliases + normalized variants)
    available_map: dict[str, Tool] = {}
    for tool in allowed_available_tools:
        available_map[tool.name] = tool
        # Index aliases so agent markdown can use dot-separated original names
        for alias in tool.aliases or ():
            if alias not in available_map:
                available_map[alias] = tool
        # Index the "reversed" normalization: kebab → dots (for backward compat)
        normalized_dots = tool.name.replace("-", ".").replace("_", ".")
        if normalized_dots != tool.name and normalized_dots not in available_map:
            available_map[normalized_dots] = tool

    valid_tools: list[str] = []
    invalid_tools: list[str] = []
    resolved: list[Tool] = []
    resolved_set: set[str] = set()
    allowed_agent_types: list[str] | None = None

    for tool_spec in agent_tools:
        tool_name = _extract_tool_name(tool_spec)
        rule_content = _extract_rule_content(tool_spec)

        # Special case: Agent tool carries allowedAgentTypes
        if tool_name == AGENT_TOOL_NAME:
            if rule_content:
                allowed_agent_types = [s.strip() for s in rule_content.split(",")]
            valid_tools.append(tool_spec)
            continue

        tool = available_map.get(tool_name)
        if tool is None:
            # Try kebab-case normalization: "LLM.invoke" → "llm-invoke"
            tool = available_map.get(_normalize_tool_name(tool_name))
        if tool:
            valid_tools.append(tool_spec)
            if tool.name not in resolved_set:
                resolved.append(tool)
                resolved_set.add(tool.name)
        else:
            invalid_tools.append(tool_spec)

    return ResolvedAgentTools(
        has_wildcard=False,
        valid_tools=valid_tools,
        invalid_tools=invalid_tools,
        resolved_tools=resolved,
        allowed_agent_types=allowed_agent_types,
    )


@dataclass
class AgentToolResult:
    """Result data from a completed agent run."""

    agent_id: str
    agent_type: str
    content: list[dict[str, Any]]
    total_duration_ms: int
    total_tokens: int
    total_tool_use_count: int
    # ``truncated=True`` when the assistant text was clipped to fit under
    # ``DEFAULT_TRUNCATE_CHAR_LIMIT`` / ``DEFAULT_TRUNCATE_LINE_LIMIT``.
    # The full transcript lives at ``transcript_path`` (relative to the
    # session dir) so callers can ``TaskOutput`` into it for the rest.
    # Defaults preserve the pre-WI-2.6 positional-arg signature, so
    # direct construction in tests / fixtures keeps working.
    truncated: bool = False
    transcript_path: str | None = None


def count_tool_uses(messages: list[Message]) -> int:
    """Count the number of tool_use blocks across all assistant messages."""
    count = 0
    for m in messages:
        if isinstance(m, AssistantMessage):
            content = m.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolUseBlock):
                        count += 1
    return count


def finalize_agent_tool(
    agent_messages: list[Message],
    agent_id: str,
    metadata: dict[str, Any],
    *,
    progress: Any | None = None,
    last_assistant_msg: Message | None = None,
    transcript_path: str | None = None,
) -> AgentToolResult:
    """Extract final result from agent messages.

    Mirrors finalizeAgentTool() from typescript/src/tools/AgentTool/agentToolUtils.ts.

    Chapter-10 / Chunk C / WI-2.4: token aggregation now goes through
    ``ProgressTracker``. Pre-WI-2.4 ``total_tokens`` was hard-coded to
    ``0`` (gap-analysis §2.2 row); post-WI-2.4 callers pass the live
    tracker via the ``progress`` keyword and we read the chapter-correct
    ``latest_input_tokens + cumulative_output_tokens`` total off it. If
    the caller can't supply a tracker (e.g. a sync agent that didn't
    feed one during iteration), we fall back to recomputing from
    ``message.usage`` so the behavior degrades gracefully rather than
    silently returning zero.

    Chapter-12 / WI-2.6: a sync subagent that produced multi-MB reports
    would OOM the parent session because the entire last-assistant text
    was inlined into the parent ``state.messages`` tool_result. The fix
    is two-pronged: callers that already know the last assistant (the
    streaming path) pass it via ``last_assistant_msg`` to skip the
    ``reversed(agent_messages)`` scan; and the resulting text is run
    through :func:`_truncate_text_blocks` so the parent only ever sees
    a bounded preview. The full transcript lives at ``transcript_path``
    for ``TaskOutput`` recovery.
    """
    # Find the last assistant message — prefer the caller's already-known
    # value (streaming path tracks it incrementally; the reversed scan
    # would be O(n) for no benefit). Fall back to the historical scan for
    # callers that don't track it.
    last_assistant: AssistantMessage | None = None
    if last_assistant_msg is not None and isinstance(last_assistant_msg, AssistantMessage):
        last_assistant = last_assistant_msg
    else:
        for msg in reversed(agent_messages):
            if isinstance(msg, AssistantMessage):
                last_assistant = msg
                break

    if last_assistant is None:
        raise ValueError("No assistant messages found")

    # Extract text content from the agent's response
    content: list[dict[str, Any]] = []
    raw_content = last_assistant.content
    if isinstance(raw_content, str):
        if raw_content:
            content = [{"type": "text", "text": raw_content}]
    elif isinstance(raw_content, list):
        for block in raw_content:
            if hasattr(block, "type") and block.type == "text":
                content.append({"type": "text", "text": block.text})

    # If no text content in last message, search backwards
    if not content:
        for msg in reversed(agent_messages):
            if isinstance(msg, AssistantMessage):
                raw = msg.content
                if isinstance(raw, list):
                    for block in raw:
                        if hasattr(block, "type") and block.type == "text":
                            content.append({"type": "text", "text": block.text})
                    if content:
                        break

    # WI-2.6: bound the parent-visible content so a multi-MB subagent
    # report doesn't OOM the parent. The full text is recoverable via
    # ``transcript_path`` + ``TaskOutput``. ``truncate`` is a no-op when
    # the content is already under the limit.
    truncated, content = _truncate_text_blocks(
        content,
        char_limit=DEFAULT_TRUNCATE_CHAR_LIMIT,
        line_limit=DEFAULT_TRUNCATE_LINE_LIMIT,
        transcript_path=transcript_path,
    )

    total_tool_use_count = count_tool_uses(agent_messages)
    start_time = metadata.get("start_time", time.time())
    duration_ms = int((time.time() - start_time) * 1000)
    total_tokens = _resolve_total_tokens(progress, agent_messages)

    return AgentToolResult(
        agent_id=agent_id,
        agent_type=metadata.get("agent_type", ""),
        content=content,
        total_duration_ms=duration_ms,
        total_tokens=total_tokens,
        total_tool_use_count=total_tool_use_count,
        truncated=truncated,
        transcript_path=transcript_path,
    )


# WI-2.6: text-block truncation thresholds. Tuned so the parent
# tool_result stays under ~8 KB regardless of how chatty the subagent
# was; line cap mirrors a reasonable "summary"-shaped answer. These are
# conservative — the OOM repro on WSL2 (3.8 GB RAM) only needed ~9 KB
# of pre-truncation content to wedge the parent context.
DEFAULT_TRUNCATE_CHAR_LIMIT: int = 8_192
DEFAULT_TRUNCATE_LINE_LIMIT: int = 200


def _truncate_text_blocks(
    content: list[dict[str, Any]],
    *,
    char_limit: int,
    line_limit: int,
    transcript_path: str | None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Bound the size of ``content`` text blocks for parent-side injection.

    Joins every ``{"type": "text", "text": "..."}`` block, applies the
    char + line cap, and (if either cap fires) replaces the trailing
    content with a single text block containing the head of the original
    text plus a one-line notice pointing at the on-disk transcript.
    Non-text blocks (e.g. tool_use placeholders) are dropped — by the
    point we get here the assistant has already finished its work, so
    tool_use in the "last message" would be a model error, and the
    transcript is the authoritative record.

    Returns ``(truncated, new_content)``:
      * ``truncated`` is True iff either cap fired.
      * ``new_content`` is always a single-element list when
        truncated, otherwise the input list is returned untouched.
    """
    if not content:
        return False, content

    text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
    if not text_blocks:
        return False, content

    joined = "\n".join(str(b.get("text", "")) for b in text_blocks)
    total_chars = len(joined)
    total_lines = joined.count("\n") + (0 if joined.endswith("\n") else 1)

    if total_chars <= char_limit and total_lines <= line_limit:
        return False, content

    # Truncate: keep head of the text, drop the rest, append notice.
    kept = joined
    if len(kept) > char_limit:
        kept = kept[:char_limit]
    # Line cap: walk the kept string and stop after ``line_limit`` newlines.
    if kept.count("\n") >= line_limit:
        # Find the position of the line_limit-th newline and cut there.
        cut_pos = -1
        seen = 0
        for idx, ch in enumerate(kept):
            if ch == "\n":
                seen += 1
                if seen == line_limit:
                    cut_pos = idx
                    break
        if cut_pos >= 0:
            kept = kept[:cut_pos]

    if transcript_path:
        notice = (
            f"\n\n[truncated: showing first {len(kept)} chars / "
            f"{kept.count(chr(10)) + (0 if kept.endswith(chr(10)) else 1)} lines "
            f"of {total_chars} chars — full transcript at {transcript_path}]"
        )
    else:
        notice = (
            f"\n\n[truncated: showing first {len(kept)} chars / "
            f"{kept.count(chr(10)) + (0 if kept.endswith(chr(10)) else 1)} lines "
            f"of {total_chars} chars — transcript not available]"
        )

    return True, [{"type": "text", "text": kept + notice}]


def _resolve_total_tokens(
    progress: Any | None,
    agent_messages: list[Message],
) -> int:
    """Compute the chapter-correct total token count.

    Preferred path: if the caller fed a ``ProgressTracker`` during
    iteration, read its accumulated totals. Fallback: recompute from
    ``message.usage`` payloads using the same arithmetic semantics
    (cumulative-per-call inputs → keep latest; per-turn outputs → sum).

    Local imports (here and below) defer the cycle: ``agent_tool_utils``
    must not import ``src.tasks.progress`` at module load because some
    test fixtures construct AgentToolResult directly.
    """
    if progress is not None:
        try:
            from clawcodex_ext.tasks.progress import (
                ProgressTracker,
                total_tokens_from_tracker,
            )

            if isinstance(progress, ProgressTracker):
                return total_tokens_from_tracker(progress)
        except ImportError:
            pass  # Fall through to message-based recompute.

    # Fallback: walk the messages and apply the same arithmetic the
    # tracker would have used. Defensive: a sync agent that didn't
    # feed a tracker still gets a non-zero total instead of WI-2.4's
    # pre-fix ``total_tokens=0``.
    latest_input = 0
    cumulative_output = 0
    for msg in agent_messages:
        if not isinstance(msg, AssistantMessage):
            continue
        usage = msg.usage if isinstance(msg.usage, dict) else None
        if usage is None:
            continue
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        latest_input = input_tokens + cache_creation + cache_read
        cumulative_output += int(usage.get("output_tokens", 0) or 0)
    return latest_input + cumulative_output


def extract_partial_result(messages: list[Message]) -> str | None:
    """Extract a partial result string from agent messages.

    Used when an async agent is killed to preserve what it accomplished.
    """
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            raw = msg.content
            if isinstance(raw, str) and raw.strip():
                return raw
            if isinstance(raw, list):
                texts = []
                for block in raw:
                    if hasattr(block, "type") and block.type == "text" and block.text:
                        texts.append(block.text)
                if texts:
                    return "\n".join(texts)
    return None


def _extract_tool_name(tool_spec: str) -> str:
    """Extract the base tool name from a spec like 'Tool(arg)'."""
    paren_idx = tool_spec.find("(")
    if paren_idx != -1:
        return tool_spec[:paren_idx].strip()
    return tool_spec.strip()


def _extract_rule_content(tool_spec: str) -> str | None:
    """Extract rule content from a spec like 'Tool(arg1, arg2)'."""
    paren_idx = tool_spec.find("(")
    if paren_idx != -1 and tool_spec.endswith(")"):
        return tool_spec[paren_idx + 1 : -1].strip()
    return None


def _normalize_tool_name(name: str) -> str:
    """Normalize an agent-markdown tool name for ToolRegistry lookup.

    Converts dot.separated / snake_case → kebab-case so that names
    written by ``pos convert`` (e.g. ``"LLM.invoke"``, ``"video_ops.transcode"``)
    can find their registered counterparts (``"llm-invoke"``, ``"video-ops-transcode"``).

    Only dots, double-underscores, and single underscores act as word
    separators.  CamelCase within a segment stays as one word.

    >>> _normalize_tool_name("LLM.invoke")
    'llm-invoke'
    >>> _normalize_tool_name("video_ops.transcode")
    'video-ops-transcode'
    >>> _normalize_tool_name("already-kebab")
    'already-kebab'
    >>> _normalize_tool_name("VideoProcessor.transcode")
    'videoprocessor-transcode'
    """
    s = name.replace(".", "-").replace("__", "-")
    s = s.replace("_", "-")
    s = re.sub(r"-+", "-", s)
    return s.strip("-").lower()
