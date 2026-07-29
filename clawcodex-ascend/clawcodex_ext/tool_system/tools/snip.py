"""SnipTool — extract conversation history fragments for context analysis.

Allows the agent to retrieve past message content by index range, role
filter, or search text.  The conversation history is read from
``context.messages`` (populated by the agent loop before each tool round).
"""

from __future__ import annotations

import json
from typing import Any

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..protocol import ToolResult
from clawcodex_ext.types.content_blocks import (
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from clawcodex_ext.types.messages import Message, MessageContent


def _extract_text_from_content(content: MessageContent, max_chars: int = 500) -> str:
    """Extract a plain-text preview from a ``MessageContent`` value.

    * If *content* is a plain ``str``, return it directly (truncated).
    * If *content* is a ``list[ContentBlock]``, join text blocks and
      summarise tool blocks.
    """
    if isinstance(content, str):
        text = content.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\u2026"
        return text

    parts: list[str] = []
    for block in content:
        if isinstance(block, TextBlock):
            t = block.text.strip()
            if len(t) > max_chars:
                t = t[:max_chars] + "\u2026"
            parts.append(t)
        elif isinstance(block, ToolUseBlock):
            parts.append(f"[ToolUse: {block.name}({json.dumps(block.input, ensure_ascii=False)})]")
        elif isinstance(block, ToolResultBlock):
            status = "error" if block.is_error else "ok"
            content_preview = ""
            if isinstance(block.content, str):
                content_preview = block.content[:80].replace("\n", " ")
            parts.append(f"[ToolResult({status}): {content_preview}]")
        else:
            parts.append(f"[{type(block).__name__}]")
    return "\n".join(parts)


def _format_message_text(msg: Message, include_metadata: bool = False) -> str:
    """Format a single message as one-line human-readable text."""
    content_preview = _extract_text_from_content(msg.content)
    if include_metadata:
        return (
            f"[{msg.role}] {msg.timestamp[:19] if msg.timestamp else ''} uuid={msg.uuid[:8]}\u2026  {content_preview}"
        )
    return f"[{msg.role}] {content_preview}"


def _format_message_json(msg: Message) -> dict[str, Any]:
    """Format a single message as a JSON-serialisable dict."""
    return {
        "role": msg.role,
        "type": msg.type,
        "uuid": msg.uuid,
        "timestamp": msg.timestamp,
        "content": _extract_text_from_content(msg.content, max_chars=2000),
    }


def _format_message_summary(msg: Message) -> str:
    """Produce a very compact one-line summary of a message."""
    role = msg.role
    content = msg.content
    if isinstance(content, str):
        text = content.strip()
        word_count = len(text.split()) if text else 0
        return f"[{role}] {word_count} words"
    if isinstance(content, list):
        text_blocks = sum(1 for b in content if isinstance(b, TextBlock))
        tool_uses = sum(1 for b in content if isinstance(b, ToolUseBlock))
        tool_results = sum(1 for b in content if isinstance(b, ToolResultBlock))
        parts = []
        if text_blocks:
            parts.append(f"{text_blocks} text")
        if tool_uses:
            tool_names = [b.name for b in content if isinstance(b, ToolUseBlock)]
            parts.append(f"{tool_uses} tool({', '.join(tool_names)})")
        if tool_results:
            parts.append(f"{tool_results} result")
        return f"[{role}] {', '.join(parts)}" if parts else f"[{role}] (empty)"
    return f"[{role}] (unknown)"


def _snip_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Execute the SnipTool — extract conversation history fragments."""
    # ── read parameters ──────────────────────────────────────────────
    start: int = tool_input.get("start", -5)
    count: int = tool_input.get("count", 5)
    role_filter: str | None = tool_input.get("role")
    query: str | None = tool_input.get("query")
    output_format: str = tool_input.get("format", "text")
    include_metadata: bool = bool(tool_input.get("include_metadata", False))

    messages: list[Message] = list(getattr(context, "messages", []) or [])
    if not messages:
        return ToolResult(
            name="Snip",
            output="(no conversation history available)",
        )

    # ── apply filters ────────────────────────────────────────────────
    if role_filter:
        role_filter_lower = role_filter.lower()
        messages = [m for m in messages if m.role == role_filter_lower]

    if query:
        query_lower = query.lower()
        filtered: list[Message] = []
        for m in messages:
            content_text = (
                m.content
                if isinstance(m.content, str)
                else " ".join(
                    b.text if isinstance(b, TextBlock) else json.dumps(getattr(b, "input", {}), ensure_ascii=False)
                    for b in (m.content or [])
                    if isinstance(b, (TextBlock, ToolUseBlock))
                )
            )
            if query_lower in content_text.lower():
                filtered.append(m)
        messages = filtered

    if not messages:
        return ToolResult(
            name="Snip",
            output="(no messages match the specified filters)",
        )

    # ── slice by start / count ──────────────────────────────────────
    total = len(messages)
    if start < 0:
        # negative index: count from the end
        actual_start = max(0, total + start)
    else:
        actual_start = min(start, total - 1)

    actual_end = min(actual_start + count, total)
    sliced = messages[actual_start:actual_end]

    if not sliced:
        return ToolResult(
            name="Snip",
            output=f"(start index {start} out of range, total messages: {total})",
        )

    # ── format output ────────────────────────────────────────────────
    if output_format == "json":
        entries = [_format_message_json(m) for m in sliced]
        output = json.dumps(
            {
                "total_messages": total,
                "returned": len(sliced),
                "slice": [actual_start, actual_end],
                "messages": entries,
            },
            ensure_ascii=False,
            indent=2,
        )
    elif output_format == "summary":
        lines = [_format_message_summary(m) for m in sliced]
        header = f"--- Snip ({actual_start}:{actual_end} of {total}) ---"
        footer = f"--- end snip ({len(sliced)} messages) ---"
        output = "\n".join([header] + lines + [footer])
    else:
        # default: text
        lines: list[str] = []
        for i, msg in enumerate(sliced):
            idx = actual_start + i
            line = _format_message_text(msg, include_metadata=include_metadata)
            lines.append(f"#{idx}: {line}")
        header = f"--- Snip ({actual_start}:{actual_end} of {total}) ---"
        footer = f"--- end snip ({len(sliced)} messages) ---"
        output = "\n".join([header] + lines + [footer])

    # ── build result data ────────────────────────────────────────────
    data = {
        "total_messages": total,
        "returned": len(sliced),
        "slice": [actual_start, actual_end],
        "format": output_format,
        "text": output,
    }
    return ToolResult(name="Snip", output=data)


SnipTool: Tool = build_tool(
    name="Snip",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "start": {
                "type": "integer",
                "description": (
                    "Starting message index (0-based).  Negative values "
                    "count from the end (e.g. -5 = last 5 messages).  "
                    "Default: -5."
                ),
            },
            "count": {
                "type": "integer",
                "description": ("Number of consecutive messages to return.  Default: 5."),
            },
            "role": {
                "type": "string",
                "enum": ["user", "assistant", "system", "tool"],
                "description": ("If set, only messages with this role are returned.  Omit to include all roles."),
            },
            "query": {
                "type": "string",
                "description": (
                    "Case-insensitive substring search applied to message "
                    "content.  Only messages whose text content matches "
                    "are returned."
                ),
            },
            "format": {
                "type": "string",
                "enum": ["text", "json", "summary"],
                "description": (
                    "Output format.  'text' (default) produces human-"
                    "readable lines with role + content preview.  'json' "
                    "returns structured data.  'summary' shows compact "
                    "per-message statistics."
                ),
            },
            "include_metadata": {
                "type": "boolean",
                "description": (
                    "When true, each message line includes timestamp and "
                    "UUID prefix.  Only applies to 'text' format.  "
                    "Default: false."
                ),
            },
        },
    },
    call=_snip_call,
    prompt=(
        "Snip Tool: extract fragments of conversation history for "
        "context analysis.  Use it to recall what the user asked earlier, "
        "what tool results were returned, or how the assistant reasoned "
        "in prior turns.  Parameters: start (index, negative = from end), "
        "count (number of messages), role (user/assistant/system filter), "
        "query (substring search), format (text/json/summary).  The tool "
        "is read-only and concurrency-safe."
    ),
    description=(
        "Extract conversation history fragments by index range, role, or search — useful for recalling past context."
    ),
    aliases=("context_snip", "history_snip"),
    strict=True,
    max_result_size_chars=50_000,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
)
