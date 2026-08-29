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

"""Outbound text helpers: Markdown→plain-text strip + long-message handling.

WeChat iLink declares ``supports_markdown=False``; the
:class:`OutboundDispatcher` strips Markdown to plain text before
sending, and falls back to plain text on platform rejection. Long
messages are split into chunks; when a message exceeds the chunk
threshold it is truncated and a LiveView link is appended rather than
flooding the channel with dozens of fragments.
"""

from __future__ import annotations

import re

DEFAULT_CHUNK_SIZE = 4000
DEFAULT_MAX_CHUNKS = 4  # > this -> truncate + LiveView link
TRUNCATION_NOTICE = "(Content truncated"


def strip_markdown(text: str) -> str:
    """Best-effort Markdown → plain-text conversion.

    Removes code fences, inline code, bold/italic, headers, links,
    images, list markers, and blockquotes while preserving readable
    text. Not a full Markdown parser — intentionally conservative.
    """
    if not text:
        return ""
    out = text
    # Fenced code blocks: keep the inner content as plain text.
    out = re.sub(r"```[^\n]*\n?", "", out)
    out = out.replace("```", "")
    # Inline code
    out = re.sub(r"`([^`]+)`", r"\1", out)
    # Images ![alt](url) -> alt
    out = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", out)
    # Links [text](url) -> text
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)
    # Headers
    out = re.sub(r"^\s{0,3}#{1,6}\s+", "", out, flags=re.MULTILINE)
    # Bold/italic
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"__([^_]+)__", r"\1", out)
    out = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", out)
    out = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", out)
    # Strikethrough
    out = re.sub(r"~~([^~]+)~~", r"\1", out)
    # Blockquotes
    out = re.sub(r"^\s{0,3}>\s?", "", out, flags=re.MULTILINE)
    # Unordered list markers
    out = re.sub(r"^\s*[-*+]\s+", "", out, flags=re.MULTILINE)
    # Ordered list markers
    out = re.sub(r"^\s*\d+\.\s+", "", out, flags=re.MULTILINE)
    # Horizontal rules
    out = re.sub(r"^\s*[-*_]{3,}\s*$", "", out, flags=re.MULTILINE)
    return out.strip()


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Split ``text`` into ``chunk_size``-char chunks on paragraph/line boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", 0, chunk_size)
        if cut == -1:
            cut = remaining.rfind(" ", 0, chunk_size)
        if cut == -1:
            cut = chunk_size
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return [c for c in chunks if c]


def maybe_truncate_with_liveview(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    liveview_url: str | None = None,
) -> list[str]:
    """Truncate text and publish the full value to LiveView when needed."""
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than zero")
    chunks = split_text(text, chunk_size)
    if len(chunks) <= max_chunks:
        return chunks
    notice = TRUNCATION_NOTICE
    if liveview_url:
        notice += f"; view the complete content in LiveView: {liveview_url}"
    notice += ")"
    if len(notice) > chunk_size:
        notice = notice[:chunk_size]
    body_budget = max(0, chunk_size - len(notice) - 2)
    final = notice
    if body_budget:
        final = f"{chunks[max_chunks - 1][:body_budget].rstrip()}\n\n{notice}"
    return [*chunks[: max_chunks - 1], final]


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MAX_CHUNKS",
    "maybe_truncate_with_liveview",
    "split_text",
    "strip_markdown",
]
