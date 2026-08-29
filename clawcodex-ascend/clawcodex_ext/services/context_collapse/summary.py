#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Pluggable summary generator for context collapse.

The collapse store accepts pre-computed summary strings, but the
generator that produces them is pluggable so the rest of the system
can run without a live LLM. Two implementations ship in this module:

* :class:`HeadlineSummaryGenerator` — extracts a single-line headline
  per message and stitches them into a compact outline. No LLM,
  deterministic, suitable for tests and offline runs.
* :class:`LLMSummaryGenerator` — calls a registered async callback
  (the application injects its LLM client). Used in production.

Both implement :class:`SummaryGenerator` so the rest of the system
can swap them at runtime.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from .exceptions import SummaryGeneratorError


@runtime_checkable
class SummaryGenerator(Protocol):
    """Generate a summary for a chunk of collapsed messages."""

    def summarize(self, messages: list[Any]) -> str: ...


_HEADLINE_RE = re.compile(r"^([^\n]{1,200})")
_WORD_RE = re.compile(r"\w+")
_WHITESPACE_RE = re.compile(r"\s+")


class HeadlineSummaryGenerator:
    """Produce a deterministic, no-LLM summary from message text.

    The output is a numbered list of one-line headlines (the first
    non-blank line of each message, trimmed to 200 chars) followed by
    an at-a-glance count of dropped/kept lines. This is deliberately
    not a real summary — it preserves enough structure that a human
    auditor can verify the collapse behavior without trusting an LLM.
    """

    name = "headline"

    def __init__(self, *, max_headlines: int = 32, max_chars: int = 8_000) -> None:
        self._max_headlines = max_headlines
        self._max_chars = max_chars
        self._lock = threading.RLock()

    def summarize(self, messages: list[Any]) -> str:
        with self._lock:
            return self._build(messages)

    def _build(self, messages: list[Any]) -> str:
        if not messages:
            return "(empty archive)"
        headlines: list[str] = []
        total_lines = 0
        kept_lines = 0
        for i, msg in enumerate(messages[: self._max_headlines], start=1):
            text = _extract_text(msg).strip()
            if not text:
                headlines.append(f"{i}. (empty)")
                continue
            lines = _WHITESPACE_RE.split(text)
            total_lines += len(lines)
            first = lines[0][:200]
            kept_lines += 1
            headlines.append(f"{i}. {first}")
        archive_count = max(0, len(messages) - self._max_headlines)
        dropped_lines = max(0, total_lines - kept_lines)
        body = "\n".join(headlines)
        tail = ""
        if archive_count:
            tail += f"\n…({archive_count} additional message(s) archived)"
        if dropped_lines:
            tail += f"\n[~{dropped_lines} non-headline line(s) elided]"
        out = f"[{len(messages)} archived message(s)]\n{body}{tail}"
        if len(out) > self._max_chars:
            out = out[: self._max_chars] + "\n[…truncated]"
        return out


LLMSummaryFn = Callable[[list[Any]], Awaitable[str]]


class LLMSummaryGenerator:
    """Summary generator that delegates to an injected async LLM callback.

    The callback receives the list of archived messages and must return
    a string. The generator wraps the call with timeout handling and
    a fallback to the headline generator if the LLM call fails or
    exceeds the configured timeout.
    """

    name = "llm"

    def __init__(
        self,
        fn: LLMSummaryFn,
        *,
        fallback: SummaryGenerator | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not callable(fn):
            raise TypeError("LLMSummaryGenerator requires a callable fn")
        self._fn = fn
        self._fallback = fallback or HeadlineSummaryGenerator()
        self._timeout_seconds = timeout_seconds
        self._lock = threading.RLock()

    async def summarize_async(self, messages: list[Any]) -> str:
        import asyncio

        with self._lock:
            try:
                return await asyncio.wait_for(
                    self._fn(list(messages)),
                    timeout=self._timeout_seconds,
                )
            except (asyncio.TimeoutError, Exception) as exc:
                # Use the headline generator as a safety net so the
                # collapse operation never silently produces an empty
                # summary.
                try:
                    return self._fallback.summarize(messages)
                except Exception:  # pragma: no cover - extremely defensive
                    raise SummaryGeneratorError(f"LLM summary failed ({exc!r}) and fallback also failed") from exc

    def summarize(self, messages: list[Any]) -> str:
        # Synchronous callers can still get a summary by falling back
        # to the headline generator. The async path is the recommended
        # one; the sync fallback exists so the rest of the codebase
        # can be tested without an event loop.
        return self._fallback.summarize(messages)


def _extract_text(msg: Any) -> str:
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                if isinstance(block, dict):
                    t = block.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "\n".join(parts)
        return ""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    text = getattr(msg, "text", None)
    if isinstance(text, str):
        return text
    return ""


# Public helper so the rest of the codebase can extract text without
# importing the private function above.
def extract_text(msg: Any) -> str:
    return _extract_text(msg)


# Convenience regex used by callers that want a quick "is this a real
# word?" check (e.g. summary-quality probes).
def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


# Re-export for tests.
__all__ = [
    "HeadlineSummaryGenerator",
    "LLMSummaryGenerator",
    "SummaryGenerator",
    "SummaryGeneratorError",
    "count_words",
    "extract_text",
]
