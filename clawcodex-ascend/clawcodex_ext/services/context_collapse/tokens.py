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

"""Token counting primitives for context collapse.

The token counter is a small pluggable interface so the rest of the
context-collapse service can be developed and tested without depending
on a specific tokenizer. The default chain is:

  1. :class:`TiktokenCounter` — uses ``tiktoken`` with a per-model
     encoding. The most accurate; preferred for any production use.
  2. :class:`HeuristicTokenCounter` — falls back to a simple
     character-to-token ratio (~4 chars per token) when tiktoken is
     unavailable. Useful for tests and offline environments.
  3. :class:`CharTokenCounter` — exact character count; useful for
     very fast upper-bound estimation.

The chain is exposed via :class:`FallbackTokenCounter`, which tries
each registered counter in order and returns the first successful
result. This means callers can register a more accurate or more
specialized counter at any layer without changing the rest of the
system.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .exceptions import TokenCountUnavailableError


@runtime_checkable
class TokenCounter(Protocol):
    """Pluggable token counter."""

    def count(self, text: str) -> int: ...

    def count_messages(self, messages: Iterable[Any]) -> int: ...


@dataclass(frozen=True)
class TokenEstimate:
    """A token count plus the name of the counter that produced it.

    The ``counter_name`` is informational — it lets the audit log
    record which counter was used so a regression to a coarser
    counter is detectable.
    """

    tokens: int
    counter_name: str


_CHARS_PER_TOKEN = 4  # OpenAI's published rule of thumb.


class CharTokenCounter:
    """Counts tokens as ``len(text) // CHARS_PER_TOKEN``.

    Useful as a fast upper bound. Not accurate for languages with
    multi-byte characters but never throws and never blocks.
    """

    name = "char"

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("CharTokenCounter.count expects a string")
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def count_messages(self, messages: Iterable[Any]) -> int:
        total = 0
        for msg in messages:
            text = _message_to_text(msg)
            if text:
                total += self.count(text)
            # Add a small overhead per message to account for role
            # tokens and structural framing.
            total += 4
        return total


class HeuristicTokenCounter:
    """Word-count heuristic: ``round(len(words) * 1.3)``.

    Slightly more accurate than ``CharTokenCounter`` for English text
    but still cheap and dependency-free.
    """

    name = "heuristic"

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("HeuristicTokenCounter.count expects a string")
        if not text:
            return 0
        words = len(text.split())
        return max(1, round(words * 1.3))

    def count_messages(self, messages: Iterable[Any]) -> int:
        total = 0
        for msg in messages:
            text = _message_to_text(msg)
            if text:
                total += self.count(text)
            total += 4
        return total


class TiktokenCounter:
    """Counter backed by the ``tiktoken`` library.

    The encoding is looked up by model name; if the model is unknown
    to tiktoken, the caller can supply an explicit ``encoding_name``
    (e.g. ``"cl100k_base"``) instead.

    Imports are done lazily so the module is usable even when
    ``tiktoken`` is not installed. The counter raises
    :class:`TokenCountUnavailableError` if the library cannot be
    imported.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        encoding_name: str | None = None,
    ) -> None:
        if not model and not encoding_name:
            raise ValueError("TiktokenCounter requires model or encoding_name")
        self._model = model
        self._encoding_name = encoding_name

    @property
    def name(self) -> str:
        return f"tiktoken:{self._encoding_name or self._model}"

    def _get_encoding(self) -> Any:
        try:
            import tiktoken  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised in env without tiktoken
            raise TokenCountUnavailableError(
                "tiktoken is not installed; use HeuristicTokenCounter or register one"
            ) from exc
        if self._encoding_name:
            return tiktoken.get_encoding(self._encoding_name)
        return tiktoken.encoding_for_model(self._model)

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("TiktokenCounter.count expects a string")
        enc = self._get_encoding()
        return len(enc.encode(text, disallowed_special=()))

    def count_messages(self, messages: Iterable[Any]) -> int:
        # Approximate the per-message overhead that the OpenAI tokenizer
        # adds (role + name + structural framing).
        per_message_overhead = 3
        tokens_per_name = 1
        total = 0
        for msg in messages:
            total += per_message_overhead
            name = getattr(msg, "name", None)
            if isinstance(name, str) and name:
                total += tokens_per_name
                total += len(name)
            text = _message_to_text(msg)
            if text:
                total += self.count(text)
        # Every reply is primed with ``assistant: ``.
        total += 3
        return total


class FallbackTokenCounter:
    """Try each registered counter in order; return the first result.

    Useful for environments where tiktoken may or may not be available:
    the application registers ``[TiktokenCounter, HeuristicTokenCounter]``
    and the first one to succeed wins.
    """

    def __init__(self, counters: Iterable[TokenCounter]) -> None:
        self._counters: list[TokenCounter] = list(counters)
        if not self._counters:
            raise ValueError("FallbackTokenCounter requires at least one counter")
        self._lock = threading.RLock()

    def count(self, text: str) -> int:
        return self._first_working().count(text)

    def count_messages(self, messages: Iterable[Any]) -> int:
        return self._first_working().count_messages(messages)

    def estimate(self, text: str) -> TokenEstimate:
        counter = self._first_working()
        return TokenEstimate(tokens=counter.count(text), counter_name=counter.name)

    def estimate_messages(self, messages: Iterable[Any]) -> TokenEstimate:
        counter = self._first_working()
        return TokenEstimate(
            tokens=counter.count_messages(messages),
            counter_name=counter.name,
        )

    def add_counter(self, counter: TokenCounter) -> None:
        with self._lock:
            self._counters.insert(0, counter)

    def _first_working(self) -> TokenCounter:
        with self._lock:
            last_exc: Exception | None = None
            for c in self._counters:
                try:
                    # Probe with a tiny string to detect import errors
                    # before the real call.
                    c.count("a")
                    return c
                except TokenCountUnavailableError as exc:
                    last_exc = exc
                    continue
                except Exception as exc:
                    last_exc = exc
                    continue
        raise TokenCountUnavailableError(f"no usable token counter in chain: {last_exc!r}")


def _message_to_text(msg: Any) -> str:
    """Best-effort extraction of text from a Message-like object.

    Supports the project's ``Message`` dataclass via ``content``
    blocks, the OpenAI-style ``{"role": ..., "content": ...}`` dict,
    or any object with a ``text`` attribute.
    """
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
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
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


# Pre-built chains for common deployments.
def tiktoken_first_then_heuristic(*, model: str = "gpt-4", encoding_name: str | None = None) -> FallbackTokenCounter:
    """Counter that prefers tiktoken, falling back to a word heuristic."""
    return FallbackTokenCounter(
        [
            TiktokenCounter(model=model, encoding_name=encoding_name),
            HeuristicTokenCounter(),
            CharTokenCounter(),
        ]
    )


def heuristic_only() -> FallbackTokenCounter:
    """Counter that never imports tiktoken; safe for offline tests."""
    return FallbackTokenCounter([HeuristicTokenCounter(), CharTokenCounter()])
