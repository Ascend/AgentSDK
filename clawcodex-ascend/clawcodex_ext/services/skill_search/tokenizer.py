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

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Literal

# ---------------------------------------------------------------------------
# Word tokenizer type
# ---------------------------------------------------------------------------

WordTokenizer = Callable[[str], list[str]]

# ---------------------------------------------------------------------------
# LangProcessor — abstract base
# ---------------------------------------------------------------------------


class LangProcessor(ABC):
    """Language-aware tokenizer for a specific script or character range.

    Subclasses must implement:
    - ``can_handle(char)``: whether this processor handles the character.
    - ``tokenize_segment(segment)``: split a pure-script segment into tokens.

    Processors are ordered by ``priority`` (lower = higher priority).
    When multiple processors can handle the same character, the one with
    the lowest priority wins.
    """

    @abstractmethod
    def can_handle(self, char: str) -> bool: ...

    @abstractmethod
    def tokenize_segment(self, segment: str) -> list[str]: ...

    @property
    def priority(self) -> int:
        return 100


# ---------------------------------------------------------------------------
# FallbackProcessor — catches everything
# ---------------------------------------------------------------------------


class FallbackProcessor(LangProcessor):
    """Last-resort processor for characters not handled by any other processor.

    Strategy:
    - Whitespace and punctuation are silently dropped.
    - All other characters are kept as single-character tokens.

    This ensures no data is silently lost when scripts like Cyrillic,
    Thai, or Arabic appear without a dedicated processor.
    """

    def can_handle(self, char: str) -> bool:
        return True

    def tokenize_segment(self, segment: str) -> list[str]:
        result: list[str] = []
        for ch in segment:
            if not ch.isspace() and ch.isprintable():
                result.append(ch)
        return result

    @property
    def priority(self) -> int:
        return 9999


# ---------------------------------------------------------------------------
# LatinProcessor
# ---------------------------------------------------------------------------

# Regex for splitting camelCase / PascalCase / acronyms / numbers.
# Matches:
#   "[A-Z]?[a-z]+"          camelCase words: "Browser", "get"
#   "[A-Z]+(?=[A-Z][a-z])"  acronyms before camelCase: "XML" in "XMLParser"
#   "[A-Z]+"                standalone acronyms: "HTTP"
#   "[0-9]+"                digit sequences: "3"
_CAMEL_SPLIT_RE = re.compile(
    r"[A-Z]?[a-z]+"
    r"|[A-Z]+(?=[A-Z][a-z])"
    r"|[A-Z]+"
    r"|[0-9]+"
)

_DEFAULT_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "this",
        "that",
        "these",
        "those",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "from",
        "by",
        "as",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
    }
)


class LatinProcessor(LangProcessor):
    """Latin-script processor (English, etc.).

    Features:
    - camelCase / PascalCase splitting
    - snake_case and kebab-case delimiter splitting
    - digit separation
    - case folding (lowercase) — Latin-specific, not forced on other scripts
    - stopword filtering
    """

    def __init__(
        self,
        *,
        case_fold: bool = True,
        split_camel: bool = True,
        split_snake: bool = True,
        split_kebab: bool = True,
        stopwords: frozenset[str] | None = None,
    ):
        self._case_fold = case_fold
        self._split_camel = split_camel
        self._stopwords = stopwords if stopwords is not None else _DEFAULT_EN_STOPWORDS

        # Build a token-extraction regex: what characters can appear in a
        # token.  Disabled delimiters become part of the token character
        # class so they are not split away.
        token_chars = r"a-zA-Z0-9"  # nosec B105
        if not split_snake:
            token_chars += "_"
        if not split_kebab:
            token_chars += r"\-"
        self._token_re = re.compile(f"[{token_chars}]+")

    @property
    def priority(self) -> int:
        return 10

    def can_handle(self, char: str) -> bool:
        return char.isascii() and (char.isalnum() or char in "_-")

    def tokenize_segment(self, segment: str) -> list[str]:
        # Step 1: extract raw tokens (contiguous runs of valid chars)
        raw_tokens = self._token_re.findall(segment)

        # Step 2: camelCase/PascalCase split (if enabled)
        # Only camel-split pure alphanumeric tokens; tokens containing
        # preserved delimiters (e.g. "_" when split_snake=False) are
        # kept as-is to avoid _CAMEL_SPLIT_RE implicitly breaking on them.
        tokens: list[str] = []
        if self._split_camel:
            for token in raw_tokens:
                if token.isalnum():
                    tokens.extend(_CAMEL_SPLIT_RE.findall(token))
                else:
                    tokens.append(token)
        else:
            tokens = raw_tokens

        # Step 3: case fold
        if self._case_fold:
            tokens = [t.lower() for t in tokens]

        # Step 4: filter stopwords
        result: list[str] = []
        for t in tokens:
            stripped = t.strip()
            if stripped and stripped not in self._stopwords:
                result.append(stripped)
        return result


# ---------------------------------------------------------------------------
# CJKProcessor
# ---------------------------------------------------------------------------

# CJK Unified Ideographs and their extensions.
_CJK_RANGES: list[tuple[int, int]] = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
]


class CJKProcessor(LangProcessor):
    """CJK (Chinese / Japanese / Korean) character processor.

    Defaults to character bigram tokenization.  An optional
    ``word_tokenizer`` (e.g. ``jieba.lcut``) upgrades to word-level
    segmentation.
    """

    def __init__(self, word_tokenizer: WordTokenizer | None = None):
        self._word_tokenizer = word_tokenizer

    @property
    def priority(self) -> int:
        return 20

    def can_handle(self, char: str) -> bool:
        cp = ord(char)
        return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)

    def tokenize_segment(self, segment: str) -> list[str]:
        if self._word_tokenizer:
            return self._word_tokenizer(segment)
        # Fallback: character bigram
        if len(segment) <= 1:
            return [segment] if segment else []
        return [segment[i : i + 2] for i in range(len(segment) - 1)]


# ---------------------------------------------------------------------------
# _try_load_jieba
# ---------------------------------------------------------------------------


def _try_load_jieba() -> WordTokenizer | None:
    try:
        import jieba

        return jieba.lcut
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class Tokenizer:
    """Multi-language tokenizer with extensible processor architecture.

    Scans input text, segments it by script, dispatches each segment to
    the matching ``LangProcessor``, then normalizes and deduplicates.

    ``LangProcessor`` instances are ordered by ``priority`` (ascending).
    Characters not matched by any processor are handled by a built-in
    ``FallbackProcessor`` so no data is silently lost.
    """

    def __init__(
        self,
        processors: Iterable[LangProcessor] | None = None,
        *,
        min_token_length: int = 1,
        max_token_length: int = 64,
        deduplicate: bool = True,
    ):
        self._processors = sorted(
            processors if processors is not None else _default_processors(),
            key=lambda p: p.priority,
        )
        self._fallback = FallbackProcessor()
        self._min_len = min_token_length
        self._max_len = max_token_length
        self._deduplicate = deduplicate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Tokenize *text* into a normalized, deduplicated token list."""
        if not text:
            return []

        segments = self._segment_text(text)
        tokens: list[str] = []
        for processor, segment in segments:
            tokens.extend(processor.tokenize_segment(segment))

        tokens = self._normalize(tokens)

        if self._deduplicate:
            tokens = self._deduplicate_keep_order(tokens)

        return tokens

    def tokenize_batch(self, texts: Iterable[str]) -> list[list[str]]:
        """Tokenize multiple texts."""
        return [self.tokenize(t) for t in texts]

    def register_processor(self, processor: LangProcessor) -> None:
        """Register an additional language processor."""
        self._processors.append(processor)
        self._processors.sort(key=lambda p: p.priority)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _segment_text(self, text: str) -> list[tuple[LangProcessor, str]]:
        """Scan *text* and group consecutive same-processor characters.

        Characters matched by no registered processor are routed to the
        built-in ``FallbackProcessor``.
        """
        result: list[tuple[LangProcessor, str]] = []
        i = 0
        n = len(text)
        while i < n:
            proc = self._find_processor(text[i]) or self._fallback
            j = i + 1
            while j < n:
                next_proc = self._find_processor(text[j]) or self._fallback
                if next_proc is not proc:
                    break
                j += 1
            result.append((proc, text[i:j]))
            i = j
        return result

    def _find_processor(self, char: str) -> LangProcessor | None:
        for proc in self._processors:
            if proc.can_handle(char):
                return proc
        return None

    def _normalize(self, tokens: list[str]) -> list[str]:
        result: list[str] = []
        for t in tokens:
            stripped = t.strip()
            if self._min_len <= len(stripped) <= self._max_len and stripped:
                result.append(stripped)
        return result

    @staticmethod
    def _deduplicate_keep_order(tokens: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result


# ---------------------------------------------------------------------------
# Default processor list
# ---------------------------------------------------------------------------


def _default_processors() -> list[LangProcessor]:
    return [
        LatinProcessor(),
        CJKProcessor(),
    ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_tokenizer(
    *,
    case_fold: bool = True,
    stopwords: frozenset[str] | None = None,
    cjk_word_tokenizer: WordTokenizer | Literal["auto"] | None = "auto",
) -> Tokenizer:
    """Create a ``Tokenizer`` with sensible defaults.

    Default processors: ``LatinProcessor`` + ``CJKProcessor``.

    Args:
        case_fold: Whether ``LatinProcessor`` lowercases tokens.
        stopwords: Custom stopword set for ``LatinProcessor``.
        cjk_word_tokenizer:
            - ``"auto"`` — try ``jieba.lcut``, fall back to bigram.
            - ``None`` — always use bigram.
            - A callable — use the given word tokenizer directly.
    """
    word_tok: WordTokenizer | None = None
    if cjk_word_tokenizer == "auto":
        word_tok = _try_load_jieba()
    elif cjk_word_tokenizer is not None:
        word_tok = cjk_word_tokenizer

    processors: list[LangProcessor] = [
        LatinProcessor(case_fold=case_fold, stopwords=stopwords),
        CJKProcessor(word_tokenizer=word_tok),
    ]
    return Tokenizer(processors)
