# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#

"""Docstring-derived intent phrases for pos-converter ToolSearch metadata.

Complements :mod:`search_tags` (which already tokenizes identifiers, parameters,
and single-word description keywords).  This module adds **multi-word phrases**
from docstrings only — no naming-pattern assumptions (no ``run_*_cli``, etc.).
"""

from __future__ import annotations

import re

from extensions.sop_converter.core.source_parser import SourceOperation

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "and",
        "or",
        "not",
        "return",
        "returns",
        "when",
        "that",
        "this",
        "from",
        "by",
        "at",
        "as",
        "it",
        "if",
        "none",
        "true",
        "false",
        "args",
        "optional",
        "default",
        "against",
        "using",
        "given",
    }
)

_CJK_SEGMENT_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}")


def _dedupe_phrases(phrases: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        cleaned = phrase.strip()
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return tuple(out)


def _phrases_from_description(description: str, *, max_phrases: int = 8) -> list[str]:
    """Extract multi-word search phrases from a docstring."""
    desc = (description or "").strip()
    if not desc:
        return []

    phrases: list[str] = []

    first = re.split(r"[.。!?\n]", desc, maxsplit=1)[0].strip()
    if 4 <= len(first) <= 120:
        phrases.append(first)
        if first != first.lower():
            phrases.append(first.lower())

    for segment in _CJK_SEGMENT_RE.findall(desc):
        if 2 <= len(segment) <= 40:
            phrases.append(segment)

    words = [w.lower() for w in _WORD_RE.findall(desc) if w.lower() not in _STOP_WORDS]
    for n in (4, 3, 2):
        for start in range(len(words) - n + 1):
            chunk = words[start : start + n]
            if all(w in _STOP_WORDS for w in chunk):
                continue
            phrase = " ".join(chunk)
            if len(phrase) >= 4:
                phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases

    return phrases


def collect_intent_phrases(
    op: SourceOperation,
    *,
    comp_name: str = "",
) -> tuple[str, ...]:
    """Build docstring-derived intent phrases for ToolSearch and task guides."""
    return _dedupe_phrases(_phrases_from_description(op.description))


def get_intent_tags(
    op_name: str,
    *,
    description: str = "",
) -> tuple[str, ...]:
    """Backward-compatible wrapper around :func:`collect_intent_phrases`."""
    op = SourceOperation(name=op_name, description=description)
    return collect_intent_phrases(op)


def enrich_with_intent_tags(
    op: SourceOperation,
    *,
    comp_name: str = "",
    base_tags: tuple[str, ...],
) -> tuple[str, ...]:
    """Append docstring phrases to ``base_tags`` without duplicates."""
    extras = collect_intent_phrases(op, comp_name=comp_name)
    if not extras:
        return base_tags

    seen = {t.lower() for t in base_tags}
    merged = list(base_tags)
    for tag in extras:
        lowered = tag.lower()
        if lowered not in seen:
            seen.add(lowered)
            merged.append(tag)
    return tuple(merged)


def format_search_suggestions(
    op: SourceOperation,
    *,
    comp_name: str = "",
    limit: int = 5,
) -> str:
    """Comma-separated ToolSearch query suggestions for task guides."""
    phrases = list(collect_intent_phrases(op, comp_name=comp_name))
    name_phrase = op.name.replace("_", " ")
    candidates: list[str] = []
    for phrase in (*phrases, name_phrase):
        if phrase and phrase not in candidates:
            candidates.append(phrase)
        if len(candidates) >= limit:
            break
    return ", ".join(candidates)
