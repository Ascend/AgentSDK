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

"""Search-tag generation for pos-converter tools (Plan B metadata).

Derives ``AgentToolSpec.tags`` from ``SourceOperation`` metadata so ToolSearch
can match natural-language queries (e.g. ``loop coordinator iteration``) rather
than requiring exact kebab-case tool names.
"""

from __future__ import annotations

import re

from extensions.sop_converter.core.intent_tags import enrich_with_intent_tags
from extensions.sop_converter.core.source_parser import SourceOperation

_CAMEL_LOWER_UPPER_RE = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_ACRONYM_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
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
    }
)


def _camel_split(name: str) -> list[str]:
    spaced = _CAMEL_LOWER_UPPER_RE.sub(r"\1 \2", name.strip())
    spaced = _CAMEL_ACRONYM_RE.sub(r"\1 \2", spaced)
    return [part for part in re.split(r"[\s_]+", spaced) if part]


def _add_tag(tags: list[str], seen: set[str], value: str) -> None:
    cleaned = value.strip().lower()
    if len(cleaned) < 2:
        return
    if cleaned in seen:
        return
    seen.add(cleaned)
    tags.append(cleaned)


def _add_identifier_variants(tags: list[str], seen: set[str], identifier: str) -> None:
    if not identifier:
        return
    _add_tag(tags, seen, identifier)
    if "_" in identifier:
        _add_tag(tags, seen, identifier.replace("_", " "))
        for part in identifier.split("_"):
            _add_tag(tags, seen, part)
    for part in _camel_split(identifier):
        _add_tag(tags, seen, part)
        _add_tag(tags, seen, part.lower())
    if len(_camel_split(identifier)) > 1:
        _add_tag(tags, seen, " ".join(p.lower() for p in _camel_split(identifier)))


def _add_comp_name_variants(tags: list[str], seen: set[str], comp_name: str) -> None:
    if not comp_name:
        return
    _add_identifier_variants(tags, seen, comp_name.replace(".", " "))
    for segment in comp_name.split("."):
        _add_identifier_variants(tags, seen, segment)
    if "." in comp_name:
        short = comp_name.split(".", 1)[1]
        for segment in re.split(r"[._]", short):
            _add_identifier_variants(tags, seen, segment)


def _description_keywords(description: str, *, limit: int = 8) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{1,}", description.lower())
    picked: list[str] = []
    for word in words:
        if word in _STOP_WORDS:
            continue
        if word not in picked:
            picked.append(word)
        if len(picked) >= limit:
            break
    return picked


def generate_search_tags(op: SourceOperation, *, comp_name: str = "") -> tuple[str, ...]:
    """Build deduplicated search tags for a parsed SDK operation."""
    tags: list[str] = []
    seen: set[str] = set()

    _add_identifier_variants(tags, seen, op.name)
    if op.class_name:
        _add_identifier_variants(tags, seen, op.class_name)
    if op.file_stem:
        _add_identifier_variants(tags, seen, op.file_stem)
    _add_comp_name_variants(tags, seen, comp_name)

    for word in _description_keywords(op.description):
        _add_tag(tags, seen, word)

    for param in op.parameters:
        if param.name.startswith("*"):
            continue
        _add_identifier_variants(tags, seen, param.name)

    return enrich_with_intent_tags(op, comp_name=comp_name, base_tags=tuple(tags))
