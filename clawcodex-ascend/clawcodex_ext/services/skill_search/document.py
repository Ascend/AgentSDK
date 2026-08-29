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

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from clawcodex_ext.skills.model import Skill

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source type
# ---------------------------------------------------------------------------

SourceType = Literal["local", "project", "mcp", "template"]

# Mapping from Skill.loaded_from to SourceType.
# Skills from MCP builders and template generators are identified
# externally via extract_batch() parameters.
_LOADED_FROM_TO_SOURCE: dict[str, SourceType] = {
    "user": "local",
    "project": "project",
    "managed": "local",
    "plugin": "local",
    "mcp": "mcp",
    "template": "template",
}

# ---------------------------------------------------------------------------
# SkillSearchDocument
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillSearchDocument:
    """Immutable search document for a single skill.

    Each document is indexed by the TF-IDF inverted index.  The
    ``text()`` method concatenates all searchable fields; the
    ``field_text()`` method returns them separately so the ranker can
    apply per-field boosts.
    """

    id: str
    name: str
    title: str
    description: str
    body: str
    source: SourceType
    tags: tuple[str, ...] = ()
    updated_at: str | None = None
    weight: float = 1.0

    def text(self) -> str:
        """Concatenate all searchable fields for full-text tokenization."""
        parts: list[str] = []
        if self.name:
            parts.append(self.name)
        if self.title and self.title != self.name:
            parts.append(self.title)
        if self.description:
            parts.append(self.description)
        if self.body:
            parts.append(self.body)
        if self.tags:
            parts.append(" ".join(self.tags))
        return "\n".join(parts)

    def field_text(self) -> dict[str, str]:
        """Return per-field text for field-level weighted matching.

        Used by the ranker to apply ``field_boost``:
        name/title=3.0, tags=2.5, description=2.0, body=1.0.
        """
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "body": self.body,
            "tags": " ".join(self.tags),
        }

    @staticmethod
    def make_id(source: SourceType, name: str) -> str:
        """Generate a stable document ID.

        Uses SHA-256 (first 16 hex chars) so the same source + name
        always produces the same ID across processes.
        """
        raw = f"{source}:{name}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_search_document(
    skill: Skill,
    *,
    source_type: SourceType | None = None,
    source_weights: dict[str, float] | None = None,
) -> SkillSearchDocument | None:
    """Extract a ``SkillSearchDocument`` from a ``Skill``.

    Args:
        skill: Upstream ``Skill`` object.
        source_type: Explicit source type.  When ``None``, inferred from
            ``skill.loaded_from``.
        source_weights: Per-source weight map.  Defaults to the built-in
            weights: project=1.3, local=1.1, template=1.0, mcp=0.9.

    Returns:
        A ``SkillSearchDocument``, or ``None`` if the skill should be
        excluded from search (e.g. hidden skills).
    """
    if skill.is_hidden:
        return None

    if source_weights is None:
        source_weights = _default_source_weights()

    src = source_type or _infer_source_type(skill)
    weight = source_weights.get(src, 1.0)

    doc_id = SkillSearchDocument.make_id(src, skill.name)

    return SkillSearchDocument(
        id=doc_id,
        name=skill.name,
        title=skill.display_name or skill.name,
        description=_extract_description(skill),
        body=_extract_body(skill),
        source=src,
        tags=_extract_tags(skill),
        updated_at=None,
        weight=weight,
    )


def extract_batch(
    skills: Sequence[Skill],
    *,
    mcp_skill_names: set[str] | None = None,
    template_skill_names: set[str] | None = None,
    source_weights: dict[str, float] | None = None,
) -> list[SkillSearchDocument]:
    """Batch-extract documents from a list of skills.

    Args:
        skills: Upstream ``Skill`` objects.
        mcp_skill_names: Names of skills produced by MCP builders.
            These are tagged as ``source="mcp"``.
        template_skill_names: Names of skills produced by template
            generators.  These are tagged as ``source="template"``.
        source_weights: Per-source weight map.

    Returns:
        List of ``SkillSearchDocument`` (hidden skills are skipped).
    """
    mcp_names = mcp_skill_names or set()
    template_names = template_skill_names or set()

    docs: list[SkillSearchDocument] = []
    for skill in skills:
        if skill.name in mcp_names:
            src: SourceType = "mcp"
        elif skill.name in template_names:
            src = "template"
        else:
            src = _infer_source_type(skill)

        doc = extract_search_document(
            skill,
            source_type=src,
            source_weights=source_weights,
        )
        if doc is not None:
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_source_type(skill: Skill) -> SourceType:
    """Infer the source type from ``skill.loaded_from``.

    Mapping::

        "user"      → "local"
        "project"   → "project"
        "managed"   → "local"    (admin-distributed skills are local)
        "plugin"    → "local"    (plugin skills are local)
        "mcp"       → "mcp"
        "template"  → "template"
        other       → "local"    (safe fallback)
    """
    loaded_from = getattr(skill, "loaded_from", "")
    return _LOADED_FROM_TO_SOURCE.get(loaded_from, "local")


def _extract_description(skill: Skill) -> str:
    """Extract the searchable description.

    Priority: ``when_to_use`` > ``description`` > empty string.
    ``when_to_use`` is the most semantically rich field — it describes
    the exact scenarios where the skill is applicable.
    """
    when = skill.when_to_use
    if when:
        return when
    return skill.description or ""


def _extract_body(skill: Skill) -> str:
    """Extract the searchable body text.

    Priority: ``markdown_content`` (SKILL.md body without frontmatter)
    > ``content`` (fallback).
    """
    md = skill.markdown_content
    if md:
        return md
    return skill.content or ""


def _extract_tags(skill: Skill) -> tuple[str, ...]:
    """Derive search tags from existing Skill fields.

    The current ``Skill`` model has no explicit ``tags`` field, so we
    derive tags from:

    1. ``allowed_tools`` — e.g. ``["bash", "python", "read"]``
    2. Namespace prefix — ``"browser:playwright"`` → ``["browser", "playwright"]``
    3. ``source`` — e.g. ``"userSettings"``

    When a future ``tags`` frontmatter field is added to SKILL.md, this
    function can be extended to read it directly.
    """
    tags: list[str] = []

    allowed = skill.allowed_tools
    if allowed:
        tags.extend(t.lower() for t in allowed)

    if ":" in skill.name:
        tags.extend(skill.name.lower().split(":"))

    source = skill.source
    if source:
        tags.append(source.lower())

    seen: set[str] = set()
    unique: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return tuple(unique)


def _default_source_weights() -> dict[str, float]:
    return {
        "project": 1.3,
        "local": 1.1,
        "template": 1.0,
        "mcp": 0.9,
    }
