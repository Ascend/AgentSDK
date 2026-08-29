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

# pylint: disable=cyclic-import

"""Unified section registry for system prompt assembly (P119-A/B/H).

Consolidates the three-way split (builder list + inserted list + canonical
maps) into a single ``_registry: dict[str, RegisteredSection]`` where every
section — whether it overrides a known slot or injects a new one — lives in
one place with its builder, order, cache_scope, and tags.

Sub-features:
  P119-A — ``register_section(id, *, builder, order, cache_scope, tags)``
  P119-B — ``override_section`` / ``disable_section`` / ``insert_section``
           thin wrappers around ``register_section``
  P119-H — tags metadata, ``get_sections_by_tag``,
           ``collect_new_sections(runtime_ctx)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from clawcodex_ext.context_system.system_prompt_cache import (
    CacheScope,
    DANGEROUS_uncachedSystemPromptSection,
    SystemPromptSection,
)


# ---------------------------------------------------------------------------
# SectionScope — cache-scope enum independent of system_prompt_cache
# ---------------------------------------------------------------------------


class SectionScope(str, Enum):
    """Cache scope for a section registration.

    Mirrors ``CacheScope`` from ``system_prompt_cache`` but is defined
    here to avoid a circular dependency at import time.
    """

    GLOBAL = "global"
    SESSION = "session"
    REQUEST = "request"


# ---------------------------------------------------------------------------
# RegisteredSection — one structure holds all metadata
# ---------------------------------------------------------------------------


@dataclass
class RegisteredSection:
    """A registered section with builder, position, and tags.

    The builder receives the per-call ``runtime_ctx`` dict and returns
    content (or ``None`` to suppress the section).
    """

    id: str
    builder: Callable[[dict[str, Any]], str | None]
    order: int
    cache_scope: SectionScope
    tags: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Canonical defaults (used when builder does not supply order / scope)
# ---------------------------------------------------------------------------

_CANONICAL_ORDER: dict[str, int] = {
    # Static modules (0-6)
    "intro": 0,
    "system": 1,
    "doing_tasks": 2,
    "actions": 3,
    "using_tools": 4,
    "tone_style": 5,
    "output_efficiency": 6,
    # Dynamic modules (10-95)
    "tool_docs": 10,
    "environment": 20,
    "memory": 25,
    "mcp": 30,
    "agents": 40,
    "skills": 50,
    "output_style": 60,
    "proactive": 65,
    "plan_mode": 70,
    "non_interactive": 80,
    "tool_restrictions": 90,
    "iteration_meta": 95,
}

_CANONICAL_SCOPE: dict[str, SectionScope] = {
    "intro": SectionScope.GLOBAL,
    "system": SectionScope.GLOBAL,
    "doing_tasks": SectionScope.GLOBAL,
    "actions": SectionScope.GLOBAL,
    "using_tools": SectionScope.GLOBAL,
    "tone_style": SectionScope.GLOBAL,
    "output_efficiency": SectionScope.GLOBAL,
    "tool_docs": SectionScope.SESSION,
    "environment": SectionScope.REQUEST,
    "memory": SectionScope.REQUEST,
    "mcp": SectionScope.SESSION,
    "agents": SectionScope.SESSION,
    "skills": SectionScope.SESSION,
    "output_style": SectionScope.SESSION,
    "proactive": SectionScope.REQUEST,
    "plan_mode": SectionScope.REQUEST,
    "non_interactive": SectionScope.REQUEST,
    "tool_restrictions": SectionScope.REQUEST,
    "iteration_meta": SectionScope.REQUEST,
}

# IDs of built-in sections (used by ``_build_*_section`` in prompt_assembly.py).
# ``collect_new_sections`` skips these — they are handled by the dedicated
# builder functions.
_CANONICAL_IDS: frozenset[str] = frozenset(_CANONICAL_ORDER.keys())


def get_section_order(section_id: str) -> int:
    """Return the canonical order for a known section, or 0 for unknown."""
    return _CANONICAL_ORDER.get(section_id, 0)


def get_section_scope(section_id: str) -> SectionScope:
    """Return the canonical scope for a known section, or SESSION for unknown."""
    return _CANONICAL_SCOPE.get(section_id, SectionScope.SESSION)


# ---------------------------------------------------------------------------
# The registry: section_id → RegisteredSection
# ---------------------------------------------------------------------------

_registry: dict[str, RegisteredSection] = {}


# Keep the public ``id=`` keyword used by section-registration callers.
def register_section(  # pylint: disable=redefined-builtin
    id: str,
    *,
    builder: Callable[[dict[str, Any]], str | None],
    order: int | None = None,
    cache_scope: SectionScope | None = None,
    tags: list[str] | None = None,
) -> RegisteredSection:
    """Register a section provider.

    If *id* matches a known built-in section (e.g. ``"intro"``), the
    provider overrides its content.  Otherwise a new section is injected.

    The *builder* is called with the per-call ``runtime_ctx`` dict every
    time the prompt is assembled.  Return ``None`` to suppress the section.

    Args:
        id: Unique identifier.  Known IDs override built-in sections;
            unknown IDs inject a new section.
        builder: ``(runtime_ctx: dict) -> str | None``.
        order: Sort order.  ``None`` → infer from canonical map (or 50).
        cache_scope: Cache scope.  ``None`` → infer from canonical map
                     (or ``SESSION``).
        tags: Optional list of tags for filtering/grouping.

    Returns:
        The created ``RegisteredSection``.
    """
    sec = RegisteredSection(
        id=id,
        builder=builder,
        order=order if order is not None else _CANONICAL_ORDER.get(id, 50),
        cache_scope=cache_scope if cache_scope is not None else _CANONICAL_SCOPE.get(id, SectionScope.SESSION),
        tags=set(tags or []),
    )
    _registry[id] = sec
    return sec


# Keep the public ``id=`` keyword symmetric with ``register_section``.
def unregister_section(id: str) -> None:  # pylint: disable=redefined-builtin
    """Remove a previously registered section (for hot-unload or tests)."""
    _registry.pop(id, None)


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------


def consult_section_builders(
    section_id: str,
    runtime_ctx: dict[str, Any] | None = None,
) -> SystemPromptSection | None:
    """Consult the registry for *section_id*.

    If a ``RegisteredSection`` exists for *section_id*, its builder is
    called with the given *runtime_ctx*.  Returns a ``SystemPromptSection``
    with the canonical order and scope, or ``None`` if the builder returned
    ``None`` / raised / wasn't registered.

    This is the primary entry point from ``prompt_assembly.py``'s
    ``_build_*_section`` functions — they call it with the same ``id``
    they pass to every other step, keeping call sites simple.
    """
    sec = _registry.get(section_id)
    if sec is None:
        return None
    try:
        content = sec.builder(runtime_ctx or {})
    except Exception:
        return None
    # None  = "no override — caller should fall back to default"
    # str   = "use this content" (may be empty = suppress)
    if content is None:
        return None
    return SystemPromptSection(
        id=sec.id,
        content=content,
        cache_scope=_to_cache_scope(sec.cache_scope),
        order=sec.order,
    )


def collect_new_sections(
    runtime_ctx: dict[str, Any],
    *,
    tags: list[str] | None = None,
) -> list[SystemPromptSection]:
    """Build all registered sections whose *id* is NOT a built-in slot.

    This replaces the old ``get_inserted_sections()`` pattern.  New
    sections have their own ``id`` and ``order``, so they sort naturally
    alongside the built-in ones.

    Args:
        runtime_ctx: Per-call runtime context passed to every builder.
        tags: Optional OR-filter — only build sections with any of the
              given tags.

    Returns:
        A list of ``SystemPromptSection``, sorted by ``order``.
    """
    result: list[SystemPromptSection] = []
    for sec in _registry.values():
        if sec.id in _CANONICAL_IDS:
            continue
        if tags and not (sec.tags & set(tags)):
            continue
        try:
            content = sec.builder(runtime_ctx)
        except Exception:
            continue
        if content:
            result.append(
                SystemPromptSection(
                    id=sec.id,
                    content=content,
                    cache_scope=_to_cache_scope(sec.cache_scope),
                    order=sec.order,
                )
            )
    result.sort(key=lambda s: s.order)
    return result


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def get_sections_by_tag(*tags: str) -> list[RegisteredSection]:
    """Return all registered sections matching ANY of *tags* (OR logic).

    If no tags are given, returns **all** registered sections.
    """
    if not tags:
        return list(_registry.values())
    tag_set = set(tags)
    return [sec for sec in _registry.values() if sec.tags & tag_set]


def get_sections_by_all_tags(*tags: str) -> list[RegisteredSection]:
    """Return sections matching ALL of *tags* (AND logic)."""
    tag_set = set(tags)
    if not tag_set:
        return list(_registry.values())
    return [sec for sec in _registry.values() if tag_set.issubset(sec.tags)]


def get_section_tags(section_id: str) -> set[str]:
    """Return the tags for *section_id*, or an empty set."""
    sec = _registry.get(section_id)
    return sec.tags if sec else set()


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def clear_section_registry() -> None:
    """Clear all registered sections.

    Primarily used by tests to ensure clean state between test cases.
    """
    _registry.clear()


# ---------------------------------------------------------------------------
# High-level override / disable / insert API (P119-B) — thin wrappers
# ---------------------------------------------------------------------------


def _to_cache_scope(scope: SectionScope) -> CacheScope:
    return CacheScope(scope.value)


def override_section(
    section_id: str,
    content: str,
    *,
    cache_scope: CacheScope | None = None,
    order: int | None = None,
    reason: str = "downstream override",
) -> SystemPromptSection:
    """Replace the content of *section_id* with *content*.

    The section is built as *cache-breaking* (recomputed every turn) to
    prevent stale cached content from leaking through.

    The *reason* parameter is required by
    ``DANGEROUS_uncachedSystemPromptSection`` — it forces code review to
    acknowledge the cache break.

    Returns the created :class:`SystemPromptSection` for inspection.
    """
    canonical_scope = _CANONICAL_SCOPE.get(section_id, SectionScope.SESSION)
    canonical_order = order if order is not None else _CANONICAL_ORDER.get(section_id, 0)

    section = DANGEROUS_uncachedSystemPromptSection(
        name=section_id,
        content=content,
        reason=reason,
        cache_scope=cache_scope or _to_cache_scope(canonical_scope),
        order=canonical_order,
    )
    register_section(
        section_id,
        builder=lambda _ctx: content,
        order=canonical_order,
        cache_scope=canonical_scope,
        tags=["_override"],
    )
    # P119-F: invalidate cache so the override takes effect immediately.
    _invalidate_section_cache(section_id)
    return section


def disable_section(section_id: str) -> None:
    """Suppress *section_id* from the prompt.

    Registers a builder that returns an empty string.  ``consult_section_builders``
    treats ``""`` (non-``None``) as "use this content" so the empty section
    replaces the default.  The build loop then filters it out with
    ``if s.content`` — net effect: the section disappears.
    """
    register_section(
        section_id,
        builder=lambda _ctx: "",
        order=_CANONICAL_ORDER.get(section_id, 0),
        tags=["_disabled"],
    )
    _invalidate_section_cache(section_id)


def insert_section(
    after_id: str,
    new_id: str,
    content: str,
    *,
    cache_scope: CacheScope = CacheScope.SESSION,
    reason: str = "downstream insertion",
) -> SystemPromptSection:
    """Insert a new section *new_id* after *after_id*.

    The new section's order = *after_id*'s order + 0.5 (all canonical
    sections use integer orders).
    """
    base_order = _CANONICAL_ORDER.get(after_id, 50)
    new_order = base_order + 0.5

    section = DANGEROUS_uncachedSystemPromptSection(
        name=new_id,
        content=content,
        reason=reason,
        cache_scope=cache_scope,
        order=new_order,
    )
    register_section(
        new_id,
        builder=lambda _ctx: content,
        order=new_order,
        cache_scope=SectionScope(cache_scope.value),
    )
    return section


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _invalidate_section_cache(section_id: str) -> None:
    """Invalidate the prompt cache for *section_id*.

    Import is deferred to avoid a circular dependency:
    ``prompt_assembly → section_registry → prompt_assembly``.
    """
    from clawcodex_ext.context_system.prompt_assembly import get_system_prompt_cache

    get_system_prompt_cache().invalidate(section_id)
