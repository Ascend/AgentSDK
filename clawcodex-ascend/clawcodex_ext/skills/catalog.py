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

"""Immutable, workspace-scoped views over every available skill source.

The loader remains responsible for discovering skills from disk, bundled
registrations, MCP builders, and dynamic directories.  This module turns that
mutable discovery result into a stable snapshot which can be shared by prompt,
tool, and command surfaces without relying on the process-global legacy
registry.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .model import Skill

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SkillCatalogSnapshot:
    """An immutable index of the skills discovered for one workspace.

    The contained :class:`Skill` definitions are shared with their source and
    are treated as immutable configuration objects.  The tuple and lookup maps
    themselves cannot be modified.  Enabled predicates are deliberately not
    cached: :meth:`resolve` evaluates them on every invocation so a runtime
    feature gate can change without rebuilding the catalog.
    """

    project_root: str
    user_skills_dir: str | None
    version: int
    skills: tuple[Skill, ...]
    canonical: Mapping[str, Skill]
    aliases: Mapping[str, Skill]
    session_id: str | None = None
    diagnostics: tuple[str, ...] = ()

    def resolve(self, name: str, *, include_disabled: bool = False) -> Skill | None:
        """Resolve ``name``, preferring every canonical name over every alias.

        By default a disabled skill is indistinguishable from an unavailable
        skill.  Callers that need to report a dedicated disabled diagnostic may
        request the definition with ``include_disabled=True`` and evaluate its
        predicate themselves.
        """

        skill = self.canonical.get(name)
        if skill is None:
            skill = self.aliases.get(name)
        if skill is None or include_disabled:
            return skill

        try:
            return skill if skill.is_enabled() else None
        except Exception as exc:  # pragma: no cover - defensive source boundary
            logger.warning("skill %r enabled predicate failed: %s", skill.name, exc)
            return None


_CatalogKey = tuple[str, str | None, str | None]
_catalog_cache: dict[_CatalogKey, SkillCatalogSnapshot] = {}
_catalog_lock = threading.RLock()
_catalog_versions = itertools.count(1)


def _normalize_catalog_key(
    project_root: str | Path | None,
    user_skills_dir: str | Path | None,
    session_id: str | None,
) -> _CatalogKey:
    raw_root = project_root if project_root is not None else os.getcwd()
    try:
        root = Path(raw_root)
    except TypeError:
        # Partially constructed interactive surfaces may expose a placeholder
        # workspace_root. Bundled commands should still remain discoverable.
        root = Path.cwd()
    normalized_root = str(root.expanduser().resolve())
    try:
        normalized_user = str(Path(user_skills_dir).expanduser().resolve()) if user_skills_dir is not None else None
    except TypeError:
        normalized_user = None
    return normalized_root, normalized_user, str(session_id) if session_id else None


def _build_snapshot(key: _CatalogKey) -> SkillCatalogSnapshot:
    from .bundled import init_bundled_skills

    # Import lazily so loader can re-export the public catalog API without a
    # module-import cycle.
    diagnostics: list[str] = []
    if not init_bundled_skills():
        diagnostics.append("one or more bundled skills failed to initialize; retry is enabled")
    try:
        from extensions.skills_ext import init_skill_catalog_extensions

        extensions_ok = init_skill_catalog_extensions()
    except Exception as exc:
        extensions_ok = False
        logger.warning("failed to initialize extension skill adapters: %s", exc)
    if not extensions_ok:
        diagnostics.append("one or more extension skill adapters failed to initialize; retry is enabled")
    from . import loader
    from .bundled_skills import get_registered_bundled_skills

    project_root, user_skills_dir, session_id = key
    loaded = list(
        loader.discover_all_skills(
            project_root=project_root,
            user_skills_dir=user_skills_dir,
            session_id=session_id,
            diagnostics=diagnostics,
        )
    )
    bundled = list(get_registered_bundled_skills())

    # Bundled canonical names are reserved. Keep the overlay here as a defensive
    # boundary as well as in loader discovery, so custom loader adapters cannot
    # accidentally shadow a first-party definition.
    ordered: list[Skill] = []
    canonical: dict[str, Skill] = {}
    for skill in bundled + loaded:
        if skill.name in canonical:
            if canonical[skill.name] is skill:
                continue
            diagnostics.append(
                f"duplicate canonical skill {skill.name!r} from {getattr(skill, 'loaded_from', 'unknown')!r} ignored"
            )
            continue
        canonical[skill.name] = skill
        ordered.append(skill)

    # Build aliases only after the complete canonical namespace is known.  An
    # alias can therefore never shadow a canonical name, even when its owning
    # skill appeared earlier in source priority order.
    aliases: dict[str, Skill] = {}
    for skill in ordered:
        for alias in skill.aliases:
            if not alias:
                continue
            if alias in canonical:
                diagnostics.append(f"alias {alias!r} from {skill.name!r} conflicts with a canonical name")
                continue
            if alias in aliases:
                diagnostics.append(f"duplicate alias {alias!r} from {skill.name!r} ignored")
                continue
            aliases[alias] = skill

    return SkillCatalogSnapshot(
        project_root=project_root,
        user_skills_dir=user_skills_dir,
        version=next(_catalog_versions),
        skills=tuple(ordered),
        canonical=MappingProxyType(canonical),
        aliases=MappingProxyType(aliases),
        session_id=session_id,
        diagnostics=tuple(diagnostics),
    )


def get_skill_catalog(
    context: Any | None = None,
    *,
    project_root: str | Path | None = None,
    user_skills_dir: str | Path | None = None,
    session_id: str | None = None,
) -> SkillCatalogSnapshot:
    """Return the immutable catalog for a ToolContext or explicit workspace."""
    if context is not None:
        if project_root is None:
            project_root = getattr(context, "workspace_root", context)
        if user_skills_dir is None:
            user_skills_dir = getattr(context, "user_skills_dir", None)
        if session_id is None:
            raw_session_id = getattr(context, "session_id", None)
            session_id = str(raw_session_id) if raw_session_id is not None else None

    key = _normalize_catalog_key(project_root, user_skills_dir, session_id)
    with _catalog_lock:
        snapshot = _catalog_cache.get(key)
        needs_retry = snapshot is not None and any(
            diagnostic.endswith("retry is enabled") for diagnostic in snapshot.diagnostics
        )
        if snapshot is None or needs_retry:
            snapshot = _build_snapshot(key)
            _catalog_cache[key] = snapshot
        return snapshot


def _clear_dependent_skill_views() -> None:
    """Clear command aggregation and rendered skill-list prompt caches."""
    try:
        from clawcodex_ext.command_system.aggregator import clear_commands_cache

        clear_commands_cache()
    except Exception:
        logger.debug("unable to clear command skill caches", exc_info=True)

    try:
        from clawcodex_ext.context_system.prompt_assembly import get_system_prompt_cache

        prompt_cache = get_system_prompt_cache()
        for section_id in prompt_cache.get_cached_section_ids():
            if section_id.startswith("skills:"):
                prompt_cache.invalidate(section_id)
    except Exception:
        logger.debug("unable to clear skill prompt caches", exc_info=True)


def _workspace_path(workspace: Any) -> str:
    value = getattr(workspace, "workspace_root", workspace)
    return str(Path(value).expanduser().resolve())


def _invalidate_catalog_cache_only(workspace: Any | None = None) -> None:
    """Clear snapshots without recursively invalidating loader discovery."""
    with _catalog_lock:
        if workspace is None:
            _catalog_cache.clear()
        else:
            root = _workspace_path(workspace)
            for key in tuple(_catalog_cache):
                if key[0] == root:
                    _catalog_cache.pop(key, None)
    _clear_dependent_skill_views()


def invalidate_skill_catalog(
    reason: str = "unspecified",
    workspace: Any | None = None,
) -> None:
    """Invalidate catalog, discovery, command and prompt-list caches."""
    logger.debug("invalidating skill catalog: %s", reason)
    _invalidate_catalog_cache_only(workspace)

    # The loader owns disk, conditional and dynamic discovery caches. Its
    # invalidator calls the private helper again; that is idempotent and avoids
    # exposing a partially invalidated workspace to another catalog consumer.
    from .loader import clear_skill_caches

    clear_skill_caches(workspace)


def resolve(
    name: str,
    *,
    project_root: str | Path | None = None,
    user_skills_dir: str | Path | None = None,
    session_id: str | None = None,
    include_disabled: bool = False,
) -> Skill | None:
    """Resolve a canonical skill name or alias from the workspace catalog."""

    return get_skill_catalog(
        project_root=project_root,
        user_skills_dir=user_skills_dir,
        session_id=session_id,
    ).resolve(name, include_disabled=include_disabled)


# Descriptive alias for callers that prefer an explicit noun in imports.
resolve_skill = resolve


__all__ = [
    "SkillCatalogSnapshot",
    "get_skill_catalog",
    "invalidate_skill_catalog",
    "resolve",
    "resolve_skill",
]
