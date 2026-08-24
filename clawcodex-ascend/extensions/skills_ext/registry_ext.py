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
"""Compatibility registry facade for the canonical skill catalog.

Historically extensions.skills_ext performed a second discovery pass and kept
a process-global list cache. The canonical implementation now lives in
clawcodex_ext.skills; this module preserves the public extension API while
delegating discovery, alias resolution, and invalidation to that implementation.
"""

from __future__ import annotations


import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clawcodex_ext.skills.catalog import (
    get_skill_catalog,
    invalidate_skill_catalog,
)

from .hooks import SkillRegistrationCallback

if TYPE_CHECKING:
    from clawcodex_ext.skills.model import Skill

    from .agent_config import AgentSkillConfig

logger = logging.getLogger(__name__)


class SkillRegistryExt:
    """Backward-compatible view over workspace-scoped catalog snapshots."""

    def __init__(
        self,
        loader_module: Any | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        """Create a facade without owning discovery or registry state.

        loader_module remains accepted for callers that inspect
        upstream_loader; it is no longer used to run an independent discovery
        pass.
        """

        self._loader = loader_module or importlib.import_module("clawcodex_ext.skills.loader")
        self._callbacks: list[SkillRegistrationCallback] = []
        self._project_root = project_root
        self._last_project_root = project_root
        self._last_user_skills_dir: str | Path | None = None
        self._last_notified_version: int | None = None

    @property
    def upstream_loader(self) -> Any:
        """Return the legacy loader module exposed by the old wrapper."""

        return self._loader

    def get_all_skills(
        self,
        *,
        project_root: str | Path | None = None,
        user_skills_dir: str | Path | None = None,
        force_refresh: bool = False,
    ) -> list[Skill]:
        """Return skills from the canonical workspace catalog.

        force_refresh invalidates the canonical catalog and all dependent skill
        views before rebuilding it. No extension-local discovery result is
        retained.
        """

        resolved_root = project_root if project_root is not None else self._last_project_root
        resolved_user_dir = user_skills_dir if user_skills_dir is not None else self._last_user_skills_dir
        if force_refresh:
            invalidate_skill_catalog(
                "skills_ext force refresh",
                workspace=resolved_root if resolved_root is not None else Path.cwd(),
            )

        snapshot = get_skill_catalog(
            project_root=resolved_root,
            user_skills_dir=resolved_user_dir,
        )
        self._last_project_root = resolved_root
        self._last_user_skills_dir = resolved_user_dir
        if snapshot.diagnostics:
            logger.debug(
                "[skills_ext] catalog diagnostics for %s: %s",
                snapshot.project_root,
                "; ".join(snapshot.diagnostics),
            )

        if snapshot.version != self._last_notified_version:
            for skill in snapshot.skills:
                self._notify_skill_registered(skill)
            self._last_notified_version = snapshot.version

        return list(snapshot.skills)

    def get_skill(self, name: str) -> Skill | None:
        """Resolve a live skill by canonical name or alias."""

        return get_skill_catalog(
            project_root=self._last_project_root,
            user_skills_dir=self._last_user_skills_dir,
        ).resolve(name)

    def list_skills(self) -> list[Skill]:
        """List all definitions available in the current catalog."""

        return self.get_all_skills()

    def on_skill_registered(self, callback: SkillRegistrationCallback) -> None:
        """Register a callback notified once per snapshot version."""

        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def off_skill_registered(self, callback: SkillRegistrationCallback) -> None:
        """Remove a previously registered callback."""

        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_skill_registered(self, skill: Skill) -> None:
        """Notify callbacks without letting compatibility hooks break discovery."""

        for callback in self._callbacks:
            try:
                callback(skill)
            except Exception:
                logger.debug(
                    "[skills_ext] skill registration callback failed",
                    exc_info=True,
                )

    def get_skills_for_config(
        self,
        config: AgentSkillConfig,
    ) -> list[Skill]:
        """Filter the canonical catalog using the legacy bundle configuration."""

        from .bundles import MODE_BUNDLES, SKILL_BUNDLES

        if config.mode == "bare":
            return []

        all_skills = self.get_all_skills()
        skill_names_in_bundle: set[str] = set()

        if config.mode == "all":
            for bundle_skills in SKILL_BUNDLES.values():
                skill_names_in_bundle.update(bundle_skills)
        elif config.bundles is not None:
            for bundle in config.bundles:
                skill_names_in_bundle.update(SKILL_BUNDLES.get(bundle, ()))
        else:
            for bundle in MODE_BUNDLES.get(config.mode, ["default"]):
                skill_names_in_bundle.update(SKILL_BUNDLES.get(bundle, ()))

        return [
            skill
            for skill in all_skills
            if skill.name not in config.exclude and (config.mode == "all" or skill.name in skill_names_in_bundle)
        ]

    def load_bundle(self, bundle_name: str) -> list[str]:
        """Return names from a legacy bundle that exist in the catalog."""

        from .bundles import SKILL_BUNDLES

        if bundle_name not in SKILL_BUNDLES:
            raise KeyError(f"unknown bundle: {bundle_name}")

        available = {skill.name for skill in self.get_all_skills()}
        return [name for name in SKILL_BUNDLES[bundle_name] if name in available]

    def get_available_bundle_names(self) -> list[str]:
        """Return all known legacy bundle names."""

        from .bundles import ALL_BUNDLE_NAMES

        return list(ALL_BUNDLE_NAMES)

    def clear_cache(self) -> None:
        """Invalidate canonical skill caches (legacy method name)."""

        self._last_notified_version = None
        invalidate_skill_catalog("skills_ext compatibility clear")


_default_registry: SkillRegistryExt | None = None


def get_default_registry() -> SkillRegistryExt:
    """Return the process-wide compatibility facade."""

    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistryExt()
    return _default_registry


def clear_default_registry_cache() -> None:
    """Invalidate canonical caches and reset the legacy facade singleton."""

    global _default_registry
    if _default_registry is not None:
        _default_registry.clear_cache()
    else:
        invalidate_skill_catalog("skills_ext default registry clear")
    _default_registry = None
