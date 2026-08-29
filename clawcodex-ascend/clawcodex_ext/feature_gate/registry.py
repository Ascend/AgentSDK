#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

"""FeatureRegistry — core singleton for runtime feature toggles.

Implements the registry that stores ``FeatureFlag`` declarations and
resolves their effective state by checking overrides (CLI), environment
variables, persisted config, and finally the registered default.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from .config import ConfigStore
from .types import FeatureFlag

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Canonical env-prefix used for per-feature overrides.
_ENV_PREFIX = "CLAWCODEX_FEATURE_"


class FeatureRegistry:
    """Central store for runtime feature flags.

    Resolution order (highest → lowest priority):

    1. Programmatic overrides set via :meth:`set_override`.
    2. Environment variables ``CLAWCODEX_FEATURE_<NAME>``.
    3. Persisted config file (``features.json``).
    4. The ``default`` value on the registered :class:`FeatureFlag`.

    Unknown (unregistered) feature names always resolve to *disabled*.
    """

    def __init__(self, config_store: ConfigStore | None = None) -> None:
        self._features: dict[str, FeatureFlag] = {}
        self._overrides: dict[str, bool] = {}
        self._config_store = config_store or ConfigStore()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, flag: FeatureFlag) -> None:
        """Register a feature flag.

        Raises:
            ValueError: If a flag with the same name is already registered.
        """
        if flag.name in self._features:
            raise ValueError(f"Duplicate feature flag: {flag.name}")
        self._features[flag.name] = flag

    def register_many(self, flags: Sequence[FeatureFlag]) -> None:
        """Register multiple flags at once."""
        for flag in flags:
            self.register(flag)

    def unregister(self, name: str) -> None:
        """Remove a feature flag. No-op if unknown."""
        self._features.pop(name, None)

    # ------------------------------------------------------------------
    # Override helpers
    # ------------------------------------------------------------------

    def set_override(self, name: str, enabled: bool) -> None:
        """Programmatically override a feature's state (highest priority)."""
        if enabled:
            self._overrides[name] = True
        else:
            self._overrides[name] = False

    def clear_override(self, name: str) -> None:
        """Remove a programmatic override, falling through to lower tiers."""
        self._overrides.pop(name, None)

    def clear_all_overrides(self) -> None:
        """Remove all programmatic overrides."""
        self._overrides.clear()

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def is_enabled(self, name: str) -> bool:
        """Return the effective enabled state for *name*.

        Resolution order: override > env var > config file > default.
        Unregistered names return ``False``.
        """
        # 1. Programmatic override
        if name in self._overrides:
            return self._overrides[name]

        # 2. Environment variable
        env_val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes")

        # 3. Persisted config
        config_val = self._config_store.get(name)
        if config_val is not None:
            return bool(config_val)

        # 4. Registered default (or False for unknown)
        flag = self._features.get(name)
        return flag.default if flag else False

    def get_state(self, name: str) -> bool | None:
        """Return the effective state, or ``None`` if the feature is unregistered."""
        if name not in self._features:
            return None
        return self.is_enabled(name)

    def list_features(self) -> list[str]:
        """Return all registered feature names."""
        return list(self._features.keys())

    def get_flag(self, name: str) -> FeatureFlag | None:
        """Return the raw :class:`FeatureFlag` declaration, or ``None``."""
        return self._features.get(name)

    # ------------------------------------------------------------------
    # Dependency & mutual-exclusion checks
    # ------------------------------------------------------------------

    def check_deps(self, name: str) -> list[str]:
        """Return a list of dependency names that are NOT enabled.

        If all dependencies are satisfied, returns an empty list.
        Unregistered features are treated as disabled for this check.
        """
        flag = self._features.get(name)
        if not flag or not flag.deps:
            return []
        return [dep for dep in flag.deps if not self.is_enabled(dep)]

    def check_mutex(self, name: str) -> list[str]:
        """Return a list of conflicting feature names that ARE enabled.

        If there are no conflicts, returns an empty list.
        """
        flag = self._features.get(name)
        if not flag or not flag.mutex_with:
            return []
        return [m for m in flag.mutex_with if self.is_enabled(m)]

    def validate_registration(self, name: str) -> tuple[bool, list[str]]:
        """Validate that a feature can be safely enabled.

        Returns:
            ``(ok, errors)`` — ``ok`` is ``True`` when there are no
            unresolved dependencies or mutex conflicts.
        """
        errors: list[str] = []
        missing_deps = self.check_deps(name)
        if missing_deps:
            errors.append(f"Missing dependencies: {missing_deps}")
        mutex_conflicts = self.check_mutex(name)
        if mutex_conflicts:
            errors.append(f"Mutex conflicts: {mutex_conflicts}")
        return (len(errors) == 0, errors)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def enable_feature(self, name: str) -> None:
        """Enable a feature (sets a programmatic override)."""
        self.set_override(name, True)

    def disable_feature(self, name: str) -> None:
        """Disable a feature (sets a programmatic override)."""
        self.set_override(name, False)

    def save_config(self) -> None:
        """Persist the current effective states to the config file."""
        states: dict[str, bool] = {}
        for name in self._features:
            states[name] = self.is_enabled(name)
        self._config_store.save(states)

    def reload_config(self) -> None:
        """Reload the persisted config from disk."""
        self._config_store.reload()

    def get_effective_states(self) -> dict[str, bool]:
        """Return a snapshot of all registered feature states."""
        return {name: self.is_enabled(name) for name in self._features}

    # ------------------------------------------------------------------
    # Circular dependency detection
    # ------------------------------------------------------------------

    def detect_circular_deps(self) -> list[list[str]]:
        """Detect circular dependency chains among registered features.

        Returns:
            A list of cycles, where each cycle is a list of feature names
            forming a directed loop.  An empty list means no cycles exist.
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            flag = self._features.get(node)
            if flag:
                for dep in flag.deps:
                    if dep not in self._features:
                        continue  # external dep, skip
                    if dep not in visited:
                        _dfs(dep, path)
                    elif dep in rec_stack:
                        # Found a cycle — extract it.
                        cycle_start = path.index(dep)
                        cycle = path[cycle_start:] + [dep]
                        cycles.append(cycle)

            path.pop()
            rec_stack.discard(node)

        for name in self._features:
            if name not in visited:
                _dfs(name, [])

        return cycles

    # ------------------------------------------------------------------
    # Transitive dependency resolution
    # ------------------------------------------------------------------

    def get_transitive_deps(self, name: str) -> list[str]:
        """Return all transitive dependencies of *name* (direct + indirect).

        Only includes features that are actually registered.  The result
        is topologically ordered so that dependencies appear before the
        features that depend on them.

        Args:
            name: The feature flag name.

        Returns:
            A deduplicated list of transitive dependency names.
        """
        flag = self._features.get(name)
        if not flag:
            return []

        result: list[str] = []
        seen: set[str] = set()

        def _collect(n: str) -> None:
            if n in seen:
                return
            seen.add(n)
            f = self._features.get(n)
            if not f:
                return
            for dep in f.deps:
                if dep in self._features and dep != n:
                    _collect(dep)
            result.append(n)

        _collect(name)
        # Remove the feature itself; return only its deps.
        return [r for r in result if r != name]

    def resolve_all_deps(self) -> dict[str, list[str]]:
        """Compute transitive dependencies for every registered feature.

        Returns:
            A dict mapping feature name → list of transitive deps.
        """
        return {name: self.get_transitive_deps(name) for name in self._features}

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_all(self) -> tuple[bool, list[str]]:
        """Validate all registered features.

        Checks for:
        - Circular dependencies
        - Missing dependencies (referenced but not registered)
        - Self-referential deps

        Returns:
            ``(ok, errors)`` — ``ok`` is ``True`` when there are no issues.
        """
        errors: list[str] = []

        # Circular deps
        cycles = self.detect_circular_deps()
        for cycle in cycles:
            errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")

        # Missing deps
        for name, flag in self._features.items():
            for dep in flag.deps:
                if dep not in self._features:
                    errors.append(f"Feature '{name}' depends on unregistered feature '{dep}'")
            for mx in flag.mutex_with:
                if mx not in self._features:
                    errors.append(f"Feature '{name}' mutex-with unregistered feature '{mx}'")
            # Self-reference
            if name in flag.deps:
                errors.append(f"Feature '{name}' has a self-referential dependency")

        return (len(errors) == 0, errors)
