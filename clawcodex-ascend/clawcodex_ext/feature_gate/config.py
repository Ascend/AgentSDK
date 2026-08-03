#!/usr/bin/env python3
# coding=utf-8

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

"""Config persistence for feature flags.

Stores and loads feature states to/from ``~/.clawcodex/features.json``
(or ``features.yaml`` when PyYAML is available).  Supports both JSON
and YAML formats for maximum flexibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default location for the feature-gate config file.
_DEFAULT_CONFIG_DIR = Path.home() / ".clawcodex"
_DEFAULT_CONFIG_FILE_JSON = _DEFAULT_CONFIG_DIR / "features.json"
_DEFAULT_CONFIG_FILE_YAML = _DEFAULT_CONFIG_DIR / "features.yaml"


class ConfigStore:
    """Read/write feature-flag states to a JSON (or YAML) file.

    All operations are best-effort: file I/O errors are logged but
    never raised so that the feature-gate system never blocks the CLI.
    """

    def __init__(
        self,
        config_dir: Path | None = None,
        config_file: Path | None = None,
    ) -> None:
        self._config_dir = config_dir or _DEFAULT_CONFIG_DIR
        # Allow caller to override the file path entirely.
        if config_file is not None:
            self._config_file = config_file
        else:
            self._config_file = self._resolve_config_file()
        self._cache: dict[str, Any] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Config-file resolution: prefer JSON, fall back to YAML.
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse_yaml(text: str) -> dict[str, Any] | None:
        """Best-effort YAML parsing (PyYAML optional)."""
        try:
            import yaml  # noqa: F401  # ensures PyYAML is importable
        except ImportError:
            return None
        try:
            data = yaml.safe_load(text)
            if isinstance(data, dict):
                return data
        except Exception as exc:  # noqa: BLE001 - optional YAML support
            logger.debug("Could not parse feature-gate YAML", exc_info=exc)
        return None

    def _resolve_config_file(self) -> Path:
        """Pick the best config file: JSON preferred, then YAML.

        When *config_dir* already contains a features file (JSON or YAML),
        that file is used.  Otherwise the default is ``{config_dir}/features.json``
        — the hardcoded ``~/.clawcodex/features.json`` fallback is only used
        when the caller did *not* provide an explicit ``config_dir``.
        """
        json_path = self._config_dir / "features.json"
        yaml_path = self._config_dir / "features.yaml"
        if self._config_dir.exists():
            if json_path.exists():
                return json_path
            if yaml_path.exists():
                return yaml_path
        # Prefer the caller's config_dir; fall back to the global default
        # only when the caller used the implicit default directory.
        if self._config_dir == _DEFAULT_CONFIG_DIR:
            return _DEFAULT_CONFIG_FILE_JSON
        return json_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> bool | None:
        """Return the persisted value for *name*, or ``None`` if absent."""
        self._ensure_loaded()
        return self._cache.get(name)

    def save(self, states: dict[str, bool]) -> None:
        """Persist *states* to disk.

        Args:
            states: Mapping of feature name → enabled state.
        """
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            with open(self._config_file, "w", encoding="utf-8") as fh:
                json.dump(states, fh, indent=2, sort_keys=True)
            # Update cache so subsequent reads are instant.
            self._cache.update(states)
            self._loaded = True
            logger.debug("Saved feature states to %s", self._config_file)
        except OSError as exc:
            logger.warning("Failed to save feature config to %s: %s", self._config_file, exc)

    def reload(self) -> None:
        """Force-reload the config file from disk."""
        self._loaded = False
        self._cache.clear()
        self._ensure_loaded()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the config file lazily on first access."""
        if self._loaded:
            return
        try:
            if self._config_file.exists():
                text = self._config_file.read_text(encoding="utf-8")
                # Try YAML first (it's a superset of JSON for simple dicts),
                # then fall back to JSON parser.
                data = self._try_parse_yaml(text)
                if data is None:
                    data = json.loads(text)
                if isinstance(data, dict):
                    self._cache = data
                else:
                    self._cache = {}
                self._loaded = True
                logger.debug("Loaded feature config from %s", self._config_file)
            else:
                self._loaded = True  # No file yet; considered empty.
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Failed to load feature config from %s: %s", self._config_file, exc)
            self._cache.clear()
            self._loaded = True

    @property
    def config_file(self) -> Path:
        """Path to the persisted config file."""
        return self._config_file

    @property
    def config_dir(self) -> Path:
        """Path to the config directory."""
        return self._config_dir
