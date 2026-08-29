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

"""
Pydantic-settings adapter for ClawCodex configuration.

This module provides a Pydantic Settings-based configuration backend
that can replace manual JSON config management. It maintains backward
compatibility with the existing ConfigManager API.

Architecture:
    src/config.py (ConfigManager API)
        ↓
    src/settings/pydantic_adapter.py (This module - Pydantic Settings backend)
        ↓
    pydantic-settings + python-dotenv (Open source dependency)

Switch:
    CLAW_USE_PYDANTIC_SETTINGS=true (default) - use Pydantic Settings
    CLAW_USE_PYDANTIC_SETTINGS=false - fallback to manual JSON config
"""

from __future__ import annotations

# pylint: disable=no-name-in-module  # capabilities: pending patch migration
from clawcodex_ext.capabilities import AdapterRegistry, env_switch
import logging
from importlib.util import find_spec
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import (
    ConfigManager,
)

logger = logging.getLogger(__name__)

# Switching mechanism: control via environment variable
_USE_PYDANTIC_SETTINGS = env_switch("CLAW_USE_PYDANTIC_SETTINGS")


# ---------------------------------------------------------------------------
# Pydantic Models for type-safe config
# ---------------------------------------------------------------------------


@AdapterRegistry.register("pydantic_settings", env_var="CLAW_USE_PYDANTIC_SETTINGS", dependency="pydantic_settings")
class ProviderConfig(BaseModel):
    """Provider-specific configuration."""

    api_key: str = ""
    base_url: str = ""
    default_model: str = ""


class SessionConfig(BaseModel):
    """Session configuration."""

    auto_save: bool = True
    max_history: int = 100


class ClawCodexSettings(BaseSettings):
    """
    Pydantic Settings-based configuration for ClawCodex.

    Loads configuration from multiple sources in order of precedence:
    1. Environment variables (highest priority)
    2. Local config file (~/.clawcodex/config.json)
    3. Project config file (<git-root>/.claude/config.json)
    4. Defaults (lowest priority)

    Environment variables use CLAWCODEX_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAWCODEX_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Core settings
    default_provider: str = Field(default="anthropic", alias="default_provider")
    model: str = Field(default="", alias="model")
    max_turns: int = Field(default=0, ge=0, alias="max_turns")

    # Provider configs stored as nested dict
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    session: SessionConfig = Field(default_factory=SessionConfig)

    @field_validator("providers", mode="before")
    @classmethod
    def parse_providers(cls, v: Any) -> dict[str, ProviderConfig]:
        if isinstance(v, dict):
            result = {}
            for name, config in v.items():
                if isinstance(config, dict):
                    result[name] = ProviderConfig(**config)
                else:
                    result[name] = config
            return result
        return {}

    @field_validator("session", mode="before")
    @classmethod
    def parse_session(cls, v: Any) -> SessionConfig:
        if isinstance(v, dict):
            return SessionConfig(**v)
        if isinstance(v, SessionConfig):
            return v
        return SessionConfig()


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------


def load_settings_from_config_manager(
    config_manager: ConfigManager | None = None,
    cwd: str | Path | None = None,
) -> ClawCodexSettings:
    """
    Load settings from ConfigManager and convert to Pydantic Settings.

    This function bridges the legacy ConfigManager JSON-based config
    with the new Pydantic Settings model.
    """
    if config_manager is None:
        config_manager = ConfigManager(cwd=cwd)

    # Get merged config from legacy system
    merged = config_manager.get_merged()

    # Extract settings section if present
    settings_data = merged.get("settings", {})

    # Merge remaining config
    config_data = {**merged, **settings_data}

    # Convert to Pydantic Settings
    try:
        return ClawCodexSettings(**config_data)
    except Exception as e:
        logger.warning("Failed to load Pydantic settings: %s, using defaults", e)
        return ClawCodexSettings()


def settings_to_dict(settings: ClawCodexSettings) -> dict[str, Any]:
    """Convert Pydantic Settings to dict for backward compatibility."""
    return settings.model_dump(exclude_none=True, by_alias=True)


def dict_to_settings(data: dict[str, Any]) -> ClawCodexSettings:
    """Convert dict to Pydantic Settings."""
    return ClawCodexSettings(**data)


@lru_cache(maxsize=1)
def get_cached_settings(
    cwd: str | Path | None = None,
) -> ClawCodexSettings:
    """
    Get cached Pydantic Settings instance.

    Cache is invalidated when config files change.
    """
    return load_settings_from_config_manager(cwd=cwd)


def invalidate_settings_cache() -> None:
    """Clear the cached settings instance."""
    get_cached_settings.cache_clear()


# ---------------------------------------------------------------------------
# Backward compatibility helpers
# ---------------------------------------------------------------------------


def is_pydantic_settings_available() -> bool:
    """Check if pydantic-settings is available."""
    return find_spec("pydantic_settings") is not None


def get_pydantic_settings_class() -> type[ClawCodexSettings]:
    """Get the Pydantic Settings class for external use."""
    return ClawCodexSettings
