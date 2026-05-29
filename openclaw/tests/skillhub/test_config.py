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


import os
from pathlib import Path
from typing import Generator

import pytest

from skillhub.config import (
    Settings,
    PlatformConfig,
    CacheConfig,
    SecurityConfig,
    DiscoveryConfig,
    get_config,
)


@pytest.fixture
def temp_config_dir() -> Generator[Path, None, None]:
    """Create temporary config directory."""
    import tempfile
    import shutil

    directory = Path(tempfile.mkdtemp())
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


class TestPlatformConfig:
    """Tests for PlatformConfig."""

    def test_platform_config_defaults(self):
        """Test platform config default values."""
        config = PlatformConfig(api_url="https://api.example.com")
        assert config.api_url == "https://api.example.com"
        assert config.auth_type == "pat"
        assert config.rate_limit == 5000
        assert config.timeout == 30

    def test_platform_config_custom_values(self):
        """Test platform config with custom values."""
        config = PlatformConfig(
            api_url="https://custom.api.com",
            auth_type="oauth",
            rate_limit=1000,
            timeout=60,
        )
        assert config.auth_type == "oauth"
        assert config.rate_limit == 1000

    def test_platform_config_model_dump(self):
        """Test converting platform config to dict."""
        config = PlatformConfig(api_url="https://api.test.com")
        data = config.model_dump()
        assert data["api_url"] == "https://api.test.com"


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_cache_config_defaults(self):
        """Test cache config default values."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.ttl_metadata == 3600
        assert config.ttl_search == 1800
        assert config.ttl_releases == 7200
        assert config.max_size_mb == 1024

    def test_cache_config_disabled(self):
        """Test cache config when disabled."""
        config = CacheConfig(enabled=False)
        assert config.enabled is False

    def test_cache_config_custom_ttl(self):
        """Test cache config with custom TTL."""
        config = CacheConfig(
            ttl_metadata=7200,
            ttl_search=3600,
        )
        assert config.ttl_metadata == 7200
        assert config.ttl_search == 3600


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_security_config_defaults(self):
        """Test security config default values."""
        config = SecurityConfig()
        assert config.allow_unsigned is False
        assert config.sandbox_installs is True
        assert config.strict_permissions is True
        assert config.trusted_authors == []
        assert config.trusted_sources == []

    def test_security_config_allow_unsigned(self):
        """Test security config allowing unsigned."""
        config = SecurityConfig(allow_unsigned=True)
        assert config.allow_unsigned is True

    def test_security_config_trusted_sources(self):
        """Test security config with trusted sources."""
        config = SecurityConfig(
            trusted_sources=["github.com", "gitee.com"],
        )
        assert "github.com" in config.trusted_sources

    def test_security_config_trusted_authors(self):
        """Test security config with trusted authors."""
        config = SecurityConfig(
            trusted_authors=["trusted-author"],
        )
        assert "trusted-author" in config.trusted_authors


class TestDiscoveryConfig:
    """Tests for DiscoveryConfig."""

    def test_discovery_config_defaults(self):
        """Test discovery config default values."""
        config = DiscoveryConfig()
        assert config.default_sources == []
        assert config.search_timeout == 30
        assert config.max_results == 100

    def test_discovery_config_custom_values(self):
        """Test discovery config with custom values."""
        config = DiscoveryConfig(
            default_sources=["github", "gitee"],
            search_timeout=60,
            max_results=50,
        )
        assert len(config.default_sources) == 2
        assert config.search_timeout == 60


class TestSettings:
    """Tests for Settings."""

    def test_settings_default_paths(self, mock_settings: Settings):
        """Test settings with default paths from fixture."""
        assert mock_settings.config_dir is not None
        assert mock_settings.cache_dir is not None
        assert mock_settings.data_dir is not None
        assert mock_settings.skills_dir is not None

    def test_settings_cache_config(self, mock_settings: Settings):
        """Test settings contains cache config."""
        assert mock_settings.cache.enabled is True
        assert isinstance(mock_settings.cache, CacheConfig)

    def test_settings_security_config(self, mock_settings: Settings):
        """Test settings contains security config."""
        assert isinstance(mock_settings.security, SecurityConfig)
        assert mock_settings.security.sandbox_installs is True

    def test_settings_discovery_config(self, mock_settings: Settings):
        """Test settings contains discovery config."""
        assert isinstance(mock_settings.discovery, DiscoveryConfig)
        assert mock_settings.discovery.max_results == 100

    def test_settings_platform_configs(self, mock_settings: Settings):
        """Test settings contains platform configs."""
        assert mock_settings.github.api_url == "https://api.github.com"
        assert mock_settings.gitee.api_url == "https://gitee.com/api/v5"
        assert mock_settings.gitcode.api_url == "https://api.gitcode.com/api/v5"

    def test_settings_log_level(self, mock_settings: Settings):
        """Test settings log level."""
        assert mock_settings.log_level == "INFO"

    def test_settings_log_file_optional(self, mock_settings: Settings):
        """Test settings log file is optional."""
        assert mock_settings.log_file is None

    def test_settings_model_dump(self, mock_settings: Settings):
        """Test converting settings to dict."""
        data = mock_settings.model_dump()
        assert "cache" in data
        assert "security" in data

    def test_settings_dirs_created(self, temp_config_dir: Path):  # pylint: disable=redefined-outer-name
        """Test that directories are created."""
        cache_dir = temp_config_dir / "cache"
        data_dir = temp_config_dir / "data"
        skills_dir = temp_config_dir / "skills"

        settings = Settings(
            config_dir=temp_config_dir,
            cache_dir=cache_dir,
            data_dir=data_dir,
            skills_dir=skills_dir,
        )

        assert settings.config_dir.exists()
        assert settings.cache_dir.exists()
        assert settings.data_dir.exists()
        assert settings.skills_dir.exists()

    def test_settings_env_prefix(self):
        """Test that settings uses SKILLHUB_ env prefix."""
        original = os.environ.get("SKILLHUB_LOG_LEVEL")
        try:
            os.environ["SKILLHUB_LOG_LEVEL"] = "DEBUG"
            settings = Settings()
            assert settings.log_level == "DEBUG"
        finally:
            if original:
                os.environ["SKILLHUB_LOG_LEVEL"] = original
            else:
                os.environ.pop("SKILLHUB_LOG_LEVEL", None)


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_default(self):
        """Test getting default config."""
        config = get_config()
        assert isinstance(config, Settings)

    def test_get_config_with_path(self, temp_config_dir: Path):  # pylint: disable=redefined-outer-name
        """Test getting config with custom path."""
        env_file = temp_config_dir / ".env"
        env_file.write_text("SKILLHUB_LOG_LEVEL=WARNING\n")

        config = get_config(config_path=str(env_file))
        assert config.log_level == "WARNING"

    def test_get_config_returns_settings(self):
        """Test that get_config returns Settings instance."""
        config = get_config()
        assert isinstance(config, Settings)
