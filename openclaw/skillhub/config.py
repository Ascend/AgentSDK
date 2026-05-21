"""Configuration management for SkillHub CLI."""

from pathlib import Path
from typing import List, Optional
from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformConfig(BaseModel):
    """Platform-specific configuration."""

    api_url: str
    auth_type: str = "pat"
    rate_limit: int = 5000
    timeout: int = 30


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    ttl_metadata: int = 3600  # 1 hour
    ttl_search: int = 1800  # 30 minutes
    ttl_releases: int = 7200  # 2 hours
    max_size_mb: int = 1024  # 1 GB


class SecurityConfig(BaseModel):
    """Security configuration."""

    allow_unsigned: bool = False
    sandbox_installs: bool = True
    strict_permissions: bool = True
    trusted_authors: List[str] = Field(default_factory=list)
    trusted_sources: List[str] = Field(default_factory=list)


class DiscoveryConfig(BaseModel):
    """Discovery configuration."""

    default_sources: List[str] = Field(default_factory=list)
    search_timeout: int = 30
    max_results: int = 100


class Settings(BaseSettings):
    """SkillHub configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SKILLHUB_",
        env_file=".env",
        extra="ignore",
    )

    # Paths
    config_dir: Path = Field(default_factory=lambda: Path(user_config_dir("skillhub")))
    data_dir: Path = Field(default_factory=lambda: Path(user_data_dir("skillhub")))
    cache_dir: Path = Field(default_factory=lambda: Path(user_data_dir("skillhub")) / "cache")
    skills_dir: Path = Field(default_factory=lambda: Path(user_data_dir("skillhub")) / "skills")

    # Cache settings
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # Platform settings
    github: PlatformConfig = PlatformConfig(
        api_url="https://api.github.com",
        rate_limit=5000,
    )
    gitee: PlatformConfig = PlatformConfig(
        api_url="https://gitee.com/api/v5",
        rate_limit=5000,
    )
    gitcode: PlatformConfig = PlatformConfig(
        api_url="https://api.gitcode.com/api/v5",
        rate_limit=5000,
    )

    # Security settings
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Discovery settings
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    def model_post_init(self, __context) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)


def get_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Loaded settings.
    """
    if config_path:
        return Settings(_env_file=config_path)
    return Settings()
