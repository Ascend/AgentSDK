"""Configuration management for SkillHub CLI."""

import json
from pathlib import Path
from typing import List, Optional
from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


CONFIG_FILE_NAME = "config.json"


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

    # Try to load from config file
    config_file = Path(user_config_dir("skillhub")) / CONFIG_FILE_NAME
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Settings(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            # Config file invalid, use defaults
            pass

    return Settings()


def save_config(config: Settings) -> None:
    """Save configuration to file.

    Args:
        config: Settings to save.
    """
    config_file = config.config_dir / CONFIG_FILE_NAME
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2)


def set_config_value(key: str, value: str) -> Settings:
    """Set a configuration value by key path.

    Args:
        key: Dot-separated key path (e.g., "cache.ttl_metadata").
        value: Value to set (will be converted to appropriate type).

    Returns:
        Updated settings.

    Raises:
        ValueError: If key not found or value type invalid.
    """
    config = get_config()
    keys = key.split(".")

    # Navigate to the target object
    obj = config
    for i, k in enumerate(keys[:-1]):
        if hasattr(obj, k):
            obj = getattr(obj, k)
        else:
            raise ValueError(f"Configuration key not found: {'.'.join(keys[: i + 1])}")

    final_key = keys[-1]
    if not hasattr(obj, final_key):
        raise ValueError(f"Configuration key not found: {key}")

    # Get current value to determine type
    current_value = getattr(obj, final_key)

    # Convert value to appropriate type
    try:
        if isinstance(current_value, bool):
            parsed_value = value.lower() in ("true", "1", "yes", "on")
        elif isinstance(current_value, int):
            parsed_value = int(value)
        elif isinstance(current_value, float):
            parsed_value = float(value)
        elif isinstance(current_value, Path):
            parsed_value = Path(value)
        elif isinstance(current_value, list):
            parsed_value = json.loads(value) if value.startswith("[") else [value]
        elif isinstance(current_value, str):
            parsed_value = value
        else:
            parsed_value = value
    except Exception as e:
        raise ValueError(f"Invalid value type for {key}: {e}")

    # Set the value
    setattr(obj, final_key, parsed_value)

    # Save to file
    save_config(config)

    return config
