"""Shared fixtures for SkillHub tests."""

# pylint: disable=redefined-outer-name
# pylint: disable=no-name-in-module

import tempfile
import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

import pytest

from skillhub.config import Settings, CacheConfig, SecurityConfig, DiscoveryConfig
from skillhub.models.skill import SkillManifest, InstalledSkill, DiscoveredSkill, ResolvedSkill
from skillhub.models.source import Source, SourceType
from skillhub.models.cache import CacheOptions
from skillhub.models.security import InstallEvent, SandboxOptions
from skillhub.models.credential import TokenInfo
from skillhub.models.repository import Repository, Release, Tag, RateLimit


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory."""
    directory = Path(tempfile.mkdtemp())
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def temp_config_dir(temp_dir: Path) -> Path:
    """Create a temporary config directory."""
    config_dir = temp_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def temp_cache_dir(temp_dir: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Create a temporary data directory."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_skills_dir(temp_dir: Path) -> Path:
    """Create a temporary skills directory."""
    skills_dir = temp_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    return skills_dir


@pytest.fixture
def mock_settings(
    temp_config_dir: Path,
    temp_cache_dir: Path,
    temp_data_dir: Path,
    temp_skills_dir: Path,
) -> Settings:
    """Create mock settings with temporary directories."""
    return Settings(
        config_dir=temp_config_dir,
        cache_dir=temp_cache_dir,
        data_dir=temp_data_dir,
        skills_dir=temp_skills_dir,
        cache=CacheConfig(),
        security=SecurityConfig(),
        discovery=DiscoveryConfig(),
    )


@pytest.fixture
def sample_manifest() -> SkillManifest:
    """Create a sample skill manifest."""
    return SkillManifest(
        name="test-skill",
        description="A test skill for testing",
        version="1.0.0",
        author="test-author",
        tags=["test", "demo"],
        license="MIT",
        dependencies={"requests": ">=2.28.0"},
    )


@pytest.fixture
def sample_source() -> Source:
    """Create a sample source configuration."""
    return Source(
        id="test-source",
        name="Test Source",
        type=SourceType.GITHUB,
        url="https://github.com/test/skills",
        priority=1,
        enabled=True,
    )


@pytest.fixture
def sample_installed_skill(sample_manifest: SkillManifest) -> InstalledSkill:
    """Create a sample installed skill record."""
    return InstalledSkill(
        name=sample_manifest.name,
        version=sample_manifest.version or "1.0.0",
        source_id="github",
        source_type="github",
        repository="test/skills",
        ref="v1.0.0",
        install_path="/skills/test-skill",
        checksum="abc123def456",
    )


@pytest.fixture
def sample_discovered_skill() -> DiscoveredSkill:
    """Create a sample discovered skill."""
    return DiscoveredSkill(
        name="demo-skill",
        version="latest",
        description="A discovered skill",
        author="demo-author",
        tags=["demo"],
        source={"id": "github", "name": "GitHub", "type": "github"},
        repository={"owner": "demo", "name": "skills", "url": "https://github.com/demo/skills"},
        manifest_url="https://github.com/demo/skills/blob/main/skills/demo-skill/SKILL.md",
        available_versions=["latest", "v1.0.0"],
    )


@pytest.fixture
def sample_resolved_skill(sample_manifest: SkillManifest) -> ResolvedSkill:
    """Create a sample resolved skill."""
    return ResolvedSkill(
        name=sample_manifest.name,
        version=sample_manifest.version or "1.0.0",
        repository="test/skills",
        ref="v1.0.0",
        manifest=sample_manifest,
        source={"id": "github", "name": "GitHub", "type": "github"},
        download_url="https://github.com/test/skills/archive/v1.0.0.tar.gz",
    )


@pytest.fixture
def sample_release() -> Release:
    """Create a sample release."""
    return Release(
        id="release-1",
        tag_name="v1.0.0",
        name="Release 1.0.0",
        body="Initial release",
        prerelease=False,
        draft=False,
        assets=[],
        tarball_url="https://example.com/tarball",
        zipball_url="https://example.com/zipball",
    )


@pytest.fixture
def sample_tag() -> Tag:
    """Create a sample tag."""
    return Tag(
        name="v1.0.0",
        commit={"sha": "abc123", "url": "https://example.com/commit"},
        tarball_url="https://example.com/tarball",
        zipball_url="https://example.com/zipball",
    )


@pytest.fixture
def sample_repository() -> Repository:
    """Create a sample repository."""
    return Repository(
        id="repo-1",
        name="test-repo",
        full_name="test/test-repo",
        description="A test repository",
        owner={"login": "test", "id": 1},
        url="https://github.com/test/test-repo",
        clone_url="https://github.com/test/test-repo.git",
        topics=["test", "demo"],
        is_private=False,
        stars=100,
        forks=10,
        language="Python",
        default_branch="main",
    )


@pytest.fixture
def sample_rate_limit() -> RateLimit:
    """Create a sample rate limit."""
    return RateLimit(
        limit=5000,
        remaining=4999,
        reset_at=datetime.utcnow(),
        used=1,
    )


@pytest.fixture
def sample_token_info() -> TokenInfo:
    """Create a sample token info."""
    return TokenInfo(
        platform="github",
        type="pat",
        has_token=True,
        expires_at=None,
        scopes=["repo", "read:org"],
    )


@pytest.fixture
def sample_install_event() -> InstallEvent:
    """Create a sample install event."""
    return InstallEvent(
        timestamp=datetime.utcnow(),
        skill="test-skill",
        version="1.0.0",
        source="github",
        repository="test/repo",
        ref="v1.0.0",
        checksum="abc123",
        success=True,
    )


@pytest.fixture
def sample_sandbox_options() -> SandboxOptions:
    """Create sample sandbox options."""
    return SandboxOptions(
        read_only_dirs=["/read"],
        write_dirs=["/write"],
        network=False,
        timeout=300,
        memory_limit=512,
    )


@pytest.fixture
def sample_cache_options() -> CacheOptions:
    """Create sample cache options."""
    return CacheOptions(
        ttl=3600,
        tags=["test"],
    )


@pytest.fixture
def mock_keyring():
    """Mock keyring for credential tests."""
    with patch("skillhub.services.credential_manager.keyring") as mock:
        mock.get_password.return_value = "test_token_123"
        mock.set_password.return_value = None
        mock.delete_password.return_value = None
        yield mock


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for adapter tests."""
    with patch("skillhub.adapters.base.httpx.AsyncClient") as mock:
        mock_client = AsyncMock()
        mock.return_value.__aenter__.return_value = mock_client
        mock.return_value.__aexit__.return_value = None
        yield mock_client


@pytest.fixture
def mock_diskcache():
    """Mock diskcache for cache manager tests."""
    with patch("skillhub.services.cache_manager.diskcache.Cache") as mock:
        mock_cache = MagicMock()
        mock.return_value = mock_cache
        yield mock_cache
