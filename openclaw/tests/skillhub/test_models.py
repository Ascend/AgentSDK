"""Tests for SkillHub models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from skillhub.models.skill import (
    SkillManifest,
    InstalledSkill,
    DiscoveredSkill,
    ResolvedSkill,
    InstallResult,
)
from skillhub.models.source import Source, SourceType
from skillhub.models.cache import CacheOptions, CacheStats
from skillhub.models.security import InstallEvent, SandboxOptions, SandboxResult
from skillhub.models.credential import TokenInfo, TokenValidation
from skillhub.models.repository import Repository, Release, Tag, ContentItem, RateLimit


class TestSkillManifest:
    """Tests for SkillManifest model."""

    def test_manifest_creation_valid(self, sample_manifest: SkillManifest):
        """Test creating a valid manifest."""
        assert sample_manifest.name == "test-skill"
        assert sample_manifest.description == "A test skill for testing"
        assert sample_manifest.version == "1.0.0"
        assert sample_manifest.author == "test-author"
        assert sample_manifest.tags == ["test", "demo"]
        assert sample_manifest.license == "MIT"

    def test_manifest_name_validation_empty(self):
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            SkillManifest(name="", description="Test")

    def test_manifest_name_validation_short(self):
        """Test that short name raises ValidationError."""
        with pytest.raises(ValidationError):
            SkillManifest(name="x", description="Test")

    def test_manifest_name_normalization(self):
        """Test that name is normalized to lowercase and hyphens."""
        manifest = SkillManifest(name="Test Skill Name", description="Test")
        assert manifest.name == "test-skill-name"

    def test_manifest_minimal_fields(self):
        """Test creating manifest with minimal required fields."""
        manifest = SkillManifest(name="minimal", description="Minimal skill")
        assert manifest.name == "minimal"
        assert manifest.description == "Minimal skill"
        assert manifest.version is None
        assert manifest.tags == []

    def test_manifest_to_dict(self, sample_manifest: SkillManifest):
        """Test converting manifest to dictionary."""
        data = sample_manifest.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "test-skill"

    def test_manifest_from_dict(self):
        """Test creating manifest from dictionary."""
        data = {
            "name": "from-dict",
            "description": "From dict",
            "version": "0.1.0",
        }
        manifest = SkillManifest(**data)
        assert manifest.name == "from-dict"

    def test_manifest_equality(self, sample_manifest: SkillManifest):
        """Test manifest equality comparison."""
        manifest2 = SkillManifest(
            name="test-skill",
            description="A test skill for testing",
            version="1.0.0",
        )
        assert sample_manifest.name == manifest2.name


class TestInstalledSkill:
    """Tests for InstalledSkill model."""

    def test_installed_skill_creation(self, sample_installed_skill: InstalledSkill):
        """Test creating an installed skill record."""
        assert sample_installed_skill.name == "test-skill"
        assert sample_installed_skill.version == "1.0.0"
        assert sample_installed_skill.source_id == "github"
        assert sample_installed_skill.checksum == "abc123def456"

    def test_installed_skill_timestamps(self, sample_installed_skill: InstalledSkill):
        """Test that timestamps are set correctly."""
        assert sample_installed_skill.installed_at is not None
        assert sample_installed_skill.updated_at is not None

    def test_installed_skill_config_default(self):
        """Test that config defaults to empty dict."""
        installed = InstalledSkill(
            name="skill",
            version="1.0",
            source_id="test",
            source_type="github",
            repository="test/repo",
            ref="v1.0",
            install_path="/path",
            checksum="abc",
        )
        assert installed.config == {}

    def test_installed_skill_model_dump(self, sample_installed_skill: InstalledSkill):
        """Test converting installed skill to dict."""
        data = sample_installed_skill.model_dump()
        assert data["name"] == "test-skill"


class TestDiscoveredSkill:
    """Tests for DiscoveredSkill model."""

    def test_discovered_skill_creation(self, sample_discovered_skill: DiscoveredSkill):
        """Test creating a discovered skill."""
        assert sample_discovered_skill.name == "demo-skill"
        assert sample_discovered_skill.version == "latest"
        assert sample_discovered_skill.tags == ["demo"]

    def test_discovered_skill_source_dict(self, sample_discovered_skill: DiscoveredSkill):
        """Test that source is a dict."""
        assert isinstance(sample_discovered_skill.source, dict)
        assert sample_discovered_skill.source["id"] == "github"

    def test_discovered_skill_repository_dict(self, sample_discovered_skill: DiscoveredSkill):
        """Test that repository is a dict."""
        assert isinstance(sample_discovered_skill.repository, dict)
        assert sample_discovered_skill.repository["owner"] == "demo"


class TestResolvedSkill:
    """Tests for ResolvedSkill model."""

    def test_resolved_skill_creation(self, sample_resolved_skill: ResolvedSkill):
        """Test creating a resolved skill."""
        assert sample_resolved_skill.name == "test-skill"
        assert sample_resolved_skill.ref == "v1.0.0"
        assert sample_resolved_skill.manifest is not None

    def test_resolved_skill_subpath_optional(self, sample_resolved_skill: ResolvedSkill):
        """Test that subpath is optional."""
        assert sample_resolved_skill.subpath is None

    def test_resolved_skill_with_subpath(self, sample_manifest: SkillManifest):
        """Test resolved skill with subpath."""
        resolved = ResolvedSkill(
            name="skill",
            version="1.0",
            repository="test/repo",
            ref="v1.0",
            manifest=sample_manifest,
            source={"id": "test"},
            download_url="https://example.com/download",
            subpath="skills/skill",
        )
        assert resolved.subpath == "skills/skill"


class TestInstallResult:
    """Tests for InstallResult model."""

    def test_install_result_success(self):
        """Test creating successful install result."""
        installed = InstalledSkill(
            name="skill",
            version="1.0",
            source_id="test",
            source_type="github",
            repository="test/repo",
            ref="v1.0",
            install_path="/path",
            checksum="abc",
        )
        result = InstallResult(
            success=True,
            skill=installed,
            installed_dependencies=["dep1"],
            warnings=["warn1"],
            errors=[],
            duration=1.5,
        )
        assert result.success is True
        assert len(result.installed_dependencies) == 1


class TestSource:
    """Tests for Source model."""

    def test_source_creation(self, sample_source: Source):
        """Test creating a source."""
        assert sample_source.id == "test-source"
        assert sample_source.name == "Test Source"
        assert sample_source.type == SourceType.GITHUB

    def test_source_default_id(self):
        """Test that source id is auto-generated."""
        source = Source(
            name="Auto ID",
            type=SourceType.GITHUB,
            url="https://github.com/test",
        )
        assert source.id is not None
        assert len(source.id) == 8

    def test_source_default_priority(self):
        """Test that priority defaults to 0."""
        source = Source(
            name="Default Priority",
            type=SourceType.GITHUB,
            url="https://github.com/test",
        )
        assert source.priority == 0

    def test_source_enabled_default(self):
        """Test that enabled defaults to True."""
        source = Source(
            name="Default Enabled",
            type=SourceType.GITHUB,
            url="https://github.com/test",
        )
        assert source.enabled is True

    def test_source_types(self):
        """Test all source types."""
        assert SourceType.GITHUB == "github"
        assert SourceType.GITEE == "gitee"
        assert SourceType.GITCODE == "gitcode"
        assert SourceType.MANIFEST == "manifest"

    def test_source_model_dump(self, sample_source: Source):
        """Test converting source to dict."""
        data = sample_source.model_dump()
        assert data["type"] == "github"


class TestCacheOptions:
    """Tests for CacheOptions model."""

    def test_cache_options_creation(self, sample_cache_options: CacheOptions):
        """Test creating cache options."""
        assert sample_cache_options.ttl == 3600
        assert sample_cache_options.tags == ["test"]

    def test_cache_options_defaults(self):
        """Test cache options defaults."""
        options = CacheOptions()
        assert options.ttl is None
        assert options.tags == []


class TestCacheStats:
    """Tests for CacheStats model."""

    def test_cache_stats_creation(self):
        """Test creating cache stats."""
        stats = CacheStats(
            size=100,
            hit_rate=0.8,
            miss_rate=0.2,
            total_size=1024,
            oldest_entry=None,
            newest_entry=None,
        )
        assert stats.size == 100
        assert stats.hit_rate == 0.8


class TestInstallEvent:
    """Tests for InstallEvent model."""

    def test_install_event_creation(self, sample_install_event: InstallEvent):
        """Test creating install event."""
        assert sample_install_event.skill == "test-skill"
        assert sample_install_event.success is True

    def test_install_event_success_values(self):
        """Test install event with different success values."""
        for success in [True, False]:
            event = InstallEvent(
                timestamp=datetime.utcnow(),
                skill="skill",
                version="1.0",
                source="github",
                repository="test/repo",
                ref="v1.0",
                checksum="abc",
                success=success,
            )
            assert event.success == success


class TestSandboxOptions:
    """Tests for SandboxOptions model."""

    def test_sandbox_options_creation(self, sample_sandbox_options: SandboxOptions):
        """Test creating sandbox options."""
        assert sample_sandbox_options.timeout == 300
        assert sample_sandbox_options.network is False

    def test_sandbox_options_defaults(self):
        """Test sandbox options defaults."""
        options = SandboxOptions()
        assert options.timeout == 300
        assert options.network is False


class TestSandboxResult:
    """Tests for SandboxResult model."""

    def test_sandbox_result_success(self):
        """Test successful sandbox result."""
        result = SandboxResult(
            exit_code=0,
            stdout="output",
            stderr="",
            duration=1.0,
        )
        assert result.exit_code == 0

    def test_sandbox_result_failure(self):
        """Test failed sandbox result."""
        result = SandboxResult(
            exit_code=1,
            stdout="",
            stderr="error",
            duration=1.0,
        )
        assert result.exit_code == 1
        assert result.stderr == "error"


class TestTokenInfo:
    """Tests for TokenInfo model."""

    def test_token_info_creation(self, sample_token_info: TokenInfo):
        """Test creating token info."""
        assert sample_token_info.platform == "github"
        assert sample_token_info.has_token is True

    def test_token_info_no_token(self):
        """Test token info with no token."""
        info = TokenInfo(
            platform="gitee",
            type="pat",
            has_token=False,
        )
        assert info.has_token is False


class TestTokenValidation:
    """Tests for TokenValidation model."""

    def test_token_validation_valid(self):
        """Test valid token validation."""
        validation = TokenValidation(
            valid=True,
            scopes=["repo"],
            rate_limit=None,
            message="Token is valid",
        )
        assert validation.valid is True

    def test_token_validation_invalid(self):
        """Test invalid token validation."""
        validation = TokenValidation(
            valid=False,
            scopes=[],
            rate_limit=None,
            message="Invalid token",
        )
        assert validation.valid is False


class TestRepository:
    """Tests for Repository model."""

    def test_repository_creation(self, sample_repository: Repository):
        """Test creating repository."""
        assert sample_repository.name == "test-repo"
        assert sample_repository.full_name == "test/test-repo"
        assert sample_repository.is_private is False


class TestRelease:
    """Tests for Release model."""

    def test_release_creation(self, sample_release: Release):
        """Test creating release."""
        assert sample_release.tag_name == "v1.0.0"
        assert sample_release.draft is False


class TestTag:
    """Tests for Tag model."""

    def test_tag_creation(self, sample_tag: Tag):
        """Test creating tag."""
        assert sample_tag.name == "v1.0.0"
        assert sample_tag.commit["sha"] == "abc123"


class TestRateLimit:
    """Tests for RateLimit model."""

    def test_rate_limit_creation(self, sample_rate_limit: RateLimit):
        """Test creating rate limit."""
        assert sample_rate_limit.limit == 5000
        assert sample_rate_limit.remaining == 4999
        assert sample_rate_limit.used == 1


class TestContentItem:
    """Tests for ContentItem model."""

    def test_content_item_file(self):
        """Test content item as file."""
        item = ContentItem(
            type="file",
            name="file.txt",
            path="path/file.txt",
            sha="abc123",
            size=100,
            url="https://example.com/file",
            content=None,
        )
        assert item.type == "file"

    def test_content_item_dir(self):
        """Test content item as directory."""
        item = ContentItem(
            type="dir",
            name="dir",
            path="path/dir",
            sha="abc123",
            size=0,
            url="https://example.com/dir",
        )
        assert item.type == "dir"
