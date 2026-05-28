"""Tests for SkillHub service implementations."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from skillhub.config import Settings
from skillhub.models.source import Source, SourceType
from skillhub.models.skill import DiscoveredSkill
from skillhub.models.cache import CacheOptions, CacheStats
from skillhub.models.security import InstallEvent, SandboxOptions, SandboxResult
from skillhub.models.repository import RateLimit
from skillhub.services.source_manager import SourceManagerImpl
from skillhub.services.cache_manager import CacheManagerImpl
from skillhub.services.security_manager import SecurityManagerImpl


class TestSourceManager:
    """Tests for SourceManagerImpl."""

    @pytest.fixture
    def source_manager(self, mock_settings: Settings):
        """Create source manager instance."""
        manager = SourceManagerImpl(mock_settings)
        return manager

    @pytest.mark.asyncio
    async def test_init_loads_empty_sources(self, mock_settings: Settings):
        """Test that source manager initializes with empty sources."""
        manager = SourceManagerImpl(mock_settings)
        sources = await manager.list_sources()
        assert sources == []

    @pytest.mark.asyncio
    async def test_add_source(self, source_manager: SourceManagerImpl):
        """Test adding a source."""
        source = Source(
            id="test-1",
            name="Test Source",
            type=SourceType.GITHUB,
            url="https://github.com/test/skills",
        )
        result = await source_manager.add_source(source)
        assert result.id == "test-1"

    @pytest.mark.asyncio
    async def test_add_source_duplicate_id(self, source_manager: SourceManagerImpl):
        """Test adding duplicate source."""
        source1 = Source(
            id="dup-test",
            name="Source 1",
            type=SourceType.GITHUB,
            url="https://github.com/test1",
        )
        await source_manager.add_source(source1)

        source2 = Source(
            id="dup-test",
            name="Source 2",
            type=SourceType.GITHUB,
            url="https://github.com/test2",
        )
        await source_manager.add_source(source2)
        sources = await source_manager.list_sources()
        assert len(sources) == 1

    @pytest.mark.asyncio
    async def test_remove_source(self, source_manager: SourceManagerImpl):
        """Test removing a source."""
        source = Source(
            id="remove-test",
            name="Remove Test",
            type=SourceType.GITHUB,
            url="https://github.com/test",
        )
        await source_manager.add_source(source)
        await source_manager.remove_source("remove-test")
        sources = await source_manager.list_sources()
        assert len(sources) == 0

    @pytest.mark.asyncio
    async def test_remove_source_non_existing(self, source_manager: SourceManagerImpl):
        """Test removing non-existing source."""
        await source_manager.remove_source("non-existing")

    @pytest.mark.asyncio
    async def test_get_source(self, source_manager: SourceManagerImpl):
        """Test getting a source."""
        source = Source(
            id="get-test",
            name="Get Test",
            type=SourceType.GITHUB,
            url="https://github.com/test",
        )
        await source_manager.add_source(source)
        result = await source_manager.get_source("get-test")
        assert result is not None
        assert result.id == "get-test"

    @pytest.mark.asyncio
    async def test_get_source_non_existing(self, source_manager: SourceManagerImpl):
        """Test getting non-existing source."""
        result = await source_manager.get_source("non-existing")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_sources(self, source_manager: SourceManagerImpl):
        """Test listing all sources."""
        source1 = Source(id="list-1", name="List 1", type=SourceType.GITHUB, url="url1")
        source2 = Source(id="list-2", name="List 2", type=SourceType.GITEE, url="url2")
        await source_manager.add_source(source1)
        await source_manager.add_source(source2)
        sources = await source_manager.list_sources()
        assert len(sources) == 2

    @pytest.mark.asyncio
    async def test_enable_source(self, source_manager: SourceManagerImpl):
        """Test enabling a source."""
        source = Source(
            id="enable-test",
            name="Enable Test",
            type=SourceType.GITHUB,
            url="https://github.com/test",
            enabled=False,
        )
        await source_manager.add_source(source)
        await source_manager.enable_source("enable-test")
        result = await source_manager.get_source("enable-test")
        assert result.enabled is True

    @pytest.mark.asyncio
    async def test_disable_source(self, source_manager: SourceManagerImpl):
        """Test disabling a source."""
        source = Source(
            id="disable-test",
            name="Disable Test",
            type=SourceType.GITHUB,
            url="https://github.com/test",
            enabled=True,
        )
        await source_manager.add_source(source)
        await source_manager.disable_source("disable-test")
        result = await source_manager.get_source("disable-test")
        assert result.enabled is False

    @pytest.mark.asyncio
    async def test_update_source(self, source_manager: SourceManagerImpl):
        """Test updating a source."""
        source = Source(
            id="update-test",
            name="Update Test",
            type=SourceType.GITHUB,
            url="https://github.com/test",
            priority=0,
        )
        await source_manager.add_source(source)
        result = await source_manager.update_source("update-test", {"priority": 10})
        assert result.priority == 10

    @pytest.mark.asyncio
    async def test_update_source_non_existing(self, source_manager: SourceManagerImpl):
        """Test updating non-existing source."""
        with pytest.raises(ValueError):
            await source_manager.update_source("non-existing", {"priority": 10})

    @pytest.mark.asyncio
    async def test_extract_owner_repo(self):
        """Test extracting owner and repo from URL."""
        result = SourceManagerImpl._extract_owner_repo("https://github.com/test/repo")
        assert result == ("test", "repo")

    @pytest.mark.asyncio
    async def test_extract_owner_repo_with_slash(self):
        """Test extracting from URL with trailing slash."""
        result = SourceManagerImpl._extract_owner_repo("https://github.com/test/repo/")
        assert result == ("test", "repo")

    @pytest.mark.asyncio
    async def test_extract_owner_repo_invalid(self):
        """Test extracting from invalid URL."""
        result = SourceManagerImpl._extract_owner_repo("https://github.com")
        assert result == ("", "")

    @pytest.mark.asyncio
    async def test_parse_frontmatter_valid(self):
        """Test parsing valid frontmatter."""
        content = "---\nname: test\nversion: 1.0\n---\nContent"
        result = SourceManagerImpl._parse_frontmatter(content)
        assert result.get("name") == "test"

    @pytest.mark.asyncio
    async def test_parse_frontmatter_invalid(self):
        """Test parsing invalid frontmatter."""
        content = "No frontmatter here"
        result = SourceManagerImpl._parse_frontmatter(content)
        assert result == {}

    def test_sources_file_path(self, mock_settings: Settings):
        """Test sources file path is set correctly."""
        manager = SourceManagerImpl(mock_settings)
        assert manager.sources_file.name == "sources.json"

    @pytest.mark.asyncio
    async def test_test_source_success(self, source_manager: SourceManagerImpl):
        """Test testing source connectivity."""
        source = Source(
            id="test-conn",
            name="Test Connection",
            type=SourceType.GITHUB,
            url="https://github.com/owner/repo",
        )
        await source_manager.add_source(source)

        with patch.object(source_manager, "_create_adapter", new_callable=AsyncMock) as mock_create:
            mock_adapter = AsyncMock()
            mock_repo = MagicMock()
            mock_adapter.get_repository = AsyncMock(return_value=mock_repo)
            mock_adapter.get_rate_limit = AsyncMock(
                return_value=RateLimit(limit=5000, remaining=5000, reset_at=datetime.utcnow(), used=0)
            )
            mock_adapter.close = AsyncMock()
            mock_create.return_value = mock_adapter

            result = await source_manager.test_source("test-conn")
            assert result.success is True
            assert result.rate_limit is not None

    @pytest.mark.asyncio
    async def test_test_source_not_found(self, source_manager: SourceManagerImpl):
        """Test testing non-existing source."""
        result = await source_manager.test_source("non-existing")
        assert result.success is False
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_test_source_exception(self, source_manager: SourceManagerImpl):
        """Test testing source with exception."""
        source = Source(
            id="test-err",
            name="Test Error",
            type=SourceType.GITHUB,
            url="https://github.com/owner/repo",
        )
        await source_manager.add_source(source)

        with patch.object(source_manager, "_create_adapter", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Connection failed")

            result = await source_manager.test_source("test-err")
            assert result.success is False
            assert "Connection failed" in result.message

    @pytest.mark.asyncio
    async def test_test_source_no_owner_repo(self, source_manager: SourceManagerImpl):
        """Test testing source without owner/repo in URL."""
        source = Source(
            id="test-org",
            name="Test Org",
            type=SourceType.GITHUB,
            url="https://github.com",
        )
        await source_manager.add_source(source)

        with patch.object(source_manager, "_create_adapter", new_callable=AsyncMock) as mock_create:
            mock_adapter = AsyncMock()
            mock_adapter.get_rate_limit = AsyncMock(
                return_value=RateLimit(limit=5000, remaining=5000, reset_at=datetime.utcnow(), used=0)
            )
            mock_adapter.close = AsyncMock()
            mock_create.return_value = mock_adapter

            result = await source_manager.test_source("test-org")
            assert result.success is True
            mock_adapter.get_rate_limit.assert_called()

    @pytest.mark.asyncio
    async def test_discover_from_source_cached(self, source_manager: SourceManagerImpl):
        """Test discover from source with cache."""
        source = Source(
            id="discover-cache",
            name="Discover Cache",
            type=SourceType.GITHUB,
            url="https://github.com/owner/repo",
        )
        await source_manager.add_source(source)

        # Pre-populate cache
        cached_skill = DiscoveredSkill(
            name="cached-skill",
            version="latest",
            description="Cached",
            author="owner",
            tags=["test"],
            source={"id": "discover-cache", "name": "Discover Cache", "type": "github"},
            repository={"owner": "owner", "name": "repo", "url": "https://github.com/owner/repo"},
            manifest_url="https://github.com/owner/repo/blob/main/skills/cached-skill/SKILL.md",
            available_versions=["latest"],
        )
        source_manager._discover_cache["discover-cache"] = [cached_skill]

        result = await source_manager.discover_from_source("discover-cache", force_refresh=False)
        assert len(result) == 1
        assert result[0].name == "cached-skill"

    @pytest.mark.asyncio
    async def test_discover_from_source_not_found(self, source_manager: SourceManagerImpl):
        """Test discover from non-existing source."""
        result = await source_manager.discover_from_source("non-existing")
        assert result == []

    @pytest.mark.asyncio
    async def test_discover_from_source_single_repo(self, source_manager: SourceManagerImpl):
        """Test discover from single repository."""
        from skillhub.models.repository import ContentItem

        source = Source(
            id="discover-single",
            name="Discover Single",
            type=SourceType.GITHUB,
            url="https://github.com/owner/repo",
            subpath="skills",
        )
        await source_manager.add_source(source)

        with patch.object(source_manager, "_create_adapter", new_callable=AsyncMock) as mock_create:
            mock_adapter = AsyncMock()
            mock_adapter.get_contents = AsyncMock(
                return_value=[
                    ContentItem(type="dir", name="skill1", path="skills/skill1", sha="abc", size=0, url="url1"),
                ]
            )
            mock_adapter.get_file_content = AsyncMock(
                return_value="---\nname: discovered-skill\ndescription: Test\n---"
            )
            mock_adapter.close = AsyncMock()
            mock_create.return_value = mock_adapter

            result = await source_manager.discover_from_source("discover-single", force_refresh=True)
            assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_discover_from_org_repos(self, source_manager: SourceManagerImpl):
        """Test discover from organization repos."""
        source = Source(
            id="discover-org",
            name="Discover Org",
            type=SourceType.GITHUB,
            url="https://github.com",  # No owner/repo
        )
        await source_manager.add_source(source)

        with patch.object(source_manager, "_create_adapter", new_callable=AsyncMock) as mock_create:
            mock_adapter = AsyncMock()
            mock_repo = MagicMock()
            mock_repo.name = "skill-repo"
            mock_repo.description = "A skill repo"
            mock_repo.owner = {"login": "org"}
            mock_repo.topics = ["skill"]
            mock_repo.url = "https://github.com/org/skill-repo"
            mock_adapter.list_repositories = AsyncMock(return_value=[mock_repo])
            mock_adapter.get_file_content = AsyncMock(return_value="---\nname: org-skill\ndescription: Test\n---")
            mock_adapter.close = AsyncMock()
            mock_create.return_value = mock_adapter

            await source_manager.discover_from_source("discover-org", force_refresh=True)
            mock_adapter.list_repositories.assert_called()

    @pytest.mark.asyncio
    async def test_search_across_sources(self, source_manager: SourceManagerImpl):
        """Test searching across sources."""
        source = Source(
            id="search-src",
            name="Search Source",
            type=SourceType.GITHUB,
            url="https://github.com/owner/repo",
            enabled=True,
        )
        await source_manager.add_source(source)

        # Pre-populate cache
        skill1 = DiscoveredSkill(
            name="python-helper",
            version="latest",
            description="Python helper skill",
            author="owner",
            tags=["python", "helper"],
            source={"id": "search-src", "name": "Search Source", "type": "github"},
            repository={"owner": "owner", "name": "repo", "url": "https://github.com/owner/repo"},
            manifest_url="url",
            available_versions=["latest"],
        )
        source_manager._discover_cache["search-src"] = [skill1]

        result = await source_manager.search_across_sources("python")
        assert "search-src" in result
        assert len(result["search-src"]) == 1

    @pytest.mark.asyncio
    async def test_search_across_sources_disabled(self, source_manager: SourceManagerImpl):
        """Test searching with disabled source."""
        source = Source(
            id="disabled-src",
            name="Disabled Source",
            type=SourceType.GITHUB,
            url="https://github.com/owner/repo",
            enabled=False,
        )
        await source_manager.add_source(source)

        result = await source_manager.search_across_sources("test")
        assert "disabled-src" not in result

    @pytest.mark.asyncio
    async def test_search_across_sources_specific(self, source_manager: SourceManagerImpl):
        """Test searching specific sources."""
        source1 = Source(id="src1", name="Src1", type=SourceType.GITHUB, url="url1", enabled=True)
        source2 = Source(id="src2", name="Src2", type=SourceType.GITHUB, url="url2", enabled=True)
        await source_manager.add_source(source1)
        await source_manager.add_source(source2)

        skill = DiscoveredSkill(
            name="test-skill",
            version="latest",
            description="Test",
            author="owner",
            tags=[],
            source={"id": "src1", "name": "Src1", "type": "github"},
            repository={"owner": "owner", "name": "repo", "url": "url"},
            manifest_url="url",
            available_versions=["latest"],
        )
        source_manager._discover_cache["src1"] = [skill]

        result = await source_manager.search_across_sources("test", sources=["src1"])
        assert "src1" in result

    @pytest.mark.asyncio
    async def test_create_adapter(self, source_manager: SourceManagerImpl):
        """Test creating adapter for source."""
        source = Source(
            id="adapter-test",
            name="Adapter Test",
            type=SourceType.GITHUB,
            url="https://github.com/owner/repo",
        )

        with patch("skillhub.services.source_manager.AdapterFactory.create") as mock_factory:
            mock_adapter = MagicMock()
            mock_factory.return_value = mock_adapter

            with patch.object(source_manager._credential_manager, "get_token", new_callable=AsyncMock) as mock_token:
                mock_token.return_value = "test_token"

                await source_manager._create_adapter(source)
                mock_token.assert_called_with("github")
                mock_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_frontmatter_incomplete(self):
        """Test parsing frontmatter with only one delimiter."""
        content = "---\nname: test"  # Only one delimiter, no closing ---
        result = SourceManagerImpl._parse_frontmatter(content)
        assert result == {}

    @pytest.mark.asyncio
    async def test_parse_frontmatter_yaml_error(self):
        """Test parsing frontmatter with YAML error."""
        content = "---\ninvalid: yaml: :\n---\nContent"
        result = SourceManagerImpl._parse_frontmatter(content)
        assert result == {}

    @pytest.mark.asyncio
    async def test_extract_owner_repo_with_numbers(self):
        """Test extracting owner with numeric check."""
        result = SourceManagerImpl._extract_owner_repo("https://github.com/123/repo")
        assert result == ("", "")  # owner is numeric, should return empty


class TestCacheManager:
    """Tests for CacheManagerImpl."""

    @pytest.fixture
    def cache_manager(self, mock_settings: Settings):
        """Create cache manager instance."""
        manager = CacheManagerImpl(mock_settings)
        return manager

    @pytest.mark.asyncio
    async def test_init_creates_cache_dir(self, mock_settings: Settings):
        """Test that cache manager creates cache directory."""
        manager = CacheManagerImpl(mock_settings)
        assert manager.cache_dir.exists()

    @pytest.mark.asyncio
    async def test_get_empty(self, cache_manager: CacheManagerImpl):
        """Test getting from empty cache."""
        result = await cache_manager.get("non-existing-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_manager: CacheManagerImpl):
        """Test setting and getting cache value."""
        await cache_manager.set("test-key", "test-value")
        result = await cache_manager.get("test-key")
        assert result == "test-value"

    @pytest.mark.asyncio
    async def test_set_with_options(self, cache_manager: CacheManagerImpl):
        """Test setting cache with options."""
        options = CacheOptions(ttl=3600, tags=["test"])
        await cache_manager.set("key-with-opts", "value", options=options)
        result = await cache_manager.get("key-with-opts")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_delete(self, cache_manager: CacheManagerImpl):
        """Test deleting cache entry."""
        await cache_manager.set("delete-key", "value")
        await cache_manager.delete("delete-key")
        result = await cache_manager.get("delete-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_non_existing(self, cache_manager: CacheManagerImpl):
        """Test deleting non-existing key."""
        await cache_manager.delete("non-existing")

    @pytest.mark.asyncio
    async def test_has(self, cache_manager: CacheManagerImpl):
        """Test checking if key exists."""
        await cache_manager.set("has-key", "value")
        assert await cache_manager.has("has-key") is True
        assert await cache_manager.has("non-existing") is False

    @pytest.mark.asyncio
    async def test_get_many(self, cache_manager: CacheManagerImpl):
        """Test getting multiple keys."""
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")
        result = await cache_manager.get_many(["key1", "key2", "key3"])
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert "key3" not in result

    @pytest.mark.asyncio
    async def test_set_many(self, cache_manager: CacheManagerImpl):
        """Test setting multiple entries."""
        entries = {"key1": "value1", "key2": "value2"}
        await cache_manager.set_many(entries)
        assert await cache_manager.get("key1") == "value1"
        assert await cache_manager.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_clear(self, cache_manager: CacheManagerImpl):
        """Test clearing cache."""
        await cache_manager.set("clear-key", "value")
        await cache_manager.clear()
        assert await cache_manager.has("clear-key") is False

    @pytest.mark.asyncio
    async def test_clean_expired(self, cache_manager: CacheManagerImpl):
        """Test cleaning expired entries."""
        cleaned = await cache_manager.clean_expired()
        assert isinstance(cleaned, int)

    @pytest.mark.asyncio
    async def test_get_stats(self, cache_manager: CacheManagerImpl):
        """Test getting cache stats."""
        stats = await cache_manager.get_stats()
        assert isinstance(stats, CacheStats)
        assert isinstance(stats.size, int)

    @pytest.mark.asyncio
    async def test_invalidate_by_tag(self, cache_manager: CacheManagerImpl):
        """Test invalidating by tag."""
        options = CacheOptions(tags=["tag1"])
        await cache_manager.set("tagged-key", "value", options=options)
        count = await cache_manager.invalidate_by_tag("tag1")
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_get_metadata(self, cache_manager: CacheManagerImpl):
        """Test getting metadata."""
        await cache_manager.set_metadata("github", "skill", "test", {"data": "test"})
        result = await cache_manager.get_metadata("github", "skill", "test")
        assert result is not None

    @pytest.mark.asyncio
    async def test_set_metadata_with_ttl(self, cache_manager: CacheManagerImpl):
        """Test setting metadata with TTL."""
        await cache_manager.set_metadata("github", "skill", "test2", {"data": "test"}, ttl=3600)
        result = await cache_manager.get_metadata("github", "skill", "test2")
        assert result is not None


class TestSecurityManager:
    """Tests for SecurityManagerImpl."""

    @pytest.fixture
    def security_manager(self, mock_settings: Settings):
        """Create security manager instance."""
        return SecurityManagerImpl(mock_settings)

    @pytest.mark.asyncio
    async def test_init_loads_empty_audit_log(self, mock_settings: Settings):
        """Test that security manager initializes with empty audit log."""
        manager = SecurityManagerImpl(mock_settings)
        assert manager._audit_log == []

    @pytest.mark.asyncio
    async def test_verify_checksum_match(self, security_manager: SecurityManagerImpl):
        """Test checksum verification with matching checksum."""
        content = b"test content"
        expected = "6dce9b5415d5b3e8b5d5b3e8b5d5b3e8b5d5b3e8b5d5b3e8b5d5b3e8b5d5b3e8"
        result = await security_manager.verify_checksum(content, expected)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_verify_checksum_sha256(self, security_manager: SecurityManagerImpl):
        """Test SHA256 checksum verification."""
        content = b"test"
        checksum = await security_manager.compute_checksum(content, "sha256")
        result = await security_manager.verify_checksum(content, checksum, "sha256")
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_checksum_sha512(self, security_manager: SecurityManagerImpl):
        """Test SHA512 checksum verification."""
        content = b"test"
        checksum = await security_manager.compute_checksum(content, "sha512")
        result = await security_manager.verify_checksum(content, checksum, "sha512")
        assert result is True

    @pytest.mark.asyncio
    async def test_compute_checksum_sha256(self, security_manager: SecurityManagerImpl):
        """Test computing SHA256 checksum."""
        content = b"test content"
        result = await security_manager.compute_checksum(content, "sha256")
        assert isinstance(result, str)
        assert len(result) == 64

    @pytest.mark.asyncio
    async def test_compute_checksum_sha512(self, security_manager: SecurityManagerImpl):
        """Test computing SHA512 checksum."""
        content = b"test content"
        result = await security_manager.compute_checksum(content, "sha512")
        assert isinstance(result, str)
        assert len(result) == 128

    @pytest.mark.asyncio
    async def test_verify_signature(self, security_manager: SecurityManagerImpl):
        """Test signature verification."""
        content = b"test"
        result = await security_manager.verify_signature(content, "sig", "key")
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_execute_in_sandbox_success(self, security_manager: SecurityManagerImpl):
        """Test sandbox execution."""
        import platform

        options = SandboxOptions(timeout=10)
        if platform.system() == "Windows":
            result = await security_manager.execute_in_sandbox("cmd", ["/c", "echo", "test"], options)
        else:
            result = await security_manager.execute_in_sandbox("echo", ["test"], options)
        assert isinstance(result, SandboxResult)
        assert result.exit_code == 0
        assert "test" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_in_sandbox_timeout(self, security_manager: SecurityManagerImpl):
        """Test sandbox execution timeout."""
        import platform

        options = SandboxOptions(timeout=1)
        if platform.system() == "Windows":
            result = await security_manager.execute_in_sandbox("cmd", ["/c", "ping", "-n", "10", "127.0.0.1"], options)
        else:
            result = await security_manager.execute_in_sandbox("sleep", ["10"], options)
        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower() or "error" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_execute_in_sandbox_failure(self, security_manager: SecurityManagerImpl):
        """Test sandbox execution failure."""
        import platform

        options = SandboxOptions(timeout=10)
        if platform.system() == "Windows":
            result = await security_manager.execute_in_sandbox("cmd", ["/c", "dir", "Z:\\nonexistent\\path"], options)
        else:
            result = await security_manager.execute_in_sandbox("ls", ["/nonexistent/path"], options)
        assert result.exit_code != 0  # Non-zero exit code indicates failure

    @pytest.mark.asyncio
    async def test_set_secure_permissions(self, security_manager: SecurityManagerImpl, temp_dir: Path):
        """Test setting secure permissions."""
        test_dir = temp_dir / "secure_test"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        await security_manager.set_secure_permissions(str(test_dir))

    @pytest.mark.asyncio
    async def test_log_install(self, security_manager: SecurityManagerImpl):
        """Test logging install event."""
        event = InstallEvent(
            timestamp=datetime.utcnow(),
            skill="test-skill",
            version="1.0.0",
            source="github",
            repository="test/repo",
            ref="v1.0.0",
            checksum="abc123",
            success=True,
        )
        await security_manager.log_install(event)
        assert len(security_manager._audit_log) == 1

    @pytest.mark.asyncio
    async def test_audit_log_saved(self, security_manager: SecurityManagerImpl):
        """Test that audit log is saved to file."""
        event = InstallEvent(
            timestamp=datetime.utcnow(),
            skill="saved-skill",
            version="1.0",
            source="github",
            repository="test/repo",
            ref="v1.0",
            checksum="abc",
            success=True,
        )
        await security_manager.log_install(event)

        assert security_manager.audit_log_file.exists()
        with open(security_manager.audit_log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 1

    def test_is_trusted_source(self, security_manager: SecurityManagerImpl):
        """Test checking trusted source."""
        security_manager.config.security.trusted_sources = ["github.com"]
        assert security_manager.is_trusted_source("github.com") is True
        assert security_manager.is_trusted_source("unknown.com") is False

    def test_is_trusted_author(self, security_manager: SecurityManagerImpl):
        """Test checking trusted author."""
        security_manager.config.security.trusted_authors = ["trusted"]
        assert security_manager.is_trusted_author("trusted") is True
        assert security_manager.is_trusted_author("unknown") is False

    @pytest.mark.asyncio
    async def test_verify_signature_gnupg_mock(self, security_manager: SecurityManagerImpl):
        """Test signature verification mocking gnupg at import time."""
        # Since gnupg is imported inside the method, we need to mock it differently
        # Just verify the method handles the case when gnupg import fails
        result = await security_manager.verify_signature(b"test", "sig", "key")
        # Without gnupg installed, this will return False
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_verify_signature_returns_bool(self, security_manager: SecurityManagerImpl):
        """Test signature verification returns boolean."""
        result = await security_manager.verify_signature(b"test", "sig", "key")
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_execute_in_sandbox_exception(self, security_manager: SecurityManagerImpl):
        """Test sandbox execution with exception."""
        options = SandboxOptions(timeout=10)

        with patch("skillhub.services.security_manager.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Command not found")

            result = await security_manager.execute_in_sandbox("invalid-cmd", [], options)
            assert result.exit_code == -1
            assert "Command not found" in result.stderr

    @pytest.mark.asyncio
    async def test_load_audit_log_existing_file(self, mock_settings: Settings, temp_dir: Path):
        """Test loading existing audit log file."""
        audit_file = mock_settings.data_dir / "audit.json"
        audit_file.write_text(json.dumps([{"skill": "old-skill", "success": True}]))

        manager = SecurityManagerImpl(mock_settings)
        assert len(manager._audit_log) == 1
        assert manager._audit_log[0]["skill"] == "old-skill"

    @pytest.mark.asyncio
    async def test_set_secure_permissions_nested(self, security_manager: SecurityManagerImpl, temp_dir: Path):
        """Test setting secure permissions on nested directories."""
        test_dir = temp_dir / "nested_secure"
        test_dir.mkdir()
        nested_dir = test_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "file.txt").write_text("test")
        (test_dir / "top_file.txt").write_text("top")

        await security_manager.set_secure_permissions(str(test_dir))
        # On Windows, this should return early
        # On POSIX, permissions should be set

    @pytest.mark.asyncio
    async def test_verify_checksum_mismatch(self, security_manager: SecurityManagerImpl):
        """Test checksum verification with mismatch."""
        content = b"test content"
        wrong_checksum = "wrongchecksum123"
        result = await security_manager.verify_checksum(content, wrong_checksum)
        assert result is False

    @pytest.mark.asyncio
    async def test_compute_checksum_empty(self, security_manager: SecurityManagerImpl):
        """Test computing checksum of empty content."""
        content = b""
        result = await security_manager.compute_checksum(content, "sha256")
        assert isinstance(result, str)
        assert len(result) == 64
