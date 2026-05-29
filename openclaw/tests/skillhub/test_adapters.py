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


from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from skillhub.adapters.base import BaseGitAdapter
from skillhub.adapters.github import GitHubAdapter
from skillhub.adapters.gitee import GiteeAdapter
from skillhub.adapters.gitcode import GitCodeAdapter
from skillhub.adapters.factory import AdapterFactory
from skillhub.models.source import SourceType
from skillhub.models.repository import Repository, Release, Tag, RateLimit


class TestAdapterFactory:
    """Tests for AdapterFactory."""

    def test_create_github_adapter(self):
        """Test creating GitHub adapter."""
        adapter = AdapterFactory.create(SourceType.GITHUB)
        assert isinstance(adapter, GitHubAdapter)

    def test_create_gitee_adapter(self):
        """Test creating Gitee adapter."""
        adapter = AdapterFactory.create(SourceType.GITEE)
        assert isinstance(adapter, GiteeAdapter)

    def test_create_gitcode_adapter(self):
        """Test creating GitCode adapter."""
        adapter = AdapterFactory.create(SourceType.GITCODE)
        assert isinstance(adapter, GitCodeAdapter)

    def test_create_adapter_with_token(self):
        """Test creating adapter with token."""
        adapter = AdapterFactory.create(SourceType.GITHUB, token="test_token")
        assert adapter._token == "test_token"

    def test_create_adapter_invalid_type(self):
        """Test creating adapter with invalid type."""
        with pytest.raises(ValueError):
            AdapterFactory.create("invalid_type")


class TestGitHubAdapter:
    """Tests for GitHubAdapter."""

    def test_github_adapter_base_url(self):
        """Test GitHub adapter base URL."""
        adapter = GitHubAdapter("https://api.github.com")
        assert adapter.base_url == "https://api.github.com"

    def test_github_adapter_inherits_base(self):
        """Test GitHub adapter inherits from BaseGitAdapter."""
        adapter = GitHubAdapter("https://api.github.com")
        assert isinstance(adapter, BaseGitAdapter)

    def test_github_adapter_platform(self):
        """Test GitHub adapter platform property."""
        adapter = GitHubAdapter("https://api.github.com")
        assert adapter.platform == "github"

    def test_adapter_init_with_token(self):
        """Test adapter initialization with token."""
        adapter = GitHubAdapter("https://api.github.com", token="test_token")
        assert adapter._token == "test_token"

    def test_adapter_init_without_token(self):
        """Test adapter initialization without token."""
        adapter = GitHubAdapter("https://api.github.com")
        assert adapter._token is None

    def test_get_headers_with_token(self):
        """Test headers generation with token."""
        adapter = GitHubAdapter("https://api.github.com", token="test_token")
        headers = adapter._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token"

    def test_get_headers_without_token(self):
        """Test headers generation without token."""
        adapter = GitHubAdapter("https://api.github.com")
        headers = adapter._get_headers()
        assert "Authorization" not in headers
        assert "Accept" in headers
        assert "User-Agent" in headers

    def test_adapter_user_agent(self):
        """Test that adapter sets User-Agent."""
        adapter = GitHubAdapter("https://api.github.com")
        headers = adapter._get_headers()
        assert headers["User-Agent"] == "SkillHub-CLI/1.0"

    def test_adapter_accept_header(self):
        """Test that adapter sets Accept header."""
        adapter = GitHubAdapter("https://api.github.com")
        headers = adapter._get_headers()
        assert "Accept" in headers

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing adapter."""
        adapter = GitHubAdapter("https://api.github.com")
        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_rate_limit_mock(self):
        """Test get rate limit with mock."""
        with patch.object(GitHubAdapter, '_request') as mock_request:
            mock_request.return_value = {
                "resources": {
                    "core": {
                        "limit": 5000,
                        "remaining": 4999,
                        "reset": 1234567890,
                    }
                }
            }
            adapter = GitHubAdapter("https://api.github.com")
            result = await adapter.get_rate_limit()
            assert isinstance(result, RateLimit)
            assert result.limit == 5000

    @pytest.mark.asyncio
    async def test_list_repositories_mock(self):
        """Test list repositories with mock."""
        with patch.object(GitHubAdapter, '_request') as mock_request:
            mock_request.return_value = [
                {
                    "id": "repo-1",
                    "name": "repo1",
                    "full_name": "user/repo1",
                    "description": "Test repo",
                    "owner": {"login": "user", "id": 1},
                    "html_url": "https://github.com/user/repo1",
                    "clone_url": "https://github.com/user/repo1.git",
                    "topics": ["test"],
                    "private": False,
                    "stargazers_count": 100,
                    "forks_count": 10,
                    "language": "Python",
                    "default_branch": "main",
                }
            ]
            adapter = GitHubAdapter("https://api.github.com")
            result = await adapter.list_repositories("user")
            assert len(result) == 1
            assert isinstance(result[0], Repository)

    @pytest.mark.asyncio
    async def test_get_repository_mock(self):
        """Test get repository with mock."""
        with patch.object(GitHubAdapter, '_request') as mock_request:
            mock_request.return_value = {
                "id": "repo-1",
                "name": "test-repo",
                "full_name": "user/test-repo",
                "description": "Test",
                "owner": {"login": "user", "id": 1},
                "html_url": "https://github.com/user/test-repo",
                "clone_url": "https://github.com/user/test-repo.git",
                "topics": ["test"],
                "private": False,
                "stargazers_count": 100,
                "forks_count": 10,
                "language": "Python",
                "default_branch": "main",
            }
            adapter = GitHubAdapter("https://api.github.com")
            result = await adapter.get_repository("user", "test-repo")
            assert result.name == "test-repo"

    @pytest.mark.asyncio
    async def test_get_contents_mock(self):
        """Test get contents with mock."""
        with patch.object(GitHubAdapter, '_request') as mock_request:
            mock_request.return_value = [
                {
                    "type": "file",
                    "name": "file.txt",
                    "path": "file.txt",
                    "sha": "abc123",
                    "size": 100,
                    "url": "https://example.com/file",
                }
            ]
            adapter = GitHubAdapter("https://api.github.com")
            result = await adapter.get_contents("user", "repo", "")
            assert isinstance(result, list)
            assert result[0].name == "file.txt"

    @pytest.mark.asyncio
    async def test_get_file_content_mock(self):
        """Test get file content with mock."""
        with patch.object(GitHubAdapter, '_request') as mock_request:
            mock_request.return_value = {
                "type": "file",
                "name": "SKILL.md",
                "path": "SKILL.md",
                "sha": "abc123",
                "content": "dGVzdCBjb250ZW50",  # base64 encoded "test content"
                "encoding": "base64",
                "size": 12,
                "url": "https://example.com/file",
            }
            adapter = GitHubAdapter("https://api.github.com")
            result = await adapter.get_file_content("user", "repo", "SKILL.md")
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_list_releases_mock(self):
        """Test list releases with mock."""
        with patch.object(GitHubAdapter, '_request') as mock_request:
            mock_request.return_value = [
                {
                    "id": "release-1",
                    "tag_name": "v1.0.0",
                    "name": "Release 1.0.0",
                    "body": "Initial release",
                    "draft": False,
                    "prerelease": False,
                    "assets": [],
                    "tarball_url": "https://example.com/tarball",
                    "zipball_url": "https://example.com/zipball",
                }
            ]
            adapter = GitHubAdapter("https://api.github.com")
            result = await adapter.list_releases("user", "repo")
            assert len(result) == 1
            assert isinstance(result[0], Release)

    @pytest.mark.asyncio
    async def test_get_latest_release_mock(self):
        """Test get latest release with mock."""
        with patch.object(GitHubAdapter, '_request') as mock_request:
            mock_request.return_value = {
                "id": "release-1",
                "tag_name": "v1.0.0",
                "name": "Release 1.0.0",
                "body": "",
                "draft": False,
                "prerelease": False,
                "assets": [],
                "tarball_url": "https://example.com/tarball",
                "zipball_url": "https://example.com/zipball",
            }
            adapter = GitHubAdapter("https://api.github.com")
            result = await adapter.get_latest_release("user", "repo")
            assert result is not None
            assert result.tag_name == "v1.0.0"

    @pytest.mark.asyncio
    async def test_list_tags_mock(self):
        """Test list tags with mock."""
        with patch.object(GitHubAdapter, '_request') as mock_request:
            mock_request.return_value = [
                {
                    "name": "v1.0.0",
                    "commit": {"sha": "abc123"},
                }
            ]
            adapter = GitHubAdapter("https://api.github.com")
            result = await adapter.list_tags("user", "repo")
            assert len(result) == 1
            assert isinstance(result[0], Tag)


class TestGiteeAdapter:
    """Tests for GiteeAdapter."""

    def test_gitee_adapter_base_url(self):
        """Test Gitee adapter base URL."""
        adapter = GiteeAdapter("https://gitee.com/api/v5")
        assert adapter.base_url == "https://gitee.com/api/v5"

    def test_gitee_adapter_inherits_base(self):
        """Test Gitee adapter inherits from BaseGitAdapter."""
        adapter = GiteeAdapter("https://gitee.com/api/v5")
        assert isinstance(adapter, BaseGitAdapter)

    def test_gitee_adapter_platform(self):
        """Test Gitee adapter platform property."""
        adapter = GiteeAdapter("https://gitee.com/api/v5")
        assert adapter.platform == "gitee"

    @pytest.mark.asyncio
    async def test_get_rate_limit_mock(self):
        """Test Gitee rate limit with mock."""
        adapter = GiteeAdapter("https://gitee.com/api/v5")
        result = await adapter.get_rate_limit()
        assert result.limit == 1000  # Gitee default


class TestGitCodeAdapter:
    """Tests for GitCodeAdapter."""

    def test_gitcode_adapter_base_url(self):
        """Test GitCode adapter base URL."""
        adapter = GitCodeAdapter("https://api.gitcode.com/api/v5")
        assert adapter.base_url == "https://api.gitcode.com/api/v5"

    def test_gitcode_adapter_inherits_base(self):
        """Test GitCode adapter inherits from BaseGitAdapter."""
        adapter = GitCodeAdapter("https://api.gitcode.com/api/v5")
        assert isinstance(adapter, BaseGitAdapter)

    def test_gitcode_adapter_platform(self):
        """Test GitCode adapter platform property."""
        adapter = GitCodeAdapter("https://api.gitcode.com/api/v5")
        assert adapter.platform == "gitcode"

    @pytest.mark.asyncio
    async def test_get_rate_limit_mock(self):
        """Test GitCode rate limit with mock."""
        with patch.object(GitCodeAdapter, '_request') as mock_request:
            mock_request.return_value = {
                "limit": 5000,
                "remaining": 4999,
                "reset_time": "2024-01-01T00:00:00Z",
            }
            adapter = GitCodeAdapter("https://api.gitcode.com/api/v5")
            result = await adapter.get_rate_limit()
            assert result.limit == 5000


class TestAdapterCommon:
    """Tests common to all adapters."""

    @pytest.mark.asyncio
    async def test_adapter_close_cleanup(self):
        """Test that all adapters can be closed."""
        adapters = [
            GitHubAdapter("https://api.github.com"),
            GiteeAdapter("https://gitee.com/api/v5"),
            GitCodeAdapter("https://api.gitcode.com/api/v5"),
        ]
        for adapter in adapters:
            await adapter.close()

    def test_adapter_with_token_header(self):
        """Test that adapters use correct token headers."""
        github = GitHubAdapter("https://api.github.com", token="github_token")
        gitee = GiteeAdapter("https://gitee.com/api/v5", token="gitee_token")
        gitcode = GitCodeAdapter("https://api.gitcode.com/api/v5", token="gitcode_token")

        # GitHub uses Authorization: Bearer
        gh_headers = github._get_headers()
        assert "Authorization" in gh_headers

        # Gitee uses Authorization: token
        ge_headers = gitee._get_headers()
        assert "Authorization" in ge_headers

        # GitCode uses PRIVATE-TOKEN
        gc_headers = gitcode._get_headers()
        assert "PRIVATE-TOKEN" in gc_headers


class TestAdapterErrorHandling:
    """Tests for adapter error handling."""

    @pytest.mark.asyncio
    async def test_request_http_error(self):
        """Test HTTP error handling."""
        import httpx

        with patch("skillhub.adapters.base.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Error",
                request=MagicMock(),
                response=MagicMock(status_code=404),
            )
            mock_instance.request.return_value = mock_response
            mock_client.return_value = mock_instance

            adapter = GitHubAdapter("https://api.github.com")
            with pytest.raises(httpx.HTTPStatusError):
                await adapter._request("GET", "/test")

    @pytest.mark.asyncio
    async def test_request_timeout(self):
        """Test timeout handling."""
        import httpx

        with patch("skillhub.adapters.base.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.request.side_effect = httpx.TimeoutException("Timeout")
            mock_client.return_value = mock_instance

            adapter = GitHubAdapter("https://api.github.com")
            with pytest.raises(httpx.TimeoutException):
                await adapter._request("GET", "/test")

    @pytest.mark.asyncio
    async def test_request_network_error(self):
        """Test network error handling."""
        import httpx

        with patch("skillhub.adapters.base.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.request.side_effect = httpx.ConnectError("Connection failed")
            mock_client.return_value = mock_instance

            adapter = GitHubAdapter("https://api.github.com")
            with pytest.raises(httpx.ConnectError):
                await adapter._request("GET", "/test")


class TestAdapterParsing:
    """Tests for adapter parsing methods."""

    def test_github_parse_repository_full(self):
        """Test GitHub repository parsing with full data."""
        adapter = GitHubAdapter("https://github.com")

        data = {
            "id": 12345,
            "name": "skillhub",
            "full_name": "owner/skillhub",
            "description": "Skill management CLI",
            "owner": {"login": "owner", "id": 1},
            "html_url": "https://github.com/owner/skillhub",
            "clone_url": "https://github.com/owner/skillhub.git",
            "topics": ["cli", "skills"],
            "private": False,
            "stargazers_count": 500,
            "forks_count": 50,
            "language": "Python",
            "default_branch": "main",
        }

        repo = adapter._parse_repository(data)
        assert repo.id == "12345"
        assert repo.name == "skillhub"
        assert repo.full_name == "owner/skillhub"
        assert repo.description == "Skill management CLI"
        assert repo.stars == 500
        assert repo.forks == 50
        assert repo.language == "Python"
        assert repo.default_branch == "main"
        assert repo.is_private is False

    def test_github_parse_repository_minimal(self):
        """Test GitHub repository parsing with minimal data."""
        adapter = GitHubAdapter("https://github.com")

        data = {"id": 1, "name": "test"}
        repo = adapter._parse_repository(data)
        assert repo.id == "1"
        assert repo.name == "test"
        assert repo.default_branch == "main"  # default

    def test_github_parse_tag(self):
        """Test GitHub tag parsing."""
        adapter = GitHubAdapter("https://github.com")

        data = {
            "name": "v2.0.0",
            "commit": {"sha": "def456abc789"},
        }

        tag = adapter._parse_tag(data)
        assert tag.name == "v2.0.0"
        assert tag.commit == {"sha": "def456abc789"}

    def test_github_parse_content_item_file(self):
        """Test GitHub content item parsing for file."""
        adapter = GitHubAdapter("https://github.com")

        data = {
            "type": "file",
            "name": "README.md",
            "path": "README.md",
            "sha": "abc123",
            "size": 2048,
            "url": "https://api.github.com/repos/owner/repo/contents/README.md",
            "download_url": "https://raw.githubusercontent.com/owner/repo/main/README.md",
        }

        item = adapter._parse_content_item(data)
        assert item.type == "file"
        assert item.name == "README.md"
        assert item.path == "README.md"
        assert item.size == 2048

    def test_github_parse_content_item_dir(self):
        """Test GitHub content item parsing for directory."""
        adapter = GitHubAdapter("https://github.com")

        data = {
            "type": "dir",
            "name": "src",
            "path": "src",
            "sha": "dir123",
        }

        item = adapter._parse_content_item(data)
        assert item.type == "dir"
        assert item.name == "src"

    def test_github_parse_release(self):
        """Test GitHub release parsing."""
        adapter = GitHubAdapter("https://github.com")

        data = {
            "id": 999,
            "tag_name": "v1.0.0",
            "name": "First Release",
            "body": "Initial release notes",
            "draft": False,
            "prerelease": True,
            "created_at": "2024-01-01T00:00:00Z",
            "published_at": "2024-01-02T00:00:00Z",
            "tarball_url": "https://github.com/owner/repo/tarball/v1.0.0",
            "zipball_url": "https://github.com/owner/repo/zipball/v1.0.0",
            "assets": [],
        }

        release = adapter._parse_release(data)
        assert release.id == "999"
        assert release.tag_name == "v1.0.0"
        assert release.name == "First Release"
        assert release.prerelease is True
        assert release.draft is False

    def test_gitee_parse_repository(self):
        """Test Gitee repository parsing."""
        adapter = GiteeAdapter("https://gitee.com")

        data = {
            "id": 456,
            "name": "skillhub",
            "full_name": "owner/skillhub",
            "description": "Skill management",
            "owner": {"login": "owner"},
            "html_url": "https://gitee.com/owner/skillhub",
            "clone_url": "https://gitee.com/owner/skillhub.git",
            "topics": ["cli"],
            "private": False,
            "stargazers_count": 100,
            "forks_count": 10,
            "language": "Python",
            "default_branch": "master",
        }

        repo = adapter._parse_repository(data)
        assert repo.name == "skillhub"
        assert repo.url == "https://gitee.com/owner/skillhub"
        assert repo.default_branch == "master"

    def test_gitcode_parse_repository(self):
        """Test GitCode repository parsing."""
        adapter = GitCodeAdapter("https://gitcode.com")

        data = {
            "id": 789,
            "name": "skillhub",
            "full_name": "owner/skillhub",
            "description": "Skill management",
            "owner": {"login": "owner"},
            "html_url": "https://gitcode.com/owner/skillhub",
            "clone_url": "https://gitcode.com/owner/skillhub.git",
            "private": False,
            "stargazers_count": 200,
            "forks_count": 20,
            "language": "Python",
            "default_branch": "main",
        }

        repo = adapter._parse_repository(data)
        assert repo.name == "skillhub"
        assert repo.url == "https://gitcode.com/owner/skillhub"
        assert repo.stars == 200
