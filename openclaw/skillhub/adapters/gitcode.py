"""GitCode API adapter."""

import base64
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from skillhub.adapters.base import BaseGitAdapter
from skillhub.models.repository import (
    ContentItem,
    RateLimit,
    Release,
    Repository,
    Tag,
)


class GitCodeAdapter(BaseGitAdapter):
    """GitCode API adapter implementation (API v5, similar to Gitee)."""

    @property
    def platform(self) -> str:
        """Platform name."""
        return "gitcode"

    def _get_headers(self) -> Dict[str, str]:
        """GitCode-specific headers (PRIVATE-TOKEN auth)."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "SkillHub-CLI/1.0",
        }
        if self._token:
            headers["PRIVATE-TOKEN"] = self._token
        return headers

    async def list_repositories(self, owner: str, **kwargs) -> List[Repository]:
        """List repositories for owner."""
        params = {"per_page": kwargs.get("per_page", 100)}
        data = await self._request("GET", f"/users/{owner}/repos", params=params)
        return [self._parse_repository(repo) for repo in data]

    async def search_repositories(self, query: str, **kwargs) -> dict:
        """Search repositories."""
        params = {
            "q": query,
            "per_page": kwargs.get("per_page", 30),
            "page": kwargs.get("page", 1),
        }
        return await self._request("GET", "/search/repositories", params=params)

    async def get_repository(self, owner: str, repo: str) -> Repository:
        """Get repository details."""
        data = await self._request("GET", f"/repos/{owner}/{repo}")
        return self._parse_repository(data)

    async def get_contents(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None,
    ) -> Union[ContentItem, List[ContentItem]]:
        """Get repository contents."""
        params = {"ref": ref} if ref else {}
        data = await self._request("GET", f"/repos/{owner}/{repo}/contents/{path}", params=params)
        if isinstance(data, list):
            return [self._parse_content_item(item) for item in data]
        return self._parse_content_item(data)

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None,
    ) -> str:
        """Get file content as string."""
        content_item = await self.get_contents(owner, repo, path, ref)
        if isinstance(content_item, list):
            raise ValueError("Path is a directory, not a file")
        if content_item.content:
            return base64.b64decode(content_item.content).decode("utf-8")
        return ""

    async def list_releases(self, owner: str, repo: str) -> List[Release]:
        """List repository releases."""
        data = await self._request("GET", f"/repos/{owner}/{repo}/releases")
        return [self._parse_release(release) for release in data]

    async def get_latest_release(self, owner: str, repo: str) -> Optional[Release]:
        """Get latest release."""
        try:
            releases = await self.list_releases(owner, repo)
            for release in releases:
                if not release.prerelease and not release.draft:
                    return release
            return None
        except Exception:
            return None

    async def get_tag(self, owner: str, repo: str, tag: str) -> Tag:
        """Get specific tag."""
        data = await self._request("GET", f"/repos/{owner}/{repo}/tags/{tag}")
        return self._parse_tag(data)

    async def list_tags(self, owner: str, repo: str, per_page: int = 30, page: int = 1) -> List[Tag]:
        """List repository tags."""
        params = {"per_page": per_page, "page": page}
        data = await self._request("GET", f"/repos/{owner}/{repo}/tags", params=params)
        return [self._parse_tag(tag) for tag in data]

    async def download_archive(self, owner: str, repo: str, ref: str, destination: str) -> str:
        """Download archive, return path."""
        url = f"/repos/{owner}/{repo}/zipball/{ref}"
        response = await self._client.get(url, follow_redirects=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as f:
            f.write(response.content)
        return destination

    async def download_asset(self, asset_url: str, destination: str) -> str:
        """Download release asset, return path."""
        response = await self._client.get(asset_url, follow_redirects=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as f:
            f.write(response.content)
        return destination

    async def get_rate_limit(self) -> RateLimit:
        """Get current rate limit (GitCode has no dedicated endpoint)."""
        return RateLimit(
            limit=5000,
            remaining=5000,
            reset_at=datetime.now(),
            used=0,
        )

    def _parse_repository(self, data: dict) -> Repository:
        return Repository(
            id=str(data.get("id", "")),
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            description=data.get("description"),
            owner=data.get("owner", {}),
            url=data.get("html_url", ""),
            clone_url=data.get("clone_url", ""),
            topics=data.get("topics", []),
            is_private=data.get("private", False),
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            language=data.get("language"),
            default_branch=data.get("default_branch", "main"),
        )

    def _parse_content_item(self, data: dict) -> ContentItem:
        return ContentItem(
            type=data.get("type", "file"),
            name=data.get("name", ""),
            path=data.get("path", ""),
            sha=data.get("sha", ""),
            size=data.get("size", 0),
            url=data.get("url", ""),
            content=data.get("content"),
            download_url=data.get("download_url"),
        )

    def _parse_release(self, data: dict) -> Release:
        return Release(
            id=str(data.get("id", "")),
            tag_name=data.get("tag_name", ""),
            name=data.get("name", ""),
            body=data.get("body"),
            prerelease=data.get("prerelease", False),
            draft=data.get("draft", False),
            assets=data.get("assets", []),
            tarball_url=data.get("tarball_url"),
            zipball_url=data.get("zipball_url"),
        )

    def _parse_tag(self, data: dict) -> Tag:
        return Tag(
            name=data.get("name", ""),
            commit=data.get("commit", {}),
            tarball_url=None,
            zipball_url=None,
        )
