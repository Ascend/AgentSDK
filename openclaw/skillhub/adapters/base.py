"""Base Git adapter implementation."""

from typing import Any, Dict, List, Optional, Union
import httpx

from skillhub.interfaces.adapter import GitPlatformAdapter
from skillhub.models.repository import (
    ContentItem,
    RateLimit,
    Release,
    Repository,
    Tag,
)


class BaseGitAdapter(GitPlatformAdapter):
    """Base implementation for Git platform adapters."""

    def __init__(self, base_url: str, token: Optional[str] = None) -> None:
        """Initialize the adapter.

        Args:
            base_url: Base URL for the API.
            token: Optional authentication token.
        """
        super().__init__(base_url, token)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=self._get_headers(),
            http2=True,
            timeout=30.0,
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get default HTTP headers.

        Returns:
            Dictionary of headers.
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "SkillHub-CLI/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Any:
        """Make an HTTP request.

        Args:
            method: HTTP method.
            path: Request path.
            **kwargs: Additional arguments for the request.

        Returns:
            Response data.
        """
        response = await self._client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def list_repositories(self, owner: str, **kwargs) -> List[Repository]:
        """List repositories for owner."""
        raise NotImplementedError

    async def search_repositories(self, query: str, **kwargs) -> dict:
        """Search repositories."""
        raise NotImplementedError

    async def get_repository(self, owner: str, repo: str) -> Repository:
        """Get repository details."""
        raise NotImplementedError

    async def get_contents(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None,
    ) -> Union[ContentItem, List[ContentItem]]:
        """Get repository contents."""
        raise NotImplementedError

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None,
    ) -> str:
        """Get file content as string."""
        raise NotImplementedError

    async def list_releases(self, owner: str, repo: str) -> List[Release]:
        """List repository releases."""
        raise NotImplementedError

    async def get_latest_release(self, owner: str, repo: str) -> Optional[Release]:
        """Get latest release."""
        raise NotImplementedError

    async def get_tag(self, owner: str, repo: str, tag: str) -> Tag:
        """Get specific tag."""
        raise NotImplementedError

    async def list_tags(
        self,
        owner: str,
        repo: str,
        per_page: int = 30,
        page: int = 1,
    ) -> List[Tag]:
        """List repository tags."""
        raise NotImplementedError

    async def download_archive(
        self,
        owner: str,
        repo: str,
        ref: str,
        destination: str,
    ) -> str:
        """Download archive, return path."""
        raise NotImplementedError

    async def download_asset(self, asset_url: str, destination: str) -> str:
        """Download release asset, return path."""
        raise NotImplementedError

    async def get_rate_limit(self) -> RateLimit:
        """Get current rate limit."""
        raise NotImplementedError
