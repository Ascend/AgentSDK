"""Git platform adapter interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from skillhub.models.repository import (
    ContentItem,
    RateLimit,
    Release,
    Repository,
    Tag,
)


class GitPlatformAdapter(ABC):
    """Abstract base class for Git platform adapters."""

    def __init__(self, base_url: str, token: Optional[str] = None) -> None:
        """Initialize the adapter.

        Args:
            base_url: Base URL for the API.
            token: Optional authentication token.
        """
        self.base_url = base_url
        self._token = token

    @property
    @abstractmethod
    def platform(self) -> str:
        """Platform name (github, gitee, gitcode)."""
        pass

    @abstractmethod
    async def list_repositories(self, owner: str, **kwargs) -> List[Repository]:
        """List repositories for owner."""
        pass

    @abstractmethod
    async def search_repositories(self, query: str, **kwargs) -> dict:
        """Search repositories."""
        pass

    @abstractmethod
    async def get_repository(self, owner: str, repo: str) -> Repository:
        """Get repository details."""
        pass

    @abstractmethod
    async def get_contents(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None,
    ) -> Union[ContentItem, List[ContentItem]]:
        """Get repository contents."""
        pass

    @abstractmethod
    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None,
    ) -> str:
        """Get file content as string."""
        pass

    @abstractmethod
    async def list_releases(self, owner: str, repo: str) -> List[Release]:
        """List repository releases."""
        pass

    @abstractmethod
    async def get_latest_release(self, owner: str, repo: str) -> Optional[Release]:
        """Get latest release."""
        pass

    @abstractmethod
    async def get_tag(self, owner: str, repo: str, tag: str) -> Tag:
        """Get specific tag."""
        pass

    @abstractmethod
    async def list_tags(
        self,
        owner: str,
        repo: str,
        per_page: int = 30,
        page: int = 1,
    ) -> List[Tag]:
        """List repository tags."""
        pass

    @abstractmethod
    async def download_archive(
        self,
        owner: str,
        repo: str,
        ref: str,
        destination: str,
    ) -> str:
        """Download archive, return path."""
        pass

    @abstractmethod
    async def download_asset(self, asset_url: str, destination: str) -> str:
        """Download release asset, return path."""
        pass

    @abstractmethod
    async def get_rate_limit(self) -> RateLimit:
        """Get current rate limit."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close HTTP client."""
        pass
