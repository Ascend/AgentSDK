"""Adapter factory for creating platform-specific adapters."""

from typing import Optional

from skillhub.adapters.github import GitHubAdapter
from skillhub.adapters.gitee import GiteeAdapter
from skillhub.adapters.gitcode import GitCodeAdapter
from skillhub.interfaces.adapter import GitPlatformAdapter
from skillhub.models.source import SourceType


class AdapterFactory:
    """Factory for creating Git platform adapters."""

    # Default API base URLs for each platform
    DEFAULT_BASE_URLS = {
        SourceType.GITHUB: "https://api.github.com",
        SourceType.GITEE: "https://gitee.com/api/v5",
        SourceType.GITCODE: "https://api.gitcode.com/api/v5",
    }

    @staticmethod
    def create(
        source_type: SourceType,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ) -> GitPlatformAdapter:
        """Create an adapter for the given source type.

        Args:
            source_type: Type of the source (github, gitee, gitcode).
            base_url: Optional base URL for the API. If provided, it should be the API base URL.
                      For web URLs like https://github.com/user/repo, use None to get the default API URL.
            token: Optional authentication token.

        Returns:
            Configured adapter instance.
        """
        # Always use the default API base URL for each platform
        # The source URL (web URL) should not be used as API base URL
        api_base_url = AdapterFactory.DEFAULT_BASE_URLS.get(source_type)
        if not api_base_url:
            raise ValueError(f"Unsupported source type: {source_type}")

        if source_type == SourceType.GITHUB:
            return GitHubAdapter(api_base_url, token=token)
        elif source_type == SourceType.GITEE:
            return GiteeAdapter(api_base_url, token=token)
        elif source_type == SourceType.GITCODE:
            return GitCodeAdapter(api_base_url, token=token)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
