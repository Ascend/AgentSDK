"""Cache manager interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from skillhub.models.cache import CacheOptions, CacheStats


class CacheManager(ABC):
    """Content-addressable cache with TTL support."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        options: Optional[CacheOptions] = None,
    ) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete from cache."""
        pass

    @abstractmethod
    async def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        pass

    @abstractmethod
    async def set_many(
        self,
        entries: Dict[str, Any],
        options: Optional[CacheOptions] = None,
    ) -> None:
        """Set multiple values in cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache."""
        pass

    @abstractmethod
    async def clean_expired(self) -> int:
        """Remove expired entries, return count."""
        pass

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass

    @abstractmethod
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate cache entries by tag, return count."""
        pass

    @abstractmethod
    async def get_metadata(
        self,
        provider: str,
        cache_type: str,
        identifier: str,
    ) -> Optional[Any]:
        """Get metadata from cache."""
        pass

    @abstractmethod
    async def set_metadata(
        self,
        provider: str,
        cache_type: str,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set metadata in cache."""
        pass
