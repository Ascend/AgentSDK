"""Cache manager implementation."""

import os
from typing import Any, Dict, List, Optional

import diskcache

from skillhub.config import Settings
from skillhub.interfaces.cache_manager import CacheManager
from skillhub.models.cache import CacheOptions, CacheStats


class CacheManagerImpl(CacheManager):
    def __init__(self, config: Settings):
        self.config = config
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(self.cache_dir))

    async def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        options: Optional[CacheOptions] = None,
    ) -> None:
        ttl = options.ttl if options else None
        tags = options.tags if options else []
        self._cache.set(key, value, expire=ttl, tag=",".join(tags))

    async def delete(self, key: str) -> None:
        self._cache.delete(key)

    async def has(self, key: str) -> bool:
        return key in self._cache

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        result = {}
        for key in keys:
            value = self._cache.get(key)
            if value is not None:
                result[key] = value
        return result

    async def set_many(
        self,
        entries: Dict[str, Any],
        options: Optional[CacheOptions] = None,
    ) -> None:
        ttl = options.ttl if options else None
        tags = options.tags if options else []
        for key, value in entries.items():
            self._cache.set(key, value, expire=ttl, tag=",".join(tags))

    async def clear(self) -> None:
        self._cache.clear()

    async def clean_expired(self) -> int:
        return self._cache.expire()

    async def get_stats(self) -> CacheStats:
        hits, misses = self._cache.stats()
        total_requests = int(hits) + int(misses)
        return CacheStats(
            size=len(self._cache),
            hit_rate=int(hits) / max(total_requests, 1),
            miss_rate=int(misses) / max(total_requests, 1),
            total_size=sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in os.listdir(self.cache_dir)
                if os.path.isfile(os.path.join(self.cache_dir, f))
            ),
            oldest_entry=None,
            newest_entry=None,
        )

    async def invalidate_by_tag(self, tag: str) -> int:
        return self._cache.evict(tag=tag)

    async def get_metadata(
        self,
        provider: str,
        cache_type: str,
        identifier: str,
    ) -> Optional[Any]:
        key = f"metadata:{provider}:{cache_type}:{identifier}"
        return self._cache.get(key)

    async def set_metadata(
        self,
        provider: str,
        cache_type: str,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        key = f"metadata:{provider}:{cache_type}:{identifier}"
        self._cache.set(key, value, expire=ttl)
