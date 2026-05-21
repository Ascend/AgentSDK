"""Source manager interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from skillhub.models.repository import RateLimit
from skillhub.models.skill import DiscoveredSkill
from skillhub.models.source import Source


class SourceTestResult(BaseModel):
    """Source connectivity test result."""

    success: bool
    message: str
    rate_limit: Optional[RateLimit] = None
    latency: Optional[float] = None


class SourceManager(ABC):
    """Manages skill sources (registries)."""

    @abstractmethod
    async def add_source(self, source: Source) -> Source:
        """Add a new source."""
        pass

    @abstractmethod
    async def remove_source(self, source_id: str) -> None:
        """Remove a source."""
        pass

    @abstractmethod
    async def update_source(
        self,
        source_id: str,
        updates: Dict[str, Any],
    ) -> Source:
        """Update source configuration."""
        pass

    @abstractmethod
    async def get_source(self, source_id: str) -> Optional[Source]:
        """Get source by ID."""
        pass

    @abstractmethod
    async def list_sources(self) -> List[Source]:
        """List all sources."""
        pass

    @abstractmethod
    async def enable_source(self, source_id: str) -> None:
        """Enable a source."""
        pass

    @abstractmethod
    async def disable_source(self, source_id: str) -> None:
        """Disable a source."""
        pass

    @abstractmethod
    async def test_source(self, source_id: str) -> SourceTestResult:
        """Test source connectivity."""
        pass

    @abstractmethod
    async def discover_from_source(
        self,
        source_id: str,
        force_refresh: bool = False,
    ) -> List[DiscoveredSkill]:
        """Discover skills from a source."""
        pass

    @abstractmethod
    async def search_across_sources(
        self,
        query: str,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, List[DiscoveredSkill]]:
        """Search across all sources."""
        pass
