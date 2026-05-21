"""Cache-related Pydantic models."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class CacheOptions(BaseModel):
    """Cache options."""

    ttl: Optional[int] = None  # seconds
    tags: List[str] = Field(default_factory=list)
    immutable: bool = False


class CacheStats(BaseModel):
    """Cache statistics."""

    size: int
    hit_rate: float
    miss_rate: float
    total_size: int  # bytes
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None
