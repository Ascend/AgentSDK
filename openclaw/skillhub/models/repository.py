"""Repository-related Pydantic models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Repository(BaseModel):
    """Repository information."""

    id: str
    name: str
    full_name: str
    description: Optional[str]
    owner: Dict[str, Any]
    url: str
    clone_url: str
    topics: List[str]
    is_private: bool
    stars: int
    forks: int
    language: Optional[str]
    default_branch: str


class Release(BaseModel):
    """Release information."""

    id: str
    tag_name: str
    name: str
    body: Optional[str]
    prerelease: bool
    draft: bool
    assets: List[Dict[str, Any]]
    tarball_url: Optional[str]
    zipball_url: Optional[str]


class ContentItem(BaseModel):
    """Repository content item."""

    type: str  # 'file' | 'dir' | 'symlink' | 'submodule'
    name: str
    path: str
    sha: str
    size: int
    url: str
    content: Optional[str] = None
    download_url: Optional[str] = None


class Tag(BaseModel):
    """Git tag."""

    name: str
    commit: Dict[str, str]
    tarball_url: Optional[str] = None
    zipball_url: Optional[str] = None


class RateLimit(BaseModel):
    """API rate limit info."""

    limit: int
    remaining: int
    reset_at: datetime
    used: int
