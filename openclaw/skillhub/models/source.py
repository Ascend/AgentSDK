"""Source-related Pydantic models."""

import uuid
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Supported Git platform types."""

    GITHUB = "github"
    GITEE = "gitee"
    GITCODE = "gitcode"
    MANIFEST = "manifest"


class AuthType(str, Enum):
    """Authentication types."""

    PAT = "pat"
    OAUTH = "oauth"


class Source(BaseModel):
    """Skill source configuration."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Human-readable name")
    type: SourceType
    url: str = Field(..., description="Base URL or manifest URL")
    auth: Optional[Dict[str, str]] = None
    subpath: Optional[str] = Field(None, description="Subdirectory path containing skills")
    filters: Optional[Dict[str, Any]] = None
    priority: int = Field(default=0, description="Search priority (lower = higher)")
    enabled: bool = True
