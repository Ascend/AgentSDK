"""Credential-related Pydantic models."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from skillhub.models.repository import RateLimit


class TokenInfo(BaseModel):
    """Token information."""

    platform: str
    type: str  # 'pat' | 'oauth' | 'app'
    has_token: bool
    expires_at: Optional[datetime] = None
    scopes: List[str] = Field(default_factory=list)


class TokenValidation(BaseModel):
    """Token validation result."""

    valid: bool
    scopes: List[str]
    rate_limit: Optional[RateLimit] = None
    message: Optional[str] = None
