"""Credential manager interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from skillhub.models.credential import TokenInfo, TokenValidation


class CredentialManager(ABC):
    """Manages authentication credentials."""

    @abstractmethod
    async def store_token(
        self,
        platform: str,
        token: str,
        token_type: str = "pat",
        expires_at: Optional[datetime] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        """Store authentication token."""
        pass

    @abstractmethod
    async def get_token(self, platform: str) -> Optional[str]:
        """Get stored token."""
        pass

    @abstractmethod
    async def remove_token(self, platform: str) -> None:
        """Remove stored token."""
        pass

    @abstractmethod
    async def list_tokens(self) -> List[TokenInfo]:
        """List stored tokens."""
        pass

    @abstractmethod
    async def validate_token(self, platform: str, token: str) -> TokenValidation:
        """Validate token."""
        pass

    @abstractmethod
    def is_secure_storage_available(self) -> bool:
        """Check if secure storage is available."""
        pass
