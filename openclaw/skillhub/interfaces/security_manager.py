"""Security manager interface."""

from abc import ABC, abstractmethod
from typing import List

from skillhub.models.security import (
    InstallEvent,
    SandboxOptions,
    SandboxResult,
)


class SecurityManager(ABC):
    """Handles security operations."""

    @abstractmethod
    async def verify_signature(
        self,
        content: bytes,
        signature: str,
        public_key: str,
    ) -> bool:
        """Verify content signature."""
        pass

    @abstractmethod
    async def verify_checksum(
        self,
        content: bytes,
        expected_checksum: str,
        algorithm: str = "sha256",
    ) -> bool:
        """Verify content checksum."""
        pass

    @abstractmethod
    async def compute_checksum(
        self,
        content: bytes,
        algorithm: str = "sha256",
    ) -> str:
        """Compute content checksum."""
        pass

    @abstractmethod
    async def execute_in_sandbox(
        self,
        command: str,
        args: List[str],
        options: SandboxOptions,
    ) -> SandboxResult:
        """Execute command in sandbox."""
        pass

    @abstractmethod
    async def set_secure_permissions(self, path: str) -> None:
        """Set secure file permissions."""
        pass

    @abstractmethod
    async def log_install(self, event: InstallEvent) -> None:
        """Log installation event."""
        pass

    @abstractmethod
    def is_trusted_source(self, source: str) -> bool:
        """Check if source is trusted."""
        pass

    @abstractmethod
    def is_trusted_author(self, author: str) -> bool:
        """Check if author is trusted."""
        pass
