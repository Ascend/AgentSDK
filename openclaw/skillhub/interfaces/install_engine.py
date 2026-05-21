"""Install engine interface."""

from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel
from skillhub.models.skill import InstallResult, InstalledSkill, ResolvedSkill


class InstallOptions(BaseModel):
    """Installation options."""

    force: bool = False
    dry_run: bool = False
    skip_dependencies: bool = False
    target_path: Optional[str] = None
    backup: bool = False


class VerificationIssue(BaseModel):
    """Verification issue."""

    type: str  # 'missing' | 'corrupt' | 'permission' | 'unknown'
    path: str
    message: str


class VerificationResult(BaseModel):
    """Skill verification result."""

    skill: str
    valid: bool
    issues: List[VerificationIssue]
    checksum: dict


class InstallEngine(ABC):
    """Handles skill installation and management."""

    @abstractmethod
    async def install(
        self,
        skill: ResolvedSkill,
        options: Optional[InstallOptions] = None,
    ) -> InstallResult:
        """Install a skill."""
        pass

    @abstractmethod
    async def install_from_path(
        self,
        path: str,
        options: Optional[InstallOptions] = None,
    ) -> InstallResult:
        """Install from local path."""
        pass

    @abstractmethod
    async def uninstall(self, skill_name: str) -> None:
        """Uninstall a skill."""
        pass

    @abstractmethod
    async def upgrade(
        self,
        skill_name: str,
        version: Optional[str] = None,
    ) -> InstallResult:
        """Upgrade a skill."""
        pass

    @abstractmethod
    async def list_installed(self) -> List[InstalledSkill]:
        """List installed skills."""
        pass

    @abstractmethod
    async def get_installed(self, skill_name: str) -> Optional[InstalledSkill]:
        """Get installed skill details."""
        pass

    @abstractmethod
    def is_installed(self, skill_name: str, version: Optional[str] = None) -> bool:
        """Check if skill is installed."""
        pass

    @abstractmethod
    async def verify(self, skill_name: str) -> VerificationResult:
        """Verify installed skill integrity."""
        pass

    @abstractmethod
    async def repair(self, skill_name: str) -> InstallResult:
        """Repair installed skill."""
        pass

    @abstractmethod
    async def clean(self) -> int:
        """Remove orphaned files, return count."""
        pass
