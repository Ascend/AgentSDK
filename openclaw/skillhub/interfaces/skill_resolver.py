"""Skill resolver interface."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel
from skillhub.models.skill import (
    DependencyGraph,
    ResolvedSkill,
    SkillManifest,
)


class ValidationResult(BaseModel):
    """Validation result."""

    valid: bool
    errors: List[str]


class SkillResolver(ABC):
    """Resolves skill versions and dependencies."""

    @abstractmethod
    async def resolve_version(
        self,
        skill_name: str,
        version_spec: str,
        sources: Optional[List[str]] = None,
    ) -> Optional[ResolvedSkill]:
        """Resolve a specific skill version."""
        pass

    @abstractmethod
    async def list_available_versions(
        self,
        skill_name: str,
        source: Optional[str] = None,
    ) -> List[str]:
        """List available versions for a skill."""
        pass

    @abstractmethod
    async def resolve_dependencies(
        self,
        skill: ResolvedSkill,
        include_optional: bool = False,
    ) -> DependencyGraph:
        """Resolve dependency graph for a skill."""
        pass

    @abstractmethod
    async def fetch_manifest(
        self,
        repository: str,
        ref: str,
        platform: str,
    ) -> SkillManifest:
        """Fetch skill manifest from repository."""
        pass

    @abstractmethod
    def validate_manifest(self, manifest: Any) -> ValidationResult:
        """Validate skill manifest."""
        pass
