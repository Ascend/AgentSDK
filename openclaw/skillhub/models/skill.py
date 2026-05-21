"""Skill-related Pydantic models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SkillManifest(BaseModel):
    name: str
    description: str
    version: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    license: Optional[str] = None
    dependencies: Optional[Dict[str, str]] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or len(v) < 2:
            raise ValueError("Skill name must be at least 2 characters")
        return v.lower().replace(" ", "-")


class InstalledSkill(BaseModel):
    """Record of an installed skill."""

    name: str
    version: str
    source_id: str
    source_type: str
    repository: str
    ref: str  # Git ref (tag/commit)
    install_path: str
    installed_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    checksum: str
    config: Dict[str, Any] = Field(default_factory=dict)


class DiscoveredSkill(BaseModel):
    """Skill discovered from a source."""

    name: str
    version: str
    description: Optional[str]
    author: Optional[str]
    tags: List[str]
    source: Dict[str, str]
    repository: Dict[str, str]
    manifest_url: str
    available_versions: List[str]


class ResolvedSkill(BaseModel):
    """Fully resolved skill with manifest."""

    name: str
    version: str
    repository: str
    ref: str
    manifest: SkillManifest
    source: Dict[str, str]
    download_url: Optional[str]
    subpath: Optional[str] = None  # Subdirectory path within the repository


class DependencyConflict(BaseModel):
    """Dependency conflict info."""

    skill: str
    required_by: List[str]
    versions: List[str]


class DependencyGraph(BaseModel):
    """Skill dependency graph."""

    root: ResolvedSkill
    dependencies: Dict[str, ResolvedSkill]
    conflicts: List[DependencyConflict]
    order: List[str]  # Installation order


class InstallResult(BaseModel):
    """Result of skill installation operation."""

    success: bool
    skill: InstalledSkill
    installed_dependencies: List[str]
    warnings: List[str]
    errors: List[str]
    duration: float
