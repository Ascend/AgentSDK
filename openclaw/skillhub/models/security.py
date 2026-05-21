"""Security-related Pydantic models."""

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class SandboxOptions(BaseModel):
    """Sandbox execution options."""

    read_only_dirs: List[str] = Field(default_factory=list)
    write_dirs: List[str] = Field(default_factory=list)
    network: bool = False
    timeout: int = 300  # seconds
    memory_limit: int = 512  # MB


class SandboxResult(BaseModel):
    """Sandbox execution result."""

    exit_code: int
    stdout: str
    stderr: str
    duration: float


class InstallEvent(BaseModel):
    """Installation audit event."""

    timestamp: datetime
    skill: str
    version: str
    source: str
    repository: str
    ref: str
    checksum: str
    success: bool
