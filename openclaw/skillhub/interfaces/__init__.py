"""Abstract base classes and interfaces for SkillHub CLI."""

from skillhub.interfaces.adapter import GitPlatformAdapter
from skillhub.interfaces.source_manager import SourceManager
from skillhub.interfaces.skill_resolver import SkillResolver
from skillhub.interfaces.install_engine import InstallEngine
from skillhub.interfaces.cache_manager import CacheManager
from skillhub.interfaces.security_manager import SecurityManager
from skillhub.interfaces.credential_manager import CredentialManager

__all__ = [
    "GitPlatformAdapter",
    "SourceManager",
    "SkillResolver",
    "InstallEngine",
    "CacheManager",
    "SecurityManager",
    "CredentialManager",
]
