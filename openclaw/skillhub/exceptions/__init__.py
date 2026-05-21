"""Custom exceptions for SkillHub CLI."""

from skillhub.exceptions.base import (
    SkillHubError,
    ValidationError,
    NetworkError,
    AuthenticationError,
    SourceError,
    ResolutionError,
    InstallError,
    SecurityError,
    NotFoundError,
)

__all__ = [
    "SkillHubError",
    "ValidationError",
    "NetworkError",
    "AuthenticationError",
    "SourceError",
    "ResolutionError",
    "InstallError",
    "SecurityError",
    "NotFoundError",
]
