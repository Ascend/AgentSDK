"""Base exception classes for SkillHub CLI."""

from typing import Optional, Dict, Any


class SkillHubError(Exception):
    """Base exception for SkillHub CLI."""

    code: str = "E_UNKNOWN"
    exit_code: int = 1

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.cause = cause
        self.context = context or {}

    def __str__(self) -> str:
        msg = super().__str__()
        if self.cause:
            msg += f" (caused by: {self.cause})"
        return msg


class ValidationError(SkillHubError):
    """Invalid input or configuration."""

    code = "E_VALIDATION"
    exit_code = 1


class NetworkError(SkillHubError):
    """Network connectivity issue."""

    code = "E_NETWORK"
    exit_code = 2


class AuthenticationError(SkillHubError):
    """Authentication or authorization failure."""

    code = "E_AUTH"
    exit_code = 3


class SourceError(SkillHubError):
    """Source operation failed."""

    code = "E_SOURCE"
    exit_code = 4


class ResolutionError(SkillHubError):
    """Could not resolve skill version."""

    code = "E_RESOLUTION"
    exit_code = 5


class InstallError(SkillHubError):
    """Installation failed."""

    code = "E_INSTALL"
    exit_code = 6


class SecurityError(SkillHubError):
    """Security check failed."""

    code = "E_SECURITY"
    exit_code = 7


class NotFoundError(SkillHubError):
    """Resource not found."""

    code = "E_NOT_FOUND"
    exit_code = 8
