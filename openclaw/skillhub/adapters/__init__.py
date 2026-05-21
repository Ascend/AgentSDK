"""Platform adapters for SkillHub CLI."""

from skillhub.adapters.base import BaseGitAdapter
from skillhub.adapters.github import GitHubAdapter
from skillhub.adapters.gitee import GiteeAdapter
from skillhub.adapters.gitcode import GitCodeAdapter
from skillhub.adapters.factory import AdapterFactory

__all__ = [
    "BaseGitAdapter",
    "GitHubAdapter",
    "GiteeAdapter",
    "GitCodeAdapter",
    "AdapterFactory",
]
