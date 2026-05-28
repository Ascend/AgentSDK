"""SkillHub CLI - Decentralized skill management."""

__version__ = "0.1.0"

# Import submodules for proper package structure
from skillhub import config
from skillhub import models
from skillhub import adapters
from skillhub import services
from skillhub import interfaces
from skillhub import exceptions
from skillhub import utils

__all__ = [
    "__version__",
    "config",
    "models",
    "adapters",
    "services",
    "interfaces",
    "exceptions",
    "utils",
]
