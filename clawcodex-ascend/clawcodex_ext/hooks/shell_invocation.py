"""Per-hook shell selection — Chapter 12 round 2.

**Deprecated**: Import from ``clawcodex_ext.utils.shell_resolver`` instead.
This module is kept as a re-export shim for existing consumers.
"""

from __future__ import annotations

from clawcodex_ext.utils.shell_resolver import (
    DEFAULT_HOOK_SHELL,
    SHELL_TYPES,
    ShellType,
    build_powershell_args,
    find_powershell_path,
)

__all__ = [
    "DEFAULT_HOOK_SHELL",
    "SHELL_TYPES",
    "ShellType",
    "build_powershell_args",
    "find_powershell_path",
]
