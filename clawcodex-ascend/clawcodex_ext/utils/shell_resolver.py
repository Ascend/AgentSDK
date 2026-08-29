#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Unified shell-resolution infrastructure for hooks and BashTool.

Centralises shell-detection logic that was previously duplicated between
``clawcodex_ext/hooks/shell_invocation.py`` (for command hooks) and the
BashTool's inline ``["bash", "-lc", ...]`` construction.

Every consumer should import from here, not from ``shell_invocation.py``
(deprecated re-export shim).

Exports
-------
SHELL_TYPES
    Ordered tuple of recognised shell type names (``"bash"``, ``"powershell"``).
ShellType
    Literal type alias for the valid strings.
DEFAULT_HOOK_SHELL
    The historical default -- ``"bash"``.
build_powershell_args
    Build ``pwsh``-compatible argv (``-NoProfile -NonInteractive -Command <cmd>``).
find_powershell_path
    Locate ``pwsh`` or ``powershell`` on ``PATH``.
resolve_shell
    Unified entry point: normalise a user-supplied ``shell`` parameter into a
    ``(shell_kind, argv_factory)`` pair.
build_shell_argv
    Build the full subprocess argv for any supported shell.
"""

from __future__ import annotations

import shutil
import sys as _sys
from typing import Callable, Literal

# Ordered tuple, not frozenset — preserves iteration order so error messages
# list valid choices "bash" first.
SHELL_TYPES: tuple[str, ...] = ("bash", "powershell")
ShellType = Literal["bash", "powershell"]

DEFAULT_HOOK_SHELL: ShellType = "bash"


# ---------------------------------------------------------------------------
# PowerShell argument builders (extracted from shell_invocation.py)
# ---------------------------------------------------------------------------


def build_powershell_args(cmd: str) -> list[str]:
    """Return the argv for spawning ``pwsh`` with *cmd*.

    Flags mirror ``typescript/src/utils/shell/powershellProvider.ts:11-13``:

    * ``-NoProfile``      — skip user profile scripts (faster, deterministic).
    * ``-NonInteractive`` — fail fast instead of prompting for input.
    * ``-Command``        — execute the literal string that follows.
    """
    encoding_prelude = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $OutputEncoding = [System.Text.Encoding]::UTF8; "
    )
    return ["-NoProfile", "-NonInteractive", "-Command", f"{encoding_prelude}{cmd}"]


def find_powershell_path() -> str | None:
    """Locate PowerShell on ``PATH``.

    Preference order:
    1. ``pwsh`` — cross-platform PowerShell 6+ (macOS / Linux / Windows).
    2. ``powershell`` — Windows PowerShell 5.1 (Windows-only).

    Returns ``None`` if neither is found.
    """
    return shutil.which("pwsh") or shutil.which("powershell")


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def _auto_detect_shell() -> ShellType:
    """Return ``"powershell"`` on Windows when pwsh is available, else ``"bash"``."""
    if _sys.platform == "win32" and find_powershell_path() is not None:
        return "powershell"
    return "bash"


# ---------------------------------------------------------------------------
# Shell argv factories
# ---------------------------------------------------------------------------

ShellArgvFactory = Callable[[str], list[str]]


def _bash_argv(cmd: str) -> list[str]:
    return ["bash", "-lc", cmd]


def _powershell_argv(cmd: str) -> list[str]:
    pwsh = find_powershell_path()
    if pwsh is None:
        raise RuntimeError(
            "PowerShell requested but 'pwsh'/'powershell' not found on PATH. "
            "Install PowerShell 7+ (https://github.com/PowerShell/PowerShell) "
            'or use shell="bash".'
        )
    return [pwsh, *build_powershell_args(cmd)]


_SHELL_ARGV_FACTORIES: dict[str, ShellArgvFactory] = {
    "bash": _bash_argv,
    "powershell": _powershell_argv,
}


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def resolve_shell(
    shell_param: str | None,
    *,
    platform: str | None = None,
) -> tuple[str, ShellArgvFactory]:
    """Normalise a user-supplied ``shell`` parameter.

    Parameters
    ----------
    shell_param
        One of ``"bash"``, ``"powershell"``, ``"auto"``, or ``None``.
        ``None`` and ``"auto"`` trigger platform detection.
    platform
        Override ``sys.platform`` (for testing).  Defaults to
        ``sys.platform``.

    Returns
    -------
    (shell_kind, argv_factory)
        A normalised shell type and its argv-building callable.
    """
    if shell_param is None or shell_param == "auto":
        resolved = (
            _auto_detect_shell()
            if platform is None
            else ("powershell" if platform == "win32" and find_powershell_path() is not None else "bash")
        )
    elif shell_param in _SHELL_ARGV_FACTORIES:
        resolved = shell_param
        # When "powershell" is explicitly requested but pwsh is not available,
        # fall back to bash with a warning rather than crashing.
        if resolved == "powershell" and find_powershell_path() is None:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "shell='powershell' requested but pwsh not found on PATH. Falling back to bash."
            )
            resolved = "bash"
    else:
        resolved = _auto_detect_shell() if platform is None else "bash"

    return resolved, _SHELL_ARGV_FACTORIES[resolved]


# ---------------------------------------------------------------------------
# Convenience: build the full subprocess argv list
# ---------------------------------------------------------------------------


def build_shell_argv(
    shell: str,
    wrapped_command: str,
    *,
    platform: str | None = None,
) -> tuple[str, list[str]]:
    """Return ``(shell_kind, argv_list)`` for a fully resolved shell."""
    kind, factory = resolve_shell(shell, platform=platform)
    return kind, factory(wrapped_command)


# ---------------------------------------------------------------------------
# CWD-tracking wrapper builders (one per shell type)
# ---------------------------------------------------------------------------


def build_cwd_wrapper_bash(command: str, cwd_path: str) -> str:
    """Wrap *command* so the shell's final PWD is captured into *cwd_path*."""
    import shlex

    return f"{{ {command}\n}}; __rc=$?; pwd > {shlex.quote(cwd_path)} 2>/dev/null; exit $__rc"


def build_cwd_wrapper_powershell(command: str, cwd_path: str) -> str:
    """Wrap *command* for PowerShell CWD tracking."""
    # PowerShell: use $(Get-Location).Path for CWD, $LASTEXITCODE for native
    # commands, and default to success for pure cmdlets that leave it unset.
    return (
        f"& {{ {command}\n}}; $__success = $?; $__native = $LASTEXITCODE; "
        f"$__rc = if (-not $__success) {{ 1 }} elseif ($null -eq $__native) {{ 0 }} else {{ $__native }}; "
        f"(Get-Location).Path | Out-File -Encoding UTF8 '{cwd_path}'; "
        f"exit $__rc"
    )


def build_cwd_wrapper(shell: str, command: str, cwd_path: str) -> str:
    """Dispatch to the correct CWD-wrapper builder by shell type."""
    if shell == "powershell":
        return build_cwd_wrapper_powershell(command, cwd_path)
    return build_cwd_wrapper_bash(command, cwd_path)


def build_bg_wrapper(shell: str, command: str) -> str:
    """Build a background-execution wrapper for the given shell type.

    The wrapper emits ``__CLAWCODEX_EXIT__=<rc>`` on stderr so the
    background reaper can extract the exit code.
    """
    if shell == "powershell":
        return (
            f"& {{ {command}\n}}; $__success = $?; $__native = $LASTEXITCODE; "
            f"$__rc = if (-not $__success) {{ 1 }} elseif ($null -eq $__native) {{ 0 }} else {{ $__native }}; "
            f'Write-Error "__CLAWCODEX_EXIT__=$__rc"; '
            f"exit $__rc"
        )
    return f'{{ {command}\n}}; __rc=$?; echo "__CLAWCODEX_EXIT__=$__rc" >&2; exit $__rc'


__all__ = [
    "SHELL_TYPES",
    "ShellType",
    "DEFAULT_HOOK_SHELL",
    "build_powershell_args",
    "find_powershell_path",
    "resolve_shell",
    "build_shell_argv",
    "build_cwd_wrapper",
    "build_cwd_wrapper_bash",
    "build_cwd_wrapper_powershell",
    "build_bg_wrapper",
    "ShellArgvFactory",
]
