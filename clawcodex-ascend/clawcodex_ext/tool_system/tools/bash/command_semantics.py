"""Exit code interpretation for common commands.

Many commands use exit codes to convey information other than success/failure.
For example, grep returns 1 when no matches are found, which is not an error.

F-107: Extended with PowerShell-specific exit-code semantics when
``shell="powershell"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||[;|])\s*")


@dataclass(frozen=True)
class CommandInterpretation:
    is_error: bool
    message: str | None = None


CommandSemantic = Callable[[int, str, str], CommandInterpretation]


def _default_semantic(exit_code: int, _stdout: str, _stderr: str) -> CommandInterpretation:
    return CommandInterpretation(
        is_error=exit_code != 0,
        message=f"Command failed with exit code {exit_code}" if exit_code != 0 else None,
    )


def _grep_semantic(exit_code: int, _stdout: str, _stderr: str) -> CommandInterpretation:
    return CommandInterpretation(
        is_error=exit_code >= 2,
        message="No matches found" if exit_code == 1 else None,
    )


def _find_semantic(exit_code: int, _stdout: str, _stderr: str) -> CommandInterpretation:
    return CommandInterpretation(
        is_error=exit_code >= 2,
        message="Some directories were inaccessible" if exit_code == 1 else None,
    )


def _diff_semantic(exit_code: int, _stdout: str, _stderr: str) -> CommandInterpretation:
    return CommandInterpretation(
        is_error=exit_code >= 2,
        message="Files differ" if exit_code == 1 else None,
    )


def _test_semantic(exit_code: int, _stdout: str, _stderr: str) -> CommandInterpretation:
    return CommandInterpretation(
        is_error=exit_code >= 2,
        message="Condition is false" if exit_code == 1 else None,
    )


# ---------------------------------------------------------------------------
# F-107: PowerShell-specific semantics
# ---------------------------------------------------------------------------


def _select_string_semantic(exit_code: int, _stdout: str, _stderr: str) -> CommandInterpretation:
    """select-string / sls: 0 = matched, 1 = no match, 2 = error (same as grep)."""
    return CommandInterpretation(
        is_error=exit_code >= 2,
        message="No matches found" if exit_code == 1 else None,
    )


def _pwsh_default_semantic(exit_code: int, _stdout: str, _stderr: str) -> CommandInterpretation:
    """Default PowerShell semantics.

    - Native cmdlets don't set $LASTEXITCODE unless they explicitly exit.
    - External programs set $LASTEXITCODE.
    - exit_code == 0 means success, != 0 means failure.
    """
    return _default_semantic(exit_code, _stdout, _stderr)


COMMAND_SEMANTICS: dict[str, CommandSemantic] = {
    "grep": _grep_semantic,
    "rg": _grep_semantic,
    "find": _find_semantic,
    "diff": _diff_semantic,
    "test": _test_semantic,
    "[": _test_semantic,
}

# ---------------------------------------------------------------------------
# F-107: PowerShell command semantics map
# ---------------------------------------------------------------------------

PWSH_COMMAND_SEMANTICS: dict[str, CommandSemantic] = {
    "select-string": _select_string_semantic,
    "sls": _select_string_semantic,
    "findstr": _select_string_semantic,
}


def _heuristically_extract_base_command(command: str) -> str:
    """Extract the primary command name from a complex command line.

    Takes the last segment (after splitting on &&, ||, ;, |) since that
    determines the exit code.
    """
    segments = _SPLIT_RE.split(command)
    last = segments[-1].strip() if segments else command.strip()
    first_word = last.split()[0] if last.split() else ""
    return first_word


def interpret_command_result(
    command: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    shell: str = "bash",
) -> CommandInterpretation:
    """Interpret a command's exit code into a human-readable message.

    When *shell* is ``"powershell"``, uses ``PWSH_COMMAND_SEMANTICS`` in
    addition to the POSIX ``COMMAND_SEMANTICS`` map.
    """
    base = _heuristically_extract_base_command(command)

    if shell == "powershell":
        base_lower = base.lower()
        semantic = PWSH_COMMAND_SEMANTICS.get(base_lower, _pwsh_default_semantic)
        return semantic(exit_code, stdout, stderr)

    semantic = COMMAND_SEMANTICS.get(base, _default_semantic)
    return semantic(exit_code, stdout, stderr)
