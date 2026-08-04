"""PowerShell command safety analysis — F-107.

Unlike the POSIX ``bash_security.py`` which uses ``tree-sitter-bash`` AST
parsing, this module uses heuristic regex + Get-Command style analysis since:

1. ``tree-sitter-powershell`` is not stable enough (community version).
2. PowerShell's Verb-Noun naming convention is regular enough for effective
   heuristic classification.
3. The safety categories map 1:1 to the existing Bash system.

Safety levels (matching ``bash_security.BashSafetyLevel``)::

    safe         — Write-Host, $true, pure pipeline (no side effects)
    read_only    — Get-*, Select-*, Where-*, Measure-* cmdlets
    write        — Set-*, Add-*, Out-File, Export-*, Copy-Item
    destructive  — Remove-Item -Recurse -Force, Clear-*, Invoke-SqlCmd
    dangerous    — Invoke-Expression/iex, Start-Process -Verb RunAs
    unknown      — Unrecognised commands
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from clawcodex_ext.permissions.types import (
    PermissionAskDecision,
    PermissionResult,
    SafetyCheckDecisionReason,
)

PWSHSafetyLevel = Literal["safe", "read_only", "write", "destructive", "dangerous", "unknown"]

# ---------------------------------------------------------------------------
# Verb-based classification
# ---------------------------------------------------------------------------

# Verb prefixes and their safety level
# Get-*, Select-*, Where-*, Measure-*, Out-* (when not writing to file),
# Format-*, Compare-*, Group-*, Sort-*, Tee-*, Convert*-*
READ_ONLY_VERBS: frozenset[str] = frozenset(
    {
        "get",
        "select",
        "where",
        "measure",
        "out",
        "format",
        "compare",
        "group",
        "sort",
        "tee",
        "convertfrom",
        "convertto",
        "convert",
        "import",
        "write",  # Write-Host/Write-Output are safe; Write-* that writes to file handled separately
        "read",
        "show",
        "dump",
        "find",
        "search",
        "test",
        "resolve",
        "trace",
    }
)

# Set-*, Add-*, Copy-*, Move-*, Rename-*, Export-*, Out-File (file output),
# New-* (creates objects/items), Remove-* (safe removals without -Recurse),
# Clear-* (clear content without -Force), Update-*, Register-*, Unregister-*
WRITE_VERBS: frozenset[str] = frozenset(
    {
        "set",
        "add",
        "copy",
        "move",
        "rename",
        "export",
        "update",
        "register",
        "unregister",
        "publish",
        "save",
        "submit",
        "sync",
        "switch",
        "use",
        "wait",
        "enable",
        "disable",
        "mount",
        "dismount",
        "approve",
        "deny",
        "complete",
        "confirm",
        "connect",
        "disconnect",
        "install",
        "uninstall",
        "block",
        "grant",
        "revoke",
        "merge",
        "split",
        "join",
    }
)

# Remove-Item with -Recurse/-Force, Clear-* with -Force, Drop-*,
# Invoke-SqlCmd with destructive SQL, Format-* -Force (drive format),
# Stop-* (stop process/service)
DESTRUCTIVE_VERBS: frozenset[str] = frozenset(
    {
        "stop",
        "debug",
        "undefine",
    }
)

# Invoke-Expression (iex) — arbitrary code execution
# Start-Process -Verb RunAs — privilege escalation
DANGEROUS_CMD: frozenset[str] = frozenset(
    {
        "invoke-expression",
        "iex",
    }
)

# Dangerous patterns in command text
DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\biex\b", re.IGNORECASE),
    re.compile(r"\binvoke-expression\b", re.IGNORECASE),
    re.compile(r"start-process.*-verb\s+runas", re.IGNORECASE),
    re.compile(r"new-object.*net\.webclient", re.IGNORECASE),
    re.compile(r"\$\(.*\)", re.IGNORECASE),  # Subexpression in string
]

# Destructive flags
DESTRUCTIVE_FLAGS: list[re.Pattern] = [
    re.compile(r"-recurse", re.IGNORECASE),
    re.compile(r"-force", re.IGNORECASE),
    re.compile(r"-literalpath.*\*", re.IGNORECASE),
]

# Clean file write patterns (safe writes)
WRITE_REDIRECT = re.compile(r">>?|out-file|add-content|set-content", re.IGNORECASE)


@dataclass(frozen=True)
class PwshSafetyAnalysis:
    """Result of PowerShell command safety analysis."""

    safety: PWSHSafetyLevel
    commands: list[str]
    reason: str = ""


def _extract_commands(command: str) -> list[str]:
    """Split a PowerShell command pipeline/chain into individual commands."""
    # Split on pipes, semicolons, &&, ||
    parts = re.split(r"\s*(?:\||;|&&|\|\|)\s*", command)
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            result.append(part)
    return result


def _get_verb_noun(cmd_text: str) -> tuple[str, str]:
    """Extract (verb, noun) from a PowerShell command like ``Get-ChildItem -Path .``."""
    # Remove leading &, ./, .\ path prefixes
    cmd = cmd_text.strip().lstrip(".&/\\").lstrip()
    # Split on whitespace and take first token
    first = cmd.split()[0] if cmd and cmd.split() else ""
    # PowerShell is case-insensitive; normalise to lower
    first = first.lower()
    if "-" in first:
        parts = first.split("-", 1)
        return (parts[0], parts[1])
    return (first, "")


def _has_dangerous_redirection(cmd_text: str) -> bool:
    """Check for dangerous redirection patterns."""
    # > / >> to system paths
    dangerous_targets = re.compile(r"(>>?\s*(\\|\$|system32|windows|boot|dev|proc))", re.IGNORECASE)
    return bool(dangerous_targets.search(cmd_text))


def analyze_powershell_safety(command: str) -> PwshSafetyAnalysis:
    """Analyse a PowerShell command and return its safety level."""
    commands = _extract_commands(command)
    if not commands:
        return PwshSafetyAnalysis(safety="safe", commands=[])

    # Check for dangerous patterns first
    for pat in DANGEROUS_PATTERNS:
        if pat.search(command):
            return PwshSafetyAnalysis(
                safety="dangerous",
                commands=commands,
                reason=f"Dangerous pattern: {pat.pattern}",
            )

    overall_safety: PWSHSafetyLevel = "safe"
    all_commands: list[str] = []

    for cmd_text in commands:
        verb, noun = _get_verb_noun(cmd_text)
        cmd_name = f"{verb}-{noun}" if noun else verb
        all_commands.append(cmd_name)

        if not verb:
            # Check if it starts with a redirect (implicit output)
            if cmd_text.strip().startswith(">"):
                # Write-level (file output)
                overall_safety = _max_safety(overall_safety, "write")
                continue
            overall_safety = _max_safety(overall_safety, "unknown")
            continue

        if verb in DANGEROUS_CMD:
            overall_safety = _max_safety(overall_safety, "dangerous")
            continue

        # Check for destructive flags (Recurse, Force)
        has_destructive_flag = any(flag.search(cmd_text) for flag in DESTRUCTIVE_FLAGS)

        if verb in DESTRUCTIVE_VERBS:
            overall_safety = _max_safety(overall_safety, "destructive")
            continue

        if verb in ("remove", "clear", "format"):
            if has_destructive_flag:
                overall_safety = _max_safety(overall_safety, "destructive")
            else:
                overall_safety = _max_safety(overall_safety, "write")
            continue

        if verb in ("new", "set", "add", "copy", "move", "rename"):
            if has_destructive_flag:
                overall_safety = _max_safety(overall_safety, "destructive")
            else:
                overall_safety = _max_safety(overall_safety, "write")
            continue

        # Safe write-* cmdlets are NOT write operations
        # Note: checked before WRITE_VERBS since "write" is in WRITE_VERBS
        # Must use the full Verb-Noun form (verb+'-'+noun), not just verb
        cmd_lower = verb.lower() + ("-" + noun.lower() if noun else "")
        if cmd_lower in (
            "write-host",
            "write-output",
            "write-warning",
            "write-verbose",
            "write-debug",
        ):
            overall_safety = _max_safety(overall_safety, "safe")
            continue
        if verb.lower() in ("echo", "true", "false"):
            overall_safety = _max_safety(overall_safety, "safe")
            continue

        if verb in WRITE_VERBS:
            overall_safety = _max_safety(overall_safety, "write")
            continue

        if verb in READ_ONLY_VERBS:
            overall_safety = _max_safety(overall_safety, "read_only")
            continue

        # Common aliases and standalone commands
        if verb.lower() in ("dir", "ls", "type", "cat", "more", "pwd", "sleep", "start-sleep"):
            overall_safety = _max_safety(overall_safety, "safe")
            continue
        if verb.lower() in ("cd", "sl", "set-location", "pushd", "popd", "md", "mkdir"):
            overall_safety = _max_safety(overall_safety, "write")
            continue
        if verb.lower() in ("del", "rm", "ri", "remove-item", "rd", "rmdir"):
            if has_destructive_flag:
                overall_safety = _max_safety(overall_safety, "destructive")
            else:
                overall_safety = _max_safety(overall_safety, "write")
            continue
        if verb.lower() in (
            "cp",
            "copy",
            "copy-item",
            "mv",
            "move",
            "move-item",
            "ren",
            "rename-item",
        ):
            overall_safety = _max_safety(overall_safety, "write")
            continue
        if verb.lower() in ("ni", "new-item", "sc", "set-content"):
            overall_safety = _max_safety(overall_safety, "write")
            continue
        if verb.lower() in ("ac", "add-content", "out-file"):
            overall_safety = _max_safety(overall_safety, "write")
            continue

        # Check for external program execution (no Verb-Noun pattern)
        if _is_powershell_verb_noun(verb):
            overall_safety = _max_safety(overall_safety, "unknown")
        else:
            # Treat unrecognised Verb-Noun as write-level (conservative)
            if "-" in verb:
                overall_safety = _max_safety(overall_safety, "write")
            else:
                overall_safety = _max_safety(overall_safety, "unknown")

    return PwshSafetyAnalysis(
        safety=overall_safety,
        commands=all_commands,
    )


_SAFETY_ORDER: dict[str, int] = {
    "safe": 0,
    "read_only": 1,
    "write": 2,
    "destructive": 3,
    "dangerous": 4,
    "unknown": 5,
}


def _max_safety(a: PWSHSafetyLevel, b: PWSHSafetyLevel) -> PWSHSafetyLevel:
    """Return the more severe of two safety levels."""
    if _SAFETY_ORDER.get(a, 0) >= _SAFETY_ORDER.get(b, 0):
        return a
    return b


def _is_powershell_verb_noun(cmd: str) -> bool:
    """Check if *cmd* looks like a standard PowerShell Verb-Noun."""
    # Simple heuristic: if it contains "-" and the part before "-" is
    # a recognised PowerShell verb prefix
    if "-" not in cmd:
        return False
    verb = cmd.split("-")[0].lower()
    return verb in (
        "get",
        "set",
        "new",
        "remove",
        "invoke",
        "write",
        "out",
        "clear",
        "export",
        "import",
        "add",
        "where",
        "select",
        "measure",
        "push",
        "pop",
        "rename",
        "start",
        "stop",
        "restart",
        "format",
        "convert",
        "compare",
        "group",
        "sort",
        "foreach",
        "find",
        "search",
        "read",
        "show",
        "dump",
        "wait",
        "disable",
        "enable",
        "register",
        "unregister",
        "mount",
        "dismount",
        "copy",
        "move",
        "join",
        "split",
        "resume",
        "suspend",
        "use",
        "complete",
        "enter",
        "exit",
        "lock",
        "unlock",
        "optimize",
        "repair",
        "reset",
        "resolve",
        "save",
        "switch",
        "sync",
        "test",
        "trace",
        "update",
        "approve",
        "deny",
        "confirm",
        "connect",
        "disconnect",
        "install",
        "uninstall",
        "block",
        "grant",
        "revoke",
        "merge",
        "publish",
        "submit",
        "debug",
        "undefine",
    )


def check_powershell_command_safety(
    command: str,
    cwd: str | None = None,
    allowed_directories: list[str] | None = None,
) -> PermissionResult | None:
    """Analyse a PowerShell command and return a permission decision.

    Mirrors ``check_bash_command_safety`` in ``bash_security.py``.
    """
    _ = cwd, allowed_directories  # reserved for future path-based checks
    analysis = analyze_powershell_safety(command)

    if analysis.safety == "dangerous":
        return PermissionAskDecision(
            behavior="ask",
            message=f"Dangerous PowerShell command ({', '.join(analysis.commands)}) requires confirmation.",
            decision_reason=SafetyCheckDecisionReason(
                reason=f"Dangerous PowerShell: {', '.join(analysis.commands)}",
                classifier_approvable=True,
            ),
        )

    if analysis.safety == "destructive":
        return PermissionAskDecision(
            behavior="ask",
            message=f"Destructive PowerShell command ({', '.join(analysis.commands)}) requires confirmation.",
            decision_reason=SafetyCheckDecisionReason(
                reason=f"Destructive PowerShell: {', '.join(analysis.commands)}",
                classifier_approvable=True,
            ),
        )

    if analysis.safety == "unknown":
        return PermissionAskDecision(
            behavior="ask",
            message="Unknown PowerShell command requires confirmation.",
            decision_reason=SafetyCheckDecisionReason(
                reason="Unknown PowerShell command",
                classifier_approvable=True,
            ),
        )

    return None
