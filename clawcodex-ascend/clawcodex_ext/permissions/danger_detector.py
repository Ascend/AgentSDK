"""Danger Detection for Auto Mode.

This module provides hardcoded rules for detecting dangerous operations
that should never be auto-approved regardless of LLM classification.
These rules act as a safety boundary before any LLM judgment.

Protected operations include:
- Bash: rm -rf, sudo, destructive commands, system file writes
- Write/Edit: .git/, .vscode/, system directories, config files
- MCP tools: always require explicit approval
"""

from __future__ import annotations

import re
from typing import Any

DANGER_PATTERNS_BASH = [
    r"rm\s+-rf",
    r"rm\s+-r\s+-f",
    r"rm\s+--no-preserve-root",
    r"sudo\s+",
    r"chmod\s+777",
    r"chmod\s+-R\s+777",
    r">\s*/dev/sd[a-z]",
    r"dd\s+if=",
    r"mkfs",
    r"fdisk",
    r":(){ :|:& };:",
    r"wget\s+.+\s+-O\s+/",
    r"curl\s+.+\s+-o\s+/",
    r"mv\s+.+\s+/dev/null",
    r"kill\s+-9\s+1",
    r"shutdown",
    r"reboot",
    r"halt",
    r"init\s+0",
    r"init\s+6",
]

DANGER_PATTERNS_WRITE = [
    r"\.git/",
    r"\.git/config",
    r"\.git/HEAD",
    r"\.vscode/",
    r"\.clawcodex/",
    r"/etc/passwd",
    r"/etc/shadow",
    r"/etc/ssh/",
    r"C:\\Windows\\",
    r"C:\\Program Files\\",
    r"/usr/bin/",
    r"/usr/sbin/",
    r"/bin/",
    r"/sbin/",
    r"~/.ssh/",
    r"~/.bashrc$",
    r"~/.profile$",
    r"package\.json$",
    r"requirements\.txt$",
    r"Pipfile$",
    r"pyproject\.toml$",
]

DANGER_PATTERNS_EDIT = [
    r"\.git/config",
    r"\.git/HEAD",
    r"\.git/refs/",
    r"/etc/",
    r"C:\\Windows\\",
    r"~/.ssh/",
    r"~/.bashrc",
    r"~/.profile",
]


def detect_dangerous_bash_command(command: str) -> tuple[bool, str]:
    if not command:
        return True, "empty command"

    for pattern in DANGER_PATTERNS_BASH:
        if re.search(pattern, command, re.IGNORECASE):
            return True, f"matches dangerous pattern: {pattern}"

    if re.search(r"\s*>\s*/", command):
        return True, "writes to root filesystem"

    if re.search(r"/dev/null\s*;\s*rm", command):
        return True, "destructive redirection chain"

    return False, ""


def detect_dangerous_write_path(file_path: str) -> tuple[bool, str]:
    if not file_path:
        return True, "empty path"

    for pattern in DANGER_PATTERNS_WRITE:
        if re.search(pattern, file_path, re.IGNORECASE):
            return True, f"protected location: {pattern}"

    if file_path.startswith("/"):
        dangerous_roots = ["/etc", "/usr", "/bin", "/sbin", "/root", "/boot", "/proc", "/sys"]
        for root in dangerous_roots:
            if file_path.startswith(root):
                return True, f"system directory: {root}"

    return False, ""


def detect_dangerous_edit_path(file_path: str) -> tuple[bool, str]:
    if not file_path:
        return True, "empty path"

    for pattern in DANGER_PATTERNS_EDIT:
        if re.search(pattern, file_path, re.IGNORECASE):
            return True, f"protected location: {pattern}"

    return False, ""


def detect_dangerous_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
) -> tuple[bool, str]:
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        return detect_dangerous_bash_command(command)

    if tool_name == "Write":
        file_path = tool_input.get("file_path", "")
        return detect_dangerous_write_path(file_path)

    if tool_name == "Edit":
        file_path = tool_input.get("file_path", "")
        return detect_dangerous_edit_path(file_path)

    if tool_name == "MultiEdit":
        edits = tool_input.get("edits", [])
        for edit in edits:
            file_path = edit.get("file_path", "")
            is_danger, reason = detect_dangerous_edit_path(file_path)
            if is_danger:
                return True, reason

    if tool_name == "NotebookEdit":
        file_path = tool_input.get("path", "")
        return detect_dangerous_edit_path(file_path)

    if tool_name.startswith("mcp__"):
        return True, "MCP tools require explicit approval"

    return False, ""


__all__ = [
    "DANGER_PATTERNS_BASH",
    "DANGER_PATTERNS_EDIT",
    "DANGER_PATTERNS_WRITE",
    "detect_dangerous_bash_command",
    "detect_dangerous_edit_path",
    "detect_dangerous_tool_call",
    "detect_dangerous_write_path",
]
