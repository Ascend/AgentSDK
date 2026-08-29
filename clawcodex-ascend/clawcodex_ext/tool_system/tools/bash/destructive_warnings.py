#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Detects potentially destructive bash commands and returns a warning string.

Purely informational -- does not affect permission logic or auto-approval.
"""

from __future__ import annotations

import re

_DestructivePattern = tuple[re.Pattern[str], str]

_DESTRUCTIVE_PATTERNS: list[_DestructivePattern] = [
    # Git -- data loss / hard to reverse
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "Note: may discard uncommitted changes"),
    (
        re.compile(r"\bgit\s+push\b[^;&|\n]*[ \t](--force|--force-with-lease|-f)\b"),
        "Note: may overwrite remote history",
    ),
    (
        re.compile(r"\bgit\s+clean\b(?![^;&|\n]*(?:-[a-zA-Z]*n|--dry-run))[^;&|\n]*-[a-zA-Z]*f"),
        "Note: may permanently delete untracked files",
    ),
    (
        re.compile(r"\bgit\s+checkout\s+(--\s+)?\.[ \t]*($|[;&|\n])"),
        "Note: may discard all working tree changes",
    ),
    (
        re.compile(r"\bgit\s+restore\s+(--\s+)?\.[ \t]*($|[;&|\n])"),
        "Note: may discard all working tree changes",
    ),
    (
        re.compile(r"\bgit\s+stash[ \t]+(drop|clear)\b"),
        "Note: may permanently remove stashed changes",
    ),
    (
        re.compile(r"\bgit\s+branch\s+(-D[ \t]|--delete\s+--force|--force\s+--delete)\b"),
        "Note: may force-delete a branch",
    ),
    # Git -- safety bypass
    (
        re.compile(r"\bgit\s+(commit|push|merge)\b[^;&|\n]*--no-verify\b"),
        "Note: may skip safety hooks",
    ),
    (
        re.compile(r"\bgit\s+commit\b[^;&|\n]*--amend\b"),
        "Note: may rewrite the last commit",
    ),
    # File deletion
    (
        re.compile(r"(^|[;&|\n]\s*)rm\s+-[a-zA-Z]*[rR][a-zA-Z]*f|(^|[;&|\n]\s*)rm\s+-[a-zA-Z]*f[a-zA-Z]*[rR]"),
        "Note: may recursively force-remove files",
    ),
    (
        re.compile(r"(^|[;&|\n]\s*)rm\s+-[a-zA-Z]*[rR]"),
        "Note: may recursively remove files",
    ),
    (re.compile(r"(^|[;&|\n]\s*)rm\s+-[a-zA-Z]*f"), "Note: may force-remove files"),
    # Database
    (
        re.compile(r"\b(DROP|TRUNCATE)\s+(TABLE|DATABASE|SCHEMA)\b", re.IGNORECASE),
        "Note: may drop or truncate database objects",
    ),
    (
        re.compile(r"\bDELETE\s+FROM\s+\w+[ \t]*(;|\"|'|\n|$)", re.IGNORECASE),
        "Note: may delete all rows from a database table",
    ),
    # Infrastructure
    (re.compile(r"\bkubectl\s+delete\b"), "Note: may delete Kubernetes resources"),
    (
        re.compile(r"\bterraform\s+destroy\b"),
        "Note: may destroy Terraform infrastructure",
    ),
    # ---------------------------------------------------------------------------
    # PowerShell-specific destructive patterns
    # ---------------------------------------------------------------------------
    # File deletion
    (
        re.compile(r"\bremove-item\b.*-(recurse|r)\b.*-(force|f)\b", re.IGNORECASE),
        "Note: may recursively force-remove files",
    ),
    (
        re.compile(r"\bremove-item\b.*-(force|f)\b.*-(recurse|r)\b", re.IGNORECASE),
        "Note: may recursively force-remove files",
    ),
    (re.compile(r"\bclear-disk\b", re.IGNORECASE), "Note: may erase disk contents"),
    (re.compile(r"\bformat-volume\b", re.IGNORECASE), "Note: may format a volume"),
    # Privilege escalation / arbitrary execution
    (
        re.compile(r"\bstart-process\b.*-verb\s+runas\b", re.IGNORECASE),
        "Note: may run with elevated privileges",
    ),
    (
        re.compile(r"\binvoke-expression\b|\biex\b", re.IGNORECASE),
        "Note: may execute arbitrary code",
    ),
    # Safety bypass
    (
        re.compile(
            r"\bset-executionpolicy\b.*-(executionpolicy\s+)?(unrestricted|bypass|remoteSigned)",
            re.IGNORECASE,
        ),
        "Note: may weaken script execution policy",
    ),
]


def get_destructive_command_warning(command: str) -> str | None:
    """Return a human-readable warning if *command* matches a destructive pattern."""
    for pattern, warning in _DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            return warning
    return None
