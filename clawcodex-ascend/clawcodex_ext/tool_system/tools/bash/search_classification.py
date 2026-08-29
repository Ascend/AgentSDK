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

"""Classify bash commands as search, read, list, or silent.

Handles compound commands (&&, ||, ;, |), semantic-neutral commands
(echo, printf, true, false, :), and redirect target skipping.

Extended with PowerShell command sets for ``shell="powershell"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SPLIT_RE = re.compile(r"(>>|>&|>|&&|\|\||[|;])")

SEARCH_COMMANDS: frozenset[str] = frozenset(
    [
        "find",
        "grep",
        "rg",
        "ag",
        "ack",
        "locate",
        "which",
        "whereis",
    ]
)

READ_COMMANDS: frozenset[str] = frozenset(
    [
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "wc",
        "stat",
        "file",
        "strings",
        "jq",
        "awk",
        "cut",
        "sort",
        "uniq",
        "tr",
    ]
)

LIST_COMMANDS: frozenset[str] = frozenset(
    [
        "ls",
        "tree",
        "du",
    ]
)

SEMANTIC_NEUTRAL_COMMANDS: frozenset[str] = frozenset(
    [
        "echo",
        "printf",
        "true",
        "false",
        ":",
    ]
)

SILENT_COMMANDS: frozenset[str] = frozenset(
    [
        "mv",
        "cp",
        "rm",
        "mkdir",
        "rmdir",
        "chmod",
        "chown",
        "chgrp",
        "touch",
        "ln",
        "cd",
        "export",
        "unset",
        "wait",
    ]
)

# ---------------------------------------------------------------------------
# PowerShell command sets
# ---------------------------------------------------------------------------

PWSH_SEARCH_COMMANDS: frozenset[str] = frozenset(
    [
        "select-string",
        "sls",
        "findstr",
        "where-object",
        "where",
    ]
)

PWSH_READ_COMMANDS: frozenset[str] = frozenset(
    [
        "get-content",
        "gc",
        "type",
        "get-item",
        "gi",
        "measure-object",
        "measure",
        "get-command",
        "gcm",
        "get-help",
        "help",
        "man",
        "get-process",
        "ps",
    ]
)

PWSH_LIST_COMMANDS: frozenset[str] = frozenset(
    [
        "get-childitem",
        "gci",
        "dir",
        "ls",
        "tree",
    ]
)

PWSH_SEMANTIC_NEUTRAL_COMMANDS: frozenset[str] = frozenset(
    [
        "write-host",
        "write-output",
        "echo",
        "write-warning",
        "write-verbose",
        "write-debug",
        "out-null",
        "true",
        "false",
    ]
)

PWSH_SILENT_COMMANDS: frozenset[str] = frozenset(
    [
        "new-item",
        "ni",
        "md",
        "mkdir",
        "remove-item",
        "ri",
        "rm",
        "del",
        "erase",
        "move-item",
        "mi",
        "mv",
        "move",
        "copy-item",
        "ci",
        "cp",
        "copy",
        "set-content",
        "sc",
        "set-location",
        "sl",
        "cd",
        "push-location",
        "pushd",
        "pop-location",
        "popd",
        "rename-item",
        "rni",
        "ren",
        "export-csv",
        "export-clixml",
        "out-file",
        "add-content",
        "ac",
        "clear-content",
        "clear-item",
        "cli",
        "remove-variable",
        "rv",
        "set-variable",
        "set",
        "remove-psdrive",
        "rdr",
        "new-psdrive",
        "mount",
        "new-variable",
        "nv",
        "start-sleep",
        "sleep",
    ]
)


@dataclass(frozen=True)
class SearchOrReadResult:
    is_search: bool = False
    is_read: bool = False
    is_list: bool = False


def _split_with_operators(command: str) -> list[str]:
    """Split command into parts and operators, preserving operator tokens."""
    return [p for p in _SPLIT_RE.split(command) if p.strip()]


# ---------------------------------------------------------------------------
# PowerShell command classification helpers
# ---------------------------------------------------------------------------

# Pre-compute lowercase set for O(1) lookup
_PWSH_SILENT_LOWER = frozenset(c.lower() for c in PWSH_SILENT_COMMANDS)


def _is_powershell_verb_noun(base: str) -> bool:
    """Check if *base* looks like a PowerShell Verb-Noun command."""
    b = base.lower()
    if "-" not in b:
        return False
    verb = b.split("-")[0]
    # Common PowerShell verbs
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
    )


def _classify_pwsh_base(base: str) -> tuple[bool, bool, bool]:
    """Classify a PowerShell command. Returns (is_search, is_read, is_list)."""
    b = base.lower()
    if b in PWSH_SEARCH_COMMANDS:
        return (True, False, False)
    if b in PWSH_READ_COMMANDS:
        return (False, True, False)
    if b in PWSH_LIST_COMMANDS:
        return (False, False, True)
    # Verb-Noun heuristic
    if "-" in b:
        verb = b.split("-")[0]
        if verb in ("select", "find", "search", "where", "measure"):
            return (True, False, False)
        if verb in (
            "get",
            "read",
            "show",
            "dump",
            "format",
            "compare",
            "group",
            "sort",
        ):
            return (False, True, False)
        if verb in (
            "set",
            "new",
            "remove",
            "clear",
            "write",
            "out",
            "export",
            "import",
            "add",
            "push",
            "pop",
            "rename",
            "copy",
            "move",
            "mount",
            "disable",
            "enable",
            "register",
            "unregister",
            "start",
            "stop",
            "restart",
            "resume",
            "suspend",
            "lock",
            "unlock",
            "save",
            "switch",
            "sync",
            "test",
        ):
            # These are typically write/modify operations, not search/read
            return (False, False, False)
    return (False, False, False)


def is_search_or_read_command(command: str, shell: str = "bash") -> SearchOrReadResult:
    """Classify a bash command for UI collapsing.

    For pipelines, ALL non-neutral parts must be search/read/list commands
    for the whole command to be considered collapsible.

    When *shell* is ``"powershell"``, uses PowerShell-specific command sets.
    """
    try:
        parts = _split_with_operators(command)
    except Exception:  # nosec
        return SearchOrReadResult()

    if not parts:
        return SearchOrReadResult()

    has_search = False
    has_read = False
    has_list = False
    has_non_neutral = False
    skip_next = False

    is_pwsh = shell == "powershell"

    for part in parts:
        if skip_next:
            skip_next = False
            continue

        stripped = part.strip()

        if stripped in (">", ">>", ">&"):
            skip_next = True
            continue
        if stripped in ("||", "&&", "|", ";"):
            continue

        base = stripped.split()[0] if stripped.split() else ""
        if not base:
            continue

        if is_pwsh:
            bl = base.lower()
            if bl in PWSH_SEMANTIC_NEUTRAL_COMMANDS:
                continue
        else:
            if base in SEMANTIC_NEUTRAL_COMMANDS:
                continue

        has_non_neutral = True

        if is_pwsh:
            is_search, is_read, is_list = _classify_pwsh_base(base)
        else:
            is_search = base in SEARCH_COMMANDS
            is_read = base in READ_COMMANDS
            is_list = base in LIST_COMMANDS

        if not is_search and not is_read and not is_list:
            return SearchOrReadResult()

        if is_search:
            has_search = True
        if is_read:
            has_read = True
        if is_list:
            has_list = True

    if not has_non_neutral:
        return SearchOrReadResult()

    return SearchOrReadResult(
        is_search=has_search,
        is_read=has_read,
        is_list=has_list,
    )


def is_silent_command(command: str, shell: str = "bash") -> bool:
    """Return True when *command* is expected to produce no stdout on success.

    When *shell* is ``"powershell"``, uses PowerShell-specific silent command sets.
    """
    try:
        parts = _split_with_operators(command)
    except Exception:  # nosec
        return False

    if not parts:
        return False

    has_non_fallback = False
    last_operator: str | None = None
    skip_next = False
    is_pwsh = shell == "powershell"

    for part in parts:
        if skip_next:
            skip_next = False
            continue

        stripped = part.strip()

        if stripped in (">", ">>", ">&"):
            skip_next = True
            continue
        if stripped in ("||", "&&", "|", ";"):
            last_operator = stripped
            continue

        base = stripped.split()[0] if stripped.split() else ""
        if not base:
            continue

        if last_operator == "||":
            if is_pwsh:
                if base.lower() in PWSH_SEMANTIC_NEUTRAL_COMMANDS:
                    continue
            elif base in SEMANTIC_NEUTRAL_COMMANDS:
                continue

        has_non_fallback = True
        if is_pwsh:
            if base.lower() not in _PWSH_SILENT_LOWER:
                return False
        elif base not in SILENT_COMMANDS:
            return False

    return has_non_fallback
