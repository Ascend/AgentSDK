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

"""Permission-mode cycling for the Shift+Tab keybinding.

Mirrors ``typescript/src/utils/permissions/getNextPermissionMode.ts``. The TS
reference has an Anthropic-internal ``USER_TYPE === 'ant'`` branch and a
``TRANSCRIPT_CLASSIFIER``-gated ``auto`` cycle target.

Cycle order (with ``bypassPermissions`` enabled):

    ``default → acceptEdits → plan → bypassPermissions → auto → default``

The ``bypassPermissions → auto`` step is guarded by :func:`can_cycle_to_auto`
so it only appears when the environment is safe (no protected directory, no
recent dangerous-operation denials).  When unavailable, the cycle wraps
``bypassPermissions → default`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clawcodex_ext.permissions.types import PermissionMode, PermissionUpdateSetMode, ToolPermissionContext

# ---------------------------------------------------------------------------
# Cycle table registry — downstream extensions can inject additional steps.
# The default table matches the upstream cycle.  Extensions call
# ``register_cycle_step()`` to insert transitions (e.g.  bypassPermissions →
# dontAsk) without modifying this file.
# ---------------------------------------------------------------------------

# Each entry is (source_mode, target_mode).  The table is consulted in order;
# first match wins.  The final fallback for any unmatched mode is "default".
_CYCLE_TABLE: list[tuple[str, str]] = [
    ("default", "acceptEdits"),
    ("acceptEdits", "plan"),
    ("plan", "bypassPermissions"),  # guarded by is_bypass_permissions_mode_available
    # Downstream keeps ``dontAsk`` in the interactive cycle. ``auto`` remains
    # available through explicit activation and its safety classifier.
    ("bypassPermissions", "dontAsk"),
]


def register_cycle_step(source: str, target: str, *, after: str | None = None) -> None:
    """Register an additional cycle transition.

    Args:
        source: The mode to transition *from*.
        target: The mode to transition *to*.
        after: If given, insert after the existing entry whose *source*
            equals this value.  Otherwise append at the end (but before
            the implicit ``→ default`` fallback).

    Example::

        # Insert bypassPermissions → dontAsk after the bypassPermissions row
        register_cycle_step("bypassPermissions", "dontAsk", after="bypassPermissions")
    """
    entry = (source, target)
    # A mode has exactly one Shift+Tab successor.  Downstream registration
    # therefore overrides the upstream transition for the same source; merely
    # inserting another row would leave it unreachable because first match
    # wins in ``get_next_permission_mode``.
    for idx, (existing_source, _existing_target) in enumerate(_CYCLE_TABLE):
        if existing_source == source:
            _CYCLE_TABLE[idx] = entry
            return
    if after is not None:
        for idx, (s, _t) in enumerate(_CYCLE_TABLE):
            if s == after:
                _CYCLE_TABLE.insert(idx + 1, entry)
                return
    _CYCLE_TABLE.append(entry)


def get_next_permission_mode(context: ToolPermissionContext) -> PermissionMode:
    """Return the next mode when the user presses Shift+Tab.

    Mirrors ``getNextPermissionMode`` in
    ``typescript/src/utils/permissions/getNextPermissionMode.ts:34-79``.

    Default cycle:

    - ``default`` → ``acceptEdits``
    - ``acceptEdits`` → ``plan``
    - ``plan`` → ``bypassPermissions`` (when available) else ``default``
    - ``bypassPermissions`` → ``auto`` (when :func:`can_cycle_to_auto`)
      else ``default``
    - ``auto`` → ``default``
    - ``bubble`` → ``default`` (escape hatch — this
      mode is not part of the user-facing cycle but we still need a defined
      transition so Shift+Tab never strands the user.)

    Downstream extensions can extend the cycle via :func:`register_cycle_step`.
    """
    mode = context.mode
    for source, target in _CYCLE_TABLE:
        if mode != source:
            continue
        # Guard: "plan → bypassPermissions" is only valid when the mode is
        # available.  Fall through to default otherwise.
        if target == "bypassPermissions" and not context.is_bypass_permissions_mode_available:
            return "default"
        # Guard: "bypassPermissions → auto" is only valid when the environment
        # is safe.  Fall through to default otherwise.
        if target == "auto" and not can_cycle_to_auto(context, check_protected_directory=False):
            return "default"
        return target
    # bubble and any unrecognised mode fall through to default.
    return "default"


def cycle_permission_mode(
    context: ToolPermissionContext,
) -> tuple[PermissionMode, ToolPermissionContext]:
    """Compute the next mode and return the (mode, updated_context) pair.

    Mirrors ``cyclePermissionMode`` in
    ``typescript/src/utils/permissions/getNextPermissionMode.ts:88-101``.
    The updated context is produced via :func:`apply_permission_update` with
    a ``setMode`` update so any future hooks that observe context updates fire
    consistently.
    """
    next_mode = get_next_permission_mode(context)
    from src.permissions.updates import apply_permission_update

    next_context = apply_permission_update(
        context,
        PermissionUpdateSetMode(
            type="setMode",
            destination="session",
            mode=next_mode,
        ),
    )
    return next_mode, next_context


PROTECTED_DIRECTORIES: tuple[str, ...] = (
    ".git",
    ".vscode",
    ".clawcodex",
    ".idea",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
)


def _is_in_protected_directory(cwd: str | Path | None) -> bool:
    """Check if current working directory is inside a protected location."""
    if cwd is None:
        return False
    cwd_path = Path(cwd).resolve()
    for protected in PROTECTED_DIRECTORIES:
        if protected in cwd_path.parts:
            return True
        protected_path = cwd_path / protected
        if protected_path.exists():
            return True
    return False


def _has_recent_dangerous_operations(
    denial_tracker: Any | None = None,
) -> bool:
    """Check if there have been recent denied dangerous operations."""
    if denial_tracker is None:
        try:
            from src.permissions.check import get_denial_tracker

            denial_tracker = get_denial_tracker()
        except Exception:
            return False
    dangerous_tools = ("Bash", "Write", "Edit")
    for tool in dangerous_tools:
        if denial_tracker.get_denial_count(tool) >= 3:
            return True
    return False


def can_cycle_to_auto(
    context: ToolPermissionContext,
    *,
    check_protected_directory: bool = True,
    check_danger_history: bool = True,
) -> bool:
    """Check if auto mode is available and safe to activate.

    Auto mode is not part of the Shift+Tab cycle because it requires
    explicit activation and safety checks. This function validates:
    1. Current permission configuration allows auto mode
    2. Not operating in a protected directory (.git, .vscode, etc.)
    3. No recent dangerous operation denials

    Args:
        context: Current permission context
        check_protected_directory: Whether to check for protected dirs
        check_danger_history: Whether to check denial history

    Returns:
        True if auto mode can be safely activated
    """
    if context.mode == "auto":
        return True
    if check_protected_directory:
        cwd = getattr(context, "cwd", None)
        if _is_in_protected_directory(cwd):
            return False
    if check_danger_history:
        if _has_recent_dangerous_operations():
            return False
    return True


def get_auto_mode_availability_reason(
    context: ToolPermissionContext,
) -> str | None:
    """Return human-readable reason why auto mode is unavailable.

    Args:
        context: Current permission context

    Returns:
        None if auto mode is available, otherwise the blocking reason
    """
    if context.mode == "auto":
        return None
    cwd = getattr(context, "cwd", None)
    if _is_in_protected_directory(cwd):
        return "Working directory is inside a protected location (.git, .vscode, etc.)"
    if _has_recent_dangerous_operations():
        return "Recent dangerous operations were denied (auto mode paused for safety)"
    return None


__all__ = [
    "PROTECTED_DIRECTORIES",
    "can_cycle_to_auto",
    "cycle_permission_mode",
    "get_auto_mode_availability_reason",
    "get_next_permission_mode",
    "register_cycle_step",
]
