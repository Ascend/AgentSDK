#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Downstream CLI permissions -- owns resolve_permission_state(args)."""

from __future__ import annotations


def resolve_permission_state(args) -> None:
    """Resolve and stash permission state on ``args``.

    Computes the effective :class:`PermissionMode` from the CLI flags and
    settings, runs the root/sudo safety gate, and emits a single log line
    when either bypass flag was passed. Stashes the result on ``args`` so
    every downstream mode (print, TUI, REPL) can read it without re-deriving.

    Mirrors the wiring in ``typescript/src/main.tsx`` lines 1087-1392 plus
    the safety check in ``typescript/src/setup.ts:382-401``.
    """
    import logging as _logging

    from src.permissions.dangerous_safety import (
        enforce_dangerous_skip_permissions_safety,
    )
    from src.permissions.modes import (
        has_allow_bypass_permissions_mode,
        initial_permission_mode_from_cli,
    )

    # F-47: plumb ``settings.permissions.default_mode`` into the mode
    # resolver. Import lazily so the CLI module stays importable in tests
    # that never touch settings (e.g. permission-only unit tests).
    try:
        from src.settings.settings import get_settings as _get_settings
    except Exception:  # pragma: no cover - defensive
        _get_settings = None

    dangerously = bool(getattr(args, "dangerously_skip_permissions", False))
    allow_dangerously = bool(getattr(args, "allow_dangerously_skip_permissions", False))
    permission_mode_cli = getattr(args, "permission_mode", None)

    # Safety gate first -- refuse to run as root outside a sandbox.
    enforce_dangerous_skip_permissions_safety(
        bypass_requested=dangerously or allow_dangerously,
    )

    # F-47: resolve settings-side default mode from the structured
    # ``permissions.default_mode`` field. The legacy top-level
    # ``settings.permission_mode`` channel has been removed; on-disk
    # values at that key are no longer consulted at startup.
    settings_default_mode: str | None = None
    if _get_settings is not None:
        try:
            s = _get_settings()
        except Exception:
            s = None
        if s is not None:
            pc = getattr(s, "permissions", None)
            structured_default = getattr(pc, "default_mode", None) if pc is not None else None
            if structured_default:
                settings_default_mode = structured_default

    mode = initial_permission_mode_from_cli(
        permission_mode_cli=permission_mode_cli,
        dangerously_skip_permissions=dangerously,
        settings_default_mode=settings_default_mode,
    )

    # When the user explicitly passes --permission-mode bypassPermissions,
    # treat bypass as available for the Shift+Tab cycle even without
    # --dangerously-skip-permissions.  Without this, the cycle guard
    # in get_next_permission_mode blocks plan → bypassPermissions →
    # default, and the user can never return to bypass after cycling away.
    is_bypass_available = (
        dangerously or allow_dangerously or has_allow_bypass_permissions_mode() or mode == "bypassPermissions"
    )

    # Stash on args so downstream entrypoints don't need to re-derive.
    args._resolved_permission_mode = mode
    args._resolved_is_bypass_available = is_bypass_available

    if dangerously or allow_dangerously:
        _logging.getLogger("clawcodex.permissions").info(
            "permission flags: dangerously_skip=%s allow_dangerously_skip=%s mode=%s",
            dangerously,
            allow_dangerously,
            mode,
        )
