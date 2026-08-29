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

"""Allowlist gate for slash commands received from IM channels.

Runtime allowlists come from ``command_allowlists`` in ``channels.yaml``.
Plain text passes through unchanged; unknown slash commands are rejected at
the gateway boundary.
"""

# This split feature branch is linted without the aggregate package initializer.
# pylint: disable=relative-beyond-top-level

from __future__ import annotations

from collections.abc import Collection

from .command_allowlist import (
    DEFAULT_ORCHESTRATOR_COMMAND_ALLOWLIST,
    DEFAULT_REPL_COMMAND_ALLOWLIST,
)

# Backward-compatible aliases for callers that enumerate the built-in defaults.
# Runtime dispatch receives its effective values from GatewayConfig instead.
REPL_ALLOWED_COMMANDS = frozenset(DEFAULT_REPL_COMMAND_ALLOWLIST)
ORCHESTRATOR_ALLOWED_COMMANDS = frozenset(DEFAULT_ORCHESTRATOR_COMMAND_ALLOWLIST)


def _block_reason(cmd_token: str) -> str:
    """Build a rejection message without echoing the complete input."""
    return f"`{cmd_token}` is not in the command allowlist and was blocked by the gateway."


def _unsupported_reason(command_display: str) -> str:
    return f"Command {command_display} is not supported."


def _slash_parts(text: str) -> list[str]:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return []
    return stripped.split(maxsplit=2)


def check_repl_command(
    text: str,
    *,
    allowed_commands: Collection[str] | None = None,
) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for a REPL slash command.

    REPL allowlist entries contain only the command token, so later arguments
    do not affect matching. Plain text and a lone slash pass through.
    """
    parts = _slash_parts(text)
    if not parts:
        return True, ""
    # Compare the first token, including its leading slash, case-insensitively.
    cmd_token = parts[0].lower()
    # A lone slash opens the REPL command palette.
    if cmd_token == "/":  # nosec B105 - This is a command token, not a password.
        return True, ""
    effective_allowlist = REPL_ALLOWED_COMMANDS if allowed_commands is None else allowed_commands
    if cmd_token in effective_allowlist:
        return True, ""
    return False, _block_reason(cmd_token)


def check_orchestrator_command(
    text: str,
    *,
    allowed_commands: Collection[str] | None = None,
) -> tuple[bool, str]:
    """Return (allowed, reason) for orchestrator IM slash commands.

    Allowlist entries without an argument contain only the command name, such
    as ``/server``. If the first argument is significant it must be part of the
    entry, such as ``/server status``; it will not match ``/server restart``.
    Arguments after the first one do not affect matching. Plain text remains
    pass-through so operator follow-up/context messages still work.
    """
    parts = _slash_parts(text)
    if not parts:
        return True, ""
    first = parts[0].lower()
    second = parts[1].lower() if len(parts) > 1 else ""
    command_display = first if not second else f"{first} {second}"
    command_key = command_display
    effective_allowlist = ORCHESTRATOR_ALLOWED_COMMANDS if allowed_commands is None else allowed_commands
    if command_key in effective_allowlist:
        return True, ""
    return False, _unsupported_reason(command_display)


__all__ = [
    "ORCHESTRATOR_ALLOWED_COMMANDS",
    "REPL_ALLOWED_COMMANDS",
    "check_orchestrator_command",
    "check_repl_command",
]
