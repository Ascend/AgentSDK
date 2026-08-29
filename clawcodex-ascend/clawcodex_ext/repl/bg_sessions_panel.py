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

"""P94-F Background-session status panel rendering."""

from __future__ import annotations
# pylint: disable=E0611

from pathlib import Path

from clawcodex_ext.tasks.bg_session import BgSession, is_bg_sessions_enabled
from clawcodex_ext.tasks.bg_session_registry import BgSessionRegistry


# ---------------------------------------------------------------------------
# Footer statistics
# ---------------------------------------------------------------------------


def footer_summary(
    registry: BgSessionRegistry,
    *,
    workspace_root: Path | None = None,
) -> str:
    """Format background-session counts for the status footer."""
    if not is_bg_sessions_enabled(registry.config):
        return ""
    sessions = registry.list(workspace_root=workspace_root)
    running = sum(1 for s in sessions if s.is_active())
    orphaned = sum(1 for s in sessions if s.status == "orphaned")
    if running == 0 and orphaned == 0:
        return ""
    parts = [f"bg:{running}"]
    if orphaned:
        parts.append(f"orphan:{orphaned}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Grouped task-list rendering
# ---------------------------------------------------------------------------


def format_bg_sessions_status(
    registry: BgSessionRegistry,
    *,
    workspace_root: Path | None = None,
    include_completed: bool = False,
) -> str:
    """Format the background-session status panel."""
    if not is_bg_sessions_enabled(registry.config):
        return "(bg_sessions disabled)"
    sessions = registry.list(workspace_root=workspace_root)
    if not include_completed:
        sessions = [s for s in sessions if not s.is_terminal()]
    if not sessions:
        return "BG sessions: (none)"
    lines = ["BG sessions:"]
    for s in sessions:
        marker = _status_marker(s.status)
        lines.append(f"  {marker} {s.id}  pid={s.pid}  agent={s.agent_name or '-'}  ws={s.workspace_root}")
    return "\n".join(lines)


def _status_marker(status: str) -> str:
    return {
        "running": "▶",
        "starting": "…",
        "paused": "⏸",
        "orphaned": "⚠",
        "completed": "✓",
        "failed": "✗",
        "stopped": "■",
        "unknown": "?",
    }.get(status, "?")


# ---------------------------------------------------------------------------
# Completion notification
# ---------------------------------------------------------------------------


def format_completion_notification(session: BgSession) -> str:
    """Format a background-session completion notice."""
    return (
        f"<bg-session-notification>\n"
        f"  session_id: {session.session_id}\n"
        f"  status: {session.status}\n"
        f"  workspace: {session.workspace_root}\n"
        f"  resume: clawcodex --resume {session.session_id}\n"
        f"</bg-session-notification>"
    )


# ---------------------------------------------------------------------------
# Build and scan a registry from the environment.
# ---------------------------------------------------------------------------


def make_panel_registry() -> BgSessionRegistry:
    """Build and scan a background-session registry."""
    return BgSessionRegistry()


__all__ = [
    "footer_summary",
    "format_bg_sessions_status",
    "format_completion_notification",
    "make_panel_registry",
]
