#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
# This file is derived from Clawd Codex (https://github.com/agentforce314/clawcodex),
# which is licensed under the MIT License.
# Copyright (c) 2026 Clawd Codex Team
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

"""Swarm team helpers.

Mirrors TypeScript swarm/helpers.ts — utility functions for team coordination.
"""

from __future__ import annotations

from .teammate import Teammate, TeammateManager, TeammateStatus


def get_active_teammates(manager: TeammateManager) -> list[Teammate]:
    """Get all currently active teammates."""
    return [t for t in manager.all_teammates if t.is_active]


def format_team_summary(manager: TeammateManager) -> str:
    """Format a human-readable summary of the team status."""
    teammates = manager.all_teammates
    if not teammates:
        return "No teammates."

    lines = [f"Team: {len(teammates)} teammate(s)"]
    for t in teammates:
        status_icon = {
            TeammateStatus.RUNNING: "🔄",
            TeammateStatus.COMPLETED: "✅",
            TeammateStatus.FAILED: "❌",
            TeammateStatus.KILLED: "⛔",
            TeammateStatus.PENDING: "⏳",
        }.get(t.status, "?")
        prompt_preview = t.config.prompt[:40] + ("..." if len(t.config.prompt) > 40 else "")
        lines.append(f"  {status_icon} [{t.id}] {t.status.value}: {prompt_preview}")

    active = manager.active_count
    if active > 0:
        lines.append(f"  Active: {active}")

    return "\n".join(lines)
