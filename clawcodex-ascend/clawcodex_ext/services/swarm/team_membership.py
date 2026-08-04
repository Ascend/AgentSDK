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

"""Team-membership predicates — Chunk F / WI-6.4."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool_system.context import ToolContext


def is_team_lead(context: ToolContext) -> bool:
    """True iff the active agent is the team lead."""
    team = getattr(context, "team", None)
    agent_id = getattr(context, "agent_id", None)
    if not isinstance(team, dict) or not agent_id:
        return False
    lead_agent_id = team.get("lead_agent_id")
    return bool(lead_agent_id) and agent_id == lead_agent_id


__all__ = [
    "is_team_lead",
]
