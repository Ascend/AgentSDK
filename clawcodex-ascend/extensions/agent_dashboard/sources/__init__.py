#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Agent Dashboard — built-in data sources.

This package ships the two data sources the plan §3 commits
to in Phase 2: :class:`GoalDashboardSource` (reads from
``GoalService``) and :class:`TasksDashboardSource` (reads from
``ToolContext.tasks``). Optional Orchestrator / SOP sources live in
``extensions/orchestrator`` and ``extensions/sop_converter`` and
register themselves against the default registry on import.
"""

from __future__ import annotations

from .goal_source import GoalDashboardSource
from .orchestrator_source import OrchestratorDashboardSource
from .sop_source import SOPDashboardSource, register_sop_dashboard_source
from .tasks_source import TasksDashboardSource

__all__ = [
    "GoalDashboardSource",
    "OrchestratorDashboardSource",
    "SOPDashboardSource",
    "TasksDashboardSource",
    "register_sop_dashboard_source",
]
