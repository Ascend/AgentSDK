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

"""Agent Dashboard — Agent model tools.

The :class:`DashboardGet` and :class:`DashboardList` tools expose
the dashboard's read-only snapshot to the model. They mirror the
:func:`/dashboard` command's semantics (per-source filters, status
filter, etc.) so the model and the user see a consistent view.

Both tools honour the plan §4.3 "read-only" invariant:
they never call any subsystem's write methods, and
``is_read_only`` is wired to ``True`` so the agent-loop's
permission layer treats them as observability tools.

The tools are registered in :data:`ALL_STATIC_TOOLS` for the
default tool pool, but the public surface is the two tool
instances (``DashboardGetTool`` / ``DashboardListTool``) so
other registries (test fixtures, the visualizer's "debug"
pool) can opt in independently.
"""

from __future__ import annotations

from .dashboard_get import DashboardGetTool
from .dashboard_list import DashboardListTool

__all__ = [
    "DashboardGetTool",
    "DashboardListTool",
]
