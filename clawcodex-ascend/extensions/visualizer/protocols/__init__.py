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

"""Dashboard and asciicast recorder protocols owned by the visualizer package.

Both modules are *local copies* of the corresponding
``extensions.capabilities`` modules; see ``dashboard.py`` and
``recorder.py`` for the drift-tracking rationale.
"""

from __future__ import annotations

from extensions.visualizer.protocols.dashboard import (
    DASHBOARD_STATUS_BLOCKED,
    DASHBOARD_STATUS_COMPLETED,
    DASHBOARD_STATUS_FAILED,
    DASHBOARD_STATUS_IN_PROGRESS,
    DASHBOARD_STATUS_PENDING,
    DASHBOARD_STATUSES,
    DashboardEntry,
    DashboardSink,
    DashboardSource,
    filter_entries,
    normalize_source_name,
)
from extensions.visualizer.protocols.recorder import (
    AsciicastCapture,
    AsciicastEvent,
    AsciicastHeader,
    RecordableSource,
)

__all__ = [
    "DASHBOARD_STATUS_BLOCKED",
    "DASHBOARD_STATUS_COMPLETED",
    "DASHBOARD_STATUS_FAILED",
    "DASHBOARD_STATUS_IN_PROGRESS",
    "DASHBOARD_STATUS_PENDING",
    "DASHBOARD_STATUSES",
    "AsciicastCapture",
    "AsciicastEvent",
    "AsciicastHeader",
    "DashboardEntry",
    "DashboardSink",
    "DashboardSource",
    "RecordableSource",
    "filter_entries",
    "normalize_source_name",
]
