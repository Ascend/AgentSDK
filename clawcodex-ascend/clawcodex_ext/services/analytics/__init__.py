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

"""Analytics subsystem.

Event logging, session metadata, and event sinks.
Mirrors TypeScript analytics/ directory.
"""

from __future__ import annotations

from .events import AnalyticsEvent, EventType, get_analytics_sink, log_event, set_analytics_sink
from .metadata import SessionAnalyticsMetadata, collect_session_metadata
from .sink import AnalyticsSink, ConsoleSink, FileSink, NullSink

__all__ = [
    "AnalyticsEvent",
    "AnalyticsSink",
    "ConsoleSink",
    "EventType",
    "FileSink",
    "NullSink",
    "SessionAnalyticsMetadata",
    "collect_session_metadata",
    "get_analytics_sink",
    "log_event",
    "set_analytics_sink",
]
