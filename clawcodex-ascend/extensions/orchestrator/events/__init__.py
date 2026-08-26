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

"""Orchestrator → IM event bridge (P3).

Defines the event types, the :class:`OrchestratorEventEmitter` (a
:class:`ProgressSink` that also exposes an explicit ``emit()``), the
event → IM text formatter, and :class:`ChannelProgressSink` that
delivers formatted events to the gateway. Per-sink exception isolation
is baked in so an IM failure never breaks the orchestrator main flow.
"""

from __future__ import annotations

from .emitter import OrchestratorEventEmitter
from .formatter import format_event
from .types import EventLevel, OrchestratorEvent

__all__ = [
    "EventLevel",
    "OrchestratorEvent",
    "OrchestratorEventEmitter",
    "format_event",
]
