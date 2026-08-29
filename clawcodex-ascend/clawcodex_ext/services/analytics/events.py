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

"""Analytics event types and logging.

Mirrors TypeScript analytics/events.ts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .sink import AnalyticsSink, NullSink


class EventType(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    COMPACT = "compact"
    ERROR = "error"
    PERMISSION_PROMPT = "permission_prompt"
    PERMISSION_DECISION = "permission_decision"
    MODEL_SWITCH = "model_switch"
    AGENT_SPAWN = "agent_spawn"
    AGENT_COMPLETE = "agent_complete"
    # Image pipeline events (resize, compress, PDF extraction, pre-API
    # validation). Subtype in data["subtype"] distinguishes specific
    # operations; mirrors TS tengu_image_* events from imageResizer.ts.
    IMAGE_PROCESSING = "image_processing"


@dataclass
class AnalyticsEvent:
    """A single analytics event."""

    type: EventType
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    model: str = ""
    data: dict[str, Any] = field(default_factory=dict)


_global_sink: AnalyticsSink = NullSink()


def set_analytics_sink(sink: AnalyticsSink) -> None:
    """Set the global analytics sink."""
    global _global_sink
    _global_sink = sink


def get_analytics_sink() -> AnalyticsSink:
    """Get the current global analytics sink."""
    return _global_sink


def log_event(
    event_type: EventType,
    session_id: str = "",
    model: str = "",
    **data: Any,
) -> AnalyticsEvent:
    """Log an analytics event to the global sink."""
    event = AnalyticsEvent(
        type=event_type,
        session_id=session_id,
        model=model,
        data=data,
    )
    _global_sink.emit(event)
    return event
