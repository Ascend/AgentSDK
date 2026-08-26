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

"""Orchestrator event types for the IM bridge."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    SUCCESS = "success"


# Terminal session reasons (from SessionComplete) that map to IM levels.
TERMINAL_REASON_LEVEL: dict[str, EventLevel] = {
    "success": EventLevel.SUCCESS,
    "task_complete": EventLevel.SUCCESS,
    "already_completed": EventLevel.SUCCESS,
    "stagnation": EventLevel.WARN,
    "loop_detected": EventLevel.WARN,
    "noop_completed": EventLevel.INFO,
    "budget_exhausted": EventLevel.WARN,
    "max_turns_exceeded": EventLevel.WARN,
    "rate_limit_circuit_open": EventLevel.ERROR,
}


@dataclass
class OrchestratorEvent:
    """One orchestrator event destined for IM / audit / LiveView."""

    event_type: str
    issue_id: str
    level: EventLevel = EventLevel.INFO
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "issue_id": self.issue_id,
            "level": self.level.value,
            "message": self.message,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }


__all__ = ["EventLevel", "OrchestratorEvent", "TERMINAL_REASON_LEVEL"]
