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

from __future__ import annotations

from .constants import (
    CONTEXT_BLOCKED_TTL_SEC,
    DEFAULT_FOCUS_LEVEL,
    DEFAULT_JITTER_FRACTION,
    TICK_INTERVAL_MS,
    TICK_TAG,
)
from .controller import (
    ProactiveController,
    get_default_controller,
    reset_default_controller_for_tests,
)
from .prompts import get_proactive_section
from .state import AutomationPhase, AutomationState, FocusLevel
from .tick_emitter import TickEmitter

__all__ = [
    "CONTEXT_BLOCKED_TTL_SEC",
    "DEFAULT_FOCUS_LEVEL",
    "DEFAULT_JITTER_FRACTION",
    "TICK_INTERVAL_MS",
    "TICK_TAG",
    "AutomationPhase",
    "AutomationState",
    "FocusLevel",
    "ProactiveController",
    "TickEmitter",
    "get_default_controller",
    "get_proactive_section",
    "reset_default_controller_for_tests",
]
