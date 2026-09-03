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

"""Orchestrator runtime protocols (no implementations).

Protocols here must not import ``clawcodex_ext.*``, ``src.*``, or
``extensions.orchestrator.*``.
"""

from __future__ import annotations

from .agent_runtime import AgentRuntime, SessionContext
from .backend import BackendUnavailable, OrchestratordBackend
from .coordinator import CoordinatorContextProvider
from .diagnostics import DiagnosticsProbe, HeartbeatState, HeartbeatStatus
from .git_backend import FileStatusLike, GitBackend
from .im_channel import ImChannel, ImCommandRouter, ImInbound, ImOutbound
from .intent_focus import FocusArea, IntentFocus, IssueLike
from .messages import (
    AgentEvent,
    AgentEventType,
    PhaseComplete,
    SessionComplete,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from .provider import LLMProvider
from .session_storage import ConversationLike, SessionMeta, SessionStorage
from .workspace_tooling import ToolContextLike, WorkspaceTooling

__all__ = [
    "AgentEvent",
    "AgentEventType",
    "AgentRuntime",
    "BackendUnavailable",
    "ConversationLike",
    "CoordinatorContextProvider",
    "DiagnosticsProbe",
    "FileStatusLike",
    "FocusArea",
    "GitBackend",
    "HeartbeatState",
    "HeartbeatStatus",
    "ImChannel",
    "ImCommandRouter",
    "ImInbound",
    "ImOutbound",
    "IntentFocus",
    "IssueLike",
    "LLMProvider",
    "OrchestratordBackend",
    "PhaseComplete",
    "SessionComplete",
    "SessionContext",
    "SessionMeta",
    "SessionStorage",
    "TextDelta",
    "ToolCallEvent",
    "ToolContextLike",
    "ToolResultEvent",
    "WorkspaceTooling",
]
