#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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
"""Spec-1 goal boundary.

Old goal v1 modules and the legacy model-callable `Goal` tool are not part
of this package anymore. Later F-122 specs will rebuild the upstream
contract in place.
"""

from __future__ import annotations

from .command import GOAL_COMMAND, GoalCommand
from .gate import GOALS_FEATURE, ensure_goals_feature_registered, goal_enabled
from .model import ThreadGoal, ThreadGoalStatus
from .protocol import (
    GoalEventLog,
    ThreadGoalClearParams,
    ThreadGoalClearResponse,
    ThreadGoalClearedNotification,
    ThreadGoalDTO,
    ThreadGoalGetParams,
    ThreadGoalGetResponse,
    ThreadGoalProtocol,
    ThreadGoalSetParams,
    ThreadGoalSetResponse,
    ThreadGoalUpdatedNotification,
)
from .accounting import (
    BudgetLimitedGoalDisposition,
    GoalAccountingState,
    GoalProgressSnapshot,
    IdleGoalProgressSnapshot,
    goal_token_delta_for_usage,
)
from .files import (
    GOAL_FILE_NAME,
    MAX_THREAD_GOAL_OBJECTIVE_CHARS,
    materialize_goal_objective,
    objective_text_for_edit,
)
from .observability import (
    GoalObservation,
    GoalObservationRecorder,
)
from .runtime import (
    GoalContinuationRequest,
    GoalRuntime,
    goal_runtime_for_context,
)
from .steering import (
    BUDGET_LIMIT_STEERING_MARKER,
    CONTINUATION_STEERING_MARKER,
    OBJECTIVE_UPDATED_STEERING_MARKER,
    budget_limit_steering_message,
    continuation_steering_message,
    objective_updated_steering_message,
)
from .service import GoalService, GoalServiceError
from .store import GoalStore, GoalUpdate, current_goal_thread_id, goals_db_filename, goals_db_path
from .tools import (
    CREATE_GOAL_TOOL_NAME,
    GET_GOAL_TOOL_NAME,
    GOAL_MODEL_TOOL_NAMES,
    UPDATE_GOAL_TOOL_NAME,
    filter_goal_model_tools_for_context,
    goal_model_tools_visible,
    make_goal_model_tools,
)

__all__ = [
    "GOAL_COMMAND",
    "GOALS_FEATURE",
    "CREATE_GOAL_TOOL_NAME",
    "BUDGET_LIMIT_STEERING_MARKER",
    "BudgetLimitedGoalDisposition",
    "CONTINUATION_STEERING_MARKER",
    "GET_GOAL_TOOL_NAME",
    "GOAL_FILE_NAME",
    "GoalAccountingState",
    "GoalContinuationRequest",
    "GoalObservation",
    "GoalObservationRecorder",
    "GoalProgressSnapshot",
    "GoalRuntime",
    "GoalStore",
    "GoalEventLog",
    "GoalUpdate",
    "GoalCommand",
    "GoalService",
    "GoalServiceError",
    "GOAL_MODEL_TOOL_NAMES",
    "MAX_THREAD_GOAL_OBJECTIVE_CHARS",
    "OBJECTIVE_UPDATED_STEERING_MARKER",
    "ThreadGoal",
    "ThreadGoalClearParams",
    "ThreadGoalClearResponse",
    "ThreadGoalClearedNotification",
    "ThreadGoalDTO",
    "ThreadGoalGetParams",
    "ThreadGoalGetResponse",
    "ThreadGoalProtocol",
    "ThreadGoalSetParams",
    "ThreadGoalSetResponse",
    "ThreadGoalStatus",
    "ThreadGoalUpdatedNotification",
    "UPDATE_GOAL_TOOL_NAME",
    "budget_limit_steering_message",
    "current_goal_thread_id",
    "continuation_steering_message",
    "ensure_goals_feature_registered",
    "filter_goal_model_tools_for_context",
    "goal_enabled",
    "goal_model_tools_visible",
    "goal_runtime_for_context",
    "goal_token_delta_for_usage",
    "goals_db_filename",
    "goals_db_path",
    "IdleGoalProgressSnapshot",
    "make_goal_model_tools",
    "materialize_goal_objective",
    "objective_text_for_edit",
    "objective_updated_steering_message",
]
