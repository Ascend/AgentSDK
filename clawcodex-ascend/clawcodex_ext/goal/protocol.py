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

"""Equivalent thread-goal API facade for this Python project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoal, ThreadGoalStatus
from clawcodex_ext.goal.service import KEEP_TOKEN_BUDGET, GoalService


@dataclass(frozen=True)
class ThreadGoalDTO:
    """API thread-goal shape matching upstream v2 protocol fields."""

    thread_id: str
    objective: str
    status: ThreadGoalStatus
    token_budget: int | None
    tokens_used: int
    time_used_seconds: int
    completion_mode: GoalCompletionMode
    evaluation_count: int
    last_evaluation_reason: str | None
    created_at: int
    updated_at: int

    @classmethod
    def from_model(cls, goal: ThreadGoal) -> "ThreadGoalDTO":
        return cls(
            thread_id=goal.thread_id,
            objective=goal.objective,
            status=goal.status,
            token_budget=goal.token_budget,
            tokens_used=goal.tokens_used,
            time_used_seconds=goal.time_used_seconds,
            completion_mode=goal.completion_mode,
            evaluation_count=goal.evaluation_count,
            last_evaluation_reason=goal.last_evaluation_reason,
            created_at=int(goal.created_at.timestamp()),
            updated_at=int(goal.updated_at.timestamp()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "threadId": self.thread_id,
            "objective": self.objective,
            "status": self.status.to_wire(),
            "tokenBudget": self.token_budget,
            "tokensUsed": self.tokens_used,
            "timeUsedSeconds": self.time_used_seconds,
            "completionMode": self.completion_mode.to_wire(),
            "evaluationCount": self.evaluation_count,
            "lastEvaluationReason": self.last_evaluation_reason,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


@dataclass(frozen=True)
class ThreadGoalSetParams:
    thread_id: str
    objective: str | None = None
    status: ThreadGoalStatus | str | None = None
    token_budget: int | None | object = KEEP_TOKEN_BUDGET
    completion_mode: GoalCompletionMode | str | None = None


@dataclass(frozen=True)
class ThreadGoalSetResponse:
    goal: ThreadGoalDTO


@dataclass(frozen=True)
class ThreadGoalReplaceParams:
    """Parameters for atomically replacing a thread's active goal."""

    thread_id: str
    objective: str
    token_budget: int | None = None
    completion_mode: GoalCompletionMode | str = GoalCompletionMode.TOOL


@dataclass(frozen=True)
class ThreadGoalGetParams:
    thread_id: str


@dataclass(frozen=True)
class ThreadGoalGetResponse:
    goal: ThreadGoalDTO | None


@dataclass(frozen=True)
class ThreadGoalClearParams:
    thread_id: str


@dataclass(frozen=True)
class ThreadGoalClearResponse:
    cleared: bool


@dataclass(frozen=True)
class ThreadGoalUpdatedNotification:
    thread_id: str
    turn_id: str | None
    goal: ThreadGoalDTO


@dataclass(frozen=True)
class ThreadGoalClearedNotification:
    thread_id: str


@dataclass(frozen=True)
class GoalProtocolMessage:
    kind: str
    method: str
    payload: object


class GoalEventLog:
    """Small in-process recorder for response/notification ordering tests."""

    def __init__(self) -> None:
        self.messages: list[GoalProtocolMessage] = []

    def response(self, method: str, payload: object) -> None:
        self.messages.append(GoalProtocolMessage("response", method, payload))

    def notification(self, method: str, payload: object) -> None:
        self.messages.append(GoalProtocolMessage("notification", method, payload))

    def clear(self) -> None:
        self.messages.clear()


class ThreadGoalProtocol:
    """Local equivalent of upstream `thread/goal/*` request processors."""

    def __init__(
        self,
        *,
        service: GoalService | None = None,
        events: GoalEventLog | None = None,
    ) -> None:
        self.service = service if service is not None else GoalService()
        self.events = events if events is not None else GoalEventLog()

    def thread_goal_set(self, params: ThreadGoalSetParams) -> ThreadGoalSetResponse:
        goal = self.service.set_goal(
            params.thread_id,
            params.objective,
            status=params.status,
            token_budget=params.token_budget,
            completion_mode=params.completion_mode,
        )
        return self._record_set_response("thread/goal/set", params.thread_id, goal)

    def thread_goal_replace(
        self,
        params: ThreadGoalReplaceParams,
    ) -> ThreadGoalSetResponse:
        """Replace a goal through the store's single atomic mutation."""

        goal = self.service.replace_goal(
            params.thread_id,
            params.objective,
            token_budget=params.token_budget,
            completion_mode=params.completion_mode,
        )
        return self._record_set_response("thread/goal/replace", params.thread_id, goal)

    def _record_set_response(
        self,
        method: str,
        thread_id: str,
        goal: ThreadGoal,
    ) -> ThreadGoalSetResponse:
        dto = ThreadGoalDTO.from_model(goal)
        response = ThreadGoalSetResponse(goal=dto)
        self.events.response(method, response)
        self.events.notification(
            "thread/goal/updated",
            ThreadGoalUpdatedNotification(
                thread_id=thread_id,
                turn_id=None,
                goal=dto,
            ),
        )
        return response

    def thread_goal_get(self, params: ThreadGoalGetParams) -> ThreadGoalGetResponse:
        goal = self.service.get_goal(params.thread_id)
        response = ThreadGoalGetResponse(goal=ThreadGoalDTO.from_model(goal) if goal is not None else None)
        self.events.response("thread/goal/get", response)
        return response

    def thread_goal_clear(self, params: ThreadGoalClearParams) -> ThreadGoalClearResponse:
        cleared = self.service.clear_goal(params.thread_id)
        response = ThreadGoalClearResponse(cleared=cleared)
        self.events.response("thread/goal/clear", response)
        if cleared:
            self.events.notification(
                "thread/goal/cleared",
                ThreadGoalClearedNotification(thread_id=params.thread_id),
            )
        return response


__all__ = [
    "GoalEventLog",
    "GoalProtocolMessage",
    "ThreadGoalClearParams",
    "ThreadGoalClearResponse",
    "ThreadGoalClearedNotification",
    "ThreadGoalDTO",
    "ThreadGoalGetParams",
    "ThreadGoalGetResponse",
    "ThreadGoalProtocol",
    "ThreadGoalReplaceParams",
    "ThreadGoalSetParams",
    "ThreadGoalSetResponse",
    "ThreadGoalUpdatedNotification",
]
