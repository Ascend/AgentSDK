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
"""Thread goal state model for upstream-compatible goal mode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ThreadGoalStatus(str, Enum):
    """Persisted thread-goal statuses, matching upstream Codex."""

    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    USAGE_LIMITED = "usage_limited"
    BUDGET_LIMITED = "budget_limited"
    COMPLETE = "complete"

    @classmethod
    def from_wire(cls, value: str) -> "ThreadGoalStatus":
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown thread goal status `{value}`") from exc

    def to_wire(self) -> str:
        return self.value

    def is_active(self) -> bool:
        return self is ThreadGoalStatus.ACTIVE

    def is_terminal(self) -> bool:
        return self in {ThreadGoalStatus.BUDGET_LIMITED, ThreadGoalStatus.COMPLETE}


class GoalCompletionMode(str, Enum):
    """Authority responsible for deciding that a goal is complete."""

    TOOL = "tool"
    EVALUATOR = "evaluator"

    @classmethod
    def from_wire(cls, value: str) -> "GoalCompletionMode":
        """Parse a completion-mode value from persisted or protocol data."""

        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown goal completion mode `{value}`") from exc

    def to_wire(self) -> str:
        """Return the stable persisted and protocol representation."""

        return self.value


@dataclass(frozen=True)
class ThreadGoal:
    """Persisted goal for one recoverable thread/session."""

    thread_id: str
    goal_id: str
    objective: str
    status: ThreadGoalStatus
    token_budget: int | None
    tokens_used: int
    time_used_seconds: int
    created_at: datetime
    updated_at: datetime
    completion_mode: GoalCompletionMode = GoalCompletionMode.TOOL
    evaluation_count: int = 0
    last_evaluation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "goal_id": self.goal_id,
            "objective": self.objective,
            "status": self.status.to_wire(),
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "time_used_seconds": self.time_used_seconds,
            "completion_mode": self.completion_mode.to_wire(),
            "evaluation_count": self.evaluation_count,
            "last_evaluation_reason": self.last_evaluation_reason,
            "created_at": _normalize_datetime(self.created_at).isoformat(),
            "updated_at": _normalize_datetime(self.updated_at).isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThreadGoal":
        return cls(
            thread_id=str(data["thread_id"]),
            goal_id=str(data["goal_id"]),
            objective=str(data["objective"]),
            status=ThreadGoalStatus.from_wire(str(data["status"])),
            token_budget=_optional_int(data.get("token_budget")),
            tokens_used=int(data["tokens_used"]),
            time_used_seconds=int(data["time_used_seconds"]),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            completion_mode=GoalCompletionMode.from_wire(
                str(data.get("completion_mode", GoalCompletionMode.TOOL.value))
            ),
            evaluation_count=int(data.get("evaluation_count", 0)),
            last_evaluation_reason=_optional_str(data.get("last_evaluation_reason")),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    text = str(value)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return _normalize_datetime(datetime.fromisoformat(text))


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"datetime {value!r} cannot be represented in UTC") from exc


__all__ = ["GoalCompletionMode", "ThreadGoal", "ThreadGoalStatus"]
