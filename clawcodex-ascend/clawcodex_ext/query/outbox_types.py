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

"""P102-C Typed outbox events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class CronPromptEvent:
    """Represent a scheduler-triggered cron prompt."""

    prompt: str = ""
    task_id: str = ""
    run_id: str = ""

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        if key == "type":
            return "cron_prompt"
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        if key == "type":
            return "cron_prompt"
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key == "type" or hasattr(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return other == {
                "type": "cron_prompt",
                "prompt": self.prompt,
                "task_id": self.task_id,
                "run_id": self.run_id,
            }
        if isinstance(other, CronPromptEvent):
            return self.prompt == other.prompt and self.task_id == other.task_id and self.run_id == other.run_id
        return NotImplemented


@dataclass
class CronMissedEvent:
    """Represent a missed one-shot cron notification."""

    tasks: list[str] = field(default_factory=list)
    notification: str = ""

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        if key == "type":
            return "cron_missed"
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        if key == "type":
            return "cron_missed"
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key == "type" or hasattr(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return other == {
                "type": "cron_missed",
                "tasks": self.tasks,
                "notification": self.notification,
            }
        if isinstance(other, CronMissedEvent):
            return self.tasks == other.tasks and self.notification == other.notification
        return NotImplemented


@dataclass
class ProactivePromptEvent:
    """Prompt injected by proactive tick/sleep wake-up."""

    prompt: str = ""
    source: str = "tick"

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        if key == "type":
            return "proactive_prompt"
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        if key == "type":
            return "proactive_prompt"
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key == "type" or hasattr(self, key)


@dataclass
class GenericOutboxEvent:
    """Represent an outbox event with arbitrary payload fields."""

    payload: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        return self.payload.get(key, default)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        return self.payload[key]

    def __contains__(self, key: str) -> bool:
        return key in self.payload

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenericOutboxEvent:
        return cls(payload=dict(d))


OutboxEvent = Union[CronPromptEvent, CronMissedEvent, ProactivePromptEvent, GenericOutboxEvent]


def outbox_event_from_dict(d: dict[str, Any]) -> OutboxEvent:
    """Deserialize a typed outbox event from a mapping."""
    etype = d.get("type", "")
    if etype == "cron_prompt":
        return CronPromptEvent(
            prompt=d.get("prompt", ""),
            task_id=d.get("task_id", ""),
            run_id=d.get("run_id", ""),
        )
    if etype == "cron_missed":
        return CronMissedEvent(
            tasks=d.get("tasks", []),
            notification=d.get("notification", ""),
        )
    if etype == "proactive_prompt":
        return ProactivePromptEvent(
            prompt=d.get("prompt", ""),
            source=d.get("source", "tick"),
        )
    return GenericOutboxEvent.from_dict(d)
