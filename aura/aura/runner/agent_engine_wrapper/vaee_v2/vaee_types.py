#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import List, Optional, Any

from pydantic.dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field


@dataclass
class RequestRecord:
    """Request Track Record Data Class

    Corresponds to the request_records table.
    """

    unique_id: str
    request_id: str
    session_id: str
    model: str
    messages: List[Any]
    run_id: Optional[str] = None
    tokenizer_path: Optional[str] = None

    raw_request: Optional[Any] = None
    raw_response: Optional[Any] = None

    text_request: Optional[Any] = None
    text_response: Optional[Any] = None

    prompt_text: Optional[str] = None
    token_ids: Optional[List[int]] = None
    token_request: Optional[Any] = None
    token_response: Optional[Any] = None

    response_text: Optional[str] = None
    response_ids: Optional[List[int]] = None

    full_conversation_text: Optional[str] = None
    full_conversation_token_ids: Optional[List[int]] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_duration_ms: Optional[float] = None

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cache_hit_tokens: int = 0

    error: Optional[str] = None
    error_traceback: Optional[str] = None

    created_at: Optional[datetime] = None

    archive_location: Optional[str] = None
    archived_at: Optional[datetime] = None


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self):
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolOutput:
    name: str
    output: str | list | dict | None = None
    error: str | None = None
    metadata: dict | None = None

    def __str__(self) -> str:
        """Convert the tool output to a string representation."""
        if self.error:
            return f"Error: {self.error}"
        elif self.output is None:
            return ""
        elif isinstance(self.output, list | dict):
            import json

            return json.dumps(self.output)
        else:
            return str(self.output)

    def to_string(self) -> str:
        """
        Convert the tool output to a string representation.

        Example usage:
            tool_output = ToolOutput(name="calculator", output=42)
            result_string = tool_output.to_string()
        """
        return str(self)


@dataclass
class ModelOutput:
    text: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None
    multi_modal_inputs: dict[str, list] | None = None
    logprobs: list[float] | None = None  # completion logprobs
    prompt_logprobs: list[float] | None = None  # prompt logprobs aligned to prompt_ids
    prompt_length: int = 0
    completion_length: int = 0
    finish_reason: str | None = None

    def to_dict(self):
        return {
            "text": self.text,
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls] if self.tool_calls else [],
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "multi_modal_inputs": self.multi_modal_inputs,
            "logprobs": self.logprobs,
            "prompt_logprobs": self.prompt_logprobs,
            "prompt_length": self.prompt_length,
            "completion_length": self.completion_length,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            text=data.get("text"),
            content=data.get("content"),
            reasoning=data.get("reasoning"),
            tool_calls=[ToolCall(**tool_call) for tool_call in data.get("tool_calls", [])]
            if data.get("tool_calls")
            else None,
            prompt_ids=data.get("prompt_ids"),
            completion_ids=data.get("completion_ids"),
            multi_modal_inputs=data.get("multi_modal_inputs"),
            logprobs=data.get("logprobs"),
            prompt_logprobs=data.get("prompt_logprobs"),
            prompt_length=data.get("prompt_length", 0),
            completion_length=data.get("completion_length", 0),
            finish_reason=data.get("finish_reason"),
        )


class _StepBase(BaseModel):
    """A single interaction step (one LLM call with optional reward)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: Any | None = None
    output: Any | None = None
    action: Any | None = None
    reward: float = 0.0
    done: bool = False
    metadata: dict | None = None


class _TrajectoryBase(BaseModel):
    """A sequence of Steps forming one agent trajectory."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "agent"
    task: Any = None
    steps: list[Step] = Field(default_factory=list)
    reward: float | None = None
    input: dict | None = None  # Function arguments (SDK usage)
    output: Any = None  # Function return value (SDK usage)
    signals: dict[str, float] = Field(default_factory=dict)  # Evaluation signals
    metadata: dict | None = None

    @property
    def result(self):
        """Get the output from the trajectory (backward compatibility)."""
        return self.output


class _EpisodeBase(BaseModel):
    """A rollout episode containing one or more Trajectories."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: Any = None
    termination_reason: Any | None = None
    is_correct: bool = False
    trajectories: list[Trajectory] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.id.split(":")[0]

    @property
    def rollout_idx(self) -> str:
        return self.id.split(":")[1]


class Step(_StepBase):
    """Training step with token IDs, logprobs, advantage."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    # Training-specific fields
    prompt_ids: list[int] | list[Any] = Field(default_factory=list)
    response_ids: list[int] = Field(default_factory=list)
    logprobs: list[float] = Field(default_factory=list)
    routing_matrices: list[str] | None = None  # per-token routing matrices (R3, transient)

    chat_completions: list[dict[str, Any]] = Field(default_factory=list)

    observation: Any = None
    thought: str = ""
    # action: inherited from _StepBase
    model_response: str = ""
    model_output: Any = None  # Runtime type is ModelOutput | None; uses Any to avoid circular import

    # reward, done: inherited from _StepBase
    mc_return: float = 0.0

    # Per-token or scalar advantages
    advantage: list[float] | float | None = None

    # weight version at time of generation (for async training staleness tracking)
    weight_version: int | None = None

    @property
    def info(self) -> dict:
        """Alias for metadata. Auto-initializes to {} if None so mutation works."""
        if self.metadata is None:
            self.metadata = {}
        return self.metadata

    @info.setter
    def info(self, value: dict) -> None:
        self.metadata = value

    def model_post_init(self, __context: Any) -> None:
        self.chat_completions = deepcopy(self.chat_completions)
        if self.model_output is None:
            return
        # backfill fields like prompt_ids, response_ids, logprobs, etc.
        if len(self.prompt_ids) == 0 and self.model_output.prompt_ids is not None:
            self.prompt_ids = self.model_output.prompt_ids
        if len(self.response_ids) == 0 and self.model_output.completion_ids is not None:
            self.response_ids = self.model_output.completion_ids
        if len(self.logprobs) == 0 and self.model_output.logprobs is not None:
            self.logprobs = self.model_output.logprobs
        if self.routing_matrices is None and getattr(self.model_output, "routing_matrices", None) is not None:
            self.routing_matrices = self.model_output.routing_matrices
        if self.weight_version is None and hasattr(self.model_output, "weight_version"):
            self.weight_version = self.model_output.weight_version

        # check that the lengths would match up
        if len(self.logprobs) > 0:
            if len(self.response_ids) != len(self.logprobs):
                raise ValueError(
                    f"length mismatch between response_ids and logprobs, "
                    f"got {len(self.response_ids)}, {len(self.logprobs)}"
                )

    def to_dict(self) -> dict:
        # Helper function to recursively convert ToolCall and ToolOutput objects to dicts
        def _serialize_value(value):
            if isinstance(value, ToolCall | ToolOutput):
                return value.to_dict()
            elif isinstance(value, list):
                return [_serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            else:
                return value

        return {
            "prompt_ids": self.prompt_ids,
            "response_ids": self.response_ids,
            "logprobs": self.logprobs,
            "routing_matrices": self.routing_matrices,
            "chat_completions": _serialize_value(self.chat_completions),
            "observation": self.observation,
            "thought": self.thought,
            "action": self.action.action if isinstance(self.action, Action) else self.action,
            "model_response": self.model_response,
            "model_output": self.model_output.to_dict() if self.model_output is not None else None,
            "info": self.info,
            "reward": self.reward,
            "done": self.done,
            "mc_return": self.mc_return,
            "advantage": self.advantage,
            "weight_version": self.weight_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Step:
        return cls(
            prompt_ids=data["prompt_ids"],
            response_ids=data["response_ids"],
            logprobs=data["logprobs"],
            routing_matrices=data.get("routing_matrices"),
            chat_completions=data["chat_completions"],
            observation=data["observation"],
            thought=data["thought"],
            action=data["action"],
            model_response=data["model_response"],
            model_output=ModelOutput.from_dict(data["model_output"])
            if data.get("model_output", None) is not None
            else None,
            metadata=data.get("info", data.get("metadata", {})),
            reward=data["reward"],
            done=data["done"],
            mc_return=data["mc_return"],
            advantage=data.get("advantage", 0.0),
            weight_version=data.get("weight_version"),
        )

    @classmethod
    def from_model_output(
        cls, model_output: ModelOutput, messages: list[dict] | None = None, action: Any | None = None
    ) -> Step:
        return cls(
            prompt_ids=model_output.prompt_ids or [],
            response_ids=model_output.completion_ids or [],
            logprobs=model_output.logprobs or [],
            routing_matrices=getattr(model_output, "routing_matrices", None),
            chat_completions=(messages or [])
            + [{"role": "assistant", "content": model_output.content, "reasoning": model_output.reasoning}],
            thought=model_output.reasoning or "",
            action=action,
            model_response=model_output.content or "",
            model_output=model_output,
            weight_version=model_output.weight_version,
        )


class Action(BaseModel):
    action: Any = None


_DEFAULT_TRAJ_NAME = "default_traj_name"


class Trajectory(_TrajectoryBase):
    """Training trajectory extending the canonical Trajectory with core defaults."""

    name: str = _DEFAULT_TRAJ_NAME  # Override canonical default for core compat
    steps: list[Step] = Field(default_factory=list)  # Narrow type to training Step

    @property
    def info(self) -> dict:
        """Alias for metadata. Auto-initializes to {} if None so mutation works."""
        if self.metadata is None:
            self.metadata = {}
        return self.metadata

    @info.setter
    def info(self, value: dict) -> None:
        self.metadata = value

    def to_dict(self):
        # Remove large/non-serializable payloads (e.g., images) from task
        def _sanitize_task(task_obj):
            if isinstance(task_obj, dict):
                cleaned = {k: v for k, v in task_obj.items() if k not in ("image", "images")}
                return cleaned
            return task_obj

        return {
            "uid": self.uid,
            "name": self.name,
            "task": _sanitize_task(self.task),
            "steps": [step.to_dict() for step in self.steps],
            "reward": float(self.reward) if self.reward is not None else None,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Trajectory:
        """Create Trajectory from dictionary, properly deserializing Step objects."""
        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            name=data["name"],
            task=data["task"],
            steps=[Step.from_dict(step_data) for step_data in data.get("steps", [])],
            reward=data["reward"],
            metadata=data.get("info", data.get("metadata", {})),
        )

    def is_cumulative(self) -> bool:
        """
        Returns True if for every step after the first, its chat_completions is an exact superset
        of the previous step's chat_completions (i.e., the previous chat_completions is a prefix).
        """
        prev = None
        for step in self.steps:
            if prev is not None:
                prev_cc = prev.chat_completions
                curr_cc = step.chat_completions
                if not (len(curr_cc) >= len(prev_cc) and curr_cc[: len(prev_cc)] == prev_cc):
                    return False
            prev = step
        return True


class Episode(_EpisodeBase):
    """Training episode extending the canonical Episode."""

    trajectories: list[Trajectory] = Field(default_factory=list)  # Narrow type

    @property
    def info(self) -> dict:
        """Alias for metadata. Auto-initializes to {} if None."""
        return self.metadata

    @info.setter
    def info(self, value: dict) -> None:
        self.metadata = value

    def to_dict(self):
        # Remove large/non-serializable payloads (e.g., images) from task
        def _sanitize_task(task_obj):
            if isinstance(task_obj, dict):
                cleaned = {k: v for k, v in task_obj.items() if k not in ("image", "images")}
                return cleaned
            return task_obj

        return {
            "id": self.id,
            "task": _sanitize_task(self.task),
            "termination_reason": self.termination_reason.value if self.termination_reason is not None else None,
            "is_correct": bool(self.is_correct),
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
            "metrics": self.metrics,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Episode:
        """Create Episode from dictionary, properly deserializing Trajectory objects."""

        return cls(
            id=data["id"],
            task=data["task"],
            termination_reason=data.get("termination_reason", "unknown"),
            is_correct=data["is_correct"],
            trajectories=[Trajectory.from_dict(trajectory_data) for trajectory_data in data["trajectories"]],
            metrics=data.get("metrics", {}),
            metadata=data.get("info", data.get("metadata", {})),
        )

    @property
    def task_id(self) -> str:
        return self.id.split(":")[0]

    @property
    def rollout_idx(self) -> str:
        return self.id.split(":")[1]


class TrajectoryGroup(BaseModel):
    """
    A group of trajectories for advantage computation.

    Unlike Episode (which represents raw rollout data), TrajectoryGroup is specifically
    structured for advantage computation. All trajectories in a group will have their
    rewards compared to compute advantages (e.g., via GRPO).

    Attributes:
        trajectories: List of trajectories to compare for advantage computation
        group_id: Optional identifier for the group (e.g., "task1:agent_0")
        metadata: List of metadata for each trajectory in the group
    """

    trajectories: list[Trajectory]
    group_id: str = ""
    metadata: list[dict] = Field(default_factory=list)
    weight_version: int = 0

    @property
    def group_role(self) -> str:
        return self.group_id.split(":")[1] if ":" in self.group_id[:-1] else "all_groups"

    @property
    def task_id(self) -> str:
        return self.group_id.split(":")[0]


class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Converts agent's internal state into a list of OAI chat completions."""
        return []

    @property
    def trajectory(self) -> Trajectory:
        """Converts agent's internal state into a Trajectory object."""
        return Trajectory()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.

        Args:
            observation (Any): The observation after stepping through environment.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended due to termination.
            info (dict): Additional metadata from the environment.
        """
        raise NotImplementedError("Subclasses must implement this method if using AgentExecutionEngine")

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state after the model generates a response.

        Args:
            response (str): The response from the model.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement this method if using AgentExecutionEngine")

    @abstractmethod
    def reset(self):
        """
        Resets the agent's internal state, typically called at the beginning of a new episode.

        This function should clear any stored history or state information necessary
        for a fresh interaction.

        Returns:
            None
        """
        return

    def get_current_state(self) -> Step | None:
        """
        Returns the agent's current state as a dictionary.

        This method provides access to the agent's internal state at the current step,
        which can be useful for debugging, logging, or state management.

        Returns:
            Step: The agent's current state.
        """
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]
