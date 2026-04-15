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
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any


@pytest.fixture(autouse=True, scope="function")
def mock_ray():
    mock_ray = MagicMock()
    mock_ray.remote = lambda cls: cls
    
    with patch.dict('sys.modules', {'ray': mock_ray}):
        yield mock_ray


@dataclass
class MockStep:
    chat_completions: list = field(default_factory=list)
    reward: float = 0.0
    mc_return: float = 0.0
    done: bool = False
    step_id: int = 0

    def to_dict(self):
        return {
            "chat_completions": self.chat_completions,
            "reward": self.reward,
            "mc_return": float(self.mc_return),
            "done": self.done,
            "step_id": self.step_id,
        }


@dataclass
class MockTrajectory:
    task: Any = None
    steps: list = field(default_factory=list)
    reward: float = 0.0
    toolcall_reward: float = 0.0
    res_reward: float = 0.0
    prompt_id: int = 0
    data_id: str = None
    training_id: str = None
    epoch_id: int = 0
    iteration_id: int = 0
    sample_id: int = 0
    trajectory_id: int = 0
    application_id: str = ""
    termination_reason: str = "unknown"

    def to_dict(self):
        return {
            "task": self.task,
            "steps": [step.to_dict() for step in self.steps],
            "reward": float(self.reward),
            "prompt_id": self.prompt_id,
            "data_id": self.data_id,
            "training_id": self.training_id,
            "epoch_id": self.epoch_id,
            "iteration_id": self.iteration_id,
            "sample_id": self.sample_id,
            "trajectory_id": self.trajectory_id,
            "application_id": self.application_id,
            "termination_reason": self.termination_reason,
        }


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(mock_ray, mock_requests, mock_aiohttp, mock_pydantic, mock_torch, mock_fastapi, mock_sse_starlette):
    """Mock all external dependencies for episode tests."""
    with (
        patch("agentic_rl.memory.episode.episode.Trajectory", MockTrajectory),
    ):
        yield {}


class TestTerminationReason:
    """Tests for TerminationReason enum."""

    def test_termination_reason_values(self, mock_dependencies):
        """Test that TerminationReason enum has expected values."""
        from agentic_rl.memory.episode.episode import TerminationReason
        
        assert TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED.value == "max_prompt_length_exceeded"
        assert TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED.value == "max_response_length_exceeded"
        assert TerminationReason.ENV_DONE.value == "env_done"
        assert TerminationReason.MAX_TURNS_EXCEEDED.value == "max_turns_exceeded"
        assert TerminationReason.TIMEOUT.value == "timeout"

    def test_termination_reason_enum_count(self, mock_dependencies):
        """Test that TerminationReason has correct number of values."""
        from agentic_rl.memory.episode.episode import TerminationReason
        
        assert len(list(TerminationReason)) == 5


class TestEpisode:
    """Tests for Episode class."""

    def test_episode_initialization(self, mock_dependencies):
        """Test Episode initialization with default values."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode(episode_id="test-episode-123")
        
        assert episode.episode_id == "test-episode-123"
        assert episode.task is None
        assert episode.termination_reason is None
        assert episode.trajectories == []

    def test_episode_initialization_without_id(self, mock_dependencies):
        """Test Episode initialization without episode_id."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode()
        
        assert episode.episode_id is None
        assert episode.task is None
        assert episode.termination_reason is None
        assert episode.trajectories == []

    def test_set_task(self, mock_dependencies):
        """Test setting task for an episode."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode(episode_id="test-episode-456")
        task = {"problem": "solve x + 1 = 2", "ground_truth": "x = 1"}
        
        episode.set_task(task)
        
        assert episode.task == task

    def test_set_termination_reason(self, mock_dependencies):
        """Test setting termination reason for an episode."""
        from agentic_rl.memory.episode.episode import Episode, TerminationReason
        
        episode = Episode(episode_id="test-episode-789")
        
        episode.set_termination_reason(TerminationReason.ENV_DONE)
        
        assert episode.termination_reason == TerminationReason.ENV_DONE

    def test_add_trajectory(self, mock_dependencies):
        """Test adding a trajectory to an episode."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode(episode_id="test-episode-101")
        trajectory = MockTrajectory(task="test task")
        
        episode.add_trajectory("agent_1", trajectory)
        
        assert len(episode.trajectories) == 1
        assert episode.trajectories[0] == ("agent_1", trajectory)

    def test_add_multiple_trajectories(self, mock_dependencies):
        """Test adding multiple trajectories to an episode."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode(episode_id="test-episode-102")
        trajectory1 = MockTrajectory(task="task 1")
        trajectory2 = MockTrajectory(task="task 2")
        trajectory3 = MockTrajectory(task="task 3")
        
        episode.add_trajectory("agent_1", trajectory1)
        episode.add_trajectory("agent_2", trajectory2)
        episode.add_trajectory("agent_1", trajectory3)
        
        assert len(episode.trajectories) == 3
        assert episode.trajectories[0] == ("agent_1", trajectory1)
        assert episode.trajectories[1] == ("agent_2", trajectory2)
        assert episode.trajectories[2] == ("agent_1", trajectory3)

    def test_get_trajectory_by_agent_name(self, mock_dependencies):
        """Test retrieving trajectories by agent name."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode(episode_id="test-episode-103")
        trajectory1 = MockTrajectory(task="task 1")
        trajectory2 = MockTrajectory(task="task 2")
        trajectory3 = MockTrajectory(task="task 3")
        
        episode.add_trajectory("agent_1", trajectory1)
        episode.add_trajectory("agent_2", trajectory2)
        episode.add_trajectory("agent_1", trajectory3)
        
        agent1_trajectories = episode.get_trajectory_by_agent_name("agent_1")
        agent2_trajectories = episode.get_trajectory_by_agent_name("agent_2")
        agent3_trajectories = episode.get_trajectory_by_agent_name("agent_3")
        
        assert len(agent1_trajectories) == 2
        assert agent1_trajectories[0] == trajectory1
        assert agent1_trajectories[1] == trajectory3
        assert len(agent2_trajectories) == 1
        assert agent2_trajectories[0] == trajectory2
        assert len(agent3_trajectories) == 0

    def test_remove_trajectory_by_agent_name(self, mock_dependencies):
        """Test removing trajectories by agent name."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode(episode_id="test-episode-104")
        trajectory1 = MockTrajectory(task="task 1")
        trajectory2 = MockTrajectory(task="task 2")
        trajectory3 = MockTrajectory(task="task 3")
        
        episode.add_trajectory("agent_1", trajectory1)
        episode.add_trajectory("agent_2", trajectory2)
        episode.add_trajectory("agent_1", trajectory3)
        
        episode.remove_trajectory_by_agent_name("agent_1")
        
        assert len(episode.trajectories) == 1
        assert episode.trajectories[0] == trajectory2

    def test_remove_trajectory_nonexistent_agent(self, mock_dependencies):
        """Test removing trajectories for non-existent agent."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode(episode_id="test-episode-105")
        trajectory1 = MockTrajectory(task="task 1")
        
        episode.add_trajectory("agent_1", trajectory1)
        episode.remove_trajectory_by_agent_name("agent_nonexistent")
        
        assert len(episode.trajectories) == 1
        assert episode.trajectories[0] == trajectory1

    def test_to_dict(self, mock_dependencies):
        """Test converting episode to dictionary."""
        from agentic_rl.memory.episode.episode import Episode, TerminationReason
        
        episode = Episode(episode_id="test-episode-106")
        episode.set_task({"problem": "test problem"})
        episode.set_termination_reason(TerminationReason.ENV_DONE)
        
        trajectory = MockTrajectory(task="test task", reward=1.0)
        episode.add_trajectory("agent_1", trajectory)
        
        result = episode.to_dict()
        
        assert result["episode_id"] == "test-episode-106"
        assert result["task"] == {"problem": "test problem"}
        assert result["termination_reason"] == TerminationReason.ENV_DONE
        assert len(result["trajectories"]) == 1
        assert result["trajectories"][0][0] == "agent_1"
        assert result["trajectories"][0][1]["task"] == "test task"

    def test_to_dict_empty_episode(self, mock_dependencies):
        """Test converting empty episode to dictionary."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode(episode_id="test-episode-107")
        result = episode.to_dict()
        
        assert result["episode_id"] == "test-episode-107"
        assert result["task"] is None
        assert result["termination_reason"] is None
        assert result["trajectories"] == []

    def test_episode_with_complex_task(self, mock_dependencies):
        """Test episode with complex task structure."""
        from agentic_rl.memory.episode.episode import Episode
        
        episode = Episode(episode_id="test-episode-108")
        complex_task = {
            "problem": "complex problem",
            "ground_truth": "answer",
            "metadata": {
                "difficulty": "hard",
                "category": "math"
            },
            "constraints": ["constraint1", "constraint2"]
        }
        
        episode.set_task(complex_task)
        
        assert episode.task == complex_task
        assert episode.task["metadata"]["difficulty"] == "hard"

    def test_episode_with_multiple_termination_reasons(self, mock_dependencies):
        """Test setting different termination reasons."""
        from agentic_rl.memory.episode.episode import Episode, TerminationReason
        
        episode = Episode(episode_id="test-episode-109")
        
        for reason in TerminationReason:
            episode.set_termination_reason(reason)
            assert episode.termination_reason == reason
