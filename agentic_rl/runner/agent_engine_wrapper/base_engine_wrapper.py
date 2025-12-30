#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Callable

from agentic_rl.base.utils.checker import validate_params
from agentic_rl.runner.agent_engine_wrapper.base import Trajectory


class BaseEngineWrapper(ABC):
    """
    Abstract base class for an engine wrapper that manages agent execution.

    Attributes:
        agent_name (str): The name of the agent.
        tokenizer: Tokenizer object for processing input.
        sampling_params (dict): Parameters for sampling during agent execution.
        max_prompt_length (int): Maximum length of the prompt, must be between 0 and 128K (inclusive).
        max_response_length (int): Maximum length of the response, must be between 1 and 8K (inclusive).
        n_parallel_agents (int): Number of agents to run in parallel, must be between 1 and 64 (inclusive).
        max_steps (int): Maximum number of steps for agent execution, must be between 1 and 10 (inclusive).
    """

    @validate_params(
        agent_name=dict(
            validator=lambda x: isinstance(x, str) and x.isidentifier(),
            message="agent_name must be a non-empty valid Python identifier",
        ),
        sampling_params=dict(
            validator=lambda x: x is None or (isinstance(x, dict) and all(isinstance(key, str) for key in x.keys())),
            message="sampling_params must be a dictionary or None and all keys must be strings",
        ),
        max_prompt_length=dict(
            validator=lambda x: isinstance(x, int) and 1 <= x <= 128 * 1024,
            message="max_prompt_length must be an integer between [1, 128K]",
        ),
        max_response_length=dict(
            validator=lambda x: isinstance(x, int) and 1 <= x <= 8 * 1024,
            message="max_response_length must be an integer between [1, 8K]",
        ),
        n_parallel_agents=dict(
            validator=lambda x: isinstance(x, int) and 1 <= x <= 64,
            message="n_parallel_agents must be an integer between [1, 64]",
        ),
        max_steps=dict(
            validator=lambda x: isinstance(x, int) and 1 <= x <= 10,
            message="max_steps must be an integer between [1, 10]",
        ),
    )
    def __init__(
            self,
            agent_name: str,
            tokenizer: Any,
            sampling_params: Dict[str, Any],
            max_prompt_length=128 * 1024,
            max_response_length=8 * 1024,
            n_parallel_agents=8,
            max_steps=5
    ):
        if tokenizer is not None:
            try:
                from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
            except ImportError as e:
                raise ImportError("Failed to import required module: transformers") from e
            if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                raise TypeError(
                    f"Expected a PreTrainedTokenizer or PreTrainedTokenizerFast, "f"got {type(tokenizer).__name__}")
        else:
            raise ValueError("tokenizer must be provided")

        self.agent_name = agent_name
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.n_parallel_agents = n_parallel_agents
        self.max_steps = max_steps
        self.completions = None

    @abstractmethod
    def initialize(self):
        """
        Perform some necessary initialize procedure for agent engine.
        """
        raise NotImplementedError("This method must be implemented by subclass")

    @abstractmethod
    def generate_agent_trajectories_async(self, tasks: List[dict]) -> List[Trajectory]:
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        Args:
            tasks: List of tasks to be executed by the agent.


        Returns:
            List[Trajectory]: The generated trajectories.

        Implementation note:
            This abstract method defines the interface for trajectory generation.
        """
        raise NotImplementedError("This method must be implemented by subclass")
