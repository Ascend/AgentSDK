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

from typing import List, Optional

import ray
from transformers import AutoTokenizer

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import TrajectoryChecker, validate_params
from agentic_rl.base.utils.file_utils import FileCheck
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from agentic_rl.runner.agent_engine_wrapper.base import Trajectory
from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import BaseEngineWrapper

logger = Loggers(__name__)


@ray.remote
class RunnerWorker:
    """
    The RunnerWorker class is used to manage the execution of parallel agents, responsible for initializing the
    execution environment and asynchronously generating agent trajectories.
    """

    @validate_params(
        sampling_params=dict(
            validator=lambda x: x is None or isinstance(x, dict), message="sampling_params must be a dictionary"
        ),
        max_prompt_length=dict(
            validator=lambda x: isinstance(x, int) and x > 0, message="max_prompt_length must be an positive integer"
        ),
        max_model_len=dict(
            validator=lambda x: isinstance(x, int) and x > 0, message="max_model_len must be an positive integer"
        ),
        n_parallel_agents=dict(
            validator=lambda x: isinstance(x, int) and 0 < x <= 100,
            message="n_parallel_agents must be in range [1, 100]",
        ),
        servers=dict(
            validator=lambda x: isinstance(x, list) and len(x) > 0,
            message="servers must be a non-empty list",
        ),
        addresses=dict(
            validator=lambda x: isinstance(x, list) and len(x) > 0,
            message="addresses must be a non-empty list",
        ),
        agentic_rl_config=dict(
            validator=lambda x: x is None or isinstance(x, AgenticRLConfig),
            message="agentic_rl_config must be an AgenticRLConfig",
        ),
    )
    def __init__(
            self,
            tokenizer_name_or_path,
            sampling_params=None,
            max_prompt_length=8192,
            max_model_len=16384,
            n_parallel_agents=8,
            agent_engine_wrapper_path: Optional[str] = None,
            servers=None,
            addresses=None,
            agentic_rl_config=None,
    ):
        """
        Initialize RunnerWorker.

        Args:
            tokenizer_name_or_path (str): The name or path of the tokenizer.
            sampling_params (dict): Sampling parameters.
            max_prompt_length (int): Maximum prompt length.
            max_model_len (int): Maximum length of the model.
            n_parallel_agents (int): Number of parallel agents.
            agent_engine_wrapper_path (str): Path to the agent engine wrapper implementation.
            servers: Inference instance.
            addresses: Inference server addresses.
            agentic_rl_config (dict): Agent environment configuration.
        """
        FileCheck.check_data_path_is_valid(tokenizer_name_or_path)
        FileCheck.check_data_path_is_valid(agent_engine_wrapper_path)
        self.agentic_rl_config = agentic_rl_config

        # Load tokenizer with exception handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, local_files_only=True, weights_only=True
            )
        except OSError as e:
            logger.error(f"Failed to load tokenizer from '{tokenizer_name_or_path}': {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Invalid tokenizer configuration at '{tokenizer_name_or_path}': {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading tokenizer from '{tokenizer_name_or_path}': {str(e)}")
            raise

        from agentic_rl.base.utils.class_loader import load_subclasses_from_file

        # Load engine wrapper class with exception handling
        try:
            engine_wrapper_class = load_subclasses_from_file(agent_engine_wrapper_path, BaseEngineWrapper)
        except ImportError as e:
            logger.error(f"Failed to load engine wrapper class from '{agent_engine_wrapper_path}': {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading engine wrapper from '{agent_engine_wrapper_path}': {str(e)}")
            raise

        # Initialize engine wrapper with exception handling
        try:
            def get_completion(servers):
                async def _internal_func(*args, **kwargs):
                    return await servers.completions.remote(*args, **kwargs)
                return _internal_func

            self.agent_executor_wrapper = engine_wrapper_class(
                agent_name=self.agentic_rl_config.agent_name,
                tokenizer=self.tokenizer,
                sampling_params=sampling_params,
                max_prompt_length=max_prompt_length,
                max_response_length=max_model_len - max_prompt_length,
                n_parallel_agents=n_parallel_agents,
                max_steps=self.agentic_rl_config.max_steps
            )
            self.agent_executor_wrapper.completions = [get_completion(server) for server in servers]
            self.agent_executor_wrapper.server_addresses = addresses
            self.agent_executor_wrapper.initialize()
        except TypeError as e:
            logger.error(f"Invalid arguments for engine wrapper initialization: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Invalid configuration for engine wrapper: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during engine wrapper initialization: {str(e)}")
            raise

    @staticmethod
    def _validate_trajectory_params(tasks):
        """Validate parameters for trajectory generation."""
        if not isinstance(tasks, list):
            raise TypeError("tasks must be a list")

        if not tasks:
            raise ValueError("tasks list cannot be empty")

        for task in tasks:
            if not isinstance(task, dict):
                raise TypeError("task must be a dictionary")

    def generate_agent_trajectories_async(self, tasks: List[dict]):
        """
        Asynchronously generate agent trajectories.

        Args:
            tasks (list): List of tasks.

        Returns:
            The result of agent trajectory generation.
        """
        self._validate_trajectory_params(tasks)

        # Generate trajectories with exception handling
        try:
            trajectories = self.agent_executor_wrapper.generate_agent_trajectories_async(tasks)
            if not isinstance(trajectories, list) or not all((isinstance(traj, Trajectory) for traj in trajectories)):
                raise TypeError("Trajectories must be a list of Trajectory objects")
            for traj in trajectories:
                TrajectoryChecker.validate_param({"prompt_tokens": traj.prompt_tokens,
                                                  "response_tokens": traj.response_tokens,
                                                  "response_masks": traj.response_masks,
                                                  "idx": traj.idx,
                                                  "trajectory_reward": traj.trajectory_reward,
                                                  "chat_completions": traj.chat_completions,
                                                  "metrics": traj.metrics})
            return trajectories
        except RuntimeError as e:
            logger.error(f"Agent execution failed for {len(tasks)} tasks: {str(e)}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during trajectory generation for {len(tasks)} tasks: {str(e)}"
            )
            raise