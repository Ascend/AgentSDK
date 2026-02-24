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
import logging
import torch
import random
from typing import List, Dict, Any
import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from agentic_rl import BaseEngineWrapper, Trajectory
from agentic_rl.base.utils.checker import ValidatorReturnTypeError

logger = logging.getLogger(__name__)


class MockEngineWrapper(BaseEngineWrapper):
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
        super().__init__(agent_name, tokenizer, sampling_params, max_prompt_length,
                         max_response_length, n_parallel_agents, max_steps)

    def initialize(self):
        pass

    def generate_agent_trajectories_async(
            self, tasks
    ) -> List[Trajectory]:

        if not isinstance(tasks, list):
            raise TypeError("tasks must be a list")

        if not tasks:
            raise ValueError("tasks list cannot be empty")

        for task in tasks:
            if not isinstance(task, dict):
                raise TypeError("tasks must be a list of dictionary")

        logger.info(f"Generating trajectories for {len(tasks)} tasks")

        mock_results = {
            "prompt_tokens": torch.tensor([101, 200, 300, 400], dtype=torch.long),
            "response_tokens": torch.tensor([500, 600, 700], dtype=torch.long),
            "response_masks": torch.tensor([1, 1, 1], dtype=torch.long),
            "trajectory_reward": random.uniform(-1, 1),
            "idx": random.randint(0, 9999),
            "chat_completions": [
                {"role": "assistant", "content": "This is a mock response."}
            ],
            "metrics": {
                "steps": 3,
                "reward_time": round(random.uniform(0.001, 0.01), 4),
                "env_time": round(random.uniform(0.01, 0.05), 4),
                "llm_time": round(random.uniform(0.01, 0.02), 4),
                "total_time": 0.08,
            },
        }
        trajectories = [Trajectory(**mock_results) for _ in tasks]
        return trajectories


tasks_input = [
    {
        "id": "00",
        "question": "Please describe Shanghai.",
        "ground_truth": "Shanghai is a big city."
    },
    {
        "id": "01",
        "question": "Please describe Hangzhou.",
        "ground_truth": "Hangzhou is a big city."
    },
    {
        "id": "02",
        "question": "Please describe Shenzhen.",
        "ground_truth": "Shenzhen is a big city."
    }
]

_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
_tokenizer.pre_tokenizer = Whitespace()
tokenizer_demo = PreTrainedTokenizerFast(
    tokenizer_object=_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

sampling_params_input = {
    "temperature": 0.9,
    "top_k": 50,
    "logprobs": 1,
    "max_tokens": 8192,
    "top_p": 0.9,
    "min_p": 0.01,
    "detokenize": False
}


class TestMockEngineWrapper:

    def test_init_agent_name_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name="", tokenizer=tokenizer_demo,
                              sampling_params=sampling_params_input)
        assert "agent_name must be a non-empty valid Python identifier" in str(exc_info.value)

    def test_init_tokenizer_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name="agent", tokenizer=None,
                              sampling_params=sampling_params_input)
        assert "tokenizer must be provided" in str(exc_info.value)

    def test_init_sampling_params_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                              sampling_params=["", "123"])
        assert "sampling_params must be a dictionary or None and all keys must be strings" in str(exc_info.value)

    def test_init_agent_name_type_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name=123, tokenizer=tokenizer_demo,
                              sampling_params=sampling_params_input)
        assert "agent_name must be a non-empty valid Python identifier" in str(exc_info.value)

    def test_init_sampling_params_type_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                              sampling_params="sampling_params_input")
        assert "sampling_params must be a dictionary or None and all keys must be strings" in str(exc_info.value)

    def test_init_sampling_params_content_type_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                              sampling_params={123: 0.9})
        assert "sampling_params must be a dictionary or None and all keys must be strings" in str(exc_info.value)

    def test_generate_agent_trajectories_async_success(self):
        mock_engine = MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                                        sampling_params=sampling_params_input)
        trajectories = mock_engine.generate_agent_trajectories_async(tasks_input)
        assert isinstance(trajectories, List)
        assert all(isinstance(item, Trajectory) for item in trajectories)

    def test_generate_agent_trajectories_async_empty_task_fail(self):
        mock_engine = MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                                        sampling_params=sampling_params_input)
        with pytest.raises(ValueError) as exc_info:
            mock_engine.generate_agent_trajectories_async([])
        assert "tasks list cannot be empty" in str(exc_info.value)

    def test_generate_agent_trajectories_async_task_type_fail(self):
        mock_engine = MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                                        sampling_params=sampling_params_input)
        with pytest.raises(TypeError) as exc_info:
            mock_engine.generate_agent_trajectories_async("tasks_input")
        assert "tasks must be a list" in str(exc_info.value)

    def test_generate_agent_trajectories_async_task_content_type_fail(self):
        mock_engine = MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                                        sampling_params=sampling_params_input)
        with pytest.raises(TypeError) as exc_info:
            mock_engine.generate_agent_trajectories_async(tasks=[[]])
        assert "tasks must be a list of dictionary" in str(exc_info.value)

    def test_generate_agent_trajectories_async_max_prompt_length_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                              sampling_params=sampling_params_input, max_prompt_length=256 * 1024)
        assert "max_prompt_length must be an integer between [1, 128K]" in str(exc_info.value)

    def test_generate_agent_trajectories_async_max_response_length_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                              sampling_params=sampling_params_input, max_response_length="64 * 1024")
        assert "max_response_length must be an integer between [1, 64K]" in str(exc_info.value)

    def test_generate_agent_trajectories_async_n_parallel_agents_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                              sampling_params=sampling_params_input, n_parallel_agents=[128])
        assert "n_parallel_agents must be an integer between [1, 64]" in str(exc_info.value)

    def test_generate_agent_trajectories_async_max_steps_fail(self):
        with pytest.raises(ValueError) as exc_info:
            MockEngineWrapper(agent_name="agent", tokenizer=tokenizer_demo,
                              sampling_params=sampling_params_input, max_steps={})
        assert "max_steps must be an integer between [1, 100]" in str(exc_info.value)
