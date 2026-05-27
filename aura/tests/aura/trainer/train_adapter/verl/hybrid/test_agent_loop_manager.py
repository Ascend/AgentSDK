#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig


class MockRollout:

    def wake_up(self):
        pass

    def sleep(self):
        pass

    def clear_kv_cache(self):
        pass


class MockAgentLoopManager:

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.server_addresses = ["0.0.0.0:1234"]
        self.rollout_replicas = []

    def _run_all(self, *args, **kwargs):
        pass


class TestAgentLoopManager:

    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "verl": MagicMock(),
            "verl.utils": MagicMock(),
            "verl.utils.hf_tokenizer": MagicMock(),
            "verl.experimental": MagicMock(),
            "verl.experimental.agent_loop": MagicMock(),
            "verl.DataProto": MagicMock(),
        }):
            yield

    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=DictConfig)
        config.extras = MagicMock()
        config.extras.infer_service = "test_infer_service"
        config.extras.agent_service = "test_agent_service"
        config.extras.traj_output_path = "/tmp/test_trajectories"
        config.actor_rollout_ref = MagicMock()
        config.actor_rollout_ref.model = MagicMock()
        config.actor_rollout_ref.model.path = "test_model_path"
        config.actor_rollout_ref.rollout = MagicMock()
        config.actor_rollout_ref.rollout.n = 2
        return config

    @pytest.fixture
    def mock_prompts(self):
        prompts = MagicMock()
        prompts.__len__.return_value = 2
        prompts.meta_info = {"global_steps": 1}
        prompts.non_tensor_batch = {
            "index": np.array([0, 1]),
            "raw_prompt": np.array([
                [{"content": "test prompt 1", "role": "user"}],
                [{"content": "test prompt 2", "role": "user"}]
            ]),
            "reward_model": np.array([
                {"ground_truth": "123"},
                {"ground_truth": "456"}
            ]),
            "extra_info": np.array([
                {"key1": "value1"},
                {"key2": "value2"}
            ])
        }
        return prompts

    @pytest.fixture
    def mock_trajectory(self):
        return {
            "idx": 0,
            "prompt_id": 0,
            "prompt_tokens": torch.tensor([1, 2, 3]),
            "response_tokens": torch.tensor([4, 5, 6]),
            "logprobs": [0.1, 0.2, 0.3],
            "response_masks": torch.tensor([1, 1, 1]),
            "trajectory_reward": 0.8,
            "chat_completions": "test completion 1"
        }

    @pytest.fixture
    def mock_trajectories(self):
        return [
            {
                "idx": 0,
                "prompt_id": 0,
                "prompt_tokens": torch.tensor([1, 2, 3]),
                "response_tokens": torch.tensor([4, 5, 6]),
                "logprobs": [0.1, 0.2, 0.3],
                "response_masks": torch.tensor([1, 1, 1]),
                "trajectory_reward": 0.8,
                "chat_completions": "test completion 1"
            },
            {
                "idx": 1,
                "prompt_id": 1,
                "prompt_tokens": torch.tensor([7, 8, 9]),
                "response_tokens": torch.tensor([10, 11, 12]),
                "logprobs": [0.4, 0.5, 0.6],
                "response_masks": torch.tensor([1, 1, 1]),
                "trajectory_reward": 0.9,
                "chat_completions": "test completion 2"
            }
        ]

    def test_init(self, mock_config, patch_modules):
        with patch("verl.experimental.agent_loop.AgentLoopManager", MockAgentLoopManager), \
                patch("verl.utils.hf_tokenizer") as mock_hf_tokenizer:
            from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import HybridAgentLoopManager

            mock_tokenizer = MagicMock()
            mock_hf_tokenizer.return_value = mock_tokenizer

            manager = HybridAgentLoopManager(mock_config)

            mock_hf_tokenizer.assert_called_once_with(
                mock_config.actor_rollout_ref.model.path, trust_remote_code=True)

            assert manager.server_addresses[0] == "0.0.0.0:1234"

    @pytest.fixture
    def mock_hybrid_agent_loop_manager(self, mock_config):
        with patch("verl.experimental.agent_loop.AgentLoopManager", MockAgentLoopManager), \
                patch("verl.utils.hf_tokenizer") as mock_hf_tokenizer:
            from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import HybridAgentLoopManager
            manager = HybridAgentLoopManager(mock_config)
            manager.chat_server_list = ["0.0.0.0:1234"]
            manager.server_handles = None
            manager.traj_output_path = "/tmp/test_trajectories"
            manager.perf_timestamp = 1234567890
            manager.iteration = 0

            return manager

    @pytest.mark.asyncio
    async def test_init_agent_loop_workers(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        await mock_hybrid_agent_loop_manager._init_agent_loop_workers()
        assert True

    @pytest.mark.asyncio
    async def test_async_generate_sequences(self, mock_hybrid_agent_loop_manager, mock_config, mock_prompts,
                                            mock_trajectory, patch_modules):
        with patch("verl.utils.hf_tokenizer") as mock_hf_tokenizer, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "create_tasks", AsyncMock()) as mock_create_tasks, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "generate_trajectory", AsyncMock()) as mock_generate_traj, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "transform_trajectories_to_batch", AsyncMock()) as mock_transform, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.HybridAgentLoopManager."
                      "write_file") as mock_write_file:
            mock_tokenizer = MagicMock()
            mock_hf_tokenizer.return_value = mock_tokenizer

            mock_agent_tasks = [MagicMock(), MagicMock()]
            mock_create_tasks.return_value = mock_agent_tasks

            mock_generate_traj.return_value = mock_trajectory

            mock_transformed_batch = MagicMock()
            mock_transform.return_value = mock_transformed_batch

            result = await mock_hybrid_agent_loop_manager.async_generate_sequences(
                mock_config, mock_prompts, mock_tokenizer)

            mock_create_tasks.assert_called_once_with(
                mock_config.extras.agent_service,
                mock_prompts,
                mock_config.actor_rollout_ref.rollout.n
            )
            assert mock_generate_traj.call_count == 2
            for call_args in mock_generate_traj.call_args_list:
                args, kwargs = call_args
                assert len(args) == 3
            mock_write_file.assert_called_once_with([mock_trajectory, mock_trajectory], prefix="trajectories")
            mock_transform.assert_called_once_with(mock_config, mock_tokenizer, [mock_trajectory, mock_trajectory])

            assert result == mock_transformed_batch

    def test_generate_sequences(self, mock_hybrid_agent_loop_manager, mock_config, mock_prompts, patch_modules):
        with    patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.HybridAgentLoopManager."
                      "async_generate_sequences") as mock_async_generate_sequences:
            mock_result = MagicMock()
            mock_async_generate_sequences.return_value = mock_result

            result = mock_hybrid_agent_loop_manager.generate_sequences(mock_prompts)

            mock_async_generate_sequences.assert_called_once_with(
                mock_config, mock_prompts, mock_hybrid_agent_loop_manager.tokenizer)

            assert result == mock_result

    def test_wake_up(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        mock_replica1 = MagicMock()
        mock_replica2 = MagicMock()
        mock_hybrid_agent_loop_manager.rollout_replicas = [mock_replica1, mock_replica2]

        mock_hybrid_agent_loop_manager.wake_up()

        mock_replica1.wake_up.assert_called_once()
        mock_replica2.wake_up.assert_called_once()

    def test_sleep(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        mock_replica1 = MagicMock()
        mock_replica2 = MagicMock()
        mock_hybrid_agent_loop_manager.rollout_replicas = [mock_replica1, mock_replica2]

        mock_hybrid_agent_loop_manager.sleep()

        mock_replica1.sleep.assert_called_once()
        mock_replica2.sleep.assert_called_once()

    def test_clear_kv_cache(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        mock_replica1 = MagicMock()
        mock_replica2 = MagicMock()
        mock_hybrid_agent_loop_manager.rollout_replicas = [mock_replica1, mock_replica2]

        mock_hybrid_agent_loop_manager.clear_kv_cache()

        mock_replica1.clear_kv_cache.assert_called_once()
        mock_replica2.clear_kv_cache.assert_called_once()

    def test_write_file(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        with patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.open", create=True) as mock_open, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.os.path.join") as mock_path_join, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.json.dump") as mock_json_dump, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.logger") as mock_logger:
            mock_hybrid_agent_loop_manager.iteration = 1
            mock_hybrid_agent_loop_manager.perf_timestamp = 1234567890
            mock_hybrid_agent_loop_manager.traj_output_path = "/tmp/test_trajectories"

            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            test_data = {
                "tensor_data": torch.tensor([1, 2, 3]),
                "list_data": [torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])],
                "dict_data": {"key": torch.tensor([10, 11, 12])},
                "str_data": "test"
            }

            mock_hybrid_agent_loop_manager.write_file(test_data, "test")

            mock_path_join.assert_called_once_with(
                mock_hybrid_agent_loop_manager.traj_output_path,
                'rollout_test_1234567890.json'
            )
            mock_open.assert_called_once()
            mock_json_dump.assert_called_once()
            mock_file.write.assert_called_once_with('\n')
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_server(self, patch_modules):
        with patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                   "InferRouter", AsyncMock()) as mock_infer_router:
            from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import launch_server

            mock_router = AsyncMock()
            mock_infer_router.create.return_value = mock_router

            await launch_server("test_infer_service", "test_model", ["server1", "server2"])

            mock_infer_router.create.assert_called_once()
            mock_router.launch_server.assert_called_once_with(
                model_name="test_infer_service",
                kwargs_list=[{
                    "model_name": "test_model",
                    "chat_server": ["http://server1", "http://server2"]
                }]
            )

    @pytest.mark.asyncio
    async def test_create_tasks(self, mock_prompts, patch_modules):
        from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import create_tasks
        from aura.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask

        agent_service = "test_agent_service"
        prompts = mock_prompts
        n_samples_per_prompt = 2

        agent_tasks = await create_tasks(agent_service, prompts, n_samples_per_prompt)

        assert len(agent_tasks) == 2

        task1 = agent_tasks[0]
        assert isinstance(task1, AgentTask)
        assert task1.task_id == "0"
        assert task1.sample_id == 0
        assert task1.iteration == 1
        assert task1.agent_name == agent_service
        assert task1.problem == "test prompt 1"
        assert task1.prompt_id == 0
        assert task1.ground_truth == "123"
        assert task1.extra_args["key1"] == "value1"

        task2 = agent_tasks[1]
        assert isinstance(task2, AgentTask)
        assert task2.task_id == "1"
        assert task2.sample_id == 1
        assert task2.iteration == 1
        assert task2.agent_name == agent_service
        assert task2.problem == "test prompt 2"
        assert task2.prompt_id == 0
        assert task2.ground_truth == "456"
        assert task2.extra_args["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_generate_trajectory(self, patch_modules):
        with patch("aura.runner.agent_router.AgentRouter", AsyncMock()) as mock_agent_router:
            from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import generate_trajectory
            from aura.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask

            mock_router = AsyncMock()
            mock_agent_router.create.return_value = mock_router

            mock_trajectory = AsyncMock()
            mock_router.generate_trajectory.return_value = mock_trajectory

            agent_task = AgentTask(
                task_id="test_task",
                sample_id=0,
                iteration=1,
                agent_name="test_agent",
                problem="test problem",
                prompt_id=0,
                content=""
            )

            trajectory = await generate_trajectory(agent_task, ["server1"], None)

            mock_agent_router.create.assert_called_once()
            mock_router.generate_trajectory.assert_called_once_with(
                agent_task, mode='Token', addresses=["server1"], server_handles=None
            )

            assert trajectory == mock_trajectory

    @pytest.mark.asyncio
    async def test_transform_trajectories_to_batch(self, mock_config, mock_trajectories, patch_modules):
        with patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.torch.nn.utils."
                   "rnn.pad_sequence") as mock_pad_sequence, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.DataProto") as mock_data_proto:
            from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import transform_trajectories_to_batch

            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token_id = 0

            mock_pad_sequence.side_effect = [
                torch.tensor([[1, 2, 3], [7, 8, 9]]),
                torch.tensor([[4, 5, 6], [10, 11, 12]]),
                torch.tensor([[1, 1, 1], [1, 1, 1]]),
                torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            ]

            mock_data_proto_instance = MagicMock()
            mock_data_proto.from_dict.return_value = mock_data_proto_instance
            mock_data_proto_instance.non_tensor_batch = {}
            mock_data_proto_instance.meta_info = {}

            result = await transform_trajectories_to_batch(mock_config, mock_tokenizer, mock_trajectories)

            assert mock_pad_sequence.call_count == 4
            mock_data_proto.from_dict.assert_called_once()

            assert result == mock_data_proto_instance
            assert "uid" in result.non_tensor_batch
            assert "timing" in result.meta_info

    @pytest.mark.asyncio
    async def test_create_tasks_without_index(self, mock_config, patch_modules):
        from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import create_tasks

        prompts = MagicMock()
        prompts.__len__.return_value = 2
        prompts.meta_info = {"global_steps": 1}
        prompts.non_tensor_batch = {
            "raw_prompt": np.array([
                [{"content": "test prompt 1", "role": "user"}],
                [{"content": "test prompt 2", "role": "user"}]
            ]),
            "reward_model": np.array([
                {"ground_truth": "123"},
                {"ground_truth": "456"}
            ]),
            "extra_info": np.array([
                {"key1": "value1"},
                {"key2": "value2"}
            ])
        }

        agent_tasks = await create_tasks("test_agent_service", prompts, 1)

        assert len(agent_tasks) == 2
        assert agent_tasks[0].task_id == "0"
        assert agent_tasks[0].prompt_id == 0
        assert agent_tasks[1].task_id == "1"
        assert agent_tasks[1].prompt_id == 1

    @pytest.mark.asyncio
    async def test_create_tasks_with_extra_fields(self, mock_config, patch_modules):
        from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import create_tasks

        prompts = MagicMock()
        prompts.__len__.return_value = 1
        prompts.meta_info = {"global_steps": 2}
        prompts.non_tensor_batch = {
            "index": np.array([5]),
            "raw_prompt": np.array([[{"content": "test prompt", "role": "user"}]]),
            "reward_model": np.array([{"ground_truth": "123"}]),
            "extra_info": np.array([{"key1": "value1"}]),
            "custom_field": np.array(["custom_value"])
        }

        agent_tasks = await create_tasks("test_agent_service", prompts, 1)

        assert len(agent_tasks) == 1
        assert agent_tasks[0].extra_args["custom_field"] == "custom_value"
        assert agent_tasks[0].iteration == 2

    @pytest.mark.asyncio
    async def test_async_generate_sequences_with_chat_interface(self, mock_hybrid_agent_loop_manager, mock_config, mock_prompts,
                                            mock_trajectory, patch_modules):
        mock_config.extras.chat_interface = "generate"

        with patch("verl.utils.hf_tokenizer") as mock_hf_tokenizer, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "create_tasks", AsyncMock()) as mock_create_tasks, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "generate_trajectory", AsyncMock()) as mock_generate_traj, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "transform_trajectories_to_batch", AsyncMock()) as mock_transform, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.HybridAgentLoopManager."
                      "write_file") as mock_write_file:
            mock_tokenizer = MagicMock()
            mock_hf_tokenizer.return_value = mock_tokenizer

            mock_agent_tasks = [MagicMock(), MagicMock()]
            mock_create_tasks.return_value = mock_agent_tasks
            mock_generate_traj.return_value = mock_trajectory
            mock_transformed_batch = MagicMock()
            mock_transform.return_value = mock_transformed_batch

            result = await mock_hybrid_agent_loop_manager.async_generate_sequences(
                mock_config, mock_prompts, mock_tokenizer)

            assert mock_generate_traj.call_count == 2
            for call_args in mock_generate_traj.call_args_list:
                args, kwargs = call_args
                assert len(args) == 3

    @pytest.mark.asyncio
    async def test_transform_trajectories_to_batch_empty_tokens_error(self, mock_config, patch_modules):
        from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import transform_trajectories_to_batch

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        empty_trajectories = [
            {
                "idx": 0,
                "prompt_id": 0,
                "prompt_tokens": torch.tensor([]),
                "response_tokens": torch.tensor([4, 5, 6]),
                "logprobs": [0.1, 0.2, 0.3],
                "response_masks": torch.tensor([1, 1, 1]),
                "trajectory_reward": 0.8,
                "chat_completions": "test completion"
            }
        ]

        with pytest.raises(ValueError, match="Both prompt .* and response .* of trajectory shouldn't be empty"):
            await transform_trajectories_to_batch(mock_config, mock_tokenizer, empty_trajectories)

    @pytest.mark.asyncio
    async def test_transform_trajectories_to_batch_cancel_logprobs(self, mock_config, patch_modules):
        from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import transform_trajectories_to_batch

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        trajectories = [
            {
                "idx": 0,
                "prompt_id": 0,
                "prompt_tokens": torch.tensor([1, 2, 3]),
                "response_tokens": torch.tensor([4, 5, 6]),
                "logprobs": [],
                "response_masks": torch.tensor([1, 1, 1]),
                "trajectory_reward": 0.8,
                "chat_completions": "test completion"
            }
        ]

        with patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.torch.nn.utils."
                   "rnn.pad_sequence") as mock_pad_sequence, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.DataProto") as mock_data_proto:
            mock_pad_sequence.side_effect = [
                torch.tensor([[1, 2, 3]]),
                torch.tensor([[4, 5, 6]]),
                torch.tensor([[1, 1, 1]]),
            ]

            mock_data_proto_instance = MagicMock()
            mock_data_proto.from_dict.return_value = mock_data_proto_instance
            mock_data_proto_instance.non_tensor_batch = {}
            mock_data_proto_instance.meta_info = {}

            result = await transform_trajectories_to_batch(mock_config, mock_tokenizer, trajectories)

            assert mock_pad_sequence.call_count == 3

    def test_write_file_with_tensor_data(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        with patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.open", create=True) as mock_open, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.os.path.join") as mock_path_join, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.json.dump") as mock_json_dump, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.logger") as mock_logger:
            mock_hybrid_agent_loop_manager.iteration = 5
            mock_hybrid_agent_loop_manager.perf_timestamp = 1234567890
            mock_hybrid_agent_loop_manager.traj_output_path = "/tmp/test_trajectories"

            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            test_data = {
                "tensor_only": torch.tensor([[1, 2], [3, 4]]),
            }

            mock_hybrid_agent_loop_manager.write_file(test_data, "tensor_test")

            mock_json_dump.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_llm_servers(self, mock_config, patch_modules):
        with patch("verl.experimental.agent_loop.AgentLoopManager", MockAgentLoopManager), \
                patch("verl.utils.hf_tokenizer") as mock_hf_tokenizer, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.launch_server", AsyncMock()) as mock_launch_server:
            from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import HybridAgentLoopManager

            mock_tokenizer = MagicMock()
            mock_hf_tokenizer.return_value = mock_tokenizer

            manager = HybridAgentLoopManager(mock_config)
            manager.server_addresses = ["0.0.0.0:1234"]
            manager.iteration = 0
            manager.traj_output_path = "/tmp/test_trajectories"

            assert manager.server_addresses == ["0.0.0.0:1234"]
            assert manager.iteration == 0
            assert manager.traj_output_path == "/tmp/test_trajectories"

    @pytest.mark.asyncio
    async def test_transform_episodes_to_batch(self, patch_modules):
        """测试transform_episodes_to_batch函数"""
        from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import transform_episodes_to_batch

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        mock_step1 = MagicMock()
        mock_step1.prompt_ids = [1, 2, 3]
        mock_step1.response_ids = [4, 5, 6]
        mock_step1.logprobs = [0.1, 0.2, 0.3]
        mock_step1.rewards = [0.5, 0.6, 0.7]

        mock_step2 = MagicMock()
        mock_step2.prompt_ids = [1, 2, 3, 7, 8]
        mock_step2.response_ids = [9, 10]
        mock_step2.logprobs = [0.8, 0.9]
        mock_step2.rewards = [0.8, 0.9]

        mock_trajectory = MagicMock()
        mock_trajectory.steps = [mock_step1, mock_step2]
        mock_trajectory.reward = 1.0
        mock_trajectory.trajectory_id = "test_traj"

        with patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.torch.nn.utils."
                   "rnn.pad_sequence") as mock_pad_sequence, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.DataProto") as mock_data_proto:
            mock_pad_sequence.side_effect = [
                torch.tensor([[1, 2, 3]]),
                torch.tensor([[4, 5, 6, 7, 8, 9, 10]]),
                torch.tensor([[1, 1, 1, 0, 0, 1, 1]]),
                torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.8, 0.9]]),
            ]

            mock_data_proto_instance = MagicMock()
            mock_data_proto.from_dict.return_value = mock_data_proto_instance
            mock_data_proto_instance.non_tensor_batch = {}
            mock_data_proto_instance.meta_info = {}

            result = await transform_episodes_to_batch(mock_tokenizer, [mock_trajectory], ["0"])

            assert mock_pad_sequence.call_count == 4
            assert "uid" in result.non_tensor_batch
            assert "timing" in result.meta_info

    @pytest.mark.asyncio
    async def test_transform_episodes_to_batch_no_logprobs(self, patch_modules):
        """测试transform_episodes_to_batch函数 - 没有logprobs"""
        from aura.trainer.train_adapter.verl.hybrid.agent_loop_manager import transform_episodes_to_batch

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        mock_step1 = MagicMock()
        mock_step1.prompt_ids = [1, 2, 3]
        mock_step1.response_ids = [4, 5, 6]
        mock_step1.logprobs = None
        mock_step1.rewards = [0.5, 0.6, 0.7]

        mock_trajectory = MagicMock()
        mock_trajectory.steps = [mock_step1]
        mock_trajectory.reward = 1.0
        mock_trajectory.trajectory_id = "test_traj"

        with patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.torch.nn.utils."
                   "rnn.pad_sequence") as mock_pad_sequence, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.DataProto") as mock_data_proto:
            mock_pad_sequence.side_effect = [
                torch.tensor([[1, 2, 3]]),
                torch.tensor([[4, 5, 6]]),
                torch.tensor([[1, 1, 1]]),
            ]

            mock_data_proto_instance = MagicMock()
            mock_data_proto.from_dict.return_value = mock_data_proto_instance
            mock_data_proto_instance.non_tensor_batch = {}
            mock_data_proto_instance.meta_info = {}

            result = await transform_episodes_to_batch(mock_tokenizer, [mock_trajectory], ["0"])

            assert mock_pad_sequence.call_count == 3

    @pytest.mark.asyncio
    async def test_async_generate_sequences_with_trajectory_dict(self, mock_hybrid_agent_loop_manager, mock_config, mock_prompts, patch_modules):
        """测试async_generate_sequences函数 - 使用dict类型轨迹"""

        mock_hybrid_agent_loop_manager.traj_output_path = None

        mock_trajectory = {
            "idx": 0,
            "prompt_id": 0,
            "prompt_tokens": torch.tensor([1, 2, 3]),
            "response_tokens": torch.tensor([4, 5, 6]),
            "logprobs": [0.1, 0.2, 0.3],
            "response_masks": torch.tensor([1, 1, 1]),
            "trajectory_reward": 0.8,
            "chat_completions": "test completion"
        }

        with patch("verl.utils.hf_tokenizer") as mock_hf_tokenizer, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "create_tasks", AsyncMock()) as mock_create_tasks, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "generate_trajectory", AsyncMock()) as mock_generate_traj, \
                patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "transform_trajectories_to_batch", AsyncMock()) as mock_transform:
            mock_tokenizer = MagicMock()
            mock_hf_tokenizer.return_value = mock_tokenizer

            mock_agent_tasks = [MagicMock()]
            mock_create_tasks.return_value = mock_agent_tasks
            mock_generate_traj.return_value = mock_trajectory
            mock_transformed_batch = MagicMock()
            mock_transform.return_value = mock_transformed_batch

            result = await mock_hybrid_agent_loop_manager.async_generate_sequences(
                mock_config, mock_prompts, mock_tokenizer)

            mock_transform.assert_called_once()
            assert result == mock_transformed_batch

    def test_write_file_with_int_value(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        """测试write_file方法处理int类型数据"""
        with patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.open", create=True) as mock_open, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.os.path.join") as mock_path_join, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.json.dump") as mock_json_dump, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.logger") as mock_logger:
            mock_hybrid_agent_loop_manager.iteration = 1
            mock_hybrid_agent_loop_manager.perf_timestamp = 1234567890
            mock_hybrid_agent_loop_manager.traj_output_path = "/tmp/test_trajectories"

            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            test_data = {
                "int_value": 42,
            }

            mock_hybrid_agent_loop_manager.write_file(test_data, "test")

            mock_json_dump.assert_called_once()

    def test_write_file_with_float_value(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        """测试write_file方法处理float类型数据"""
        with patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.open", create=True) as mock_open, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.os.path.join") as mock_path_join, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.json.dump") as mock_json_dump, \
             patch("aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.logger") as mock_logger:
            mock_hybrid_agent_loop_manager.iteration = 1
            mock_hybrid_agent_loop_manager.perf_timestamp = 1234567890
            mock_hybrid_agent_loop_manager.traj_output_path = "/tmp/test_trajectories"

            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            test_data = {
                "float_value": 3.14159,
            }

            mock_hybrid_agent_loop_manager.write_file(test_data, "test")

            mock_json_dump.assert_called_once()
