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
            # 在fixture内部导入，确保patch生效
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
        """创建模拟的DataProto提示"""
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
        """创建模拟的轨迹数据"""
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
                patch("verl.utils.hf_tokenizer") as mock_hf_tokenizer, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "launch_server", AsyncMock()) as mock_launch_s:
            from agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager import HybridAgentLoopManager

            mock_tokenizer = MagicMock()
            mock_hf_tokenizer.return_value = mock_tokenizer

            manager = HybridAgentLoopManager(mock_config)

            mock_hf_tokenizer.assert_called_once_with(
                mock_config.actor_rollout_ref.model.path, trust_remote_code=True)

            mock_launch_s.assert_called_once()

            assert manager.server_addresses[0] == "0.0.0.0:1234"

    @pytest.fixture
    def mock_hybrid_agent_loop_manager(self, mock_config):
        with patch("verl.experimental.agent_loop.AgentLoopManager", MockAgentLoopManager), \
                patch("verl.utils.hf_tokenizer") as mock_hf_tokenizer, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "launch_server", AsyncMock()) as mock_launch_s:
            from agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager import HybridAgentLoopManager
            manager = HybridAgentLoopManager(mock_config)

            return manager

    def test_init_agent_loop_workers(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        mock_hybrid_agent_loop_manager._init_agent_loop_workers()
        assert True

    @pytest.mark.asyncio
    async def test_async_generate_sequences(self, mock_hybrid_agent_loop_manager, mock_config, mock_prompts,
                                            mock_trajectory, patch_modules):
        with patch("verl.utils.hf_tokenizer") as mock_hf_tokenizer, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "create_tasks", AsyncMock()) as mock_create_tasks, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "generate_trajectory", AsyncMock()) as mock_generate_traj, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                      "transform_trajectories_to_batch", AsyncMock()) as mock_transform, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.HybridAgentLoopManager."
                      "write_file") as mock_write_file:
            # 分词器
            mock_tokenizer = MagicMock()
            mock_hf_tokenizer.return_value = mock_tokenizer

            # 构造task
            mock_agent_tasks = [MagicMock(), MagicMock()]
            mock_create_tasks.return_value = mock_agent_tasks

            # 生成轨迹
            mock_generate_traj.return_value = mock_trajectory

            # 轨迹转换
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
            mock_write_file.assert_called_once_with([mock_trajectory, mock_trajectory], prefix="trajectories")
            mock_transform.assert_called_once_with(mock_config, mock_tokenizer, [mock_trajectory, mock_trajectory])

            # 验证结果
            assert result == mock_transformed_batch

    def test_generate_sequences(self, mock_hybrid_agent_loop_manager, mock_config, mock_prompts, patch_modules):
        with    patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.HybridAgentLoopManager."
                      "async_generate_sequences") as mock_async_generate_sequences, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.HybridAgentLoopManager."
                      "wake_up") as mock_wake_up, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.HybridAgentLoopManager."
                      "sleep") as mock_sleep:
            # 设置模拟
            mock_result = MagicMock()
            mock_async_generate_sequences.return_value = mock_result

            # 调用测试方法
            result = mock_hybrid_agent_loop_manager.generate_sequences(mock_prompts)

            # 验证调用，来自于输入的global_steps
            assert mock_hybrid_agent_loop_manager.iteration == 1

            mock_wake_up.assert_called_once()
            mock_async_generate_sequences.assert_called_once_with(
                mock_config, mock_prompts, mock_hybrid_agent_loop_manager.tokenizer)
            mock_sleep.assert_called_once()

            # 验证结果
            assert result == mock_result

    def test_wake_up(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        """测试wake_up方法"""

        # 设置模拟
        mock_replica1 = MagicMock()
        mock_replica2 = MagicMock()
        mock_hybrid_agent_loop_manager.rollout_replicas = [mock_replica1, mock_replica2]

        # 调用测试方法
        mock_hybrid_agent_loop_manager.wake_up()

        # 验证调用
        mock_replica1.wake_up.assert_called_once()
        mock_replica2.wake_up.assert_called_once()

    def test_sleep(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        """测试sleep方法"""

        # 设置模拟
        mock_replica1 = MagicMock()
        mock_replica2 = MagicMock()
        mock_hybrid_agent_loop_manager.rollout_replicas = [mock_replica1, mock_replica2]

        # 调用测试方法
        mock_hybrid_agent_loop_manager.sleep()

        # 验证调用
        mock_replica1.sleep.assert_called_once()
        mock_replica2.sleep.assert_called_once()

    def test_clear_kv_cache(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        """测试clear_kv_cache方法"""

        # 设置模拟
        mock_replica1 = MagicMock()
        mock_replica2 = MagicMock()
        mock_hybrid_agent_loop_manager.rollout_replicas = [mock_replica1, mock_replica2]

        # 调用测试方法
        mock_hybrid_agent_loop_manager.clear_kv_cache()

        # 验证调用
        mock_replica1.clear_kv_cache.assert_called_once()
        mock_replica2.clear_kv_cache.assert_called_once()

    def test_write_file(self, mock_hybrid_agent_loop_manager, mock_config, patch_modules):
        """测试write_file方法"""

        with patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.datetime") as mock_datetime, \
             patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.os.path.join") as mock_path_join, \
             patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.os.path.realpath") as mock_realpath, \
             patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.os.open") as mock_os_open, \
             patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.os.fdopen") as mock_fdopen, \
             patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.json.dump") as mock_json_dump, \
             patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.logger") as mock_logger:
            mock_hybrid_agent_loop_manager.iteration = 1

            mock_timestamp = 1234567890
            mock_datetime.datetime.now.return_value.timestamp.return_value = mock_timestamp
            mock_file_path = "/tmp/test_trajectories/rollout_test_1234567890.json"
            mock_path_join.return_value = mock_file_path
            mock_realpath.return_value = mock_file_path
            mock_fd = 42
            mock_os_open.return_value = mock_fd
            mock_file = MagicMock()
            mock_fdopen.return_value.__enter__.return_value = mock_file

            test_data = {
                "tensor_data": torch.tensor([1, 2, 3]),
                "list_data": [torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])],
                "dict_data": {"key": torch.tensor([10, 11, 12])},
                "str_data": "test"
            }

            mock_hybrid_agent_loop_manager.write_file(test_data, "test")

            mock_path_join.assert_called_once_with(
                mock_hybrid_agent_loop_manager.traj_output_path,
                f'rollout_test_{mock_timestamp}.json'
            )
            mock_realpath.assert_called_once_with(mock_file_path)
            mock_os_open.assert_called_once_with(
                mock_file_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            mock_fdopen.assert_called_once_with(mock_fd, 'a')
            mock_json_dump.assert_called_once()
            mock_file.write.assert_called_once_with('\n')
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_server(self, patch_modules):
        """测试launch_server函数"""
        with patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager."
                   "InferRouter", AsyncMock()) as mock_infer_router:
            # 在测试函数内部导入，确保patch生效
            from agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager import launch_server

            # 设置模拟
            mock_router = AsyncMock()
            mock_infer_router.create.return_value = mock_router

            # 调用测试函数
            await launch_server("test_infer_service", "test_model", ["server1", "server2"])

            # 验证调用
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
        """测试create_tasks函数"""
        from agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager import create_tasks
        from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask

        # 设置模拟
        agent_service = "test_agent_service"
        prompts = mock_prompts
        n_samples_per_prompt = 2

        # 调用测试函数
        agent_tasks = await create_tasks(agent_service, prompts, n_samples_per_prompt)

        # 验证结果
        assert len(agent_tasks) == 2

        # 验证第一个任务
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

        # 验证第二个任务
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
        """测试generate_trajectory函数"""
        with patch("agentic_rl.runner.agent_router.AgentRouter", AsyncMock()) as mock_agent_router:
            from agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager import generate_trajectory
            from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask

            # 设置模拟
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

            # 调用测试函数
            trajectory = await generate_trajectory(agent_task)

            # 验证调用
            mock_agent_router.create.assert_called_once()
            mock_router.generate_trajectory.assert_called_once_with(agent_task, mode='Token')

            # 验证结果
            assert trajectory == mock_trajectory

    @pytest.mark.asyncio
    async def test_transform_trajectories_to_batch(self, mock_config, mock_trajectories, patch_modules):
        """测试transform_trajectories_to_batch函数"""
        with patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.torch.nn.utils."
                   "rnn.pad_sequence") as mock_pad_sequence, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.DataProto") as mock_data_proto:
            from agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager import transform_trajectories_to_batch

            # 设置模拟
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token_id = 0

            mock_pad_sequence.side_effect = [
                torch.tensor([[1, 2, 3], [7, 8, 9]]),  # prompts_batch
                torch.tensor([[4, 5, 6], [10, 11, 12]]),  # response_batch
                torch.tensor([[1, 1, 1], [1, 1, 1]]),  # traj_mask
                torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # rollout_log_probs_batch
            ]

            mock_data_proto_instance = MagicMock()
            mock_data_proto.from_dict.return_value = mock_data_proto_instance
            mock_data_proto_instance.non_tensor_batch = {}
            mock_data_proto_instance.meta_info = {}

            # 调用测试函数
            result = await transform_trajectories_to_batch(mock_config, mock_tokenizer, mock_trajectories)

            # 验证调用
            assert mock_pad_sequence.call_count == 4
            mock_data_proto.from_dict.assert_called_once()

            # 验证结果
            assert result == mock_data_proto_instance
            assert "uid" in result.non_tensor_batch
            assert "timing" in result.meta_info
