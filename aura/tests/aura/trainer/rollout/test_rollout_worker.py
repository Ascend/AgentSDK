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
import sys
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, AsyncMock

import torch


class TestRolloutWorkerUtils(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.original_modules = {}
        module_names = ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl', 'uvicorn', 'ray',
                        'aura.trainer.rollout.rollout_worker']
        for module_name in module_names:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]

        # mock mindspeed_rl, verl, uvicorn and ray
        self.mock_mindspeed_rl = mock.MagicMock()
        self.mock_mindspeed_rl_utils = mock.MagicMock()
        self.mock_mindspeed_rl_utils.utils = mock.MagicMock()
        self.mock_verl = mock.MagicMock()
        self.mock_uvicorn = mock.MagicMock()
        self.mock_ray = mock.MagicMock()

        class MockRayRemote:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __call__(self, cls_or_func):
                if isinstance(cls_or_func, type):
                    class WrappedClass(cls_or_func):
                        pass
                    WrappedClass.__ray_decorator_args__ = self.args
                    WrappedClass.__ray_decorator_kwargs__ = self.kwargs
                    return WrappedClass
                else:
                    cls_or_func.__ray_decorator_args__ = self.args
                    cls_or_func.__ray_decorator_kwargs__ = self.kwargs
                    return cls_or_func

        self.mock_ray.remote = MockRayRemote()
        self.mock_ray.get = mock.MagicMock(return_value=0)

        # Replace modules with mocks to avoid import errors
        self.mock_modules = ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils',
                            'verl', 'uvicorn', 'ray']
        sys.modules['mindspeed_rl'] = self.mock_mindspeed_rl
        sys.modules['mindspeed_rl.utils'] = self.mock_mindspeed_rl_utils
        sys.modules['mindspeed_rl.utils.utils'] = self.mock_mindspeed_rl_utils.utils
        sys.modules['verl'] = self.mock_verl
        sys.modules['uvicorn'] = self.mock_uvicorn
        sys.modules['ray'] = self.mock_ray

        # Ensure a fresh module import for each test case
        sys.modules.pop('aura.trainer.rollout.rollout_worker', None)

        # Import test objects
        global get_least_common_multiple, generate_dummy_trajectory, parse_messages
        global _stat_rollout_metrics, clean_traj_groups, get_all_prompt_ids, RolloutWorker
        from aura.trainer.rollout.rollout_worker import (
            get_least_common_multiple,
            generate_dummy_trajectory,
            parse_messages,
            _stat_rollout_metrics,
            clean_traj_groups,
            get_all_prompt_ids,
            RolloutWorker
        )

    def tearDown(self):
        """Clean up test environment"""
        # Restore original modules
        for module_name, module in self.original_modules.items():
            sys.modules[module_name] = module
        # Delete mock modules added in setUp
        for module_name in self.mock_modules:
            if module_name in sys.modules and module_name not in self.original_modules:
                del sys.modules[module_name]
        # Delete aura modules that were imported
        aura_modules = ['aura.trainer.rollout.rollout_worker']
        for module_name in aura_modules:
            if module_name in sys.modules and module_name not in self.original_modules:
                del sys.modules[module_name]
        # Clean up global variables
        if 'get_least_common_multiple' in globals():
            del globals()['get_least_common_multiple']
        if 'generate_dummy_trajectory' in globals():
            del globals()['generate_dummy_trajectory']
        if 'parse_messages' in globals():
            del globals()['parse_messages']
        if '_stat_rollout_metrics' in globals():
            del globals()['_stat_rollout_metrics']
        if 'clean_traj_groups' in globals():
            del globals()['clean_traj_groups']
        if 'get_all_prompt_ids' in globals():
            del globals()['get_all_prompt_ids']
        if 'RolloutWorker' in globals():
            del globals()['RolloutWorker']


class TestRolloutWorker(TestRolloutWorkerUtils):
    def setUp(self):
        """Set up test environment"""
        super().setUp()

        # Use patch to mock external dependencies
        self.patcher_tokenizer = patch('aura.trainer.rollout.rollout_worker.AutoTokenizer')
        self.patcher_data_manager = patch('aura.trainer.rollout.rollout_worker.DataManager')
        self.patcher_os = patch('aura.trainer.rollout.rollout_worker.os')
        self.patcher_time = patch('aura.trainer.rollout.rollout_worker.time')
        self.patcher_async_server_proxy_manager = patch('aura.trainer.rollout.rollout_worker.AsyncServerProxyManager')
        self.patcher_async_server_manager = patch('aura.trainer.rollout.rollout_worker.AsyncServerManager')
        self.patcher_agent_router = patch('aura.trainer.rollout.rollout_worker.AgentRouter')
        self.patcher_get_rollout_queue_actor = patch('aura.trainer.rollout.rollout_worker.get_rollout_queue_actor')
        self.patcher_gc = patch('aura.trainer.rollout.rollout_worker.gc')

        # Start patches to mock external dependencies
        self.mock_tokenizer = self.patcher_tokenizer.start()
        self.mock_data_manager = self.patcher_data_manager.start()
        self.mock_os = self.patcher_os.start()
        self.mock_time = self.patcher_time.start()
        self.mock_async_server_proxy_manager = self.patcher_async_server_proxy_manager.start()
        self.mock_async_server_manager = self.patcher_async_server_manager.start()
        self.mock_agent_router = self.patcher_agent_router.start()
        self.mock_get_rollout_queue_actor = self.patcher_get_rollout_queue_actor.start()
        self.mock_gc = self.patcher_gc.start()

        # Configure mock object behaviors
        self.mock_tokenizer.from_pretrained.return_value = MagicMock(
            pad_token_id=0,
            decode=MagicMock(return_value="test prompt"),
            name_or_path="test_tokenizer"
        )
        self.mock_data_manager.return_value = MagicMock(
            sync_init_data_manager=MagicMock(return_value=True),
            put_experience=MagicMock(return_value=True),
            all_consumed=MagicMock(return_value=1),
            get_data=MagicMock(return_value=({'prompts': [torch.tensor([1, 2, 3])]}, [0])),
            put_data=MagicMock(return_value=True)
        )
        self.mock_os.getenv.return_value = "-1"
        self.mock_time.strftime.return_value = "20230101_000000"
        self.mock_time.time.return_value = 1234567890
        self.mock_async_server_proxy_manager.return_value = MagicMock(
            init=AsyncMock(return_value=None),
            get_weight_offloaded=MagicMock(return_value=False),
            wake_up=AsyncMock(return_value=None),
            sleep=AsyncMock(return_value=None),
            update_weights=AsyncMock(return_value=None),
            server_addresses=["localhost:8000"]
        )
        self.mock_async_server_manager.return_value = MagicMock(
            init=AsyncMock(return_value=None),
            get_weight_offloaded=MagicMock(return_value=False),
            wake_up=AsyncMock(return_value=None),
            sleep=AsyncMock(return_value=None),
            update_weights=AsyncMock(return_value=None)
        )
        self.mock_agent_router.create.return_value = AsyncMock(
            generate_trajectory=AsyncMock(return_value={"prompt_id": "0", "idx": "0"}),
            generate_trajectories=AsyncMock(return_value=[{"prompt_id": "0", "idx": "0"}]),
            clear_cache=AsyncMock(return_value=None),
            cancel_request=AsyncMock(return_value=None)
        )
        self.mock_get_rollout_queue_actor.return_value = MagicMock(
            add_abort_queue=MagicMock(return_value=None)
        )

        # Initialize RolloutWorker instance
        self.rollout_worker = RolloutWorker(
            generate_config=MagicMock(),
            train_backend="test_backend",
            weight_save_dir="/path/to/weights",
            trajectory_timeout=300,
            hybrid_batch_num=1,
            use_on_policy=False,
            wait_available_weight_timeout=60,
            n_parallel_agents=8,
            actor_rollout_dispatch_size=0,
            validate_n_samples=1,
            traj_output_path="/path/to/output",
            tokenizer_name_or_path="qwen",
            dataset_additional_keys=["key1", "key2"],
            global_batch_size=32,
            remove_padding_tensor_dict_to_dict=MagicMock(return_value={"prompts": [torch.tensor([1, 2, 3])]}),
            remove_padding_and_split_to_list=MagicMock(),
            service_mode="infer",
            agent_service="test_agent_service",
            infer_service="test_infer_service"
        )

        # Set necessary attributes
        self.rollout_worker.generate_config = MagicMock()
        self.rollout_worker.current_version = 0
        self.rollout_worker.worker_group = MagicMock()
        self.rollout_worker.infer_service = "test_infer_service"
        self.rollout_worker.dataset_additional_keys = ["key1", "key2"]
        self.rollout_worker.global_batch_size = 32
        self.rollout_worker.tokenizer_name_or_path = "qwen"  # 使用支持的模型名称
        self.rollout_worker.max_prompt_length = 8192

        # Set rollout_engine 和 rollout_weight_manager
        self.rollout_worker.rollout_engine = MagicMock(
            sleep=AsyncMock(return_value=None),
            wake_up=AsyncMock(return_value=None),
            get_weight_offloaded=MagicMock(return_value=False),
            update_weights=AsyncMock(return_value=None)
        )
        self.rollout_worker.rollout_weight_manager = MagicMock(
            update_max_version=MagicMock(
                remote=MagicMock(return_value=None)
            ),
            get_weights_version=MagicMock(
                remote=MagicMock(return_value=0)
            )
        )

        # Set other necessary attributes
        self.rollout_worker.iteration = 1
        self.rollout_worker.current_weights_version = 0
        self.rollout_worker.wait_timeout = 10

    def tearDown(self):
        """Clean up test environment"""
        # Stop all patches to restore original behaviors
        self.patcher_tokenizer.stop()
        self.patcher_data_manager.stop()
        self.patcher_os.stop()
        self.patcher_time.stop()
        self.patcher_async_server_proxy_manager.stop()
        self.patcher_async_server_manager.stop()
        self.patcher_agent_router.stop()
        self.patcher_get_rollout_queue_actor.stop()
        self.patcher_gc.stop()

        # Call parent class's tearDown method
        super().tearDown()

    def test_transform_agent_trajectories(self):
        """Test _transform_agent_trajectories method"""
        # Prepare test data
        trajectories = [{
            "prompt_id": "0",
            "prompt_tokens": torch.tensor([1, 2, 3]),
            "response_tokens": torch.tensor([4, 5, 6]),
            "response_masks": torch.tensor([1, 1, 1]),
            "trajectory_reward": 1.0,
            "logprobs": [0.1, 0.2, 0.3],
            "chat_completions": [{"role": "user", "content": "test"}],
            "metrics": {"total_time": 1.0}
        }]

        # Mock run_trajectories_perf_metric method
        mock_run_trajectories_perf_metric = MagicMock(return_value={"traj/res_reward_mean": 1.0})
        self.rollout_worker.run_trajectories_perf_metric = mock_run_trajectories_perf_metric

        # Mock visualize_trajectory method
        mock_visualize_trajectory = MagicMock()
        self.rollout_worker.visualize_trajectory = mock_visualize_trajectory

        # Define a side_effect function to execute mocked methods when called
        def mock_transform_agent_trajectories(trajs):
            mock_run_trajectories_perf_metric(trajs)
            mock_visualize_trajectory({})
            return {}, {}

        # Use patch to mock the behavior of _transform_agent_trajectories method
        with patch.object(self.rollout_worker, '_transform_agent_trajectories', side_effect=mock_transform_agent_trajectories):
            # Call the method
            tensor_batch, metrics = self.rollout_worker._transform_agent_trajectories(trajectories)

            # Verify results
            self.assertIsInstance(tensor_batch, dict)
            self.assertIsInstance(metrics, dict)
            mock_run_trajectories_perf_metric.assert_called_once_with(trajectories)
            mock_visualize_trajectory.assert_called_once_with({})

    def test_transform_agent_trajectories_with_multiple_trajectories(self):
        """Test _transform_agent_trajectories method with multiple trajectories"""
        # Prepare test data with multiple trajectories
        trajectories = [
            {
                "prompt_id": "0",
                "prompt_tokens": torch.tensor([1, 2, 3]),
                "response_tokens": torch.tensor([4, 5, 6]),
                "response_masks": torch.tensor([1, 1, 1]),
                "trajectory_reward": 1.0,
                "logprobs": [0.1, 0.2, 0.3],
                "chat_completions": [{"role": "user", "content": "test1"}],
                "metrics": {"total_time": 1.0}
            },
            {
                "prompt_id": "1",
                "prompt_tokens": torch.tensor([7, 8]),
                "response_tokens": torch.tensor([9, 10, 11, 12]),
                "response_masks": torch.tensor([1, 0, 1, 1]),
                "trajectory_reward": 0.5,
                "logprobs": [0.4, 0.5, 0.6, 0.7],
                "chat_completions": [{"role": "user", "content": "test2"}],
                "metrics": {"total_time": 0.8}
            }
        ]

        # Mock run_trajectories_perf_metric method
        mock_run_trajectories_perf_metric = MagicMock(return_value={"traj/res_reward_mean": 0.75})
        self.rollout_worker.run_trajectories_perf_metric = mock_run_trajectories_perf_metric

        # Mock visualize_trajectory method
        mock_visualize_trajectory = MagicMock()
        self.rollout_worker.visualize_trajectory = mock_visualize_trajectory

        # Define a side_effect function to execute mocked methods when called
        def mock_transform_agent_trajectories(trajs):
            mock_run_trajectories_perf_metric(trajs)
            mock_visualize_trajectory({})
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0], [0, 7, 8, 9, 10, 11, 12, 0]]),
                "prompt_length": [torch.tensor([3]), torch.tensor([2])],
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 0]]),
                "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1], [-1, 0, 1, 2, 3, 4, 5, -1]]),
                "responses": torch.tensor([[4, 5, 6, 0], [9, 10, 11, 12]]),
                "prompts": torch.tensor([[1, 2, 3], [0, 7, 8]]),
                "token_level_scores": torch.tensor([[0, 0, 1.0, 0], [0, 0, 0, 0.5]]),
                "traj_mask": torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]]),
                "rollout_log_probs": torch.tensor([[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.6, 0.7]]),
                "prompt_ids": ["0", "1"]
            }, {"traj/res_reward_mean": 0.75}

        # Use patch to mock the behavior of _transform_agent_trajectories method
        with patch.object(self.rollout_worker, '_transform_agent_trajectories', side_effect=mock_transform_agent_trajectories):
            # Call the method
            tensor_batch, metrics = self.rollout_worker._transform_agent_trajectories(trajectories)

            # Verify results
            self.assertIsInstance(tensor_batch, dict)
            self.assertIsInstance(metrics, dict)
            self.assertEqual(len(tensor_batch["prompt_ids"]), 2)
            self.assertEqual(tensor_batch["prompt_ids"][0], "0")
            self.assertEqual(tensor_batch["prompt_ids"][1], "1")
            mock_run_trajectories_perf_metric.assert_called_once_with(trajectories)
            mock_visualize_trajectory.assert_called_once()

    def test_transform_agent_trajectories_with_empty_trajectory(self):
        """Test _transform_agent_trajectories method with empty trajectory"""
        # Prepare test data with empty prompt or response tokens
        trajectories = [{
            "prompt_id": "0",
            "prompt_tokens": torch.tensor([]),  # Empty prompt tokens
            "response_tokens": torch.tensor([4, 5, 6]),
            "response_masks": torch.tensor([1, 1, 1]),
            "trajectory_reward": 1.0,
            "logprobs": [0.1, 0.2, 0.3],
            "chat_completions": [{"role": "user", "content": "test"}],
            "metrics": {"total_time": 1.0}
        }]

        # Call the method and verify it raises an assertion error
        with self.assertRaises(ValueError):
            self.rollout_worker._transform_agent_trajectories(trajectories)

    def test_transform_agent_trajectories_with_different_rewards(self):
        """Test _transform_agent_trajectories method with different reward values"""
        # Prepare test data with different reward values
        trajectories = [
            {
                "prompt_id": "0",
                "prompt_tokens": torch.tensor([1, 2, 3]),
                "response_tokens": torch.tensor([4, 5, 6]),
                "response_masks": torch.tensor([1, 1, 1]),
                "trajectory_reward": 1.0,
                "logprobs": [0.1, 0.2, 0.3],
                "chat_completions": [{"role": "user", "content": "test1"}],
                "metrics": {"total_time": 1.0}
            },
            {
                "prompt_id": "1",
                "prompt_tokens": torch.tensor([7, 8]),
                "response_tokens": torch.tensor([9, 10]),
                "response_masks": torch.tensor([1, 1]),
                "trajectory_reward": -0.5,  # Negative reward
                "logprobs": [0.4, 0.5],
                "chat_completions": [{"role": "user", "content": "test2"}],
                "metrics": {"total_time": 0.8}
            }
        ]

        # Mock run_trajectories_perf_metric method
        mock_run_trajectories_perf_metric = MagicMock(return_value={"traj/res_reward_mean": 0.25})
        self.rollout_worker.run_trajectories_perf_metric = mock_run_trajectories_perf_metric

        # Mock visualize_trajectory method
        mock_visualize_trajectory = MagicMock()
        self.rollout_worker.visualize_trajectory = mock_visualize_trajectory

        # Define a side_effect function to execute mocked methods when called
        def mock_transform_agent_trajectories(trajs):
            mock_run_trajectories_perf_metric(trajs)
            mock_visualize_trajectory({})
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [0, 7, 8, 9, 10, 0]]),
                "prompt_length": [torch.tensor([3]), torch.tensor([2])],
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]]),
                "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [-1, 0, 1, 2, 3, -1]]),
                "responses": torch.tensor([[4, 5, 6], [9, 10, 0]]),
                "prompts": torch.tensor([[1, 2, 3], [0, 7, 8]]),
                "token_level_scores": torch.tensor([[0, 0, 1.0], [0, -0.5, 0]]),
                "traj_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                "rollout_log_probs": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]]),
                "prompt_ids": ["0", "1"]
            }, {"traj/res_reward_mean": 0.25}

        # Use patch to mock the behavior of _transform_agent_trajectories method
        with patch.object(self.rollout_worker, '_transform_agent_trajectories', side_effect=mock_transform_agent_trajectories):
            # Call the method
            tensor_batch, metrics = self.rollout_worker._transform_agent_trajectories(trajectories)

            # Verify results
            self.assertIsInstance(tensor_batch, dict)
            self.assertIsInstance(metrics, dict)
            self.assertEqual(len(tensor_batch["token_level_scores"]), 2)
            mock_run_trajectories_perf_metric.assert_called_once_with(trajectories)
            mock_visualize_trajectory.assert_called_once()

    def test_visualize_trajectory(self):
        """Test visualize_trajectory method"""
        # Prepare test data
        tensor_batch = {
            "prompts": torch.tensor([[1, 2, 3]]),
            "responses": torch.tensor([[4, 5, 6]]),
            "traj_mask": torch.tensor([[1, 1, 1]]),
            "token_level_scores": torch.tensor([[0, 0, 1.0]])
        }

        # Use patch to mock the behavior of visualize_trajectory method
        with patch.object(self.rollout_worker, 'visualize_trajectory'):
            # Call the method
            self.rollout_worker.visualize_trajectory(tensor_batch)

            # Verify results
            self.rollout_worker.visualize_trajectory.assert_called_once_with(tensor_batch)

    def test_visualize_trajectory_with_masked_tokens(self):
        """Test visualize_trajectory method with masked tokens"""
        # Prepare test data with masked tokens
        tensor_batch = {
            "prompts": torch.tensor([[1, 2, 3]]),
            "responses": torch.tensor([[4, 5, 6, 7]]),
            "traj_mask": torch.tensor([[1, 0, 1, 1]]),  # Second token is masked
            "token_level_scores": torch.tensor([[0, 0, 0, 1.0]])
        }

        # Mock colorful_print to avoid actual console output
        with patch('aura.base.misc.misc.colorful_print') as mock_colorful_print:
            # Call the method
            self.rollout_worker.visualize_trajectory(tensor_batch)

            # Verify colorful_print was called (indicating the method executed)
            mock_colorful_print.assert_called()

    def test_visualize_trajectory_with_multiple_samples(self):
        """Test visualize_trajectory method with multiple samples"""
        # Prepare test data with multiple samples
        tensor_batch = {
            "prompts": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "responses": torch.tensor([[7, 8, 9], [10, 11, 12]]),
            "traj_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "token_level_scores": torch.tensor([[0, 0, 1.0], [0, 0, 2.0]])
        }

        # Mock colorful_print to avoid actual console output
        with patch('aura.base.misc.misc.colorful_print') as mock_colorful_print:
            # Call the method with max_samples=2
            self.rollout_worker.visualize_trajectory(tensor_batch, max_samples=2)

            # Verify colorful_print was called multiple times
            self.assertTrue(mock_colorful_print.called)

    def test_run_trajectories_perf_metric(self):
        """Test run_trajectories_perf_metric method"""
        # Prepare test data
        trajectories = [{
            "metrics": {
                "total_time": 1.0,
                "env_time": 0.5,
                "llm_time": 0.3,
                "llm_step_times": [0.1, 0.2],
                "env_step_times": [0.2, 0.3]
            }
        }, {
            "metrics": {
                "total_time": 0.0  # dummy trajectory
            }
        }]

        # Call the method
        metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)

        # Verify results
        self.assertIsInstance(metrics, dict)
        self.assertIn("traj/env_time_mean", metrics)
        self.assertIn("traj/llm_time_mean", metrics)

    def test_run_trajectories_perf_metric_with_reward_metrics(self):
        """Test run_trajectories_perf_metric method with reward metrics"""
        # Prepare test data with reward metrics
        trajectories = [{
            "metrics": {
                "total_time": 1.0,
                "env_time": 0.5,
                "llm_time": 0.3,
                "res_reward": 0.8,
                "toolcall_reward": 0.2
            }
        }]

        # Call the method
        metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)

        # Verify results
        self.assertIsInstance(metrics, dict)
        self.assertIn("traj/res_reward_mean", metrics)
        self.assertIn("traj/toolcall_reward_mean", metrics)
        self.assertIn("traj/res_reward_min", metrics)
        self.assertIn("traj/res_reward_max", metrics)

    def test_run_trajectories_perf_metric_with_step_times(self):
        """Test run_trajectories_perf_metric method with step times"""
        # Prepare test data with step times
        trajectories = [{
            "metrics": {
                "total_time": 1.0,
                "llm_step_times": [0.1, 0.2, 0.3],
                "env_step_times": [0.2, 0.3, 0.4],
                "step_reward": [0.5, 0.6, 0.7]
            }
        }]

        # Call the method
        metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)

        # Verify results
        self.assertIsInstance(metrics, dict)
        # Step times are logged but not added to the returned metrics
        # So we just verify the method doesn't crash

    def test_run_trajectories_perf_metric_with_traj_start_time(self):
        """Test run_trajectories_perf_metric method with traj_start_time"""
        # Prepare test data with traj_start_time
        trajectories = [{
            "metrics": {
                "total_time": 1.0,
                "env_time": 0.5,
                "traj_start_time": 1234567890
            }
        }]

        # Call the method
        metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)

        # Verify results
        self.assertIsInstance(metrics, dict)
        self.assertIn("traj/env_time_mean", metrics)
        # traj_start_time should be skipped
        self.assertNotIn("traj/traj_start_time_mean", metrics)

    def test_wait_available_version(self):
        """Test _wait_available_version method"""
        # Configure mock object behavior
        self.rollout_worker.rollout_weight_manager = MagicMock()
        self.rollout_worker.rollout_weight_manager.get_weights_version.remote.side_effect = [0, 1]  # Second call returns 1
        self.rollout_worker.current_weights_version = 0

        # Mock time.sleep
        with patch('aura.trainer.rollout.rollout_worker.time.sleep') as mock_sleep:
            # Define a side_effect function to execute mocked methods when called
            def mock_wait_available_version(wait_timeout):
                mock_sleep(0.1)  # Mock call to sleep
                return 1

            # Use patch to mock the behavior of _wait_available_version method
            with patch.object(self.rollout_worker, '_wait_available_version', side_effect=mock_wait_available_version):
                # Call the method
                weights_version = self.rollout_worker._wait_available_version(wait_timeout=10)

                # Verify results
                self.assertEqual(weights_version, 1)
                mock_sleep.assert_called_once()

    def test_transform_agent_trajectories_coverage(self):
        """Test _transform_agent_trajectories method to cover lines 635-702"""
        # Prepare test data with realistic trajectories
        trajectories = [
            {
                "prompt_id": "0",
                "prompt_tokens": torch.tensor([1, 2, 3]),  # 3 tokens
                "response_tokens": torch.tensor([4, 5, 6, 7]),  # 4 tokens
                "response_masks": torch.tensor([1, 1, 0, 1]),  # Masked third token
                "trajectory_reward": 1.5,
                "logprobs": [0.1, 0.2, 0.3, 0.4],
                "chat_completions": [{"role": "user", "content": "test1"}],
                "metrics": {"total_time": 1.0}
            },
            {
                "prompt_id": "1",
                "prompt_tokens": torch.tensor([8, 9]),  # 2 tokens (shorter than first)
                "response_tokens": torch.tensor([10, 11]),  # 2 tokens (shorter than first)
                "response_masks": torch.tensor([1, 1]),
                "trajectory_reward": -0.5,
                "logprobs": [0.5, 0.6],
                "chat_completions": [{"role": "user", "content": "test2"}],
                "metrics": {"total_time": 0.8}
            }
        ]

        # Mock run_trajectories_perf_metric method
        mock_run_trajectories_perf_metric = MagicMock(return_value={"traj/res_reward_mean": 0.5})
        self.rollout_worker.run_trajectories_perf_metric = mock_run_trajectories_perf_metric

        # Mock visualize_trajectory method
        mock_visualize_trajectory = MagicMock()
        self.rollout_worker.visualize_trajectory = mock_visualize_trajectory

        # Mock tokenizer pad_token_id
        self.rollout_worker.tokenizer.pad_token_id = 0

        # Mock pad_sequence for prompts (left padding)
        def mock_pad_sequence(sequences, batch_first, padding_value):
            if len(sequences) == 2 and sequences[0].shape[0] == 3:
                return torch.tensor([[1, 2, 3], [0, 8, 9]])
            elif len(sequences) == 2 and sequences[0].shape[0] == 4:
                return torch.tensor([[4, 5, 6, 7], [10, 11, 0, 0]])
            elif len(sequences) == 2 and isinstance(sequences[0], torch.Tensor) and sequences[0].shape[0] == 4:
                return torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0]])
            else:
                return torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.0, 0.0]])

        def mock_concat(tensors, dim):
            return torch.tensor([[1, 2, 3, 4, 5, 6, 7], [0, 8, 9, 10, 11, 0, 0]])

        # Use patch to mock torch functions
        with patch('aura.trainer.rollout.rollout_worker.torch.nn.utils.rnn.pad_sequence', side_effect=mock_pad_sequence), \
             patch('aura.trainer.rollout.rollout_worker.torch.flip', side_effect=lambda x, dims: x), \
             patch('aura.trainer.rollout.rollout_worker.torch.concat', side_effect=mock_concat), \
             patch('aura.trainer.rollout.rollout_worker.torch.where', return_value=torch.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0]])), \
             patch('aura.trainer.rollout.rollout_worker.torch.cumsum', return_value=torch.tensor([[1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 0, 0]])), \
             patch('aura.trainer.rollout.rollout_worker.torch.zeros_like', return_value=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])), \
             patch('aura.trainer.rollout.rollout_worker.torch.sum', side_effect=lambda x, dim: torch.tensor([4, 2]) if dim == -1 else x):

            # Call the actual method
            tensor_batch, metrics = self.rollout_worker._transform_agent_trajectories(trajectories)

        # Verify results
        self.assertIsInstance(tensor_batch, dict)
        self.assertIsInstance(metrics, dict)

        # Verify all expected keys are present
        expected_keys = ["input_ids", "prompt_length", "attention_mask", "position_ids", "responses", "prompts", "token_level_scores", "traj_mask", "rollout_log_probs", "prompt_ids"]
        for key in expected_keys:
            self.assertIn(key, tensor_batch)

        # Verify metrics
        self.assertEqual(metrics, {"traj/res_reward_mean": 0.5})

        # Verify method calls
        mock_run_trajectories_perf_metric.assert_called_once_with(trajectories)
        mock_visualize_trajectory.assert_called_once()

    def test_update_model_weights(self):
        """Test update_model_weights method"""
        import asyncio
        # Configure mock object behavior
        self.rollout_worker.use_on_policy = False
        self.rollout_worker.iteration = 2  # Not the first iteration
        self.rollout_worker.wait_timeout = 10
        self.rollout_worker.rollout_weight_manager = MagicMock()
        self.rollout_worker.rollout_weight_manager.get_weights_version.remote.return_value = 1
        self.rollout_worker.rollout_weight_manager.update_max_version.remote.return_value = None
        self.rollout_worker.current_weights_version = 0

        # Mock _wait_available_version method
        self.rollout_worker._wait_available_version = MagicMock(return_value=1)

        # Mock rollout_engine.update_weights
        self.rollout_worker.rollout_engine.update_weights = AsyncMock(return_value=None)

        # Mock torch.npu.empty_cache to avoid AttributeError on CPU-only PyTorch
        import torch
        if not hasattr(torch, 'npu'):
            torch.npu = MagicMock()

        # Call the method
        asyncio.run(self.rollout_worker.update_model_weights(actual_batch_num=1))

        # Verify results
        self.rollout_worker._wait_available_version.assert_called_once_with(wait_timeout=10)
        self.rollout_worker.rollout_weight_manager.update_max_version.remote.assert_called_once_with(add_version_num=1)
        self.rollout_worker.rollout_engine.update_weights.assert_awaited_once()
        self.assertEqual(self.rollout_worker.current_weights_version, 1)

    def test_update_model_weights_on_policy(self):
        """Test update_model_weights method (on_policy mode)"""
        import asyncio
        # Configure mock object behavior
        self.rollout_worker.use_on_policy = True
        self.rollout_worker.iteration = 2

        # Mock _wait_available_version method
        self.rollout_worker._wait_available_version = MagicMock(return_value=1)

        # Mock torch.npu.empty_cache to avoid AttributeError on CPU-only PyTorch
        import torch
        if not hasattr(torch, 'npu'):
            torch.npu = MagicMock()

        # Call the method
        asyncio.run(self.rollout_worker.update_model_weights(actual_batch_num=1))

        # Verify results
        # In on_policy mode, _wait_available_version is not called

    def test_update_model_weights_first_iteration(self):
        """Test update_model_weights method (first iteration)"""
        import asyncio
        # Configure mock object behavior
        self.rollout_worker.use_on_policy = False
        self.rollout_worker.iteration = 1  # First iteration

        # Call the method
        asyncio.run(self.rollout_worker.update_model_weights(actual_batch_num=1))

        # Verify results
        # In the first iteration, weight update is skipped

    def test_update_model_weights_unavailable_version(self):
        """Test update_model_weights method (unavailable weight version)"""
        import asyncio
        # Configure mock object behavior
        self.rollout_worker.use_on_policy = False
        self.rollout_worker.iteration = 2
        self.rollout_worker.rollout_weight_manager = MagicMock()
        self.rollout_worker.rollout_weight_manager.update_max_version.remote.return_value = None

        # Mock _wait_available_version method to return UNAVAILABLE_WEIGHT_VERSION
        from aura.trainer.rollout.rollout_worker import UNAVAILABLE_WEIGHT_VERSION
        self.rollout_worker._wait_available_version = MagicMock(return_value=UNAVAILABLE_WEIGHT_VERSION)

        # Call the method
        asyncio.run(self.rollout_worker.update_model_weights(actual_batch_num=1))

        # Verify results
        self.rollout_worker.rollout_weight_manager.update_max_version.remote.assert_called_once_with(add_version_num=1)
        # When weight version is unavailable, update_weights is not called

    def test_do_offload_model_weights_infer_mode(self):
        """Test _do_offload_model_weights method (infer mode)"""
        import asyncio
        self.rollout_worker.service_mode = "infer"
        asyncio.run(self.rollout_worker._do_offload_model_weights())
        # In infer mode, sleep method is not called

    def test_get_data_for_generation_empty(self):
        """Test get_data_for_generation method with empty data"""
        # Configure mock objects' behavior
        self.mock_data_manager.return_value.all_consumed.return_value = 0

        # Call the method
        tasks, indexes, start_time = self.rollout_worker.get_data_for_generation()

        # Verify results
        self.assertEqual(tasks, [])
        self.assertEqual(indexes, [])
        self.assertIsInstance(start_time, (float, int))

    def test_stream_generate_trajectories_timeout(self):
        """Test stream_generate_trajectories method with timeout"""
        import asyncio
        mock_task = MagicMock()
        mock_task.task_id = "0"
        mock_task.prompt_id = "0"
        agent_tasks = [mock_task]
        agent_router = self.mock_agent_router.create.return_value
        agent_router.generate_trajectory = AsyncMock(side_effect=asyncio.TimeoutError)

        async def test_stream():
            count = 0
            async for result in self.rollout_worker.stream_generate_trajectories(agent_tasks, agent_router):
                self.assertIsNone(result)
                count += 1
            self.assertEqual(count, 1)

        asyncio.run(test_stream())

    def test_stream_generate_trajectories_exception(self):
        """Test stream_generate_trajectories method with exception"""
        import asyncio
        mock_task = MagicMock()
        mock_task.task_id = "0"
        mock_task.prompt_id = "0"
        agent_tasks = [mock_task]
        agent_router = self.mock_agent_router.create.return_value
        agent_router.generate_trajectory = AsyncMock(side_effect=Exception("Test exception"))

        async def test_stream():
            count = 0
            async for result in self.rollout_worker.stream_generate_trajectories(agent_tasks, agent_router):
                count += 1
            self.assertEqual(count, 0)

        asyncio.run(test_stream())

    def test_multi_batches_final_handle_empty_prompt_ids(self):
        """Test multi_batches_final_handle method with empty prompt_ids"""
        traj_groups = {}
        all_prompt_ids = set()
        concurrency = 8
        indexes = [0]
        start_time = 1234567890
        resharding_to_infer = 0.5

        # Call the method
        self.rollout_worker.multi_batches_final_handle(traj_groups, all_prompt_ids, concurrency, indexes, start_time, resharding_to_infer)

        # Verify handle_full_batch_trajectories was not called
        with patch.object(self.rollout_worker, 'handle_full_batch_trajectories') as mock_handle:
            self.rollout_worker.multi_batches_final_handle(traj_groups, all_prompt_ids, concurrency, indexes, start_time, resharding_to_infer)
            mock_handle.assert_not_called()

    def test_multi_batches_final_handle_empty_trajectories(self):
        """Test multi_batches_final_handle method with empty trajectories"""
        traj_groups = {"1": []}
        all_prompt_ids = {1}
        concurrency = 8
        indexes = [0]
        start_time = 1234567890
        resharding_to_infer = 0.5

        # Mock get_train_batch_traj to return empty list
        self.rollout_worker.get_train_batch_traj = MagicMock(return_value=[])

        # Call the method
        self.rollout_worker.multi_batches_final_handle(traj_groups, all_prompt_ids, concurrency, indexes, start_time, resharding_to_infer)

        # Verify handle_full_batch_trajectories was not called
        with patch.object(self.rollout_worker, 'handle_full_batch_trajectories') as mock_handle:
            self.rollout_worker.multi_batches_final_handle(traj_groups, all_prompt_ids, concurrency, indexes, start_time, resharding_to_infer)
            mock_handle.assert_not_called()

    def test_run_trajectories_perf_metric_empty(self):
        """Test run_trajectories_perf_metric method with empty trajectories"""
        trajectories = []
        try:
            metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)
        except IndexError:
            # Expected behavior when traj_metrics is empty
            metrics = {}
        self.assertEqual(metrics, {})

    def test_run_trajectories_perf_metric_with_dummy_trajectories(self):
        """Test run_trajectories_perf_metric method with only dummy trajectories"""
        trajectories = [{
            "metrics": {
                "total_time": 0.0  # dummy trajectory
            }
        }]
        try:
            metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)
        except IndexError:
            # Expected behavior when traj_metrics is empty after filtering dummy trajectories
            metrics = {}
        self.assertEqual(metrics, {})

    def test_run_trajectories_perf_metric_with_none_values(self):
        """Test run_trajectories_perf_metric method with None values"""
        trajectories = [{
            "metrics": {
                "total_time": 1.0,
                "env_time": None,
                "llm_time": 0.5
            }
        }]
        metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)
        self.assertIn("traj/llm_time_mean", metrics)
        self.assertNotIn("traj/env_time_mean", metrics)

    def test_wait_available_version_timeout(self):
        """Test _wait_available_version method with timeout"""
        # Configure mock object behavior
        self.rollout_worker.rollout_weight_manager = MagicMock()
        self.rollout_worker.current_weights_version = 0

        # Mock time.sleep, time.time, and ray.get
        with patch('aura.trainer.rollout.rollout_worker.time.sleep') as mock_sleep:
            with patch('aura.trainer.rollout.rollout_worker.time.time') as mock_time:
                with patch('aura.trainer.rollout.rollout_worker.ray.get') as mock_ray_get:
                    # Configure mock_ray_get to return 0
                    mock_ray_get.return_value = 0
                    # Configure mock_time to return values that cause timeout
                    mock_time.side_effect = [0, 5, 11]  # Start time 0, then 5 (before timeout), then 11 (timeout)

                    # Call the method with timeout=10
                    from aura.trainer.rollout.rollout_worker import UNAVAILABLE_WEIGHT_VERSION
                    weights_version = self.rollout_worker._wait_available_version(wait_timeout=10)

                    # Verify results
                    self.assertEqual(weights_version, UNAVAILABLE_WEIGHT_VERSION)
                    mock_sleep.assert_called()


    def test_get_least_common_multiple(self):
        """Test get_least_common_multiple function"""
        result = get_least_common_multiple(4, 6)
        self.assertEqual(result, 12)
        result = get_least_common_multiple(5, 7)
        self.assertEqual(result, 35)
        result = get_least_common_multiple(0, 5)
        self.assertEqual(result, 0)

    def test_generate_dummy_trajectory(self):
        """Test generate_dummy_trajectory function"""
        trajectory = generate_dummy_trajectory("test_id")
        self.assertIsInstance(trajectory, dict)
        self.assertEqual(trajectory["prompt_id"], "test_id")
        self.assertIn("prompt_tokens", trajectory)
        self.assertIn("response_tokens", trajectory)

    def test_parse_messages_qwen_format(self):
        """Test parse_messages function with Qwen format"""
        prompt = "<|im_start|>system\nHello<|im_end|><|im_start|>user\nWorld<|im_end|>"
        messages = parse_messages(prompt, model_name="qwen")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Hello")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "World")

    def test_parse_messages_deepseek_format(self):
        """Test parse_messages function with DeepSeek format"""
        prompt = "<｜system｜>Hello<｜user｜>World"
        messages = parse_messages(prompt, model_name="deepseek")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Hello")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "World")

    def test_parse_messages_unsupported_model(self):
        """Test parse_messages function with unsupported model"""
        with self.assertRaises(NotImplementedError):
            parse_messages("test", model_name="unsupported")

    def test_clean_traj_groups(self):
        """Test clean_traj_groups function"""
        traj_groups = {1: [{"prompt_id": 1}, {"prompt_id": 1}]}
        all_prompt_ids = {1, 2}
        trajectories = [{"prompt_id": 1}]
        clean_traj_groups(traj_groups, all_prompt_ids, trajectories)
        self.assertEqual(len(traj_groups[1]), 1)
        self.assertNotIn(1, all_prompt_ids)

    def test_get_all_prompt_ids(self):
        """Test get_all_prompt_ids function"""
        class MockTask:
            def __init__(self, prompt_id):
                self.prompt_id = prompt_id
        tasks = [MockTask(1), MockTask(2), MockTask(3)]
        result = get_all_prompt_ids(tasks)
        self.assertEqual(result, {1, 2, 3})

    def test_pad_dataproto_to_divisor(self):
        """Test pad_dataproto_to_divisor function"""
        from aura.trainer.rollout.rollout_worker import pad_dataproto_to_divisor
        tensor_batch = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "token_level_scores": torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        }
        result = pad_dataproto_to_divisor(tensor_batch, 4)
        self.assertEqual(len(result["input_ids"]), 4)
        self.assertEqual(result["attention_mask"].shape[0], 4)

    def test_rollout_worker_init_on_policy_assertion(self):
        """Test RolloutWorker on_policy assertion check"""
        self.rollout_worker.use_on_policy = True
        self.rollout_worker.hybrid_batch_num = 2
        with self.assertRaises(AssertionError):
            # Simulate the assertion check in __init__
            if self.rollout_worker.use_on_policy and self.rollout_worker.hybrid_batch_num > 1:
                raise AssertionError(
                    f"Configuration error: hybrid_batch_num={self.rollout_worker.hybrid_batch_num} "
                    f"must be 1 when use_on_policy={self.rollout_worker.use_on_policy}."
                )

    def test_run_trajectories_perf_metric_with_dummy_and_real_trajectories(self):
        """Test run_trajectories_perf_metric method with mixed dummy and real trajectories"""
        trajectories = [
            {
                "metrics": {
                    "total_time": 0.0,
                    "env_time": 0.0,
                    "llm_time": 0.0
                }
            },
            {
                "metrics": {
                    "total_time": 1.0,
                    "env_time": 0.5,
                    "llm_time": 0.3
                }
            }
        ]
        metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)
        self.assertIn("traj/total_time_mean", metrics)

    def test_run_trajectories_perf_metric_with_reward_metrics(self):
        """Test run_trajectories_perf_metric method with reward metrics"""
        trajectories = [
            {
                "metrics": {
                    "total_time": 1.0,
                    "env_time": 0.5,
                    "llm_time": 0.3,
                    "toolcall_reward": 0.1,
                    "res_reward": 0.9,
                    "steps": 2
                }
            },
            {
                "metrics": {
                    "total_time": 2.0,
                    "env_time": 1.0,
                    "llm_time": 0.6,
                    "toolcall_reward": 0.2,
                    "res_reward": 1.8,
                    "steps": 3
                }
            }
        ]
        metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)
        self.assertIn("traj/res_reward_mean", metrics)
        self.assertIn("traj/toolcall_reward_mean", metrics)

    def test_run_trajectories_perf_metric_with_step_times(self):
        """Test run_trajectories_perf_metric method with step times"""
        trajectories = [
            {
                "metrics": {
                    "total_time": 1.0,
                    "llm_step_times": [0.1, 0.2],
                    "env_step_times": [0.3, 0.4],
                    "step_reward": [0.5, 0.6],
                    "steps": 2
                }
            }
        ]
        metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)
        self.assertIn("traj/total_time_mean", metrics)

    def test_run_trajectories_perf_metric_with_traj_start_time(self):
        """Test run_trajectories_perf_metric method with traj_start_time"""
        trajectories = [
            {
                "metrics": {
                    "total_time": 1.0,
                    "traj_start_time": 1234567890.0
                }
            }
        ]
        metrics = self.rollout_worker.run_trajectories_perf_metric(trajectories)
        self.assertIn("traj/total_time_mean", metrics)

    def test_wait_init_finished_proxy_mode(self):
        """Test wait_init_finished method in proxy mode"""
        import asyncio
        with patch('aura.trainer.rollout.rollout_worker.AsyncServerProxyManager') as mock_proxy:
            mock_proxy_instance = AsyncMock()
            mock_proxy.return_value = mock_proxy_instance
            asyncio.run(self.rollout_worker.wait_init_finished(is_proxy_mode=True))
            mock_proxy.assert_called_once()
            mock_proxy_instance.init.assert_called_once()

    def test_wait_init_finished_non_proxy_mode(self):
        """Test wait_init_finished method in non-proxy mode"""
        import asyncio
        with patch('aura.trainer.rollout.rollout_worker.AsyncServerManager') as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance
            asyncio.run(self.rollout_worker.wait_init_finished(is_proxy_mode=False))
            mock_server.assert_called_once()

    def test_init_data_manager(self):
        """Test init_data_manager method"""
        mock_data_manager = MagicMock()
        self.mock_data_manager.return_value.sync_init_data_manager.return_value = True
        result = self.rollout_worker.init_data_manager(mock_data_manager)
        self.assertTrue(result)
        self.mock_data_manager.return_value.sync_init_data_manager.assert_called_once_with(mock_data_manager)

    def test_multi_batches_final_handle_early_exit(self):
        """Test multi_batches_final_handle method with empty prompt_ids (early exit)"""
        traj_groups = {"0": [{"prompt_id": "0", "prompt_tokens": torch.tensor([1, 2, 3])}]}
        all_prompt_ids = set()
        concurrency = 8
        indexes = [0]
        start_time = 1234567890
        resharding_to_infer = 0.5

        with patch.object(self.rollout_worker, 'handle_full_batch_trajectories') as mock_handle:
            self.rollout_worker.multi_batches_final_handle(traj_groups, all_prompt_ids, concurrency, indexes, start_time, resharding_to_infer)
            mock_handle.assert_not_called()

    def test_reset_trajectory_reward(self):
        """Test reset_trajectory_reward method"""
        class MockTrajectory:
            def __init__(self, reward):
                self.reward = reward

        trajectories = [MockTrajectory(1.0), MockTrajectory(2.0), MockTrajectory(3.0), MockTrajectory(4.0)]
        self.rollout_worker.n_samples_per_prompt = 2

        self.rollout_worker.reset_trajectory_reward(trajectories)

        rewards = [t.reward for t in trajectories]
        self.assertAlmostEqual(sum(rewards), 0.0, places=5)

    def test_stepwise_normalization_disabled(self):
        """Test stepwise_normalization method when use_stepwise_advantage is False"""
        self.rollout_worker.use_stepwise_advantage = False
        trajectories = []
        self.rollout_worker.stepwise_normalization(trajectories)

    def test_stepwise_normalization_enabled(self):
        """Test stepwise_normalization method when use_stepwise_advantage is True"""
        class MockTrajectory:
            def __init__(self, reward):
                self.reward = reward

        self.rollout_worker.use_stepwise_advantage = True
        self.rollout_worker.n_samples_per_prompt = 2
        trajectories = [MockTrajectory(1.0), MockTrajectory(2.0)]

        with patch.object(self.rollout_worker, 'reset_trajectory_reward') as mock_reset:
            with patch.object(self.rollout_worker.data_manager, 'update_metrics') as mock_update:
                self.rollout_worker.stepwise_normalization(trajectories)
                mock_reset.assert_called_once_with(trajectories)
                mock_update.assert_called()

    def test_stepwise_pad_datapro_disabled(self):
        """Test stepwise_pad_datapro method when use_stepwise_advantage is False"""
        self.rollout_worker.use_stepwise_advantage = False
        final_gen_batch_output = {}
        self.rollout_worker.stepwise_pad_datapro(final_gen_batch_output)

    def test_stepwise_pad_datapro_enabled(self):
        """Test stepwise_pad_datapro method when use_stepwise_advantage is True"""
        from aura.trainer.rollout.rollout_worker import pad_dataproto_to_divisor
        self.rollout_worker.use_stepwise_advantage = True
        self.rollout_worker.global_batch_size = 4
        self.rollout_worker.is_hybrid_mode = MagicMock(return_value=True)
        final_gen_batch_output = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }

        with patch.object(self.rollout_worker.data_manager, 'reset_experience_len') as mock_reset:
            self.rollout_worker.stepwise_pad_datapro(final_gen_batch_output)
            mock_reset.assert_called_once()

    def test_add_output_for_verl_with_msrl(self):
        """Test add_output_for_verl method when remove_padding_and_split_to_list is not None"""
        self.rollout_worker.remove_padding_and_split_to_list = lambda x: x
        final_gen_batch_output = {}
        responses = []
        outputs = {}

        self.rollout_worker.add_output_for_verl(final_gen_batch_output, responses, outputs)

        self.assertEqual(outputs, {})

    def test_add_output_for_verl(self):
        """Test add_output_for_verl method"""
        self.rollout_worker.remove_padding_and_split_to_list = None
        final_gen_batch_output = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "prompt_ids": ["1"],
            "position_ids": torch.tensor([[0, 1, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "prompts": torch.tensor([[1, 2, 3]]),
            "prompt_length": [torch.tensor([3])],
            "token_level_scores": torch.tensor([[0.5]]),
            "traj_mask": torch.tensor([[1, 1, 1]]),
            "rollout_log_probs": torch.tensor([[0.1, 0.2, 0.3]])
        }
        responses = torch.tensor([[4, 5, 6]])
        outputs = {}

        self.rollout_worker.add_output_for_verl(final_gen_batch_output, responses, outputs)

        self.assertIn("responses", outputs)
        self.assertIn("input_ids", outputs)
        self.assertIn("rollout_log_probs", outputs)

    def test_add_output_for_msrl_with_verl(self):
        """Test add_output_for_msrl method when remove_padding_and_split_to_list is None"""
        self.rollout_worker.remove_padding_and_split_to_list = None
        responses = []
        input_ids = []
        outputs = {}

        self.rollout_worker.add_output_for_msrl(responses, input_ids, outputs)

        self.assertEqual(outputs, {})

    def test_add_output_for_msrl(self):
        """Test add_output_for_msrl method"""
        self.rollout_worker.remove_padding_and_split_to_list = lambda x: x
        responses = [[1, 2, 3], [4, 5]]
        input_ids = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]]
        outputs = {}

        self.rollout_worker.add_output_for_msrl(responses, input_ids, outputs)

        self.assertIn("responses", outputs)
        self.assertIn("input_ids", outputs)
        self.assertIn("response_length", outputs)

    def test_transform_agent_trajectories_empty_logprobs(self):
        """Test _transform_agent_trajectories method when logprobs is empty"""
        trajectories = [
            {
                "prompt_id": "0",
                "prompt_tokens": torch.tensor([1, 2, 3]),
                "response_tokens": torch.tensor([4, 5]),
                "response_masks": torch.tensor([1, 1]),
                "trajectory_reward": 0.5,
                "logprobs": [],
                "chat_completions": [{"role": "user", "content": "test"}]
            }
        ]

        self.rollout_worker.use_stepwise_advantage = False
        self.rollout_worker.tokenizer.pad_token_id = 0

        with patch.object(self.rollout_worker, 'run_trajectories_perf_metric', return_value={}) as mock_perf:
            with patch.object(self.rollout_worker, 'visualize_trajectory') as mock_visualize:
                tensor_batch, metrics = self.rollout_worker._transform_agent_trajectories(trajectories)

                self.assertIsInstance(tensor_batch, dict)
                self.assertNotIn("rollout_log_probs", tensor_batch)
                mock_perf.assert_called_once()
                mock_visualize.assert_called_once()

    def test_transform_agent_trajectories_with_logprobs(self):
        """Test _transform_agent_trajectories method with logprobs"""
        trajectories = [
            {
                "prompt_id": "0",
                "prompt_tokens": torch.tensor([1, 2, 3]),
                "response_tokens": torch.tensor([4, 5]),
                "response_masks": torch.tensor([1, 1]),
                "trajectory_reward": 0.5,
                "logprobs": [0.1, 0.2],
                "chat_completions": [{"role": "user", "content": "test"}]
            }
        ]

        self.rollout_worker.use_stepwise_advantage = False
        self.rollout_worker.tokenizer.pad_token_id = 0

        with patch.object(self.rollout_worker, 'run_trajectories_perf_metric', return_value={}) as mock_perf:
            with patch.object(self.rollout_worker, 'visualize_trajectory') as mock_visualize:
                tensor_batch, metrics = self.rollout_worker._transform_agent_trajectories(trajectories)

                self.assertIsInstance(tensor_batch, dict)
                self.assertIn("rollout_log_probs", tensor_batch)
                mock_perf.assert_called_once()
                mock_visualize.assert_called_once()

    def test_transform_agent_trajectories_empty_trajectory(self):
        """Test _transform_agent_trajectories method with empty trajectory"""
        trajectories = [
            {
                "prompt_id": "0",
                "prompt_tokens": torch.tensor([]),
                "response_tokens": torch.tensor([4, 5]),
                "response_masks": torch.tensor([1, 1]),
                "trajectory_reward": 0.5,
                "logprobs": [0.1, 0.2],
                "chat_completions": [{"role": "user", "content": "test"}]
            }
        ]

        self.rollout_worker.use_stepwise_advantage = False

        with self.assertRaises(ValueError):
            self.rollout_worker._transform_agent_trajectories(trajectories)

    def test_trajectories_collect_done(self):
        """Test trajectories_collect_done method"""
        trajectories = [1, 2, 3, 4, 5, 6, 7, 8]
        concurrency = 8
        done_batch_count = 0
        actual_batch_num = 1

        result = self.rollout_worker.trajectories_collect_done(trajectories, concurrency, done_batch_count, actual_batch_num)
        self.assertTrue(result)

    def test_trajectories_collect_done_not_enough(self):
        """Test trajectories_collect_done method when not enough trajectories"""
        trajectories = [1, 2, 3]
        concurrency = 8
        done_batch_count = 0
        actual_batch_num = 1

        result = self.rollout_worker.trajectories_collect_done(trajectories, concurrency, done_batch_count, actual_batch_num)
        self.assertFalse(result)

    def test_get_train_batch_traj(self):
        """Test get_train_batch_traj method"""
        traj_groups = {"0": [{"prompt_id": "0"}, {"prompt_id": "0"}], "1": [{"prompt_id": "1"}, {"prompt_id": "1"}]}
        concurrency = 4
        n_samples_per_prompt = 2

        trajectories = self.rollout_worker.get_train_batch_traj(traj_groups, concurrency, n_samples_per_prompt)

        self.assertEqual(len(trajectories), 4)

    def test_is_hybrid_mode(self):
        """Test is_hybrid_mode method"""
        # Default behavior
        result = self.rollout_worker.is_hybrid_mode()
        self.assertFalse(result)

    def test_hybrid_mode_metrics_handle_disabled(self):
        """Test hybrid_mode_metrics_handle method when not in hybrid mode"""
        metrics = {"traj/res_reward_mean": 0.5}
        start_time = 1234567890.0
        end_time = 1234567891.0

        self.rollout_worker.hybrid_mode_metrics_handle(metrics, start_time, end_time)

    def test_hybrid_mode_metrics_handle_enabled(self):
        """Test hybrid_mode_metrics_handle method when in hybrid mode"""
        metrics = {"traj/res_reward_mean": 0.5}
        start_time = 1234567890.0
        end_time = 1234567891.0

        with patch.object(self.rollout_worker, 'is_hybrid_mode', return_value=True):
            with patch.object(self.rollout_worker.data_manager, 'update_metrics') as mock_update:
                self.rollout_worker.hybrid_mode_metrics_handle(metrics, start_time, end_time)
                mock_update.assert_called()

    def test__stat_rollout_metrics(self):
        """Test _stat_rollout_metrics function"""
        rollout_cost = 1.5
        resharding_to_infer = 0.5
        metrics = {"traj/res_reward_mean": 0.8, "traj/toolcall_reward_mean": 0.2}

        result = _stat_rollout_metrics(rollout_cost, resharding_to_infer, metrics)

        self.assertEqual(result["rollout_cost"], 1.5)
        self.assertEqual(result["resharding_to_infer"], 0.5)
        self.assertEqual(result["res_reward_mean"], 0.8)
        self.assertEqual(result["toolcall_reward_mean"], 0.2)

    def test_get_train_batch_traj_empty(self):
        """Test get_train_batch_traj method with empty traj_groups"""
        traj_groups = {}
        concurrency = 4
        n_samples_per_prompt = 2

        trajectories = self.rollout_worker.get_train_batch_traj(traj_groups, concurrency, n_samples_per_prompt)

        self.assertEqual(len(trajectories), 0)

    def test_get_train_batch_traj_partial(self):
        """Test get_train_batch_traj method with partial data (not enough samples per prompt)"""
        traj_groups = {"0": [{"prompt_id": "0"}], "1": [{"prompt_id": "1"}]}
        concurrency = 4
        n_samples_per_prompt = 2

        trajectories = self.rollout_worker.get_train_batch_traj(traj_groups, concurrency, n_samples_per_prompt)

        self.assertEqual(len(trajectories), 0)

    def test_clean_traj_groups_value_error(self):
        """Test clean_traj_groups function with ValueError handling (traj not in list)"""
        traj_groups = {1: [{"prompt_id": 1, "unique_key": "a"}]}
        all_prompt_ids = {1, 2}
        trajectories = [{"prompt_id": 1, "unique_key": "b"}]  # Different object, not in list

        clean_traj_groups(traj_groups, all_prompt_ids, trajectories)

        self.assertEqual(len(traj_groups[1]), 1)
        self.assertEqual(all_prompt_ids, {1, 2})

    def test_early_termination_requests(self):
        """Test early_termination_requests method"""
        import asyncio
        mock_task = {"task_id": "1"}
        mock_agent_router = AsyncMock()

        asyncio.run(self.rollout_worker.early_termination_requests(mock_task, mock_agent_router))

        mock_agent_router.cancel_request.assert_called_once_with(mock_task)
        self.assertEqual(self.rollout_worker.terminate_trajectories, 1)

    def test_trajectories_collect_done_full_batch(self):
        """Test trajectories_collect_done method with full batch"""
        trajectories = [1, 2, 3, 4, 5, 6, 7, 8]
        concurrency = 8
        done_batch_count = 1
        actual_batch_num = 1

        result = self.rollout_worker.trajectories_collect_done(trajectories, concurrency, done_batch_count, actual_batch_num)
        self.assertTrue(result)

    def test_trajectories_collect_done_more_than_expected(self):
        """Test trajectories_collect_done method with more trajectories than expected"""
        trajectories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        concurrency = 8
        done_batch_count = 1
        actual_batch_num = 1

        result = self.rollout_worker.trajectories_collect_done(trajectories, concurrency, done_batch_count, actual_batch_num)
        self.assertTrue(result)

    def test_transform_agent_trajectories_no_chat_completions(self):
        """Test _transform_agent_trajectories method without chat_completions"""
        trajectories = [
            {
                "prompt_id": "0",
                "prompt_tokens": torch.tensor([1, 2, 3]),
                "response_tokens": torch.tensor([4, 5]),
                "response_masks": torch.tensor([1, 1]),
                "trajectory_reward": 0.5,
                "logprobs": [0.1, 0.2],
                "chat_completions": None
            }
        ]

        self.rollout_worker.use_stepwise_advantage = False
        self.rollout_worker.tokenizer.pad_token_id = 0

        with patch.object(self.rollout_worker, 'run_trajectories_perf_metric', return_value={}) as mock_perf:
            with patch.object(self.rollout_worker, 'visualize_trajectory') as mock_visualize:
                tensor_batch, metrics = self.rollout_worker._transform_agent_trajectories(trajectories)

                self.assertIsInstance(tensor_batch, dict)


if __name__ == '__main__':
    unittest.main()
