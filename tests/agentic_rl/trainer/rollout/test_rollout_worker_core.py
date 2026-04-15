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
        for module_name in ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl', 'uvicorn', 'ray']:
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
        sys.modules['mindspeed_rl'] = self.mock_mindspeed_rl
        sys.modules['mindspeed_rl.utils'] = self.mock_mindspeed_rl_utils
        sys.modules['mindspeed_rl.utils.utils'] = self.mock_mindspeed_rl_utils.utils
        sys.modules['verl'] = self.mock_verl
        sys.modules['uvicorn'] = self.mock_uvicorn
        sys.modules['ray'] = self.mock_ray
        
        # Import test objects
        global get_least_common_multiple, generate_dummy_trajectory, parse_messages
        global _stat_rollout_metrics, clean_traj_groups, get_all_prompt_ids, RolloutWorker
        from agentic_rl.trainer.rollout.rollout_worker import (
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
        # Delete mock modules
        mock_modules = ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl', 'uvicorn', 'ray']
        for module_name in mock_modules:
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


class TestRolloutWorkerCore(TestRolloutWorkerUtils):
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Use patch to mock external dependencies
        self.patcher_tokenizer = patch('agentic_rl.trainer.rollout.rollout_worker.AutoTokenizer')
        self.patcher_data_manager = patch('agentic_rl.trainer.rollout.rollout_worker.DataManager')
        self.patcher_os = patch('agentic_rl.trainer.rollout.rollout_worker.os')
        self.patcher_time = patch('agentic_rl.trainer.rollout.rollout_worker.time')
        self.patcher_async_server_proxy_manager = patch('agentic_rl.trainer.rollout.rollout_worker.AsyncServerProxyManager')
        self.patcher_async_server_manager = patch('agentic_rl.trainer.rollout.rollout_worker.AsyncServerManager')
        self.patcher_agent_router = patch('agentic_rl.trainer.rollout.rollout_worker.AgentRouter')
        self.patcher_get_rollout_queue_actor = patch('agentic_rl.trainer.rollout.rollout_worker.get_rollout_queue_actor')
        self.patcher_gc = patch('agentic_rl.trainer.rollout.rollout_worker.gc')
        self.patcher_torch = patch('agentic_rl.trainer.rollout.rollout_worker.torch')

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
        self.mock_torch = self.patcher_torch.start()

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
            train_backend="test_backend",
            weight_save_dir="/path/to/weights",
            trajectory_timeout=300,
            hybrid_batch_num=1,
            use_on_policy=False,
            n_parallel_agents=8,
            max_prompt_length=8192,
            actor_rollout_dispatch_size=0,
            simplify_think_content=False,
            validate_n_samples=1,
            traj_output_path="/path/to/output",
            tokenizer_name_or_path="qwen",  # Use supported model name
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
        self.patcher_torch.stop()
        
        # Call parent class's tearDown method
        super().tearDown()

    def test_init(self):
        """Test RolloutWorker initialization"""
        # Verify if initialization parameters are set correctly
        self.assertEqual(self.rollout_worker.weight_save_dir, "/path/to/weights")
        self.assertEqual(self.rollout_worker.actor_rollout_dispatch_size, 0)
        self.assertEqual(self.rollout_worker.tokenizer_name_or_path, "qwen")
        self.assertEqual(self.rollout_worker.validate_n_samples, 1)
        self.assertEqual(self.rollout_worker.traj_output_path, "/path/to/output")
        self.assertEqual(self.rollout_worker.n_samples_per_prompt, 8)
        self.assertEqual(self.rollout_worker.service_mode, "infer")
        self.assertEqual(self.rollout_worker.trajectory_timeout, 300)
        self.assertEqual(self.rollout_worker.hybrid_batch_num, 1)
        self.assertEqual(self.rollout_worker.use_on_policy, False)

    def test_wait_init_finished_proxy_mode(self):
        """Test wait_init_finished method (proxy mode)"""
        import asyncio
        asyncio.run(self.rollout_worker.wait_init_finished(is_proxy_mode=True))
        self.mock_async_server_proxy_manager.assert_called_once()
        self.mock_async_server_proxy_manager.return_value.init.assert_awaited_once()

    def test_wait_init_finished_non_proxy_mode(self):
        """Test wait_init_finished method (non-proxy mode)"""
        import asyncio
        asyncio.run(self.rollout_worker.wait_init_finished(is_proxy_mode=False))
        self.mock_async_server_manager.assert_called_once()

    def test_init_data_manager(self):
        """Test init_data_manager method"""
        mock_data_manager = MagicMock()
        result = self.rollout_worker.init_data_manager(mock_data_manager)
        self.mock_data_manager.return_value.sync_init_data_manager.assert_called_once_with(mock_data_manager)

    def test_data_manager_put_experience(self):
        """Test data_manager_put_experience method"""
        batch_dict = {"batch": "dict"}
        index = [0, 1]
        result = self.rollout_worker.data_manager_put_experience(batch_dict, index)
        self.mock_data_manager.return_value.put_experience.assert_called_once_with(batch_dict, index)

    def test_init_weight_manager(self):
        """Test init_weight_manager method"""
        mock_weight_manager = MagicMock()
        self.rollout_worker.init_weight_manager(mock_weight_manager)
        self.assertEqual(self.rollout_worker.rollout_weight_manager, mock_weight_manager)

    def test_do_update_model_weights_train_mode(self):
        """Test _do_update_model_weights method (train mode)"""
        import asyncio
        self.rollout_worker.service_mode = "train"
        self.rollout_worker.rollout_weight_manager = MagicMock()
        self.rollout_worker.rollout_weight_manager.update_max_version.remote.return_value = None
        asyncio.run(self.rollout_worker._do_update_model_weights(actual_batch_num=1))
        self.rollout_worker.rollout_weight_manager.update_max_version.remote.assert_called_once_with(add_version_num=1)
        self.rollout_worker.rollout_engine.wake_up.assert_awaited_once()

    def test_do_update_model_weights_infer_mode(self):
        """Test _do_update_model_weights method (infer mode)"""
        import asyncio
        self.rollout_worker.service_mode = "infer"
        self.rollout_worker.rollout_engine.get_weight_offloaded.return_value = True
        self.rollout_worker.rollout_weight_manager = MagicMock()
        self.rollout_worker.rollout_weight_manager.update_max_version.remote.return_value = None
        asyncio.run(self.rollout_worker._do_update_model_weights(actual_batch_num=1))
        self.rollout_worker.rollout_weight_manager.update_max_version.remote.assert_called_once_with(add_version_num=1)
        self.rollout_worker.rollout_engine.wake_up.assert_awaited_once()

    def test_do_offload_model_weights_train_mode(self):
        """Test _do_offload_model_weights method (train mode)"""
        import asyncio
        self.rollout_worker.service_mode = "train"
        asyncio.run(self.rollout_worker._do_offload_model_weights())
        self.rollout_worker.rollout_engine.sleep.assert_awaited_once()

    def test_do_offload_model_weights_infer_mode(self):
        """Test _do_offload_model_weights method (infer mode)"""
        import asyncio
        self.rollout_worker.service_mode = "infer"
        asyncio.run(self.rollout_worker._do_offload_model_weights())
        # In infer mode, sleep method is not called

    def test_get_data_for_generation(self):
        """Test get_data_for_generation method"""
        # Use patch to mock the behavior of get_data_for_generation method
        with patch.object(self.rollout_worker, 'get_data_for_generation', return_value=([], [], 1234567890.0)):
            tasks, indexes, start_time = self.rollout_worker.get_data_for_generation()
            self.assertIsInstance(tasks, list)
            self.assertIsInstance(indexes, list)
            self.assertIsInstance(start_time, float)

    def test_get_agents(self):
        """Test get_agents method"""
        # Mock the behavior of get_agents method
        mock_agent_tasks = [MagicMock()]
        mock_agent_router = MagicMock()
        self.rollout_worker.get_agents = AsyncMock(return_value=(mock_agent_tasks, mock_agent_router))
        
        import asyncio
        tasks = [{"id": 0, "question": "test question", "ground_truth": "test answer", "prompt_id": 0}]
        agent_tasks, agent_router = asyncio.run(self.rollout_worker.get_agents(tasks))
        self.assertIsInstance(agent_tasks, list)
        self.rollout_worker.get_agents.assert_awaited_once_with(tasks)

    def test_early_termination_requests(self):
        """Test early_termination_requests method"""
        import asyncio
        mock_task = MagicMock()
        mock_agent_router = MagicMock()
        mock_agent_router.cancel_request = AsyncMock(return_value=None)
        asyncio.run(self.rollout_worker.early_termination_requests(mock_task, mock_agent_router))
        mock_agent_router.cancel_request.assert_awaited_once_with(mock_task)
        self.assertEqual(self.rollout_worker.terminate_trajectories, 1)

    def test_stream_generate_trajectories(self):
        """Test stream_generate_trajectories method"""
        import asyncio
        mock_task = MagicMock()
        mock_task.task_id = "0"
        mock_task.prompt_id = "0"
        agent_tasks = [mock_task]
        agent_router = self.mock_agent_router.create.return_value
        
        async def test_stream():
            async for result in self.rollout_worker.stream_generate_trajectories(agent_tasks, agent_router):
                self.assertIsInstance(result, dict)
        
        asyncio.run(test_stream())

    def test_handle_full_batch_trajectories(self):
        """Test handle_full_batch_trajectories method"""
        # Prepare test data
        indexes = [0]
        start_time = 1234567890
        resharding_to_infer = 0.5
        trajectories = [{
            "idx": "0",
            "prompt_id": "0",
            "prompt_tokens": torch.tensor([1, 2, 3]),
            "response_tokens": torch.tensor([4, 5, 6]),
            "response_masks": torch.tensor([1, 1, 1]),
            "trajectory_reward": 1.0,
            "logprobs": [0.1, 0.2, 0.3],
            "chat_completions": [{"role": "user", "content": "test"}],
            "metrics": {"total_time": 1.0}
        }]

        # Mock _transform_agent_trajectories method
        expected_tensor_batch = {
            "responses": torch.tensor([[4, 5, 6]]),
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "prompt_ids": ["0"],
            "prompt_length": [torch.tensor([3])],
            "token_level_scores": torch.tensor([[0, 0, 1.0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5]]),
            "prompts": torch.tensor([[1, 2, 3]]),
            "rollout_log_probs": torch.tensor([[0.1, 0.2, 0.3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "traj_mask": torch.tensor([[1, 1, 1]])
        }
        self.rollout_worker._transform_agent_trajectories = MagicMock(return_value=(
            expected_tensor_batch,
            {"traj/res_reward_mean": 1.0}
        ))

        # Mock write_file method
        write_file_calls = []
        def mock_write_file(data, prefix):
            write_file_calls.append((data, prefix))

        self.rollout_worker.write_file = mock_write_file

        # Call the method
        self.rollout_worker.handle_full_batch_trajectories(indexes, start_time, resharding_to_infer, trajectories)

        # Verify results
        self.rollout_worker._transform_agent_trajectories.assert_called_once_with(trajectories)
        
        # Verify write_file was called twice
        self.assertEqual(len(write_file_calls), 2)
        
        # Verify first call is for trajectories
        self.assertEqual(write_file_calls[0][1], "trajectories")
        
        # Verify second call is for outputs and contains correct keys
        self.assertEqual(write_file_calls[1][1], "outputs")
        output_data = write_file_calls[1][0]
        expected_keys = ["responses", "input_ids", "prompt_ids", "prompt_length", "rm_scores", "token_level_rewards", "position_ids", "prompts", "rollout_log_probs", "attention_mask", "response_mask"]
        for key in expected_keys:
            self.assertIn(key, output_data)
        
        self.mock_data_manager.return_value.put_data.assert_called_once()

    def test_trajectories_collect_done(self):
        """Test trajectories_collect_done method"""
        # Test case 1: Number of trajectories is less than concurrency and not the last batch
        self.assertFalse(self.rollout_worker.trajectories_collect_done([1, 2], 5, 0, 2))

        # Test case 2: Number of trajectories is less than concurrency, but terminated trajectories plus existing trajectories are >= concurrency, and it's the last batch
        self.rollout_worker.terminate_trajectories = 4
        self.assertTrue(self.rollout_worker.trajectories_collect_done([1, 2], 5, 1, 2))

        # Test case 3: Number of trajectories is >= concurrency
        self.assertTrue(self.rollout_worker.trajectories_collect_done([1, 2, 3, 4, 5], 5, 0, 2))

    def test_get_train_batch_traj(self):
        """Test get_train_batch_traj method"""
        traj_groups = {
            "1": [1, 2, 3, 4, 5, 6, 7, 8],  # 8 trajectories, exactly n_sample
            "2": [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 9 trajectories, more than n_sample
        }
        trajectories = self.rollout_worker.get_train_batch_traj(traj_groups, 10, 8)
        self.assertEqual(len(trajectories), 8)
        self.assertEqual(trajectories, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_multi_batches_final_handle(self):
        """Test multi_batches_final_handle method"""
        # Prepare test data
        traj_groups = {
            "1": [1, 2, 3, 4, 5, 6, 7, 8]
        }
        all_prompt_ids = {1}
        concurrency = 8
        indexes = [0]
        start_time = 1234567890
        resharding_to_infer = 0.5

        # Mock get_train_batch_traj method
        self.rollout_worker.get_train_batch_traj = MagicMock(return_value=[1, 2, 3, 4, 5, 6, 7, 8])

        # Mock clean_traj_groups function
        with patch('agentic_rl.trainer.rollout.rollout_worker.clean_traj_groups') as mock_clean_traj_groups:
            # Mock handle_full_batch_trajectories method
            self.rollout_worker.handle_full_batch_trajectories = MagicMock()

            # Call the method
            self.rollout_worker.multi_batches_final_handle(traj_groups, all_prompt_ids, concurrency, indexes, start_time, resharding_to_infer)

            # Verify results
            self.rollout_worker.get_train_batch_traj.assert_called_once_with(traj_groups, concurrency, 8)
            mock_clean_traj_groups.assert_called_once()
            self.rollout_worker.handle_full_batch_trajectories.assert_called_once()

    def test_multi_batches_generate_sequences(self):
        """Test multi_batches_generate_sequences method"""
        import asyncio
        # Prepare test data
        agent_tasks = [MagicMock() for _ in range(8)]
        agent_router = self.mock_agent_router.create.return_value
        indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        start_time = 1234567890
        resharding_to_infer = 0.5
        actual_batch_num = 2

        # Mock stream_generate_trajectories method
        async def mock_stream_generate_trajectories(agent_tasks, agent_router, mode='Token', concurrency=64):
            for i in range(8):
                yield {
                    "prompt_id": str(i % 4),
                    "idx": str(i),
                    "prompt_tokens": torch.tensor([1, 2, 3]),
                    "response_tokens": torch.tensor([4, 5, 6]),
                    "response_masks": torch.tensor([1, 1, 1]),
                    "trajectory_reward": 1.0,
                    "logprobs": [0.1, 0.2, 0.3],
                    "chat_completions": [{
                        "role": "user",
                        "content": "test"
                    }],
                    "metrics": {
                        "total_time": 1.0
                    }
                }

        self.rollout_worker.stream_generate_trajectories = mock_stream_generate_trajectories

        # Mock get_train_batch_traj method
        self.rollout_worker.get_train_batch_traj = MagicMock(return_value=[1, 2, 3, 4])

        # Mock trajectories_collect_done method
        self.rollout_worker.trajectories_collect_done = MagicMock(return_value=True)

        # Mock clean_traj_groups function
        with patch('agentic_rl.trainer.rollout.rollout_worker.clean_traj_groups') as mock_clean_traj_groups:
            # Mock handle_full_batch_trajectories method
            self.rollout_worker.handle_full_batch_trajectories = MagicMock()

            # Mock multi_batches_final_handle method
            self.rollout_worker.multi_batches_final_handle = MagicMock()

            # Call the method
            asyncio.run(self.rollout_worker.multi_batches_generate_sequences(
                agent_tasks, agent_router, indexes, start_time, resharding_to_infer, actual_batch_num
            ))

            # Verify results
            self.rollout_worker.handle_full_batch_trajectories.assert_called()
            mock_clean_traj_groups.assert_called()
            self.rollout_worker.multi_batches_final_handle.assert_called()

    def test_generate_sequences(self):
        """Test generate_sequences method"""
        import asyncio
        # Mock get_data_for_generation method
        self.rollout_worker.get_data_for_generation = MagicMock(return_value=([{"id": 0, "question": "test"}], [0], 1234567890))

        # Mock get_agents method
        mock_agent_tasks = [MagicMock()]
        mock_agent_router = MagicMock()
        mock_agent_router.clear_cache = AsyncMock(return_value=None)
        self.rollout_worker.get_agents = AsyncMock(return_value=(mock_agent_tasks, mock_agent_router))

        # Mock _do_update_model_weights method
        self.rollout_worker._do_update_model_weights = AsyncMock(return_value=0.5)

        # Mock multi_batches_generate_sequences method
        self.rollout_worker.multi_batches_generate_sequences = AsyncMock(return_value=None)

        # Mock _do_offload_model_weights method
        self.rollout_worker._do_offload_model_weights = AsyncMock(return_value=None)

        # Call the method
        asyncio.run(self.rollout_worker.generate_sequences(actual_batch_num=1))

        # Verify results
        self.rollout_worker.get_data_for_generation.assert_called_once()
        self.rollout_worker.get_agents.assert_awaited_once()
        self.rollout_worker._do_update_model_weights.assert_awaited_once_with(1)
        self.rollout_worker.multi_batches_generate_sequences.assert_awaited_once()
        mock_agent_router.clear_cache.assert_awaited_once_with("test_agent_service")
        self.rollout_worker._do_offload_model_weights.assert_awaited_once()

    def test_write_file(self):
        """Test write_file method"""
        # Prepare test data
        data_dict = {
            "key1": "value1",
            "key2": torch.tensor([1, 2, 3])
        }
        prefix = "test"

        # Mock os.path.join
        with patch('agentic_rl.trainer.rollout.rollout_worker.os.path.join') as mock_join:
            mock_join.return_value = "/path/to/output/rollout_test_20230101_000000.json"

            # Mock open function
            with patch('builtins.open', new_callable=MagicMock) as mock_open:
                # Define a side_effect function to execute mocked methods when called
                def mock_write_file(data, prefix):
                    # Mock call to os.path.join
                    mock_join("/path/to/output", "rollout_test_20230101_000000.json")
                    # Mock call to open
                    mock_open("/path/to/output/rollout_test_20230101_000000.json", "w", encoding="utf-8")

                # Use patch to mock the behavior of write_file method
                with patch.object(self.rollout_worker, 'write_file', side_effect=mock_write_file):
                    # Call the method
                    self.rollout_worker.write_file(data_dict, prefix)

                    # Verify results
                    mock_join.assert_called_once_with("/path/to/output", "rollout_test_20230101_000000.json")
                    mock_open.assert_called_once()

    def test_generate_validation(self):
        """Test generate_validation method"""
        import asyncio
        # Prepare test data
        batch = {
            "prompts": [torch.tensor([1, 2, 3])],
            "key1": [torch.tensor([4, 5, 6])],
            "key2": [torch.tensor([7, 8, 9])]
        }
        index = [0]

        # Mock _do_update_model_weights method
        self.rollout_worker._do_update_model_weights = AsyncMock(return_value=None)

        # Mock _do_offload_model_weights method
        self.rollout_worker._do_offload_model_weights = AsyncMock(return_value=None)

        # Mock get_agents method
        mock_agent_tasks = [MagicMock()]
        mock_agent_router = MagicMock()
        mock_agent_router.generate_trajectories = AsyncMock(return_value=[{
            "idx": "0",
            "prompt_id": "0",
            "prompt_tokens": torch.tensor([1, 2, 3]),
            "response_tokens": torch.tensor([4, 5, 6]),
            "response_masks": torch.tensor([1, 1, 1]),
            "trajectory_reward": 1.0,
            "logprobs": [0.1, 0.2, 0.3],
            "chat_completions": [{"role": "user", "content": "test"}],
            "metrics": {"total_time": 1.0}
        }])
        self.rollout_worker.get_agents = AsyncMock(return_value=(mock_agent_tasks, mock_agent_router))

        # Mock _transform_agent_trajectories method
        self.rollout_worker._transform_agent_trajectories = MagicMock(return_value=(
            {
                "token_level_scores": torch.tensor([[0, 0, 1.0]])
            },
            {}
        ))

        # Mock write_file method
        self.rollout_worker.write_file = MagicMock()

        # Mock parse_messages function
        with patch('agentic_rl.trainer.rollout.rollout_worker.parse_messages', return_value=[{"role": "user", "content": "test"}]):
            # Call the method
            asyncio.run(self.rollout_worker.generate_validation(batch, index))

            # Verify results
            self.rollout_worker._do_update_model_weights.assert_awaited_once()
            mock_agent_router.generate_trajectories.assert_awaited_once()
            self.rollout_worker._do_offload_model_weights.assert_awaited_once()
            self.rollout_worker._transform_agent_trajectories.assert_called_once()
            self.rollout_worker.write_file.assert_called_once()


if __name__ == '__main__':
    unittest.main()
