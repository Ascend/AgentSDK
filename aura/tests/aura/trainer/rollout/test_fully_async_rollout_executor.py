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
import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch


class TestFullyAsyncRolloutExecutor(unittest.TestCase):
    def setUp(self):
        """Set up test environment with mocked heavy dependencies."""
        # Ensure aura package is importable when running from repo root
        aura_src = str(Path(__file__).resolve().parents[4])
        if aura_src not in sys.path:
            sys.path.insert(0, aura_src)

        # Save original modules to restore them later
        self.original_modules = {}
        module_names = [
            'sentence_transformers', 'mindspeed_rl', 'mindspeed_rl.utils',
            'mindspeed_rl.utils.utils', 'mindspeed_rl.utils.pad_process',
            'mindspeed_rl.trainer', 'mindspeed_rl.trainer.utils',
            'mindspeed_rl.trainer.utils.transfer_dock', 'verl', 'ray',
            'ray.util', 'ray.util.scheduling_strategies', 'uvicorn', 'fastapi',
            'aura.controllers.rollout_controller.rollout_controller',
            'aura.controllers.rollout_controller.rollout_queue',
            'aura.controllers.utils.utils',
            'aura.trainer.rollout.rollout_executor',
            'aura.trainer.rollout.rollout_worker',
            'aura.trainer.rollout.fully_async_rollout_executor',
            'aura.base.log.loggers', 'aura.base.utils.pad_process',
            'aura.base.exceptions.exceptions',
        ]
        for module_name in module_names:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]

        # Mock heavy external dependencies
        self.mock_sentence_transformers = mock.MagicMock()
        self.mock_mindspeed_rl = mock.MagicMock()
        self.mock_mindspeed_rl_utils = mock.MagicMock()
        self.mock_mindspeed_rl_utils.utils = mock.MagicMock()
        self.mock_mindspeed_rl_utils.pad_process = mock.MagicMock()
        self.mock_mindspeed_rl_trainer = mock.MagicMock()
        self.mock_mindspeed_rl_trainer.utils = mock.MagicMock()
        self.mock_mindspeed_rl_trainer.utils.transfer_dock = mock.MagicMock()
        self.mock_verl = mock.MagicMock()
        self.mock_ray = mock.MagicMock()
        self.mock_ray.util = mock.MagicMock()
        self.mock_ray.util.scheduling_strategies = mock.MagicMock()
        self.mock_uvicorn = mock.MagicMock()
        self.mock_fastapi = mock.MagicMock()

        # Mock ray.get_runtime_context
        self.mock_ray_get_runtime_context = mock.Mock()
        self.mock_ray_get_runtime_context.node_id = "test_node_id"
        self.mock_ray.get_runtime_context.return_value = self.mock_ray_get_runtime_context
        # ray.get returns its argument unchanged so tests can use .remote() return values directly
        self.mock_ray.get = lambda x: x
        # ray.get_actor returns a MagicMock for any name
        self.mock_ray.get_actor = mock.Mock(return_value=mock.MagicMock())
        self.mock_ray.remote = lambda x: x

        # Replace modules
        sys.modules['sentence_transformers'] = self.mock_sentence_transformers
        sys.modules['mindspeed_rl'] = self.mock_mindspeed_rl
        sys.modules['mindspeed_rl.utils'] = self.mock_mindspeed_rl_utils
        sys.modules['mindspeed_rl.utils.utils'] = self.mock_mindspeed_rl_utils.utils
        sys.modules['mindspeed_rl.utils.pad_process'] = self.mock_mindspeed_rl_utils.pad_process
        sys.modules['mindspeed_rl.trainer'] = self.mock_mindspeed_rl_trainer
        sys.modules['mindspeed_rl.trainer.utils'] = self.mock_mindspeed_rl_trainer.utils
        sys.modules['mindspeed_rl.trainer.utils.transfer_dock'] = self.mock_mindspeed_rl_trainer.utils.transfer_dock
        sys.modules['verl'] = self.mock_verl
        sys.modules['ray'] = self.mock_ray
        sys.modules['ray.util'] = self.mock_ray.util
        sys.modules['ray.util.scheduling_strategies'] = self.mock_ray.util.scheduling_strategies
        sys.modules['uvicorn'] = self.mock_uvicorn
        sys.modules['fastapi'] = self.mock_fastapi

        # Mock aura.base.log.loggers
        mock_loggers = mock.MagicMock()
        mock_loggers.Loggers.return_value.get_logger.return_value = mock.MagicMock()
        sys.modules['aura.base.log.loggers'] = mock_loggers

        # Mock aura.base.utils.pad_process
        mock_pad_process_module = mock.MagicMock()
        mock_pad_process_module.remove_padding_tensor_dict_to_dict = mock.MagicMock()
        mock_pad_process_module.remove_padding_and_split_to_list = mock.MagicMock()
        mock_pad_process_module.padding_dict_to_tensor_dict = mock.MagicMock()
        mock_pad_process_module.put_prompts_experience = mock.MagicMock()
        sys.modules['aura.base.utils.pad_process'] = mock_pad_process_module

        # Mock aura.base.exceptions.exceptions
        mock_exceptions_module = mock.MagicMock()

        class RolloutShutdownException(Exception):
            """Raised when SampleQueue is shut down, signaling rollout to stop generation."""

            pass

        mock_exceptions_module.RolloutShutdownException = RolloutShutdownException
        sys.modules['aura.base.exceptions.exceptions'] = mock_exceptions_module

        # Mock aura.controllers.rollout_controller.rollout_queue
        mock_queue_module = mock.MagicMock()
        mock_queue_module.get_rollout_queue_actor = mock.MagicMock()
        mock_queue_module.MIN_SLEEP_TIME = 0.1
        sys.modules['aura.controllers.rollout_controller.rollout_queue'] = mock_queue_module

        # Mock aura.controllers.utils.utils
        mock_utils_module = mock.MagicMock()
        mock_utils_module.MIN_SLEEP_TIME = 0.1
        mock_utils_module.TRAIN_CONTROLLER_NAMESPACE = "test_namespace"
        sys.modules['aura.controllers.utils.utils'] = mock_utils_module

        # Mock aura.controllers.rollout_controller.rollout_controller
        mock_controller_module = mock.MagicMock()
        mock_controller_module.RolloutController = mock.MagicMock()
        sys.modules['aura.controllers.rollout_controller.rollout_controller'] = mock_controller_module

        # Mock aura.trainer.rollout.rollout_worker
        mock_worker_module = mock.MagicMock()
        mock_worker_module.RolloutWorker = mock.MagicMock()
        sys.modules['aura.trainer.rollout.rollout_worker'] = mock_worker_module

        # Mock aura.trainer.rollout.rollout_executor
        mock_executor_module = mock.MagicMock()
        mock_executor_module.OneStepOffRolloutExecutor = mock.MagicMock()
        sys.modules['aura.trainer.rollout.rollout_executor'] = mock_executor_module

        # Remove cached fully_async_rollout_executor module to force reimport
        sys.modules.pop('aura.trainer.rollout.fully_async_rollout_executor', None)

        # Import the test object
        executor_mod = importlib.import_module('aura.trainer.rollout.fully_async_rollout_executor')
        self.FullyAsyncRolloutExecutor = executor_mod.FullyAsyncRolloutExecutor
        self.RolloutShutdownException = RolloutShutdownException

    def tearDown(self):
        """Clean up test environment."""
        for module_name, module in self.original_modules.items():
            sys.modules[module_name] = module
        mock_modules = [
            'sentence_transformers', 'mindspeed_rl', 'mindspeed_rl.utils',
            'mindspeed_rl.utils.utils', 'mindspeed_rl.utils.pad_process',
            'mindspeed_rl.trainer', 'mindspeed_rl.trainer.utils',
            'mindspeed_rl.trainer.utils.transfer_dock', 'verl', 'ray',
            'ray.util', 'ray.util.scheduling_strategies', 'uvicorn', 'fastapi',
            'aura.controllers.rollout_controller.rollout_controller',
            'aura.controllers.rollout_controller.rollout_queue',
            'aura.controllers.utils.utils',
            'aura.trainer.rollout.rollout_executor',
            'aura.trainer.rollout.rollout_worker',
            'aura.trainer.rollout.fully_async_rollout_executor',
            'aura.base.log.loggers', 'aura.base.utils.pad_process',
            'aura.base.exceptions.exceptions',
        ]
        for module_name in mock_modules:
            if module_name in sys.modules and module_name not in self.original_modules:
                del sys.modules[module_name]

    def _create_executor(self, train_iters=10, hybrid_batch_num=2):
        """Helper to create a FullyAsyncRolloutExecutor with mocked deps."""
        mock_controller = MagicMock()
        mock_rollout_worker = MagicMock()
        mock_padding_fn = MagicMock()
        mock_put_prompts_fn = MagicMock()

        executor = self.FullyAsyncRolloutExecutor(
            controller=mock_controller,
            rollout_worker=mock_rollout_worker,
            train_iters=train_iters,
            padding_dict_to_tensor_dict=mock_padding_fn,
            put_prompts_experience=mock_put_prompts_fn,
            data_optimized=False,
            dataset_additional_keys=["key1"],
            n_samples_per_prompt=2,
            hybrid_batch_num=hybrid_batch_num,
        )
        return executor, mock_controller, mock_rollout_worker

    def test_init_sets_attributes(self):
        """Test that __init__ correctly stores all kwargs as attributes."""
        executor, controller, rollout_worker = self._create_executor(
            train_iters=100, hybrid_batch_num=4
        )

        self.assertEqual(executor.train_iters, 100)
        self.assertEqual(executor.hybrid_batch_num, 4)
        self.assertEqual(executor.n_samples_per_prompt, 2)
        self.assertFalse(executor.data_optimized)
        self.assertEqual(executor.dataset_additional_keys, ["key1", "response_mask"])
        self.assertIs(executor.controller, controller)
        self.assertIs(executor.rollout_worker, rollout_worker)

    def test_init_calls_init_weight_manager(self):
        """Test that __init__ calls rollout_worker.init_weight_manager."""
        executor, controller, rollout_worker = self._create_executor()

        rollout_worker.init_weight_manager.remote.assert_called_once()

    def test_get_batch_dict_non_optimized(self):
        """Test get_batch_dict when data_optimized is False (uses put_prompts_experience)."""
        executor, _, _ = self._create_executor()
        executor.data_optimized = False

        mock_batch = {"prompts": ["p1", "p2"]}
        expected_result = ({"input_ids": [1, 2]}, [0, 1])
        executor.put_prompts_experience = MagicMock(return_value=expected_result)

        result = executor.get_batch_dict(mock_batch)

        self.assertEqual(result, expected_result)
        executor.put_prompts_experience.assert_called_once_with(
            mock_batch, executor.n_samples_per_prompt, executor.dataset_additional_keys
        )

    def test_get_batch_dict_optimized(self):
        """Test get_batch_dict when data_optimized is True (uses optimized path)."""
        executor, _, _ = self._create_executor()
        executor.data_optimized = True

        mock_batch = {"prompts": ["p1", "p2"]}
        expected_result = ({"input_ids": [1, 2]}, [0, 1])

        with patch('aura.trainer.rollout.rollout_dataset.optimized_preprocess_input',
                   return_value=(["mb1"], ["pid1"])) as mock_preprocess, \
             patch('aura.trainer.rollout.rollout_dataset.optimized_put_prompt_experience',
                   return_value=expected_result) as mock_put:
            result = executor.get_batch_dict(mock_batch)

        self.assertEqual(result, expected_result)
        mock_preprocess.assert_called_once_with(mock_batch)
        mock_put.assert_called_once_with(["mb1"], ["pid1"], executor.padding_dict_to_tensor_dict)

    def test_merge_batch_list_empty(self):
        """Test merge_batch_list with empty list returns empty dict."""
        result = self.FullyAsyncRolloutExecutor.merge_batch_list([])

        self.assertEqual(result, {})

    def test_merge_batch_list_single_batch(self):
        """Test merge_batch_list with a single batch."""
        batches = [{"prompts": ["p1"], "responses": ["r1"]}]

        result = self.FullyAsyncRolloutExecutor.merge_batch_list(batches)

        self.assertEqual(result, {"prompts": ["p1"], "responses": ["r1"]})

    def test_merge_batch_list_multiple_batches(self):
        """Test merge_batch_list with multiple batches extends values."""
        batches = [
            {"prompts": ["p1"], "responses": ["r1"]},
            {"prompts": ["p2", "p3"], "responses": ["r2"]},
        ]

        result = self.FullyAsyncRolloutExecutor.merge_batch_list(batches)

        self.assertEqual(result, {"prompts": ["p1", "p2", "p3"], "responses": ["r1", "r2"]})

    def test_fit_shutdown_immediately(self):
        """Test fit() exits immediately when queue is already shut down."""
        executor, controller, _ = self._create_executor(train_iters=10)

        # Queue reports shutdown=True on first check
        executor.queue_actor = MagicMock()
        executor.queue_actor.is_shutdown.remote.return_value = True

        executor.fit()

        controller.finish_rollout.assert_called_once()

    def test_fit_empty_queue_timeout(self):
        """Test fit() sleeps and exits when queue stays empty and then shuts down."""
        executor, controller, _ = self._create_executor(train_iters=10)

        executor.queue_actor = MagicMock()
        # First check: not shutdown, queue_size=0, not running
        # Second check: shutdown=True
        executor.queue_actor.is_shutdown.remote.side_effect = [False, True]
        executor.queue_actor.queue_size.remote.return_value = 0
        executor.queue_actor.is_running.remote.return_value = False

        with patch('aura.trainer.rollout.fully_async_rollout_executor.time.sleep') as mock_sleep:
            executor.fit()

        # Should have slept at least once waiting for queue
        mock_sleep.assert_called()
        controller.finish_rollout.assert_called_once()

    def test_fit_normal_processing(self):
        """Test fit() processes batches normally and increments iteration."""
        executor, controller, rollout_worker = self._create_executor(
            train_iters=4, hybrid_batch_num=2
        )

        executor.queue_actor = MagicMock()
        # Loop 1: process 2 batches, iteration=2
        # Loop 2: process 2 batches, iteration=4 -> loop exits (iteration >= train_iters)
        executor.queue_actor.is_shutdown.remote.side_effect = [False, False]
        executor.queue_actor.queue_size.remote.return_value = 2
        executor.queue_actor.is_running.remote.return_value = True
        executor.queue_actor.pop_queue.remote.return_value = {"prompts": ["p1"]}

        # Mock the dispatch_actor (set in __init__ via ray.get_actor)
        executor.dispatch_actor = MagicMock()

        # Mock get_batch_dict to avoid complex data processing
        executor.get_batch_dict = MagicMock(return_value=({"input_ids": [1]}, [0]))

        with patch('aura.trainer.rollout.fully_async_rollout_executor.time.sleep'):
            executor.fit()

        # Should have processed 2 loops * 2 batches = 4 iterations
        self.assertEqual(rollout_worker.generate_sequences_fully_async.remote.call_count, 2)
        controller.finish_rollout.assert_called_once()

    def test_fit_rollout_shutdown_exception_breaks_loop(self):
        """Test fit() breaks the loop and calls finish_rollout on RolloutShutdownException."""
        executor, controller, rollout_worker = self._create_executor(
            train_iters=10, hybrid_batch_num=2
        )

        executor.queue_actor = MagicMock()
        executor.queue_actor.is_shutdown.remote.return_value = False
        executor.queue_actor.queue_size.remote.return_value = 2
        executor.queue_actor.is_running.remote.return_value = True
        executor.queue_actor.pop_queue.remote.return_value = {"prompts": ["p1"]}

        executor.dispatch_actor = MagicMock()
        executor.get_batch_dict = MagicMock(return_value=({"input_ids": [1]}, [0]))

        # Simulate RolloutShutdownException raised by generate_sequences_fully_async
        rollout_worker.generate_sequences_fully_async.remote.side_effect = self.RolloutShutdownException(
            "SampleQueue shutdown"
        )

        with patch('aura.trainer.rollout.fully_async_rollout_executor.time.sleep'):
            # Should not raise, should break the loop gracefully
            executor.fit()

        rollout_worker.generate_sequences_fully_async.remote.assert_called_once()
        controller.finish_rollout.assert_called_once()

    def test_fit_clamps_batch_num_at_train_iters(self):
        """Test fit() clamps actual_batch_num when approaching train_iters limit."""
        executor, controller, rollout_worker = self._create_executor(
            train_iters=3, hybrid_batch_num=2
        )

        executor.queue_actor = MagicMock()
        # First loop: queue_size=2, iteration=0 -> actual_batch_num=min(2,2)=2
        # After first loop: iteration=2, remaining=1
        # Second loop: queue_size=2, iteration=2 -> actual_batch_num=min(2,2)=2,
        #   but iteration+actual_batch_num=4 > 3, so clamped to 4-3=1
        executor.queue_actor.is_shutdown.remote.side_effect = [False, False]
        executor.queue_actor.queue_size.remote.return_value = 2
        executor.queue_actor.is_running.remote.return_value = True
        executor.queue_actor.pop_queue.remote.return_value = {"prompts": ["p1"]}

        executor.dispatch_actor = MagicMock()
        executor.get_batch_dict = MagicMock(return_value=({"input_ids": [1]}, [0]))

        with patch('aura.trainer.rollout.fully_async_rollout_executor.time.sleep'):
            executor.fit()

        # First call with actual_batch_num=2, second call with actual_batch_num=1 (clamped)
        call_args_list = rollout_worker.generate_sequences_fully_async.remote.call_args_list
        self.assertEqual(call_args_list[0].args[0], 2)
        self.assertEqual(call_args_list[1].args[0], 1)
        controller.finish_rollout.assert_called_once()

    def test_fit_init_wait_times_decreases(self):
        """Test fit() uses INIT_WAIT_TIMES to wait for queue to fill up initially."""
        executor, controller, _ = self._create_executor(
            train_iters=10, hybrid_batch_num=4
        )

        executor.queue_actor = MagicMock()
        # First call: queue_size=2 < hybrid_batch_num=4, init_wait_times=100 -> sleep
        # Second call: shutdown=True -> exit
        executor.queue_actor.is_shutdown.remote.side_effect = [False, True]
        executor.queue_actor.queue_size.remote.return_value = 2
        executor.queue_actor.is_running.remote.return_value = True
        executor.queue_actor.pop_queue.remote.return_value = None

        with patch('aura.trainer.rollout.fully_async_rollout_executor.time.sleep') as mock_sleep:
            executor.fit()

        # Should have slept once due to init_wait_times logic
        mock_sleep.assert_called()
        controller.finish_rollout.assert_called_once()


if __name__ == '__main__':
    unittest.main()
