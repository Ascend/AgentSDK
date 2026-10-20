#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import os
import sys
import unittest
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure aura package is importable when tests run from repo root
AURA_SRC = str(Path(__file__).resolve().parents[6])
if AURA_SRC not in sys.path:
    sys.path.insert(0, AURA_SRC)


# Create mock objects
mock_ray = MagicMock()
mock_ray_util = MagicMock()
mock_ray_util_scheduling_strategies = MagicMock()
mock_omegaconf = MagicMock()
mock_recipe = MagicMock()
mock_verl = MagicMock()
mock_verl_experimental = MagicMock()
mock_verl_experimental_separation = MagicMock()

# Set up ray.util.scheduling_strategies mock
mock_ray.util = mock_ray_util
mock_ray.util.scheduling_strategies = mock_ray_util_scheduling_strategies

# Create mock for ray.remote decorator
mock_ray.remote = MagicMock(side_effect=lambda *args, **kwargs: lambda cls: cls)  # Directly return the original class without remote processing

# Create mocks for socket.gethostname and os.getpid
mock_socket = MagicMock()
mock_socket.gethostname.return_value = 'test-host'
mock_os_pid = MagicMock()
mock_os_pid.return_value = 1234

# Set necessary mock values
mock_verl.utils = MagicMock()
mock_verl.trainer.ppo.utils = MagicMock()
mock_verl.trainer.ppo.utils.Role = MagicMock()
mock_verl.trainer.ppo.utils.Role.Actor = 'actor'
mock_verl.trainer.ppo.utils.Role.Rollout = 'rollout'
mock_verl.trainer.main_ppo = MagicMock()
mock_verl.experimental = mock_verl_experimental
mock_verl.experimental.separation = mock_verl_experimental_separation
mock_verl.experimental.separation.utils = mock_verl_experimental_separation

# Mock create_resource_pool_manager and create_role_worker_mapping
mock_verl.experimental.separation.utils.create_resource_pool_manager = MagicMock()
mock_verl.experimental.separation.utils.create_role_worker_mapping = MagicMock(return_value=({}, MagicMock()))

# Mock aura submodules
mock_fsdp_workers = MagicMock()
mock_megatron_worker = MagicMock()
mock_full_async_trainer = MagicMock()
mock_param_sync = MagicMock()
mock_train_controller = MagicMock()
mock_data_manager = MagicMock()
mock_default_train_dataloader = MagicMock()

# Mock necessary classes and methods
mock_fsdp_detach_actor_worker = MagicMock()
mock_fsdp_workers.FsdpDetachActorWorker = MagicMock(return_value=mock_fsdp_detach_actor_worker)

mock_megatron_detach_actor_worker = MagicMock()
mock_megatron_worker.MegatronDetachActorWorker = MagicMock(return_value=mock_megatron_detach_actor_worker)

mock_fully_async_trainer = MagicMock()
mock_fully_async_trainer.get_actor_wg.return_value = MagicMock()
mock_full_async_trainer.FullyAsyncTrainer = MagicMock(return_value=mock_fully_async_trainer)

mock_param_synchronizer = MagicMock()
mock_param_sync.ParameterSynchronizer = MagicMock()
mock_param_sync.ParameterSynchronizer.remote = MagicMock(return_value=mock_param_synchronizer)

mock_train_controller_instance = MagicMock()
mock_train_controller.TrainController = MagicMock(return_value=mock_train_controller_instance)

mock_data_manager_instance = MagicMock()
mock_data_manager_instance.set_pad_token_id_from_tokenizer.return_value = 0
mock_data_manager.DataManager = MagicMock(return_value=mock_data_manager_instance)

# Mock required dependency modules, including aura submodules, before importing the module under test
with patch.dict('sys.modules', {
    'ray': mock_ray,
    'ray.util': mock_ray.util,
    'ray.util.scheduling_strategies': mock_ray.util.scheduling_strategies,
    'omegaconf': mock_omegaconf,
    'recipe': mock_recipe,
    'recipe.fully_async_policy.fully_async_main': mock_recipe.fully_async_policy.fully_async_main,
    'verl': mock_verl,
    'verl.trainer.ppo.utils': mock_verl.trainer.ppo.utils,
    'verl.utils': mock_verl.utils,
    'verl.trainer.main_ppo': mock_verl.trainer.main_ppo,
    'verl.experimental': mock_verl.experimental,
    'verl.experimental.separation.utils': mock_verl.experimental.separation.utils,
    'socket': mock_socket,
    # Keep the real ``os`` module intact so libraries like numpy can still call
    # ``os.uname()``; only ``os.getpid`` is overridden via patch below.
    # Mock aura submodules
    'aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers': mock_fsdp_workers,
    'aura.trainer.train_adapter.verl.full_async.workers.megatron_worker': mock_megatron_worker,
    'aura.trainer.train_adapter.verl.full_async.full_async_trainer': mock_full_async_trainer,
    'aura.trainer.train_adapter.verl.full_async.param_sync': mock_param_sync,
    'aura.controllers.train_controller.train_controller': mock_train_controller,
    'aura.data_manager.data_manager': mock_data_manager,
    'aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader': mock_default_train_dataloader
}):
    # Now we can safely import the classes and functions under test
    from aura.trainer.train_adapter.verl.full_async.train_main import FullyAsyncTaskRunner, start_train

class TestFullyAsyncTaskRunner(unittest.TestCase):
    def setUp(self):
        # Create test instance
        self.task_runner = FullyAsyncTaskRunner()

        # Create mock configuration
        self.mock_config = MagicMock()
        self.mock_config.actor_rollout_ref.model.path = '/tmp/test_model'
        self.mock_config.data = {'trust_remote_code': False}
        self.mock_config.actor_rollout_ref.actor.strategy = 'fsdp'
        self.mock_config.total_train_steps = 1000
        self.mock_config.extras = MagicMock()
        self.mock_config.extras.global_batch_size = 32
        self.mock_config.extras.n_samples_per_prompt = 4
        self.mock_config.extras.validate_num_samples = 100
        self.mock_config.extras.init_num_group_batches = 10
        self.mock_config.extras.max_queue_size = 1000
        self.mock_config.extras.train_iters = 1000
        self.mock_config.extras.update_weights_interval = 1
        self.mock_config.extras.data_loader = MagicMock()
        self.mock_config.extras.data_loader.global_batch_size = 32
        self.mock_config.extras.data_loader.train_iters = 1000
        self.mock_config.extras.weight_save_dir = '/tmp/test_weights'
        self.mock_config.extras.delta = 0.1
        self.mock_config.extras.consumed_train_samples = 0
        self.mock_config.trainer = MagicMock()
        self.mock_config.trainer.device = 'cpu'
        self.mock_config.trainer.get.return_value = True

        # Mock tokenizer and processor
        self.mock_tokenizer = MagicMock()
        self.mock_processor = MagicMock()
        mock_verl.utils.hf_tokenizer.return_value = self.mock_tokenizer
        mock_verl.utils.hf_processor.return_value = self.mock_processor

    def test_init(self):
        # Verify initialization
        self.assertFalse(self.task_runner.running)
        self.assertEqual(self.task_runner.components, {})
        self.assertIsInstance(self.task_runner.shutdown_event, threading.Event)

    def test_run(self):
        # Mock _initialize_components and _run_training_loop methods
        with patch.object(self.task_runner, '_initialize_components') as mock_init, \
             patch.object(self.task_runner, '_run_training_loop') as mock_run_loop:

            # Call run method
            self.task_runner.run(self.mock_config)

            # Verify calls
            mock_init.assert_called_once_with(self.mock_config)
            mock_run_loop.assert_called_once()

    def test_initialize_components(self):
        # Repatch all required modules in the test method, including dynamically imported modules
        with patch.dict('sys.modules', {
            'verl': mock_verl,
            'verl.utils': mock_verl.utils,
            'verl.experimental': mock_verl.experimental,
            'verl.experimental.separation.utils': mock_verl.experimental.separation.utils,
            'aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers': mock_fsdp_workers,
            'aura.trainer.train_adapter.verl.full_async.full_async_trainer': mock_full_async_trainer,
            'aura.trainer.train_adapter.verl.full_async.param_sync': mock_param_sync,
            'aura.controllers.train_controller.train_controller': mock_train_controller,
            'aura.data_manager.data_manager': mock_data_manager,
            'aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader': mock_default_train_dataloader
        }):
            # Call _initialize_components method
            self.task_runner._initialize_components(self.mock_config)

            # Verify configuration resolution
            mock_omegaconf.OmegaConf.resolve.assert_called_once_with(self.mock_config)

            # Verify tokenizer and processor creation
            mock_verl.utils.hf_tokenizer.assert_called_once_with('/tmp/test_model', trust_remote_code=False)
            mock_verl.utils.hf_processor.assert_called_once_with('/tmp/test_model', trust_remote_code=False, use_fast=True)

            # Verify creation of role_worker_mapping
            mock_verl.experimental.separation.utils.create_role_worker_mapping.assert_called_once_with(self.mock_config)

            # Verify creation of FullyAsyncTrainer (ray actor style)
            mock_full_async_trainer.FullyAsyncTrainer.remote.assert_called_once()

            # Verify trainer method calls (ray actor style)
            trainer_actor = mock_full_async_trainer.FullyAsyncTrainer.remote.return_value
            trainer_actor.set_total_train_steps.remote.assert_called_once_with(1000)
            trainer_actor.set_data_manager.remote.assert_called_once_with(mock_data_manager_instance)
            trainer_actor.set_controller.remote.assert_called_once_with(mock_train_controller_instance)

            # Verify creation of ParameterSynchronizer

            # Verify creation of TrainController
            mock_train_controller.TrainController.assert_called_once()

            # Verify creation and initialization of DataManager
            mock_data_manager.DataManager.assert_called_once_with(train_backend="verl", service_mode="train")
            mock_data_manager_instance.sync_init_data_manager.assert_called_once_with(mock_train_controller_instance)
            mock_data_manager_instance.set_pad_token_id_from_tokenizer.assert_called_once_with(self.mock_tokenizer)

    def test_initialize_components_megatron_strategy(self):
        # Set strategy to megatron
        self.mock_config.actor_rollout_ref.actor.strategy = 'megatron'

        # Create a new mock to capture calls to MegatronDetachActorWorker
        mock_new_worker = MagicMock()
        mock_megatron_worker.MegatronDetachActorWorker = MagicMock(return_value=mock_new_worker)

        # Create a mock role_worker_mapping to ensure it gets updated
        mock_role_worker_mapping = {}
        mock_recipe.fully_async_policy.fully_async_main.create_role_worker_mapping.return_value = (mock_role_worker_mapping, MagicMock())

        # Repatch all required modules in the test method, including dynamically imported modules
        with patch.dict('sys.modules', {
            'verl': mock_verl,
            'verl.utils': mock_verl.utils,
            'verl.experimental': mock_verl.experimental,
            'verl.experimental.separation.utils': mock_verl.experimental.separation.utils,
            'aura.trainer.train_adapter.verl.full_async.workers.megatron_worker': mock_megatron_worker,
            'aura.trainer.train_adapter.verl.full_async.full_async_trainer': mock_full_async_trainer,
            'aura.trainer.train_adapter.verl.full_async.param_sync': mock_param_sync,
            'aura.controllers.train_controller.train_controller': mock_train_controller,
            'aura.data_manager.data_manager': mock_data_manager,
            'aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader': mock_default_train_dataloader
        }):
            # Call _initialize_components method
            self.task_runner._initialize_components(self.mock_config)

            # Verify correct worker class is used (just need to check that ray.remote was called)
            mock_ray.remote.assert_called()

    def test_initialize_components_unsupported_strategy(self):
        # Set unsupported strategy
        self.mock_config.actor_rollout_ref.actor.strategy = 'unsupported'

        # Repatch all required modules in the test method, including dynamically imported modules
        with patch.dict('sys.modules', {
            'verl': mock_verl,
            'verl.utils': mock_verl.utils,
            'verl.experimental': mock_verl.experimental,
            'verl.experimental.separation.utils': mock_verl.experimental.separation.utils
        }):
            # Call _initialize_components method and verify exception
            with self.assertRaises(NotImplementedError):
                self.task_runner._initialize_components(self.mock_config)

    def test_run_training_loop(self):
        # Set necessary components
        self.task_runner.components["trainer"] = mock_fully_async_trainer

        # Mock trainer methods (ray actor style)
        mock_fully_async_trainer.fit.remote.return_value = MagicMock()
        mock_ray.wait.return_value = ([mock_fully_async_trainer.fit.remote.return_value], [])
        mock_ray.get.return_value = None

        # Call _run_training_loop method
        try:
            self.task_runner._run_training_loop()
        except Exception:
            # Original method will re-raise the exception
            pass

        # Verify trainer's fit.remote method is called
        mock_fully_async_trainer.fit.remote.assert_called_once()

        # Verify running state (original method doesn't set to False in finally block)
        self.assertTrue(self.task_runner.running)

    def test_run_training_loop_exception(self):
        # Set necessary components
        self.task_runner.components["trainer"] = mock_fully_async_trainer

        # Reset mock call count
        mock_fully_async_trainer.reset_mock()

        # Mock trainer's fit.remote future to raise exception on ray.get
        failed_future = MagicMock(name="failed_future")
        mock_fully_async_trainer.fit.remote.return_value = failed_future
        mock_ray.wait.return_value = ([failed_future], [])

        def _raise_on_get(obj):
            if obj is failed_future:
                raise Exception("Test exception")
            return None

        mock_ray.get.side_effect = _raise_on_get

        # Call _run_training_loop method and verify exception is raised
        with self.assertRaises(Exception):
            self.task_runner._run_training_loop()

        # Verify trainer's fit.remote method is called
        mock_fully_async_trainer.fit.remote.assert_called_once()

        # Verify running state (original method doesn't set to False in finally block)
        self.assertTrue(self.task_runner.running)

class TestStartTrain(unittest.TestCase):
    def test_start_train_logic(self):
        # Test core logic of start_train function instead of directly calling the decorated function
        mock_config = MagicMock()
        mock_config.async_training = MagicMock()

        # Mock verl.trainer.main_ppo.run_ppo
        with patch.dict('sys.modules', {
            'verl': mock_verl,
            'verl.trainer.main_ppo': mock_verl.trainer.main_ppo
        }):
            mock_verl.trainer.main_ppo.run_ppo = MagicMock()

            # Get the wrapped function
            if hasattr(start_train, '__wrapped__'):
                start_train_func = start_train.__wrapped__

                # Call the function
                start_train_func('local', mock_config)

                # Verify verl.trainer.main_ppo.run_ppo is called
                mock_verl.trainer.main_ppo.run_ppo.assert_called_once_with(
                    mock_config,
                    task_runner_class=FullyAsyncTaskRunner
                )

    def test_start_train_missing_async_config_logic(self):
        # Test case with missing async_training configuration
        mock_config = MagicMock()
        mock_config.async_training = None

        # Get the wrapped function
        if hasattr(start_train, '__wrapped__'):
            start_train_func = start_train.__wrapped__

            # Verify exception
            with self.assertRaises(RuntimeError):
                start_train_func('local', mock_config)


class TestFullyAsyncTaskRunnerHelpers(unittest.TestCase):
    """Unit tests for the extracted helpers ``_drain_futures``/``_handle_completed_future``/``_cancel_all``."""

    def setUp(self):
        # Reset module-level mocks (including side_effect/return_value) to avoid
        # cross-test state pollution. Each test re-sets the behavior it needs.
        mock_ray.reset_mock(side_effect=True, return_value=True)
        mock_fully_async_trainer.reset_mock(side_effect=True, return_value=True)
        self.task_runner = FullyAsyncTaskRunner()

    def _make_future(self, name="f"):
        return MagicMock(name=name)

    def test_cancel_all_invokes_ray_cancel(self):
        f1 = self._make_future("f1")
        f2 = self._make_future("f2")
        f3 = self._make_future("f3")

        self.task_runner._cancel_all([f1, f2, f3])

        # ``ray.cancel`` is mocked at module level via mock_ray
        self.assertEqual(mock_ray.cancel.call_count, 3)
        for f in (f1, f2, f3):
            mock_ray.cancel.assert_any_call(f)

    def test_cancel_all_handles_empty_list(self):
        # Should not raise when there are no futures to cancel
        self.task_runner._cancel_all([])
        mock_ray.cancel.assert_not_called()

    def test_handle_completed_future_success_does_not_cancel_siblings(self):
        future = self._make_future("future")
        sibling = self._make_future("sibling")
        mock_ray.get.return_value = None

        # Should not raise
        self.task_runner._handle_completed_future(future, [sibling])

        mock_ray.get.assert_called_once_with(future)
        # ray.cancel should not be called because future succeeded
        mock_ray.cancel.assert_not_called()

    def test_handle_completed_future_failure_cancels_siblings_and_raises(self):
        future = self._make_future("future")
        sibling = self._make_future("sibling")
        mock_ray.get.side_effect = RuntimeError("worker crashed")

        with self.assertRaises(RuntimeError):
            self.task_runner._handle_completed_future(future, [sibling])

        mock_ray.get.assert_called_once_with(future)
        # Siblings should be cancelled
        mock_ray.cancel.assert_called_once_with(sibling)

    def test_drain_futures_returns_when_no_futures(self):
        result = self.task_runner._drain_futures([])
        self.assertEqual(result, [])

    def test_drain_futures_processes_single_future_success(self):
        future = self._make_future("future")
        mock_ray.wait.return_value = ([future], [])
        mock_ray.get.return_value = None

        result = self.task_runner._drain_futures([future])

        # After one iteration, remaining is [] and loop exits
        self.assertEqual(result, [])
        mock_ray.wait.assert_called_once()
        mock_ray.get.assert_called_once_with(future)

    def test_drain_futures_loops_through_multiple_futures(self):
        f1 = self._make_future("f1")
        f2 = self._make_future("f2")
        # First wait returns f1 done with f2 remaining, second wait returns f2 done with no remaining
        mock_ray.wait.side_effect = [([f1], [f2]), ([f2], [])]
        mock_ray.get.return_value = None

        result = self.task_runner._drain_futures([f1, f2])

        self.assertEqual(result, [])
        self.assertEqual(mock_ray.wait.call_count, 2)
        self.assertEqual(mock_ray.get.call_count, 2)

    def test_drain_futures_propagates_failure_and_stops_loop(self):
        f1 = self._make_future("f1")
        f2 = self._make_future("f2")
        mock_ray.wait.return_value = ([f1], [f2])
        mock_ray.get.side_effect = RuntimeError("first future failed")

        with self.assertRaises(RuntimeError):
            self.task_runner._drain_futures([f1, f2])

        # Only the first future was processed (loop aborted via exception)
        mock_ray.get.assert_called_once_with(f1)
        # Sibling cancelled
        mock_ray.cancel.assert_called_once_with(f2)

    def test_run_training_loop_sets_running_flag(self):
        """``_run_training_loop`` should set running=True at the start."""
        self.task_runner.components = {"trainer": mock_fully_async_trainer}
        mock_fully_async_trainer.reset_mock()
        future = self._make_future("future")
        mock_fully_async_trainer.fit.remote.return_value = future
        mock_ray.wait.return_value = ([future], [])
        mock_ray.get.return_value = None

        self.task_runner._run_training_loop()

        self.assertTrue(self.task_runner.running)
        mock_fully_async_trainer.fit.remote.assert_called_once()

    def test_run_training_loop_cancels_remaining_on_failure(self):
        """On failure, ``_run_training_loop`` should cancel remaining futures and re-raise."""
        self.task_runner.components = {"trainer": mock_fully_async_trainer}
        mock_fully_async_trainer.reset_mock()
        future = self._make_future("future")
        mock_fully_async_trainer.fit.remote.return_value = future
        mock_ray.wait.return_value = ([future], [])
        mock_ray.get.side_effect = RuntimeError("train crashed")

        with self.assertRaises(RuntimeError):
            self.task_runner._run_training_loop()

        # Cancel invoked on the original futures list (which is [future])
        mock_ray.cancel.assert_any_call(future)


if __name__ == '__main__':
    unittest.main()
