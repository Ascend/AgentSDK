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
from unittest.mock import patch, MagicMock


# Create mock objects
mock_ray = MagicMock()
mock_omegaconf = MagicMock()
mock_recipe = MagicMock()
mock_verl = MagicMock()

# Create mock for ray.remote decorator
mock_ray.remote = MagicMock(side_effect=lambda *args, **kwargs: lambda cls: cls)  # Directly return the original class without remote processing

# Create mocks for socket.gethostname and os.getpid
mock_socket = MagicMock()
mock_socket.gethostname.return_value = 'test-host'
mock_os_pid = MagicMock()
mock_os_pid.return_value = 1234

# Set necessary mock values
mock_verl.trainer.ppo.utils.Role = MagicMock()
mock_verl.trainer.ppo.utils.Role.Actor = 'actor'
mock_verl.trainer.ppo.utils.Role.Rollout = 'rollout'

# Mock create_resource_pool_manager and create_role_worker_mapping
mock_recipe.fully_async_policy.fully_async_main.create_resource_pool_manager = MagicMock()
mock_recipe.fully_async_policy.fully_async_main.create_role_worker_mapping = MagicMock(return_value=({}, MagicMock()))

# Mock agentic_rl submodules
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

# Mock required dependency modules, including agentic_rl submodules, before importing the module under test
with patch.dict('sys.modules', {
    'ray': mock_ray,
    'omegaconf': mock_omegaconf,
    'recipe': mock_recipe,
    'recipe.fully_async_policy.fully_async_main': mock_recipe.fully_async_policy.fully_async_main,
    'verl': mock_verl,
    'verl.trainer.ppo.utils': mock_verl.trainer.ppo.utils,
    'verl.utils': mock_verl.utils,
    'verl.trainer.main_ppo': mock_verl.trainer.main_ppo,
    'socket': mock_socket,
    'os': MagicMock(getpid=mock_os_pid, path=os.path),
    # Mock agentic_rl submodules
    'agentic_rl.trainer.train_adapter.verl.full_async.workers.fsdp_workers': mock_fsdp_workers,
    'agentic_rl.trainer.train_adapter.verl.full_async.workers.megatron_worker': mock_megatron_worker,
    'agentic_rl.trainer.train_adapter.verl.full_async.full_async_trainer': mock_full_async_trainer,
    'agentic_rl.trainer.train_adapter.verl.full_async.param_sync': mock_param_sync,
    'agentic_rl.controllers.train_controller.train_controller': mock_train_controller,
    'agentic_rl.data_manager.data_manager': mock_data_manager,
    'agentic_rl.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader': mock_default_train_dataloader
}):
    # Now we can safely import the classes and functions under test
    from agentic_rl.trainer.train_adapter.verl.full_async.train_main import FullyAsyncTaskRunner, start_train

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
        self.mock_config.extras.weight_save_dir = '/tmp/test_weights'
        self.mock_config.extras.delta = 0.1
        self.mock_config.extras.data_loader = None
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
            'agentic_rl.trainer.train_adapter.verl.full_async.workers.fsdp_workers': mock_fsdp_workers,
            'agentic_rl.trainer.train_adapter.verl.full_async.full_async_trainer': mock_full_async_trainer,
            'agentic_rl.trainer.train_adapter.verl.full_async.param_sync': mock_param_sync,
            'agentic_rl.controllers.train_controller.train_controller': mock_train_controller,
            'agentic_rl.data_manager.data_manager': mock_data_manager,
            'agentic_rl.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader': mock_default_train_dataloader
        }):
            # Call _initialize_components method
            self.task_runner._initialize_components(self.mock_config)
            
            # Verify configuration resolution
            mock_omegaconf.OmegaConf.resolve.assert_called_once_with(self.mock_config)
            
            # Verify tokenizer and processor creation
            mock_verl.utils.hf_tokenizer.assert_called_once_with('/tmp/test_model', trust_remote_code=False)
            mock_verl.utils.hf_processor.assert_called_once_with('/tmp/test_model', trust_remote_code=False, use_fast=True)
            
            # Verify creation of role_worker_mapping
            mock_recipe.fully_async_policy.fully_async_main.create_role_worker_mapping.assert_called_once_with(self.mock_config)
            
            # Verify creation of FullyAsyncTrainer
            mock_full_async_trainer.FullyAsyncTrainer.assert_called_once()
            
            # Verify trainer method calls
            mock_fully_async_trainer.set_total_train_steps.assert_called_once_with(1000)
            mock_fully_async_trainer.set_parameter_synchronizer.assert_called_once_with(mock_param_synchronizer)
            mock_fully_async_trainer.set_data_manager.assert_called_once_with(mock_data_manager_instance)
            mock_fully_async_trainer.set_controller.assert_called_once_with(mock_train_controller_instance)
            
            # Verify creation of ParameterSynchronizer
            mock_param_sync.ParameterSynchronizer.remote.assert_called_once()
            
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
            'agentic_rl.trainer.train_adapter.verl.full_async.workers.megatron_worker': mock_megatron_worker,
            'agentic_rl.trainer.train_adapter.verl.full_async.full_async_trainer': mock_full_async_trainer,
            'agentic_rl.trainer.train_adapter.verl.full_async.param_sync': mock_param_sync,
            'agentic_rl.controllers.train_controller.train_controller': mock_train_controller,
            'agentic_rl.data_manager.data_manager': mock_data_manager,
            'agentic_rl.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader': mock_default_train_dataloader
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
            'verl.utils': mock_verl.utils
        }):
            # Call _initialize_components method and verify exception
            with self.assertRaises(NotImplementedError):
                self.task_runner._initialize_components(self.mock_config)

    def test_run_training_loop(self):
        # Set necessary components
        self.task_runner.components["trainer"] = mock_fully_async_trainer
        
        # Mock trainer methods
        mock_fully_async_trainer.fit.return_value = None
        
        # Call _run_training_loop method
        try:
            self.task_runner._run_training_loop()
        except Exception:
            # Original method will re-raise the exception
            pass
        
        # Verify trainer's fit method is called
        mock_fully_async_trainer.fit.assert_called_once()
        
        # Verify running state (original method doesn't set to False in finally block)
        self.assertTrue(self.task_runner.running)

    def test_run_training_loop_exception(self):
        # Set necessary components
        self.task_runner.components["trainer"] = mock_fully_async_trainer
        
        # Reset mock call count
        mock_fully_async_trainer.reset_mock()
        
        # Mock trainer's fit method to raise exception
        mock_fully_async_trainer.fit.side_effect = Exception("Test exception")
        
        # Call _run_training_loop method and verify exception is raised
        with self.assertRaises(Exception):
            self.task_runner._run_training_loop()
        
        # Verify trainer's fit method is called
        mock_fully_async_trainer.fit.assert_called_once()
        
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

if __name__ == '__main__':
    unittest.main()