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
import tempfile
from unittest.mock import patch, MagicMock, mock_open


# Create mock objects
mock_ray = MagicMock()
mock_recipe = MagicMock()
mock_verl = MagicMock()
mock_omegaconf = MagicMock()
mock_tqdm = MagicMock()
mock_np = MagicMock()
mock_torch = MagicMock()

# Set up recipe module internal structure
mock_recipe.fully_async_policy = MagicMock()
mock_recipe.fully_async_policy.ray_trainer = MagicMock()

# Create a mock FullyAsyncRayPPOTrainer class
class MockFullyAsyncRayPPOTrainer:
    def __init__(self, *args, **kwargs):
        # Initialize some necessary properties
        self.resource_pool_to_cls = {}
        self.all_wg = {}
        self.actor_wg = None
        self.actor_rollout_wg = None
        
# Set up mock FullyAsyncRayPPOTrainer
mock_recipe.fully_async_policy.ray_trainer.FullyAsyncRayPPOTrainer = MockFullyAsyncRayPPOTrainer

# Set necessary mock values
mock_verl.DataProto = MagicMock()
mock_verl.Role = MagicMock()
mock_verl.Role.Actor = 'actor'
mock_verl.Role.Critic = 'critic'
mock_verl.Role.RefPolicy = 'ref_policy'
mock_verl.Role.RewardModel = 'reward_model'
mock_verl.WorkerType = MagicMock()

# Mock necessary functions
mock_verl.utils.checkpoint.checkpoint_manager.find_latest_ckpt_path = MagicMock()
mock_verl.utils.checkpoint.checkpoint_manager.should_save_ckpt_esi = MagicMock()
mock_verl.utils.debug.marked_timer = MagicMock(return_value=MagicMock())
mock_verl.trainer.ppo.core_algos.get_kl_controller = MagicMock()
mock_verl.trainer.ppo.utils.need_critic = MagicMock(return_value=False)
mock_verl.trainer.ppo.utils.need_reference_policy = MagicMock(return_value=False)
mock_verl.trainer.ppo.utils.need_reward_model = MagicMock(return_value=False)

# Set necessary mock values
mock_verl.DataProto = MagicMock()
mock_verl.Role = MagicMock()
mock_verl.Role.Actor = 'actor'
mock_verl.Role.Critic = 'critic'
mock_verl.Role.RefPolicy = 'ref_policy'
mock_verl.Role.RewardModel = 'reward_model'
mock_verl.WorkerType = MagicMock()

# Mock necessary functions
mock_verl.utils.checkpoint.checkpoint_manager.find_latest_ckpt_path = MagicMock()
mock_verl.utils.checkpoint.checkpoint_manager.should_save_ckpt_esi = MagicMock()
mock_verl.utils.debug.marked_timer = MagicMock(return_value=MagicMock())
mock_verl.trainer.ppo.core_algos.get_kl_controller = MagicMock()
mock_verl.trainer.ppo.utils.need_critic = MagicMock(return_value=False)
mock_verl.trainer.ppo.utils.need_reference_policy = MagicMock(return_value=False)
mock_verl.trainer.ppo.utils.need_reward_model = MagicMock(return_value=False)

# Mock required dependency modules before importing the module under test
with patch.dict('sys.modules', {
    'ray': mock_ray,
    'omegaconf': mock_omegaconf,
    'tqdm': mock_tqdm,
    'numpy': mock_np,
    'torch': mock_torch,
    'recipe': mock_recipe,
    'recipe.fully_async_policy': mock_recipe.fully_async_policy,
    'recipe.fully_async_policy.detach_utils': mock_recipe.fully_async_policy.detach_utils,
    'recipe.fully_async_policy.ray_trainer': mock_recipe.fully_async_policy.ray_trainer,
    'verl': mock_verl,
    'verl.single_controller.ray': mock_verl.single_controller.ray,
    'verl.trainer.ppo': mock_verl.trainer.ppo,
    'verl.trainer.ppo.core_algos': mock_verl.trainer.ppo.core_algos,
    'verl.trainer.ppo.ray_trainer': mock_verl.trainer.ppo.ray_trainer,
    'verl.trainer.ppo.reward': mock_verl.trainer.ppo.reward,
    'verl.trainer.ppo.utils': mock_verl.trainer.ppo.utils,
    'verl.utils.checkpoint.checkpoint_manager': mock_verl.utils.checkpoint.checkpoint_manager,
    'verl.utils.debug': mock_verl.utils.debug,
}):
    # Now we can safely import the class under test
    from agentic_rl.trainer.train_adapter.verl.full_async.full_async_trainer import FullyAsyncTrainer


class TestFullyAsyncTrainer(unittest.TestCase):
    def setUp(self):
        # Create mock configuration
        self.mock_config = MagicMock()
        self.mock_config.actor_rollout_ref.hybrid_engine = False
        
        # Set up model object's get method
        self.mock_config.actor_rollout_ref.model = MagicMock()
        self.mock_config.actor_rollout_ref.model.get.return_value = 0
        self.mock_config.algorithm.use_kl_in_reward = False
        self.mock_config.async_training.trigger_parameter_sync_step = 1
        self.mock_config.async_training.require_batches = 1
        self.mock_config.actor_rollout_ref.actor.ppo_mini_batch_size = 32
        self.mock_config.async_training.compute_prox_log_prob = False
        self.mock_config.trainer.device = 'cpu'
        self.mock_config.trainer.project_name = 'test_project'
        self.mock_config.trainer.experiment_name = 'test_experiment'
        self.mock_config.trainer.logger = 'test_logger'
        self.mock_config.trainer.default_local_dir = '/tmp/test_ckpt'
        self.mock_config.trainer.default_hdfs_dir = None
        self.mock_config.trainer.save_freq = 10
        self.mock_config.trainer.esi_redundant_time = 300
        self.mock_config.trainer.resume_mode = 'disable'
        self.mock_config.trainer.nnodes = 1
        self.mock_config.trainer.n_gpus_per_node = 1
        self.mock_config.rollout.nnodes = 1
        self.mock_config.rollout.n_gpus_per_node = 1
        self.mock_config.rollout.test_freq = 10
        self.mock_config.reward_model = {}
        
        # Create mock objects
        self.mock_tokenizer = MagicMock()
        self.mock_role_worker_mapping = {mock_verl.Role.Actor: MagicMock()}
        self.mock_resource_pool_manager = MagicMock()
        
        # Mock load_reward_manager
        mock_verl.trainer.ppo.reward.load_reward_manager = MagicMock(return_value=MagicMock())
        
        # Create test instance
        self.trainer = FullyAsyncTrainer(
            config=self.mock_config,
            tokenizer=self.mock_tokenizer,
            role_worker_mapping=self.mock_role_worker_mapping,
            resource_pool_manager=self.mock_resource_pool_manager
        )
        
        # Set necessary properties
        self.trainer.data_manager = MagicMock()
        self.trainer.controller = MagicMock()
        self.trainer.param_synchronizer = MagicMock()
        self.trainer.actor_wg = MagicMock()
        self.trainer.actor_rollout_wg = self.trainer.actor_wg
        self.trainer.max_steps_duration = 300
        
        # Mock logger
        self.trainer.logger = MagicMock()
        
        # Mock progress_bar
        self.trainer.progress_bar = MagicMock()
    
    def tearDown(self):
        # Clear all mock side_effect and return_value
        mock_ray.reset_mock()
        mock_recipe.reset_mock()
        mock_verl.reset_mock()
        mock_omegaconf.reset_mock()
        mock_tqdm.reset_mock()
        mock_torch.reset_mock()
    
    def test_init(self):
        # Verify initialized properties
        self.assertEqual(self.trainer.delta, None)
        self.assertEqual(self.trainer.weight_save_dir, None)
        self.assertEqual(self.trainer.update_weights_interval, 1)
        self.assertEqual(self.trainer.tokenizer, self.mock_tokenizer)
        self.assertEqual(self.trainer.config, self.mock_config)
        self.assertEqual(self.trainer.hybrid_engine, False)
        self.assertEqual(self.trainer.role_worker_mapping, self.mock_role_worker_mapping)
        self.assertEqual(self.trainer.resource_pool_manager, self.mock_resource_pool_manager)
        self.assertEqual(self.trainer.use_reference_policy, False)
        self.assertEqual(self.trainer.use_rm, False)
        self.assertEqual(self.trainer.use_critic, False)
        self.assertEqual(self.trainer.ref_in_actor, False)
        self.assertEqual(self.trainer.global_steps, 1)
        self.assertEqual(self.trainer.local_trigger_step, 1)
        self.assertEqual(self.trainer.processed_samples, 0)
        self.assertEqual(self.trainer.stale_samples_processed, 0)
        self.assertEqual(self.trainer.stale_trajectory_processed, 0)
        self.assertEqual(self.trainer.current_param_version, 0)
        self.assertEqual(self.trainer.total_train_steps, None)
        self.assertEqual(self.trainer.trigger_parameter_sync_step, 1)
        self.assertEqual(self.trainer.last_ckpt_version, 0)
        self.assertEqual(self.trainer.require_batches, 1)
        self.assertEqual(self.trainer.required_samples, 32)
        self.assertEqual(self.trainer.compute_prox_log_prob, False)
    
    def test_set_controller(self):
        # Create mock controller
        mock_controller = MagicMock()
        
        # Call the method
        self.trainer.set_controller(mock_controller)
        
        # Verify the setting
        self.assertEqual(self.trainer.controller, mock_controller)
    
    def test_set_data_manager(self):
        # Create mock data manager
        mock_data_manager = MagicMock()
        
        # Call the method
        self.trainer.set_data_manager(mock_data_manager)
        
        # Verify the setting
        self.assertEqual(self.trainer.data_manager, mock_data_manager)
    
    def test_set_parameter_synchronizer(self):
        # Create mock parameter synchronizer
        mock_param_synchronizer = MagicMock()
        
        # Call the method
        self.trainer.set_parameter_synchronizer(mock_param_synchronizer)
        
        # Verify the setting
        self.assertEqual(self.trainer.param_synchronizer, mock_param_synchronizer)
    
    def test_set_total_train_steps(self):
        # Set total training steps
        total_steps = 1000
        
        # Call the method
        self.trainer.set_total_train_steps(total_steps)
        
        # Verify the setting
        self.assertEqual(self.trainer.total_train_steps, total_steps)
        mock_tqdm.tqdm.assert_called_once_with(total=total_steps, initial=0, desc="Training Progress")
    
    def test_get_actor_wg(self):
        # Create mock actor worker group
        mock_actor_wg = MagicMock()
        self.trainer.actor_wg = mock_actor_wg
        
        # Call the method
        result = self.trainer.get_actor_wg()
        
        # Verify the result
        self.assertEqual(result, mock_actor_wg)
    
    def test_get_samples_from_queue(self):
        # Create mock processed batch data
        mock_processed_batch = {
            'prompts': mock_np.array([[1, 2, 3]]),
            'responses': mock_np.array([[4, 5, 6]]),
            'input_ids': mock_np.array([[7, 8, 9]]),
            'rm_scores': mock_np.array([[10.0]]),
            'token_level_rewards': mock_np.array([[0.1, 0.2, 0.3]]),
            'position_ids': mock_np.array([[0, 1, 2]]),
            'attention_mask': mock_np.array([[1, 1, 1]]),
            'response_mask': mock_np.array([[0, 1, 1]]),
            'rollout_log_probs': mock_np.array([[0.01, 0.02, 0.03]]),
            'prompt_ids': mock_np.array([[0]])
        }
        
        # Set up mock
        self.trainer.data_manager.get_data.return_value = (mock_processed_batch, None)
        
        # Call the method
        epoch, batch = self.trainer._get_samples_from_queue()
        
        # Verify the call
        self.trainer.data_manager.get_data.assert_called_once_with(
            experience_consumer_stage="train",
            experience_columns=None,
            experience_count=32
        )
        
        # Verify the result
        self.assertEqual(epoch, 0)
        self.assertEqual(batch, mock_processed_batch)
    
    def test_prepare_single_generation_data(self):
        # Reset sys.modules to handle internal imports
        with patch.dict('sys.modules', {'verl': mock_verl}):
            # Create mock batch data
            mock_batch_dict = {
                '_prompt_id': mock_np.array([0]),
                'key1': mock_np.array([1]),
                'key2': mock_np.array([2])
            }
            
            # Create mock DataProto
            mock_data_proto = MagicMock()
            mock_data_proto.non_tensor_batch = {}
            mock_verl.DataProto.from_single_dict.return_value = mock_data_proto
            
            # Call the method
            result = self.trainer._prepare_single_generation_data(mock_batch_dict, self.mock_config)
            
            # Verify the call
            mock_verl.DataProto.from_single_dict.assert_called_once_with({
                'key1': mock_np.array([1]),
                'key2': mock_np.array([2])
            })
            mock_data_proto.pop.assert_called_once()
            
            # Verify the result
            self.assertEqual(result, mock_data_proto)
            self.assertEqual(result.non_tensor_batch['uid'], mock_np.array([0]))
    
    def test_check_save_checkpoint_no_save(self):
        # Set condition: current version is the same as last saved version
        self.trainer.current_param_version = 1
        self.trainer.last_ckpt_version = 1
        
        # Call the method
        self.trainer._check_save_checkpoint({})
        
        # Verify no checkpoint is saved
        self.trainer.actor_rollout_wg.save_checkpoint.assert_not_called()
    
    def test_check_save_checkpoint_save(self):
        # Set conditions: current version is different from last saved version and save frequency is met
        self.trainer.current_param_version = 10
        self.trainer.last_ckpt_version = 0
        self.mock_config.trainer.save_freq = 10
        mock_verl.utils.checkpoint.checkpoint_manager.should_save_ckpt_esi.return_value = False
        
        # Mock file operations
        with patch('os.open', return_value=99) as mock_os_open:
            with patch('os.fdopen', mock_open()) as mock_os_fdopen:
                with patch('os.path.realpath', side_effect=lambda x: x):
                    # Call the method
                    self.trainer._check_save_checkpoint({})
                    
                    # Verify checkpoint is saved
                    self.trainer.actor_rollout_wg.save_checkpoint.assert_called_once()
                    self.assertEqual(self.trainer.last_ckpt_version, 10)
    
    def test_trigger_parameter_sync_after_step(self):
        # Set initial values
        self.trainer.current_param_version = 0
        self.trainer.local_trigger_step = 1
        self.trainer.metrics_aggregator.get_aggregated_metrics.return_value = {'metric1': 0.5}
        
        # Call the method
        self.trainer._trigger_parameter_sync_after_step()
        
        # Verify calls
        self.assertEqual(self.trainer.current_param_version, 1)
        self.assertEqual(self.trainer.local_trigger_step, 1)
        self.trainer.metrics_aggregator.get_aggregated_metrics.assert_called_once()
        self.trainer.logger.log.assert_called_once()
        self.trainer.progress_bar.update.assert_called_once_with(1)
        self.trainer.metrics_aggregator.reset.assert_called_once()
    
    def test_load_checkpoint_disable(self):
        # Set resume_mode to disable
        self.mock_config.trainer.resume_mode = 'disable'
        
        # Call the method
        result = self.trainer.load_checkpoint()
        
        # Verify the call
        self.trainer.actor_rollout_wg.load_checkpoint.assert_called_once_with(None)
        self.assertEqual(result, 0)
    
    def test_load_checkpoint_from_scratch(self):
        # Set resume_mode to auto and no checkpoint exists
        self.mock_config.trainer.resume_mode = 'auto'
        mock_verl.utils.checkpoint.checkpoint_manager.find_latest_ckpt_path.return_value = None
        
        # Call the method
        result = self.trainer.load_checkpoint()
        
        # Verify the call
        self.trainer.actor_rollout_wg.load_checkpoint.assert_called_once_with(None)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()