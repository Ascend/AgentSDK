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
import unittest
from unittest.mock import MagicMock, patch, call
import sys
from unittest import mock
import torch


class TestOneStepOffRollouter(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Save original modules
        self.original_modules = {}
        for module_name in ['ray']:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]
        
        # Mock ray to avoid import errors
        self.mock_ray = mock.MagicMock()
        
        # Configure mock_ray behavior
        self.mock_ray.get_actor.return_value = mock.MagicMock()
        
        # Replace modules
        sys.modules['ray'] = self.mock_ray
        
        # Save original rollouter module
        self.original_rollouter_module = None
        if 'agentic_rl.trainer.rollout.rollouter' in sys.modules:
            self.original_rollouter_module = sys.modules['agentic_rl.trainer.rollout.rollouter']
        
        # Reload rollouter module to ensure we get the original OneStepOffRollouter class
        import importlib
        import agentic_rl.trainer.rollout.rollouter
        importlib.reload(agentic_rl.trainer.rollout.rollouter)
        from agentic_rl.trainer.rollout.rollouter import OneStepOffRollouter
        
        # Create mock objects
        self.mock_controller = MagicMock()
        self.mock_rollout_worker = MagicMock()
        self.mock_padding_dict_to_tensor_dict = MagicMock()
        self.mock_put_prompts_experience = MagicMock()
        self.mock_queue_actor = MagicMock()
        
        # Configure mock objects' behavior
        self.mock_controller.get_weight_manager.return_value = MagicMock()
        self.mock_rollout_worker.init_weight_manager.remote.return_value = None
        self.mock_put_prompts_experience.return_value = ({"batch_dict": "value"}, [0, 1])
        
        # Save original get_rollout_queue_actor function
        from agentic_rl.controllers.rollout_controller.rollout_queue import get_rollout_queue_actor
        self.original_get_rollout_queue_actor = get_rollout_queue_actor
        
        # Replace get_rollout_queue_actor function
        import agentic_rl.controllers.rollout_controller.rollout_queue
        agentic_rl.controllers.rollout_controller.rollout_queue.get_rollout_queue_actor = lambda: self.mock_queue_actor
        
        # Initialize OneStepOffRollouter instance
        self.rollouter = OneStepOffRollouter(
            controller=self.mock_controller,
            rollout_worker=self.mock_rollout_worker,
            train_iters=10,
            padding_dict_to_tensor_dict=self.mock_padding_dict_to_tensor_dict,
            put_prompts_experience=self.mock_put_prompts_experience,
            data_optimized=False,
            dataset_additional_keys=["key1", "key2"],
            n_samples_per_prompt=8,
            hybrid_batch_num=2
        )
        
        # Save OneStepOffRollouter class reference
        self.OneStepOffRollouter = OneStepOffRollouter
    
    def tearDown(self):
        """Clean up test environment"""
        # Restore original get_rollout_queue_actor function
        import agentic_rl.controllers.rollout_controller.rollout_queue
        agentic_rl.controllers.rollout_controller.rollout_queue.get_rollout_queue_actor = self.original_get_rollout_queue_actor
        
        # Restore original rollouter module
        if self.original_rollouter_module is not None:
            sys.modules['agentic_rl.trainer.rollout.rollouter'] = self.original_rollouter_module
        else:
            # If original module doesn't exist, delete the module we created
            if 'agentic_rl.trainer.rollout.rollouter' in sys.modules:
                del sys.modules['agentic_rl.trainer.rollout.rollouter']
        
        # Restore original modules
        for module_name, module in self.original_modules.items():
            sys.modules[module_name] = module
        # Remove mock modules
        mock_modules = ['ray']
        for module_name in mock_modules:
            if module_name in sys.modules and module_name not in self.original_modules:
                del sys.modules[module_name]

    def test_init(self):
        """Test OneStepOffRollouter initialization"""
        # Verify initialization parameters are set correctly
        self.assertEqual(self.rollouter.controller, self.mock_controller)
        self.assertEqual(self.rollouter.rollout_worker, self.mock_rollout_worker)
        self.assertEqual(self.rollouter.train_iters, 10)
        self.assertEqual(self.rollouter.padding_dict_to_tensor_dict, self.mock_padding_dict_to_tensor_dict)
        self.assertEqual(self.rollouter.put_prompts_experience, self.mock_put_prompts_experience)
        self.assertEqual(self.rollouter.data_optimized, False)
        self.assertEqual(self.rollouter.dataset_additional_keys, ["key1", "key2", "response_mask"])
        self.assertEqual(self.rollouter.n_samples_per_prompt, 8)
        self.assertEqual(self.rollouter.hybrid_batch_num, 2)
        
        # Verify init_weight_manager was called
        self.mock_rollout_worker.init_weight_manager.remote.assert_called_once_with(
            self.mock_controller.get_weight_manager.return_value
        )

    def test_get_batch_dict_not_optimized(self):
        """Test get_batch_dict method (non-optimized mode)"""
        # Prepare test data
        batch = {"key": "value"}
        expected_batch_dict = {"batch": "dict"}
        expected_indexes = [0, 1, 2]
        
        # Configure mock objects' behavior
        self.rollouter.data_optimized = False
        self.mock_put_prompts_experience.return_value = (expected_batch_dict, expected_indexes)
        
        # Call method
        batch_dict, indexes = self.rollouter.get_batch_dict(batch)
        
        # Verify results
        self.assertEqual(batch_dict, expected_batch_dict)
        self.assertEqual(indexes, expected_indexes)
        self.mock_put_prompts_experience.assert_called_once_with(
            batch, 8, ["key1", "key2", "response_mask"]
        )

    def test_get_batch_dict_optimized(self):
        """Test get_batch_dict method (optimized mode)"""
        # Prepare test data
        batch = {"key": "value"}
        expected_mini_batches = {"mini": "batches"}
        expected_prompt_ids = {"prompt": "ids"}
        expected_batch_dict = {"batch": "dict"}
        expected_indexes = [0, 1, 2]
        
        # Configure mock objects' behavior
        self.rollouter.data_optimized = True
        
        # Use patch to mock imported functions
        with patch('agentic_rl.trainer.rollout.rollout_dataset.optimized_preprocess_input') as mock_optimized_preprocess_input:
            with patch('agentic_rl.trainer.rollout.rollout_dataset.optimized_put_prompt_experience') as mock_optimized_put_prompt_experience:
                mock_optimized_preprocess_input.return_value = (expected_mini_batches, expected_prompt_ids)
                mock_optimized_put_prompt_experience.return_value = (expected_batch_dict, expected_indexes)
                
                # Call method
                batch_dict, indexes = self.rollouter.get_batch_dict(batch)
                
                # Verify results
                self.assertEqual(batch_dict, expected_batch_dict)
                self.assertEqual(indexes, expected_indexes)
                mock_optimized_preprocess_input.assert_called_once_with(batch)
                mock_optimized_put_prompt_experience.assert_called_once_with(
                    expected_mini_batches, expected_prompt_ids, self.mock_padding_dict_to_tensor_dict
                )

    def test_merge_batch_list(self):
        """Test merge_batch_list method"""
        # Prepare test data
        batches = [
            {"key1": [1, 2], "key2": [3, 4]},
            {"key1": [5, 6], "key2": [7, 8]}
        ]
        
        # Call method
        merged_batch = self.OneStepOffRollouter.merge_batch_list(batches)
        
        # Verify results
        self.assertEqual(merged_batch, {"key1": [1, 2, 5, 6], "key2": [3, 4, 7, 8]})

    def test_merge_batch_list_empty(self):
        """Test merge_batch_list method with empty list"""
        # Call method
        merged_batch = self.OneStepOffRollouter.merge_batch_list([])
        
        # Verify results
        self.assertEqual(merged_batch, {})

    def test_fit(self):
        """Test fit method"""
        # Configure mock objects' behavior
        self.mock_queue_actor.is_shutdown.remote.side_effect = lambda: False
        self.mock_queue_actor.queue_size.remote.side_effect = lambda: 3
        self.mock_queue_actor.is_running.remote.side_effect = lambda: True
        self.mock_queue_actor.pop_queue.remote.side_effect = [
            {"batch": [1]}, {"batch": [2]}  # Return two batches of data, batch value is a list
        ]

        # Configure rollouter properties
        self.rollouter.queue_actor = self.mock_queue_actor
        self.rollouter.train_iters = 2
        self.rollouter.hybrid_batch_num = 2

        # Use patch to mock time.sleep and ray.get
        with patch('agentic_rl.trainer.rollout.rollouter.time.sleep') as mock_sleep:
            with patch('agentic_rl.trainer.rollout.rollouter.ray.get') as mock_ray_get:
                # Configure mock_ray_get return value
                mock_ray_get.side_effect = lambda x: x
                # Call method
                self.rollouter.fit()

                # Verify results
                self.mock_controller.finish_rollout.assert_called_once()
                # Verify data_manager_put_experience was called
                self.mock_rollout_worker.data_manager_put_experience.remote.assert_called_once_with(
                    batch_dict={"batch_dict": "value"}, index=[0, 1]
                )
                # Verify generate_sequences was called
                self.mock_rollout_worker.generate_sequences.remote.assert_called_once_with(2)

    def test_fit_with_queue_size_less_than_hybrid_batch_num(self):
        """Test fit method with queue size less than hybrid batch number"""
        # Configure mock objects' behavior
        self.mock_queue_actor.is_shutdown.remote.side_effect = lambda: False
        self.mock_queue_actor.queue_size.remote.side_effect = lambda: 1
        self.mock_queue_actor.is_running.remote.side_effect = lambda: True
        self.mock_queue_actor.pop_queue.remote.side_effect = [
            {"batch": [1]}, {"batch": [1]}  # Return two batches of data, batch value is a list
        ]

        # Configure rollouter properties
        self.rollouter.queue_actor = self.mock_queue_actor
        self.rollouter.train_iters = 2
        self.rollouter.hybrid_batch_num = 2

        # Use patch to mock time.sleep and ray.get
        with patch('agentic_rl.trainer.rollout.rollouter.time.sleep') as mock_sleep:
            with patch('agentic_rl.trainer.rollout.rollouter.ray.get') as mock_ray_get:
                # Configure mock_ray_get return value
                mock_ray_get.side_effect = lambda x: x
                # Call method
                self.rollouter.fit()

                # Verify results
                self.mock_controller.finish_rollout.assert_called_once()
                # Verify data_manager_put_experience was called 2 times (because train_iters=2)
                self.assertEqual(self.mock_rollout_worker.data_manager_put_experience.remote.call_count, 2)
                # Verify generate_sequences was called 2 times (because train_iters=2)
                self.assertEqual(self.mock_rollout_worker.generate_sequences.remote.call_count, 2)

    def test_fit_with_iteration_exceeding_train_iters(self):
        """Test fit method with iteration exceeding train iterations"""
        # Configure mock objects' behavior
        self.mock_queue_actor.is_shutdown.remote.side_effect = lambda: False
        self.mock_queue_actor.queue_size.remote.side_effect = lambda: 3
        self.mock_queue_actor.is_running.remote.side_effect = lambda: True
        self.mock_queue_actor.pop_queue.remote.side_effect = [
            {"batch": [1]}  # Return one batch of data, batch value is a list
        ]

        # Configure rollouter properties
        self.rollouter.queue_actor = self.mock_queue_actor
        self.rollouter.train_iters = 1  # Only allow 1 iteration
        self.rollouter.hybrid_batch_num = 2

        # Use patch to mock time.sleep and ray.get
        with patch('agentic_rl.trainer.rollout.rollouter.time.sleep') as mock_sleep:
            with patch('agentic_rl.trainer.rollout.rollouter.ray.get') as mock_ray_get:
                # Configure mock_ray_get return value
                mock_ray_get.side_effect = lambda x: x
                # Call method
                self.rollouter.fit()

                # Verify results
                self.mock_controller.finish_rollout.assert_called_once()
                # Verify data_manager_put_experience was called
                self.mock_rollout_worker.data_manager_put_experience.remote.assert_called_once_with(
                    batch_dict={"batch_dict": "value"}, index=[0, 1]
                )
                # Verify generate_sequences was called with actual batch count of 1
                self.mock_rollout_worker.generate_sequences.remote.assert_called_once_with(1)

    def test_fit_with_queue_shutdown(self):
        """Test fit method with queue shutdown"""
        # Configure mock objects' behavior
        self.mock_queue_actor.is_shutdown.remote.side_effect = lambda: True

        # Configure rollouter properties
        self.rollouter.queue_actor = self.mock_queue_actor
        self.rollouter.train_iters = 1

        # Use patch to mock time.sleep and ray.get
        with patch('agentic_rl.trainer.rollout.rollouter.time.sleep') as mock_sleep:
            with patch('agentic_rl.trainer.rollout.rollouter.ray.get') as mock_ray_get:
                # Configure mock_ray_get return value
                mock_ray_get.side_effect = lambda x: x
                # Call method
                self.rollouter.fit()

                # Verify results
                self.mock_controller.finish_rollout.assert_called_once()
                # Verify data_manager_put_experience was not called
                self.mock_rollout_worker.data_manager_put_experience.remote.assert_not_called()
                # Verify generate_sequences was not called
                self.mock_rollout_worker.generate_sequences.remote.assert_not_called()
                # Verify sleep was not called
                mock_sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
