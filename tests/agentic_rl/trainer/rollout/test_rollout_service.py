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
import sys
from unittest.mock import MagicMock, patch, call

class TestRolloutService(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Save original modules
        self.original_modules = {}
        for module_name in ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl']:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]

        # Mock mindspeed_rl and verl to avoid import errors
        self.mock_mindspeed_rl = MagicMock()
        self.mock_mindspeed_rl_utils = MagicMock()
        self.mock_mindspeed_rl_utils.utils = MagicMock()
        self.mock_verl = MagicMock()

        # Replace modules
        sys.modules['mindspeed_rl'] = self.mock_mindspeed_rl
        sys.modules['mindspeed_rl.utils'] = self.mock_mindspeed_rl_utils
        sys.modules['mindspeed_rl.utils.utils'] = self.mock_mindspeed_rl_utils.utils
        sys.modules['verl'] = self.mock_verl

    def tearDown(self):
        """Clean up test environment"""
        # Restore original modules
        for module_name, module in self.original_modules.items():
            sys.modules[module_name] = module
        # Remove mock modules
        mock_modules = ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl']
        for module_name in mock_modules:
            if module_name in sys.modules and module_name not in self.original_modules:
                del sys.modules[module_name]
    def test_start_async_rollout_worker(self):
        """Test start_async_rollout_worker function"""
        # Prepare test data
        mock_config = {"model": {"test_model": {}}}
        mock_rl_config = MagicMock()
        mock_rl_config.n_samples_per_prompt = 8
        mock_rl_config.max_prompt_length = 8192
        mock_rl_config.actor_rollout_dispatch_size = 0
        mock_rl_config.simplify_think_content = False
        mock_rl_config.validate_n_samples = 1
        mock_rl_config.dict.return_value = {"key": "value"}
        
        mock_agentic_env_config = MagicMock()
        mock_agentic_env_config.rollout_output_path = "/path/to/output"
        
        mock_actor_config = MagicMock()
        mock_actor_config.tokenizer_name_or_path = "test_tokenizer"
        mock_actor_config.dataset_additional_keys = ["key1", "key2"]
        mock_actor_config.global_batch_size = 32
        mock_actor_config.train_iters = 100
        
        mock_generate_config = MagicMock()
        mock_generate_config.dict.return_value = {"generate_key": "generate_value"}
        
        mock_agent_service = "test_agent_service"
        mock_infer_service = "test_infer_service"
        mock_remove_padding_tensor_dict_to_dict = MagicMock()
        mock_remove_padding_and_split_to_list = MagicMock()
        mock_padding_dict_to_tensor_dict = MagicMock()
        mock_put_prompts_experience = MagicMock()
        
        # Import module
        import agentic_rl.trainer.rollout.rollout_service
        
        # Use patch to mock dependencies
        with patch('agentic_rl.trainer.rollout.rollout_service.ray') as mock_ray:
            with patch('agentic_rl.trainer.rollout.rollout_service.RolloutWorker') as mock_rollout_worker:
                with patch('ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy') as mock_node_affinity:
                    with patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutController') as mock_rollout_controller:
                        with patch('agentic_rl.trainer.rollout.rollouter.OneStepOffRollouter') as mock_one_step_off_rollouter:
                            # Configure mock objects' behavior
                            mock_ray.get_runtime_context.return_value.node_id = "test_node_id"
                            mock_rollout_worker_instance = MagicMock()
                            mock_rollout_worker.options.return_value.remote.return_value = mock_rollout_worker_instance
                            mock_ray.get.return_value = None
                            
                            mock_controller_instance = MagicMock()
                            mock_rollout_controller.return_value = mock_controller_instance
                            
                            mock_executor_instance = MagicMock()
                            mock_one_step_off_rollouter.return_value = mock_executor_instance
                            
                            # Get original function (without ray.remote decorator)
                            import types
                            original_function = agentic_rl.trainer.rollout.rollout_service.start_async_rollout_worker._function
                            
                            # Call original function directly
                            original_function(
                                config=mock_config,
                                rl_config=mock_rl_config,
                                agentic_env_config=mock_agentic_env_config,
                                actor_config=mock_actor_config,
                                generate_config=mock_generate_config,
                                agent_service=mock_agent_service,
                                infer_service=mock_infer_service,
                                remove_padding_tensor_dict_to_dict=mock_remove_padding_tensor_dict_to_dict,
                                remove_padding_and_split_to_list=mock_remove_padding_and_split_to_list,
                                padding_dict_to_tensor_dict=mock_padding_dict_to_tensor_dict,
                                put_prompts_experience=mock_put_prompts_experience
                            )
                            
                            # Verify functions are called correctly
                            mock_rollout_worker.options.assert_called_once()
                            mock_rollout_worker_instance.wait_init_finished.remote.assert_called_once_with(is_proxy_mode=True)
                            mock_ray.get.assert_called_once()
                            mock_rollout_controller.assert_called_once_with(mock_actor_config, mock_generate_config, "test_model")
                            mock_controller_instance.send_ready_to_train.assert_called_once()
                            mock_one_step_off_rollouter.assert_called_once()
                            mock_executor_instance.fit.assert_called_once()

    def test_start_async_rollout_worker_with_empty_model_config(self):
        """Test start_async_rollout_worker function with empty model configuration"""
        # Prepare test data
        mock_config = {"model": {"test_model": {}}}
        mock_rl_config = MagicMock()
        mock_rl_config.n_samples_per_prompt = 8
        mock_rl_config.max_prompt_length = 8192
        mock_rl_config.actor_rollout_dispatch_size = 0
        mock_rl_config.simplify_think_content = False
        mock_rl_config.validate_n_samples = 1
        mock_rl_config.dict.return_value = {}
        
        mock_agentic_env_config = MagicMock()
        mock_agentic_env_config.rollout_output_path = "/path/to/output"
        
        mock_actor_config = MagicMock()
        mock_actor_config.tokenizer_name_or_path = "test_tokenizer"
        mock_actor_config.dataset_additional_keys = ["key1", "key2"]
        mock_actor_config.global_batch_size = 32
        mock_actor_config.train_iters = 100
        
        mock_generate_config = MagicMock()
        mock_generate_config.dict.return_value = {}
        
        mock_agent_service = "test_agent_service"
        mock_infer_service = "test_infer_service"
        mock_remove_padding_tensor_dict_to_dict = MagicMock()
        mock_remove_padding_and_split_to_list = MagicMock()
        mock_padding_dict_to_tensor_dict = MagicMock()
        mock_put_prompts_experience = MagicMock()
        
        # Import module
        import agentic_rl.trainer.rollout.rollout_service
        
        # Use patch to mock dependencies
        with patch('agentic_rl.trainer.rollout.rollout_service.ray') as mock_ray:
            with patch('agentic_rl.trainer.rollout.rollout_service.RolloutWorker') as mock_rollout_worker:
                with patch('ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy') as mock_node_affinity:
                    with patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutController') as mock_rollout_controller:
                        with patch('agentic_rl.trainer.rollout.rollouter.OneStepOffRollouter') as mock_one_step_off_rollouter:
                            # Configure mock objects' behavior
                            mock_ray.get_runtime_context.return_value.node_id = "test_node_id"
                            mock_rollout_worker_instance = MagicMock()
                            mock_rollout_worker.options.return_value.remote.return_value = mock_rollout_worker_instance
                            mock_ray.get.return_value = None
                            
                            mock_controller_instance = MagicMock()
                            mock_rollout_controller.return_value = mock_controller_instance
                            
                            mock_executor_instance = MagicMock()
                            mock_one_step_off_rollouter.return_value = mock_executor_instance
                            
                            # Get original function (without ray.remote decorator)
                            import types
                            original_function = agentic_rl.trainer.rollout.rollout_service.start_async_rollout_worker._function
                            
                            # Call original function directly
                            original_function(
                                config=mock_config,
                                rl_config=mock_rl_config,
                                agentic_env_config=mock_agentic_env_config,
                                actor_config=mock_actor_config,
                                generate_config=mock_generate_config,
                                agent_service=mock_agent_service,
                                infer_service=mock_infer_service,
                                remove_padding_tensor_dict_to_dict=mock_remove_padding_tensor_dict_to_dict,
                                remove_padding_and_split_to_list=mock_remove_padding_and_split_to_list,
                                padding_dict_to_tensor_dict=mock_padding_dict_to_tensor_dict,
                                put_prompts_experience=mock_put_prompts_experience
                            )
                            
                            # Verify functions are called correctly
                            mock_rollout_worker.options.assert_called_once()
                            mock_rollout_worker_instance.wait_init_finished.remote.assert_called_once_with(is_proxy_mode=True)
                            mock_ray.get.assert_called_once()
                            mock_rollout_controller.assert_called_once_with(mock_actor_config, mock_generate_config, "test_model")
                            mock_controller_instance.send_ready_to_train.assert_called_once()
                            mock_one_step_off_rollouter.assert_called_once()
                            mock_executor_instance.fit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
