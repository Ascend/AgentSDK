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
from unittest import mock
import sys


class TestRolloutMain(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Save original modules
        self.original_modules = {}
        for module_name in ['sentence_transformers', 'mindspeed_rl', 'mindspeed_rl.utils', 
                           'mindspeed_rl.utils.utils', 'mindspeed_rl.utils.pad_process', 'mindspeed_rl.trainer', 
                           'mindspeed_rl.trainer.utils', 'mindspeed_rl.trainer.utils.transfer_dock', 'verl', 'ray', 
                           'ray.util', 'ray.util.scheduling_strategies', 'uvicorn']:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]
        
        # Mock sentence_transformers, mindspeed_rl, verl and ray to avoid import errors
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
        
        # Mock uvicorn to avoid import errors
        self.mock_uvicorn = mock.MagicMock()
        
        # Mock ray.get_runtime_context
        self.mock_ray_get_runtime_context = mock.Mock()
        self.mock_ray_get_runtime_context.node_id = "test_node_id"
        self.mock_ray.get_runtime_context.return_value = self.mock_ray_get_runtime_context
        
        # Mock ray.get
        self.mock_ray.get = mock.Mock()
        
        # Mock ray.remote
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
        
        # Replace uvicorn module
        sys.modules['uvicorn'] = self.mock_uvicorn
        
        # Import test object
        global start_rollout
        from agentic_rl.trainer.rollout.rollout_main import start_rollout
    
    def tearDown(self):
        """Clean up test environment"""
        # Restore original modules
        for module_name, module in self.original_modules.items():
            sys.modules[module_name] = module
        # Remove mock modules
        mock_modules = ['sentence_transformers', 'mindspeed_rl', 'mindspeed_rl.utils', 
                       'mindspeed_rl.utils.utils', 'mindspeed_rl.utils.pad_process', 'mindspeed_rl.trainer', 
                       'mindspeed_rl.trainer.utils', 'mindspeed_rl.trainer.utils.transfer_dock', 'verl', 'ray', 
                       'ray.util', 'ray.util.scheduling_strategies', 'uvicorn']
        for module_name in mock_modules:
            if module_name in sys.modules and module_name not in self.original_modules:
                del sys.modules[module_name]
        # Clean up global variables
        if 'start_rollout' in globals():
            del globals()['start_rollout']
    @mock.patch('agentic_rl.trainer.rollout.rollout_main.RolloutWorker')
    @mock.patch('agentic_rl.trainer.rollout.rollout_main.logger')
    def test_start_rollout(self, mock_logger, mock_rollout_worker):
        # Prepare test data
        mock_cluster_mode = "test_cluster_mode"
        
        # Create mock rollout_config
        class MockRolloutConfig:
            train_backend = "test_train_backend"
            trajectory_timeout = 3600
            weight_save_dir = "/test/weight/save/dir"
            hybrid_batch_num = 4
            use_on_policy = False
            n_samples_per_prompt = 8
            max_prompt_length = 512
            actor_rollout_dispatch_size = 16
            simplify_think_content = True
            validate_n_samples = 10
            traj_output_path = "/test/traj/output/path"
            tokenizer_name_or_path = "test_tokenizer"
            dataset_additional_keys = ["key1", "key2"]
            global_batch_size = 32
            trust_remote_code = True
            infer_tensor_parallel_size = 1
            train_tensor_parallel_size = 1
            infer_expert_parallel_size = 1
            enable_version_control = True
            train_iters = 1000
            data_optimized = True
        
        mock_rollout_config = MockRolloutConfig()
        mock_agent_service = "test_agent_service"
        mock_infer_service = "test_infer_service"
        
        # Mock NodeAffinitySchedulingStrategy
        mock_node_affinity_scheduling_strategy = mock.Mock()
        self.mock_ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy = mock_node_affinity_scheduling_strategy
        
        # Mock mindspeed_rl imports
        mock_put_prompts_experience = mock.Mock()
        mock_remove_padding_tensor_dict_to_dict = mock.Mock()
        mock_remove_padding_and_split_to_list = mock.Mock()
        mock_padding_dict_to_tensor_dict = mock.Mock()
        
        # Set the mocks
        self.mock_mindspeed_rl_trainer.utils.transfer_dock.put_prompts_experience = mock_put_prompts_experience
        self.mock_mindspeed_rl_utils.pad_process.remove_padding_tensor_dict_to_dict = mock_remove_padding_tensor_dict_to_dict
        self.mock_mindspeed_rl_utils.pad_process.remove_padding_and_split_to_list = mock_remove_padding_and_split_to_list
        self.mock_mindspeed_rl_utils.pad_process.padding_dict_to_tensor_dict = mock_padding_dict_to_tensor_dict
        
        # Mock OneStepOffRollouter
        mock_one_step_off_rollouter = mock.Mock()
        mock_one_step_off_rollouter_instance = mock.Mock()
        mock_one_step_off_rollouter.return_value = mock_one_step_off_rollouter_instance
        
        # Set the mock
        import agentic_rl.trainer.rollout.rollouter
        agentic_rl.trainer.rollout.rollouter.OneStepOffRollouter = mock_one_step_off_rollouter
        
        # Mock RolloutController
        mock_rollout_controller = mock.Mock()
        mock_rollout_controller_instance = mock.Mock()
        mock_rollout_controller.return_value = mock_rollout_controller_instance
        
        # Set the mock
        import agentic_rl.controllers.rollout_controller.rollout_controller
        agentic_rl.controllers.rollout_controller.rollout_controller.RolloutController = mock_rollout_controller
        
        # Mock RolloutWorker
        mock_rollout_worker_instance = mock.Mock()
        mock_rollout_worker.options.return_value.remote.return_value = mock_rollout_worker_instance
        
        # Execute test
        start_rollout(mock_cluster_mode, mock_rollout_config, mock_agent_service, mock_infer_service)
        
        # Verify ray.get_runtime_context was called
        self.mock_ray.get_runtime_context.assert_called_once()
        
        # Verify NodeAffinitySchedulingStrategy was called
        mock_node_affinity_scheduling_strategy.assert_called_once()
        
        # Verify RolloutWorker was initialized correctly
        mock_rollout_worker.options.assert_called_once()
        mock_rollout_worker.options.return_value.remote.assert_called_once_with(
            train_backend=mock_rollout_config.train_backend,
            trajectory_timeout=mock_rollout_config.trajectory_timeout,
            weight_save_dir=mock_rollout_config.weight_save_dir,
            hybrid_batch_num=mock_rollout_config.hybrid_batch_num,
            use_on_policy=mock_rollout_config.use_on_policy,
            n_parallel_agents=mock_rollout_config.n_samples_per_prompt,
            max_prompt_length=mock_rollout_config.max_prompt_length,
            actor_rollout_dispatch_size=mock_rollout_config.actor_rollout_dispatch_size,
            simplify_think_content=mock_rollout_config.simplify_think_content,
            validate_n_samples=mock_rollout_config.validate_n_samples,
            traj_output_path=mock_rollout_config.traj_output_path,
            tokenizer_name_or_path=mock_rollout_config.tokenizer_name_or_path,
            dataset_additional_keys=mock_rollout_config.dataset_additional_keys,
            global_batch_size=mock_rollout_config.global_batch_size,
            remove_padding_tensor_dict_to_dict=mock_remove_padding_tensor_dict_to_dict,
            remove_padding_and_split_to_list=mock_remove_padding_and_split_to_list,
            service_mode="infer",
            agent_service=mock_agent_service,
            infer_service=mock_infer_service
        )
        
        # Verify wait_init_finished was called
        mock_rollout_worker_instance.wait_init_finished.remote.assert_called_once_with(is_proxy_mode=True)
        self.mock_ray.get.assert_called_once_with(mock_rollout_worker_instance.wait_init_finished.remote.return_value)
        
        # Verify RolloutController was initialized correctly
        mock_rollout_controller.assert_called_once_with(
            weight_save_dir=mock_rollout_config.weight_save_dir,
            tokenizer_name_or_path=mock_rollout_config.tokenizer_name_or_path,
            trust_remote_code=mock_rollout_config.trust_remote_code,
            infer_tensor_parallel_size=mock_rollout_config.infer_tensor_parallel_size,
            train_tensor_parallel_size=mock_rollout_config.train_tensor_parallel_size,
            infer_expert_parallel_size=mock_rollout_config.infer_expert_parallel_size,
            enable_version_control=mock_rollout_config.enable_version_control,
            use_on_policy=mock_rollout_config.use_on_policy,
            model_name=mock_infer_service
        )
        
        # Verify send_ready_to_train was called
        mock_rollout_controller_instance.send_ready_to_train.assert_called_once()
        
        # Verify OneStepOffRollouter was initialized correctly
        mock_one_step_off_rollouter.assert_called_once_with(
            mock_rollout_controller_instance,
            mock_rollout_worker_instance,
            train_iters=mock_rollout_config.train_iters,
            padding_dict_to_tensor_dict=mock_padding_dict_to_tensor_dict,
            put_prompts_experience=mock_put_prompts_experience,
            dataset_additional_keys=mock_rollout_config.dataset_additional_keys,
            data_optimized=mock_rollout_config.data_optimized,
            n_samples_per_prompt=mock_rollout_config.n_samples_per_prompt,
            hybrid_batch_num=mock_rollout_config.hybrid_batch_num,
        )
        
        # Verify fit was called
        mock_one_step_off_rollouter_instance.fit.assert_called_once()
        
        # Verify log was recorded
        mock_logger.info.assert_called_once_with("one step off rollout process successfully!")


if __name__ == '__main__':
    unittest.main()
