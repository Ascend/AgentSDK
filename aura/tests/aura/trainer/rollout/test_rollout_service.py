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
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch, call


class TestRolloutService(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.original_modules = {}
        aura_src = str(Path(__file__).resolve().parents[5] / 'aura')
        if aura_src not in sys.path:
            sys.path.insert(0, aura_src)
        module_names = ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl',
                        'aura.trainer.rollout.rollout_service', 'aura.trainer.rollout.rollout_worker',
                        'aura.trainer.rollout.rollout_executor', 'aura.controllers.rollout_controller',
                        'aura.controllers.rollout_controller.rollout_controller', 'aura.base.utils.pad_process',
                        'aura.base.log.loggers', 'ray', 'ray.util', 'ray.util.scheduling_strategies']
        for module_name in module_names:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]

        self.mock_ray = MagicMock()
        self.mock_ray.get.return_value = None
        self.mock_ray.get_runtime_context.return_value.node_id = "test_node_id"
        # Make @ray.remote just return the original function!
        self.mock_ray.remote.side_effect = lambda func: func
        sys.modules['ray'] = self.mock_ray

        mock_ray_util = MagicMock()
        mock_ray_scheduling = MagicMock()
        self.mock_node_affinity = MagicMock()
        mock_ray_scheduling.NodeAffinitySchedulingStrategy = self.mock_node_affinity

        sys.modules['ray.util'] = mock_ray_util
        sys.modules['ray.util.scheduling_strategies'] = mock_ray_scheduling

        self.mock_mindspeed_rl = MagicMock()
        self.mock_mindspeed_rl_utils = MagicMock()
        self.mock_mindspeed_rl_utils.utils = MagicMock()
        self.mock_verl = MagicMock()

        sys.modules['mindspeed_rl'] = self.mock_mindspeed_rl
        sys.modules['mindspeed_rl.utils'] = self.mock_mindspeed_rl_utils
        sys.modules['mindspeed_rl.utils.utils'] = self.mock_mindspeed_rl_utils.utils
        sys.modules['verl'] = self.mock_verl

        self.mock_logger = MagicMock()
        mock_loggers = MagicMock()
        mock_loggers.Loggers.return_value.get_logger.return_value = self.mock_logger
        sys.modules['aura.base.log.loggers'] = mock_loggers

        self.mock_pad_process = MagicMock()
        sys.modules['aura.base.utils.pad_process'] = self.mock_pad_process

        self.mock_rollout_worker_cls = MagicMock()
        self.mock_rollout_worker_instance = MagicMock()
        self.mock_rollout_worker_instance.wait_init_finished.return_value = None  # 不用 .remote()
        self.mock_rollout_worker_cls.options.return_value = self.mock_rollout_worker_cls
        self.mock_rollout_worker_cls.remote.return_value = self.mock_rollout_worker_instance
        mock_rollout_worker_module = MagicMock()
        mock_rollout_worker_module.RolloutWorker = self.mock_rollout_worker_cls
        sys.modules['aura.trainer.rollout.rollout_worker'] = mock_rollout_worker_module

        self.mock_executor_cls = MagicMock()
        self.mock_executor_instance = MagicMock()
        self.mock_executor_cls.return_value = self.mock_executor_instance
        mock_executor_module = MagicMock()
        mock_executor_module.OneStepOffRolloutExecutor = self.mock_executor_cls
        sys.modules['aura.trainer.rollout.rollout_executor'] = mock_executor_module

        self.mock_controller_cls = MagicMock()
        self.mock_controller_instance = MagicMock()
        self.mock_controller_cls.return_value = self.mock_controller_instance
        self.mock_controller_instance.send_ready_to_train = MagicMock()
        mock_controller_module = MagicMock()
        mock_controller_module.RolloutController = self.mock_controller_cls
        sys.modules['aura.controllers.rollout_controller'] = MagicMock()
        sys.modules['aura.controllers.rollout_controller.rollout_controller'] = mock_controller_module

        mock_rollout_queue_module = MagicMock()
        mock_rollout_queue_module.get_rollout_queue_actor = MagicMock(return_value=MagicMock())
        sys.modules['aura.controllers.rollout_controller.rollout_queue'] = mock_rollout_queue_module

        # Import target module, then patch key symbols on module object to avoid stale imports.
        self.rollout_service = importlib.import_module('aura.trainer.rollout.rollout_service')
        self.rollout_service.ray = self.mock_ray
        self.rollout_service.NodeAffinitySchedulingStrategy = self.mock_node_affinity
        self.rollout_service.RolloutWorker = self.mock_rollout_worker_cls
        self.rollout_service.OneStepOffRolloutExecutor = self.mock_executor_cls
        self.rollout_service.RolloutController = self.mock_controller_cls

    def tearDown(self):
        """Clean up test environment"""
        mock_modules = ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl',
                        'ray', 'ray.util', 'ray.util.scheduling_strategies',
                        'aura.base.log.loggers', 'aura.base.utils.pad_process',
                        'aura.trainer.rollout.rollout_worker', 'aura.trainer.rollout.rollout_executor',
                        'aura.controllers.rollout_controller', 'aura.controllers.rollout_controller.rollout_controller',
                        'aura.controllers.rollout_controller.rollout_queue', 'aura.trainer.rollout.rollout_service']
        for module_name in mock_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        for module_name, module in self.original_modules.items():
            sys.modules[module_name] = module

    def test_start_async_rollout_worker(self):
        """Test start_async_rollout_worker function"""
        mock_config = {"model": {"test_model": {}}}
        mock_rl_config = MagicMock()
        mock_rl_config.n_samples_per_prompt = 8
        mock_rl_config.max_prompt_length = 8192
        mock_rl_config.actor_rollout_dispatch_size = 0
        mock_rl_config.simplify_think_content = False
        mock_rl_config.validate_n_samples = 1
        mock_rl_config.dict.return_value = {"key": "value"}
        mock_rl_config.wait_available_weight_timeout = 60

        mock_agentic_env_config = MagicMock()
        mock_agentic_env_config.rollout_output_path = "/path/to/output"

        mock_actor_config = MagicMock()
        mock_actor_config.tokenizer_name_or_path = "test_tokenizer"
        mock_actor_config.dataset_additional_keys = ["key1", "key2"]
        mock_actor_config.global_batch_size = 32
        mock_actor_config.train_iters = 100
        mock_actor_config.tensor_model_parallel_size = 1

        mock_generate_config = MagicMock()
        mock_generate_config.dict.return_value = {"generate_key": "generate_value"}
        mock_generate_config.train_backend = "verl"
        mock_generate_config.weight_save_dir = "/path/to/weights"
        mock_generate_config.hybrid_batch_num = 2
        mock_generate_config.use_on_policy = False
        mock_generate_config.trust_remote_code = True
        mock_generate_config.infer_tensor_parallel_size = 1
        mock_generate_config.infer_expert_parallel_size = 1
        mock_generate_config.enable_version_control = True

        mock_agent_service = "test_agent_service"
        mock_infer_service = "test_infer_service"

        # Now actually call the function (without mocking the whole function)
        self.rollout_service.start_async_rollout_worker(
            config=mock_config,
            rl_config=mock_rl_config,
            agentic_env_config=mock_agentic_env_config,
            actor_config=mock_actor_config,
            generate_config=mock_generate_config,
            agent_service=mock_agent_service,
            infer_service=mock_infer_service
        )

        # Verify calls
        self.mock_rollout_worker_cls.options.assert_called_once()
        self.mock_controller_cls.assert_called_once()
        self.mock_executor_cls.assert_called_once()
        self.mock_executor_instance.fit.assert_called_once()
        self.mock_controller_instance.send_ready_to_train.assert_called_once()

    def test_start_async_rollout_worker_with_empty_model_config(self):
        """Test start_async_rollout_worker function with empty model configuration"""
        mock_config = {"model": {"test_model": {}}}
        mock_rl_config = MagicMock()
        mock_rl_config.n_samples_per_prompt = 8
        mock_rl_config.max_prompt_length = 8192
        mock_rl_config.actor_rollout_dispatch_size = 0
        mock_rl_config.simplify_think_content = False
        mock_rl_config.validate_n_samples = 1
        mock_rl_config.dict.return_value = {}
        mock_rl_config.wait_available_weight_timeout = 60

        mock_agentic_env_config = MagicMock()
        mock_agentic_env_config.rollout_output_path = "/path/to/output"

        mock_actor_config = MagicMock()
        mock_actor_config.tokenizer_name_or_path = "test_tokenizer"
        mock_actor_config.dataset_additional_keys = ["key1", "key2"]
        mock_actor_config.global_batch_size = 32
        mock_actor_config.train_iters = 100
        mock_actor_config.tensor_model_parallel_size = 1

        mock_generate_config = MagicMock()
        mock_generate_config.dict.return_value = {}
        mock_generate_config.train_backend = "verl"
        mock_generate_config.weight_save_dir = "/path/to/weights"
        mock_generate_config.hybrid_batch_num = 2
        mock_generate_config.use_on_policy = False
        mock_generate_config.trust_remote_code = True
        mock_generate_config.infer_tensor_parallel_size = 1
        mock_generate_config.infer_expert_parallel_size = 1
        mock_generate_config.enable_version_control = True

        mock_agent_service = "test_agent_service"
        mock_infer_service = "test_infer_service"

        # Now actually call the function
        self.rollout_service.start_async_rollout_worker(
            config=mock_config,
            rl_config=mock_rl_config,
            agentic_env_config=mock_agentic_env_config,
            actor_config=mock_actor_config,
            generate_config=mock_generate_config,
            agent_service=mock_agent_service,
            infer_service=mock_infer_service
        )

        # Verify calls
        self.mock_rollout_worker_cls.options.assert_called()
        self.mock_controller_cls.assert_called()
        self.mock_executor_cls.assert_called()
        self.mock_executor_instance.fit.assert_called()

    def test_start_rollout(self):
        """Test start_rollout function"""
        mock_rollout_config = MagicMock()
        mock_rollout_config.train_backend = "verl"
        mock_rollout_config.trajectory_timeout = 300
        mock_rollout_config.weight_save_dir = "/path/to/weights"
        mock_rollout_config.hybrid_batch_num = 2
        mock_rollout_config.use_on_policy = False
        mock_rollout_config.wait_available_weight_timeout = 60
        mock_rollout_config.n_samples_per_prompt = 8
        mock_rollout_config.actor_rollout_dispatch_size = 0
        mock_rollout_config.validate_n_samples = 1
        mock_rollout_config.traj_output_path = "/path/to/output"
        mock_rollout_config.tokenizer_name_or_path = "test_tokenizer"
        mock_rollout_config.dataset_additional_keys = ["key1", "key2"]
        mock_rollout_config.global_batch_size = 32
        mock_rollout_config.train_iters = 100
        mock_rollout_config.trust_remote_code = True
        mock_rollout_config.infer_tensor_parallel_size = 1
        mock_rollout_config.train_tensor_parallel_size = 1
        mock_rollout_config.infer_expert_parallel_size = 1
        mock_rollout_config.enable_version_control = True
        mock_rollout_config.data_optimized = True

        mock_agent_service = "test_agent_service"
        mock_infer_service = "test_infer_service"

        # Call the start_rollout function
        self.rollout_service.start_rollout(
            rollout_config=mock_rollout_config,
            agent_service=mock_agent_service,
            infer_service=mock_infer_service
        )

        # Verify calls
        self.mock_rollout_worker_cls.options.assert_called_once()
        self.mock_controller_cls.assert_called_once()
        self.mock_executor_cls.assert_called_once()
        self.mock_executor_instance.fit.assert_called_once()

    def test_start_rollout_with_llm_tokenizer_path(self):
        """Test start_rollout with llm_tokenizer_path"""
        mock_rollout_config = MagicMock()
        mock_rollout_config.train_backend = "verl"
        mock_rollout_config.trajectory_timeout = 300
        mock_rollout_config.weight_save_dir = "/path/to/weights"
        mock_rollout_config.hybrid_batch_num = 2
        mock_rollout_config.use_on_policy = False
        mock_rollout_config.wait_available_weight_timeout = 60
        mock_rollout_config.n_samples_per_prompt = 8
        mock_rollout_config.actor_rollout_dispatch_size = 0
        mock_rollout_config.validate_n_samples = 1
        mock_rollout_config.traj_output_path = "/path/to/output"
        mock_rollout_config.tokenizer_name_or_path = "test_tokenizer"
        mock_rollout_config.dataset_additional_keys = ["key1", "key2"]
        mock_rollout_config.global_batch_size = 32
        mock_rollout_config.train_iters = 100
        mock_rollout_config.trust_remote_code = True
        mock_rollout_config.infer_tensor_parallel_size = 1
        mock_rollout_config.train_tensor_parallel_size = 1
        mock_rollout_config.infer_expert_parallel_size = 1
        mock_rollout_config.enable_version_control = True
        mock_rollout_config.data_optimized = True
        mock_rollout_config.llm_tokenizer_path = "/path/to/llm_tokenizer"

        mock_agent_service = "test_agent_service"
        mock_infer_service = "test_infer_service"

        # Call the start_rollout function
        self.rollout_service.start_rollout(
            rollout_config=mock_rollout_config,
            agent_service=mock_agent_service,
            infer_service=mock_infer_service
        )

        # Verify calls
        self.mock_rollout_worker_cls.options.assert_called_once()
        self.mock_controller_cls.assert_called_once()
        self.mock_executor_cls.assert_called_once()


if __name__ == '__main__':
    unittest.main()
