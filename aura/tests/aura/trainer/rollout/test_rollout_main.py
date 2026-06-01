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
import importlib
from pathlib import Path


class TestRolloutMain(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Ensure aura package is importable when running from repo root
        aura_src = str(Path(__file__).resolve().parents[4])
        if aura_src not in sys.path:
            sys.path.insert(0, aura_src)

        # Save original modules
        self.original_modules = {}
        for module_name in ['sentence_transformers', 'mindspeed_rl', 'mindspeed_rl.utils',
                           'mindspeed_rl.utils.utils', 'mindspeed_rl.utils.pad_process', 'mindspeed_rl.trainer',
                           'mindspeed_rl.trainer.utils', 'mindspeed_rl.trainer.utils.transfer_dock', 'verl', 'ray',
                           'ray.util', 'ray.util.scheduling_strategies', 'uvicorn',
                           'aura.controllers.rollout_controller.rollout_controller',
                           'aura.trainer.rollout.rollout_executor', 'aura.trainer.rollout.rollout_worker',
                           'aura.base.log.loggers', 'aura.base.utils.pad_process']:
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

        # Mock rollout dependencies to avoid importing heavy real stack
        mock_loggers = mock.MagicMock()
        mock_loggers.Loggers.return_value.get_logger.return_value = mock.MagicMock()
        sys.modules['aura.base.log.loggers'] = mock_loggers

        mock_pad_process_module = mock.MagicMock()
        mock_pad_process_module.remove_padding_tensor_dict_to_dict = mock.MagicMock()
        mock_pad_process_module.remove_padding_and_split_to_list = mock.MagicMock()
        mock_pad_process_module.padding_dict_to_tensor_dict = mock.MagicMock()
        mock_pad_process_module.put_prompts_experience = mock.MagicMock()
        sys.modules['aura.base.utils.pad_process'] = mock_pad_process_module

        mock_worker_module = mock.MagicMock()
        mock_worker_module.RolloutWorker = mock.MagicMock()
        sys.modules['aura.trainer.rollout.rollout_worker'] = mock_worker_module

        mock_executor_module = mock.MagicMock()
        mock_executor_module.OneStepOffRolloutExecutor = mock.MagicMock()
        sys.modules['aura.trainer.rollout.rollout_executor'] = mock_executor_module

        mock_controller_module = mock.MagicMock()
        mock_controller_module.RolloutController = mock.MagicMock()
        sys.modules['aura.controllers.rollout_controller.rollout_controller'] = mock_controller_module

        # Import test object
        global start_rollout
        rollout_service_mod = importlib.import_module('aura.trainer.rollout.rollout_service')
        start_rollout = rollout_service_mod.start_rollout

    def tearDown(self):
        """Clean up test environment"""
        # Restore original modules
        for module_name, module in self.original_modules.items():
            sys.modules[module_name] = module
        # Remove mock modules
        mock_modules = ['sentence_transformers', 'mindspeed_rl', 'mindspeed_rl.utils',
                       'mindspeed_rl.utils.utils', 'mindspeed_rl.utils.pad_process', 'mindspeed_rl.trainer',
                       'mindspeed_rl.trainer.utils', 'mindspeed_rl.trainer.utils.transfer_dock', 'verl', 'ray',
                       'ray.util', 'ray.util.scheduling_strategies', 'uvicorn',
                       'aura.controllers.rollout_controller.rollout_controller',
                       'aura.trainer.rollout.rollout_executor', 'aura.trainer.rollout.rollout_worker',
                       'aura.base.log.loggers', 'aura.base.utils.pad_process',
                       'aura.trainer.rollout.rollout_service']
        for module_name in mock_modules:
            if module_name in sys.modules and module_name not in self.original_modules:
                del sys.modules[module_name]
        # Clean up global variables
        if 'start_rollout' in globals():
            del globals()['start_rollout']
    @mock.patch('aura.trainer.rollout.rollout_service.RolloutWorker')
    @mock.patch('aura.trainer.rollout.rollout_service.logger')
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
            wait_available_weight_timeout = 60
            actor_rollout_dispatch_size = 16
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

        # Mock OneStepOffRolloutExecutor
        mock_one_step_off_rollout_executor = mock.Mock()
        mock_one_step_off_rollout_executor_instance = mock.Mock()
        mock_one_step_off_rollout_executor.return_value = mock_one_step_off_rollout_executor_instance

        # Set the mock on the real import target
        rollout_service_mod = importlib.import_module('aura.trainer.rollout.rollout_service')
        rollout_service_mod.OneStepOffRolloutExecutor = mock_one_step_off_rollout_executor

        # Mock RolloutController
        mock_rollout_controller = mock.Mock()
        mock_rollout_controller_instance = mock.Mock()
        mock_rollout_controller.return_value = mock_rollout_controller_instance

        # Set the mock
        rollout_controller_mod = importlib.import_module('aura.controllers.rollout_controller.rollout_controller')
        rollout_controller_mod.RolloutController = mock_rollout_controller

        # Mock RolloutWorker
        mock_rollout_worker_instance = mock.Mock()
        mock_rollout_worker.options.return_value.remote.return_value = mock_rollout_worker_instance

        # Execute test
        start_rollout(mock_rollout_config, mock_agent_service, mock_infer_service)

        # Verify ray.get_runtime_context was called
        self.mock_ray.get_runtime_context.assert_called_once()

        # Verify scheduling strategy was attached via RolloutWorker.options
        mock_rollout_worker.options.assert_called_once()

        # Verify RolloutWorker was initialized correctly
        mock_rollout_worker.options.assert_called_once()
        remote_kwargs = mock_rollout_worker.options.return_value.remote.call_args.kwargs
        self.assertEqual(remote_kwargs["train_backend"], mock_rollout_config.train_backend)
        self.assertEqual(remote_kwargs["trajectory_timeout"], mock_rollout_config.trajectory_timeout)
        self.assertEqual(remote_kwargs["weight_save_dir"], mock_rollout_config.weight_save_dir)
        self.assertEqual(remote_kwargs["hybrid_batch_num"], mock_rollout_config.hybrid_batch_num)
        self.assertEqual(remote_kwargs["use_on_policy"], mock_rollout_config.use_on_policy)
        self.assertEqual(remote_kwargs["wait_available_weight_timeout"], mock_rollout_config.wait_available_weight_timeout)
        self.assertEqual(remote_kwargs["n_parallel_agents"], mock_rollout_config.n_samples_per_prompt)
        self.assertEqual(remote_kwargs["service_mode"], "infer")
        self.assertEqual(remote_kwargs["agent_service"], mock_agent_service)
        self.assertEqual(remote_kwargs["infer_service"], mock_infer_service)

        # Verify wait_init_finished was called
        mock_rollout_worker_instance.wait_init_finished.remote.assert_called_once_with(is_proxy_mode=True)
        self.mock_ray.get.assert_called_once_with(mock_rollout_worker_instance.wait_init_finished.remote.return_value)

        # Verify controller interaction happened
        self.assertTrue(mock_one_step_off_rollout_executor.called)

        # Verify send_ready_to_train was called on executor's controller arg
        executor_call_args = mock_one_step_off_rollout_executor.call_args
        controller_arg = executor_call_args.args[0]
        controller_arg.send_ready_to_train.assert_called_once()

        # Verify OneStepOffRolloutExecutor call args in a robust way
        self.assertTrue(mock_one_step_off_rollout_executor.called)
        executor_args, executor_kwargs = mock_one_step_off_rollout_executor.call_args
        self.assertIs(executor_args[1], mock_rollout_worker_instance)
        self.assertEqual(executor_kwargs["train_iters"], mock_rollout_config.train_iters)
        self.assertEqual(executor_kwargs["dataset_additional_keys"], mock_rollout_config.dataset_additional_keys)
        self.assertEqual(executor_kwargs["data_optimized"], mock_rollout_config.data_optimized)
        self.assertEqual(executor_kwargs["n_samples_per_prompt"], mock_rollout_config.n_samples_per_prompt)
        self.assertEqual(executor_kwargs["hybrid_batch_num"], mock_rollout_config.hybrid_batch_num)

        # Verify fit was called
        mock_one_step_off_rollout_executor_instance.fit.assert_called_once()

        # Verify success log was recorded
        self.assertIn(
            mock.call("one step off rollout process successfully!"),
            mock_logger.info.call_args_list,
        )


if __name__ == '__main__':
    unittest.main()
