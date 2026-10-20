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
                        'aura.trainer.rollout.rollout_executor', 'aura.trainer.rollout.fully_async_rollout_executor',
                        'aura.controllers.rollout_controller',
                        'aura.controllers.rollout_controller.rollout_controller',
                        'aura.controllers.rollout_controller.rollout_queue',
                        'aura.controllers.rollout_controller.sample_queue',
                        'aura.base.utils.pad_process',
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

        self.mock_fully_async_executor_cls = MagicMock()
        self.mock_fully_async_executor_instance = MagicMock()
        self.mock_fully_async_executor_cls.return_value = self.mock_fully_async_executor_instance
        mock_fully_async_executor_module = MagicMock()
        mock_fully_async_executor_module.FullyAsyncRolloutExecutor = self.mock_fully_async_executor_cls
        sys.modules['aura.trainer.rollout.fully_async_rollout_executor'] = mock_fully_async_executor_module

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

        # Mock sample_queue module so the lazy import in start_fully_async_rollout resolves.
        self.mock_sample_queue = MagicMock()
        self.mock_sample_queue_obj = MagicMock()
        self.mock_sample_queue.get_sample_queue.return_value = self.mock_sample_queue_obj
        sys.modules['aura.controllers.rollout_controller.sample_queue'] = self.mock_sample_queue

        # Import target module, then patch key symbols on module object to avoid stale imports.
        self.rollout_service = importlib.import_module('aura.trainer.rollout.rollout_service')
        self.rollout_service.ray = self.mock_ray
        self.rollout_service.NodeAffinitySchedulingStrategy = self.mock_node_affinity
        self.rollout_service.RolloutWorker = self.mock_rollout_worker_cls
        self.rollout_service.OneStepOffRolloutExecutor = self.mock_executor_cls
        self.rollout_service.FullyAsyncRolloutExecutor = self.mock_fully_async_executor_cls
        self.rollout_service.RolloutController = self.mock_controller_cls

    def tearDown(self):
        """Clean up test environment"""
        mock_modules = ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl',
                        'ray', 'ray.util', 'ray.util.scheduling_strategies',
                        'aura.base.log.loggers', 'aura.base.utils.pad_process',
                        'aura.trainer.rollout.rollout_worker', 'aura.trainer.rollout.rollout_executor',
                        'aura.trainer.rollout.fully_async_rollout_executor',
                        'aura.controllers.rollout_controller', 'aura.controllers.rollout_controller.rollout_controller',
                        'aura.controllers.rollout_controller.rollout_queue',
                        'aura.controllers.rollout_controller.sample_queue',
                        'aura.trainer.rollout.rollout_service']
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

    def _build_rollout_config(self, **overrides):
        """Build a MagicMock rollout_config with default verl-like field values."""
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
        # Default: no SampleQueue injection (max_required_samples=0).
        mock_rollout_config.max_required_samples = 0
        for k, v in overrides.items():
            setattr(mock_rollout_config, k, v)
        return mock_rollout_config

    # ---------- _create_rollout_worker ----------
    def test_create_rollout_worker_default(self):
        """Test _create_rollout_worker helper with default args."""
        mock_rollout_config = self._build_rollout_config()
        self.rollout_service._create_rollout_worker(
            mock_rollout_config, "test_agent_service", "test_infer_service"
        )
        # Rollup worker construction via options().remote()
        self.mock_rollout_worker_cls.options.assert_called_once()
        self.mock_rollout_worker_cls.remote.assert_called_once()
        kwargs = self.mock_rollout_worker_cls.remote.call_args.kwargs
        self.assertEqual(kwargs["train_backend"], "verl")
        self.assertEqual(kwargs["service_mode"], "infer")
        self.assertEqual(kwargs["agent_service"], "test_agent_service")
        self.assertEqual(kwargs["infer_service"], "test_infer_service")
        self.assertEqual(kwargs["generate_config"], None)
        self.assertEqual(kwargs["remove_padding_and_split_to_list"], None)

    def test_create_rollout_worker_with_split_to_list(self):
        """Test _create_rollout_worker helper with custom split_to_list and generate_config."""
        mock_rollout_config = self._build_rollout_config()
        mock_generate_config = MagicMock()
        mock_split = MagicMock()
        self.rollout_service._create_rollout_worker(
            mock_rollout_config, "agent_svc", "infer_svc",
            generate_config=mock_generate_config, split_to_list=mock_split
        )
        kwargs = self.mock_rollout_worker_cls.remote.call_args.kwargs
        self.assertEqual(kwargs["generate_config"], mock_generate_config)
        self.assertEqual(kwargs["remove_padding_and_split_to_list"], mock_split)

    def test_create_rollout_worker_llm_tokenizer_path_missing(self):
        """Test _create_rollout_worker helper falls back to None when llm_tokenizer_path missing."""
        mock_rollout_config = self._build_rollout_config()
        # Delete the auto-generated MagicMock attribute to simulate missing field
        del mock_rollout_config.llm_tokenizer_path
        self.rollout_service._create_rollout_worker(
            mock_rollout_config, "agent_svc", "infer_svc"
        )
        kwargs = self.mock_rollout_worker_cls.remote.call_args.kwargs
        self.assertIsNone(kwargs["llm_tokenizer_path"])

    # ---------- _create_rollout_controller ----------
    def test_create_rollout_controller(self):
        """Test _create_rollout_controller helper."""
        mock_rollout_config = self._build_rollout_config()
        self.rollout_service._create_rollout_controller(mock_rollout_config, "test_model")
        self.mock_controller_cls.assert_called_once()
        kwargs = self.mock_controller_cls.call_args.kwargs
        self.assertEqual(kwargs["model_name"], "test_model")
        self.assertEqual(kwargs["weight_save_dir"], "/path/to/weights")
        self.assertEqual(kwargs["tokenizer_name_or_path"], "test_tokenizer")
        self.assertEqual(kwargs["train_tensor_parallel_size"], 1)

    # ---------- start_fully_async_rollout ----------
    def test_start_fully_async_rollout(self):
        """Test start_fully_async_rollout function."""
        mock_rollout_config = self._build_rollout_config()
        self.rollout_service.start_fully_async_rollout(
            rollout_config=mock_rollout_config,
            agent_service="test_agent_service",
            infer_service="test_infer_service",
        )
        self.mock_rollout_worker_cls.options.assert_called_once()
        self.mock_controller_cls.assert_called_once()
        self.mock_fully_async_executor_cls.assert_called_once()
        self.mock_fully_async_executor_instance.fit.assert_called_once()
        self.mock_controller_instance.send_ready_to_train.assert_called_once()

    def test_start_fully_async_rollout_with_llm_tokenizer_path(self):
        """Test start_fully_async_rollout function with llm_tokenizer_path set."""
        mock_rollout_config = self._build_rollout_config(llm_tokenizer_path="/path/to/llm_tokenizer")
        self.rollout_service.start_fully_async_rollout(
            rollout_config=mock_rollout_config,
            agent_service="agent_svc",
            infer_service="infer_svc",
        )
        kwargs = self.mock_rollout_worker_cls.remote.call_args.kwargs
        self.assertEqual(kwargs["llm_tokenizer_path"], "/path/to/llm_tokenizer")
        self.mock_fully_async_executor_instance.fit.assert_called_once()

    def test_start_fully_async_rollout_passes_executor_args(self):
        """Test start_fully_async_rollout forwards expected kwargs to FullyAsyncRolloutExecutor."""
        mock_rollout_config = self._build_rollout_config()
        self.rollout_service.start_fully_async_rollout(
            rollout_config=mock_rollout_config,
            agent_service="agent_svc",
            infer_service="infer_svc",
        )
        kwargs = self.mock_fully_async_executor_cls.call_args.kwargs
        self.assertEqual(kwargs["train_iters"], 100)
        self.assertEqual(kwargs["dataset_additional_keys"], ["key1", "key2"])
        self.assertEqual(kwargs["data_optimized"], True)
        self.assertEqual(kwargs["n_samples_per_prompt"], 8)
        self.assertEqual(kwargs["hybrid_batch_num"], 2)

    def test_start_fully_async_rollout_no_injection_when_max_required_samples_zero(self):
        """Test start_fully_async_rollout does not inject SampleQueue when max_required_samples=0."""
        mock_rollout_config = self._build_rollout_config(max_required_samples=0)
        self.rollout_service.start_fully_async_rollout(
            rollout_config=mock_rollout_config,
            agent_service="agent_svc",
            infer_service="infer_svc",
        )
        self.mock_sample_queue.get_sample_queue.assert_not_called()
        self.mock_rollout_worker_instance.set_fully_async_config.remote.assert_not_called()
        self.mock_fully_async_executor_instance.fit.assert_called_once()

    def test_start_fully_async_rollout_injects_sample_queue_from_attr(self):
        """Test start_fully_async_rollout injects SampleQueue from object attribute."""
        mock_rollout_config = self._build_rollout_config(max_required_samples=4)
        self.rollout_service.start_fully_async_rollout(
            rollout_config=mock_rollout_config,
            agent_service="agent_svc",
            infer_service="infer_svc",
        )
        self.mock_sample_queue.get_sample_queue.assert_called_once()
        self.mock_rollout_worker_instance.set_fully_async_config.remote.assert_called_once_with(
            self.mock_sample_queue_obj, 4
        )
        self.mock_fully_async_executor_instance.fit.assert_called_once()

    def test_start_fully_async_rollout_injects_sample_queue_from_dict(self):
        """Test start_fully_async_rollout injects SampleQueue when rollout_config is a dict."""
        # Use a dict subclass that also exposes items as attributes, so isinstance(.., dict)
        # is True (exercising the dict branch) while _create_rollout_worker can still read fields.
        class AttrDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)

        rollout_dict = AttrDict({
            "train_backend": "verl",
            "trajectory_timeout": 300,
            "weight_save_dir": "/path/to/weights",
            "hybrid_batch_num": 2,
            "use_on_policy": False,
            "wait_available_weight_timeout": 60,
            "n_samples_per_prompt": 8,
            "actor_rollout_dispatch_size": 0,
            "validate_n_samples": 1,
            "traj_output_path": "/path/to/output",
            "tokenizer_name_or_path": "test_tokenizer",
            "dataset_additional_keys": ["key1", "key2"],
            "global_batch_size": 32,
            "train_iters": 100,
            "trust_remote_code": True,
            "infer_tensor_parallel_size": 1,
            "train_tensor_parallel_size": 1,
            "infer_expert_parallel_size": 1,
            "enable_version_control": True,
            "data_optimized": True,
            "max_required_samples": 8,
        })
        self.rollout_service.start_fully_async_rollout(
            rollout_config=rollout_dict,
            agent_service="agent_svc",
            infer_service="infer_svc",
        )
        self.mock_sample_queue.get_sample_queue.assert_called_once()
        self.mock_rollout_worker_instance.set_fully_async_config.remote.assert_called_once_with(
            self.mock_sample_queue_obj, 8
        )
        self.mock_fully_async_executor_instance.fit.assert_called_once()

    def test_start_fully_async_rollout_injection_failure_reraises(self):
        """Test start_fully_async_rollout re-raises when SampleQueue injection fails."""
        mock_rollout_config = self._build_rollout_config(max_required_samples=4)
        self.mock_rollout_worker_instance.set_fully_async_config.remote.side_effect = RuntimeError("rpc timeout")

        with self.assertRaises(RuntimeError):
            self.rollout_service.start_fully_async_rollout(
                rollout_config=mock_rollout_config,
                agent_service="agent_svc",
                infer_service="infer_svc",
            )
        # Executor.fit should not be called when injection fails
        self.mock_fully_async_executor_instance.fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
