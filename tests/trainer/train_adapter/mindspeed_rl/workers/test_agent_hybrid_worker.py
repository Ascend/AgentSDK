#!/usr/bin/env python3
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from unittest.mock import patch, MagicMock, Mock

import pytest
import torch

from agentic_rl.configs.agentic_rl_config import AgenticRLConfig


@dataclass
class MockGenerateConfig:
    offload_train_optimizer: bool = True
    offload_train_grad: bool = True
    offload_train_param: bool = True
    max_num_seqs: int = 10
    max_model_len: int = 10
    dtype = "float16"
    gpu_memory_utilization: float = 0.7
    infer_tensor_parallel_size: int = 10
    infer_pipeline_parallel_size: int = 1
    infer_expert_parallel_size: int = 1
    trust_remote_code: bool = False
    sampling_config: dict = field(default_factory=lambda: {"temperature": 1.0})
    limit_mm_image_per_prompt = None
    limit_mm_video_per_prompt = None
    tokenizer_name_or_path: str = ""
    max_num_batched_tokens: int = 1024
    enable_prefix_caching: bool = False
    num_scheduler_steps: int = 10
    enforce_eager: bool = False
    torchair_graph: bool = False
    enable_expert_parallel: bool = False
    ascend_scheduler_config_enabled: bool = False


class MockActorState(Enum):
    NONE = "none"
    INFER = "infer"


@dataclass
class MockMegatronConfig:
    tokenizer_name_or_path: str = "/abs/path"
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    context_parallel_size: int = 1
    context_parallel_algo: str = "megatron_cp_algo"
    stage: MockActorState = MockActorState.NONE
    micro_batch_size: int = 10
    global_batch_size: int = 20
    dataset_additional_keys = ["question", "answer"]
    train_iters: int = 10
    save_interval: int = 100
    load: str = "/test/path"
    save: str = "/test/path"
    data_path: str = "/test/path"
    split = None
    seq_length: int = 100
    seed = 1234
    num_workers = 1
    no_shuffle = True


@dataclass
class MockRLConfig:
    beta: float = 0.1
    epochs: int = 10
    shuffle_mini_batch: bool = False
    clip_ratio: float = 0.1
    use_dynamic_bsz: bool = False
    max_packing_token_size: int = 10
    dynamic_max_batch_size: int = 10
    use_remove_padding: bool = False
    entropy_coeff: float = 0.1
    kl_penalty: str = 'kl'
    token_level_loss: bool = True
    clip_higher_enable: bool = False
    clip_ratio_low: float = 0.1
    clip_ratio_high: float = 0.1
    mini_batch_size: int = 10
    actor_update_dispatch_size: int = None
    guarantee_order: bool = True
    n_samples_per_prompt: int = 2
    actor_forward_micro_batch_size: int = 1
    ref_forward_micro_batch_size: int = 1
    integrated_mode_config = None
    max_prompt_length: int = 10
    actor_rollout_dispatch_size: int = 10
    adv_dispatch_size: int = 10
    use_integrated_worker: bool = True
    rule_reward: bool = True
    num_cpus_for_local_task: int = 1

    def dict(self):
        return {}


class MockActorRolloutHybrid:
    def __init__(self, *args, **kwargs):
        pass

    def update_actor(self, batch_data, kl_ctrl):
        pass


class MockBaseTokenizer:
    def tokenize(self, *args, **kwargs):
        return torch.Tensor([1])


class MockMegatronOffLoader:

    def __init__(self,
                 megatron_model=None,
                 optimizer=None,
                 wrap_with_ddp=True,
                 megatron_config=None,
                 distributed_optimizer=None,
                 float16_optimizer_with_float16_params=None):
        pass

    def onload_param(self):
        pass

    def offload_optimizer(self):
        pass

    def offload_grad(self):
        pass

    def offload_param(self):
        pass


class MockBaseInferEngine:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockAsyncVLLMInferEngine:
    def __init__(self, *args, **kwargs):
        pass

    def sleep(self):
        pass

    def wake_up(self):
        pass

    def load_model(self):
        pass

    def init_worker(self):
        pass

    def execute_method(self, *args, **kwargs):
        pass


class MockActorHybridWorkerBase:
    def __init__(self,
                 megatron_config: MockMegatronConfig,
                 rl_config: MockRLConfig,
                 generate_config: MockGenerateConfig,
                 model_provider: Callable,
                 initialize_func: Callable,
                 tokenizer: MockBaseTokenizer = None,
                 get_megatron_module: Callable = None,
                 **kwargs):
        self.megatron_config = megatron_config
        self.rl_config = rl_config
        self.generate_config = generate_config
        self.model_provider = model_provider
        self.initialize_func = initialize_func
        self.tokenizer = tokenizer
        self.get_megatron_module = get_megatron_module
        self.kwargs = kwargs

        self.args = None

        self._all_consumed = 1
        self.enable_partial_rollout = False
        self.num_floating_point_operations_so_far = 0
        self.prof_iteration = 1

    def setup_distributed_rank(self):
        self.distributed_optimizer = MagicMock()
        self.float16_optimizer_with_float16_params = MagicMock()
        self.parallel_state = MagicMock()
        self.set_actual_seq_len = MagicMock()
        self.get_actual_seq_len = MagicMock()
        self.set_position_ids = MagicMock()
        self.forward_backward_func = MagicMock()
        self.args = MagicMock()

        def fake_get_data_parallel_world_size():
            return 1

        self.parallel_state.get_data_parallel_world_size.side_effect = fake_get_data_parallel_world_size

    def _build_model_optimizer(self):
        pass

    def _set_no_sync_func(self):
        pass

    def _build_sharding_manager(self):
        pass

    def empty_cache(self):
        pass

    def get_dp_range_indexes(self, experience_count, use_vllm=False):
        return [1]

    def all_consumed(self, experience_consumer_stage, sorted_indexes):
        _all_consumed = self._all_consumed
        self._all_consumed = self._all_consumed - 1
        return _all_consumed

    def dispatch_transfer_dock_data(self, *args, **kwargs):
        return MagicMock(), MagicMock()


class TestAgentActorHybridWorkerBase:

    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.config_cls": MagicMock(),
            "mindspeed_rl.config_cls.generate_config": MagicMock(),
            "mindspeed_rl.config_cls.megatron_config": MagicMock(),
            "mindspeed_rl.config_cls.rl_config": MagicMock(),
            "mindspeed_rl.models": MagicMock(),
            "mindspeed_rl.models.actor_rollout_hybrid": MagicMock(),
            "mindspeed_rl.utils": MagicMock(),
            "mindspeed_rl.utils.tokenizer": MagicMock(),
            "mindspeed_rl.utils.utils": MagicMock(),
            "mindspeed_rl.workers": MagicMock(),
            "mindspeed_rl.workers.actor_hybrid_worker": MagicMock(),
            "mindspeed_rl.workers.resharding": MagicMock(),
            "mindspeed_rl.workers.resharding.megatron_off_loader": MagicMock(),
            "mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state": MagicMock(),
            "mindspeed_rl.models.base": MagicMock(),
            "mindspeed_rl.models.base.base_inference_engine": MagicMock(),
            "vllm": MagicMock(),
            "vllm.config": MagicMock(),
            "vllm.worker": MagicMock(),
            "vllm.worker.worker_base": MagicMock(),
            "vllm_ascend": MagicMock(),
            "vllm_ascend.patch": MagicMock(),
        }):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        def fake_mstx_timer_decorator(func):
            return func

        def mock_ray_get(*args):
            return args

        def mock_remote(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                obj = args[0]
                obj.remote = obj
                return obj
            else:
                def decorator(obj):
                    obj.remote = obj
                    return obj

                return decorator

        with patch("ray.remote", mock_remote), \
                patch("ray.get", mock_ray_get), \
                patch("mindspeed_rl.config_cls.generate_config.GenerateConfig", MockGenerateConfig), \
                patch("mindspeed_rl.config_cls.megatron_config.MegatronConfig", MockMegatronConfig), \
                patch("mindspeed_rl.config_cls.rl_config.RLConfig", MockRLConfig), \
                patch("mindspeed_rl.models.actor_rollout_hybrid.ActorRolloutHybrid", MockActorRolloutHybrid), \
                patch("mindspeed_rl.utils.tokenizer.BaseTokenizer", MockBaseTokenizer), \
                patch("mindspeed_rl.workers.actor_hybrid_worker.ActorHybridWorkerBase", MockActorHybridWorkerBase), \
                patch("mindspeed_rl.workers.actor_hybrid_worker.ActorState", MockActorState), \
                patch("mindspeed_rl.workers.resharding.megatron_off_loader.MegatronOffLoader", MockMegatronOffLoader), \
                patch("mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state.initialize_parallel_state"), \
                patch("mindspeed_rl.models.base.base_inference_engine.BaseInferEngine", MockBaseInferEngine), \
                patch("vllm.worker.worker_base.WorkerWrapperBase", MagicMock), \
                patch("vllm.config.VllmConfig"), \
                patch("vllm_ascend.patch.platform"), \
                patch("vllm_ascend.patch.worker"), \
                patch("mindspeed_rl.utils.utils.mstx_timer_decorator", fake_mstx_timer_decorator), \
                patch("agentic_rl.runner.infer_adapter.vllm.patch.apply_patch"), \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.apply_patch"), \
                patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker."
                      "AsyncVLLMInferEngine", MockAsyncVLLMInferEngine):
            yield

    @pytest.fixture
    def actor_hybrid_worker(self, patch_target):
        from mindspeed_rl.config_cls.generate_config import GenerateConfig
        from mindspeed_rl.config_cls.megatron_config import MegatronConfig
        from mindspeed_rl.config_cls.rl_config import RLConfig
        from mindspeed_rl.utils.tokenizer import BaseTokenizer

        from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        agentic_rl_config = AgenticRLConfig()
        megatron_config = MegatronConfig()
        rl_config = RLConfig()
        generate_config = GenerateConfig()
        model_provider = MagicMock()
        initialize_func = MagicMock()
        tokenizer = BaseTokenizer()
        get_megatron_module = MagicMock()

        worker = AgentActorHybridWorkerBase(agentic_rl_config=agentic_rl_config,
                                            megatron_config=megatron_config,
                                            rl_config=rl_config,
                                            generate_config=generate_config,
                                            model_provider=model_provider,
                                            initialize_func=initialize_func,
                                            tokenizer=tokenizer,
                                            get_megatron_module=get_megatron_module)

        from mindspeed_rl import Metric
        worker.td = MagicMock()
        worker.td.metrics = Metric()
        worker.td.update_metrics = MagicMock()

        def fake_update_metrics(key="", value=None, cumulate=False):
            worker.td.metrics.update(key, value, cumulate=cumulate)

        worker.td.update_metrics.remote = fake_update_metrics

        yield (
            worker,
            {"agentic_rl_config": agentic_rl_config,
             "megatron_config": megatron_config,
             "rl_config": rl_config,
             "generate_config": generate_config,
             "model_provider": model_provider,
             "initialize_func": initialize_func,
             "tokenizer": tokenizer,
             "get_megatron_module": get_megatron_module},
            {}
        )

    def test_init_success_with_no_error(self, actor_hybrid_worker):
        worker, targets, _ = actor_hybrid_worker

        assert worker.agentic_rl_config == targets["agentic_rl_config"]
        assert worker.rl_config.zmq_communication is False
        assert worker.continue_infer_running is False

    def test_init_failed_with_params_error(self, actor_hybrid_worker):
        worker, targets, _ = actor_hybrid_worker

        from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        params = targets.copy()
        params["agentic_rl_config"] = 1
        with pytest.raises(ValueError, match="agentic_rl_config must not be none and is an instance of"):
            AgentActorHybridWorkerBase(**params)

        params = targets.copy()
        params["megatron_config"] = 1
        with pytest.raises(ValueError, match="megatron_config must not be none and is an instance of"):
            AgentActorHybridWorkerBase(**params)

        params = targets.copy()
        params["rl_config"] = 1
        with pytest.raises(ValueError, match="rl_config must not be none and is an instance of"):
            AgentActorHybridWorkerBase(**params)

        params = targets.copy()
        params["generate_config"] = 1
        with pytest.raises(ValueError, match="generate_config must not be none and is an instance of"):
            AgentActorHybridWorkerBase(**params)

        params = targets.copy()
        params["model_provider"] = 1
        with pytest.raises(ValueError, match="model_provider must not be none and is a Callable func"):
            AgentActorHybridWorkerBase(**params)

        params = targets.copy()
        params["initialize_func"] = 1
        with pytest.raises(ValueError, match="initialize_func must not be none and is a Callable func"):
            AgentActorHybridWorkerBase(**params)

        params = targets.copy()
        params["tokenizer"] = 1
        with pytest.raises(ValueError, match="tokenizer must not be none and is an instance of"):
            AgentActorHybridWorkerBase(**params)

        params = targets.copy()
        params["get_megatron_module"] = 1
        with pytest.raises(ValueError, match="get_megatron_module must not be none and is a Callable func"):
            AgentActorHybridWorkerBase(**params)

    def test_init_failed_with_super_class(self, actor_hybrid_worker):
        worker, targets, _ = actor_hybrid_worker

        from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "ActorHybridWorkerBase.__init__") as mock_super_class:
            mock_super_class.side_effect = AttributeError("error")
            with pytest.raises(AttributeError, match="AgentActorHybridWorkerBase initialize failed with missing"):
                AgentActorHybridWorkerBase(**targets)

            mock_super_class.side_effect = OSError("error")
            with pytest.raises(OSError, match="AgentActorHybridWorkerBase initialize failed with environment"):
                AgentActorHybridWorkerBase(**targets)

            mock_super_class.side_effect = Exception("error")
            with pytest.raises(Exception, match="Unexpected error occurred when AgentActorHybridWorkerBase"):
                AgentActorHybridWorkerBase(**targets)

    def test_initialize_failed_with_setup_distributed_rank_error(self, actor_hybrid_worker):
        worker, _, _ = actor_hybrid_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "AgentActorHybridWorkerBase.setup_distributed_rank") as mock_setup_distributed_rank:
            mock_setup_distributed_rank.side_effect = AttributeError("error")
            with pytest.raises(AttributeError,
                               match="actor worker setup distributed rank failed with missing attributes"):
                worker.initialize()

            mock_setup_distributed_rank.side_effect = Exception("error")
            with pytest.raises(Exception,
                               match="Unexpected error occurred when actor worker setup distributed rank"):
                worker.initialize()

    def test_initialize_failed_with_build_model_optimizer_error(self, actor_hybrid_worker):
        worker, _, _ = actor_hybrid_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "AgentActorHybridWorkerBase._build_model_optimizer") as mock_setup_distributed_rank:
            mock_setup_distributed_rank.side_effect = AttributeError("error")
            with pytest.raises(AttributeError,
                               match="actor worker initialize model failed with missing attributes"):
                worker.initialize()

            mock_setup_distributed_rank.side_effect = ValueError("error")
            with pytest.raises(ValueError,
                               match="actor worker initialize model failed with parameters conflict"):
                worker.initialize()

            mock_setup_distributed_rank.side_effect = Exception("error")
            with pytest.raises(Exception,
                               match="Unexpected error occurred when actor worker initialize model"):
                worker.initialize()

    def test_initialize_failed_with_build_megatron_offloader_error(self, actor_hybrid_worker):
        worker, _, _ = actor_hybrid_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "AgentActorHybridWorkerBase._build_model_optimizer") as mock_build_model_optimizer, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                      "AgentActorHybridWorkerBase._set_no_sync_func") as mock_set_no_sync_func, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                      "MegatronOffLoader") as mock_megatron_offloader:
            mock_build_model_optimizer.return_value = MagicMock(), MagicMock(), MagicMock()

            mock_megatron_offloader.side_effect = AttributeError("error")
            with pytest.raises(AttributeError,
                               match="actor worker failed to create offloader with missing attributes"):
                worker.initialize()

            mock_megatron_offloader.side_effect = OSError("error")
            with pytest.raises(OSError,
                               match="actor worker failed to create offloader with system error"):
                worker.initialize()

            mock_megatron_offloader.side_effect = Exception("error")
            with pytest.raises(Exception,
                               match="Unexpected error occurred when actor worker create offloader"):
                worker.initialize()

    def test_initialize_failed_with_offload_param_error(self, actor_hybrid_worker):
        worker, _, _ = actor_hybrid_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "AgentActorHybridWorkerBase._build_model_optimizer") as mock_build_model_optimizer, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                      "AgentActorHybridWorkerBase._set_no_sync_func") as mock_set_no_sync_func, \
                patch("mindspeed_rl.workers.resharding.megatron_off_loader."
                      "MegatronOffLoader.offload_optimizer") as mock_offload_optimizer, \
                patch("mindspeed_rl.workers.resharding.megatron_off_loader."
                      "MegatronOffLoader.offload_param") as mock_offload_param, \
                patch("mindspeed_rl.workers.resharding.megatron_off_loader."
                      "MegatronOffLoader.offload_grad") as mock_offload_grad, \
                patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained:
            mock_build_model_optimizer.return_value = MagicMock(), MagicMock(), MagicMock()

            mock_offload_param.side_effect = RuntimeError("error")
            with pytest.raises(RuntimeError,
                               match="actor worker offload optimizer/grad/param failed"):
                worker.initialize()

            mock_build_model_optimizer.assert_called_once()
            mock_set_no_sync_func.assert_called_once()
            mock_offload_optimizer.assert_called_once()
            mock_offload_grad.assert_called_once()
            mock_offload_param.assert_called_once()
            mock_from_pretrained.assert_not_called()

            mock_offload_param.side_effect = Exception("error")
            with pytest.raises(Exception,
                               match="Unexpected error occurred when actor worker offload optimizer/grad/param"):
                worker.initialize()

    @pytest.fixture
    def actor_hybrid_worker_initialized(self, actor_hybrid_worker):
        worker, targets, patches = actor_hybrid_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "AgentActorHybridWorkerBase._build_model_optimizer") as mock_build_model_optimizer, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                      "AgentActorHybridWorkerBase._set_no_sync_func") as mock_set_no_sync_func, \
                patch("mindspeed_rl.workers.resharding.megatron_off_loader."
                      "MegatronOffLoader.offload_optimizer") as mock_offload_optimizer, \
                patch("mindspeed_rl.workers.resharding.megatron_off_loader."
                      "MegatronOffLoader.offload_param") as mock_offload_param, \
                patch("mindspeed_rl.workers.resharding.megatron_off_loader."
                      "MegatronOffLoader.offload_grad") as mock_offload_grad, \
                patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained, \
                patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid") as mock_file_check:
            mock_build_model_optimizer.return_value = MagicMock(), MagicMock(), MagicMock()

            worker.initialize()

            yield (
                worker,
                {**targets},
                {**patches,
                 "mock_build_model_optimizer": mock_build_model_optimizer,
                 "mock_set_no_sync_func": mock_set_no_sync_func,
                 "mock_offload_optimizer": mock_offload_optimizer,
                 "mock_offload_param": mock_offload_param,
                 "mock_offload_grad": mock_offload_grad,
                 "mock_from_pretrained": mock_from_pretrained,
                 "mock_file_check": mock_file_check}
            )

    def test_initialize_success_with_no_error(self, actor_hybrid_worker_initialized):
        _, targets, patches = actor_hybrid_worker_initialized
        patches["mock_build_model_optimizer"].assert_called_once()
        patches["mock_set_no_sync_func"].assert_called_once()
        patches["mock_offload_optimizer"].assert_called_once()
        patches["mock_offload_grad"].assert_called_once()
        patches["mock_offload_param"].assert_called_once()
        patches["mock_from_pretrained"].assert_called_once()
        patches["mock_file_check"].assert_called()

    def test_init_sharding_manager_failed_with_not_initialized(self, actor_hybrid_worker):
        worker, targets, patches = actor_hybrid_worker

        with pytest.raises(AttributeError,
                           match="'AgentActorHybridWorkerBase' object has no attribute 'inference_model'"):
            worker.init_sharding_manager()

    def test_init_sharding_manager_failed_with_build_sharding_manager(self, actor_hybrid_worker_initialized):
        worker, targets, patches = actor_hybrid_worker_initialized
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "AgentActorHybridWorkerBase._build_sharding_manager") as mock_build_sharding_manager:
            mock_build_sharding_manager.side_effect = AttributeError("error")
            with pytest.raises(AttributeError, match="actor worker build sharding manager failed with missing"):
                worker.init_sharding_manager()

            mock_build_sharding_manager.side_effect = RuntimeError("error")
            with pytest.raises(RuntimeError, match="actor worker build sharding manager failed with runtime"):
                worker.init_sharding_manager()

            mock_build_sharding_manager.side_effect = Exception("error")
            with pytest.raises(Exception, match="Unexpected error occurred when actor worker build sharding"):
                worker.init_sharding_manager()

    def test_init_sharding_manager_failed_with_on_load_param(self, actor_hybrid_worker_initialized):
        worker, targets, patches = actor_hybrid_worker_initialized
        with patch("mindspeed_rl.workers.resharding.megatron_off_loader."
                   "MegatronOffLoader.onload_param") as mock_onload_param:
            mock_onload_param.side_effect = RuntimeError("error")
            with pytest.raises(RuntimeError, match="actor worker onload parameters failed"):
                worker.init_sharding_manager()

            mock_onload_param.side_effect = Exception("error")
            with pytest.raises(Exception, match="Unexpected error occurred when actor worker onload parameters"):
                worker.init_sharding_manager()

    def test_init_sharding_manager_failed_by_dp_size_zero(self, actor_hybrid_worker_initialized):
        worker, targets, patches = actor_hybrid_worker_initialized

        def fake_get_data_parallel_world_size():
            return 0

        worker.parallel_state.get_data_parallel_world_size = fake_get_data_parallel_world_size

        with pytest.raises(Exception, match="actor worker init actor hybrid failed") as outer_error:
            worker.init_sharding_manager()

        assert "actor worker get data parallel world size equal 0" in str(outer_error.value.__cause__)

    def test_init_sharding_manager_failed_by_empty_cache(self, actor_hybrid_worker_initialized):
        worker, targets, patches = actor_hybrid_worker_initialized

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "AgentActorHybridWorkerBase.empty_cache") as mock_empty_cache:
            mock_empty_cache.side_effect = RuntimeError("error")
            with pytest.raises(RuntimeError, match="actor worker failed to empty the cache"):
                worker.init_sharding_manager()

            mock_empty_cache.side_effect = Exception("error")
            with pytest.raises(Exception, match="Unexpected error occurred when actor worker empty the cache"):
                worker.init_sharding_manager()

    @pytest.fixture
    def actor_hybrid_worker_init_shard_manager(self, actor_hybrid_worker_initialized):
        worker, targets, patches = actor_hybrid_worker_initialized
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "AsyncVLLMInferEngine.sleep", Mock()) as mock_sleep, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                      "AgentActorHybridWorkerBase._build_sharding_manager") as mock_build_sharding_manager, \
                patch("mindspeed_rl.workers.resharding.megatron_off_loader."
                      "MegatronOffLoader.onload_param") as mock_onload_param:
            sharding_manager = MagicMock()
            mock_build_sharding_manager.return_value = sharding_manager

            worker.init_sharding_manager()

            yield (
                worker,
                {**targets,
                 "sharding_manager": sharding_manager},
                {**patches,
                 "mock_sleep": mock_sleep,
                 "mock_build_sharding_manager": mock_build_sharding_manager,
                 "mock_onload_param": mock_onload_param}
            )

    def test_init_sharding_manager_success_with_no_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        patches["mock_sleep"].assert_called_once()
        assert worker.sharding_manager == targets["sharding_manager"]
        patches["mock_onload_param"].assert_called_once()
        assert isinstance(worker.actor_hybrid, MockActorRolloutHybrid)

    def test_get_worker_info(self, actor_hybrid_worker):
        worker, targets, patches = actor_hybrid_worker

        with patch("os.environ.get") as mock_getenv, \
                patch("ray.get_runtime_context") as mock_get_runtime_context:
            mock_getenv.return_value = 2

            def fake_get_node_id():
                return 1

            mock_runtime_context = MagicMock()
            mock_runtime_context.get_node_id.side_effect = fake_get_node_id

            def fake_get_runtime_context():
                return mock_runtime_context

            mock_get_runtime_context.side_effect = fake_get_runtime_context

            rank, node_id = worker.get_worker_info()

            assert rank == 2
            assert node_id == 1

    def test_enter_infer_mode_success_with_no_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        worker.state = MockActorState.NONE
        mock_enter_infer_mode = MagicMock()
        targets["sharding_manager"].enter_infer_mode = mock_enter_infer_mode

        worker.enter_infer_mode()

        assert worker.state == MockActorState.INFER
        mock_enter_infer_mode.assert_called_once()
        worker.td.metrics.update.assert_called()

    def test_enter_infer_mode_skip_with_state_infer(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        worker.state = MockActorState.INFER
        mock_enter_infer_mode = MagicMock()
        targets["sharding_manager"].enter_infer_mode = mock_enter_infer_mode

        worker.enter_infer_mode()

        mock_enter_infer_mode.assert_not_called()

    def test_enter_infer_mode_failed_with_enter_infer_mode_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        worker.state = MockActorState.NONE
        mock_enter_infer_mode = MagicMock()
        targets["sharding_manager"].enter_infer_mode = mock_enter_infer_mode

        mock_enter_infer_mode.side_effect = RuntimeError("error")
        with pytest.raises(RuntimeError, match="actor worker enter infer mode failed"):
            worker.enter_infer_mode()

        mock_enter_infer_mode.side_effect = Exception("error")
        with pytest.raises(Exception, match="Unexpected error occurred when actor worker enter infer mode"):
            worker.enter_infer_mode()

    def test_exit_infer_mode_success_with_no_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        worker.state = MockActorState.INFER
        mock_exit_infer_mode = MagicMock()
        targets["sharding_manager"].exit_infer_mode = mock_exit_infer_mode

        worker.exit_infer_mode()

        assert worker.state == MockActorState.NONE
        mock_exit_infer_mode.assert_called_once()
        worker.td.metrics.update.assert_called()

    def test_exit_infer_mode_failed_with_state_not_infer(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        worker.state = MockActorState.NONE

        with pytest.raises(RuntimeError, match="current state is not INFER, it is no available to exit infer mode"):
            worker.exit_infer_mode()

    def test_exit_infer_mode_failed_with_exit_infer_mode_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        worker.state = MockActorState.INFER
        mock_exit_infer_mode = MagicMock()
        targets["sharding_manager"].exit_infer_mode = mock_exit_infer_mode

        mock_exit_infer_mode.side_effect = RuntimeError("error")
        with pytest.raises(RuntimeError, match="actor worker exit infer mode failed"):
            worker.exit_infer_mode()

        mock_exit_infer_mode.side_effect = Exception("error")
        with pytest.raises(Exception, match="Unexpected error occurred when actor worker exit infer mode"):
            worker.exit_infer_mode()

    def test_init_worker_success_with_no_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.AsyncVLLMInferEngine."
                   "init_worker") as mock_model_init_worker:
            worker.init_worker([])

            mock_model_init_worker.assert_called_once()

    def test_load_model_success_with_no_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.AsyncVLLMInferEngine."
                   "load_model") as mock_model_load_model:
            worker.load_model()

            mock_model_load_model.assert_called_once()

    def test_sleep_success_with_no_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.AsyncVLLMInferEngine."
                   "sleep") as mock_model_sleep:
            worker.inference_model.is_sleep = False
            worker.state = MockActorState.INFER
            targets["sharding_manager"].exit_infer_node = MagicMock()

            worker.sleep()

            assert worker.inference_model.is_sleep is True
            mock_model_sleep.assert_called_once()
            worker.td.metrics.update.assert_called()

    def test_wake_up_success_with_no_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.AsyncVLLMInferEngine."
                   "wake_up") as mock_model_wake_up:
            worker.inference_model.is_sleep = True
            worker.state = MockActorState.NONE
            worker.continue_infer_running = True
            targets["sharding_manager"].enter_forward_mode = MagicMock()
            targets["sharding_manager"].exit_infer_mode = MagicMock()

            worker.wake_up()

            assert worker.inference_model.is_sleep is False
            mock_model_wake_up.assert_called_once()
            worker.td.metrics.update.assert_called()

    def test_execute_method_success_with_no_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "AgentActorHybridWorkerBase.init_worker") as mock_init_worker, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                      "AgentActorHybridWorkerBase.load_model") as mock_load_model, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                      "AgentActorHybridWorkerBase.sleep") as mock_sleep, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                      "AgentActorHybridWorkerBase.wake_up") as mock_wake_up, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.AsyncVLLMInferEngine."
                      "execute_method") as mock_execute_method:
            worker.execute_method("init_worker")
            worker.execute_method("load_model")
            worker.execute_method("sleep")
            worker.execute_method("wake_up")
            worker.execute_method("other_method")

            mock_init_worker.assert_called_once()
            mock_load_model.assert_called_once()
            mock_sleep.assert_called_once()
            mock_wake_up.assert_called_once()
            mock_execute_method.assert_called_once()

    def test_update_success_with_no_error(self, actor_hybrid_worker_init_shard_manager):
        worker, targets, patches = actor_hybrid_worker_init_shard_manager

        kl_ctrl = MagicMock()
        targets["sharding_manager"].enter_train_mode = MagicMock()
        targets["rl_config"].actor_update_dispatch_size = 10
        skip_actor_log_prob = True
        worker.iteration = 1

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                   "ActorRolloutHybrid.update_actor") as mock_update_actor:
            worker.update(kl_ctrl, skip_actor_log_prob=skip_actor_log_prob)
            mock_update_actor.assert_called_once()
            worker.td.metrics.update.assert_called()
