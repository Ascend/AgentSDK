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
from typing import Callable
from unittest.mock import patch, MagicMock

import pytest
import ray

from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from test_agent_hybrid_worker import (MockRLConfig,
                                      MockMegatronConfig,
                                      MockGenerateConfig,
                                      MockBaseTokenizer,
                                      MockMegatronOffLoader)


class BaseWorker:
    def __init__(self, *args, **kwargs):
        self.model_type = None

    pass


class MockReferenceWorkerBase(BaseWorker):
    pass


class MockRewardWorkerBase(BaseWorker):
    pass


class MockActorHybridWorkerBase(BaseWorker):

    def __init__(self, megatron_config: MockMegatronConfig,
                 rl_config: MockRLConfig,
                 generate_config: MockGenerateConfig,
                 model_provider: Callable,
                 initialize_func: Callable,
                 tokenizer: MockBaseTokenizer = None,
                 get_megatron_module: Callable = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.megatron_config = megatron_config
        self.rl_config = rl_config
        self.generate_config = generate_config
        self.model_provider = model_provider
        self.initialize_func = initialize_func
        self.tokenizer = tokenizer
        self.get_megatron_module = get_megatron_module
        self.kwargs = kwargs

    def initialize(self):
        self.get_model = MagicMock()
        self.model_provider = MagicMock()
        self.forward_backward_func = MagicMock()
        self.set_actual_seq_len = MagicMock()
        self.get_actual_seq_len = MagicMock()
        self.set_position_ids = MagicMock()

    def compute_log_prob(self):
        pass

    def _build_rollout(self):
        pass

    def enter_infer_mode(self):
        pass

    def exit_infer_mode(self):
        pass


class MockAgentActorHybridWorkerBase(MockActorHybridWorkerBase):
    def __init__(self,
                 agentic_rl_config: AgenticRLConfig,
                 megatron_config: MockMegatronConfig,
                 rl_config: MockRLConfig,
                 generate_config: MockGenerateConfig,
                 model_provider: Callable,
                 initialize_func: Callable,
                 tokenizer: MockBaseTokenizer = None,
                 get_megatron_module: Callable = None,
                 **kwargs):
        super().__init__(megatron_config,
                         rl_config,
                         generate_config,
                         model_provider,
                         initialize_func,
                         tokenizer,
                         get_megatron_module,
                         **kwargs)
        self.agentic_rl_config = agentic_rl_config

    def initialize(self):
        super().initialize()

    def compute_log_prob(self):
        pass

    def _build_rollout(self):
        pass

    def enter_infer_mode(self):
        pass

    def exit_infer_mode(self):
        pass

    def update(self, kl_ctrl=None, skip_actor_log_prob: bool = False):
        pass


@ray.remote
class MockIntegratedWorker(MockActorHybridWorkerBase, MockReferenceWorkerBase, MockRewardWorkerBase):

    def compute_ref_log_prob(self):
        pass

    def load_checkpoint_with_path(self, *args, **kwargs):
        pass


class TestIntegratedWorker:

    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.config_cls": MagicMock(),
            "mindspeed_rl.config_cls.generate_config": MagicMock(),
            "mindspeed_rl.config_cls.megatron_config": MagicMock(),
            "mindspeed_rl.config_cls.rl_config": MagicMock(),
            "mindspeed_rl.models": MagicMock(),
            "mindspeed_rl.models.reference": MagicMock(),
            "mindspeed_rl.utils": MagicMock(),
            "mindspeed_rl.utils.tokenizer": MagicMock(),
            "mindspeed_rl.workers": MagicMock(),
            "mindspeed_rl.workers.integrated_worker": MagicMock(),
            "mindspeed_rl.workers.reference_worker": MagicMock(),
            "mindspeed_rl.workers.resharding": MagicMock(),
            "mindspeed_rl.workers.resharding.megatron_off_loader": MagicMock(),
            "mindspeed_rl.workers.reward_worker": MagicMock(),
            "agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker": MagicMock(),
        }):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        with patch("mindspeed_rl.config_cls.generate_config.GenerateConfig", MockGenerateConfig), \
                patch("mindspeed_rl.config_cls.megatron_config", MockMegatronConfig), \
                patch("mindspeed_rl.config_cls.rl_config.RLConfig", MockRLConfig), \
                patch("mindspeed_rl.models.reference.Reference"), \
                patch("mindspeed_rl.utils.tokenizer.BaseTokenizer", MockBaseTokenizer), \
                patch("mindspeed_rl.workers.integrated_worker.IntegratedWorker", MockIntegratedWorker), \
                patch("mindspeed_rl.workers.reference_worker.ReferenceWorkerBase", MockReferenceWorkerBase), \
                patch("mindspeed_rl.workers.resharding.megatron_off_loader.MegatronOffLoader", MockMegatronOffLoader), \
                patch("mindspeed_rl.workers.reward_worker.RewardWorkerBase", MockRewardWorkerBase), \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker."
                      "AgentActorHybridWorkerBase", MockAgentActorHybridWorkerBase), \
                patch("ray.remote") as mock_remote:
            def fake_remote(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    arg = args[0]
                    arg.remote = arg
                    return arg
                else:
                    def decorator(obj):
                        obj.remote = obj
                        return obj

                    return decorator

            mock_remote.side_effect = fake_remote
            yield

    @pytest.fixture
    def integrated_worker(self, patch_target):
        from mindspeed_rl.config_cls.generate_config import GenerateConfig
        from mindspeed_rl.config_cls.megatron_config import MegatronConfig
        from mindspeed_rl.config_cls.rl_config import RLConfig
        from mindspeed_rl.utils.tokenizer import BaseTokenizer

        from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker import IntegratedWorker

        agentic_rl_config = AgenticRLConfig()
        megatron_config = MegatronConfig()
        rl_config = RLConfig()
        generate_config = GenerateConfig()
        model_provider = MagicMock()
        initialize_func = MagicMock()
        tokenizer = BaseTokenizer()
        get_megatron_module = MagicMock()

        worker = IntegratedWorker(agentic_rl_config=agentic_rl_config,
                                  megatron_config=megatron_config,
                                  rl_config=rl_config,
                                  generate_config=generate_config,
                                  model_provider=model_provider,
                                  initialize_func=initialize_func,
                                  tokenizer=tokenizer,
                                  get_megatron_module=get_megatron_module)

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

    def test_init_success_with_no_error(self, integrated_worker):
        worker, targets, _ = integrated_worker

        assert worker.actor_forward_micro_batch_size == targets["rl_config"].actor_forward_micro_batch_size
        assert worker.ref_forward_micro_batch_size == targets["rl_config"].ref_forward_micro_batch_size
        assert worker.reference is None
        assert worker.ref_model is None
        assert worker.ref_manager is None

    def test_initialize_failed_with_get_model_failed(self, integrated_worker):
        worker, _, _ = integrated_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker."
                   "AgentActorHybridWorkerBase.initialize") as mock_initialize:
            mock_get_model = MagicMock()

            def fake_initialize(self):
                self.get_model = mock_get_model

            mock_initialize.side_effect = fake_initialize

            mock_get_model.side_effect = AttributeError("error")
            with pytest.raises(AttributeError,
                               match="actor worker failed to initialize reference model with missing attributes"):
                worker.initialize()
            mock_initialize.assert_called_once()

            mock_get_model.side_effect = Exception("error")
            with pytest.raises(Exception,
                               match="Unexpected error occurred when actor worker initialize reference model"):
                worker.initialize()

    def test_initialize_failed_with_load_checkpoint(self, integrated_worker):
        worker, _, _ = integrated_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker."
                   "IntegratedWorker.load_checkpoint_with_path") as mock_load_checkpoint:
            mock_load_checkpoint.side_effect = AttributeError("error")
            with pytest.raises(AttributeError,
                               match="actor worker load checkpoint for ref model failed with missing attributes"):
                worker.initialize()

            mock_load_checkpoint.side_effect = Exception("error")
            with pytest.raises(Exception,
                               match="Unexpected error occurred when actor worker load checkpoint for ref model"):
                worker.initialize()

    def test_initialize_failed_with_create_offloader(self, integrated_worker):
        worker, _, _ = integrated_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker."
                   "MegatronOffLoader") as mock_megatron_offloader:
            mock_megatron_offloader.side_effect = AttributeError("error")
            with pytest.raises(AttributeError,
                               match="actor worker create offloader for ref model failed with missing attributes"):
                worker.initialize()

            mock_megatron_offloader.side_effect = OSError("error")
            with pytest.raises(OSError,
                               match="actor worker create offloader for ref model failed with system error"):
                worker.initialize()

            mock_megatron_offloader.side_effect = Exception("error")
            with pytest.raises(Exception,
                               match="Unexpected error occurred when actor worker create offloader for ref model"):
                worker.initialize()

    def test_initialize_failed_with_offload_param(self, integrated_worker):
        worker, _, _ = integrated_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker."
                   "MegatronOffLoader.offload_param") as mock_offload_param:
            mock_offload_param.side_effect = RuntimeError("error")
            with pytest.raises(RuntimeError,
                               match="actor worker offload param for ref model failed"):
                worker.initialize()

            mock_offload_param.side_effect = Exception("error")
            with pytest.raises(Exception,
                               match="Unexpected error occurred when actor worker offload param for ref model"):
                worker.initialize()

    def test_initialize_failed_with_create_reference(self, integrated_worker):
        worker, _, _ = integrated_worker

        del worker.generate_config.sampling_config
        with pytest.raises(AttributeError, match="actor worker init reference failed with missing attributes"):
            worker.initialize()

        worker.generate_config = MockGenerateConfig()
        del worker.generate_config.sampling_config["temperature"]
        with pytest.raises(KeyError, match="actor worker init reference failed with missing keys"):
            worker.initialize()

    def test_initialize_success_with_no_error(self, integrated_worker):
        worker, _, _ = integrated_worker

        worker.initialize()

    def test_build_rollout_success_with_no_error(self, integrated_worker):
        worker, _, _ = integrated_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker."
                   "AgentActorHybridWorkerBase._build_rollout") as mock_actor_build_rollout:
            worker._build_rollout()

            mock_actor_build_rollout.assert_called_once()

    def test_enter_infer_mode_success_with_no_error(self, integrated_worker):
        worker, _, _ = integrated_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker."
                   "AgentActorHybridWorkerBase.enter_infer_mode") as mock_enter_infer_mode:
            worker.enter_infer_mode()

            mock_enter_infer_mode.assert_called_once()

    def test_exit_infer_mode_success_with_no_error(self, integrated_worker):
        worker, _, _ = integrated_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker."
                   "AgentActorHybridWorkerBase.exit_infer_mode") as mock_exit_infer_mode:
            worker.exit_infer_mode()

            mock_exit_infer_mode.assert_called_once()

    def test_update_success_with_no_error(self, integrated_worker):
        worker, _, _ = integrated_worker

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker."
                   "AgentActorHybridWorkerBase.update") as mock_update:
            worker.update(kl_ctrl=MagicMock())

            mock_update.assert_called_once()
