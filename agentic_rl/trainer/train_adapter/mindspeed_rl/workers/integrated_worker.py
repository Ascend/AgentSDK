#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from typing import Callable

import ray
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.models.reference import Reference
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.integrated_worker import IntegratedWorker as _IntegratedWorker
from mindspeed_rl.workers.integrated_worker import temporary_micro_batch_size
from mindspeed_rl.workers.reference_worker import ReferenceWorkerBase
from mindspeed_rl.workers.resharding.megatron_off_loader import MegatronOffLoader
from mindspeed_rl.workers.reward_worker import RewardWorkerBase

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.file_utils import FileCheck
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

logger = Loggers(__name__)

# IntegratedWorker is wrapped by ray.remote, so we need to get the real class
try:
    raw_integrated_worker_cls = _IntegratedWorker.__ray_metadata__.modified_class
except AttributeError as attribute_error:
    logger.error(f"Unable to get IntegratedWorker from msrl: {attribute_error}")
    raise AttributeError("Unable to get IntegratedWorker from msrl.") from attribute_error


@ray.remote(resources={"NPU": 0.7})
class IntegratedWorker(raw_integrated_worker_cls, AgentActorHybridWorkerBase, ReferenceWorkerBase, RewardWorkerBase):
    """
    IntegratedWorker class. This class implements the integrated worker for the Actor, Reference and Reward Worker.
    """

    def __init__(
            self,
            agentic_rl_config: AgenticRLConfig,
            megatron_config: MegatronConfig,
            rl_config: RLConfig,
            generate_config: GenerateConfig,
            model_provider: Callable,
            initialize_func: Callable,
            tokenizer: BaseTokenizer = None,
            get_megatron_module: Callable = None,
            **kwargs
    ):
        """
        Initialization for IntegratedWorker.

        Args:
            megatron_config: MegatronConfig Configuration for Megatron-LM (e.g., model parallelism settings).
            rl_config: RLConfig Configuration for reinforcement learning (e.g., PPO settings).
            generate_config: GenerateConfig Configuration for generation/inference (e.g., vLLM settings).
            model_provider: Callable Function to provide the model instance.
            initialize_func: Callable Function to initialize the model and environment.
            tokenizer: BaseTokenizer = None Object to retrieve the tokenizer.
            get_megatron_module: Callable = megatron_module from get_megatron_module.
            kwargs: dict = extra args for megatron_config
        """
        # We use Actor as main worker, so only do init for Actor here.
        AgentActorHybridWorkerBase.__init__(
            self,
            agentic_rl_config=agentic_rl_config,
            megatron_config=megatron_config,
            rl_config=rl_config,
            generate_config=generate_config,
            model_provider=model_provider,
            initialize_func=initialize_func,
            tokenizer=tokenizer,
            get_megatron_module=get_megatron_module,
            **kwargs
        )

        self.actor_forward_micro_batch_size = rl_config.actor_forward_micro_batch_size
        self.ref_forward_micro_batch_size = rl_config.ref_forward_micro_batch_size

        self.reference = None
        self.ref_model = None
        self.ref_manager = None

    def get_master_addr_port(self):
        """
        get master's ip and port
        """
        if getattr(self, "_master_port", None) is None:
            logger.error("IntegratedWorker is not correct initialized, unable to get master port")
            raise ValueError("IntegratedWorker is not correct initialized, unable to get master port")

        return "127.0.0.1", self._master_port

    def initialize(self):
        """
        Initialize actor worker's model for training and inference. When using Integrated worker, reference model will
            also be initialized here.
        """
        AgentActorHybridWorkerBase.initialize(self)

        try:
            self.ref_model = self.get_model(self.model_provider, self.model_type, wrap_with_ddp=False)
        except AttributeError as e:
            raise AttributeError("actor worker failed to initialize reference model with missing attributes") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker initialize reference model") from e
        logger.debug(f"actor worker initialize reference model successfully")

        ref_model_load_path = getattr(
            self.rl_config.integrated_mode_config, "ref_model_load_path", None
        ) if self.rl_config.integrated_mode_config is not None else None
        if ref_model_load_path is not None:
            FileCheck.check_data_path_is_valid(ref_model_load_path)

        try:
            self.load_checkpoint_with_path(self.ref_model, ref_model_load_path, ckpt_only=True)
        except AttributeError as e:
            raise AttributeError("actor worker load checkpoint for ref model failed with missing attributes") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker load checkpoint for ref model") from e
        logger.debug(f"actor worker load weight for reference model successfully")

        try:
            self.ref_manager = MegatronOffLoader(self.ref_model, wrap_with_ddp=False)
        except AttributeError as e:
            raise AttributeError("actor worker create offloader for ref model failed with missing attributes") from e
        except OSError as e:
            raise OSError("actor worker create offloader for ref model failed with system error") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker create offloader for ref model") from e

        try:
            self.ref_manager.offload_param()
        except RuntimeError as e:
            raise RuntimeError("actor worker offload param for ref model failed") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker offload param for ref model") from e

        try:
            self._init_reference()
        except KeyError as e:
            raise KeyError("actor worker init reference failed with missing keys") from e
        except AttributeError as e:
            raise AttributeError("actor worker init reference failed with missing attributes") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker init reference") from e

    def _init_reference(self):
        self.reference = Reference(
            self.ref_model,
            megatron_config=self.megatron_config,
            beta=self.rl_config.beta,
            mini_batch_size=self.rl_config.mini_batch_size,
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            micro_batch_size=self.megatron_config.micro_batch_size,
            use_dynamic_bsz=self.rl_config.use_dynamic_bsz,
            max_packing_token_size=self.rl_config.max_packing_token_size,
            dynamic_max_batch_size=self.rl_config.dynamic_max_batch_size,
            use_remove_padding=self.rl_config.use_remove_padding,
            set_actual_seq_len=self.set_actual_seq_len,
            get_actual_seq_len=self.get_actual_seq_len,
            set_position_ids=self.set_position_ids,
            context_parallel_size=self.megatron_config.context_parallel_size,
            temperature=self.generate_config.sampling_config["temperature"]
        )

    def compute_log_prob(self):
        if self.actor_forward_micro_batch_size is not None:
            with temporary_micro_batch_size(
                    worker=self.actor_hybrid.train_actor,
                    args=self.get_args(),
                    new_mbs=self.actor_forward_micro_batch_size
            ):
                AgentActorHybridWorkerBase.compute_log_prob(self)
        else:
            AgentActorHybridWorkerBase.compute_log_prob(self)

    def _build_rollout(self):
        return AgentActorHybridWorkerBase._build_rollout(self)

    def enter_infer_mode(self):
        AgentActorHybridWorkerBase.enter_infer_mode(self)

    def exit_infer_mode(self):
        AgentActorHybridWorkerBase.exit_infer_mode(self)

    def update(self, kl_ctrl=None, skip_actor_log_prob: bool = False):
        AgentActorHybridWorkerBase.update(self, kl_ctrl, skip_actor_log_prob)

    def update_ref_dispatch_size(self, new_ref_dispatch_size: int):
        self.rl_config.ref_dispatch_size = new_ref_dispatch_size // self.parallel_state.get_data_parallel_world_size()
