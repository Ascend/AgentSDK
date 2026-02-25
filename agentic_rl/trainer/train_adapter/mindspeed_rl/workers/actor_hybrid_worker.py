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
import os
import time
from typing import Any, Dict, List, Union, Callable

import torch.cuda

from agentic_rl.trainer.train_adapter.mindspeed_rl import patch

patch.apply_patch()

import ray
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.models.actor_rollout_hybrid import ActorRolloutHybrid
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.utils.utils import replace_torch_compile, mstx_timer_decorator
from mindspeed_rl.workers.actor_hybrid_worker import (ActorHybridWorkerBase,
                                                      num_floating_point_operations,
                                                      ActorState)
from mindspeed_rl.workers.resharding.megatron_off_loader import MegatronOffLoader
from transformers import AutoConfig

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from agentic_rl.trainer.train_adapter.mindspeed_rl.vllm_infer_engine import AsyncVLLMInferEngine
from agentic_rl.base.utils.file_utils import FileCheck

logger = Loggers(__name__)


class AgentActorHybridWorkerBase(ActorHybridWorkerBase):
    """
    AgentActorHybridWorkerBase class. This class implements the hybrid worker logic for training and inference.
    """

    def __init__(self,
                 agentic_rl_config: AgenticRLConfig,
                 megatron_config: MegatronConfig,
                 rl_config: RLConfig,
                 generate_config: GenerateConfig,
                 model_provider: Callable,
                 initialize_func: Callable,
                 tokenizer: BaseTokenizer = None,
                 get_megatron_module: Callable = None,
                 **kwargs):
        """
        Initialization for AgentActorHybridWorkerBase.

        Args:
            agentic_rl_config: AgenticRLConfig Configuration for agent trajectory.
            megatron_config: MegatronConfig Configuration for Megatron-LM (e.g., model parallelism settings).
            rl_config: RLConfig Configuration for reinforcement learning (e.g., PPO settings).
            generate_config: GenerateConfig Configuration for generation/inference (e.g., vLLM settings).
            model_provider: Callable Function to provide the model instance.
            initialize_func: Callable Function to initialize the model and environment.
            tokenizer: BaseTokenizer = None Object to retrieve the tokenizer.
            get_megatron_module: Callable = megatron_module from get_megatron_module.
        """
        if agentic_rl_config is None or not isinstance(agentic_rl_config, AgenticRLConfig):
            raise ValueError(f"agentic_rl_config must not be none and is an instance of {AgenticRLConfig.__name__}")
        if megatron_config is None or not isinstance(megatron_config, MegatronConfig):
            raise ValueError(f"megatron_config must not be none and is an instance of {MegatronConfig.__name__}")
        if rl_config is None or not isinstance(rl_config, RLConfig):
            raise ValueError(f"rl_config must not be none and is an instance of {RLConfig.__name__}")
        if generate_config is None or not isinstance(generate_config, GenerateConfig):
            raise ValueError(f"generate_config must not be none and is an instance of {GenerateConfig.__name__}")
        if model_provider is None or not isinstance(model_provider, Callable):
            raise ValueError(f"model_provider must not be none and is a Callable func")
        if initialize_func is None or not isinstance(initialize_func, Callable):
            raise ValueError(f"initialize_func must not be none and is a Callable func")
        if tokenizer is not None and not isinstance(tokenizer, BaseTokenizer):
            raise ValueError(f"tokenizer must not be none and is an instance of {BaseTokenizer.__name__}")
        if get_megatron_module is not None and not isinstance(get_megatron_module, Callable):
            raise ValueError(f"get_megatron_module must not be none and is a Callable func")

        try:
            super().__init__(megatron_config=megatron_config,
                             rl_config=rl_config,
                             generate_config=generate_config,
                             model_provider=model_provider,
                             initialize_func=initialize_func,
                             tokenizer=tokenizer,
                             get_megatron_module=get_megatron_module,
                             **kwargs)
        except AttributeError as e:
            raise AttributeError("AgentActorHybridWorkerBase initialize failed with missing attributes") from e
        except OSError as e:
            raise OSError("AgentActorHybridWorkerBase initialize failed with environment error") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when AgentActorHybridWorkerBase initialize") from e

        self.agentic_rl_config = agentic_rl_config
        self.rl_config.zmq_communication = False
        self.continue_infer_running = False
        self.start_time_defined = False

    def initialize(self):
        """
        Initialize actor worker's model for training and inference.
        """
        try:
            self.setup_distributed_rank()
        except AttributeError as e:
            raise AttributeError("actor worker setup distributed rank failed with missing attributes") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker setup distributed rank") from e
        logger.debug("actor worker setup distributed rank success")

        try:
            self.model, self.optimizer, self.opt_param_scheduler = self._build_model_optimizer()
            self._set_no_sync_func()
        except AttributeError as e:
            raise AttributeError("actor worker initialize model failed with missing attributes") from e
        except ValueError as e:
            raise ValueError("actor worker initialize model failed with parameters conflict") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker initialize model") from e
        logger.debug("actor worker initialize training model success")

        try:
            self.actor_offloader = MegatronOffLoader(
                self.model,
                self.optimizer,
                megatron_config=self.megatron_config,
                distributed_optimizer=self.distributed_optimizer,
                float16_optimizer_with_float16_params=self.float16_optimizer_with_float16_params)
        except AttributeError as e:
            raise AttributeError("actor worker failed to create offloader with missing attributes") from e
        except OSError as e:
            raise OSError("actor worker failed to create offloader with system error") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker create offloader") from e
        logger.debug("actor worker create offloader success")

        try:
            if self.generate_config.offload_train_optimizer:
                self.actor_offloader.offload_optimizer()
            if self.generate_config.offload_train_grad:
                self.actor_offloader.offload_grad()
            if self.generate_config.offload_train_param:
                self.actor_offloader.offload_param()
        except RuntimeError as e:
            raise RuntimeError("actor worker offload optimizer/grad/param failed") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker offload optimizer/grad/param") from e

        with replace_torch_compile():
            self.inference_model = self._build_rollout()

    def _build_rollout(self) -> AsyncVLLMInferEngine:
        FileCheck.check_data_path_is_valid(self.megatron_config.tokenizer_name_or_path)
        self.actor_model_config = AutoConfig.from_pretrained(
            self.megatron_config.tokenizer_name_or_path,
            trust_remote_code=False,
            local_files_only=True,
            weights_only=True,
        )

        rollout = AsyncVLLMInferEngine(
            tokenizer_name_or_path=self.megatron_config.tokenizer_name_or_path,
            train_tensor_parallel_size=self.megatron_config.tensor_model_parallel_size,
            train_pipeline_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            train_expert_parallel_size=self.megatron_config.expert_model_parallel_size,
            train_context_parallel_size=self.megatron_config.context_parallel_size,
            infer_tensor_parallel_size=self.generate_config.infer_tensor_parallel_size,
            infer_pipeline_parallel_size=self.generate_config.infer_pipeline_parallel_size,
            infer_expert_parallel_size=self.generate_config.infer_expert_parallel_size,
            max_num_seqs=self.generate_config.max_num_seqs,
            max_model_len=self.generate_config.max_model_len,
            dtype=self.generate_config.dtype,
            gpu_memory_utilization=self.generate_config.gpu_memory_utilization,
            trust_remote_code=self.generate_config.trust_remote_code,
            enable_sleep_mode=self.agentic_rl_config.enable_sleep_mode,
        )

        return rollout

    def init_sharding_manager(self):
        """
        Initialize actor worker's sharding manager and actor_hybrid. This operation should be called after
            RolloutWorker init.
        """
        self.inference_model.sleep()

        try:
            self.sharding_manager = self._build_sharding_manager()
        except AttributeError as e:
            raise AttributeError("actor worker build sharding manager failed with missing attributes") from e
        except RuntimeError as e:
            raise RuntimeError("actor worker build sharding manager failed with runtime error") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker build sharding manager") from e

        if self.generate_config.offload_train_param:
            try:
                self.actor_offloader.onload_param()
            except RuntimeError as e:
                raise RuntimeError("actor worker onload parameters failed") from e
            except Exception as e:
                raise Exception("Unexpected error occurred when actor worker onload parameters") from e

        try:
            self._init_actor_hybrid()
        except AttributeError as e:
            raise AttributeError("actor worker init actor hybrid failed with missing attributes") from e
        except ValueError as e:
            raise ValueError("actor worker init actor hybrid failed with parameters conflict") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker init actor hybrid") from e
        logger.debug("actor worker init actor hybrid success")

        try:
            self.empty_cache()
        except RuntimeError as e:
            raise RuntimeError("actor worker failed to empty the cache") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker empty the cache") from e

    def _init_actor_hybrid(self):
        dp_size = self.parallel_state.get_data_parallel_world_size()
        if dp_size == 0:
            raise ValueError("actor worker get data parallel world size equal 0")

        """Create sharding manager and actor hybrid"""
        self.actor_hybrid = ActorRolloutHybrid(
            self.model,
            megatron_config=self.megatron_config,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            inference_model=self.inference_model,
            sharding_manager=self.sharding_manager,
            beta=self.rl_config.beta,
            mini_batch_size_per_dp=self.rl_config.mini_batch_size // dp_size,
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            clip_ratio=self.rl_config.clip_ratio,
            micro_batch_size=self.megatron_config.micro_batch_size,
            use_dynamic_bsz=self.rl_config.use_dynamic_bsz,
            max_packing_token_size=self.rl_config.max_packing_token_size,
            dynamic_max_batch_size=self.rl_config.dynamic_max_batch_size,
            use_remove_padding=self.rl_config.use_remove_padding,
            set_actual_seq_len=self.set_actual_seq_len,
            get_actual_seq_len=self.get_actual_seq_len,
            set_position_ids=self.set_position_ids,
            context_parallel_size=self.megatron_config.context_parallel_size,
            entropy_coeff=self.rl_config.entropy_coeff,
            kl_penalty=self.rl_config.kl_penalty,
            temperature=self.generate_config.sampling_config["temperature"],
            token_level_loss=self.rl_config.token_level_loss,
            clip_higher_enable=self.rl_config.clip_higher_enable,
            clip_ratio_low=self.rl_config.clip_ratio_low,
            clip_ratio_high=self.rl_config.clip_ratio_high
        )

    def get_worker_info(self):
        """
        Get worker's node info.

        Returns:
            tuple: the RANK of current worker and node id.
        """
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is None:
            return None, None

        try:
            local_rank = int(local_rank_env)
            if local_rank < 0 or local_rank >= 8:
                raise ValueError("LOCAL_RANK must be in [0, 8)")
        except ValueError as e:
            raise ValueError(f"Invalid LOCAL_RANK value: {e}") from e

        return local_rank, ray.get_runtime_context().get_node_id()

    def enter_infer_mode(self):
        if self.state == ActorState.INFER:
            return

        start_time = time.time()
        try:
            self.sharding_manager.enter_infer_mode()
        except RuntimeError as e:
            raise RuntimeError("actor worker enter infer mode failed") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker enter infer mode") from e

        self.state = ActorState.INFER
        end_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/resharding_to_infer",
                value=[end_time - start_time],
                cumulate=True
            )
        )
        logger.debug("actor worker enter infer mode")

        torch.cuda.empty_cache()

    def exit_infer_mode(self):
        if self.state != ActorState.INFER:
            raise RuntimeError("current state is not INFER, it is no available to exit infer mode")

        start_time = time.time()
        try:
            self.sharding_manager.exit_infer_mode()
        except RuntimeError as e:
            raise RuntimeError("actor worker exit infer mode failed") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker exit infer mode") from e

        self.state = ActorState.NONE
        end_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/resharding_exit_infer",
                value=[end_time - start_time],
                cumulate=True
            )
        )
        logger.debug("actor worker exit infer mode")

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Init inference model."""
        self.inference_model.init_worker(all_kwargs)
        logger.debug("actor worker init inference model worker success")

    def load_model(self, *args, **kwargs):
        """Inference_model load model weights"""
        self.inference_model.load_model(*args, **kwargs)
        logger.debug("actor worker load inference model success")

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.inference_model.is_sleep:
            return
        self.exit_infer_mode()
        self.inference_model.sleep(*args, **kwargs)
        self.inference_model.is_sleep = True
        self.continue_infer_running = True
        logger.debug("actor inference model sleep success")

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.inference_model.is_sleep:
            return

        if self.continue_infer_running:
            try:
                self.sharding_manager.enter_forward_mode()
            except RuntimeError as e:
                raise RuntimeError("actor worker enter forward mode failed") from e
            except Exception as e:
                raise Exception("Unexpected error occurred when actor worker enter forward mode") from e

        self.enter_infer_mode()

        self.inference_model.wake_up(*args, **kwargs)
        self.inference_model.is_sleep = False
        logger.debug("actor inference model wake up success")

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """The proxy method for execution with ray"""
        # torch.compile has been mocked by mindspeed-rl and megatron
        with replace_torch_compile():
            if method == "init_worker":
                return self.init_worker(*args, **kwargs)
            elif method == "load_model":
                return self.load_model(*args, **kwargs)
            elif method == "sleep":
                return self.sleep(*args, **kwargs)
            elif method == "wake_up":
                return self.wake_up(*args, **kwargs)
            else:
                result = self.inference_model.execute_method(method, *args, **kwargs)
                return result

    @mstx_timer_decorator
    def update(self, kl_ctrl=None, skip_actor_log_prob: bool = False):
        """
        Use data to update actor model

        Vars:
            kl_ctrl: controller of KL divergence.
            skip_actor_log_prob (bool): if skip process of old_log_prob.
        """
        if kl_ctrl is None:
            raise ValueError("kl_ctrl must not be none for actor worker to perform update")

        start_sharding_enter_train = time.time()
        try:
            self.sharding_manager.enter_train_mode()
        except RuntimeError as e:
            raise RuntimeError("actor worker enter train mode failed") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker enter train mode") from e
        sharding_train_interval = time.time() - start_sharding_enter_train

        self.args.curr_iteration = self.iteration

        experience_consumer_stage = 'actor_train'
        experience_columns = [
            'responses', 'advantages', 'old_log_prob', 'input_ids', 'response_length', 'prompt_length', 'response_mask']
        if self.megatron_config.stage != "ray_dapo":
            experience_columns.append('ref_log_prob')

        if self.rl_config.actor_update_dispatch_size is None:
            raise ValueError("actor worker rl_config.actor_update_dispatch_size is none")
        experience_count = self.rl_config.actor_update_dispatch_size

        if skip_actor_log_prob:
            experience_columns.remove('old_log_prob')

        learning_rate = None
        temp_learning_rates = []
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr'] if 'lr' in param_group else learning_rate
            if learning_rate is not None and learning_rate not in temp_learning_rates:
                temp_learning_rates.append(learning_rate)
        if len(temp_learning_rates) > 1:
            logger.warning(f"Multiple learning rates found, using the last one in {temp_learning_rates}.")
        ray.get(self.td.update_metrics.remote(key='actor/lr', value=learning_rate))

        sorted_indexes = None
        try:
            if self.rl_config.guarantee_order:
                sorted_indexes = self.get_dp_range_indexes(experience_count, use_vllm=False)
        except AttributeError as e:
            raise AttributeError("actor worker get data parallel range indexes failed with missing attributes") from e
        except RuntimeError as e:
            raise RuntimeError("actor worker get data parallel range indexes failed with runtime error") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker get data parallel range indexes") from e

        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            self._do_update(kl_ctrl, experience_consumer_stage, experience_columns, experience_count, sorted_indexes)

        self.iteration += 1
        self.prof_iteration += 1

        start_sharding_exit_train = time.time()
        try:
            self.sharding_manager.exit_train_mode()
        except RuntimeError as e:
            raise RuntimeError("actor worker exit train mode failed") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker exit train mode") from e

        sharding_train_interval += (time.time() - start_sharding_exit_train)
        ray.get(
            self.td.update_metrics.remote('timing/resharding_to_train', value=[sharding_train_interval], cumulate=True)
        )

        self.continue_infer_running = False
        logger.debug("finish actor update")

    def _do_update(self, kl_ctrl, experience_consumer_stage, experience_columns, experience_count, sorted_indexes):
        try:
            batch_data, index = self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                self.megatron_config.tensor_model_parallel_size,
                self.megatron_config.context_parallel_size,
                self.megatron_config.context_parallel_algo,
                indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None,
                get_n_samples=self.enable_partial_rollout)
        except KeyError as e:
            raise KeyError("actor worker dispatch transfer dock data failed with key mismatching") from e
        except ValueError as e:
            raise ValueError("actor worker dispatch transfer dock data failed with parameters conflict") from e
        except RuntimeError as e:
            raise RuntimeError("actor worker dispatch transfer dock data failed with runtime error") from e
        except Exception as e:
            raise Exception("Unexpected error occurred when actor worker dispatch transfer dock data") from e

        if not self.start_time_defined:
            self.update_start_time = time.time()
            self.start_time_defined = True

        if batch_data and index:
            try:
                metrics = self.actor_hybrid.update_actor(batch_data, kl_ctrl)
            except ValueError as e:
                raise ValueError("actor worker update actor failed with parameters conflict") from e
            except Exception as e:
                raise Exception("Unexpected error occurred when actor worker update actor") from e

            if self.rl_config.n_samples_per_prompt == 0:
                raise ValueError("actor worker rl_config.n_samples_per_prompt should not be zero")
            self.args.consumed_train_samples += (
                    self.megatron_config.global_batch_size // self.rl_config.n_samples_per_prompt)
            self.num_floating_point_operations_so_far += (
                num_floating_point_operations(self.args, self.megatron_config.global_batch_size))

            if (self.parallel_state.is_pipeline_last_stage(ignore_virtual=True) and
                    self.parallel_state.get_tensor_model_parallel_rank() == 0 and
                    self.parallel_state.get_context_parallel_rank() == 0):
                ray.get(self.td.update_metrics.remote(value=metrics, cumulate=True))
                ray.get(
                    self.td.update_metrics.remote(
                        "timing/update",
                        value=[round(time.time(), 4), round(self.update_start_time, 4)],
                        cumulate=True
                    )
                )

    def update_actor_logprob_dispatch_size(self, new_actor_logprob_dispatch_size: int):
        """Update actor_logprob_dispatch_size, experience count every forward step for actor_logprob"""
        self.rl_config.actor_logprob_dispatch_size = (new_actor_logprob_dispatch_size //
                                                      self.parallel_state.get_data_parallel_world_size())

    def update_actor_update_dispatch_size(self, new_actor_update_dispatch_size: int):
        """Update actor_update_dispatch_size, experience count every forward step for actor update"""
        self.rl_config.actor_update_dispatch_size = (new_actor_update_dispatch_size //
                                                     self.parallel_state.get_data_parallel_world_size())

    def update_mini_batch_size(self, original_n_samples_per_prompt: int, new_samples_per_prompt: int,
                               use_stepwise_advantage: bool):
        """Update mini_batch_size, mini batch size"""
        self.actor_hybrid.update_mini_batch_size(original_n_samples_per_prompt, new_samples_per_prompt,
                                                 use_stepwise_advantage)
