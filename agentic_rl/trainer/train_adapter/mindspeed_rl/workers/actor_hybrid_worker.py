# -*- coding: utf-8 -*-
#
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# 
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import megatron.core.parallel_state as ps
import ray
import safetensors.torch as sf
import torch
from transformers import AutoConfig

from mindspeed_rl.models.actor_rollout_hybrid import ActorRolloutHybrid
from mindspeed_rl.utils.utils import (
    MsProbe,
    mstx_timer_decorator,
    profiler_start,
    profiler_step,
    replace_torch_compile,
)
from mindspeed_rl.workers.actor_hybrid_worker import (
    ActorHybridWorkerBase,
    ActorState,
    is_multimodal,
    num_floating_point_operations,
)
from mindspeed_rl.workers.resharding.megatron_off_loader import MegatronOffLoader

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.infer_adapter.vllm.extension.custom_worker_extensions import (
    resolve_device,
    split_tensors_and_meta,
)

logger = Loggers(__name__).get_logger()


def _do_tensors_save(
    save_dir: str, file_path: str, params: Dict[str, Any], meta_header: Optional[Dict[str, str]]
) -> None:
    """Save tensor parameters to a safetensors file and notify the weight updater actor.

    Args:
        save_dir: Directory where the tensors are saved.
        file_path: Full path of the target safetensors file.
        params: Dictionary mapping parameter names to tensor values.
        meta_header: Optional metadata header to embed in the safetensors file.
    """
    try:
        
        logger.info(f"===save tensors to {file_path}")
        if meta_header:
            sf.save_file(params, file_path, metadata=meta_header)
        else:
            sf.save_file(params, file_path)

        logger.info(f"===save tensors to {file_path} succeed")
        w_actor = ray.get_actor("weight_updater", namespace="controller_raygroup")
        w_actor.weight_saved.remote(save_dir)
        logger.info("===weights save success")
    except Exception:
        logger.exception(f"Failed to save tensors to {file_path}")
        raise


def async_tensors_save(
    save_dir: str, file_path: str, params: Dict[str, Any], meta_header: Optional[Dict[str, str]] = None
) -> None:
    """Spawn a daemon thread to save tensor parameters asynchronously.

    Args:
        save_dir: Directory where the tensors are saved.
        file_path: Full path of the target safetensors file.
        params: Dictionary mapping parameter names to tensor values.
        meta_header: Optional metadata header to embed in the safetensors file.
    """
    threading.Thread(
        target=_do_tensors_save, args=(save_dir, file_path, params, meta_header), daemon=True
    ).start()
    logger.info("===async saving tensor with threading")


class AgentActorHybridWorkerBase(ActorHybridWorkerBase):
    """Actor-inference hybrid worker base that manages training, resharding, and vLLM inference."""

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        super().__init__(*args, **kwargs)
        self.rl_config.zmq_communication = False
        self.continue_infer_running = False

    def initialize(self) -> None:
        """Set up distributed rank, build model/optimizer, offloader, and inference engine."""
        self.setup_distributed_rank()
        self.model, self.optimizer, self.opt_param_scheduler = self._build_model_optimizer()
        self._set_no_sync_func()
        self.actor_offloader = MegatronOffLoader(
            self.model,
            self.optimizer,
            megatron_config=self.megatron_config,
            distributed_optimizer=self.distributed_optimizer,
            float16_optimizer_with_float16_params=self.float16_optimizer_with_float16_params)

        if self.generate_config.offload_train_optimizer:
            self.actor_offloader.offload_optimizer()
        if self.generate_config.offload_train_grad:
            self.actor_offloader.offload_grad()
        if self.generate_config.offload_train_param:
            self.actor_offloader.offload_param()
        with replace_torch_compile():
            self.inference_model = self._build_rollout()

        self.actor_profiler = profiler_start(self.profiler_config, self.profiler_config.role)
        MsProbe.config_init(self.msprobe_config)

    def get_worker_info(self) -> Tuple[Optional[str], str]:
        """Return the current RANK environment variable and the Ray node ID."""
        return os.getenv('RANK'), ray.get_runtime_context().get_node_id()

    def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
        """Forward worker initialisation kwargs to the inference engine."""
        self.inference_model.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs) -> None:
        """Delegate model loading to the inference engine."""
        self.inference_model.load_model(*args, **kwargs)

    def enter_infer_mode(self) -> None:
        """Transition the worker into inference mode and record resharding time."""
        if self.state == ActorState.INFER:
            return

        start_time = time.time()
        self.sharding_manager.enter_infer_mode()
        self.state = ActorState.INFER
        end_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/resharding_to_infer",
                value=[end_time - start_time],
                cumulate=True
            )
        )

    def init_sharding_manager(self) -> None:
        """Build the sharding manager and the actor-rollout hybrid after sleeping the inference engine."""
        logger.info(f"===init_sharding_manager, inference_model: {self.inference_model}")
        self.inference_model.sleep()
        self.sharding_manager = self._build_sharding_manager()
        self.sharding_manager.enable_sleep_mode = self.generate_config.enable_sleep_mode

        if self.generate_config.offload_train_param:
            self.actor_offloader.onload_param()

        self.actor_hybrid = ActorRolloutHybrid(
            self.model,
            megatron_config=self.megatron_config,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            inference_model=self.inference_model,
            sharding_manager=self.sharding_manager,
            beta=self.rl_config.beta,
            mini_batch_size_per_dp=self.rl_config.mini_batch_size
                                   // self.parallel_state.get_data_parallel_world_size(),
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
        self.empty_cache()

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.inference_model.is_sleep:
            return
        self.inference_model.sleep(*args, **kwargs)
        self.inference_model.is_sleep = True
        self.exit_infer_mode()
        self.continue_infer_running = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.inference_model.is_sleep:
            return
        if self.continue_infer_running:
            self.sharding_manager.enter_forward_mode()
        self.enter_infer_mode()  # pylint: disable=C2801
        self.inference_model.wake_up(*args, **kwargs)
        self.inference_model.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs) -> Any:
        """Dispatch a named method call, wrapping execution with a torch.compile replacement.

        Args:
            method: Name of the method to invoke.

        Returns:
            The return value of the dispatched method.
        """
        with replace_torch_compile():
            dispatch = {
                "init_worker": self.init_worker,
                "load_model": self.load_model,
                "sleep": self.sleep,
                "wake_up": self.wake_up,
            }
            handler = dispatch.get(method)
            if handler is not None:
                return handler(*args, **kwargs)
            return self.inference_model.execute_method(method, *args, **kwargs)

    def _build_rollout(self) -> Any:
        """Construct and return the asynchronous vLLM inference engine."""
        self.actor_model_config = AutoConfig.from_pretrained(
            self.megatron_config.tokenizer_name_or_path, trust_remote_code=self.generate_config.trust_remote_code)

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

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
            enable_sleep_mode=self.generate_config.enable_sleep_mode,
        )

        return rollout

    @mstx_timer_decorator
    def update(self, kl_ctrl: Any = None, skip_actor_log_prob: bool = False) -> None:
        """Run one actor-update step: enter train mode, consume experience, and update metrics.

        Args:
            kl_ctrl: Optional KL divergence controller.
            skip_actor_log_prob: If True, omit old_log_prob from the experience columns.
        """
        start_sharding_enter_train = time.time()
        self.sharding_manager.enter_train_mode()
        sharding_train_interval = time.time() - start_sharding_enter_train

        self.args.curr_iteration = self.iteration

        experience_consumer_stage = 'actor_train'

        if self.megatron_config.stage == "ray_dapo":
            experience_columns = ['responses', 'advantages', 'old_log_prob',
                                  'input_ids', 'response_length', 'prompt_length']
        else:
            experience_columns = ['responses', 'advantages', 'old_log_prob',
                                  'ref_log_prob', 'input_ids', 'response_length', 'prompt_length']

        experience_columns.append("response_mask")

        if is_multimodal():
            experience_columns.extend(['attention_mask', 'position_ids'])

        experience_count = self.rl_config.actor_update_dispatch_size

        if skip_actor_log_prob:
            experience_columns.remove('old_log_prob')

        learning_rate = None
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
        ray.get(self.td.update_metrics.remote(key='actor/lr', value=learning_rate))
        sorted_indexes = self.get_dp_range_indexes(
            experience_count,
            use_vllm=False
        ) if self.rl_config.guarantee_order else None

        actor_update_profiler = profiler_start(
            self.profiler_config,
            role="actor_update",
            profiler_iteration=self.prof_iteration
        )

        MsProbe.debugger_start(self.model[0], tag='actor_update')

        start_time_defined = False
        first_dispatch_data_defined = False
        first_dispatch_start_time = time.time()
        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            if not first_dispatch_data_defined:
                first_dispatch_start_time = time.time()
            batch_data, index = self.dispatch_transfer_dock_data(experience_consumer_stage,
                                                                 experience_columns,
                                                                 experience_count,
                                                                 self.megatron_config.tensor_model_parallel_size,
                                                                 self.megatron_config.context_parallel_size,
                                                                 self.megatron_config.context_parallel_algo,
                                                                 indexes=sorted_indexes.pop(
                                                                     0) if self.rl_config.guarantee_order else None,
                                                                 get_n_samples=self.enable_partial_rollout)
            if batch_data and index:
                if not first_dispatch_data_defined:
                    ray.get(self.td.update_metrics.remote(
                        "dispatch_timing(first)/update",
                        value=[time.time(), first_dispatch_start_time],
                        cumulate=True
                    ))
                    first_dispatch_data_defined = True

                if not start_time_defined:
                    start_time = time.time()
                    start_time_defined = True
                metrics = self.actor_hybrid.update_actor(batch_data, kl_ctrl)

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
                            value=[round(time.time(), 4), round(start_time, 4)],
                            cumulate=True
                        )
                    )

        self.iteration += 1
        profiler_step(actor_update_profiler)
        MsProbe.debugger_stop(tag='actor_update')
        MsProbe.step()
        self.prof_iteration += 1
        start_sharding_exit_train = time.time()
        self.sharding_manager.exit_train_mode()
        sharding_train_interval += (time.time() - start_sharding_exit_train)
        ray.get(
            self.td.update_metrics.remote(
                "timing/resharding_to_train",
                value=[sharding_train_interval],
                cumulate=True
            )
        )
        profiler_step(self.actor_profiler)
        self.continue_infer_running = False
        logger.info("finish actor update")

    def get_meta_and_param_from_dev(
        self, dev: str, to_cpu: bool
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
        """Materialise inference parameters on the given device and split into tensors and metadata.

        Args:
            dev: Target device identifier (e.g. ``"cpu"`` or ``"npu"``).
            to_cpu: Whether to move the resulting tensors to CPU.

        Returns:
            A tuple of (tensor_params, meta_header).
        """
        self.onload_infer_params_with_device(dev)
        params = self.sharding_manager.vllm_weight_container.get_infer_params()
        params = {k: (v.to(torch.float16) if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        if to_cpu:
            params = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in params.items()}

        tensor_params, meta_header = split_tensors_and_meta(params)
        return tensor_params, meta_header

    @mstx_timer_decorator
    def onload_infer_params_with_device(self, device: str = "cpu") -> None:
        """Rebuild weight buffers on the specified device.

        Args:
            device: Target device string (default ``"cpu"``).
        """
        dev = resolve_device(device)
        for buffer in self.sharding_manager.vllm_weight_container.weight_buffers:
            buffer.rebuild_with_device(dev)

    def get_file_name_and_dev(self, save_dir: str) -> Optional[Tuple[str, str]]:
        """Determine the safetensors file path and target device for the current rank.

        Args:
            save_dir: Base directory to save model shards.

        Returns:
            A tuple of (file_path, device) or ``None`` when the current data-parallel
            rank should not save.
        """
        dp_rank = ps.get_data_parallel_rank()
        pp_rank = ps.get_pipeline_model_parallel_rank()
        tp_rank = ps.get_tensor_model_parallel_rank()
        ep_rank = None
        if self.megatron_config.expert_model_parallel_size != 1:
            dev = "npu"
            ep_rank = ps.get_expert_model_parallel_rank()
        else:
            dev = "cpu"
            if dp_rank != 0:
                return None

        ep_name = f"_ep{ep_rank}" if ep_rank is not None else "_ep0"
        save_dir = os.path.realpath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"pp{pp_rank}_tp{tp_rank}{ep_name}.safetensors"
        file_path = os.path.join(save_dir, file_name)
        return file_path, dev

    @mstx_timer_decorator
    def prepare_infer_params_to_cpu(self, save_dir: str, to_cpu: bool = True):
        """Synchronously materialize weights on CPU and free actor buffers."""
        file_path, dev = self.get_file_name_and_dev(save_dir)

        tensor_params, meta_header = self.get_meta_and_param_from_dev(dev, to_cpu)
        self.sharding_manager.offload_infer_params()

        logger.info(f"saving {len(tensor_params)} tensors to {file_path}")
        async_tensors_save(save_dir, file_path, tensor_params, meta_header)
        return file_path


@ray.remote(resources={"NPU": 0.7})
class ActorHybridWorker(AgentActorHybridWorkerBase):
    """Ray remote actor-hybrid worker bound to an NPU resource."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
