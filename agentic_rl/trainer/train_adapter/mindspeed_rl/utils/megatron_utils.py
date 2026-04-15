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
from datetime import timedelta
from typing import Any, Callable, Dict, Optional
import sys

import ray
import torch

from mindspeed_rl.utils import seed_all
from mindspeed_rl.utils.utils import parse_args_from_config, init_torch_compile
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig, MsprobeConfig

from agentic_rl.trainer.train_adapter.mindspeed_rl.config_cls.extend_generate import ExtendedGenerateConfig
from agentic_rl.trainer.train_adapter.mindspeed_rl.config_cls.agentic_env import AgenticEnvConfig
from agentic_rl.trainer.train_adapter.mindspeed_rl.config_cls.extend_rl_config import ExtendedRLConfig
from agentic_rl.trainer.train_adapter.mindspeed_rl.config_cls.extend_megatron_config import ExtendMegatronConfig
from agentic_rl.trainer.train_adapter.mindspeed_rl.config_cls.validate_config import validate_agent_rl_args

from agentic_rl.base.log.loggers import Loggers

logger = Loggers("train_hybrid").get_logger()


def parse_training_config(config: Dict) -> Dict[str, Any]:
    """
    Parse the training configuration and extract sub-configs for each component.

    Args:
        config: Global configuration dictionary containing keys such as
            ``megatron_training``, ``actor_config``, ``rl_config``, etc.

    Returns:
        A dict with keys: actor_config, ref_config, reward_config,
        rl_config, generate_config, profiler_config, msprobe_config,
        agentic_env_config.

    Raises:
        ValueError: If conflicting config keys are set for integrated worker mode.
    """
    logger.info("=" * 10)
    logger.info(config)
    logger.info("=" * 10)
    actor_config = ExtendMegatronConfig(
        {**config.get("megatron_training"), **config.get("actor_config")}, config.get('model'))
    rl_config = ExtendedRLConfig(config.get("rl_config"))

    if rl_config.use_integrated_worker:
        if "ref_config" in config:
            raise ValueError(
                "ref_config should not be set when use_integrated_worker mode is on.")
        ref_config = actor_config

        if "reward_config" in config:
            raise ValueError(
                "reward_config should not be set when use_integrated_worker mode is on.")
        reward_config = actor_config

    else:
        ref_config = ExtendMegatronConfig(
            {**config.get("megatron_training"), **config.get("ref_config")}, config.get('model'))

        reward_config = ExtendMegatronConfig(
            {**config.get("megatron_training"), **config.get("reward_config")}, config.get('model'))

    generate_config = ExtendedGenerateConfig(config.get("generate_config"))
    agentic_config = AgenticEnvConfig(config.get("agentic_env_config"))
    validate_agent_rl_args(actor_config, ref_config, reward_config,
                           rl_config, generate_config, agentic_config)

    profiler_config = {}
    profiler_config.update({
        "integrated": ProfilerConfig(
            config.get("profiler_config", {}).get("integrated", {}),
            role="integrated"
        ),
    })
    actor_config.max_prompt_length = rl_config.max_prompt_length

    msprobe_config = MsprobeConfig(
        config.get("msprobe_config", {}),
        role="integrated"
    )

    return {
        "actor_config": actor_config,
        "ref_config": ref_config,
        "reward_config": reward_config,
        "rl_config": rl_config,
        "generate_config": generate_config,
        "profiler_config": profiler_config,
        "msprobe_config": msprobe_config,
        "agentic_env_config": agentic_config
    }


def get_megatron_module() -> Dict[str, Any]:
    """
    Lazily import and return a dict of core Megatron modules and utilities.

    Returns:
        A mapping from short names to Megatron/MindSpeed objects such as
        ``parallel_state``, ``get_model``, ``load_checkpoint``, etc.
    """
    from megatron.core import parallel_state
    from megatron.core import DistributedDataParallel
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.training.checkpointing import load_checkpoint, save_checkpoint
    from megatron.training.training import get_optimizer_param_scheduler
    from megatron.training import get_args
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core import DistributedDataParallel as LocalDDP
    from megatron.legacy.model import Float16Module
    from megatron.training.training import get_model, unwrap_model
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
    from megatron.training.training import setup_model_and_optimizer
    from megatron.core.enums import ModelType
    from megatron.core.distributed import finalize_model_grads
    from mindspeed.utils import set_actual_seq_len, set_position_ids, get_actual_seq_len
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    return {
        'parallel_state': parallel_state,
        'get_model': get_model,
        'get_megatron_optimizer': get_megatron_optimizer,
        'get_optimizer_param_scheduler': get_optimizer_param_scheduler,
        'load_checkpoint': load_checkpoint,
        'save_checkpoint': save_checkpoint,
        'get_args': get_args,
        'get_forward_backward_func': get_forward_backward_func,
        'float16_module': Float16Module,
        'unwrap_model': unwrap_model,
        'local_ddp': LocalDDP,
        'distributed_data_parallel_config': DistributedDataParallelConfig,
        'vocab_parallel_cross_entropy': vocab_parallel_cross_entropy,
        'setup_model_and_optimizer': setup_model_and_optimizer,
        'model_type': ModelType,
        'distributed_data_parallel': DistributedDataParallel,
        'finalize_model_grads': finalize_model_grads,
        'set_actual_seq_len': set_actual_seq_len,
        'get_actual_seq_len': get_actual_seq_len,
        'set_position_ids': set_position_ids,
        'distributed_optimizer': DistributedOptimizer,
        'float16_optimizer_with_float16_params': Float16OptimizerWithFloat16Params
    }


def gpt_model_provider(pre_process: bool, post_process: bool):
    """
    Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    from megatron.training import get_args
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training.arguments import core_transformer_config_from_args
    args = get_args()

    logger.info('building GPT model ...')
    # Experimental loading arguments from configs
    config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm,
                                                          qk_layernorm=args.qk_layernorm)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
    )

    return model


def rm_model_provider(pre_process: bool, post_process: bool):
    """
    Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.

    Returns:
        GPTRewardModel: The returned model
    """
    from megatron.training import get_args
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training.arguments import core_transformer_config_from_args
    from mindspeed_llm.tasks.posttrain.orm.orm_model import GPTRewardModel
    args = get_args()
    logger.info('building RM GPT model ...')
    # Experimental loading arguments from configs
    config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm,
                                                          qk_layernorm=args.qk_layernorm)

    if (not args.untie_embeddings_and_output_weights) and (args.pipeline_model_parallel_size > 1):
        args.untie_embeddings_and_output_weights = True
        logger.warning(
            "untie_embeddings_and_output_weights is set to True, "
            "since output_layer is not used in Outcome Reward model training."
        )
    model = GPTRewardModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        post_layer_norm=not args.no_post_layer_norm,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )

    return model


def initialize_megatron(
        extra_args_provider: Optional[Callable] = None,
        args_defaults: Optional[Dict] = None,
        ignore_unknown_args: bool = False,
        allow_no_cuda: bool = False,
        skip_mpu_initialization: bool = False,
        config: Optional[Dict] = None,
) -> Optional[Callable]:
    """
    Set global variables, initialize distributed, and set autoresume and random seeds.

    ``allow_no_cuda`` should not be set unless using Megatron for CPU-only
    data processing.

    Args:
        extra_args_provider: Optional callable that adds extra CLI arguments.
        args_defaults: Default argument overrides. Defaults to an empty dict.
        ignore_unknown_args: Whether to ignore unknown CLI arguments.
        allow_no_cuda: Skip CUDA availability check when True.
        skip_mpu_initialization: Skip model-parallel initialization.
        config: Configuration dict forwarded to ``parse_args_from_config``.

    Returns:
        A ``finish_mpu_init`` callable when ``lazy_mpu_init`` is True, else None.
    """
    if args_defaults is None:
        args_defaults = {}

    origin_sys_argv = sys.argv
    sys.argv = [sys.argv[0]]
    parse_args_from_config(config)

    # Initialize torch.compile global variables to avoid training-related patches affecting vLLM graph mode enabling.
    init_torch_compile(torch.compile)
    # Note: Importing this line activates the megatron_adapter.
    from mindspeed_llm.training.arguments import parse_args_decorator
    import megatron

    parse_args = parse_args_decorator(megatron.training.arguments.parse_args)
    args = parse_args(extra_args_provider, ignore_unknown_args)
    sys.argv = origin_sys_argv

    if not allow_no_cuda:
        if not torch.cuda.is_available():
            raise ValueError("Megatron requires CUDA.")

    from megatron.core import parallel_state
    from megatron.training import get_args
    from megatron.training.arguments import validate_args
    from megatron.training.checkpointing import load_args_from_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import _set_random_seed, \
        _init_autoresume, _compile_dependencies, \
        _initialize_tp_communicators

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        if args.load is None:
            raise ValueError("--use-checkpoints-args requires --load argument.")
        load_args_from_checkpoint(args)

    # TODO: improve argument validation error handling
    validate_args(args, args_defaults)

    set_global_variables(args)

    from mindspeed.core.tensor_parallel.lcal_coc.user_config import initialize_coc_from_cfg
    initialize_coc_from_cfg(args)

    if args.use_deter_comp:
        seed_all(args.seed)
        logger.info("deterministic computing is applied for npu.")

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            logger.info(f"> setting random seeds to {args.seed} ...")
        _set_random_seed(args.seed, args.data_parallel_random_init)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        parallel_state.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        parallel_state.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            _initialize_tp_communicators()

        # No continuation function
        return None


def _initialize_distributed() -> None:
    """Initialize torch.distributed and core model parallel."""
    from megatron.core import parallel_state
    from megatron.training import get_args
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            logger.info("torch distributed is already initialized, skipping initialization...")
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            logger.info("> initializing torch distributed...")
        # Manually set the device ids.
        if device_count > 0:
            if args.stage in ["ray_ppo", "ray_online_dpo", "ray_grpo", "ray_dapo"]:
                allocated_device = int(ray.get_runtime_context().get_accelerator_ids()["NPU"][0])
                torch.cuda.set_device(allocated_device)
            else:
                device = args.rank % device_count
                if args.local_rank is not None:
                    if args.local_rank != device:
                        raise ValueError("expected local-rank to be the same as rank % device-count.")
                else:
                    args.local_rank = device
                torch.cuda.set_device(device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            logger.info("model parallel is already initialized")
        else:
            parallel_state.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank,
                context_parallel_size=args.context_parallel_size,
                expert_model_parallel_size=args.expert_model_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
                nccl_communicator_config_path=args.nccl_communicator_config_path,
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',
            )
            if args.rank == 0:
                logger.info(
                    f"> initialized tensor model parallel with size "
                    f"{parallel_state.get_tensor_model_parallel_world_size()}"
                )
                logger.info(
                    f"> initialized pipeline model parallel with size "
                    f"{parallel_state.get_pipeline_model_parallel_world_size()}"
                )