# -*- coding: utf-8 -*-
#
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
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
from time import time
from pathlib import Path
from functools import cached_property

import ray
import torch
import torch.distributed as dist
import transformers
from transformers.configuration_utils import PretrainedConfig

from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.single_controller.base.decorator import Dispatch, register

from aura.base.log.loggers import Loggers


# Supported HuggingFace architecture suffixes for causal generation models
SUPPORTED_HF_ARCHITECTURES: tuple[str, ...] = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "NemotronH_Nano_VL_V2",
)

# Preformatted display string for error/help messages
SUPPORTED_HF_ARCHITECTURES_DISPLAY = " or ".join(f"'{s}'" for s in SUPPORTED_HF_ARCHITECTURES)

logger = Loggers(__name__).get_logger()


class MegatronDetachActorWorker(DetachActorWorker):
    """Megatron-based actor worker that supports detached weight saving and synchronization."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def prepare_infer_params_to_cpu(self, weight_save_dir: str) -> None:
        """Save model weights to CPU and notify the weight updater.

        Args:
            weight_save_dir: Directory path where weights will be saved.
            sync_group_name: Name of the synchronization group.
        """
        logger.info(f"start saving weights: rank={self.rank}, path={weight_save_dir}")
        os.makedirs(weight_save_dir, exist_ok=True)

        start_time = time()
        self._save_weights(
            model=self.actor.engine.module, path=weight_save_dir, show_progress=True, distributed_save=True
        )
        logger.info(f"weight saving completed: rank={self.rank}, path={weight_save_dir}, time={time() - start_time}")

        w_actor = ray.get_actor("weight_updater", namespace="controller_raygroup")
        w_actor.weight_saved.remote(weight_save_dir)

    def _save_weights(
        self,
        model,
        path: str | Path,
        show_progress: bool = True,
        strict: bool = True,
        distributed_save: bool = False,
        save_every_n_ranks: int = 1,
    ):
        from verl.utils.megatron_utils import (
            get_megatron_module_device,
            offload_megatron_model_to_cpu,
            load_megatron_model_to_gpu,
        )

        engine = self.actor.engine
        origin_module_device = get_megatron_module_device(engine.module)
        if engine.engine_config.param_offload or origin_module_device == "cpu":
            logger.info("=====weight is on cpu and needs to be loaded to npu=====")
            load_megatron_model_to_gpu(engine.module, load_grad=True)
        logger.info("=====start to save hf weights=====")
        self._save_hf_weights(model, path, show_progress, strict, distributed_save, save_every_n_ranks)
        torch.distributed.barrier()
        if engine.engine_config.param_offload:
            offload_megatron_model_to_cpu(engine.module)

    @staticmethod
    def _get_model_instance(model):
        model_instance = model[0]
        while hasattr(model_instance, "module"):
            model_instance = model_instance.module
        return model_instance

    @cached_property
    def _causal_lm_architecture(self):
        from megatron.bridge.models.conversion.utils import get_causal_lm_class_name_via_auto_map
        from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

        if isinstance(self.hf_pretrained, PreTrainedCausalLM):
            config = self.hf_pretrained.config
        else:
            config = self.hf_pretrained

        architectures = getattr(config, "architectures", [])

        if not architectures:
            raise ValueError(
                "\n✗ No architectures found in model config\n\n"
                "The model configuration does not specify any architectures.\n"
                "This is required for determining the model type."
            )

        causal_lm_arch = None
        for architecture_name in architectures:
            if architecture_name.endswith(SUPPORTED_HF_ARCHITECTURES):
                causal_lm_arch = architecture_name
                break

        if not causal_lm_arch:
            raise ValueError(
                f"\n✗ No CausalLM architecture found\n\n"
                f"Model architectures: {architectures}\n\n"
                f"None of the architectures end with {SUPPORTED_HF_ARCHITECTURES_DISPLAY}.\n"
                f"This bridge only supports causal language models.\n"
                f"For other model types, use a different bridge class."
            )

        # Try auto_map first (returns class name string if available)
        cls_name = get_causal_lm_class_name_via_auto_map(config=config)
        if cls_name is not None:
            # For auto_map models, return the class name as a string
            return cls_name

        try:
            return getattr(transformers, causal_lm_arch)
        except AttributeError:
            raise ValueError(
                f"\n✗ Architecture class '{causal_lm_arch}' not found in transformers\n\n"
                f"This could mean:\n"
                f"1. The model requires a newer version of transformers\n"
                f"2. The model uses a custom modeling file not in the standard library\n"
                f"3. There's a typo in the architecture name\n\n"
                f"Please verify your transformers installation and the model requirements."
            )

    def _save_hf_weights(
        self,
        model,
        path: str | Path,
        show_progress: bool = True,
        strict: bool = True,
        distributed_save: bool = False,
        save_every_n_ranks: int = 1,
    ) -> None:
        """
        Save Megatron model weights in HuggingFace safetensors format.

        This method exports only the model weights (not configuration or tokenizer)
        to safetensors files compatible with HuggingFace. It uses streaming save
        to handle large models efficiently without requiring all weights in memory
        at once.

        The weights are gathered from distributed ranks and saved in the standard
        HuggingFace sharded format when the model is large.

        Args:
            model: Megatron model instance or list of instances
            path: Directory path where weight files will be saved
            show_progress: Display progress bar during export
            distributed_save: Whether to enable distributed saving mode where each rank saves
                part of weights independently.
            save_every_n_ranks: Interval for saving weights across ranks in distributed mode.
                For example, if set to 2, only ranks 0, 2, 4, ... will save weights.

        Raises:
            ValueError: If the state source doesn't support streaming save

        Note:
            - This method is collective and must be called by all ranks
            - Uses safetensors format for efficient loading and security
            - Automatically handles model sharding for large models
            - The saved weights can be loaded with HuggingFace's from_pretrained
        """
        from megatron.bridge import AutoBridge
        from megatron.bridge.models.conversion import model_bridge
        from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
        from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

        bridge: AutoBridge = self.actor.engine.bridge
        self.hf_pretrained: PreTrainedCausalLM | PretrainedConfig = bridge.hf_pretrained

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        dispatch_instance = (self._causal_lm_architecture, self._get_model_instance(model))
        # peft不支持，merge_adapter_weights设置为False
        generator = model_bridge.stream_weights_megatron_to_hf(
            dispatch_instance,
            model,
            self.hf_pretrained,
            cpu=True,
            show_progress=show_progress,
            merge_adapter_weights=False,
        )
        # 判断量化层的逻辑暂时删除
        # Check if the state source is SafeTensorsStateSource for streaming save.
        if (
            hasattr(self.hf_pretrained, "state")
            and hasattr(self.hf_pretrained.state, "source")
            and isinstance(self.hf_pretrained.state.source, SafeTensorsStateSource)
        ):
            self.hf_pretrained.state.source.save_generator(
                generator,
                path,
                strict=strict,
                distributed_save=distributed_save,
                save_every_n_ranks=save_every_n_ranks,
            )
        else:
            raise ValueError("The state source is not a SafeTensorsStateSource, cannot save in streaming mode.")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
