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
import torch
import gc

from mindspeed_rl.utils.utils import mstx_timer_decorator

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


@mstx_timer_decorator
def enter_infer_mode_patch(self) -> None:
    """
    Before:
        Empty or with training param on NPU.

    After:
        Empty.

    Process:
        1. onload training param if needed
        2. onload inference param
        3. do resharding
        4. offload training param
    """
    logger.info("enter_infer_mode_patch ...")

    self.onload_infer_params()
    infer_params = self.vllm_weight_container.get_infer_params()
    torch.cuda.empty_cache()
    if self.train_param_offload:
        self.megatron_offloader.offload_param()
        torch.cuda.empty_cache()

    gc.collect()
    self.inference_engine.sync_model_weights(infer_params, load_format='megatron')

    if self.enable_sleep_mode:
        infer_params = None
        self.offload_infer_params()
        torch.cuda.empty_cache()


@mstx_timer_decorator
def exit_infer_mode_patch(self) -> None:
    """
    Before:
        With inference param on NPU.

    After:
        Empty.

    Process:
        1. offload inference param
    """
    logger.info("exit_infer_mode_patch ...")

    self.inference_engine.offload_model_weights()
    if not self.enable_sleep_mode:
        self.offload_infer_params()
    torch.cuda.empty_cache()
    gc.collect()


@mstx_timer_decorator
def exit_train_mode_patch(self) -> None:
    """
    Before:
        With training param, optimizer and grad on NPU.

    After:
        With training param on NPU.

    Process:
        1. offload training optimizer
        2. offload training grad
    """
    if self.optimizer_offload:
        self.megatron_offloader.offload_optimizer()
        torch.cuda.empty_cache()
    if self.grad_offload:
        self.megatron_offloader.offload_grad()
        torch.cuda.empty_cache()
    gc.collect()