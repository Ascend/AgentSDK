#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
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
# -------------------------------------------------------------------------

# Copyright 2024 Bytedance Ltd. and/or its affiliates

import torch
import torch_npu

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def logprobs_from_logits_torch_npu_patch(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log-probabilities using Ascend NPU's optimized cross-entropy.

    Uses torch_npu's native cross-entropy implementation for efficient
    computation on Huawei Ascend NPU devices.

    Args:
        logits: Model output logits of shape (..., vocab_size).
        labels: Target token indices of shape (...,).

    Returns:
        torch.Tensor: Log-probabilities for target labels, same shape as labels.
    """
    logger.debug("logprobs_from_logits_torch_npu_patch")
    batch_dim = logits.shape[:-1]
    logits = logits.reshape(-1, logits.shape[-1])
    ## [Aura feature sp] patch begin
    if logits.shape[0] != labels.shape[0]:
        repeat_factor = logits.shape[0] // labels.shape[0]
        labels = labels.repeat(repeat_factor)
    ## [Aura feature sp] patch end
    loss, _, _, _ = torch_npu.npu_cross_entropy_loss(logits, labels.reshape(-1), reduction="none")
    return -loss.view(*batch_dim)
