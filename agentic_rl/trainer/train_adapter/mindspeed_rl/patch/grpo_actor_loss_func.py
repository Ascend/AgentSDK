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

from typing import Dict

import torch
from mindspeed_rl.utils.utils import generate_mask

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


def get_policy_loss_input_patch(self, batch: Dict[str, torch.Tensor]):
    """
    Generate Grpo actor loss mask for response.

    Args:
        self: GRPOActorLossFunc instance.
        batch (Dict[str, torch.Tensor]): map of tensor for calculate loss.

    Returns:
        response_mask (torch.Tensor): response mask.
        old_log_prob, advantages, ref_log_prob (torch.Tensor): extract tensor from batch.
    """

    if 'responses' not in batch or 'response_length' not in batch:
        logger.error("The responses or response_length is None")
        raise ValueError("The responses or response_length is None")
    try:
        response_mask_ori = generate_mask(batch['responses'], batch['response_length']).npu()
    except AttributeError as e:
        logger.error(f"Failed to generate mask with error: {e}")
        raise AttributeError("Failed to generate mask") from e
    except Exception as e:
        logger.error(f"Unexpected error occurred when generate mask: {e}")
        raise RuntimeError("Unexpected error occurred when generate mask.") from e

    if len(response_mask_ori.shape) != 2:
        logger.error("Generated response mask tensor dimension should be equal to 2")
        raise ValueError("Generated response mask tensor dimension should be equal to 2")

    if 'response_mask' in batch:
        response_mask = batch["response_mask"]

        if response_mask.shape[0] != response_mask_ori.shape[0]:
            logger.error("The length of dim 0 of response_mask is not equal to generated mask")
            raise ValueError("The length of dim 0 of response_mask is not equal to generated mask")

        if response_mask.shape[1] < response_mask_ori.shape[1]:
            logger.error("The length of dim 1 of response_mask is less than generated mask")
            raise ValueError("The length of dim 1 of response_mask is less than generated mask")

        response_mask = response_mask[:, :response_mask_ori.shape[1]] * response_mask_ori  # use mask from agent
    else:
        response_mask = response_mask_ori

    old_log_prob = batch['old_log_prob'] if 'old_log_prob' in batch else None
    advantages = batch['advantages'] if 'advantages' in batch else None
    ref_log_prob = batch['ref_log_prob'] if 'ref_log_prob' in batch else None
    return response_mask, old_log_prob, advantages, ref_log_prob
