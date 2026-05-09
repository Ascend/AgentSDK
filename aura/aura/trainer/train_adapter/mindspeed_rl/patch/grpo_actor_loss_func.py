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
from typing import Dict, Optional, Tuple

import torch

from mindspeed_rl.utils.utils import generate_mask


def _get_policy_loss_input(
    self, batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract response mask and optional log-prob / advantage tensors from the batch.

    Args:
        batch: Training batch containing at least 'responses' and 'response_length'.

    Returns:
        Tuple of (response_mask, old_log_prob, advantages, ref_log_prob).

    Raises:
        ValueError: If 'responses' key is missing from the batch.
    """
    if 'responses' not in batch:
        raise ValueError("The responses is None")
    response_mask_ori = generate_mask(batch['responses'], batch['response_length']).npu()
    if 'response_mask' in batch:
        # Apply tool-usage mask on top of the original response mask
        response_mask = batch['response_mask'][:, :response_mask_ori.shape[1]] * response_mask_ori
    else:
        response_mask = response_mask_ori
    old_log_prob = batch.get('old_log_prob')
    advantages = batch.get('advantages')
    ref_log_prob = batch.get('ref_log_prob')
    return response_mask, old_log_prob, advantages, ref_log_prob