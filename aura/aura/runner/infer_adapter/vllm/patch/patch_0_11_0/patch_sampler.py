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

import torch
import os
from vllm_ascend.sample.sampler import AscendTopKTopPSampler
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

from ..comm.vllm_execute_stat import StatTimeUtil, vllm_output_statics, StatPhase


def forward_native_patch(self, logits, generators, k, p):
    """Override pytorch native implementation to torch_npu"""
    time_util_sample = StatTimeUtil()
    logits = self._apply_top_k_top_p(logits, k, p)
    vllm_output_statics.add_stat(StatPhase.post_samper_sample_topk_topp_apply_time, time_util_sample.get_duration())
    logits_to_return = None
    if self.logprobs_mode == "processed_logits":
        logits_to_return = logits
    elif self.logprobs_mode == "processed_logprobs":
        logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
    vllm_output_statics.add_stat(
        StatPhase.post_samper_sample_topk_topp_logits_log_softmax_time, time_util_sample.get_duration()
    )

    probs = logits.softmax(dim=-1, dtype=torch.float32)
    vllm_output_statics.add_stat(
        StatPhase.post_samper_sample_topk_topp_probs_softmax_time, time_util_sample.get_duration()
    )
    output = random_sample(probs, generators), logits_to_return
    vllm_output_statics.add_stat(
        StatPhase.post_samper_sample_topk_topp_random_sample_time, time_util_sample.get_duration()
    )
    return output


is_vllm_statistic = os.getenv('ENABLE_VLLM_STAT', "False").lower() == "true"
vllm_statistic_level = int(os.environ.get('VLLM_STAT_LEVEL', "0"))
if is_vllm_statistic and vllm_statistic_level == 1:
    AscendTopKTopPSampler.forward_native = forward_native_patch
