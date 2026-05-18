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

from vllm_ascend.utils import vllm_version_is
from ..comm.vllm_execute_stat import StatTimeUtil, vllm_output_statics, StatPhase

if not (vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1")):
    from vllm.config import LogprobsMode
else:
    LogprobsMode = None


def forward_native_patch(self, logits, generators, k, p):
    """Override pytorch native implementation to torch_npu"""
    # print("===hucuihua AscendSampler forward_native, generators:", generators, " k:", generators, " p:", p)
    time_util_sample = StatTimeUtil()
    logits = self._apply_top_k_top_p(logits, k, p)
    vllm_output_statics.add_stat(StatPhase.post_samper_sample_topk_topp_apply_time, time_util_sample.get_duration())
    if not (vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1")):
        logits_to_return = None
        if self.logprobs_mode == LogprobsMode.PROCESSED_LOGITS:
            logits_to_return = logits
        elif self.logprobs_mode == LogprobsMode.PROCESSED_LOGPROBS:
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
    vllm_output_statics.add_stat(
        StatPhase.post_samper_sample_topk_topp_logits_log_softmax_time, time_util_sample.get_duration()
    )

    probs = logits.softmax(dim=-1, dtype=torch.float32)
    vllm_output_statics.add_stat(
        StatPhase.post_samper_sample_topk_topp_probs_softmax_time, time_util_sample.get_duration()
    )
    output = None
    if vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1"):
        output = random_sample(probs, generators)
    else:
        output = (random_sample(probs, generators), logits_to_return)
    vllm_output_statics.add_stat(
        StatPhase.post_samper_sample_topk_topp_random_sample_time, time_util_sample.get_duration()
    )
    return output


is_vllm_statistic = os.getenv('ENABLE_VLLM_STAT', "False").lower() == "true"
if is_vllm_statistic:
    AscendTopKTopPSampler.forward_native = forward_native_patch
