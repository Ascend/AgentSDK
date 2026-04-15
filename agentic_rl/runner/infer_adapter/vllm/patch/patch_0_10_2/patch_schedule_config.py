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

# SPDX-License-Identifier: Apache-2.0
from vllm_ascend.core.schedule_config import AscendSchedulerConfig


def ascend_scheduler_config_post_init(self) -> None:
    self.max_num_encoder_input_tokens = self.max_num_batched_tokens
    self.encoder_cache_size = self.max_num_batched_tokens
    self.chunked_prefill_enabled = self.enable_chunked_prefill
    
    if self.policy != "fcfs":
        raise NotImplementedError(
            f"currently AscendScheduler only supports fcfs policy, got {self.policy}"
        )
    
    if self.is_multimodal_model:
        raise NotImplementedError(
            "currently AscendScheduler only supports LLM models.")
    if self.num_scheduler_steps > 1:
        raise NotImplementedError(
            "currently AscendScheduler doesn't support multi-step.")
    if self.send_delta_data:
        raise NotImplementedError(
            "currently AscendScheduler doesn't support send_delta_data.")
    if self.delay_factor > 0:
        raise NotImplementedError(
            "currently AscendScheduler doesn't support scheduler_delay_factor."
        )


AscendSchedulerConfig.__post_init__ = ascend_scheduler_config_post_init
