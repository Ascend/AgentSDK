#!/usr/bin/env python3
# coding=utf-8

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


import os
from mindspeed_rl import GenerateConfig


class ExtendedGenerateConfig(GenerateConfig):
    def __init__(self, config_dict):
        # Extended parameters with default values
        self.base_url = ""
        self.api_key = "empty"
        self.train_backend = "mindspeed_rl"
        self.enable_sleep_mode = False
        self.load_format = "megatron"
        self.agent_engine = "rllm"
        self.infer_backend = "vllm"
        self.cudagraph_capture_sizes = None
        # Enable inference statistics by default for load balancing scheduling, False means enable statistics
        self.disable_log_stats = False
        self.enable_chunked_prefill = True

        self.validate_sampling = {
            "max_tokens": 8192,
            "top_p": 0.5,
            "top_k": 50,
            "min_p": 0.01,
            "temperature": 0.2
        }

        self.init_num_group_batches = 1
        self.hybrid_batch_num = 1
        self.enable_version_control = False
        self.use_on_policy = False
        self.max_queue_size = 1
        self.weight_save_dir = None
        self.update_weights_interval = 1
        self.ckpt_delta = 1
        self.data_optimized = False

        # add prefill params
        self.prefill_enforce_eager = None
        self.prefill_max_num_seqs = None
        self.prefill_max_num_batched_tokens = None
        self.prefill_gpu_memory_utilization = None
        self.prefill_max_model_len = None

        super().__init__(config_dict)
