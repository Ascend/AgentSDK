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


from mindspeed_rl import GenerateConfig


class ExtendedGenerateConfig(GenerateConfig):
    def __init__(self, config_dict):
        # Extended parameters with default values
        defaults = {
            "base_url": "",
            "api_key": "empty",
            "train_backend": "mindspeed_rl",
            "enable_sleep_mode": False,
            "load_format": "megatron",
            "agent_engine": "rllm",
            "infer_backend": "vllm",
            "cudagraph_capture_sizes": None,
            "disable_log_stats": False,
            "enable_chunked_prefill": True,
            "validate_sampling": {
                "max_tokens": 8192,
                "top_p": 0.5,
                "top_k": 50,
                "min_p": 0.01,
                "temperature": 0.2,
            },
            "init_num_group_batches": 1,
            "max_queue_size": 1,
            "weight_save_dir": None,
            "update_weights_interval": 1,
            "ckpt_delta": 1,
            "data_optimized": False,
            "hybrid_batch_num": 1,
            "enable_version_control": False,
            "use_on_policy": False,
            "wait_available_weight_timeout": -1,
            "prefill_enforce_eager": None,
            "prefill_max_num_seqs": None,
            "prefill_max_num_batched_tokens": None,
            "prefill_gpu_memory_utilization": None,
            "prefill_max_model_len": None,
        }
        for key, value in defaults.items():
            setattr(self, key, value)
        # Enable inference statistics by default for load balancing scheduling, False means enable statistics
        super().__init__(config_dict)
        for key, value in defaults.items():
            setattr(self, key, config_dict.get(key, value))
