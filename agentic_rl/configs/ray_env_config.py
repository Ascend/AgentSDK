#!/usr/bin/env python3
# coding=utf-8
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

from dataclasses import dataclass
from typing import Dict


@dataclass
class RayEnvVarsConfig:
    """Environment variables configuration class"""

    # Ray related configuration
    ray_experimental_noset_ascend_rt_visible_devices: str = 'true'

    # Tokenizers configuration
    tokenizers_parallelism: str = 'true'

    # NCCL configuration
    nccl_debug: str = 'WARN'

    # HCCL related configuration
    hccl_connect_timeout: str = '1800'
    hccl_exec_timeout: str = '3600'
    hccl_if_base_port: str = '64000'
    hccl_buffsize: str = '256'

    # CUDA configuration
    cuda_device_max_connections: str = '1'

    # Hydra configuration
    hydra_full_error: str = '1'

    # vLLM related configuration
    vllm_dp_size: str = '1'
    vllm_use_v1: str = '1'
    vllm_version: str = '0.9.0'
    vllm_enable_graph_mode: str = '0'
    vllm_enable_mc2: str = '0'
    vllm_ascend_enable_topk_optimize: str = '1'

    # Communication related configuration
    using_lccl_com: str = '0'
    lcal_comm_id: str = '127.0.0.1:27001'

    # Task queue configuration
    task_queue_enable: str = '2'

    # CPU configuration
    cpu_affinity_conf: str = '1'

    # Training engine configuration
    train_backend: str = 'mindspeed_rl'

    def to_env_dict(self) -> Dict[str, dict | str]:
        """Convert to environment variables dictionary format"""
        return {
            'env_vars':
                {
                    'GLOO_SOCKET_IFNAME': "lo",
                    'NCCL_SOCKET_IFNAME': "lo",
                    'HCCL_SOCKET_IFNAME': "lo",
                    'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES':
                        self.ray_experimental_noset_ascend_rt_visible_devices,
                    'TOKENIZERS_PARALLELISM': self.tokenizers_parallelism,
                    'NCCL_DEBUG': self.nccl_debug,
                    'HCCL_CONNECT_TIMEOUT': self.hccl_connect_timeout,
                    'HCCL_EXEC_TIMEOUT': self.hccl_exec_timeout,
                    'HCCL_IF_IP': "127.0.0.1",
                    'HCCL_IF_BASE_PORT': self.hccl_if_base_port,
                    'CUDA_DEVICE_MAX_CONNECTIONS': self.cuda_device_max_connections,
                    'HYDRA_FULL_ERROR': self.hydra_full_error,
                    'VLLM_DP_SIZE': self.vllm_dp_size,
                    'USING_LCCL_COM': self.using_lccl_com,
                    'HCCL_BUFFSIZE': self.hccl_buffsize,
                    'VLLM_LOGGING_LEVEL': "ERROR",
                    'VLLM_USE_V1': self.vllm_use_v1,
                    'VLLM_VERSION': self.vllm_version,
                    'VLLM_ENABLE_GRAPH_MODE': self.vllm_enable_graph_mode,
                    'VLLM_ENABLE_MC2': self.vllm_enable_mc2,
                    'VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE': self.vllm_ascend_enable_topk_optimize,
                    'TASK_QUEUE_ENABLE': self.task_queue_enable,
                    'CPU_AFFINITY_CONF': self.cpu_affinity_conf,
                    'LCAL_COMM_ID': self.lcal_comm_id,
                    'TRAIN_BACKEND': self.train_backend,
                },
            "worker_process_setup_hook": 'agentic_rl.base.utils.logger_patch.patch'
        }
