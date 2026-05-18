#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# 训练任务启动入口

base_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))
scripts_dir=$(realpath $(dirname ${base_dir}))
root_dir=$(realpath $(dirname ${scripts_dir}))

export HCCL_SOCKET_FAMILY=AF_INET
export LD_LIBRARY_PATH=\
$LD_LIBRARY_PATH:/usr/local/python3.11.14/lib/python3.11/site-packages/torch/lib/:\
/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib/

export VLLM_VERSION=0.11.0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export LOG_WORKLOAD_ENABLE="0"

export WORKSPACE=${root_dir}
export RLLM_PATH=${WORKSPACE}/third_party/agent_engine/rllm
export VLLM_PATH=${WORKSPACE}/third_party/infer/vllm
export VLLM_ASCEND_PATH=${WORKSPACE}/third_party/infer/vllm_ascend
export MINDSPEED_RL_PATH=${WORKSPACE}/third_party/rl/mindspeed_rl
export MEGATRON_PATH=${WORKSPACE}/third_party/rl/megatron
export MINDSPEED_PATH=${WORKSPACE}/third_party/rl/mindspeed
export MINDSPEED_LLM_PATH=${WORKSPACE}/third_party/rl/mindspeed_llm

export LOG_PATH=${root_dir}/logs/
function log_init()
{
  if [ ! -d "$LOG_PATH" ]; then
      mkdir -p "$LOG_PATH"
      echo "Dir created: $LOG_PATH"
  fi
}

log_init
