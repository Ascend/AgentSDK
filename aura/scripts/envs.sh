#!/bin/bash

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

me=$(basename $0)
root_dir=$(realpath $(dirname $0))


export VLLM_VERSION=0.10.2
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export LOG_WORKLOAD_ENABLE="0"
export WORKSPACE=/${root_dir}
export RLLM_PATH=$WORKSPACE/third_party/agent_engine/rllm
export VLLM_PATH=$WORKSPACE/third_party/infer/vllm
export VLLM_ASCEND_PATH=$WORKSPACE/third_party/infer/vllm_ascend
export MINDSPEED_RL_PATH=$WORKSPACE/third_party/rl/mindspeed_rl
export MEGATRON_PATH=$WORKSPACE/third_party/rl/megatron
export MINDSPEED_PATH=$WORKSPACE/third_party/rl/mindspeed
export MINDSPEED_LLM_PATH=$WORKSPACE/third_party/rl/mindspeed_llm

export PYTHONPATH=${RLLM_PATH}:${VLLM_PATH}:${VLLM_ASCEND_PATH}:${MINDSPEED_RL_PATH}:${MEGATRON_PATH}:${MINDSPEED_PATH}:${MINDSPEED_LLM_PATH}:${PYTHONPATH}