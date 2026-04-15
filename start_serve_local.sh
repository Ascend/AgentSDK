#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
# 训练任务启动入口

me=$(basename $0)
root_dir=$(realpath $(dirname $0))

export VLLM_VERSION=0.10.2
export GLOO_SOCKET_IFNAME=bond19
export TP_SOCKET_IFNAME=bond19
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WORKSPACE=/${root_dir}
export RLLM_PATH=$WORKSPACE/third_party/agent_engine/rllm
export VLLM_PATH=$WORKSPACE/third_party/infer/vllm
export VLLM_ASCEND_PATH=$WORKSPACE/third_party/infer/vllm_ascend
export MINDSPEED_RL_PATH=$WORKSPACE/third_party/rl/mindspeed_rl
export MEGATRON_PATH=$WORKSPACE/third_party/rl/megatron
export MINDSPEED_PATH=$WORKSPACE/third_party/rl/mindspeed
export MINDSPEED_LLM_PATH=$WORKSPACE/third_party/rl/mindspeed_llm

export PYTHONPATH=${RLLM_PATH}:${VLLM_PATH}:${VLLM_ASCEND_PATH}:${MINDSPEED_RL_PATH}:${MEGATRON_PATH}:${MINDSPEED_PATH}:${MINDSPEED_LLM_PATH}:${PYTHONPATH}

logs_path=./logs/

if [ ! -d "$logs_path" ]; then
    mkdir -p "$logs_path"
    echo "dir created: $logs_path"
else
    echo "dir exists: $logs_path"
fi

function show_help()
{
    echo "usage: ${me} <master {master_ip}|worker {master_ip}>"
}

if [[ $1 == "" || $2 == "" ]]; then
  show_help
  exit 0
fi

ray stop
if [[ $1 == "master" ]]; then
  ray start --head --port 7099
else
  ray start --address="$2:7099"

  # 非0结点循环检查ray集群状态
  while true; do
    ray status > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
    sleep 30
  done
fi

if [[ $1 == "master" ]]; then
  timestamp=$(date +%s%3N)
  python agentic_rl/start.py --config-name=serve_1node_qwen235b 2>&1 | tee ${logs_path}/logs_${timestamp}.log
  ray stop
fi
