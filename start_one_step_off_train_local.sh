#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
# 训练任务启动入口

source scripts/envs.sh
source scripts/input_args.sh

logs_path=./logs/
if [ ! -d "$logs_path" ]; then
    mkdir -p "$logs_path"
    echo "dir created: $logs_path"
else
    echo "dir exists: $logs_path"
fi

export TRAIN_NODE=${train_ip}
export ROLLOUT_NODE=${rollout_ip}

echo "===TRAIN_NODE: ${TRAIN_NODE}, ROLLOUT_NODE: ${ROLLOUT_NODE}==="

ray stop
if [[ ${node_type} == "master" ]]; then
  ray start --head --port 7099
else
  if [[ ${task_type} == "train" ]]; then
    ray start --address="${TRAIN_NODE}:7099"
  else
    ray start --address="${ROLLOUT_NODE}:7099"
  fi

  # 非0节点循环检查ray集群状态
  while true; do
    ray status > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
    sleep 30
  done
fi

timestamp=$(date +%s%3N)
if [[ ${node_type} == "master" && ${task_type} == "train" ]]; then
  python agentic_rl/start.py --config-name=one_step_off_1node_qwen25_7b_separate_train 2>&1 | tee ${logs_path}/logs_${timestamp}.log
  ray stop
elif [[ ${node_type} == "master" && ${task_type} == "rollout" ]]; then
  python agentic_rl/start.py --config-name=one_step_off_1node_qwen25_7b_separate_rollout 2>&1 | tee ${logs_path}/logs_${timestamp}.log
  ray stop
fi
