#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
# 训练任务启动入口

# # 104执行
# ray stop && ./start_one_step_off_train_local_separate_pd.sh master 10.50.89.103 10.50.89.104 rollout ranktable_104_136.json
# # 136执行
# ray stop && ./start_one_step_off_train_local_separate_pd.sh worker 10.50.89.103 10.50.89.104 rollout ranktable_104_136.json
# # 103执行
# ray stop && ./start_one_step_off_train_local_separate_pd.sh master 10.50.89.103 10.50.89.104 train ranktable_104_136.json


me=$(basename $0)
root_dir=$(realpath $(dirname $0))

export VLLM_VERSION=0.10.2
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

export CONTROLLER_MODE=separate

export WORKSPACE=/${root_dir}
export RLLM_PATH=$WORKSPACE/third_party/agent_engine/rllm
export VLLM_PATH=$WORKSPACE/third_party/infer/vllm
export VLLM_ASCEND_PATH=$WORKSPACE/third_party/infer/vllm_ascend
export MINDSPEED_RL_PATH=$WORKSPACE/third_party/rl/mindspeed_rl
export MEGATRON_PATH=$WORKSPACE/third_party/rl/megatron
export MINDSPEED_PATH=$WORKSPACE/third_party/rl/mindspeed
export MINDSPEED_LLM_PATH=$WORKSPACE/third_party/rl/mindspeed_llm

export PYTHONPATH=${RLLM_PATH}:${VLLM_PATH}:${VLLM_ASCEND_PATH}:${MINDSPEED_RL_PATH}:${MEGATRON_PATH}:${MINDSPEED_PATH}:${MINDSPEED_LLM_PATH}:${PYTHONPATH}
# params for PD seperate
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="/opt/DPC/models/l30060498/Code/AgenticRL-A2/v5.0.0-dev/ranktable_143_103.json"
export USE_PD=1 # use (1) prefill-decode seperate or not (0)
export P_INSTANCE_NUM_DEVICE=8
export D_INSTANCE_NUM_DEVICE=8

logs_path=./logs/

if [ ! -d "$logs_path" ]; then
    mkdir -p "$logs_path"
    echo "dir created: $logs_path"
else
    echo "dir exists: $logs_path"
fi

function show_help()
{
    echo "usage: ${me} <master {train_master_ip} {rollout_master_ip} {train|rollout}|worker {train_master_ip} {rollout_master_ip} {train|rollout} {rank_table}>"
}

if [[ $1 == "" || $2 == "" || $3 == "" || $4 == "" ]]; then
  show_help
  exit 0
fi

export TRAIN_NODE=$2
export ROLLOUT_NODE=$3
if [[ "${USE_PD}" == "1" && $# -ge 5 ]]; then
  export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="$5"
fi

echo "===TRAIN_NODE: ${TRAIN_NODE}, ROLLOUT_NODE: ${ROLLOUT_NODE}==="

ray stop
if [[ $1 == "master" ]]; then
  ray start --head --port 2358
else
  if [[ $3 == "train" ]]; then
    ray start --address="${TRAIN_NODE}:2358"
  else
    ray start --address="${ROLLOUT_NODE}:2358"
  fi

  # 非0结点循环检查ray集群状态
  while true; do
    ray status > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
    sleep 30
  done
fi

if [[ $1 == "master" && $4 == "train" ]]; then
  timestamp=$(date +%s%3N)
#  python agentic_rl/start.py --config-name=one_step_off_1node_qwen25_7b_separate_dummy_train 2>&1 | tee ${logs_path}/logs_${timestamp}.log
  python agentic_rl/start.py --config-name=one_step_off_qwq_32b_separate_dummy_train 2>&1 | tee ${logs_path}/logs_${timestamp}.log
  ray stop
elif [[ $1 == "master" && $4 == "rollout" ]]; then
  timestamp=$(date +%s%3N)
#  python agentic_rl/start.py --config-name=one_step_off_1node_qwen25_7b_separate_rollout 2>&1 | tee ${logs_path}/logs_${timestamp}.log
  python agentic_rl/start.py --config-name=one_step_off_qwq_32b_separate_rollout 2>&1 | tee ${logs_path}/logs_${timestamp}.log
  ray stop
fi
