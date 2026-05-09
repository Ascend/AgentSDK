#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
# 训练任务启动入口

# 注: VLLM_CACHE_ROOT和TORCH_EXTENSIONS_DIR需根据实际cache路径配置(可以不配置)
export VLLM_CACHE_ROOT=/models/.cache/vllm/
export TORCH_EXTENSIONS_DIR=/home/.cache/torch_extension
export RAY_heartbeat_timeout_milliseconds=120000
export RAY_raylet_heartbeat_period_milliseconds=120000


HOSTS="$VC_WORKER_HOSTS"
MASTER_HOST_0="${HOSTS%%,*}"
MASTER_HOST_32="$(echo "$VC_WORKER_HOSTS" | cut -d ',' -f 33)"

export ROLLOUT_NODE=$MASTER_HOST_0
export TRAIN_NODE=$MASTER_HOST_32

rm -rf third_party/rl/mindspeed_llm/mindspeed_llm/core/context_parallel/ring_context_parallel.py
cp cp_opt/ring_context_parallel_opt.py third_party/rl/mindspeed_llm/mindspeed_llm/core/context_parallel/ring_context_parallel.py

rm -rf third_party/rl/mindspeed/mindspeed/core/context_parallel/utils.py
cp cp_opt/utils_opt.py third_party/rl/mindspeed/mindspeed/core/context_parallel/utils.py

rm -rf third_party/rl/mindspeed_llm/mindspeed_llm/core/transformer/dot_product_attention.py
cp cp_opt/dot_product_attention_opt.py third_party/rl/mindspeed_llm/mindspeed_llm/core/transformer/dot_product_attention.py

rm -rf third_party/rl/mindspeed_rl/mindspeed_rl/utils/seqlen_balancing.py
cp cp_opt/seqlen_balancing_opt.py third_party/rl/mindspeed_rl/mindspeed_rl/utils/seqlen_balancing.py

python ./scripts/gmm.py

# index 0: 推理master节点
if [ "$VC_TASK_INDEX" = "0" ]; then
  echo "********** infer work-0 starts"
  bash start_one_step_off_train_local_separate.sh --node_type master \
  --train_ip "${TRAIN_NODE}" \
  --rollout_ip "${ROLLOUT_NODE}" \
  --task_type rollout

# index 32: 训练master节点
elif [ "$VC_TASK_INDEX" = "32" ]; then
  echo "********** train work-0 starts"
  bash start_one_step_off_train_local_separate.sh --node_type master \
  --train_ip "${TRAIN_NODE}" \
  --rollout_ip "${ROLLOUT_NODE}" \
  --task_type train

# 判断 index 是否在 1 到 31 之间: 推理worker节点
elif [[ "$VC_TASK_INDEX" -ge 0 && "$VC_TASK_INDEX" -le 32 ]]; then
  echo "********** infer work-$VC_TASK_INDEX starts"
  echo "$ROLLOUT_NODE:6344"
  bash start_one_step_off_train_local_separate.sh --node_type worker \
  --train_ip "${TRAIN_NODE}" \
  --rollout_ip "${ROLLOUT_NODE}" \
  --task_type rollout


# 判断 index 是否在 33 到 48 之间: 训练worker节点
else
  echo "********** train work-$VC_TASK_INDEX starts"
  echo "$TRAIN_NODE:6344"
  bash start_one_step_off_train_local_separate.sh --node_type worker \
  --train_ip "${TRAIN_NODE}" \
  --rollout_ip "${ROLLOUT_NODE}" \
  --task_type train
fi

echo "Training End ..."