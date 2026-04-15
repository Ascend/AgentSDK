#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# 训练任务启动入口


export HCCL_SOCKET_FAMILY=AF_INET
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.10.16/lib/python3.10/site-packages/torch/lib/:/usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/lib/

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

logs_path=./logs/

if [ ! -d "$logs_path" ]; then
    mkdir -p "$logs_path"
    echo "dir created: $logs_path"
else
    echo "dir exists: $logs_path"
fi

# node index for rollout and train cluster
# [MASTER_ROLLOUT_INDEX : MASTER_TRAIN_INDEX) -> rollout; [MASTER_TRAIN_INDEX: ] -> train
MASTER_ROLLOUT_INDEX=0
MASTER_TRAIN_INDEX=16

HOSTS="$VC_WORKER_HOSTS"
MASTER_HOST_ROLLOUT="${HOSTS%%,*}"
MASTER_HOST_TRAIN="$(echo "$VC_WORKER_HOSTS" | cut -d ',' -f $((MASTER_TRAIN_INDEX + 1)))"
export ROLLOUT_NODE=$MASTER_HOST_ROLLOUT
export TRAIN_NODE=$MASTER_HOST_TRAIN

#############################
# Used for PD seperate mode
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="/home/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json"
# use (1) prefill-decode seperate or not (0)
export USE_PD=1
# num of devices used for prefill and decode
export P_INSTANCE_NUM_DEVICE=64
export D_INSTANCE_NUM_DEVICE=64

# python ./scripts/gmm.py

##########################################################
function start_rollout_ray_cluster() {
  need_stuck=$1
  if [[ "$VC_TASK_INDEX" = "$MASTER_ROLLOUT_INDEX" ]]; then
    echo "**********rollout work-0 starts"
    ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources='{"NPU": 8}'
    sleep 30s
  elif [[ "$VC_TASK_INDEX" -gt 0 && "$VC_TASK_INDEX" -lt $MASTER_TRAIN_INDEX ]]; then
    echo "**********rollout work-$VC_TASK_INDEX starts"
    echo "$MASTER_HOST_ROLLOUT:6344"
    sleep 20s
    ray start --address="$MASTER_HOST_ROLLOUT:6344" --resources='{"NPU": 8}'

    # 非0结点循环检查ray集群状态
    if [ ${need_stuck} -eq 1 ]; then
      while true; do
        ray status > /dev/null 2>&1
        if [ $? -ne 0 ]; then
          break
        fi
        sleep 30
      done
    fi
  fi
}

function start_train_ray_cluster() {
  if [ "$VC_TASK_INDEX" = "$MASTER_TRAIN_INDEX" ]; then
    export USE_PD=0
    echo "**********train work-0 starts"
    ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources='{"NPU": 8}'
    sleep 30s
  elif [[ "$VC_TASK_INDEX" -gt $MASTER_TRAIN_INDEX ]]; then
    export USE_PD=0
    echo "**********train work-$VC_TASK_INDEX starts"
    echo "$MASTER_HOST_TRAIN:6344"
    sleep 30s
    ray start --address="$MASTER_HOST_TRAIN:6344" --resources='{"NPU": 8}'
    # 非0结点循环检查ray集群状态
    while true; do
      ray status > /dev/null 2>&1
      if [ $? -ne 0 ]; then
        break
      fi
      sleep 30
    done
  fi
}

# start rollout workers without stuck
start_rollout_ray_cluster 0
start_train_ray_cluster

#########################################################
# generate ranktable for PD seperate mode
if [[ "${USE_PD:-0}" = "1" && "$VC_TASK_INDEX" -lt $MASTER_TRAIN_INDEX ]]; then
    echo "USE_PD=1: Attempting to generate ranktable using Ray cluster IPs..."
    RES_STR=$(python3 -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
    nodes = ray.nodes()
    ips = [node.get('NodeManagerAddress') for node in nodes if node.get('NodeManagerAddress')]
    res = ' '.join(ips)
    print('===ips:', res)
except Exception as e:
    import sys
    print(f'Ray IP fetch failed: {e}', file=sys.stderr)
    sys.exit(1)
finally:
    ray.shutdown()
")
    ray stop
    RAY_IPS_STR=$(echo "$RES_STR" | grep "===ips" | awk -F ':' '{print $2}' | xargs)
    if [ $? -eq 0 ] && [ -n "$RAY_IPS_STR" ]; then
        RAY_IPS=$(echo $RAY_IPS_STR | tr ' ' '\n' | sort -t . -k 1,1n -k 2,2n -k 3,3n -k 4,4n | tr '\n' ' ' | sed 's/ $//')
        echo "Ray node IPs detected: $RAY_IPS"
        CURR_PATH=$(pwd)
        cd /home/vllm-ascend/examples/disaggregated_prefill_v1/
        sed -i 's#LOCAL_HOSTS=($(hostname -I))#LOCAL_HOSTS=($(ifconfig eth0 | awk '"'"'/inet /{print $2}'"'"'))#' gen_ranktable.sh
        rm -f ranktable.json
        bash gen_ranktable.sh \
            --ips $RAY_IPS \
            --npus-per-node 8 \
            --network-card-name eth0 \
            --prefill-device-cnt $P_INSTANCE_NUM_DEVICE \
            --decode-device-cnt $D_INSTANCE_NUM_DEVICE 2>&1
        echo "Ranktable generation completed."
        cd $CURR_PATH
    else
        echo "ERROR: Failed to fetch Ray IPs or no IPs found. Generating ranktable failed." >&2
        exit 0
    fi
    # start rollout workers with stuck
    start_rollout_ray_cluster 1
fi

################################################
# start agentic rl
sleep 1m
ray status
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "current_time: $timestamp"
# worker0 启动rollout任务
if [ "$VC_TASK_INDEX" = "$MASTER_ROLLOUT_INDEX" ]; then
  echo "********** work-$MASTER_ROLLOUT_INDEX rollout"
  sleep 1m
  ray status
  python agentic_rl/start.py --config-name=one_step_off_qwen3_235b_separate_rollout 2>&1 | tee ${logs_path}/rollout_unit_${timestamp}.log
  # 结束ray集群
  ray stop
fi

if [ "$VC_TASK_INDEX" = "$MASTER_TRAIN_INDEX" ]; then
  echo "********** work-$MASTER_TRAIN_INDEX training"
  sleep 1m
  ray status
  python agentic_rl/start.py --config-name=one_step_off_qwen3_235b_separate_dummy_train 2>&1 | tee ${logs_path}/train_unit_${timestamp}.log
  # 结束ray集群
  ray stop
fi

ps -ef | grep "python"| grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {};pkill -9 python; pkill -9 torchrun;

ps -ef | grep "defunct"|grep python| awk '{print $3}'|xargs -t -i kill -9 {};ps -ef | grep "defunct"|grep torchrun| awk '{print $3}'|xargs -t -i kill -9 {}