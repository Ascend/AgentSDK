#!/bin/bash
# start ray
export HCCL_SOCKET_FAMILY=AF_INET
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.10.16/lib/python3.10/site-packages/torch/lib/:/usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/lib/
export PYTHONPATH=$PYTHONPATH:/code/vllm-ascend/

root_dir=$(realpath $(dirname $0))
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
    echo "目录已创建: $logs_path"
else
    echo "目录已存在: $logs_path"
fi


HOSTS="$VC_TASK_HOSTS"
MASTER_HOST="${HOSTS%%,*}"

if [ "$VC_TASK_INDEX" = "0" ]; then
  echo "********** work-0 starts"
  ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources='{"NPU": 8}'
  sleep 1m
else
  echo "********** work-$VC_TASK_INDEX starts"
  echo "$MASTER_HOST:6344"
  sleep 30s
  ray start --address="$MASTER_HOST:6344" --resources='{"NPU": 8}'
  sleep 30
  # 非0结点循环检查ray集群状态
  while true; do
    ray status > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
    sleep 30
  done
fi
ray status

# worker0 启动训练任务
if [ "$VC_TASK_INDEX" = "0" ]; then
  echo "********** work-0 training"
  sleep 1m
  ray status
  python agentic_rl/start.py --config-name=agentic_rl_direct_mode_one_step_off 2>&1 | tee ${logs_path}/logs_${JOB_NAME}_${timestamp}.log
  # 结束ray集群
  ray stop
fi

ps -ef | grep "python"| grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {};pkill -9 python; pkill -9 torchrun;

ps -ef | grep "defunct"|grep python| awk '{print $3}'|xargs -t -i kill -9 {};ps -ef | grep "defunct"|grep torchrun| awk '{print $3}'|xargs -t -i kill -9 {}