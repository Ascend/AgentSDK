#!/bin/bash
# start ray
export HCCL_SOCKET_FAMILY=AF_INET
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.10.16/lib/python3.10/site-packages/torch/lib/:/usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/lib/
export PYTHONPATH=$PYTHONPATH:/code/vllm-ascend/
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

DEFAULT_YAML="integrated_grpo_qwq32_2node"
DEFAULT_LOG_PATH="logs"
YAML=${1:-$DEFAULT_YAML}
LOG_PATH=${2:-$DEFAULT_LOG_PATH}

current_time=$(date +"%Y%m%d_%H%M%S")
echo "current_time: $current_time"

mkdir -p "$LOG_PATH"
echo "[log path]: $LOG_PATH"

HOSTS="$VC_TASK_HOSTS"
MASTER_HOST="${HOSTS%%,*}"

python ./scripts/gmm.py

if [ "$VC_TASK_INDEX" = "0" ]; then
  echo "********** work-0 starts"
  ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources='{"NPU": 8}'
  sleep 1m
else
  echo "********** work-$VC_TASK_INDEX starts"
  echo "$MASTER_HOST:6344"
  sleep 30
  ray start --address="$MASTER_HOST:6344" --resources='{"NPU": 8}'
  sleep 30
  while true; do
    ray status > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
    sleep 30
  done
  exit 1
fi
ray status

if [ "$VC_TASK_INDEX" = "0" ]; then
  echo "********** work-0 training"
  sleep 2m
  ray status
  python aura/trainer/train_grpo.py --config-name=${YAML} 2>&1 | tee ${LOG_PATH}/logs_${JOB_NAME}_${current_time}.log

  python_exit_code=${PIPESTATUS[0]}
  if [ $python_exit_code -eq 0 ]; then
      ray stop
  else
    exit 1
  fi
fi

ps -ef | grep "python"| grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {};pkill -9 python; pkill -9 torchrun;

ps -ef | grep "defunct"|grep python| awk '{print $3}'|xargs -t -i kill -9 {};ps -ef | grep "defunct"|grep torchrun| awk '{print $3}'|xargs -t -i kill -9 {}
