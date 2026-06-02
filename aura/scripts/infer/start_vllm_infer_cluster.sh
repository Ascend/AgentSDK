#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# 推理集群启动入口

infer_dir=$(realpath $(dirname $0))
scripts_dir=$(realpath $(dirname ${infer_dir}))
root_dir=$(realpath $(dirname ${scripts_dir}))

source ${scripts_dir}/base/envs.sh
source ${scripts_dir}/base/utils.sh

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config-name)
      export CONFIG_NAME=$2
      shift
      ;;
    *)
      log_error "unknown arg: $1"
      exit 1
      ;;
  esac
  shift
done

source ${scripts_dir}/infer/vllm/parse_infer_config.sh

export PYTHONPATH=${WORKSPACE}:${PYTHONPATH}

export VC_TASK_INDEX=${VC_TASK_INDEX:-$1}

# 将超时从 60s 延长到 10 分钟
export VLLM_RPC_TIMEOUT=600000
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
# 针对 v1 引擎的共享内存同步超时
export VLLM_SHM_BROADCAST_TIMEOUT=600

function start_infer_instances()
{
  URL="http://${P_INSTANCE_HOSTS_ARRAY[0]}:20012/metrics"
  log_info "URL: ${URL}"
  STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$URL")
  log_info "STATUS_CODE: ${STATUS_CODE}"
  if [ "$STATUS_CODE" -eq 200 ]; then
    log_info "Server ${P_INSTANCE_HOSTS_ARRAY[0]} is ready! Status code: $STATUS_CODE"
    return
  fi

  # patch vllm
  bash ${scripts_dir}/infer/vllm/patch_vllm.sh

  # start vllm cluster
  timestamp=$(date +"%Y%m%d_%H%M%S")
  cd ${scripts_dir}/infer/vllm

  export DEFAULT_SOCKET_IFNAME=${DEFAULT_SOCKET_IFNAME:-"eth0"}
  # 阻塞运行
  bash vllm_launch.sh \
     --prefill-instances ${PREFILL_INSTANCE_COUNT}\
     --decode-instances ${DECODE_INSTANCE_COUNT}\
     --prefill-cards-per-instance ${PREFILL_CARDS_PER_INSTANCE}\
     --decode-cards-per-instance ${DECODE_CARDS_PER_INSTANCE}\
     --node-cards ${NPU_NUM_PER_NODE} --socket-ifname ${DEFAULT_SOCKET_IFNAME} 2>&1|tee ${LOG_PATH}/infer_unit_${VC_TASK_INDEX}_${timestamp}.log

  # 如果上面没有阻塞住, 表示推理集群异常退出了
  log_error "start vllm infer failed!!!"
  exit 1
}

function qwen35_moe_model_handle()
{
  if [[ "${SERVED_MODEL_NAME}" == "Qwen3.5-35B-A3B" ||
    "${SERVED_MODEL_NAME}" == "Qwen3.5-122B-A10B" ||
    "${SERVED_MODEL_NAME}" == "Qwen3.5-397B-A17B" ]]; then
    export TRANSPOSE_EXPERT_SHAPE="true"
    log_info "TRANSPOSE_EXPERT_SHAPE: ${TRANSPOSE_EXPERT_SHAPE}"
  fi
}

get_infer_configs

log_info "[infer] ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES}"

log_info "[infer] PYTHONPATH: ${PYTHONPATH}"

qwen35_moe_model_handle
start_infer_instances
