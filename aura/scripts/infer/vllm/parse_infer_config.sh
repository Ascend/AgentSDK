#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# 外挂推理集群配置解析

vllm_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))
infer_dir=$(realpath $(dirname ${vllm_dir}))
scripts_dir=$(realpath $(dirname ${infer_dir}))
root_dir=$(realpath $(dirname $scripts_dir))

source ${scripts_dir}/base/utils.sh
export CONFIG_FILE=${root_dir}/configs/infer/${INFER_CONF_NAME}.yaml

log_info "CONFIG_FILE: ${CONFIG_FILE}"
log_info "================parse infer params begin================="
export VLLM_VERSION=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} vllm_version)

export SERVED_MODEL_NAME=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} infer_model_name)
export MODEL_PATH=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} infer_mode_path)

export USE_PD=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} pd_mode)

export ENABLE_EXPERT_PARALLEL=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} enable_expert_parallel)

export PREFILL_INSTANCE_COUNT=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} prefill_instance_count)
export DECODE_INSTANCE_COUNT=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} decode_instance_count)
export PREFILL_TENSOR_PARALLEL_SIZE=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} tensor_parallel_size)
export PREFILL_DATA_PARALLEL_SIZE=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} data_parallel_size)
export DECODE_TENSOR_PARALLEL_SIZE=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} tensor_parallel_size)
export DECODE_DATA_PARALLEL_SIZE=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} data_parallel_size)

export MAX_MODEL_LEN=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} max_model_len)
export MAX_NUM_BATCHED_TOKENS=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} max_num_batched_tokens)
export GPU_MEMORY_UTILIZATION=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} gpu_memory_utilization)
export MAX_NUM_SEQS=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} max_num_seqs)
export CUDAGRAPH_CAPTURE_SIZES=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} cudagraph_capture_sizes)

export KV_BACKEND=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} kv_backend)

export ENABLE_VLLM_STAT=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} enable_vllm_stat)
export ENABLE_TENSOR_SIMILARITY_CHECK=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} enable_tensor_similarity_check)

export VLLM_ASCEND_ENABLE_FLASHCOMM=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} vllm_ascend_enable_flashcomm)
export VLLM_ASCEND_ENABLE_FLASHCOMM1=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} vllm_ascend_enable_flashcomm1)

export TOOL_CALL_ENABLE=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} tool_call_enable)
export USE_VLLM_OPT=$(python3 ${scripts_dir}/base/get_yaml.py ${CONFIG_FILE} use_vllm_opt)

log_info "VLLM_VERSION: ${VLLM_VERSION}"
log_info "SERVED_MODEL_NAME: ${SERVED_MODEL_NAME}"
log_info "MODEL_PATH: ${MODEL_PATH}"
log_info "USE_PD: ${USE_PD}"
log_info "ENABLE_EXPERT_PARALLEL: ${ENABLE_EXPERT_PARALLEL}"
log_info "PREFILL_INSTANCE_COUNT: ${PREFILL_INSTANCE_COUNT}"
log_info "DECODE_INSTANCE_COUNT: ${DECODE_INSTANCE_COUNT}"
log_info "PREFILL_TENSOR_PARALLEL_SIZE: ${PREFILL_TENSOR_PARALLEL_SIZE}"
log_info "PREFILL_DATA_PARALLEL_SIZE: ${PREFILL_DATA_PARALLEL_SIZE}"
log_info "DECODE_TENSOR_PARALLEL_SIZE: ${DECODE_TENSOR_PARALLEL_SIZE}"
log_info "DECODE_DATA_PARALLEL_SIZE: ${DECODE_DATA_PARALLEL_SIZE}"
log_info "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
log_info "MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS}"
log_info "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
log_info "MAX_NUM_SEQS: ${MAX_NUM_SEQS}"
log_info "CUDAGRAPH_CAPTURE_SIZES: ${CUDAGRAPH_CAPTURE_SIZES}"
log_info "KV_BACKEND: ${KV_BACKEND}"
log_info "ENABLE_VLLM_STAT: ${ENABLE_VLLM_STAT}"
log_info "ENABLE_TENSOR_SIMILARITY_CHECK: ${ENABLE_TENSOR_SIMILARITY_CHECK}"
log_info "VLLM_ASCEND_ENABLE_FLASHCOMM: ${VLLM_ASCEND_ENABLE_FLASHCOMM}"
log_info "VLLM_ASCEND_ENABLE_FLASHCOMM1: ${VLLM_ASCEND_ENABLE_FLASHCOMM1}"
log_info "TOOL_CALL_ENABLE: ${TOOL_CALL_ENABLE}"
log_info "USE_VLLM_OPT: ${USE_VLLM_OPT}"
log_info "================parse infer params end================="

# A2每节点8卡, A3每节点16卡
export NPU_NUM_PER_NODE=8
if [[ "$CARD_TYPE" -eq "${A3_CARD}" ]]; then
  export NPU_NUM_PER_NODE=16
fi

PREFILL_DATA_PARALLEL_SIZE_LOCAL=$((NPU_NUM_PER_NODE / PREFILL_TENSOR_PARALLEL_SIZE))
DECODE_DATA_PARALLEL_SIZE_LOCAL=$((NPU_NUM_PER_NODE / DECODE_TENSOR_PARALLEL_SIZE))

export PREFILL_DATA_PARALLEL_SIZE_LOCAL=$((PREFILL_DATA_PARALLEL_SIZE == 1 ? 1 : PREFILL_DATA_PARALLEL_SIZE_LOCAL))
export DECODE_DATA_PARALLEL_SIZE_LOCAL=$((DECODE_DATA_PARALLEL_SIZE == 1 ? 1 : DECODE_DATA_PARALLEL_SIZE_LOCAL))
export PREFILL_DATA_PARALLEL_SIZE_LOCAL=$((PREFILL_DATA_PARALLEL_SIZE_LOCAL < 1 ? 1 : PREFILL_DATA_PARALLEL_SIZE_LOCAL))
export DECODE_DATA_PARALLEL_SIZE_LOCAL=$((DECODE_DATA_PARALLEL_SIZE_LOCAL < 1 ? 1 : DECODE_DATA_PARALLEL_SIZE_LOCAL))

export PREFILL_CARDS_PER_INSTANCE=$((PREFILL_TENSOR_PARALLEL_SIZE * PREFILL_DATA_PARALLEL_SIZE))
export DECODE_CARDS_PER_INSTANCE=$((DECODE_TENSOR_PARALLEL_SIZE * DECODE_DATA_PARALLEL_SIZE))

export P_INSTANCE_NUM_DEVICE=$((PREFILL_INSTANCE_COUNT * PREFILL_CARDS_PER_INSTANCE))
export D_INSTANCE_NUM_DEVICE=$((DECODE_INSTANCE_COUNT * DECODE_CARDS_PER_INSTANCE))

# 只有开启这个才能调用collective_rpc接口, 使能外部注册扩展接口
export VLLM_SERVER_DEV_MODE=1

log_info "============================================================"
log_info "CARD_TYPE: ${CARD_TYPE}"
log_info "IS_SHARED_FILESYSTEM: ${IS_SHARED_FILESYSTEM}"
log_info "NPU_NUM_PER_NODE: ${NPU_NUM_PER_NODE}"
log_info "PREFILL_DATA_PARALLEL_SIZE: ${PREFILL_DATA_PARALLEL_SIZE}"
log_info "DECODE_DATA_PARALLEL_SIZE: ${DECODE_DATA_PARALLEL_SIZE}"
log_info "PREFILL_DATA_PARALLEL_SIZE_LOCAL: ${PREFILL_DATA_PARALLEL_SIZE_LOCAL}"
log_info "DECODE_DATA_PARALLEL_SIZE_LOCAL: ${DECODE_DATA_PARALLEL_SIZE_LOCAL}"
log_info "PREFILL_CARDS_PER_INSTANCE: ${PREFILL_CARDS_PER_INSTANCE}"
log_info "DECODE_CARDS_PER_INSTANCE: ${DECODE_CARDS_PER_INSTANCE}"
log_info "P_INSTANCE_NUM_DEVICE: ${P_INSTANCE_NUM_DEVICE}"
log_info "D_INSTANCE_NUM_DEVICE: ${D_INSTANCE_NUM_DEVICE}"
log_info "============================================================"

###################################################################################

function allocate_pd_hosts()
{
  local vc_worker_hosts_str="${VC_WORKER_HOSTS}"
  local p_instance_device_num="${P_INSTANCE_NUM_DEVICE}"
  local p_devices_per_instance="${PREFILL_CARDS_PER_INSTANCE}"
  local d_instance_device_num="${D_INSTANCE_NUM_DEVICE}"
  local d_devices_per_instance="${DECODE_CARDS_PER_INSTANCE}"

  log_info "========================allocate_pd_hosts begin======================="
  log_info "vc_worker_hosts_str: ${vc_worker_hosts_str}"

  IFS=',' read -r -a host_array <<< "$vc_worker_hosts_str"
  local p_required_ips=$((p_instance_device_num / NPU_NUM_PER_NODE))
  local p_select_ips=$((p_devices_per_instance / NPU_NUM_PER_NODE))
  local d_required_ips=$((d_instance_device_num / NPU_NUM_PER_NODE))
  local d_select_ips=$((d_devices_per_instance / NPU_NUM_PER_NODE))

  p_required_ips=$((p_required_ips < 1 ? 1 : p_required_ips))
  p_select_ips=$((p_select_ips < 1 ? 1 : p_select_ips))
  d_required_ips=$((d_required_ips < 1 ? 1 : d_required_ips))
  d_select_ips=$((d_select_ips < 1 ? 1 : d_select_ips))

  log_info "p_required_ips: ${p_required_ips}"
  log_info "p_select_ips: ${p_select_ips}"
  log_info "d_required_ips: ${d_required_ips}"
  log_info "d_select_ips: ${d_select_ips}"

  local total_required_ips=$((p_required_ips + (USE_PD == 1 ? d_required_ips : 0)))
  local available_ips=${#host_array[@]}
  if (( total_required_ips > available_ips )); then
    log_error "allocate_hosts failed! required ips: $total_required_ips, available ips: $available_ips"
    return
  fi

  local p_hosts_str=""
  local d_hosts_str=""

  # 分配给P实例
  for (( i=0; i<p_required_ips; i++ )); do
    if (( i % p_select_ips != 0 )); then
      continue
    fi
    if [ -n "$p_hosts_str" ]; then
      p_hosts_str+=","
    fi
    p_hosts_str+="${host_array[i]}"
  done

  if [[ "$USE_PD" -eq 1 ]]; then
    # 分配给D实例
    for (( i=0; i<d_required_ips; i++ )); do
      if (( i % d_select_ips != 0 )); then
        continue
      fi
      if [ -n "$d_hosts_str" ]; then
        d_hosts_str+=","
      fi
      d_hosts_str+="${host_array[i+p_required_ips]}"
    done
  fi

  # 格式: P_IP_LIST;D_IP_LIST
  export PD_HOSTS_STR="${p_hosts_str};${d_hosts_str}"
  log_info "PD_HOSTS_STR: ${PD_HOSTS_STR}"
  log_info "========================allocate_pd_hosts begin======================="
}

function clean_old_files()
{
  if [[ ! -d "${infer_dir}/conf_for_train" ]]; then
    return
  fi

  rm -f ${infer_dir}/conf_for_train/config_done

  rm -f ${infer_dir}/conf_for_train/prefill_server_list
  rm -f ${infer_dir}/conf_for_train/decode_server_list
  rm -f ${infer_dir}/conf_for_train/tensor_parallel_size
  rm -f ${infer_dir}/conf_for_train/data_parallel_size
  rm -f ${infer_dir}/conf_for_train/enable_expert_parallel
  rm -f ${infer_dir}/conf_for_train/vllm_version
}

function write_infer_server_list()
{
  if [[ ! -d "${infer_dir}/conf_for_train" ]]; then
    mkdir -p ${infer_dir}/conf_for_train
  fi

  # prefill和decode实例地址转替换到配置文件中, 权重更新的时候需要用到该配置
  PREFILL_SERVER_LIST=$(printf '"http://%s:20012",' "${P_INSTANCE_HOSTS_ARRAY[@]}" | sed 's/,$//')
  DECODE_SERVER_LIST=$(printf '"http://%s:20012",' "${D_INSTANCE_HOSTS_ARRAY[@]}" | sed 's/,$//')  # 去掉末尾逗号

  # 如果是PD混部, 则把DECODE_SERVER_LIST置空
  if [ ${USE_PD} -eq 0 ]; then
    DECODE_SERVER_LIST=""
  fi

  echo ${PREFILL_SERVER_LIST} > ${infer_dir}/conf_for_train/prefill_server_list
  echo ${DECODE_SERVER_LIST} > ${infer_dir}/conf_for_train/decode_server_list
}

function write_infer_parallel_size()
{
  # 受限当前权重更新策略影响, p和d实例的tp和dp参数需配置一致
  echo ${PREFILL_TENSOR_PARALLEL_SIZE} > ${infer_dir}/conf_for_train/tensor_parallel_size
  echo ${PREFILL_DATA_PARALLEL_SIZE} > ${infer_dir}/conf_for_train/data_parallel_size
  echo ${ENABLE_EXPERT_PARALLEL} > ${infer_dir}/conf_for_train/enable_expert_parallel
  echo ${VLLM_VERSION} > ${infer_dir}/conf_for_train/vllm_version

  echo "done" > ${infer_dir}/conf_for_train/config_done
}

function get_infer_configs_for_shared_filesystem()
{
  # 共享存储配置仅一份,训练节点不需要写入，只读就可以了
  if [[ ${VC_TASK_INDEX} -ge ${MASTER_TRAIN_INDEX} ]]; then
      return
  fi

  # 兼容续训脚本 第一个推理节点写推理配置和清理配置
  if [ -z "$START_RESUME_FLAG" ] || [ "$START_RESUME_FLAG" = "false" ]; then
    log_info "START_RESUME_FLAG: $START_RESUME_FLAG | clean old infer config files ..."
    if [[ ${VC_TASK_INDEX} -eq 0 ]]; then
      clean_old_files
    fi
  fi

  log_info "start infer instances begin..."
  allocate_pd_hosts
  if [[ -z "${PD_HOSTS_STR}" ]]; then
    log_error "start infer instances failed!!!"
    exit 1
  fi
  IFS=';' read -r p_hosts_result d_hosts_result <<< "${PD_HOSTS_STR}"
  IFS=',' read -r -a P_INSTANCE_HOSTS_ARRAY <<< "$p_hosts_result"
  IFS=',' read -r -a D_INSTANCE_HOSTS_ARRAY <<< "$d_hosts_result"

  # 如果是PD混部, 则把D_INSTANCE_HOSTS_ARRAY置空
  if [ ${USE_PD} -eq 0 ]; then
    D_INSTANCE_HOSTS_ARRAY=()
  fi

  log_info "USE_PD=${USE_PD}"
  log_info "P_INSTANCE_HOSTS_ARRAY=${P_INSTANCE_HOSTS_ARRAY[*]}"
  log_info "D_INSTANCE_HOSTS_ARRAY=${D_INSTANCE_HOSTS_ARRAY[*]}"

  # 第一个推理节点去写PD Server配置, 给训练端使用, 通过写文件方式保证训练和推理脚本解耦
  if [[ ${VC_TASK_INDEX} -eq 0 ]]; then
    write_infer_server_list
    write_infer_parallel_size
  fi
}

function get_infer_configs_for_non_shared_filesystem()
{
  clean_old_files
  allocate_pd_hosts
  if [[ -z "${PD_HOSTS_STR}" ]]; then
    log_error "get_infer_configs_for_non_shared_filesystem failed!!! PD_HOSTS_STR is empty"
    exit 1
  fi
  IFS=';' read -r p_hosts_result d_hosts_result <<< "${PD_HOSTS_STR}"
  IFS=',' read -r -a P_INSTANCE_HOSTS_ARRAY <<< "$p_hosts_result"
  IFS=',' read -r -a D_INSTANCE_HOSTS_ARRAY <<< "$d_hosts_result"

  # 如果是PD混部, 则把D_INSTANCE_HOSTS_ARRAY置空
  if [ ${USE_PD} -eq 0 ]; then
    D_INSTANCE_HOSTS_ARRAY=()
  fi

  log_info "USE_PD=${USE_PD}"
  log_info "P_INSTANCE_HOSTS_ARRAY=${P_INSTANCE_HOSTS_ARRAY[*]}"
  log_info "D_INSTANCE_HOSTS_ARRAY=${D_INSTANCE_HOSTS_ARRAY[*]}"

  # 每个节点各自写各自的配置
  write_infer_server_list
  write_infer_parallel_size
}

function get_infer_configs()
{
  if [[ ${IS_SHARED_FILESYSTEM} -eq 1 ]]; then
    get_infer_configs_for_shared_filesystem
  else
    get_infer_configs_for_non_shared_filesystem
  fi
}
