#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# RL（GRPO等）任务启动入口

scripts_dir=$(realpath $(dirname $0))
root_dir=$(realpath $(dirname $scripts_dir))

echo "=========set cann env================"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source ${scripts_dir}/base/utils.sh

parse_train_conf

if [[ "${WORK_MODE}" == "hybrid" ]]; then
  # 共卡模式, MASTER_TRAIN_INDEX默认为0
  export MASTER_TRAIN_INDEX=0
fi

export RL_TRAIN_BACKEND="verl"
export ACLNN_ALLOW_RUNTIME_CACHE=1

# 配置使用的是VC_TASK_HOSTS
# 云道配置使用的是VC_WORKER_HOSTS
# 本地调测使用hosts文件配置
config_vc_hosts

log_info "============================================================"
log_info "VC_TASK_HOSTS: ${VC_TASK_HOSTS}"
log_info "VC_WORKER_HOSTS: ${VC_WORKER_HOSTS}"
log_info "VC_TASK_INDEX: ${VC_TASK_INDEX}"
log_info "MASTER_TRAIN_INDEX: ${MASTER_TRAIN_INDEX}"
log_info "MASTER_INFER_INDEX: ${MASTER_INFER_INDEX}"
log_info "RL_TRAIN_BACKEND: ${RL_TRAIN_BACKEND}"
log_info "============================================================"

check_env

# 集群IP, 如果是集群域名, 需要转换成IP, 并配置集群相关信息(训练和推理集群IP等)
# 是配置VC_TASK_HOSTS, 云道是配置VC_WORKER_HOSTS
prepare_cluster_info

log_info "after parsing domain, VC_WORKER_HOSTS: ${VC_WORKER_HOSTS}"

rm -rf /tmp/ray/*

function start_train()
{
  # 训练节点启动训练集群, train进程, rollout进程
  # 如果是断点续训, 则使用RESUME_TRAIN_CONF_NAME配置文件名称
  if [[ -n "${RESUME_TRAIN_CONF_NAME}" ]]; then
    real_train_conf_name=${RESUME_TRAIN_CONF_NAME}
  else
    real_train_conf_name=${TRAIN_CONF_NAME}
  fi

  source ${scripts_dir}/infer/vllm/parse_infer_config.sh
  get_infer_configs

  log_info "start verl train cluster, work_mode: ${WORK_MODE}, config name: ${real_train_conf_name}"
  log_info "start mode: $1"
  if [[ "$1" == "daemon" ]]; then
    (
      bash ${scripts_dir}/train/start_verl_train_cluster.sh --config-name ${real_train_conf_name} \
        | sed "s/^/[train_cluster] /"
      # 子脚本退出码作为子 shell 的退出码
      exit ${PIPESTATUS[0]}
    ) &
    pid=$!
    wait $pid
    exit_code=$?
  else
    bash ${scripts_dir}/train/start_verl_train_cluster.sh --config-name ${real_train_conf_name} | sed "s/^/[train_cluster] /"
    exit_code=${PIPESTATUS[0]}
  fi
  log_info "start_verl_train_cluster.sh end with exit code $exit_code"
  exit $exit_code
}

function start_infer()
{
  if [[ "${WORK_MODE}" == "hybrid" ]]; then
    # 常规配置, 共卡模式的MASTER_TRAIN_INDEX是从0开始, 不会预留机器给独立的推理集群部署
    # 但是预防异常的配置, 这里判断共卡模式直接退出
    log_warn "hybrid mode, skip start external vllm infer cluster"
    return
  fi
  # 推理节点启动外挂模式的推理集群, 在后台执行, 和训练并行启动
  log_info "start external vllm infer cluster"
  # 如果是断点续训, 则使用RESUME_INFER_CONF_NAME配置文件名称
  if [[ -n "${RESUME_INFER_CONF_NAME}" ]]; then
    real_infer_conf_name=${RESUME_INFER_CONF_NAME}
  else
    real_infer_conf_name=${INFER_CONF_NAME}
  fi
  if [[ "$1" == "daemon" ]]; then
    (
      bash ${scripts_dir}/infer/start_vllm_infer_cluster.sh --config-name ${real_infer_conf_name} \
        | sed "s/^/[infer_cluster] /"
      # 子脚本退出码作为子 shell 的退出码
      exit ${PIPESTATUS[0]}
    ) &
    pid=$!
    wait $pid
    exit_code=$?
  else
    bash ${scripts_dir}/infer/start_vllm_infer_cluster.sh --config-name ${real_infer_conf_name} | sed "s/^/[infer_cluster] /"
    exit_code=${PIPESTATUS[0]}
  fi
  log_info "start_vllm_infer_cluster.sh end with exit code $exit_code"
  exit $exit_code
}

NODE_TYPE="NULL"
function get_node_type()
{
  if [[ "${VC_TASK_INDEX}" -lt "${MASTER_TRAIN_INDEX}" && -z "${MASTER_INFER_INDEX}" ]]; then
    NODE_TYPE="infer"
  elif [[ "${VC_TASK_INDEX}" -ge "${MASTER_TRAIN_INDEX}" && -z "${MASTER_INFER_INDEX}" ]]; then
    NODE_TYPE="train"
  elif [[ "${VC_TASK_INDEX}" -eq "${MASTER_TRAIN_INDEX}" &&
    -n "${MASTER_INFER_INDEX}" &&
    "${VC_TASK_INDEX}" -eq "${MASTER_INFER_INDEX}" ]]; then
    NODE_TYPE="hybrid"
  fi
}

function set_infer_visible_devices()
{
  # 训推共节点部署场景, 默认前面的卡是给训练的, 后面的卡给推理
  yaml_file="${root_dir}/configs/infer/${INFER_CONF_NAME}.yaml"
  local num_npus=$(python3 -c "import yaml; \
    print(yaml.safe_load(open('${yaml_file}'))['tensor_parallel_size'])")

  # 提取 Total Count (对应 NPU 逻辑卡数)
  local total_count=$(npu-smi info -l | grep "Total Count" | awk -F: '{print $2}' | tr -d ' ')
  # 提取 Chip Count (由于每张卡结构一致，我们取第一个 NPU 的 Chip Count)
  local chip_count=$(npu-smi info -l | grep "Chip Count" | head -n 1 | awk -F: '{print $2}' | tr -d ' ')
  # 计算实际的总芯片（卡）数
  local total_cards=$((total_count * chip_count))

  export ASCEND_RT_VISIBLE_DEVICES=$(seq 0 $((total_cards - 1)) | tail -n "${num_npus}" | paste -sd "," -)
  log_info "for infer, ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES}"
}

# 定义清理函数, 主进程ctrl+c, 回收子进程
function cleanup()
{
  # 杀掉当前进程组中的所有后台进程
  echo "receive $1, stopping child processes..."
  ray stop
  pgid=$(ps -o pgid= -p $$)
  pgrep -g $pgid | grep -vE "^ *($pgid|$$) *$" | xargs -r kill -TERM 2>/dev/null
  if [[ "$1" == "SIGUSR1" ]]; then
    exit 0
  fi
}

trap 'cleanup SIGINT' SIGINT
trap 'cleanup SIGTERM' SIGTERM
trap 'cleanup SIGUSR1' SIGUSR1

get_node_type
log_info "NODE_TYPE: ${NODE_TYPE}"
if [[ "${NODE_TYPE}" == "infer" ]]; then
  start_infer non_daemon
elif [[ "${NODE_TYPE}" == "train" ]]; then
  start_train non_daemon
elif [[ "${NODE_TYPE}" == "hybrid" ]]; then
  # 捕获SIGINT(Ctrl+C), SIGTERM, SIGUSR1信号, 触发cleanup
  # 训推共节点部署场景, 进程在后台运行, 需要主进程回收
  # 训推单节点共部署场景, 主要用于功能调测
  # 共节点部署场景, 推理需要指定可见的卡数, 从后面的卡启动推理
  set_infer_visible_devices
  start_infer daemon
  # 训练不需要指定可看见的卡, 训练默认需要从0开始读取
  unset ASCEND_RT_VISIBLE_DEVICES
  start_train daemon

  wait
else
  log_error "unknown node type: ${NODE_TYPE}"
fi
