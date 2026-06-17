#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# 训练任务启动入口

train_dir=$(realpath $(dirname $0))
scripts_dir=$(realpath $(dirname ${train_dir}))
root_dir=$(realpath $(dirname ${scripts_dir}))

source ${root_dir}/scripts/base/envs.sh
source ${root_dir}/scripts/base/utils.sh

export HCCL_BUFFSIZE="200" #默认大小
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
if [[ "${WORK_MODE}" == "hybrid" ]]; then
  # export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64
  export HCCL_HOST_SOCKET_PORT_RANGE="auto"
  export HCCL_NPU_SOCKET_PORT_RANGE="auto"
fi

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

export GLOO_SOCKET_IFNAME=${DEFAULT_SOCKET_IFNAME:-"eth0"}
export TP_SOCKET_IFNAME=${DEFAULT_SOCKET_IFNAME:-"eth0"}

export PYTHONPATH=/MindSpeed:/Megatron-Bridge/src:${RLLM_PATH}:${PYTHONPATH}

export TRITON_DISABLE_AUTOTUNE=1

export VC_TASK_INDEX=${VC_TASK_INDEX:-$1}
export USE_PD=0 # 训练端的推理是个假的推理, 默认不开PD分离

[ "$CARD_TYPE" = "${A3_CARD}" ] && NPU_RESOURCES=16 || NPU_RESOURCES=8
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES}" ]]; then
  IFS=',' read -ra ids <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  NPU_RESOURCES=${#ids[@]}
fi
log_info "NPU_RESOURCES: ${NPU_RESOURCES}"

function start_ray_master()
{
  log_info "********** train master-$VC_TASK_INDEX starts, card type: $CARD_TYPE **********"
  if [[ "$CARD_TYPE" -eq "${A3_CARD}" ]]; then
    # A3机器需要指定cpu数量, 否则CPU资源不够会抛异常
    ray start --head --port 6344 --num-cpus 192 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources="{\"NPU\": ${NPU_RESOURCES}}"
  else
    ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources="{\"NPU\": ${NPU_RESOURCES}}"
  fi
  sleep 30
}

function start_ray_worker()
{
  log_info "********** train work-$VC_TASK_INDEX starts, card type: $CARD_TYPE **********"
  log_info "$MASTER_TRAIN_HOST:6344"
  sleep 30

  if [[ "$CARD_TYPE" -eq "${A3_CARD}" ]]; then
    # A3机器需要指定cpu数量, 否则CPU资源不够会抛异常
    ray start --address="$MASTER_TRAIN_HOST:6344" --num-cpus 192 --resources="{\"NPU\": ${NPU_RESOURCES}}"
  else
    ray start --address="$MASTER_TRAIN_HOST:6344" --resources="{\"NPU\": ${NPU_RESOURCES}}"
  fi

  # 非master节点循环检查ray集群状态
  while true; do
    ray status > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
    sleep 30
  done
  exit 1
}

function start_ray_cluster()
{
  if [[ $VC_TASK_INDEX -eq $MASTER_TRAIN_INDEX ]]; then
    start_ray_master
  elif [[ $VC_TASK_INDEX -gt $MASTER_TRAIN_INDEX ]]; then
    start_ray_worker
  fi
}

function get_infer_server_config()
{
  # 如果训练先启动, 需要等待配置文件的生成
  config_done_file="${scripts_dir}/infer/conf_for_train/config_done"
  while [ ! -f "${config_done_file}" ]; do
    log_warn "external vllm cluster is not ready, waiting 5 seconds..."
    sleep 5
  done

  # 提供给修改配置使用
  PREFILL_SERVER_LIST=$(cat ${scripts_dir}/infer/conf_for_train/prefill_server_list)
  DECODE_SERVER_LIST=$(cat ${scripts_dir}/infer/conf_for_train/decode_server_list)

  # 提供给数组解析使用
  PREFILL_SERVER_LIST_FOR_ARRAY=$(cat ${scripts_dir}/infer/conf_for_train/prefill_server_list | tr -d '"')
  DECODE_SERVER_LIST_FOR_ARRAY=$(cat ${scripts_dir}/infer/conf_for_train/decode_server_list | tr -d '"')

  # 设置 IFS 为逗号，将字符串转换为数组
  IFS=',' read -r -a PREFILL_ARRAY <<< "$PREFILL_SERVER_LIST_FOR_ARRAY"
  IFS=',' read -r -a DECODE_ARRAY <<< "$DECODE_SERVER_LIST_FOR_ARRAY"

  # 获取tp和dp参数
  TENSOR_PARALLEL_SIZE=$(cat ${scripts_dir}/infer/conf_for_train/tensor_parallel_size)
  DATA_PARALLEL_SIZE=$(cat ${scripts_dir}/infer/conf_for_train/data_parallel_size)
  ENABLE_EXPERT_PARALLEL=$(cat ${scripts_dir}/infer/conf_for_train/enable_expert_parallel)
}

function replace_infer_server_config()
{
  # 修改训练的yaml配置 (仅修改主节点配置，避免冲突)
  if [ "$VC_TASK_INDEX" = "$MASTER_TRAIN_INDEX" ]; then
    sed -e "s|chat_server:.*|chat_server: \"http://${MASTER_ROLLOUT_HOST}:8080\"|" \
        -e "s|prefill_server_list:.*|prefill_server_list: [${PREFILL_SERVER_LIST}]|" \
        -e "s|decode_server_list:.*|decode_server_list: [${DECODE_SERVER_LIST}]|" \
        -e "s|\btensor_parallel_size:.*|tensor_parallel_size: ${TENSOR_PARALLEL_SIZE}|" \
        -e "s|\bdata_parallel_size:.*|data_parallel_size: ${DATA_PARALLEL_SIZE}|" \
        -e "s|enable_expert_parallel:.*|enable_expert_parallel: ${ENABLE_EXPERT_PARALLEL}|" \
        ${root_dir}/configs/train/${CONFIG_NAME}.yaml > tmp.yaml
    cp -f tmp.yaml ${root_dir}/configs/train/${CONFIG_NAME}.yaml
    rm -f tmp.yaml
  fi
}


function regitster_sandbox_infer_model() {
    if [[ $VC_TASK_INDEX -ne $MASTER_TRAIN_INDEX ]]; then
       return
    fi
    local yaml_file=${root_dir}/configs/train/${CONFIG_NAME}.yaml

    # 1. 提取核心配置（并生成临时的 run_id）
    # 修改点：使用更直接的 get(0) 逻辑获取列表元素，并确保 run_id 打印
    local config_info=$(python3 - <<EOF
import yaml
import sys
import os
import uuid

try:
    with open('$yaml_file', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 提取 agent 实例
    agents = config.get('agent_instances', [])
    if not agents or len(agents) == 0:
        print("ERROR: agent_instances is empty", file=sys.stderr)
        sys.exit(0)

    agent_kwargs = agents[0].get('executor_kwargs', {})
    p_url = str(agent_kwargs.get('traj_proxy_url', '')).strip().rstrip('/')
    r_id = str(agent_kwargs.get('traj_proxy_run_id', '')).strip()

    # 如果 r_id 为空或 None
    if not r_id or r_id.lower() == 'none' or r_id == '':
        pid_hex = hex(os.getpid())[2:]
        r_id = f"{pid_hex}_{uuid.uuid4().hex[:12]}"
        print(f"NEED_UPDATE|{p_url}|{r_id}")
    else:
        print(f"READY|{p_url}|{r_id}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

    status=$(echo "$config_info" | cut -d'|' -f1)
    p_url=$(echo "$config_info" | cut -d'|' -f2)
    r_id=$(echo "$config_info" | cut -d'|' -f3)
    if [ "${p_url}" == "" ]; then
      return
    fi

    # 2. 如果需要更新，使用更强力的 sed 写入文件
    if [ "$status" == "NEED_UPDATE" ]; then
        # 兼容 traj_proxy_run_id: 或 traj_proxy_run_id: ""
        sed -i "/traj_proxy_run_id:/d" "$yaml_file"
        sed -i "/traj_proxy_url:/a \      traj_proxy_run_id: $r_id" "$yaml_file"
    fi

    # 3. 再次调用 Python 获取完整的配置（此时文件已更新）
    model_configs=$(python3 - <<EOF
import yaml
import sys
try:
    with open('$yaml_file', 'r') as f:
        config = yaml.safe_load(f)

    agent_kwargs = config.get('agent_instances', [])[0].get('executor_kwargs', {})
    proxy_url = str(agent_kwargs.get('traj_proxy_url', '')).strip().rstrip('/')
    run_id = str(agent_kwargs.get('traj_proxy_run_id', '')).strip()

    for instance in config.get('infer_instances', []):
        m_name = instance.get('executor_kwargs', {}).get('engine_kwargs', {}).get('model_name', '')
        if m_name and proxy_url and run_id:
            print(f"{m_name}|{proxy_url}|{run_id}")
except:
    sys.exit(1)
EOF
)
    echo "$model_configs" | while IFS='|' read -r m_name p_url r_id; do
        m_url="http://${MASTER_ROLLOUT_HOST}:8080/v1"
        curl -s -X DELETE "${p_url}/models?model_name=${m_name}&run_id=${r_id}"
        echo "register new model to TrajProxy:  ${p_url}/models/register model_name：${m_name} url:${m_url}"
        curl -s -X POST "${p_url}/models/register" \
          -H "Content-Type: application/json" \
          -d "{\"model_name\": \"$m_name\", \"url\": \"$m_url\", \"run_id\": \"$r_id\", \"api_key\": \"sk-1234\", \"token_in_token_out\": false}"
        echo -e "\n--------------------------"
    done
}

function check_pd_server_ready()
{
  INTERVAL=5

  # ADDR格式为: "http://ip:port"
  ADDR=$1
  MAX_WAIT=$2
  URL="${ADDR}/metrics"
  for ((i=0; i<MAX_WAIT; i+=INTERVAL)); do
    # 使用 curl 获取 HTTP 状态码
    log_info "begin to curl ${ADDR}"
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$URL")
    if [ "$STATUS_CODE" -eq 200 ]; then
        log_info "server $ADDR is ready, status code: $STATUS_CODE"
        return 0
    fi
    log_warn "server $ADDR not ready, status code: $STATUS_CODE, waiting $INTERVAL seconds..."
    sleep $INTERVAL
  done
  return 1
}

function wait_for_infer_cluster_ready()
{
  is_ready="false"
  while [[ "${is_ready}" == "false" ]]; do
    get_infer_server_config
    check_pd_server_ready ${PREFILL_ARRAY[0]} 1
    if [[ $? -eq 1 ]]; then
      log_error "waiting infer cluster (prefill instance ${PREFILL_ARRAY[0]}) ready failed!!!"
      is_ready="false"
      sleep 10
    else
      if (( ${#DECODE_ARRAY[@]} == 0 )); then
        is_ready="true"
        break
      fi

      for addr in "${DECODE_ARRAY[@]}"; do
          check_pd_server_ready "$addr" 100
          if [[ $? -eq 1 ]]; then
            log_error "waiting infer cluster (decode instance $addr) ready failed!!!"
            is_ready="false"
            break
          else
            is_ready="true"
          fi
      done
    fi
  done
  log_info "wait infer ready end"

  # 推理集群启动成功, 需要替换训练侧的配置
  replace_infer_server_config
}

function start_rollout_and_train()
{
  sleep 1m
  ray status
  timestamp=$(date +"%Y%m%d_%H%M%S")
  log_info "start rollout and train process, current_time: $timestamp"
  # MASTER_TRAIN_INDEX worker启动管理进程
  if [ "$VC_TASK_INDEX" = "$MASTER_TRAIN_INDEX" ]; then
    log_info "********** work-$MASTER_TRAIN_INDEX training **********"
    sleep 10
    ray status
    cd ${root_dir}/
    python aura/start.py --config-name=${CONFIG_NAME} 2>&1 | tee ${LOG_PATH}/train_unit_${timestamp}.log
    # 结束ray集群
    python_exit_code=${PIPESTATUS[0]}
    if [[ "${python_exit_code}" -eq 0 ]]; then
      ray stop
    else
      exit ${python_exit_code}
    fi
  fi
}


function set_hccl_timeout()
{
  export HCCL_CONNECT_TIMEOUT=1800
  export HCCL_EXEC_TIMEOUT=1800
}

function patch_verl()
{
  if [[ "$CONFIG_NAME" == *"megatron"* ]]; then
    sed -i 's/\${model_engine}/megatron/g' /verl/verl/trainer/config/ppo_trainer.yaml
    export WEIGHT_SAVE_STRATEGY="megatron"
  else
    sed -i 's/\${model_engine}/dp/g' /verl/verl/trainer/config/ppo_trainer.yaml
    sed -i 's/logits_rmpad\.div_/logits_rmpad = logits_rmpad.div/g' /verl/verl/workers/engine/fsdp/transformer_impl.py
    export WEIGHT_SAVE_STRATEGY="fsdp"
  fi
}

function disable_compile()
{
  if [[ "$CONFIG_NAME" == *"megatron"* ]]; then
    export TORCH_COMPILE_DISABLE=1
  fi
}

function patch_megatron_bridge()
{
  if [[ "$CONFIG_NAME" != *"megatron"* ]]; then
    return 0
  fi

  # npu上megatron-bridge暂时不支持peft权重保存，需要先注释掉
  local megatron_path=$(pip show megatron-bridge 2>/dev/null | sed -n 's/^Location: [[:space:]]*//p')
  if [ -z "${megatron_path}" ]; then
    log_info "megatron-bridge does not install"
    return 0
  fi

  local peft_bridge_file=${megatron_path}/megatron/bridge/models/conversion/peft_bridge.py
  if [ ! -f "${peft_bridge_file}" ]; then
    log_info "${peft_bridge_file} does not exist"
    return 0
  fi

  # 注释掉peft相关的代码
  sed -i '/from megatron.bridge.peft.canonical_lora import ModuleDict/s/^/# /' ${peft_bridge_file}
  sed -i '/from megatron.bridge.peft.lora import LoRAMerge/s/^/# /' ${peft_bridge_file}
  sed -i '383,385s/^/#/' ${peft_bridge_file}
  sed -i '505,506s/^/#/' ${peft_bridge_file}
  sed -i '833,847s/^/#/' ${peft_bridge_file}
  sed -i '832a \        return base_weight' ${peft_bridge_file}
}

###################################################################################
ray stop

set_hccl_timeout
patch_verl
patch_megatron_bridge
disable_compile

log_info "[train] ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES}"

regitster_sandbox_infer_model

if [[ "${WORK_MODE}" == "one_step_off" ]]; then
  # 训推全异步分离场景, 需要等待推理集群启动完成
  wait_for_infer_cluster_ready
fi

start_ray_cluster
start_rollout_and_train

# 训练正常结束, 通知主进程退出
kill -USR1 $PPID
