#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# 公共函数

base_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))
scripts_dir=$(realpath $(dirname ${base_dir}))
root_dir=$(realpath $(dirname ${scripts_dir}))

source ${scripts_dir}/base/color.sh

function log_info()
{
    echo -e "${COLOR_GREEN}[INFO] [" $(date +"%y-%m-%d %H:%M:%S") "] $1${COLOR_RESET}"
}

function log_error()
{
    echo -e "${COLOR_RED}[ERROR] [" $(date +"%y-%m-%d %H:%M:%S") "] $1${COLOR_RESET}"
}

function log_warn()
{
    echo -e "${COLOR_YELLOW}[WARNING] [" $(date +"%y-%m-%d %H:%M:%S") "] $1${COLOR_RESET}"
}

function translate_domain_to_ips()
{
  TASK_HOSTS_DOMAINS=${VC_WORKER_HOSTS//,/ }
  first_element=$(echo "$TASK_HOSTS_DOMAINS" | awk -F' ' '{print $1}')
  if [[ "$first_element" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    NODE_IP_LIST=${VC_WORKER_HOSTS}
  else
    NODE_IP_LIST=""
    log_info "   - change domain to ip list (NODE_IP_LIST)..."
    for domain in ${TASK_HOSTS_DOMAINS}; do
        IP_ADDRESS=$(python -c "import socket; print(socket.gethostbyname('$domain'))")
        if [ $? -ne 0 ] || [ -z "$IP_ADDRESS" ]; then
            log_error "can not parse domain: $domain"
            exit 1
        fi
        if [ "X${NODE_IP_LIST}" == "X" ]; then
          NODE_IP_LIST="${IP_ADDRESS}"
        else
          NODE_IP_LIST="${NODE_IP_LIST},${IP_ADDRESS}"
        fi
    done
  fi

  export VC_WORKER_HOSTS=${NODE_IP_LIST}
}

function get_this_pod_ip()
{
  export THIS_POD_IP="$(echo "$VC_WORKER_HOSTS" | cut -d ',' -f $((VC_TASK_INDEX + 1)))"
}

function get_master_hosts()
{
  export MASTER_ROLLOUT_HOST="${VC_WORKER_HOSTS%%,*}"
  export MASTER_TRAIN_HOST="$(echo "$VC_WORKER_HOSTS" | cut -d ',' -f $((MASTER_TRAIN_INDEX + 1)))"
}

function replace_domain_task_hosts_to_ips()
{
  export VC_TASK_HOSTS=$(echo "$VC_WORKER_HOSTS" | tr ',' ',' | cut -d ',' -f 1-$(( MASTER_TRAIN_INDEX > 0 ? MASTER_TRAIN_INDEX : 1)))
}

function get_is_config_shared_filesystem()
{
  # 是配置VC_TASK_HOSTS, 云道是配置VC_WORKER_HOSTS
  if [[ -n "${VC_TASK_HOSTS}" ]]; then
    export VC_WORKER_HOSTS=${VC_TASK_HOSTS}
    export IS_SHARED_CONF=1
  else
    # 云道环境非共享配置文件路径
    export IS_SHARED_CONF=0
  fi
}

function prepare_cluster_info()
{
  get_is_config_shared_filesystem
  translate_domain_to_ips
  get_this_pod_ip
  get_master_hosts
  replace_domain_task_hosts_to_ips
}

function check_env()
{
  if [[ -z "${VC_TASK_HOSTS}" && -z "${VC_WORKER_HOSTS}" ]]; then
    log_error "You should exec \"export VC_TASK_HOSTS=XXX | export VC_WORKER_HOSTS=XXX\" to configure hosts "
    log_error "for infer and train, the infer hosts must be prioritized in the configuration."
    exit 1
  fi

  if [[ -z "${VC_TASK_INDEX}" ]]; then
    log_error "You should exec \"export VC_TASK_INDEX={0|1|2...}\" to configure the host index"
    exit 1
  fi

  if [[ -z "${MASTER_TRAIN_INDEX}" ]]; then
    log_error "You should exec \"export MASTER_TRAIN_INDEX={16|...}\" to configure the host index for the train master"
    exit 1
  fi
}

function config_vc_hosts()
{
  # 配置使用的是VC_TASK_HOSTS, 云道配置使用的是VC_WORKER_HOSTS
  if [[ -n "${VC_TASK_HOSTS}" || -n "${VC_WORKER_HOSTS}" ]]; then
    log_warn "use global \"VC_TASK_HOSTS | VC_WORKER_HOSTS\" configure"
    return
  fi

  # 本地调测使用hosts文件配置
  if [[ ! -f "${root_dir}/configs/hosts.conf" ]]; then
    log_error "config file hosts.conf is not exist"
    exit 1
  fi

  if [[ ! -s "${root_dir}/configs/hosts.conf" ]]; then
    log_error "config file hosts.conf exist, but the content is null"
    exit 1
  fi

  export DEFAULT_SOCKET_IFNAME=${DEFAULT_SOCKET_IFNAME:-"eth0"}
  # ifconfig输出的格式可能为"inet x.x.x.x"或"inet addr:x.x.x.x"
  local_ip=$(ifconfig ${DEFAULT_SOCKET_IFNAME} | grep 'inet ' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
  if [[ -z "$local_ip" ]]; then
      log_error "can not get IP from ${DEFAULT_SOCKET_IFNAME}"
      exit 1
  fi

  while IFS=',' read -r ip id train_master_id infer_master_id || [ -n "$ip" ]; do
    ip=$(echo "$ip" | tr -d '[:space:]')
    id=$(echo "$id" | tr -d '[:space:]')
    train_master_id=$(echo "$train_master_id" | tr -d '[:space:]')
    infer_master_id=$(echo "$infer_master_id" | tr -d '[:space:]')

    # 过滤空行（防止文件末尾有空行）
    [[ -z "$ip" ]] && continue

    if [[ -z "${VC_TASK_HOSTS}" ]]; then
      export VC_TASK_HOSTS=${ip}
    else
      export VC_TASK_HOSTS="$VC_TASK_HOSTS,$ip"
    fi
    if [[ "$local_ip" == "$ip" ]]; then
      export VC_TASK_INDEX=$id
    fi
    if [[ "$train_master_id" -eq 1 ]]; then
      export MASTER_TRAIN_INDEX=$id
    fi
    if [[ -n "$infer_master_id" && "$infer_master_id" -eq 1 ]]; then
      export MASTER_INFER_INDEX=$id
    fi
  done < <(grep -vE '^\s*#|^\s*$' ${root_dir}/configs/hosts.conf)

  if [[ -z "${VC_TASK_INDEX}" ]]; then
    log_error "no host in hosts.conf matches local IP (${local_ip}), please check configs/hosts.conf"
    exit 1
  fi
}

function get_conf_val()
{
    local key=$1
    local file=$2
    grep -vE '^[[:space:]]*#' "$file" | grep "^${key}=" | cut -d'=' -f2- | tr -d '\r' | xargs
}

function parse_train_conf()
{
  BASE_CONF="${root_dir}/configs/base.conf"
  if [[ ! -f "${BASE_CONF}" ]]; then
    log_error "config file ${BASE_CONF} is not exist"
    exit 1
  fi

  export WORK_MODE=$(get_conf_val "work_mode" "${BASE_CONF}")
  export TRAIN_CONF_NAME=$(get_conf_val "train_config_name" "${BASE_CONF}")
  export INFER_CONF_NAME=$(get_conf_val "infer_config_name" "${BASE_CONF}")
  if [[ -z "${WORK_MODE}" ]]; then
    log_error "get WORK_MODE from ${BASE_CONF} failed, please confirm the conf"
    exit 1
  fi

  case "${WORK_MODE}" in
    hybrid|one_step_off)
      ;;
    *)
      log_error "invalid WORK_MODE '${WORK_MODE}' in ${BASE_CONF}, supported values: hybrid, one_step_off"
      exit 1
      ;;
  esac
}

function parse_resume_conf()
{
  BASE_CONF="${root_dir}/configs/base.conf"
  if [[ ! -f "${BASE_CONF}" ]]; then
    log_error "config file ${BASE_CONF} is not exist"
    exit 1
  fi

  export MONITOR_CMD=$(get_conf_val "monitor_cmd" "${BASE_CONF}")
  if [[ -z "${MONITOR_CMD}" ]]; then
    log_error "get MONITOR_CMD from ${BASE_CONF} failed, please confirm the conf"
    exit 1
  fi

  export MAX_RETRIES=$(get_conf_val "max_retries" "${BASE_CONF}")
  if [[ -z "${MAX_RETRIES}" ]]; then
    log_warn "get MAX_RETRIES from ${BASE_CONF} failed, set to default 100"
    export MAX_RETRIES=100
  fi

  export CLEAN_OLD_CKPT=$(get_conf_val "clean_old_ckpt" "${BASE_CONF}")
  if [[ -z "${CLEAN_OLD_CKPT}" ]]; then
    log_warn "get CLEAN_OLD_CKPT from ${BASE_CONF} failed, set to default 0"
    export CLEAN_OLD_CKPT=0
  fi
}

function parse_base_conf()
{
  parse_train_conf
  parse_resume_conf
}

export A2_CARD=0
export A3_CARD=1
export UNKNOWN_CARD=255

function set_card_type()
{
  a3_card_num=$(npu-smi info | grep "Ascend910" | awk '{print $2}' | wc -l)
  a2_card_num=$(npu-smi info | grep "910B" | awk '{print $2}' | wc -l)
  if [[ "$a3_card_num" -eq 16 ]]; then
    export CARD_TYPE=${A3_CARD}
  elif [[ "$a2_card_num" -eq 8 ]]; then
    export CARD_TYPE=${A2_CARD}
  else
    export CARD_TYPE=${UNKNOWN_CARD}
  fi
}

set_card_type

function get_actor_conf_from_yaml()
{
  local yaml_file=$(echo "$1" | tr -d '\r\n\t ')
  local key=$(echo "$2" | tr -d '\r\n\t ')

  local target_key=$(
  awk '
    /^\s*actor_config:/ {flag=1; next}
    /^\s*[_a-z]+_config:/ {if(flag) exit}
    flag && /^[[:space:]]*'"$key"':/ {
        # 移除行首的key:和空格
        sub(/^[[:space:]]*'"$key"':[[:space:]]*/, "")
        # 移除行尾的注释（如果有）
        sub(/[[:space:]]*#.*$/, "")
        print
        exit
    }' "${yaml_file}"
  )
  target_key=$(echo "${target_key}" | tr -d '\r\n\t' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  echo "${target_key}"
}

function get_verl_conf_val() {
    local file=$1
    local key=$2
    # 先检查文件是否存在
    if [ ! -f "$file" ]; then
        echo "Error: config file $file does not exist."
        exit 1
    fi
    # 用 grep 查找对应的 key 行
    line=$(grep -E "^[[:space:]]*${key}:" "$file")
    if [ -z "$line" ]; then
        echo "Error: key '${key}' not found in $file"
        exit 1
    fi
    # 提取值
    val=$(echo "$line" \
        | sed -E "s/^[[:space:]]*${key}:[[:space:]]*//; s/[[:space:]]*#.*//" \
        | tr -d '\r' \
        | xargs)
    if [ -z "$val" ]; then
        echo "Error: key '${key}' has empty value in $file"
        exit 1
    fi
    echo "$val"
}

function replace_verl_conf_val() {
    local file=$1
    local key=$2
    local value=$3
    # 替换file中key对应的值，不保留原来的注释部分
    sed -i -E "s|^([[:space:]]*)${key}:[[:space:]]*.*|\1${key}: ${value}|" "$file"
    log_info "successfully replace key=$key value=$value in file=$file"
}

function grep_global_step_from_path()
{
    local path="$1"
    local global_step=""
    # 用正则匹配路径末尾的 resume_<数字>
    if [[ "$path" =~ global_step_([0-9]+)[[:space:]]*/?$ ]]; then
        global_step="${BASH_REMATCH[1]}"
    fi
    echo "$global_step"
}
