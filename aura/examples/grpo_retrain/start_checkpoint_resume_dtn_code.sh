#!/bin/bash
MAX_RETRIES=100
RETRY_COUNT=1

DEFAULT_START_SH="start_multi_node_agent_rl_template.sh"
DEFAULT_YAML="grpo_trainer_qwen3_235b_32node_dtn_code_64k"
DEFAULT_LOG_PATH="logs"
DEFAULT_CLEAR_CKPT="0"

START_SH=${1:-$DEFAULT_START_SH}
YAML=${2:-$DEFAULT_YAML}
LOG_PATH=${3:-$DEFAULT_LOG_PATH}
CLEAR_CKPT=${4:-$DEFAULT_CLEAR_CKPT}

mkdir -p "$LOG_PATH"

WORK_DIR=$(pwd)
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_PATH}/execute_log_work${VC_TASK_INDEX}_${CURRENT_TIME}.log

HOSTS="$VC_WORKER_HOSTS"
MASTER_HOST="${HOSTS%%,*}"

# 支持不同日志级别的函数
log_info() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [INFO] [WORK_$VC_TASK_INDEX] $message" | tee -a "${LOG_FILE}"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [ERROR] [WORK_$VC_TASK_INDEX] $message" | tee -a "${LOG_FILE}" >&2
}

log_warning() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [WARNING] [WORK_$VC_TASK_INDEX] $message" | tee -a "${LOG_FILE}"
}

clean_process() {
    log_info "terminating remaining processes ..."
    ray stop --force
    ps -ef | grep "python"| grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {};pkill -9 python; pkill -9 torchrun;
    ps -ef | grep "defunct"|grep python| awk '{print $3}'|xargs -t -i kill -9 {};ps -ef | grep "defunct"|grep torchrun| awk '{print $3}'|xargs -t -i kill -9 {}
}

get_yaml_value() {
    local yaml_file=$(echo "$1" | tr -d '\r\n\t ')
    local key=$(echo "$2" | tr -d '\r\n\t ')

    # 检查文件是否存在
    if [ ! -f "$yaml_file" ]; then
        log_error "$yaml_file file not exist"
        return 1
    fi

    local target_key=$(
    awk '
        /^actor_config:/ {flag=1; next}
        /^[_a-z]+_config:/ {if(flag) exit}
        flag && /^[[:space:]]*'"$key"':/ {
            # 移除行首的key:和空格
            sub(/^[[:space:]]*'"$key"':[[:space:]]*/, "")
            # 移除行尾的注释（如果有）
            sub(/[[:space:]]*#.*$/, "")
            print
            exit
        }' "$yaml_file"
    )
    target_key=$(echo "${target_key}" | tr -d '\r\n\t' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    echo "${target_key}"
}

replace_yaml_value() {
    local load_value=$1
    local save_value=$2
    local yaml_file=$3

    log_info "[yaml_path]: ${yaml_file}"
    log_info "[load path]: ${load_value}"
    log_info "[save path]: ${save_value}"

    # 更新load值为save的值
    sed -i "/^actor_config:/,/^[_a-z]*_config:/ {s|^[[:space:]]*load:.*|  load: $save_value|}" "$yaml_file"
    # 更新finetune值为false的值
    sed -i "/^actor_config:/,/^[_a-z]*_config:/ {s|^[[:space:]]*finetune:.*|  finetune: false|}" "$yaml_file"

    if grep -q "integrated_mode_config:" "$yaml_file"; then
        sed -i "/^rl_config:/,/^[_a-z]*_config:/ {s|^[[:space:]]*ref_model_load_path:.*|    ref_model_load_path: $load_value|}" "$yaml_file"
    else
        sed -i "/rl_config:/a\  integrated_mode_config:\n    ref_model_load_path: $load_value" "$yaml_file"
    fi
}

cleanup_checkpoints() {
    # 清理检查点文件夹，只保留最新的
    local checkpoint_dir=$1

    # 检查目录是否存在
    if [ ! -d "$checkpoint_dir" ]; then
        log_error "dir $checkpoint_dir not exist"
        return 1
    fi

    # 检查latest_checkpointed_iteration.txt文件是否存在
    if [ ! -f "$checkpoint_dir/latest_checkpointed_iteration.txt" ]; then
        log_warning "[latest_checkpointed_iteration.txt] file not exist"
        return 1
    fi

    # 读取步数
    local step=$(cat "$checkpoint_dir/latest_checkpointed_iteration.txt" | tr -d ' \t\n\r')

    # 检查是否读取到有效步数
    if [ -z "$step" ]; then
        log_error "[latest_checkpointed_iteration.txt] read failed"
        return 1
    fi

    # 格式化步数为6位数字
    local formatted_step=$(printf "%07d" "$step")
    local keep_dir="$checkpoint_dir/iter_$formatted_step"

    log_info "target dir: $keep_dir"

    # 检查要保留的目录是否存在
    if [ ! -d "$keep_dir" ]; then
        log_error "$keep_dir not exist"
        return 1
    fi

    # 删除其他iter_开头的目录
    for dir in "$checkpoint_dir"/iter_*; do
        if [ -d "$dir" ] && [ "$dir" != "$keep_dir" ]; then
            log_warning "+++++++ remove dir: $dir"
            rm -rf "$dir"
        fi
    done

    log_info "clean dir complete ..."
}

do_main() {
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        log_info "executed [${START_SH}], times: $((RETRY_COUNT))"

        if [ $RETRY_COUNT -gt 1 ] && [ "$VC_TASK_INDEX" = "0" ]; then
#          clean_process
          local yaml_path="configs/${YAML}.yaml"
          local load_value=$(get_yaml_value "$yaml_path" "load")
          local save_value=$(get_yaml_value "$yaml_path" "save")

          if [ $RETRY_COUNT -eq 2 ] && [ -f "$save_value/latest_checkpointed_iteration.txt" ]; then
            # 第二次重新拉起后修改加载权重路径
            replace_yaml_value "$load_value" "$save_value" "$yaml_path"
            if [ $? -ne 0 ]; then
                break
            fi
          fi

          if [ "$CLEAR_CKPT" = '1' ] && [ -f "$save_value/latest_checkpointed_iteration.txt" ]; then
            # 执行检查点清理操作（假设检查点在$path目录下）
            log_info "starting to clean up the checkpoint folder ..."
            cleanup_checkpoints "$save_value"
          fi
        fi

        bash "examples/grpo_retrain/${START_SH}" $YAML $LOG_PATH

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log_info "script terminated successfully (code: $exit_code)"
            break
        else
            log_error "script abnormal exit (code: $exit_code)"
            clean_process
            RETRY_COUNT=$((RETRY_COUNT + 1))

            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
              # 主从节点restart差距最大应该在1分钟
              if [ "$VC_TASK_INDEX" = "0" ]; then
                log_warning "waiting for 5 min before restarting ..."
                sleep 5m
              else
                log_warning "waiting for 4 min before restarting ..."
                sleep 4m
              fi
            else
                log_info "maximum retries ($MAX_RETRIES) reached, stopping"
                exit 1
            fi
        fi
    done

    log_info "execute complete"
}

log_info "==================================================================================="
log_info "开始执行 breakpoint_reload_train.sh 脚本"
log_info "将监控 ${START_SH} 的执行状态，并在异常退出时自动重启"
log_info "[START_SH]: ${START_SH}"
log_info "[YAML]: ${YAML}"
log_info "[LOG_PATH]: ${LOG_PATH}"
log_info "[WORK_DIR]: ${WORK_DIR}"
log_info "==================================================================================="

mkdir -p "$LOG_PATH"

do_main
