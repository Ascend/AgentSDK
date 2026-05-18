#!/bin/bash
MAX_RETRIES=100
RETRY_COUNT=1

DEFAULT_START_SH="start_roma_vllm_proxy_pd_resume.sh" # 5.0 startup script (vllm plugin)
DEFAULT_YAML="direct_p1d1_qwen25_7b_train_one_step_off" # Modify the configuration file, not msrl_conf
DEFAULT_LOG_PATH="logs" # Modify the log path
DEFAULT_CLEAR_CKPT="0" # Enable clearing non-latest saveckpt during resuming training
DEFAULT_CLEAR_ALL_CKPT="0" # Enable clearing all saveckpt on first run
MASTER_TRAIN_INDEX=2 # Index of the main training node

START_SH=${1:-$DEFAULT_START_SH}
YAML=${2:-$DEFAULT_YAML}
RESUME_YAML="${YAML}_resume"
LOG_PATH=${3:-$DEFAULT_LOG_PATH}
CLEAR_CKPT=${4:-$DEFAULT_CLEAR_CKPT}
CLEAR_ALL_CKPT=${5:-$DEFAULT_CLEAR_ALL_CKPT}

mkdir -p "$LOG_PATH"

WORK_DIR=$(pwd)
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_PATH}/execute_log_work${VC_TASK_INDEX}_${CURRENT_TIME}.log

HOSTS="$VC_WORKER_HOSTS"
MASTER_HOST="${HOSTS%%,*}"

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

get_yaml_actor_value() {
    local yaml_file=$(echo "$1" | tr -d '\r\n\t ')
    local key=$(echo "$2" | tr -d '\r\n\t ')

    if [ ! -f "$yaml_file" ]; then
        log_error "$yaml_file file not exist"
        return 1
    fi

    local target_key=$(
    awk '
        /^\s*actor_config:/ {flag=1; next}
        /^\s*[_a-z]+_config:/ {if(flag) exit}
        flag && /^[[:space:]]*'"$key"':/ {
            sub(/^[[:space:]]*'"$key"':[[:space:]]*/, "")
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

    sed -i "/^\s*actor_config:/,/^\s*[_a-z]*_config:/ {s|^\([[:space:]]*\)load:.*|\1load: $save_value  #$load_value|}" "$yaml_file"
    if grep -q "finetune:" "$yaml_file"; then
        sed -i "/^\s*actor_config:/,/^\s*[_a-z]*_config:/ {s|^\([[:space:]]*\)finetune:.*|\1finetune: false  #true|}" "$yaml_file"
    else
        sed -i "/actor_config:/a\    finetune: false  #true" "$yaml_file"
    fi
    if grep -q "integrated_mode_config:" "$yaml_file"; then
        sed -i "/^  rl_config:/,/^  [_a-z]*_config:/ {s|^\([[:space:]]*\)ref_model_load_path:.*|\1ref_model_load_path: $load_value|}" "$yaml_file"
    else
        sed -i "/rl_config:/a\    integrated_mode_config:\n      ref_model_load_path: $load_value" "$yaml_file"
    fi
}

cleanup_checkpoints() {
    local checkpoint_dir=$1

    if [ ! -d "$checkpoint_dir" ]; then
        log_error "dir $checkpoint_dir not exist"
        return 1
    fi

    if [ ! -f "$checkpoint_dir/latest_checkpointed_iteration.txt" ]; then
        log_warning "[latest_checkpointed_iteration.txt] file not exist"
        return 1
    fi

    local step=$(cat "$checkpoint_dir/latest_checkpointed_iteration.txt" | tr -d ' \t\n\r')

    if [ -z "$step" ]; then
        log_error "[latest_checkpointed_iteration.txt] read failed"
        return 1
    fi

    local formatted_step=$(printf "%07d" "$step")
    local keep_dir="$checkpoint_dir/iter_$formatted_step"

    log_info "target dir: $keep_dir"

    if [ ! -d "$keep_dir" ]; then
        log_error "$keep_dir not exist"
        return 1
    fi

    for dir in "$checkpoint_dir"/iter_*; do
        if [ -d "$dir" ] && [ "$dir" != "$keep_dir" ]; then
            log_warning "+++++++ remove dir: $dir"
            rm -rf "$dir"
        fi
    done

    log_info "clean dir complete ..."
}

cleanup_all_checkpoints() {
    local save_checkpoint_dir="$1"

    if [ -z "$save_checkpoint_dir" ]; then
        log_info "Error: save_checkpoint_dir is not specified."
        return 1
    fi

    if [ ! -d "$save_checkpoint_dir" ]; then
        log_info "Error: $save_checkpoint_dir does not exist."
        return 1
    fi

    if [ ! -f "$checkpoint_dir/latest_checkpointed_iteration.txt" ]; then
        log_warning "[latest_checkpointed_iteration.txt] file not exist"
    fi

    log_info "Cleaning checkpoint directory: $save_checkpoint_dir"
    rm -rf "${save_checkpoint_dir:?}"/*
    log_info "Checkpoint directory cleaned."
}

is_yaml_replaced=0
do_main() {
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        log_info "executed [${START_SH}], times: $((RETRY_COUNT))"

        local yaml_path="configs/${YAML}.yaml"
        local resume_yaml_path="configs/${RESUME_YAML}.yaml"

        if [ "$VC_TASK_INDEX" = "$MASTER_TRAIN_INDEX" ]; then
            if [ $RETRY_COUNT -eq 1 ]; then
                if [ -e "$yaml_path" ]; then
                    touch $resume_yaml_path
                    \cp -f $yaml_path $resume_yaml_path
                else
                    log_error "配置路径不存在: $yaml_path"
                    return 1
                fi
                local load_value=$(get_yaml_actor_value "$resume_yaml_path" "load")
                local save_value=$(get_yaml_actor_value "$resume_yaml_path" "save")
                if [ "$CLEAR_ALL_CKPT" = '1' ]; then
                    log_info "starting to clean up the save checkpoint folder before training ..."
                    cleanup_all_checkpoints "$save_value"
                fi
            else
                if [ $is_yaml_replaced -eq 0 ] && [ -f "$save_value/latest_checkpointed_iteration.txt" ]; then
                    replace_yaml_value "$load_value" "$save_value" "$resume_yaml_path"
                    if [ $? -ne 0 ]; then
                        break
                    fi
                    is_yaml_replaced=1
                fi
            fi
            if [ "$CLEAR_CKPT" = '1' ] && [ -f "$save_value/latest_checkpointed_iteration.txt" ]; then
                log_info "starting to clean up the checkpoint folder ..."
                cleanup_checkpoints "$save_value"
            fi
        fi

        if [ $RETRY_COUNT -gt 1 ]; then
            if [ "$VC_TASK_INDEX" -gt "$MASTER_TRAIN_INDEX" ]; then
                sleep 70s
            fi
            local new_checkpoint_dir=$(get_yaml_actor_value "$resume_yaml_path" "load")
            log_info "resume load path is: $new_checkpoint_dir"
            local resume_iteration=$(cat "$new_checkpoint_dir/latest_checkpointed_iteration.txt" | tr -d ' \t\n\r')
            log_info "resume iteration is: $resume_iteration"
            export RESUME_ITERATION=$resume_iteration
        fi

        bash "${START_SH}" $RESUME_YAML

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log_info "script terminated successfully (code: $exit_code)"
            break
        else
            log_error "script abnormal exit (code: $exit_code)"
            clean_process
            RETRY_COUNT=$((RETRY_COUNT + 1))

            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
              if [ "$VC_TASK_INDEX" = "$MASTER_TRAIN_INDEX" ]; then
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
