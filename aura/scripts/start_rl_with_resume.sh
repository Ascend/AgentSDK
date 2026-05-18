#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# RL（GRPO等）断点续训任务启动入口

scripts_dir=$(realpath $(dirname $0))
root_dir=$(realpath $(dirname ${scripts_dir}))

source ${scripts_dir}/base/envs.sh
source ${scripts_dir}/base/utils.sh

parse_base_conf

export RESUME_TRAIN_CONF_NAME=${TRAIN_CONF_NAME}_resume

log_info "============================================================"
log_info "Start exec resume training with checkpoint..."
log_info "[MONITOR_CMD]: ${MONITOR_CMD}"
log_info "[TRAIN_CONF_NAME]: ${TRAIN_CONF_NAME}"
log_info "[RESUME_TRAIN_CONF_NAME]: ${RESUME_TRAIN_CONF_NAME}"
log_info "[MAX_RETRIES]: ${MAX_RETRIES}"
log_info "============================================================"

function clean_old_process()
{
    ray stop --force
}

function replace_yaml_value()
{
    local load_value=$1
    local save_value=$2
    local yaml_file=$3

    log_info "replace yaml value:"
    log_info "[yaml_file]: ${yaml_file}"
    log_info "[load path]: ${load_value}"
    log_info "[save path]: ${save_value}"

    # 更新direct的load值为save的值
    sed -i "/^\s*actor_config:/,/^\s*[_a-z]*_config:/ {s|^\([[:space:]]*\)load:.*|\1load: ${save_value}  #${load_value}|}" "${yaml_file}"
    # 更新direct的finetune值为false的值
    if grep -q "finetune:" "${yaml_file}"; then
        sed -i "/^\s*actor_config:/,/^\s*[_a-z]*_config:/ {s|^\([[:space:]]*\)finetune:.*|\1finetune: false  #true|}" "${yaml_file}"
    else
        sed -i "/actor_config:/a\    finetune: false  #true" "${yaml_file}"
    fi
    # 修改或新增direct的ref_model_load_path
    if grep -q "integrated_mode_config:" "${yaml_file}"; then
        sed -i "/^  rl_config:/,/^  [_a-z]*_config:/ {s|^\([[:space:]]*\)ref_model_load_path:.*|\1ref_model_load_path: ${load_value}|}" "${yaml_file}"
    else
        sed -i "/rl_config:/a\    integrated_mode_config:\n      ref_model_load_path: ${load_value}" "${yaml_file}"
    fi
}

function clean_all_checkpoints()
{
  local save_checkpoint_dir="$1"
  if [ -z "${save_checkpoint_dir}" ]; then
      log_warn "save dir ${save_checkpoint_dir} not config"
      return
  fi

  if [ ! -d "${save_checkpoint_dir}" ]; then
      log_warn "dir ${save_checkpoint_dir} does not exist."
      return
  fi

  # 安全删除目录下所有文件
  log_info "cleaning checkpoint directory: ${save_checkpoint_dir}"
  rm -rf "${save_checkpoint_dir:?}"/*
  log_info "checkpoint directory clean succeed"
}

exec_cmd_count=1
yaml_file="${root_dir}/configs/train/${TRAIN_CONF_NAME}.yaml"
resume_yaml_file="${root_dir}/configs/train/${RESUME_TRAIN_CONF_NAME}.yaml"

function generate_new_yaml_file()
{
  if [[ "${exec_cmd_count}" -gt 1 ]]; then
    return
  fi

  # 启动训练前，自动覆盖先前因续训被修改的resume配置
  if [[ ! -f "${resume_yaml_file}" ]]; then
    touch ${resume_yaml_file}
  fi
  cp -f ${yaml_file} ${resume_yaml_file}

  local save_value=$(get_actor_conf_from_yaml "${resume_yaml_file}" "save")

  # 第一次启动训练，将save路径残余ckpt清理
  if [[ "${CLEAN_OLD_CKPT}" -eq 1 ]]; then
    log_info "starting to clean all checkpoints from ${save_value} ..."
    clean_all_checkpoints "${save_value}"
  fi
}

is_yaml_replaced=0
function modify_weight_load_path()
{
  if [ "${exec_cmd_count}" -lt 2 ] || [ "$is_yaml_replaced" -eq 1 ]; then
    return
  fi

  local load_value=$(get_actor_conf_from_yaml "${resume_yaml_file}" "load")
  local save_value=$(get_actor_conf_from_yaml "${resume_yaml_file}" "save")
  if [[ ! -f "${save_value}/latest_checkpointed_iteration.txt" ]]; then
    return
  fi

  # 非0续训时，首次需修改加载权重路径load为save以及相关续训参数
  replace_yaml_value "${load_value}" "${save_value}" "${resume_yaml_file}"
  if [ $? -ne 0 ]; then
    # 如果修改配置文件失败异常, 则直接退出
    log_error "replace yaml config failed, exit train"
    exit 1
  fi
  # yaml只替换一次
  is_yaml_replaced=1
}

function modify_train_yaml()
{
  if [[ "$VC_TASK_INDEX" -ne "$MASTER_TRAIN_INDEX" ]]; then
    return
  fi
  generate_new_yaml_file
  modify_weight_load_path
}

function get_resume_iteration()
{
  if [[ "${exec_cmd_count}" -eq 1 ]]; then
    return
  fi

  # 等待主节点替换路径及清除过期ckpt
  if [[ "$VC_TASK_INDEX" -gt "$MASTER_TRAIN_INDEX" ]]; then
    sleep 70s
  fi
  # 传入断点续训步数环境变量
  local new_checkpoint_dir=$(get_actor_conf_from_yaml "${resume_yaml_file}" "load")
  log_info "resume checkpoint load path: ${new_checkpoint_dir}"
  # 默认从0续训
  local resume_iteration=0
  if [ $is_yaml_replaced -eq 1 ]; then
    # 非0断点续训，须从新路径下重新读取resume_iteration
    resume_iteration=$(cat ${new_checkpoint_dir}/latest_checkpointed_iteration.txt | tr -d ' \t\n\r')
  fi
  # 这个环境变量提供给训练进程, 感知断点续训, 同步最新的权重给推理
  export RESUME_ITERATION=${resume_iteration}

  log_info "resume iteration: ${resume_iteration}, export RESUME_ITERATION=${RESUME_ITERATION}"
}

function resume_train()
{
  clean_old_process

  exec_cmd_count=$((exec_cmd_count + 1))

  # 主从节点restart差距最大应该在1分钟
  if [[ "$VC_TASK_INDEX" == "$MASTER_TRAIN_INDEX" ]]; then
    log_warn "waiting for 5 min before restarting ..."
    sleep 5m
  else
    log_warn "waiting for 4 min before restarting ..."
    sleep 4m
  fi
}

function main()
{
  log_info "train process begin!!!"
  while [ ${exec_cmd_count} -lt ${MAX_RETRIES} ]; do
    log_info "execute [${MONITOR_CMD}], times: ${exec_cmd_count}"

    modify_train_yaml
    get_resume_iteration

    # 执行训练启动脚本
    sh ${scripts_dir}/${MONITOR_CMD}
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
      log_info "script terminated successfully (code: $exit_code)"
      break
    fi
    log_error "script abnormal exit (code: $exit_code)"
    resume_train
  done

  log_info "train process complete!!!"
}

main
