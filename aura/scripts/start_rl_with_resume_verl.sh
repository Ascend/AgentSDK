#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# RL（GRPO等）Verl断点续训任务启动入口

scripts_dir=$(realpath $(dirname $0))
root_dir=$(realpath $(dirname ${scripts_dir}))

save_value="./resume/save_ckpt/" # 默认权重保存路径，将被train的default_local_dir覆盖
resume_iteration=-1 # 默认值-1，表示首次训练
status_dir="./resume/status" # 节点状态日志所在路径 ./resume/status/node_${VC_TASK_INDEX}.status
org_infer_model_path="" # 记录原始hf权重tokenizer路径，需要包含模型名称

source ${scripts_dir}/base/envs.sh
source ${scripts_dir}/base/utils.sh

parse_base_conf

export RESUME_TRAIN_CONF_NAME=${TRAIN_CONF_NAME}_resume
export RESUME_INFER_CONF_NAME=${INFER_CONF_NAME}_resume
export START_RESUME_FLAG=true

log_info "============================================================"
log_info "Start exec resume training with checkpoint..."
log_info "[MONITOR_CMD]: ${MONITOR_CMD}"
log_info "[TRAIN_CONF_NAME]: ${TRAIN_CONF_NAME}"
log_info "[INFER_CONF_NAME]: ${INFER_CONF_NAME}"
log_info "[RESUME_TRAIN_CONF_NAME]: ${RESUME_TRAIN_CONF_NAME}"
log_info "[RESUME_INFER_CONF_NAME]: ${RESUME_INFER_CONF_NAME}"
log_info "[MAX_RETRIES]: ${MAX_RETRIES}"
log_info "============================================================"

function clean_old_process()
{
    log_info "terminating remaining processes ..."
    ray stop --force
    ps -ef | grep "python"| grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {};pkill -9 python; pkill -9 torchrun;
    ps -ef | grep "defunct"|grep python| awk '{print $3}'|xargs -t -i kill -9 {};ps -ef | grep "defunct"|grep torchrun| awk '{print $3}'|xargs -t -i kill -9 {}
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
train_yaml_file="${root_dir}/configs/train/${TRAIN_CONF_NAME}.yaml"
train_resume_yaml_file="${root_dir}/configs/train/${RESUME_TRAIN_CONF_NAME}.yaml"

infer_yaml_file="${root_dir}/configs/infer/${INFER_CONF_NAME}.yaml"
infer_resume_yaml_file="${root_dir}/configs/infer/${RESUME_INFER_CONF_NAME}.yaml"

function generate_new_train_yaml()
{
  if [[ "${exec_cmd_count}" -gt 1 ]]; then
    return
  fi

  # 第一次启动任务，生成用于resume的临时训练配置文件
  touch ${train_resume_yaml_file}
  cp -f ${train_yaml_file} ${train_resume_yaml_file}

  # 记录权重保存路径
  save_value=$(get_verl_conf_val "${train_resume_yaml_file}" "default_local_dir")
  log_info "found default_local_dir: $save_value"

  # 第一次启动任务，是否清理save路径的残余ckpt
  if [[ "${CLEAN_OLD_CKPT}" -eq 1 ]]; then
    log_info "starting to clean all checkpoints from ${save_value} ..."
    clean_all_checkpoints "${save_value}"
  fi
}

is_yaml_replaced=0
function modify_train_reumse_mode()
{
  if [ "${exec_cmd_count}" -lt 2 ] || [ "$is_yaml_replaced" -eq 1 ]; then
    # 在第二次启动训练之后，仅需替换一次resume mode为auto即可
    return
  fi

  # check保存路径下是否有权重可续训
  if [[ ! -f "${save_value}/latest_checkpointed_iteration.txt" ]]; then
    log_info "latest_checkpointed_iteration.txt does not exist under ${save_value}, cannot resume training"
    # 无可续训的新权重，仍从原始路径权重训练
    return
  fi

  org_resume_mode=$(get_verl_conf_val "${train_resume_yaml_file}" "resume_mode")
  log_info "original resume_mode: ${org_resume_mode}"

  # 非0续训时，首次需修改加载相关续训参数
  replace_verl_conf_val "${train_resume_yaml_file}" "resume_mode" "auto"
  if [ $? -ne 0 ]; then
    # 如果修改配置文件失败异常, 则直接退出
    log_error "replace yaml config failed, exit train"
    exit 1
  fi
  # 主节点yaml已替换为auto续训状态标识
  is_yaml_replaced=1
}

function modify_train_yaml()
{
  if [[ "$VC_TASK_INDEX" -ne "$MASTER_TRAIN_INDEX" ]]; then
    return
  fi
  generate_new_train_yaml
  modify_train_reumse_mode
}

function generate_new_infer_yaml()
{
  if [[ "${exec_cmd_count}" -gt 1 ]]; then
    return
  fi

  # 第一次启动任务，生成用于resume的临时推理配置文件
  touch ${infer_resume_yaml_file}
  cp -f ${infer_yaml_file} ${infer_resume_yaml_file}

  # 记录推理权重原始路径，tokenizer用于替换转换后的权重文件
  org_infer_model_path=$(get_verl_conf_val "${infer_resume_yaml_file}" "infer_model_path")
  log_info "found original infer_model_path: $org_infer_model_path"

  # 在续训配置文件末尾新增一行权重转换等待标识if_waiting参数，默认值为true
  sed -i -e '$a\' -e 'if_waiting: true' -e '$a\' "${infer_resume_yaml_file}"
  get_verl_conf_val "${infer_resume_yaml_file}" "if_waiting"
}

function modify_infer_model_path()
{
  # Step 1 主节点基于resume非0续训的ckpt转换生成新的hf权重
  # Step 2 替换model_path下的tokenizer.json等文件为原始infer_model_path目录下的
  # Step 3 替换infer配置的infer_model_path

  # 判断非0续训
  if [[ "${resume_iteration}" -lt 1 ]]; then
      replace_verl_conf_val "${infer_resume_yaml_file}" "if_waiting" "false"
      return
  fi

  # 推理续训权重路径
  converted_infer_model_path="${root_dir}/resume/resume_hf_path/$(basename ${org_infer_model_path%/})_resume_${resume_iteration}"

  # 主节点执行权重转换流程: 如果目录不存在，或者目录存在但没有 convert_done 文件，执行流程
  if [[ ! -d "${converted_infer_model_path}" ]] || [[ ! -f "${converted_infer_model_path}/convert_done" ]]; then
      # 如果转换后的目录存在但未完成，先删除
      if [[ -d "${converted_infer_model_path}" ]]; then
        log_info "convert_done not found, removing incomplete directory..."
        rm -rf "${converted_infer_model_path}"
      fi

      # 执行权重转换脚本
      ckpt_path="${save_value}/global_step_${resume_iteration}"
      bash ${scripts_dir}/base/verl_merge.sh $ckpt_path $converted_infer_model_path
      exit_code=$?
      if [ $exit_code -ne 0 ]; then
        log_error "Failed merge verl weights for vllm..."
        exit $exit_code
      fi

      # 在converted_infer_model_path目录下新增convert_done文件，标识转换已完成
      touch "${converted_infer_model_path}/convert_done"

      # 替换转换权重路径的tokenizer相关文件，除safetensors及model.safetensors.index.json
      # find "${org_infer_model_path}" -type f \
      # ! -name '*.safetensors' \
      # ! -name 'model.safetensors.index.json' \
      # -exec cp -f {} "${converted_infer_model_path}/" \;
  else
      # 转换后的权重已有，则跳过转换
      log_info "infer weights already converted in ${converted_infer_model_path}, skipping..."
  fi

  # 权重转换准备完成，更新推理配置
  replace_verl_conf_val "${infer_resume_yaml_file}" "infer_model_path" "${converted_infer_model_path}"
  replace_verl_conf_val "${infer_resume_yaml_file}" "if_waiting" "false"
}

function modify_infer_yaml()
{
  # 其他节点非0续训时，需等待主节点转换权重并更新配置
  if [[ "$VC_TASK_INDEX" -ne "$MASTER_TRAIN_INDEX" ]]; then
      while true; do
          if grep -q "if_waiting: false" "$infer_resume_yaml_file" 2>/dev/null; then
              log_info "Master node finished weight conversion, stop waiting."
              break
          else
              log_info "Waiting for master node to finish inference weight conversion..."
              sleep 60
          fi
      done
      return
  fi
  # 主节点创建infer续训配置文件，并新增参数if_waiting
  generate_new_infer_yaml
  # 主节点若非0续训，执行推理权重转换流程，生成新的推理可用权重,替换路径下部分文件并更新推理配置
  modify_infer_model_path
}

function get_resume_iteration()
{
  if [[ "${exec_cmd_count}" -eq 1 ]]; then
    # 首次训练，若原始resume_mode为resume_path，仍需更新推理权重及版本，属于续训
    resume_mode=$(get_verl_conf_val "${train_resume_yaml_file}" "resume_mode")
    if [[ "${resume_mode}" == "resume_path" ]]; then
      # 原始resume_mode为resume_path，第一次也要更新推理权重及版本，属于续训
      resume_from_path=$(get_verl_conf_val "${train_resume_yaml_file}" "resume_from_path")
      log_info "resume_path | found resume_from_path: $resume_from_path"
      resume_iteration=$(grep_global_step_from_path $resume_from_path)
      if [ -z "$resume_iteration" ]; then
          # 不存在正确路径 异常退出
          log_info "resume_from_path is invalid: $resume_from_path"
          exit 1
      fi
      export RESUME_ITERATION=$resume_iteration
      log_info "resume_path | resume iteration:=${resume_iteration}, export RESUME_ITERATION=${RESUME_ITERATION}"
      echo "ready resume_path ${exec_cmd_count} ${resume_iteration}" >> "${status_dir}/node_${VC_TASK_INDEX}.status"
    fi
    return
  fi

  # 断点后默认从0续训
  resume_iteration=0
  if [[ "$VC_TASK_INDEX" == "$MASTER_TRAIN_INDEX" ]]; then
    if [ $is_yaml_replaced -eq 1 ]; then
      # 非0断点续训，须从新路径下重新读取resume_iteration
      resume_iteration=$(cat ${save_value}/latest_checkpointed_iteration.txt | tr -d ' \t\n\r')
    fi
  else
    # 其他节点等待主节点ready状态码
    while true; do
      line=$(grep "ready ${exec_cmd_count}" "${status_dir}/node_${MASTER_TRAIN_INDEX}.status" 2>/dev/null)
      if [[ -n "$line" ]]; then
        # 截取最后一个字段作为 resume_iteration
        resume_iteration=$(echo "$line" | awk '{print $NF}')
        log_info "Detected master ready with resume_iteration=$resume_iteration"
        break
      fi
      sleep 10
    done
  fi
  # 这个环境变量提供给训练进程, 感知断点续训, 同步最新的权重给推理（主要是训练主节点用）
  export RESUME_ITERATION=${resume_iteration}
  log_info "resume iteration: ${resume_iteration}, export RESUME_ITERATION=${RESUME_ITERATION}"
  # 各节点发送ready信号并带上resume_iteration标识
  echo "ready ${exec_cmd_count} ${resume_iteration}" >> "${status_dir}/node_${VC_TASK_INDEX}.status"
}

function resume_train()
{
  clean_old_process

  exec_cmd_count=$((exec_cmd_count + 1))

  # 主从节点restart差距最大应该在1分钟
  if [[ "$VC_TASK_INDEX" == "$MASTER_TRAIN_INDEX" ]]; then
    log_warn "waiting for 5 min before restarting ..."
    sleep 5m
    log_info "update if_waiting to true in $infer_resume_yaml_file before next resume ..."
    sed -i 's/^if_waiting:.*/if_waiting: true/' "$infer_resume_yaml_file"
  else
    log_warn "waiting for 4 min before restarting ..."
    sleep 4m
  fi
}

function init_group_status()
{
  if [[ "$VC_TASK_INDEX" == "$MASTER_TRAIN_INDEX" ]]; then
    # 主节点负责目录初始化
    if [[ -d "${status_dir}" ]]; then
      # 清空目录下所有 .status 文件
      rm -f "${status_dir}"/*.status
      log_info "status_dir already exists, cleared old .status files"
    else
      # 创建目录
      mkdir -p "${status_dir}"
      log_info "created status_dir ${status_dir}"
    fi
    # 清除可能残留的临时resume配置文件
    rm -f ${train_resume_yaml_file}
    rm -f ${infer_resume_yaml_file}
  else
    # 其他节点等待主节点操作 确保历史状态日志已清空
    while true; do
      if [[ ! -s "${status_dir}/node_${VC_TASK_INDEX}.status" ]]; then
        break
      fi
      sleep 10
    done
  fi
  echo "init" >> "${status_dir}/node_${VC_TASK_INDEX}.status"
}

function clean_old_files()
{
  infer_dir="${scripts_dir}/infer"
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

function main()
{
  log_info "train process begin!!!"
  # 集群节点状态初始化，清除残留的resume配置文件
  init_group_status

  # 续训主流程
  while [ ${exec_cmd_count} -lt ${MAX_RETRIES} ]; do
    log_info "execute [${MONITOR_CMD}], times: ${exec_cmd_count}"

    # 仅推理主节点清理推理配置以保证时序
    if [[ "$VC_TASK_INDEX" -eq 0 ]]; then
      clean_old_files
      echo "clean ${exec_cmd_count}" >> "${status_dir}/node_${VC_TASK_INDEX}.status"
    else
      # 其他节点(主节点)等待推理主节点操作 确保历史推理配置已清空
      while true; do
        if grep -q "clean ${exec_cmd_count}" ${status_dir}/node_0.status 2>/dev/null; then
          log_info "Detected infer master has cleaned the old configs ..."
          break
        fi
        sleep 5
      done
    fi

    # 主节点自动识别并修改续训train配置
    modify_train_yaml
    # 主节点获取续训版本
    get_resume_iteration
    # 自动生成续训infer权重并更新infer配置
    modify_infer_yaml

    # 执行训练启动脚本
    bash ${scripts_dir}/${MONITOR_CMD} &
    main_pid=$!

    # 启动监控循环（后台进程）
    (
      while true; do
        # 检查共享状态目录里是否有 fail 标记
        if grep -q "fail ${exec_cmd_count}" ${status_dir}/*.status 2>/dev/null; then
          log_error "Detected another node failed, killing main process..."
          kill -9 $main_pid
          exit 1
        fi
        sleep 120
      done
    ) &

    # 等待训练脚本结束
    wait $main_pid
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_info "script terminated successfully (code: $exit_code)"
        echo "ok ${exec_cmd_count}" >> "${status_dir}/node_${VC_TASK_INDEX}.status"
        break
    else
        log_error "script abnormal exit (code: $exit_code)"
        echo "fail ${exec_cmd_count}" >> "${status_dir}/node_${VC_TASK_INDEX}.status"
        resume_train
    fi
  done

  log_info "train process complete!!!"
}

main
