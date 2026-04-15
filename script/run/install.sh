#!/bin/bash
# This script is used to install.
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# 自定义变量
install_path="${USER_PWD}"
LOG_SIZE_THRESHOLD=$((10*1024*1024))
declare -A param_dict=()               # 参数个数统计
version_number=""
mxsdk_manufacture_name=""
mxsdk_new_name=""
arch_name="aarch64"

info_record_path="${HOME}/log/AgentSDK"
info_record_file="deployment.log"

#标识符
invalid_param_flag=n
print_version_flag=n
install_flag=n
install_path_flag=n
upgrade_flag=n
quiet_flag=n

ms_deployment_log_rotate() {
  if [ -L "${info_record_path}" ]; then
    echo "The directory path of deployment.log cannot be a symlink." >&2
    exit 1
  fi
  if [[ ! -d "${info_record_path}" ]];then
    mkdir -p "${info_record_path}"
    chmod 750 "${info_record_path}"
  fi
  record_file_path="${info_record_path}"/"${info_record_file}"
  if [ -L "${record_file_path}" ]; then
    echo "The deployment.log cannot be a symlink." >&2
    exit 1
  fi
  if [[ ! -f "${record_file_path}" ]];then
    touch "${record_file_path}" 2>/dev/null
  fi
  record_file_path_bk="${info_record_path}"/"${info_record_file}".bk
  if [ -L "${record_file_path_bk}" ]; then
    echo "The deployment.log.bk cannot be a symlink." >&2
    exit 1
  fi
  log_size=$(find "${record_file_path}" -exec ls -l {} \; | awk '{ print $5 }')
  if [[ "${log_size}" -ge "${LOG_SIZE_THRESHOLD}" ]];then
    mv -f "${record_file_path}" "${record_file_path_bk}"
    touch "${record_file_path}" 2>/dev/null
    chmod 400 "${record_file_path_bk}"
  fi
  chmod 600 "${record_file_path}"
}

ms_log()
{
  ms_deployment_log_rotate
  record_file_path="${info_record_path}/${info_record_file}"
  chmod 640 "${record_file_path}"
  user_ip=$(who am i | awk '{print $NF}' | sed 's/[()]//g')
  [[ -z "${user_ip}" ]] && user_ip="localhost"
  user_name=$(whoami)
  host_name=$(hostname)
  timestamp=$(date "+%Y-%m-%d %H:%M:%S")

  log_line="[${timestamp}][${user_ip}][${user_name}][${host_name}]: $1"
  echo "${log_line}" >> "${record_file_path}"
  chmod 440 "${record_file_path}"
  echo "$1"
}

###  公用函数
function print_usage() {
  ms_log "Please input this command for more help: --help"
}

### 脚本入参的相关处理函数
function check_script_args() {
  ######################  check params confilct ###################
  if [ "${invalid_param_flag}" = y ]; then
    ms_log "ERROR: Check script para failed."
    print_usage
    exit 1
  fi

  if [ $# -lt 3 ]; then
    print_usage
  fi

  # 重复参数检查
  for key in "${!param_dict[@]}";do
    if [ "${param_dict[${key}]}" -gt 1 ]; then
      ms_log "ERROR: parameter error! ${key} is repeat."
      exit 1
    fi
  done

  if [ "${print_version_flag}" = y ]; then
    if [ "${install_flag}" = y ] || [ "${upgrade_flag}" = "y" ]; then
      ms_log "ERROR: --version param cannot config with install or upgrade param."
      exit 1
    fi
  fi

  if [ "${install_path_flag}" = y ]; then
    if [[ ! "${install_path}" =~ ^/.* ]]; then
      ms_log "ERROR: parameter error ${install_path}, must absolute path."
      exit 1
    fi
  fi

  if [ "${upgrade_flag}" = "y" ] && [ "${install_flag}" = "y" ]; then
    ms_log "ERROR: --install and --upgrade para cannot be configured together."
    exit 1
  fi

  if [ "${install_path_flag}" = y ]; then
    if [ "${install_flag}" = "n" ] && [ "${upgrade_flag}" = "n" ]; then
      ms_log "ERROR: Unsupported separate 'install-path' used independently."
      exit 1
    fi
  fi
}

check_target_dir()
{
  if [[ "${install_path}" =~ [^a-zA-Z0-9_./-] ]]; then
    ms_log "Agent SDK dir contains invalid char, please check path."
    exit 1
  fi
}

function check_sha256sum()
{
  if [ ! -e "/usr/bin/sha256sum" ] && [ ! -e "/usr/bin/shasum" ]; then
    ms_log "ERROR:Sha256 check Failed."
    exit 1
  fi
}

# 解析脚本自身的参数
function parse_script_args() {
  local all_para_len="$*"
  if [[ ${#all_para_len} -gt 1024 ]]; then
    ms_log "ERROR: The total length of the parameter is too long."
    exit 1
  fi
  local num=0
  while true; do
    if [[ "$1" == "" ]]; then
      break
    fi
    if [[ "${1: 0: 2}" == "--" ]]; then
      num=$((num + 1))
    fi
    if [[ ${num} -gt 2 ]]; then
      break
    fi
    shift 1
  done
  while true; do
    case "$1" in
    --check)
      check_sha256sum
      exit 0
      ;;
    --version)
      print_version_flag=y
      shift
      ;;
    --install)
      check_platform
      install_flag=y
      ((param_dict["install"]++)) || true
      shift
      ;;
    --install-path=*)
      check_platform
      # 去除指定安装目录后所有的 "/"
      install_path=$(echo "$1" | cut -d"=" -f2 | sed "s/\/*$//g")
      check_target_dir
      if [[ "${install_path}" != /* ]]; then
        install_path="${USER_PWD}/${install_path}"
      fi
      existing_dir="${install_path}"
      while [[ ! -d "${existing_dir}" && "${existing_dir}" != "/" ]]; do
        existing_dir=$(dirname "${existing_dir}")
      done
      abs_existing_dir=$(readlink -f "${existing_dir}")
      nonexistent_suffix="${install_path#"$existing_dir"}"
      install_path="${abs_existing_dir}${nonexistent_suffix}"
      install_path_flag=y
      ((param_dict["install-path"]++)) || true
      shift
      ;;
    --upgrade)
      check_platform
      upgrade_flag=y
      ((param_dict["upgrade"]++)) || true
      shift
      ;;
    --quiet)
      quiet_flag=y
      ((param_dict["quiet"]++)) || true
      shift
      ;;
    -*)
      ms_log "WARNING: Unsupported parameters: $1"
      print_usage
      shift
      ;;
    *)
      if [ "x$1" != "x" ]; then
        ms_log "WARNING: Unsupported parameters: $1"
        print_usage
      fi
      break
      ;;
    esac
  done
}

ms_save_upgrade_info()
{
  path="$1"
  user_ip=$(who am i | awk '{print $NF}' | sed 's/(//g' | sed 's/)//g')
  if [[ -z "${user_ip}" ]]; then
    user_ip=localhost
  fi
  user_name=$(whoami)
  host_name=$(hostname)
  append_text="[$(date "+%Y-%m-%d %H:%M:%S")][${user_ip}][${user_name}][${host_name}]:"
  echo "${append_text}" >> "${path}"
  append_text="${old_version_info}"
  append_text+="    ->    "
  append_text+=${new_version_info}
  echo "${append_text:+$append_text }Upgrade Agent SDK successfully." >> "${path}"
}

ms_save_install_info()
{
  path="$1"
  user_ip=$(who am i | awk '{print $NF}' | sed 's/(//g' | sed 's/)//g')
  if [[ -z "${user_ip}" ]]; then
    user_ip=localhost
  fi
  user_name=$(whoami)
  host_name=$(hostname)
  append_text="[$(date "+%Y-%m-%d %H:%M:%S")][${user_ip}][${user_name}][${host_name}]:"
  echo "$append_text${new_version_info:+ $new_version_info} Install Agent SDK successfully." >> "${path}"
}

ms_record_operator_info()
{
  ms_deployment_log_rotate

  find "${record_file_path}" -type f -exec chmod 750 {} +

  if [[ "${install_flag}" == "y" ]]; then
    ms_save_install_info "${record_file_path}"
    echo "INFO: Install Agent SDK successfully." >&2
  fi

  if [[ "${upgrade_flag}" == "y" ]]; then
    ms_save_upgrade_info "${record_file_path}"
    echo "INFO: Upgrade Agent SDK successfully." >&2
  fi

  find "${record_file_path}" -type f -exec chmod 440 {} +
}

function check_platform()
{
  plat="$(uname -m)"
  result="$(echo ${arch_name} | grep "${plat}")"
  if [[ -z "${result}" ]]; then
    ms_log "WARNING: Platform(${plat}) mismatch for ${arch_name}, please check it."
  fi
}

function check_owner()
{
  _local_path=$1

  owner=$(stat -c "%U" "$_local_path")

  if [ ! "$owner" = "$(whoami)" ]; then
    ms_log "ERROR: current user is not owner at $_local_path, operation failed."
    exit 1
  fi
}

function install_whl()
{
  cd "${install_path}"

  whl_file_name=$(find ./ -maxdepth 1 -type f -name 'agentic_rl-*.whl')
  if [[ "${quiet_flag}" == "n" ]]; then
    ms_log "INFO: Begin to install wheel package(${whl_file_name##*/})."
  fi

  if [[ -f "${whl_file_name}" ]];then
    if [[ "${quiet_flag}" == "y" ]]; then
      python3 -m pip install --no-index --upgrade --force-reinstall --no-dependencies "${whl_file_name##*/}" --user > /dev/null 2>&1
    else
      python3 -m pip install --no-index --upgrade --force-reinstall --no-dependencies "${whl_file_name##*/}" --user
    fi
    if test $? -ne 0; then
      ms_log "ERROR: Install wheel package failed."
      rm -rf "$whl_file_name"
      return 1
    else
      if [[ "${quiet_flag}" == "n" ]]; then
        ms_log "INFO: Install wheel package successfully."
      fi
    fi
    rm -rf "$whl_file_name"
  else
    ms_log "ERROR: There is no wheel package to install."
    return 1
  fi
  cd - > /dev/null
}


function untar_file() {
  SELF_DIR=$(dirname "$(readlink -f "$0")") || {
    ms_log "ERROR: Cannot resolve script directory."
    exit 1
  }
  if [ "${print_version_flag}" = y ]; then
    tar -xzf "${SELF_DIR}/Ascend-agentsdk__linux-aarch64.tar.gz" -C "${SELF_DIR}" --no-same-owner
    cat "${SELF_DIR}/version.info"
  elif [ "${install_flag}" = y ] || [ "${upgrade_flag}" = "y" ]; then
    mxsdk_manufacture_name="agent"
    tar -xzf "${SELF_DIR}/Ascend-agentsdk__linux-aarch64.tar.gz" -C "${SELF_DIR}" --no-same-owner
    version_number=$(head -n 1 "${SELF_DIR}/version.info" | cut -d ':' -f2 | tr -d '[:space:]')
    mxsdk_new_name="${mxsdk_manufacture_name}-${version_number}"
    new_version_info=$version_number

    if [[ "${install_flag}" == "y" ]]; then
      if [[ -d "${install_path}/${mxsdk_new_name}" ]] || \
        [[ -d "${install_path}/${mxsdk_manufacture_name}" ]]; then

        ms_log "WARNING: There is already installation at $install_path with name $mxsdk_manufacture_name or $mxsdk_new_name, install canceled."
        exit 1

      fi
    fi

    if [[ "${upgrade_flag}" == "y" ]]; then
      if [[ -d "${install_path}/${mxsdk_manufacture_name}" ]]; then
        unset doupgrade

        old_version_number=$(head -n 1 "${install_path}/${mxsdk_manufacture_name}/version.info" | cut -d ':' -f2 | tr -d '[:space:]')
        if test $? -ne 0; then
          ms_log "ERROR: Failed to read old files version from ${install_path}/${mxsdk_manufacture_name}/version.info, upgrade failed."
          exit 1
        fi
        mxsdk_old_name="${mxsdk_manufacture_name}-${old_version_number}"
        old_version_info=$old_version_number

        if [[ "${quiet_flag}" == "n" ]]; then
          echo "INFO: Check install path (\"${install_path}\")."
          echo "INFO: Found an existing installation."
          read -t 60 -n1 -re -p "Do you want to upgrade by removing the old installation? [Y/N] " answer
          case "${answer}" in
            Y|y)
              doupgrade=y
              ;;
            *)
              doupgrade=n
              ;;
          esac
        else
          doupgrade=y
        fi

        if [[ "${doupgrade}" == "n" ]]; then
          ms_log "WARNING: user rejected to upgrade, nothing changed"
          exit 1
        else
          ms_log "INFO: user choose to upgrade"

          if [ ! -L "$install_path"/$mxsdk_manufacture_name ]; then
            ms_log "ERROR: agent is not symlink at $install_path/$mxsdk_manufacture_name, upgrade canceled."
            exit 1
          fi

          real_path=$(readlink -f "$install_path"/$mxsdk_manufacture_name)
          if [ ! "$real_path" = "$install_path/$mxsdk_old_name" ]; then
            ms_log "ERROR: Symlink $install_path/$mxsdk_manufacture_name not link to valid path $install_path/$mxsdk_old_name, upgrade canceled."
            exit 1
          fi

          echo "INFO: Removing old installation at $install_path/$mxsdk_old_name ..."
          check_owner "$install_path"/"$mxsdk_old_name"

          bash "$install_path"/"$mxsdk_old_name"/script/uninstall.sh
          if test $? -ne 0; then
            ms_log "ERROR: Failed to remove old installation at $install_path/$mxsdk_old_name"
            exit 1
          else
            ms_log "INFO: Remove old installation success!"
          fi

          echo "INFO: Old installation removed. Proceeding with new installation..."
        fi

      else
        ms_log "ERROR: There is no Agent SDK installed in current install path, upgrade is invalid. Please install Agent SDK first."
        exit 1
      fi

    fi

    mkdir -p "${install_path}"/"${mxsdk_new_name}"
    if [ ! -d "${install_path}/${mxsdk_new_name}" ]; then
      ms_log "ERROR: Create path at ${install_path}/${mxsdk_new_name} failed."
      exit 1
    fi
    check_owner "${install_path}"/"${mxsdk_new_name}"

    tar -xzf "${SELF_DIR}/Ascend-agentsdk__linux-aarch64.tar.gz" -C "${install_path}/${mxsdk_new_name}" --no-same-owner
    if test $? -ne 0; then
      ms_log "ERROR: Failed to extract files to ${install_path}/${mxsdk_new_name}."
      exit 1
    fi

    cd "${install_path}"
    ln -snf "./${mxsdk_new_name}" "${install_path}/${mxsdk_manufacture_name}"

    install_path=${install_path}/${mxsdk_new_name}
    find "$install_path" -type d -exec chmod 750 {} +
    find "$install_path" -type f -exec chmod 640 {} +
    find "$install_path" -type f -name "*.sh" -exec chmod 500 {} +
    find "$install_path" -type f -name "*.info" -exec chmod 440 {} +

    cd - > /dev/null
    install_whl
    if test $? -ne 0; then
      if [[ -d "${install_path}" ]]; then
        rm -rf "${install_path}"
      fi

      if [[ -L "$(dirname "${install_path}")/${mxsdk_manufacture_name}" ]]; then
        rm -f "$(dirname "${install_path}")/${mxsdk_manufacture_name}"
      fi
      ms_log "INFO: rollback successfully."
      exit 1
    else
      ms_record_operator_info
    fi
  else
    ms_log "INFO: Do not proceed with installation or upgrade and exit."
  fi
}


function main() {
  parse_script_args "$@"
  check_script_args "$@"
  untar_file
}

main "$@"