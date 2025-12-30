#!/bin/bash
# This script is used to uninstall.
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

# Simple log helper functions
CUR_PATH=$(cd "$(dirname "$0")" || { echo "Failed to enter current path" ; exit ; } ; pwd)

info_record_path="${HOME}/log/AgentSDK"
info_record_file="deployment.log"
info_record_file_back="deployment.log.bak"
log_file=$info_record_path/$info_record_file
LOG_SIZE_THRESHOLD=1024000

USER_N="$(whoami)" || exist 1
readonly USER_N
WHO_PATH="$(which who)" || exist 1
readonly WHO_PATH
CUT_PATH="$(which cut)" || exist 1
readonly CUT_PATH
IP_N="$("${WHO_PATH}" -m | "${CUT_PATH}" -d '(' -f 2 | "${CUT_PATH}" -d ')' -f 1)"
if [ "${IP_N}" = "" ]; then
   IP_N="localhost"
fi

function rotate_log() {
    check_path "$log_file"
    mv -f "$log_file" "$info_record_path/$info_record_file_back"
    touch "$log_file" 2>/dev/null
    check_path "$info_record_path/$info_record_file_back"
    chmod 440 "$info_record_path/$info_record_file_back"
    check_path "$log_file"
    chmod 640 "$log_file"
}

function check_path() {
    if [ "$1" != "$(realpath "$1")" ]; then
      echo "Log file is not support symlink, exiting!"
      exit 1
    fi
}

function log_check() {
    local log_size
    log_size="$(find "$log_file" -exec ls -l {} \; | awk '{ print $5 }')"
    if [[ "${log_size}" -ge "${LOG_SIZE_THRESHOLD}" ]];then
        rotate_log
    fi
}

function log() {
    # print log to log file
    if [ "$log_file" = "" ]; then
        echo "[$(date +%Y%m%d-%H:%M:%S)] [$USER_N] [$IP_N]: $1" >&2
    fi
    if [ -f "$log_file" ]; then
        log_check
        if ! echo "[$(date +%Y%m%d-%H:%M:%S)] [$USER_N] [$IP_N]: $1" >> "$log_file"; then
          echo "Can not write log, exiting!"
          exit 1
        fi
    else
        echo "Log file does not exist, exiting!"
        exit 1
    fi
}

get_run_path() {
  run_path=$(pwd)
  cd ..
  if [[ "$run_path" =~ /agent ]];then
    suffix='agent'
  else
    log "Error: Directory agent does not exist, exiting!"
    echo "Directory agent does not exist in path[$run_path], exiting!"
    exit 1
  fi
  del_path=$(pwd)/"$suffix"
}

real_delete() {
  cd "${CUR_PATH}/.." || {
    echo "Cannot locate Agent SDK directory: ${CUR_PATH}/.."
    log "Error: Cannot locate Agent SDK directory!"
    exit 255
  }
  get_run_path
  if [[ -f "$del_path"/version.info ]];then
    version_info="$del_path"/version.info
    if [ ! -d "$info_record_path" ];then
        mkdir -p "$info_record_path"
        chmod 750 "$info_record_path"
    fi

    if [[ ! -f "$log_file" ]];then
        touch "$log_file"
    fi
    find "$log_file" -type f -exec chmod 640 {} +
    log "$(cat "${version_info}")"

    python3 -m pip uninstall agentic_rl -y
    if test $? -ne 0; then
      log "Error: Uninstall wheel package failed."
      exit 1
    fi

    chmod u+w -R "$del_path"
    del_real_path=$(realpath "$del_path")
    # remove soft link first
    if [ -d "$del_path" ];then
      rm -rf "$del_path"
    fi
    # remove real files
    if [ -d "$del_real_path" ];then
      rm -rf "$del_real_path"
    fi

    log "Uninstall Agent SDK package successfully."
    echo 'Uninstall Agent SDK package successfully.'
  else
    log "Uninstall Agent SDK package failed, version info is missed."
    echo 'Uninstall Agent SDK package failed, version info is missed.'
  fi
}

real_delete