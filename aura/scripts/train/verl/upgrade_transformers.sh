#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

verl_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))
train_dir=$(realpath $(dirname ${verl_dir}))
scripts_dir=$(realpath $(dirname ${train_dir}))

source ${scripts_dir}/base/utils.sh

function set_proxy_env()
{
  git config --global --unset http.proxy
  git config --global --unset https.proxy
  git config --global http.https://github.com.proxy "http://10.50.113.120:3128"
  git config --global http.sslVerify false
  git config --global https.sslVerify false

  export http_proxy="http://10.50.113.120:3128"
  export https_proxy=${http_proxy}
}

function clear_proxy_env()
{
  git config --global --unset http.proxy
  git config --global --unset https.proxy
  unset http_proxy
  unset https_proxy
}

function upgrade_transformers()
{
  log_info "transformers upgrade"

  cd /home/upgrade_0313/transformers
  pip install -e . --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
  cd /home
  pip show transformers
}

log_info "transformers version: ${TRANSFORMERS_VERSION}"
installed_version=$(pip show transformers | grep "Version:" | awk '{print $2}')
if [[ "${installed_version}" == "${TRANSFORMERS_VERSION}"* ]]; then
  log_info "transformers is already at the required version: ${installed_version}"
  return
fi
log_warn "transformers installed version: ${installed_version}, required version: ${TRANSFORMERS_VERSION}, upgrade..."

set_proxy_env
upgrade_transformers
clear_proxy_env
