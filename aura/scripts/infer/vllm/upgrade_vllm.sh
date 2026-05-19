#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

vllm_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))
infer_dir=$(realpath $(dirname ${vllm_dir}))
scripts_dir=$(realpath $(dirname ${infer_dir}))

source ${scripts_dir}/base/utils.sh

# 定义版本配套关系表 (推荐稳定组合)
# 格式: "vLLM版本 | vLLM-Commit | vLLM-Ascend-Commit"
declare -a version_matrix=(
  "0.11.0 | b5ee1e3 | e7409e9"
  "0.13.0 | 72506c9 | 6281c12"
  "0.15.0 | f176443 | 3d43ed9"
  "0.17.0 | b31e932 | e20f0b1"
)

function set_proxy_env()
{
  git config --global --unset http.proxy
  git config --global --unset https.proxy
  git config --global http.sslVerify false
  git config --global https.sslVerify false

}

function clear_proxy_env()
{
  git config --global --unset http.proxy
  git config --global --unset https.proxy
  unset http_proxy
  unset https_proxy
}

function upgrade_vllm()
{
  log_info "vllm upgrade to commit: ${VLLM_COMMIT}"

  pip uninstall vllm -y
  cd /home/upgrade
  git clone https://github.com/vllm-project/vllm.git
  cd /home/upgrade/vllm
  git checkout ${VLLM_COMMIT}
  VLLM_TARGET_DEVICE=empty pip install --no-build-isolation . --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
  cd /home
  pip show vllm
}

function upgrade_vllm_ascend()
{
  log_info "vllm-ascend upgrade to commit: ${VLLM_ASCEND_COMMIT}"

  pip uninstall vllm-ascend -y
  cd /home/upgrade
  git clone https://github.com/vllm-project/vllm-ascend.git
  cd /home/upgrade/vllm-ascend
  git checkout ${VLLM_ASCEND_COMMIT}
  pip install nanobind --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
  pip install -r requirements.txt --no-build-isolation --proxy $http_proxy --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
  pip install -e . --no-build-isolation
  cd /home
  pip show vllm-ascend
}

function upgrade_infer_version()
{
  # upgrade_vllm
  # upgrade_vllm_ascend

  log_info "upgrade vllm and vllm-ascend"

  # TODO: 镜像中备份了升级的版本, 直接替换
  cd /usr/local/python3.11.14/lib/python3.11/site-packages
  log_info "before update:"
  ls -lrt | grep vllm

  rm -rf __editable___vllm_0_16_1rc1_dev140_g4034c3d32_empty_finder.py
  rm -rf __editable__.vllm-0.16.1rc1.dev140+g4034c3d32.empty.pth
  rm -rf vllm-0.16.1rc1.dev140+g4034c3d32.empty.dist-info
  rm -rf __editable___vllm_ascend_0_16_0rc2_dev32_gfe4cad24e_finder.py
  rm -rf __editable__.vllm_ascend-0.16.0rc2.dev32+gfe4cad24e.pth
  rm -rf vllm_ascend-0.16.0rc2.dev32+gfe4cad24e.dist-info
  cp -rf vllm_0170/* ./

  log_info "after update:"
  ls -lrt | grep vllm

  cd /
  log_info "show vllm version:"
  pip show vllm
  log_info "show vllm-ascend version:"
  pip show vllm-ascend

  vllm_ascend_path=$(pip show vllm-ascend | grep "Editable project location" | cut -d ' ' -f 4)
  export PYTHONPATH=${vllm_ascend_path}:${PYTHONPATH}
  log_info "after update, PYTHONPATH: ${PYTHONPATH}"
}

log_info "VLLM_VERSION: ${VLLM_VERSION}"
installed_version=$(pip show vllm-ascend | grep "Version:" | awk '{print $2}')
if [[ "${installed_version}" == "${VLLM_VERSION}"* ]]; then
  log_info "vllm ascend is already at the required version: ${installed_version}"
  return
fi
log_warn "vllm ascend installed version: ${installed_version}, required version: ${VLLM_VERSION}, upgrade..."

for row in "${version_matrix[@]}"; do
  IFS='|' read -r vllm_v vllm_commit vllm_ascend_commit <<< "$row"
  vllm_v=${vllm_v// /}
  vllm_commit=${vllm_commit// /}
  vllm_ascend_commit=${vllm_ascend_commit// /}

  if [[ "${vllm_v}" == "${VLLM_VERSION}" ]]; then
    log_info "match version:$vllm_v, vllm commit: $vllm_commit, vllm-ascend commit: $vllm_ascend_commit"
    export VLLM_COMMIT=${vllm_commit}
    export VLLM_ASCEND_COMMIT=${vllm_ascend_commit}
    break
  fi
done

#mkdir -p /home/upgrade
#rm -rf /home/upgrade/*

set_proxy_env
upgrade_infer_version
clear_proxy_env
