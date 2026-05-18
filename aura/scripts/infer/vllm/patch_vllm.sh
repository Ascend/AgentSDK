#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# vllm patch

set -eo

vllm_dir=$(realpath "$(dirname "$0")")
infer_dir=$(realpath "$(dirname "$vllm_dir")")
scripts_dir=$(realpath "$(dirname "$infer_dir")")
root_dir=$(realpath "$(dirname "$scripts_dir")")

# 不能开启pipefail严格错误检查，utils中的grep管道命令找不到值可能会返回非0值，导致运行中断
source ${scripts_dir}/base/utils.sh

readonly VLLM_VERSION=${VLLM_VERSION:-"0.11.0"}  # 默认版本号
IFS='.' read -ra version_segments <<< "$VLLM_VERSION"
readonly PATCH_NAME="patch_${version_segments[0]}_${version_segments[1]}_${version_segments[2]}"
readonly PATCH_DIR="${root_dir}/aura/runner/infer_adapter/vllm/patch/${PATCH_NAME}"
readonly COMM_DIR="${root_dir}/aura/runner/infer_adapter/vllm/patch/comm"

# 获取Python包安装路径
function get_package_path() {
    local package_name="$1"
    local expect_subdir="$2"  # 可选的子目录名

    if [[ -z "$package_name" ]]; then
        log_error "Package name is empty"
        return 1
    fi

    # 获取pip信息
    local pip_output
    if ! pip_output=$(pip show "$package_name" 2>/dev/null); then
        log_error "Failed to get info for package: $package_name"
        return 1
    fi

    # 解析安装路径
    local install_path
    if install_path=$(grep 'Editable project location:' <<< "$pip_output" | cut -d: -f2 | xargs); then
        [[ -n "$expect_subdir" ]] && install_path+="/${expect_subdir}"
    else
        install_path=$(grep 'Location:' <<< "$pip_output" | cut -d: -f2 | xargs)
        [[ -n "$install_path" ]] && install_path+="/${package_name//-/_}"  # 处理包名差异
    fi

    if [[ -z "$install_path" ]]; then
        log_error "Could not determine path for: $package_name"
        return 1
    fi

    echo "$install_path"
}

# 安全删除目录
function safe_remove() {
    local target="$1"
    local description="$2"

    if [[ ! -d "$target" ]]; then
        log_warn "Target ${description} not exist: ${target}"
        return
    fi

    if [[ "$target" == "/" ]]; then
        log_error "Dangerous path detected: ${target}"
        exit 1
    fi

    log_info "Removing ${description}: ${target}"
    rm -rf "$target"
}

# 应用补丁到目标目录
function apply_patch() {
    local target_dir="$1"
    local module_name="$2"

    # 清理旧文件
    safe_remove "${target_dir}/comm" "comm directory"
    safe_remove "${target_dir}/${PATCH_NAME}" "patch directory"

    # 复制新文件
    log_info "Copying comm to: ${target_dir}"
    cp -rf "${COMM_DIR}" "${target_dir}/"

    log_info "Applying patch ${PATCH_NAME} to: ${target_dir}"
    cp -rf "${PATCH_DIR}" "${target_dir}/"

    # 特殊处理vllm的初始化文件
    if [[ "$module_name" == "vllm" ]]; then
        sed -i '/patch_model_runner_v1/d' "${target_dir}/${PATCH_NAME}/__init__.py"
    fi
}

# 更新初始化文件
function update_init_file() {
    local init_file="$1"
    local import_line="$2"

    if [[ ! -f "$init_file" ]]; then
        log_warn "Init file not found: ${init_file}"
        return
    fi

    # 删除旧导入
    if grep -q "$import_line" "$init_file"; then
        log_info "Removing existing import in: ${init_file}"
        sed -i.bak "/$import_line/d" "$init_file"
        rm -f "${init_file}.bak"
    fi

    # 追加新导入
    log_info "Updating init file: ${init_file}"
    echo "$import_line" >> "$init_file"
}

#######################################
## 主流程
#######################################

function main() {
    # 验证补丁目录存在
    if [[ ! -d "$PATCH_DIR" ]]; then
        log_warn "Patch directory missing: ${PATCH_DIR}"
        exit 0
    fi

    # 获取安装路径
    local vllm_path vllm_ascend_path
    vllm_path=$(get_package_path "vllm")
    vllm_ascend_path=$(get_package_path "vllm-ascend" "vllm_ascend")

    log_info "Detected vllm path: ${vllm_path}"
    log_info "Detected vllm-ascend path: ${vllm_ascend_path}"

    # 应用vllm补丁
    apply_patch "$vllm_path" "vllm"
    update_init_file "${vllm_path}/__init__.py" "import vllm.${PATCH_NAME}"

    # 应用vllm-ascend补丁
    local ascend_worker_path="${vllm_ascend_path}/worker"
    apply_patch "$ascend_worker_path" "vllm_ascend"
    update_init_file "${ascend_worker_path}/__init__.py" "import vllm_ascend.worker.${PATCH_NAME}"

    log_info "Patch applied successfully"
}

# 执行入口
main "$@"
