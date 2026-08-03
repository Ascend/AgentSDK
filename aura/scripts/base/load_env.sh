#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# 统一环境变量加载脚本
# 加载优先级: 环境变量(已export) > env.local > env.conf > 脚本默认值
# 用法: source ${scripts_dir}/base/load_env.sh
#
# ============================================================
# env.local 创建与使用流程
# ============================================================
# 1) 创建文件（与 env.conf 同目录，仅在本机生效，已加入 .gitignore）:
#      touch aura/configs/env/env.local
# 2) 按 KEY=VALUE 格式写入需要覆盖 env.conf 的变量，每行一条，例如:
#      # 覆盖默认网卡（env.conf 默认 eth0，本机使用 bond19）
#      DEFAULT_SOCKET_IFNAME=bond19
#      # 关闭 vLLM 优化开关
#      USE_VLLM_OPT=false
# 3) 加载优先级: 命令行 export  >  env.local  >  env.conf
#    例: 在 shell 中先 export FOO=1, 再 source 本脚本, FOO 保持 1, 不会被 env.conf 覆盖
# 4) 安全约束:
#    - key 必须匹配 [A-Za-z_][A-Za-z0-9_]*  (POSIX 变量命名规范), 否则跳过并告警
#    - value 中如果包含 $(...) 或 `cmd`, 不会被二次解析执行（使用 printf -v 赋值）
# 5) 本脚本可被多次 source（幂等）：派生变量 (WORKSPACE / *_PATH / LD_LIBRARY_PATH)
#    仅在 AURA_ENV_DERIVED 未设置时执行, 避免重复追加。
#
# 实现说明:
#   - load_env_file 仅在变量未被设置时才赋值（保证外部环境变量优先级最高）
#   - 为使 env.local 能覆盖 env.conf，先扫描 env.local 中已被外部设置的 key，
#     记录到 AURA_EXTERNAL_KEYS；加载 env.conf 时跳过已设置的 key；
#     加载 env.local 时仅跳过外部 key，其余 key 无条件覆盖 env.conf
# ============================================================

# 解析路径: load_env.sh 位于 aura/scripts/base/
# base_dir  = aura/scripts/base
# scripts_dir = aura/scripts
# root_dir  = aura
base_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
scripts_dir=$(realpath "$(dirname "${base_dir}")")
root_dir=$(realpath "$(dirname "${scripts_dir}")")

ENV_CONF="${root_dir}/configs/env/env.conf"
ENV_LOCAL="${root_dir}/configs/env/env.local"

# 本脚本被 source 调用, 不修改调用方的 set -e / set -u / pipefail 等 shell 选项。
# 变量未定义检测由 [[ -z "${!key+x}" ]] 显式处理, 不依赖 set -u。

# 校验 key 是否符合 POSIX 变量命名规范 [A-Za-z_][A-Za-z0-9_]*
# 防止 env.conf / env.local 中存在 A&B=1、$(id)=1 等恶意/误写
function _is_valid_key() {
    [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}

# 从配置文件中提取所有合法 key（去注释/空行/非法行）
function _extract_keys() {
    local conf_file=$1
    [[ ! -f "${conf_file}" ]] && return
    local key
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ "${line}" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line}" ]] && continue
        [[ "${line}" != *=* ]] && continue
        key="${line%%=*}"
        key="${key//[[:space:]]/}"
        _is_valid_key "${key}" || continue
        echo "${key}"
    done < "${conf_file}"
}

# 追加 LD_LIBRARY_PATH 条目，跳过空路径和已存在路径，避免产生前导/尾随冒号。
function _append_ld_library_path() {
    local path
    for path in "$@"; do
        [[ -z "${path}" ]] && continue
        if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
            LD_LIBRARY_PATH="${path}"
        elif [[ ":${LD_LIBRARY_PATH}:" != *":${path}:"* ]]; then
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${path}"
        fi
    done
    export LD_LIBRARY_PATH
}

# 记录 env.local 中已被外部设置的 key（外部环境变量优先级最高，不被 env.local 覆盖）。
# 使用换行分隔的字符串，避免依赖 bash 4.0+ 的关联数组。
AURA_EXTERNAL_KEYS=$'\n'
if [[ -f "${ENV_LOCAL}" ]]; then
    while read -r k; do
        [[ -z "${k}" ]] && continue
        if [[ -n "${!k+x}" ]]; then
            AURA_EXTERNAL_KEYS="${AURA_EXTERNAL_KEYS}${k}"$'\n'
        fi
    done < <(_extract_keys "${ENV_LOCAL}")
fi

# --- 加载配置文件的通用函数 ---
# mode=strict : 仅当变量未被设置时才赋值（用于 env.conf，避免覆盖外部/env.local）
# mode=override: 仅跳过外部 key，其余 key 无条件赋值（用于 env.local，覆盖 env.conf）
function load_env_file() {
    local conf_file=$1
    local mode=${2:-strict}
    if [[ ! -f "${conf_file}" ]]; then
        return
    fi
    local key value
    while IFS= read -r line || [[ -n "${line}" ]]; do
        # 跳过注释行和空行
        [[ "${line}" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line}" ]] && continue
        # 跳过不含=的行
        [[ "${line}" != *=* ]] && continue
        # 拆分 key=value
        key="${line%%=*}"
        value="${line#*=}"
        # 去除首尾空白
        key="${key//[[:space:]]/}"
        value="${value%"${value##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        # 跳过空 key
        [[ -z "${key}" ]] && continue
        # 跳过非法 key（防止 $(id)=1 等注入），并给出告警
        if ! _is_valid_key "${key}"; then
            echo "load_env.sh: WARN invalid key '${key}' in ${conf_file}, skipped" >&2
            continue
        fi
        if [[ "${mode}" == "override" ]]; then
            # env.local: 仅跳过外部 key，其余无条件覆盖
            [[ "${AURA_EXTERNAL_KEYS}" == *$'\n'"${key}"$'\n'* ]] && continue
            # 使用 printf -v 安全赋值, 避免 value 中的 $(...) / `cmd` 被二次解析执行
            printf -v "${key}" '%s' "${value}"
            export "${key}"
        else
            # env.conf: 仅当变量未被设置时才赋值（外部 > env.local > env.conf）
            if [[ -z "${!key+x}" ]]; then
                printf -v "${key}" '%s' "${value}"
                export "${key}"
            fi
        fi
    done < "${conf_file}"
}

# 按优先级加载: 先加载默认配置(env.conf), 再用本地覆盖(env.local)
# env.conf 使用 strict 模式（仅填充未设置的变量）
# env.local 使用 override 模式（覆盖 env.conf，但跳过外部环境变量）
load_env_file "${ENV_CONF}" "strict"
load_env_file "${ENV_LOCAL}" "override"

# --- 基于配置自动派生的变量（仅首次加载时执行，避免重复追加） ---
if [[ -z "${AURA_ENV_DERIVED+x}" ]]; then
    export AURA_ENV_DERIVED=1

    # root_dir 已由 realpath 解析为绝对路径，无需再添加前导斜杠。
    export WORKSPACE="${root_dir}"
    export LOG_PATH="${root_dir}/logs/"

    # 拼接第三方组件路径
    export RLLM_PATH=${WORKSPACE}/${RLLM_REL_PATH}
    export VLLM_PATH=${WORKSPACE}/${VLLM_REL_PATH}
    export VLLM_ASCEND_PATH=${WORKSPACE}/${VLLM_ASCEND_REL_PATH}
    export MINDSPEED_RL_PATH=${WORKSPACE}/${MINDSPEED_RL_REL_PATH}
    export MEGATRON_PATH=${WORKSPACE}/${MEGATRON_REL_PATH}
    export MINDSPEED_PATH=${WORKSPACE}/${MINDSPEED_REL_PATH}
    export MINDSPEED_LLM_PATH=${WORKSPACE}/${MINDSPEED_LLM_REL_PATH}

    # 未配置 PYTHON_HOME 时，从当前 python3 可执行文件推导安装前缀。
    python_exec=""
    python_site_dir=""
    if [[ -n "${PYTHON_HOME:-}" && -x "${PYTHON_HOME}/bin/python3" ]]; then
        python_exec="${PYTHON_HOME}/bin/python3"
    fi
    if [[ -z "${PYTHON_HOME:-}" ]]; then
        python_exec=$(command -v python3 2>/dev/null || true)
        if [[ -n "${python_exec}" ]]; then
            python_prefix=$("${python_exec}" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)
            PYTHON_HOME=${python_prefix:-$(realpath "$(dirname "${python_exec}")/..")}
            export PYTHON_HOME
        fi
    fi

    if [[ -n "${python_exec}" ]]; then
        python_site_dir=$("${python_exec}" -c 'import site; paths = site.getsitepackages(); print(paths[0] if paths else "")' 2>/dev/null || true)
    fi

    if [[ -n "${PYTHON_HOME:-}" ]]; then
        if [[ -z "${python_site_dir}" ]]; then
            # 从 PYTHON_HOME 派生 python3.x 目录，避免 PYTHON_HOME 变更时仍写死 python3.11。
            python_home_name="${PYTHON_HOME##*/}"
            if [[ "${python_home_name}" =~ ^python([0-9]+)\.([0-9]+)(\..*)?$ ]]; then
                python_lib_version="python${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
            else
                python_lib_version=$(python3 -c 'import sys; print("python%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
                python_lib_version=${python_lib_version:-python3.11}
            fi
            python_site_dir="${PYTHON_HOME}/lib/${python_lib_version}/site-packages"
        fi

        _append_ld_library_path \
            "${python_site_dir}/torch/lib" \
            "${python_site_dir}/torch_npu/lib"
    fi
fi

# 日志目录初始化
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
fi
