#!/bin/bash
echo "Starting verl_merge.sh | Converting verl actor weights to vllm HF format..."

# ==========================================
# 1. 加载全局 CANN 环境
# ==========================================
# 尝试加载标准的系统级 NPU 环境脚本
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "已加载 ascend-toolkit 环境"
elif [ -f "/usr/local/Ascend/nnae/set_env.sh" ]; then
    source /usr/local/Ascend/nnae/set_env.sh
    echo "已加载 nnae 环境"
fi

# ==========================================
# 2. 统一加载 Aura 环境变量（HCCL_SOCKET_FAMILY / LD_LIBRARY_PATH /
#    TORCH_DEVICE_BACKEND_AUTOLOAD 等已通过 env.conf 集中配置）
# ==========================================
base_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))
scripts_dir=$(realpath $(dirname ${base_dir}))
source ${scripts_dir}/base/load_env.sh

# 允许加载 NPU 后端并屏蔽物理显卡（仅本脚本专用，不入 env.conf）
export ASCEND_RT_VISIBLE_DEVICES=""

# ==========================================
# 3. 运行合并
# ==========================================
ckpt_path="$1/actor"
target_path="$2"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${ckpt_path} \
    --target_dir ${target_path} \
    --use_cpu_initialization
