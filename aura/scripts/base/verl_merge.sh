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

export HCCL_SOCKET_FAMILY=AF_INET
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.11.14/lib/python3.11/site-packages/torch/lib/:/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib/

# ==========================================
# 2. 允许加载 NPU 后端并屏蔽物理显卡
# ==========================================
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
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
