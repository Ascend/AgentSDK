#!/usr/bin/env bash
set -e

echo "============================================"
echo " AgentSDK build_env"
echo "============================================"

echo "[build_env] apt update & install"
apt-get update
apt-get install -y net-tools dos2unix ca-certificates curl wget
update-ca-certificates || true

echo "[build_env] mkdir -p /home/work"
mkdir -p /home/work

echo "[build_env] clone 依赖仓库"
bash /home/work/AgentSDK/aura/dockers/env/build_repos.sh

echo "[build_env] pip 源"
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com

echo "[build_env] 安装 vllm"
cd /home/work/vllm
git checkout 4034c3d32
VLLM_TARGET_DEVICE=empty pip install -v -e .

export CPLUS_INCLUDE_PATH=/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/inc/kernel:${CPLUS_INCLUDE_PATH:-}

echo "[build_env] 安装 vllm-ascend"
cd /home/work/vllm-ascend
git checkout fe4cad24e
export COMPILE_CUSTOM_KERNELS=1
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh ]; then
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
pip install -v -e .

echo "[build_env] 安装 MindSpeed + Megatron-LM + mbridge"
pip install -e /home/work/MindSpeed
pip install -e /home/work/Megatron-LM
pip uninstall -y triton || true
pip install mbridge

echo "[build_env] 安装 verl"
cd /verl
git checkout e9972368aa6a6078eacd7f0678bdfdd0196ce7b5
pip install -r requirements-npu.txt
pip install -v -e .

echo "[build_env] 安装 transformers"
cd /home/work/transformers
git checkout cc7ab9be508ce6ed3637bba9e50367b29b742dc6
pip install -v -e .

echo "[build_env] 安装 AgentSDK + third_party"
bash /home/work/AgentSDK/aura/dockers/env/build_common.sh

echo "[build_env] patch triton-ascend for CANN 9.0.0"
bash /home/work/AgentSDK/aura/dockers/patch/patch_triton_ascend.sh

echo "[build_env] 安装 uv"
pip install uv

echo "[build_env] 创建 qwen3_moe 虚拟环境"
mkdir -p /home/work/model_env
uv venv /home/work/model_env/qwen3_moe

echo "[build_env] 在虚拟环境中执行 build_qwen3_moe_env.sh"
cd /home/work/model_env/qwen3_moe
source bin/activate
hash -r
bash /home/work/AgentSDK/aura/dockers/env/build_qwen3_moe_env.sh
deactivate

cd /home/work/AgentSDK/aura

echo "============================================"
echo " build_env 完成"
echo "============================================"
