#!/usr/bin/env bash
set -e

echo "============================================"
echo " AgentSDK build_env"
echo "============================================"

echo "[build_env] apt update & install"
apt-get update
apt-get install -y net-tools dos2unix ca-certificates curl wget
update-ca-certificates || true
apt clean
rm -rf /var/lib/apt/lists/*

echo "[build_env] mkdir -p /home/work"
mkdir -p /home/work

echo "[build_env] clone 依赖仓库"
bash /home/work/AgentSDK/docker/aura/env/build_repos.sh

echo "[build_env] pip 源"
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host "mirrors.aliyun.com mirrors.huaweicloud.com"

echo "[build_env] 安装 verl"
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
cd /verl
git checkout e9972368aa6a6078eacd7f0678bdfdd0196ce7b5
pip install -r requirements-npu.txt
pip install -v -e .

echo "[build_env] 安装 vllm"
PYTHONUNBUFFERED=1 VLLM_TARGET_DEVICE="empty" pip install -v -e /home/work/vllm/[audio]
pip uninstall -y triton

echo "[build_env] 安装 vllm-ascend"
export PIP_EXTRA_INDEX_URL="https://mirrors.huaweicloud.com/ascend/repos/pypi"
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh ]; then
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
PYTHONUNBUFFERED=1 pip install -v -e /home/work/vllm-ascend/
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

echo "[build_env] 安装 MindSpeed + Megatron-LM + mbridge"
pip install -e /home/work/MindSpeed
pip install -e /home/work/Megatron-LM
pip install mbridge

echo "[build_env] 安装 AgentSDK + third_party"
bash /home/work/AgentSDK/docker/aura/env/build_common.sh
pip cache purge || true

echo "[build_env] 应用环境相关patch"
cd /verl
git apply /home/work/AgentSDK/aura/third_party/patch/verl.patch
cd /home/work/vllm
git apply /home/work/AgentSDK/aura/third_party/patch/vllm.patch
cd /home/work/vllm-ascend
git apply /home/work/AgentSDK/aura/third_party/patch/vllm-ascend.patch

cd /home/work/AgentSDK/aura

echo "============================================"
echo " build_env 完成"
echo "============================================"
