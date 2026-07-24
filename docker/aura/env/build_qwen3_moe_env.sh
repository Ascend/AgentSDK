#!/usr/bin/env bash
set -e

echo "============================================"
echo " AgentSDK build_env (Qwen3-MoE variant)"
echo "============================================"

echo "[build_env] pip 源"
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com

echo "[build_env] pip 安装 vllm / vllm-ascend / transformers"
pip install vllm==0.11.0
pip install vllm-ascend==0.11.0
pip install transformers==4.57.1

echo "[build_env] 安装 MindSpeed + Megatron-LM + mbridge"
pip install -e /home/work/MindSpeed
pip install -e /home/work/Megatron-LM
pip install mbridge

echo "[build_env] 安装 verl"
cd /verl
pip install -r requirements-npu.txt
pip install -v -e .
pip uninstall -y triton || true
pip uninstall -y triton-ascend || true

echo "[build_env] 安装 AgentSDK + third_party"
cd /home/work/AgentSDK/aura
bash download_third_party.sh
pip cache purge || true
pip install -r third_party/requirements_aura.txt
pip uninstall -y torch torchvision torchaudio || true
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 -f https://mirrors.aliyun.com/pytorch-wheels/cpu
pip uninstall pyarrow
pip install pyarrow==24.0.0
pip uninstall -y triton || true
pip uninstall -y triton-ascend || true
pip install triton-ascend==3.2.0

echo "[build_env] patch triton-ascend for CANN 9.0.0"
bash /home/work/AgentSDK/docker/aura/patch/patch_triton_ascend.sh

echo "[build_env] patch transformers for Qwen3-MoE"
bash /home/work/AgentSDK/docker/aura/patch/patch_transformers_qwen3_moe.sh

echo "============================================"
echo " build_qwen3_moe_env 完成"
echo "============================================"
