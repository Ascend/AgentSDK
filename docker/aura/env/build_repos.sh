#!/usr/bin/env bash
set -e

echo "============================================"
echo " AgentSDK clone 依赖仓库"
echo "============================================"

cd /home/work

git clone --depth 1 -b v0.22.1 https://gitcode.com/GitHub_Trending/vl/vllm.git
git clone --depth 1 -b v0.22.1rc1 https://gitcode.com/gh_mirrors/vl/vllm-ascend.git
git clone --depth 1 -b 2.3.0_core_r0.12.1 https://gitcode.com/Ascend/MindSpeed.git
git clone --depth 1 --branch core_v0.12.1 https://gitcode.com/GitHub_Trending/me/Megatron-LM.git
git clone https://gitcode.com/GitHub_Trending/ve/verl.git /verl

echo "[clone] 所有依赖仓库就绪"
