#!/usr/bin/env bash
set -e

echo "============================================"
echo " AgentSDK clone 依赖仓库"
echo "============================================"

cd /home/work

git clone https://gitcode.com/GitHub_Trending/vl/vllm.git
git clone https://gitcode.com/gh_mirrors/vl/vllm-ascend.git
git clone https://gitcode.com/Ascend/MindSpeed.git
cd /home/work/MindSpeed && git checkout 2.3.0_core_r0.12.1
cd ..

git clone --depth 1 --branch core_v0.12.1 https://gitcode.com/GitHub_Trending/me/Megatron-LM.git

git clone https://github.com/verl-project/verl.git /verl
git clone https://github.com/huggingface/transformers.git /home/work/transformers

echo "[clone] 所有依赖仓库就绪"
