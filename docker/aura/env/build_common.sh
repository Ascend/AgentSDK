#!/usr/bin/env bash
set -e

cd /home/work/AgentSDK/aura
bash download_third_party.sh
pip cache purge || true
pip install -r third_party/requirements_aura.txt
pip uninstall -y torch torchvision torchaudio || true
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 -f https://mirrors.aliyun.com/pytorch-wheels/cpu
pip uninstall pyarrow
pip install pyarrow==24.0.0
pip uninstall -y triton || true
pip uninstall -y triton-ascend || true
pip install triton-ascend==3.2.0
