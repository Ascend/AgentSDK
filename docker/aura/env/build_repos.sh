#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

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
