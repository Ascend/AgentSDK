#!/bin/bash
# This script is used to generate llt-cpp coverage.
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

http_proxy="${1}"
https_proxy="${2}"

workdir=$(
  cd $(dirname $0) || exit
  pwd
)

workdir=$workdir/..

# 设置运行测试所需要的第三方github仓库路径
THIRD_PARTY_DIR=$workdir/third-party/
mkdir -p $THIRD_PARTY_DIR

ALL_THIRD_PYTHONPATH=""

function pre_install() {
  cd $workdir

  # 安装python包
  pip3 install transformers==4.52.3 \
  sympy==1.13.1 \
  pylatexenc==2.10 \
  openai==1.99.6 \
  torch==2.5.1 \
  vertexai==1.64.0 \
  sentence_transformers==5.1.0 \
  hydra-core==1.3.2 \
  regex==2025.8.29 \
  tensordict==0.1.2 \
  word2number==1.1 \
  codetiming==1.4.0 \
  torchvision==0.20.1 \
  ray==2.42.1 \
  uvicorn==0.38.0 \
  datasets==4.4.1

  megatron_path=$workdir/AgenticRL/third-party/Megatron-LM-core_r0.8.0/
  mindspeed_path=$workdir/AgenticRL/third-party/MindSpeed-2.1.0_core_r0.8.0/
  mindspeed_llm_path=$workdir/AgenticRL/third-party/MindSpeed-LLM-2.1.0/
  mindspeed_rl_path=$workdir/AgenticRL/third-party/MindSpeed-RL-v2.2.0/
  vllm_path=$workdir/AgenticRL/third-party/vllm-releases-v0.9.1/
  vllm_ascend_path=$workdir/AgenticRL/third-party/vllm-ascend-0.9.1-dev/

  # 设置第三方库
  ALL_THIRD_PYTHONPATH=$megatron_path:$mindspeed_path:$mindspeed_llm_path:$mindspeed_rl_path:$vllm_path:$vllm_ascend_path
}

# 需要提前下载好pytest pytest-html pytest-cov
function run_test() {
  cd $workdir

  # DT环境需要提前导入此动态库
  export LD_PRELOAD=$LD_PRELOAD:/opt/buildtools/python-3.11.4/lib/python3.11/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
  export PYTHONPATH=$PYTHONPATH:$ALL_THIRD_PYTHONPATH

  echo ""
  echo "[INFO] >>>>>>>>>>> start running tests >>>>>>>>>>>"
  python3 -m pytest \
    --cov=agentic_rl/ \
    --cov-report=term \
    --cov-report=html:script/coverage/html \
    --cov-report=xml:script/coverage/coverage.xml \
    --junit-xml=script/coverage/final.xml \
    --html=script/coverage/final.html \
    --self-contained-html \
    --cov-branch \
    -vs tests/
  echo "[INFO] >>>>>>>>>>> finish running tests >>>>>>>>>>>"
  echo ""

  echo "[INFO] Coverage report generated:"
  echo "  HTML: ${workdir}/script/coverage/htmlcov/index.html"
  echo "  XML : ${workdir}/script/coverage/coverage.xml"
  echo "  JUnit: ${workdir}/script/coverage/final.xml"
  echo "  HTML test report: ${workdir}/script/coverage/final.html"

  LINE_RATE=$(grep -o 'line-rate="[^"]*"' ${workdir}/script/coverage/coverage.xml | head -1 | cut -d'"' -f2)
 	BRANCH_RATE=$(grep -o 'branch-rate="[^"]*"' ${workdir}/script/coverage/coverage.xml | head -1 | cut -d'"' -f2)

 	echo "[INFO] Line coverage   : $(awk "BEGIN {print ${LINE_RATE}*100}")%"
 	echo "[INFO] Branch coverage : $(awk "BEGIN {print ${BRANCH_RATE}*100}")%"

  echo "[INFO] Test done!"
}

pre_install
run_test
