#!/bin/bash
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
THIRD_PARTY_DIR=$workdir/aura/third-party/

function pre_install() {
 cd $workdir

 # 安装python包
echo "======================================"
echo "[INFO] Installing aura python packages"
echo "======================================"
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
 ray==2.55.1 \
 uvicorn==0.38.0 \
 datasets==4.4.1 \
 tensorboard==2.20.0 \
 firecrawl \
 pytest-asyncio==1.3.0 \
 pillow==12.2.0 \
 typer==0.15.0 \
 click==8.1.7 \
 rich==13.0.0 \
 pydantic==2.0.0 \
 pydantic-settings==2.0.0 \
 httpx[http2]==0.24.0 \
 keyring==23.0.0 \
 keyrings.alt==4.0.0 \
 diskcache==5.0.0 \
 platformdirs==3.0.0 \
 pyyaml==6.0.0 \
 prometheus_client==0.25.0
 echo "[INFO] >>>>>>>>>>> finish install python packages >>>>>>>>>>>"
 mkdir -p $THIRD_PARTY_DIR
 cd $THIRD_PARTY_DIR
 git clone https://gitee.com/mirrors/rLLM rllm
 cd rllm
 git checkout v0.1
 rllm_path=$THIRD_PARTY_DIR/rllm/
 export PYTHONPATH=$PYTHONPATH:$rllm_path
}

# 需要提前下载好pytest pytest-html pytest-cov
function run_test() {
  cd $workdir

  # DT环境需要提前导入此动态库
  export LD_PRELOAD=$LD_PRELOAD:/opt/buildtools/python-3.11.4/lib/python3.11/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0

  cd $workdir/aura
  echo ""
  echo "======================================"
  echo "[INFO] >>>>>>>>>>> start running aura tests >>>>>>>>>>>"
  echo "======================================"
  python3 -m pytest \
    --cov=aura/ \
    --cov-report=term \
    --cov-report=html:../script/coverage/html \
    --cov-report=xml:../script/coverage/coverage.xml \
    --junit-xml=../script/coverage/final.xml \
    --html=../script/coverage/final.html \
    --self-contained-html \
    --cov-branch \
    -vs tests/
  echo "[INFO] >>>>>>>>>>> finish running aura tests >>>>>>>>>>>"
  echo ""

  echo "[INFO] Aura coverage report generated:"
  echo "  HTML: ${workdir}/script/coverage/html/index.html"
  echo "  XML : ${workdir}/script/coverage/coverage.xml"
  echo "  JUnit: ${workdir}/script/coverage/final.xml"
  echo "  HTML test report: ${workdir}/script/coverage/final.html"

  LINE_RATE=$(grep -o 'line-rate="[^"]*"' ${workdir}/script/coverage/coverage.xml | head -1 | cut -d'"' -f2)
 	BRANCH_RATE=$(grep -o 'branch-rate="[^"]*"' ${workdir}/script/coverage/coverage.xml | head -1 | cut -d'"' -f2)

 	echo "[INFO] Aura line coverage   : $(awk "BEGIN {print ${LINE_RATE}*100}")%"
 	echo "[INFO] Aura branch coverage : $(awk "BEGIN {print ${BRANCH_RATE}*100}")%"

  echo ""
  echo "======================================"
  echo "[SUCCESS] Aura UT finished"
  echo "======================================"
}

echo ""
echo "======================================"
echo "[INFO] aura_ut.sh start"
echo "======================================"
echo "[INFO] http_proxy  : ${http_proxy}"
echo "[INFO] https_proxy : ${https_proxy}"
echo "[INFO] workdir     : ${workdir}"

pre_install
run_test
