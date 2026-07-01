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

function pre_install() {
  cd $workdir

  echo "======================================"
  echo "[INFO] Installing openclaw python packages"
  echo "======================================"
  pip3 install \
    pydantic==2.0.0 \
    pydantic-settings==2.0.0 \
    httpx[http2]==0.24.0 \
    typer==0.15.0 \
    click==8.1.7 \
    rich==13.0.0 \
    keyring==23.0.0 \
    keyrings.alt==4.0.0 \
    diskcache==5.0.0 \
    platformdirs==3.0.0 \
    pyyaml==6.0.0
  echo "[INFO] >>>>>>>>>>> finish install python packages >>>>>>>>>>>"
}

# 需要提前下载好pytest pytest-html pytest-cov
function run_test() {
  cd $workdir

  unset LD_PRELOAD
  cd $workdir/openclaw
  echo ""
  echo "======================================"
  echo "[INFO] >>>>>>>>>>> start running openclaw tests >>>>>>>>>>>"
  echo "======================================"
  python3 -m pytest \
    --cov=skillhub \
    --cov-report=term \
    --cov-report=html:../script/coverage/openclaw/html \
    --cov-report=xml:../script/coverage/openclaw/coverage.xml \
    --junit-xml=../script/coverage/openclaw/final.xml \
    --html=../script/coverage/openclaw/final.html \
    --self-contained-html \
    --cov-branch \
    -vs tests/ \
    --tb=short
  echo "[INFO] >>>>>>>>>>> finish running openclaw tests >>>>>>>>>>>"
  echo ""

  echo "[INFO] Openclaw coverage report generated:"
  echo "  HTML: ${workdir}/script/coverage/openclaw/html/index.html"
  echo "  XML : ${workdir}/script/coverage/openclaw/coverage.xml"
  echo "  JUnit: ${workdir}/script/coverage/openclaw/final.xml"
  echo "  HTML test report: ${workdir}/script/coverage/openclaw/final.html"

  OPENCLAW_LINE_RATE=$(grep -o 'line-rate="[^"]*"' ${workdir}/script/coverage/openclaw/coverage.xml | head -1 | cut -d'"' -f2)
  OPENCLAW_BRANCH_RATE=$(grep -o 'branch-rate="[^"]*"' ${workdir}/script/coverage/openclaw/coverage.xml | head -1 | cut -d'"' -f2)

  echo "[INFO] Openclaw line coverage   : $(awk "BEGIN {print ${OPENCLAW_LINE_RATE}*100}")%"
  echo "[INFO] Openclaw branch coverage : $(awk "BEGIN {print ${OPENCLAW_BRANCH_RATE}*100}")%"

  echo ""
  echo "======================================"
  echo "[SUCCESS] Openclaw UT finished"
  echo "======================================"
}

echo ""
echo "======================================"
echo "[INFO] openclaw_ut.sh start"
echo "======================================"
echo "[INFO] http_proxy  : ${http_proxy}"
echo "[INFO] https_proxy : ${https_proxy}"
echo "[INFO] workdir     : ${workdir}"

pre_install
run_test
