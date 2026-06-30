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
THIRD_PARTY_DIR=$workdir/aura/third-party/
#mkdir -p $THIRD_PARTY_DIR

# ALL_THIRD_PYTHONPATH=""

function pre_install() {
 cd $workdir

 # 安装python包
echo "======================================"
echo "[INFO] Installing python packages"
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
 pyyaml==6.0.0
 echo "[INFO] >>>>>>>>>>> finish install python packages >>>>>>>>>>>"
 mkdir -p $THIRD_PARTY_DIR
 cd $THIRD_PARTY_DIR
 git clone https://gitee.com/mirrors/rLLM rllm
 cd rllm
 git checkout v0.1
 rllm_path=$THIRD_PARTY_DIR/rllm/
 export PYTHONPATH=$PYTHONPATH:$rllm_path
#  megatron_path=$workdir/aura/third-party/Megatron-LM-core_r0.8.0/
#  mindspeed_path=$workdir/aura/third-party/MindSpeed-2.1.0_core_r0.8.0/
#  mindspeed_llm_path=$workdir/aura/third-party/MindSpeed-LLM-2.1.0/
#  mindspeed_rl_path=$workdir/aura/third-party/MindSpeed-RL-v2.2.0/
#  vllm_path=$workdir/aura/third-party/vllm-releases-v0.9.1/
#  vllm_ascend_path=$workdir/aura/third-party/vllm-ascend-0.9.1-dev/

#  # 设置第三方库
#  ALL_THIRD_PYTHONPATH=$megatron_path:$mindspeed_path:$mindspeed_llm_path:$mindspeed_rl_path:$vllm_path:$vllm_ascend_path
}

# 需要提前下载好pytest pytest-html pytest-cov
function run_test() {
  cd $workdir

  # DT环境需要提前导入此动态库
  export LD_PRELOAD=$LD_PRELOAD:/opt/buildtools/python-3.11.4/lib/python3.11/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
#  export PYTHONPATH=$PYTHONPATH:$ALL_THIRD_PYTHONPATH

  # ----------------------------------------------------------------------------
  # Change analysis: decide which module's UT needs to run.
  # The change file is produced by `git diff --name-only --no-commit-id`, so
  # each non-empty line is just a repository-relative file path, e.g.:
  #     aura/setup.py
  #     openclaw/skillhub/cli.py
  # There are NO `diff --git` headers. We detect module scope by checking
  # whether any changed path starts with `aura/` or `openclaw/`.
  # ----------------------------------------------------------------------------
  CHANGE_FILE="/workspace/change.txt"
  RUN_AURA_UT=false
  RUN_OPENCLAW_UT=false

  echo ""
  echo "======================================"
  echo "[INFO] UT change analysis"
  echo "======================================"
  echo "[INFO] Change file : $CHANGE_FILE"

  if [ ! -f "$CHANGE_FILE" ]; then
    echo "[ERROR] change file not found: $CHANGE_FILE"
    echo "[ERROR] Cannot determine which module to test, aborting."
    exit 1
  fi

  echo "[INFO] Collecting changed file paths (name-only format)..."
  mapfile -t CHANGED_FILES < <(grep -vE "^[[:space:]]*$" "$CHANGE_FILE" || true)
  echo "[INFO] Found ${#CHANGED_FILES[@]} changed file(s) in the change file."
  if [ "${#CHANGED_FILES[@]}" -gt 0 ]; then
    echo "[INFO] Changed file list:"
    for f in "${CHANGED_FILES[@]}"; do
      echo "  - $f"
    done
  fi

  for line in "${CHANGED_FILES[@]}"; do
    if [[ "$line" == aura/* ]]; then
      RUN_AURA_UT=true
      break
    fi
  done

  for line in "${CHANGED_FILES[@]}"; do
    if [[ "$line" == openclaw/* ]]; then
      RUN_OPENCLAW_UT=true
      break
    fi
  done

  echo ""
  echo "[INFO] Diff analysis result:"
  if [ "$RUN_AURA_UT" = true ]; then
    echo "  - aura     : CHANGED  -> aura UT will be executed"
  else
    echo "  - aura     : untouched -> aura UT skipped"
  fi
  if [ "$RUN_OPENCLAW_UT" = true ]; then
    echo "  - openclaw : CHANGED  -> openclaw UT will be executed"
  else
    echo "  - openclaw : untouched -> openclaw UT skipped"
  fi

  if [ "$RUN_AURA_UT" = false ] && [ "$RUN_OPENCLAW_UT" = false ]; then
    echo ""
    echo "[INFO] No changes detected under aura/ or openclaw/."
    echo "[INFO] Nothing to do for UT, exiting successfully."
    echo "======================================"
    echo "[SUCCESS] UT skipped (no relevant changes)"
    echo "======================================"
    return 0
  fi

  # ----------------------------------------------------------------------------
  # aura UT
  # ----------------------------------------------------------------------------
  if [ "$RUN_AURA_UT" = true ]; then
    pre_install
    cd $workdir/aura
    echo ""
    echo "======================================"
    echo "[INFO] >>>>>>>>>>> start running aura tests >>>>>>>>>>>"
    echo "======================================"
    python3 -m pytest \
      --cov=aura/ \
      --cov-report=term \
      --cov-report=html:script/coverage/html \
      --cov-report=xml:script/coverage/coverage.xml \
      --junit-xml=script/coverage/final.xml \
      --html=script/coverage/final.html \
      --self-contained-html \
      --cov-branch \
      -vs tests/
    echo "[INFO] >>>>>>>>>>> finish running aura tests >>>>>>>>>>>"
    echo ""

    echo "[INFO] Aura coverage report generated:"
    echo "  HTML: ${workdir}/script/coverage/htmlcov/index.html"
    echo "  XML : ${workdir}/script/coverage/coverage.xml"
    echo "  JUnit: ${workdir}/script/coverage/final.xml"
    echo "  HTML test report: ${workdir}/script/coverage/final.html"

    LINE_RATE=$(grep -o 'line-rate="[^"]*"' ${workdir}/script/coverage/coverage.xml | head -1 | cut -d'"' -f2)
   	BRANCH_RATE=$(grep -o 'branch-rate="[^"]*"' ${workdir}/script/coverage/coverage.xml | head -1 | cut -d'"' -f2)

   	echo "[INFO] Aura line coverage   : $(awk "BEGIN {print ${LINE_RATE}*100}")%"
   	echo "[INFO] Aura branch coverage : $(awk "BEGIN {print ${BRANCH_RATE}*100}")%"
  else
    echo ""
    echo "[INFO] Skipping aura UT (no aura changes detected)."
  fi

  # ----------------------------------------------------------------------------
  # openclaw UT
  # ----------------------------------------------------------------------------
  if [ "$RUN_OPENCLAW_UT" = true ]; then
    unset LD_PRELOAD
    cd $workdir/openclaw
    echo ""
    echo "======================================"
    echo "[INFO] >>>>>>>>>>> installing openclaw dependencies >>>>>>>>>>>"
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
    echo "  HTML: ${workdir}/script/coverage/openclaw/htmlcov/index.html"
    echo "  XML : ${workdir}/script/coverage/openclaw/coverage.xml"
    echo "  JUnit: ${workdir}/script/coverage/openclaw/final.xml"
    echo "  HTML test report: ${workdir}/script/coverage/openclaw/final.html"

    OPENCLAW_LINE_RATE=$(grep -o 'line-rate="[^"]*"' ${workdir}/script/coverage/openclaw/coverage.xml | head -1 | cut -d'"' -f2)
    OPENCLAW_BRANCH_RATE=$(grep -o 'branch-rate="[^"]*"' ${workdir}/script/coverage/openclaw/coverage.xml | head -1 | cut -d'"' -f2)

    echo "[INFO] Openclaw line coverage   : $(awk "BEGIN {print ${OPENCLAW_LINE_RATE}*100}")%"
    echo "[INFO] Openclaw branch coverage : $(awk "BEGIN {print ${OPENCLAW_BRANCH_RATE}*100}")%"
  else
    echo ""
    echo "[INFO] Skipping openclaw UT (no openclaw changes detected)."
  fi

  echo ""
  echo "======================================"
  echo "[INFO] UT summary"
  echo "======================================"
  if [ "$RUN_AURA_UT" = true ]; then
    echo "  - aura UT     : DONE"
  else
    echo "  - aura UT     : SKIPPED"
  fi
  if [ "$RUN_OPENCLAW_UT" = true ]; then
    echo "  - openclaw UT : DONE"
  else
    echo "  - openclaw UT : SKIPPED"
  fi
  echo "[INFO] All requested tests done!"
  echo "======================================"
}

echo ""
echo "======================================"
echo "[INFO] test.sh start"
echo "======================================"
echo "[INFO] http_proxy  : ${http_proxy}"
echo "[INFO] https_proxy : ${https_proxy}"
echo "[INFO] workdir     : ${workdir}"

run_test

echo ""
echo "======================================"
echo "[SUCCESS] test.sh finished"
echo "======================================"
