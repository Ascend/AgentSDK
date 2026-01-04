#!/bin/bash
# This script is used to generate compling.
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

workdir=$(
  cd $(dirname $0) || exit
  pwd
)

workdir=$workdir/..

# 获取第一个入参，版本号
VERSION="${1:-7.0.0}"
VERSION_FILE="${workdir}"/ci/config/config.ini
get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed -n 's/^version:[[:space:]]*//p' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  fi
}
get_version

function modify_init_py() {
  cd $workdir
  # 替换agentic_rl/__init__.py中记录的版本号，setup.py中进行解析
  sed -i "s/^__version__ = .*/__version__ = \"${VERSION}\"/" agentic_rl/__init__.py
}

function compile() {
  echo "compile AgentSDK"

  cd $workdir
  OUTPUT_DIR=$workdir/output
  mkdir -p $OUTPUT_DIR
  rm -rf $OUTPUT_DIR/*

  echo "AgentSDK: ${VERSION}" >>$workdir/output/version.info

  chmod +x $workdir/script/run/install.sh
  chmod +x $workdir/script/run/uninstall.sh

  python3 setup.py bdist_wheel
  mv dist/* $OUTPUT_DIR/

  cp -r $workdir/configs/ $OUTPUT_DIR/

  mkdir -p $OUTPUT_DIR/script/
  cp $workdir/script/run/uninstall.sh $OUTPUT_DIR/script/

  # 处理权限
  find "$OUTPUT_DIR/" -type d -name "script" -exec chmod 750 {} +
  find "$OUTPUT_DIR/" -type d -name "configs" -exec chmod 750 {} +
  find "$OUTPUT_DIR/" -type f -path "*.sh" -exec chmod 500 {} +
  find "$OUTPUT_DIR/" -type f -name "*.yaml" -exec chmod 640 {} +
  find "$OUTPUT_DIR/" -type f -name "*.info" -exec chmod 440 {} +
  find "$OUTPUT_DIR/" -type f -name "*.whl" -exec chmod 640 {} +

  TAR_NAME="Ascend-agentsdk__linux-aarch64.tar.gz"
  echo "[INFO] Packaging output to ${TAR_NAME}..."
  tar -czvf "$OUTPUT_DIR/${TAR_NAME}" --exclude="${TAR_NAME}" -C "${OUTPUT_DIR}" $(ls -A "${OUTPUT_DIR}")
}

modify_init_py
compile
