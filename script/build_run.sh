#!/bin/bash
# This script is used to build run package.
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

# 获取第一个入参，版本号
VERSION="${1:-7.0.0}"

echo $VERSION

PROCESS_DIR=$workdir/../output/process
RESOURCE_DIR=$workdir/../output/resource
TARGET_DIR=$workdir/../output/target
mkdir -p $PROCESS_DIR
mkdir -p $RESOURCE_DIR
mkdir -p $TARGET_DIR

# 拷贝脚本至process目录
cp $workdir/../opensource/makeself/makeself.sh $PROCESS_DIR
cp $workdir/../opensource/makeself/makeself-header.sh $PROCESS_DIR

# 拷贝资源至resource目录
cp $workdir/../output/Ascend-agentsdk__linux-aarch64.tar.gz $RESOURCE_DIR
cp $workdir/run/help.info $RESOURCE_DIR
cp $workdir/run/install.sh $RESOURCE_DIR

chmod 640 $RESOURCE_DIR/Ascend-agentsdk__linux-aarch64.tar.gz
chmod 500 $RESOURCE_DIR/install.sh
chmod 440 $RESOURCE_DIR/help.info

cd $RESOURCE_DIR
bash $PROCESS_DIR/makeself.sh --chown --nomd5 --sha256 --nocrc \
  --header $PROCESS_DIR/makeself-header.sh \
  --help-header help.info \
  --packaging-date "" \
  --tar-extra '--owner=root --group=root' \
  $RESOURCE_DIR \
  $TARGET_DIR/Ascend-agentsdk_${VERSION}_linux-aarch64.run \
  "ASCEND Agent SDK RUN PACKAGE" \
  ./install.sh
