#!/bin/bash
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

# 训练任务启动入口

base_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))
scripts_dir=$(realpath $(dirname ${base_dir}))
root_dir=$(realpath $(dirname ${scripts_dir}))

# 统一加载环境变量（替代原有的硬编码 export），变量来源: 外部环境变量 > env.local > env.conf > 脚本默认值
source ${scripts_dir}/base/load_env.sh
