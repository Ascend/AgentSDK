#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# 训练任务启动入口

base_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))
scripts_dir=$(realpath $(dirname ${base_dir}))
root_dir=$(realpath $(dirname ${scripts_dir}))

# 统一加载环境变量（替代原有的硬编码 export），变量来源: 外部环境变量 > env.local > env.conf > 脚本默认值
source ${scripts_dir}/base/load_env.sh
