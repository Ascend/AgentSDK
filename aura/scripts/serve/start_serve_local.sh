#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
# 训练任务启动入口

me=$(basename $0)
# start_serve_local.sh 位于 aura/scripts/serve/, 需向上两级得到 aura/
serve_dir=$(realpath $(dirname $0))
scripts_dir=$(realpath $(dirname ${serve_dir}))
root_dir=$(realpath $(dirname ${scripts_dir}))

# 统一加载环境变量（替代原有的硬编码 export）
# VLLM_VERSION / RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES / 路径 /
# GLOO_SOCKET_IFNAME / TP_SOCKET_IFNAME 等均由 env.conf 统一配置
source ${scripts_dir}/base/load_env.sh

# 本脚本为本地调测专用，bond19 为本脚本硬编码的本地调测兜底默认值
# 注意：此处不使用 env.conf 的 DEFAULT_SOCKET_IFNAME(eth0)，因为该值为训练/推理
# 集群场景的默认值，与本地调测环境无关。用户可通过 env.local 覆盖 GLOO/TP 变量
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"bond19"}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-"bond19"}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export PYTHONPATH=${RLLM_PATH}:${VLLM_PATH}:${VLLM_ASCEND_PATH}:${MINDSPEED_RL_PATH}:${MEGATRON_PATH}:${MINDSPEED_PATH}:${MINDSPEED_LLM_PATH}:${PYTHONPATH}

logs_path=./logs/

if [ ! -d "$logs_path" ]; then
    mkdir -p "$logs_path"
    echo "dir created: $logs_path"
else
    echo "dir exists: $logs_path"
fi

function show_help()
{
    echo "usage: ${me} <master {master_ip}|worker {master_ip}>"
}

if [[ $1 == "" || $2 == "" ]]; then
  show_help
  exit 0
fi

ray stop
if [[ $1 == "master" ]]; then
  ray start --head --port 7099
else
  ray start --address="$2:7099"

  # 非0结点循环检查ray集群状态
  while true; do
    ray status > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
    sleep 30
  done
fi

if [[ $1 == "master" ]]; then
  timestamp=$(date +%s%3N)
  python aura/start.py --config-name=serve_1node_qwen235b 2>&1 | tee ${logs_path}/logs_${timestamp}.log
  ray stop
fi
