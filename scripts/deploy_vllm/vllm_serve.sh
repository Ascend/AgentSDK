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

set -e

usage() {
    echo "Usage:"
    echo "  bash vllm_serve.sh  --role [prefill|decode] \\"
    echo "                      --host <local_ip> \\"
    echo "                      --port <port> \\"
    echo "                      --master_addr <master_ip> \\"
    echo "                      --local_node_rank <rank>"
    echo ""
    echo "Example:"
    echo "  bash vllm_serve.sh --role prefill --host 10.50.89.139 --port 20012 --master_addr 10.50.89.139 --local_node_rank 0"
    exit 1
}

HEADLESS_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --master_addr) MASTER_ADDR="$2"; shift 2 ;;
        --local_node_rank) LOCAL_NODE_RANK="$2"; shift 2 ;;
        --role) ROLE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [ -z "$HOST" ] || [ -z "$PORT" ] || [ -z "$MASTER_ADDR" ] || [ -z "$LOCAL_NODE_RANK" ] || [ -z "$ROLE" ]; then
    echo "错误: Missing required arguments (HOST, PORT, MASTER_ADDR, LOCAL_NODE_RANK, ROLE)"
    usage
fi

if [ "$LOCAL_NODE_RANK" -eq 0 ]; then
    HEADLESS_FLAG=""
else
    HEADLESS_FLAG="--headless"
fi


export HCCL_IF_IP="$HOST"
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export HCCL_BUFFSIZE="256"

export VLLM_USE_V1="1"
export HCCL_OP_EXPANSION_MODE="AIV"
if [[ ":$LD_PRELOAD:" != *":/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:"* ]]; then
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
fi

export VLLM_ASCEND_ENABLE_FLASHCOMM="1"
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE="1"
export TASK_QUEUE_ENABLE="1"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ASCEND_GLOBAL_LOG_LEVEL="3"
export OMP_PROC_BIND="false"
export ASCEND_LAUNCH_BLOCKING="0"
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT="600"
export ENABLE_VLLM_STAT="true"


if [ "$ROLE" = "prefill" ]; then
    export TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-8}
    export DATA_PARALLEL_SIZE=${PREFILL_DATA_PARALLEL_SIZE:-1}
    export DATA_PARALLEL_SIZE_LOCAL=${PREFILL_DATA_PARALLEL_SIZE_LOCAL:-1}
    export ENABLE_EXPERT_PARALLEL=${PREFILL_ENABLE_EXPERT_PARALLEL:-true}
else
    export TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-8}
    export DATA_PARALLEL_SIZE=${DECODE_DATA_PARALLEL_SIZE:-1}
    export DATA_PARALLEL_SIZE_LOCAL=${DECODE_DATA_PARALLEL_SIZE_LOCAL:-1}
    export ENABLE_EXPERT_PARALLEL=${DECODE_ENABLE_EXPERT_PARALLEL:-true}
fi

export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=${DISAGGREGATED_PREFILL_RANK_TABLE_PATH:-/ranktable.json}
export MODEL_PATH=${MODEL_PATH:-/models/l00649956/models/Qwen3-30B-A3B}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3-30B-A3B}
export DP_RPC_PORT=${DP_RPC_PORT:-13397}
export VLLM_ASCEND_LLMDD_RPC_PORT=${VLLM_ASCEND_LLMDD_RPC_PORT:-7778}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

DP_START_RANK=$(( LOCAL_NODE_RANK * DATA_PARALLEL_SIZE_LOCAL ))

if [ "$ENABLE_EXPERT_PARALLEL" = "true" ]; then
    EXPERT_PARALLEL_ARG="--enable-expert-parallel"
else
    EXPERT_PARALLEL_ARG=""
fi


echo "==================================================="
echo " Launching vLLM Node"
echo "---------------------------------------------------"
echo " PD Role                : $ROLE"
echo " Host IP                : $HOST"
echo " Port                   : $PORT"
echo " Master Address         : $MASTER_ADDR"
echo " Local Node Rank        : $LOCAL_NODE_RANK"
echo " DP Start Rank          : $DP_START_RANK (Calculated)"
echo " Headless Flag          : $HEADLESS_FLAG"
echo " TP Size                : $TENSOR_PARALLEL_SIZE"
echo " DP Size                : $DATA_PARALLEL_SIZE"
echo " DP Size Local          : $DATA_PARALLEL_SIZE_LOCAL"
echo " Expert Parallel        : $ENABLE_EXPERT_PARALLEL"
echo "==================================================="

if [ "$ROLE" = "prefill" ]; then
    vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    $HEADLESS_FLAG \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --data-parallel-size "$DATA_PARALLEL_SIZE" \
    --data-parallel-size-local "$DATA_PARALLEL_SIZE_LOCAL" \
    $EXPERT_PARALLEL_ARG \
    --data-parallel-start-rank "$DP_START_RANK" \
    --data-parallel-address "$MASTER_ADDR" \
    --data-parallel-rpc-port "$DP_RPC_PORT" \
    --max-model-len 40960 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-num-seqs 4 \
    --enable-prefix-caching \
    --additional-config '{"ascend_scheduler_config":{"enabled":true, "enable_chunked_prefill":true}}' \
    --kv-transfer-config \
    '{"kv_connector": "LLMDataDistCMgrConnector",
    "kv_buffer_device": "npu",
    "kv_role": "kv_producer",
    "kv_parallel_size": 1,
    "kv_port": "20011",
    "engine_id": "0",
    "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
    }'
else
    vllm serve "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        $HEADLESS_FLAG \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --data-parallel-size "$DATA_PARALLEL_SIZE" \
        --data-parallel-size-local "$DATA_PARALLEL_SIZE_LOCAL" \
        $EXPERT_PARALLEL_ARG \
        --data-parallel-start-rank "$DP_START_RANK" \
        --data-parallel-address "$MASTER_ADDR" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --max-model-len 40960 \
        --max-num-batched-tokens 128 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --enable-chunked-prefill \
        --max-num-seqs 12 \
        --enable-prefix-caching \
        --additional-config '{"ascend_scheduler_config":{"enabled":true, "enable_chunked_prefill":true}}' \
        --compilation_config '{"cudagraph_capture_sizes":[4,8,12,16]}' \
        --kv-transfer-config \
        '{"kv_connector": "LLMDataDistCMgrConnector",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_parallel_size": 1,
        "kv_port": "20011",
        "engine_id": "0",
        "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
        }'
fi