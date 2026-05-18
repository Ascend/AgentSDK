#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

set -e

########################################
# 参数解析
########################################
function usage()
{
    echo "Usage:"
    echo "  bash vllm_serve.sh  --role [prefill|decode] \\"
    echo "                      --host <local_ip> \\"
    echo "                      --port <port> \\"
    echo "                      --master_addr <master_ip> \\"
    echo "                      --local_node_rank <rank>"
    echo ""
    echo "Example:"
    echo "  bash vllm_serve.sh  --host 10.50.89.139 --port 20012 --master_addr 10.50.89.139 --local_node_rank 0 --role prefill --engine_id 0"
    exit 1
}

# 默认参数
HEADLESS_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --master_addr) MASTER_ADDR="$2"; shift 2 ;;
        --local_node_rank) LOCAL_NODE_RANK="$2"; shift 2 ;;
        --role) ROLE="$2"; shift 2 ;;
        --engine_id) ENGINE_ID="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# 参数校验
if [ -z "$HOST" ] || [ -z "$PORT" ] || [ -z "$MASTER_ADDR" ] || [ -z "$LOCAL_NODE_RANK" ] || [ -z "$ROLE" ]; then
    echo "❌ Missing required arguments (HOST, PORT, MASTER_ADDR, LOCAL_NODE_RANK, ROLE)."
    usage
fi

API_SERVER_CNT="--api-server-count 1"
# 根据 local_node_rank 设置 HEADLESS_FLAG
# local_node_rank 为 0 时，作为实例 Master 启动，启用 --headless
if [ "$LOCAL_NODE_RANK" -eq 0 ]; then
    HEADLESS_FLAG=""
else
    HEADLESS_FLAG="--headless"
    API_SERVER_CNT=""
fi

########################################
# 公共环境变量 (不允许外部覆盖，使用硬编码值)
########################################

# 公共网络/HCCL/NPU 配置
export HCCL_IF_IP="$HOST"
export GLOO_SOCKET_IFNAME="${DEFAULT_SOCKET_IFNAME}"
export TP_SOCKET_IFNAME="${DEFAULT_SOCKET_IFNAME}"
export HCCL_SOCKET_IFNAME="${DEFAULT_SOCKET_IFNAME}"
export HCCL_BUFFSIZE="256"

export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

# VLLM/NPU 内部标志/优化配置
export USE_VLLM_OPT=${USE_VLLM_OPT:-0}

export VLLM_USE_V1="1"
export HCCL_OP_EXPANSION_MODE="AIV"
# 追加 jemalloc 预加载
#if [[ ":$LD_PRELOAD:" != *":/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:"* ]]; then
#    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
#fi

export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE="1"
export TASK_QUEUE_ENABLE="1"
#export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ASCEND_GLOBAL_LOG_LEVEL="3"
export OMP_PROC_BIND="false"
export ASCEND_LAUNCH_BLOCKING="0"
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT="600"
export MC_TRANSFER_TIMEOUT=300
export HCCL_INTRA_ROCE_ENABLE=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=${SCRIPT_DIR}/../../:$PYTHONPATH

########################################
# 可自定义配置 (已有环境变量值优先，否则使用默认值)
########################################

# VLLM 分布式配置
export KV_BACKEND=${KV_BACKEND:-"llmdatadist"}
if [ "$ROLE" = "prefill" ]; then
  export TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-8}
  export DATA_PARALLEL_SIZE=${PREFILL_DATA_PARALLEL_SIZE:-1}
  export DATA_PARALLEL_SIZE_LOCAL=${PREFILL_DATA_PARALLEL_SIZE_LOCAL:-1}
else
  export TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-8}
  export DATA_PARALLEL_SIZE=${DECODE_DATA_PARALLEL_SIZE:-1}
  export DATA_PARALLEL_SIZE_LOCAL=${DECODE_DATA_PARALLEL_SIZE_LOCAL:-1}
fi

# 模型和路径配置
export ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-true}
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=${DISAGGREGATED_PREFILL_RANK_TABLE_PATH:-/ranktable.json}
export MODEL_PATH=${MODEL_PATH:-""}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-""}
export DP_RPC_PORT=${DP_RPC_PORT:-13397}
export VLLM_ASCEND_LLMDD_RPC_PORT=${VLLM_ASCEND_LLMDD_RPC_PORT:-7778}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOOL_CALL_ENABLE=${TOOL_CALL_ENABLE:-false}

# DP_START_RANK 计算: DP_START_RANK = local_node_rank * DATA_PARALLEL_SIZE_LOCAL
DP_START_RANK=$(( LOCAL_NODE_RANK * DATA_PARALLEL_SIZE_LOCAL ))

# 将 ENABLE_EXPERT_PARALLEL 转换为 VLLM 命令行参数
if [ "$ENABLE_EXPERT_PARALLEL" = "true" ]; then
    EXPERT_PARALLEL_ARG="--enable-expert-parallel"
else
    EXPERT_PARALLEL_ARG=""
fi

# 转换模型名称为全小写
served_model_name="${SERVED_MODEL_NAME,,}"
if [ "$TOOL_CALL_ENABLE" = "true" ]; then
    # 匹配 qwen3 或 qwen3.5 (支持 235b, 30b, 7b 等所有尺寸)
    if [[ "$served_model_name" == *"qwen35"* || "$served_model_name" == *"qwen3.5"* ]]; then
        TOOL_CALL_ARGS="--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice"
    # 其他默认情况（可选，通常默认回退到 hermes）
    else
        TOOL_CALL_ARGS="--tool-call-parser hermes --enable-auto-tool-choice"
    fi
else
    TOOL_CALL_ARGS=""
fi



########################################
# 信息展示
########################################
echo "============================================"
echo " Launching vLLM Node"
echo "--------------------------------------------"
echo " PD Role             : $ROLE"
echo " Host IP             : $HOST"
echo " Port                : $PORT"
echo " Master Address      : $MASTER_ADDR"
echo " Local Node Rank     : $LOCAL_NODE_RANK"
echo " DP Start Rank       : $DP_START_RANK (Calculated)"
echo " Headless Flag       : $HEADLESS_FLAG"
echo " TP Size             : $TENSOR_PARALLEL_SIZE"
echo " DP Size             : $DATA_PARALLEL_SIZE"
echo " DP Size Local       : $DATA_PARALLEL_SIZE_LOCAL"
echo " Expert Parallel     : $ENABLE_EXPERT_PARALLEL"
echo " tool call           : $TOOL_CALL_ARGS"
echo "============================================"

########################################
# 启动 vLLM 服务
########################################
function get_mooncake_kv_transfer_conf()
{
  tp_size=$1
  dp_size=$2
  engine_id=$3
  KV_PORT=$((21022 + engine_id * 2))
  KV_TRANSFER_CONFIG_TEMP='{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "kv_port": "21022",
    "engine_id": "ENGINE_ID_PLACEHOLDER",
    "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {
        "dp_size": DP_SIZE,
        "tp_size": TP_SIZE
      },
      "decode": {
        "dp_size": DP_SIZE,
        "tp_size": TP_SIZE
      }
    }
  }'
  KV_TRANSFER_CONFIG=$(echo $KV_TRANSFER_CONFIG_TEMP | sed "s/ENGINE_ID_PLACEHOLDER/$engine_id/g")
  KV_TRANSFER_CONFIG=$(echo $KV_TRANSFER_CONFIG | sed "s/21022/$KV_PORT/g")
  KV_TRANSFER_CONFIG=$(echo $KV_TRANSFER_CONFIG | sed "s/DP_SIZE/$dp_size/g")
  KV_TRANSFER_CONFIG=$(echo $KV_TRANSFER_CONFIG | sed "s/TP_SIZE/$tp_size/g")
  echo ${KV_TRANSFER_CONFIG}
}

function get_llmdatadist_kv_transfer_conf()
{
  KV_TRANSFER_CONFIG='{
    "kv_connector": "LLMDataDistCMgrConnector",
    "kv_buffer_device": "npu",
    "kv_role": "kv_consumer",
    "kv_parallel_size": 1,
    "kv_port": "21022",
    "engine_id": "0",
    "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
    }'
  echo $KV_TRANSFER_CONFIG
}

function start_vllm_serve_separate()
{
  KV_TRANSFER_CONFIG=$(get_llmdatadist_kv_transfer_conf)
  if [ $KV_BACKEND = "mooncake" ]; then
    KV_TRANSFER_CONFIG=$(get_mooncake_kv_transfer_conf $TENSOR_PARALLEL_SIZE $DATA_PARALLEL_SIZE $ENGINE_ID)
  fi
  if [ "$ROLE" = "prefill" ]; then
      KV_TRANSFER_CONFIG=$(echo $KV_TRANSFER_CONFIG | sed "s/kv_consumer/kv_producer/g")
      export TASK_QUEUE_ENABLE="2"
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
          --max-model-len ${MAX_MODEL_LEN} \
          --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
          --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
          ${TOOL_CALL_ARGS} \
          --trust-remote-code \
          --enforce-eager \
          --enable-chunked-prefill \
          --max-num-seqs ${MAX_NUM_SEQS} \
          --enable-prefix-caching \
          --worker_extension_cls "aura.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions" \
          --additional-config '{"ascend_scheduler_config":{"enabled":true,"enable_chunked_prefill":true}}' \
          --kv-transfer-config "$KV_TRANSFER_CONFIG"
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
          --max-model-len ${MAX_MODEL_LEN} \
          --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
          --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
          ${TOOL_CALL_ARGS} \
          --trust-remote-code \
          --enable-chunked-prefill \
          --max-num-seqs ${MAX_NUM_SEQS} \
          --enable-prefix-caching \
          --worker_extension_cls "aura.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions" \
          --additional-config '{"ascend_scheduler_config":{"enabled":true,"enable_chunked_prefill":true}}' \
          --compilation_config '{"cudagraph_capture_sizes":'"$CUDAGRAPH_CAPTURE_SIZES"',"cudagraph_mode":"FULL_DECODE_ONLY"}' \
          --kv-transfer-config "$KV_TRANSFER_CONFIG"
  fi
}

########################################
# 启动 vLLM PD混合 服务
########################################
function start_vllm_serve_hybrid()
{
  export HCCL_BUFFSIZE=256
  vllm serve "$MODEL_PATH" \
          --served-model-name "$SERVED_MODEL_NAME" \
          --host "$HOST" \
          --port "$PORT" \
          $HEADLESS_FLAG \
          --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
          --data-parallel-size "$DATA_PARALLEL_SIZE" \
          --data-parallel-size-local "$DATA_PARALLEL_SIZE_LOCAL" \
          $EXPERT_PARALLEL_ARG \
          ${TOOL_CALL_ARGS} \
          --data-parallel-start-rank "$DP_START_RANK" \
          --data-parallel-address "$MASTER_ADDR" \
          --data-parallel-rpc-port "$DP_RPC_PORT" \
          --max-model-len ${MAX_MODEL_LEN} \
          --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
          --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
          --trust-remote-code \
          --enable-chunked-prefill \
          ${API_SERVER_CNT} \
          --max-num-seqs ${MAX_NUM_SEQS} \
          --enable-prefix-caching \
          --worker_extension_cls "aura.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions" \
          --additional-config '{"ascend_scheduler_config":{"enabled":true,"enable_chunked_prefill":true}}' \
          --compilation_config '{"cudagraph_capture_sizes":'"$CUDAGRAPH_CAPTURE_SIZES"', "pass_config": {"enable_sp": true}}'
}

function start_vllm_serve_hybrid_opt()
{
  export HCCL_BUFFSIZE=512
  ENABLE_CPU_BINDING_ARGS='"enable_cpu_binding":true'
  CUDAGRAPH_FULL_DECODE_FULL_ARGS='"cudagraph_mode":"FULL_DECODE_ONLY"'
  ASYNC_SCHEDULING_ARGS="--async-scheduling"
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export VLLM_ASCEND_ENABLE_NZ=2
  unset VLLM_LOGGING_CONFIG_PATH

  echo "============================================"
  echo " VLLM OPT Setting"
  echo "--------------------------------------------"
  echo " Cpu Binding            : $ENABLE_CPU_BINDING_ARGS"
  echo " Cuda Graph             : $CUDAGRAPH_FULL_DECODE_FULL_ARGS"
  echo " Async Scheduling       : $ASYNC_SCHEDULING_ARGS"
  echo " Hccl Buffsize          : $HCCL_BUFFSIZE"
  echo " Gpu Memory utilization : $GPU_MEMORY_UTILIZATION"
  echo "============================================"

  vllm serve "$MODEL_PATH" \
          --served-model-name "$SERVED_MODEL_NAME" \
          --host "$HOST" \
          --port "$PORT" \
          $HEADLESS_FLAG \
          --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
          --data-parallel-size "$DATA_PARALLEL_SIZE" \
          --data-parallel-size-local "$DATA_PARALLEL_SIZE_LOCAL" \
          $EXPERT_PARALLEL_ARG \
          ${TOOL_CALL_ARGS} \
          --data-parallel-start-rank "$DP_START_RANK" \
          --data-parallel-address "$MASTER_ADDR" \
          --data-parallel-rpc-port "$DP_RPC_PORT" \
          --max-model-len ${MAX_MODEL_LEN} \
          --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
          --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
          --trust-remote-code \
          --enable-chunked-prefill \
          --max-num-seqs ${MAX_NUM_SEQS} \
          --enable-prefix-caching \
          ${API_SERVER_CNT} \
          --async-scheduling \
          --worker_extension_cls "aura.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions" \
          --additional-config '{"ascend_scheduler_config":{"enabled":true,"enable_chunked_prefill":true},"enable_cpu_binding":true}' \
          --compilation_config '{"cudagraph_capture_sizes":'"$CUDAGRAPH_CAPTURE_SIZES"',"cudagraph_mode":"FULL_DECODE_ONLY"}'
}

########################################
# 启动 vLLM 服务
########################################
echo "========USE_PD:$USE_PD, 1:PD separate, 0:PD hybrid"
if [ "$USE_PD" -eq 1 ];then
  start_vllm_serve_separate
else
  echo "============USE_VLLM_OPT:$USE_VLLM_OPT"
  if [ "$USE_VLLM_OPT" = "false" ];then
    start_vllm_serve_hybrid
  else
    start_vllm_serve_hybrid_opt
  fi
fi
