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

DEFAULT_SOCKET_IFNAME="eth0"
VLLM_PORT=20012
PROXY_PORT=8080

PREFILL_INSTANCE_COUNT=""
DECODE_INSTANCE_COUNT=""
PREFILL_CARDS_PER_INSTANCE=""
DECODE_CARDS_PER_INSTANCE=""
NODE_CARDS_COUNT=""
SOCKET_IFNAME="$DEFAULT_SOCKET_IFNAME"


echo "### 1. 解析命名参数..."

OPTIONS=p:d:s:e:c:i:
LONGOPTS=prefill-instances:,decode-instances:,prefill-cards-per-instance:,decode-cards-per-instance:,node-cards:,socket-ifname:

TEMP=$(getopt -o $OPTIONS --longoptions $LONGOPTS -n 'start_vllm_cluster.sh' -- "$@")

if [ $? != 0 ]; then
    echo "错误：参数解析失败。请使用 --help 查看使用方法。" >&2
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        -p | --prefill-instances)
            PREFILL_INSTANCE_COUNT=$2
            shift 2
            ;;
        -d | --decode-instances)
            DECODE_INSTANCE_COUNT=$2
            shift 2
            ;;
        -s | --prefill-cards-per-instance)
            PREFILL_CARDS_PER_INSTANCE=$2
            shift 2
            ;;
        -e | --decode-cards-per-instance)
            DECODE_CARDS_PER_INSTANCE=$2
            shift 2
            ;;
        -c | --node-cards)
            NODE_CARDS_COUNT=$2
            shift 2
            ;;
        -i | --socket-ifname)
            SOCKET_IFNAME=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "错误：未知参数 '$1'。"
            exit 1
            ;;
    esac
done

if [ -z "$PREFILL_INSTANCE_COUNT" ] || [ -z "$DECODE_INSTANCE_COUNT" ] || \
   [ -z "$PREFILL_CARDS_PER_INSTANCE" ] || [ -z "$DECODE_CARDS_PER_INSTANCE" ] || \
   [ -z "$NODE_CARDS_COUNT" ]; then
    echo "错误：缺少必要的参数"
    echo "必需参数：--prefill-instances, --decode-instances, --prefill-cards-per-instance, --decode-cards-per-instance, --node-cards"
    echo "可选参数：--socket-ifname（默认值：$DEFAULT_SOCKET_IFNAME）"
    exit 1
fi

PREFILL_TOTAL_CARDS=$((PREFILL_INSTANCE_COUNT * PREFILL_CARDS_PER_INSTANCE))
DECODE_TOTAL_CARDS=$((DECODE_INSTANCE_COUNT * DECODE_CARDS_PER_INSTANCE))
TOTAL_REQUIRED_CARDS=$((PREFILL_TOTAL_CARDS + DECODE_TOTAL_CARDS))

echo "  - Prefill实例数: $PREFILL_INSTANCE_COUNT"
echo "  - Decode实例数: $DECODE_INSTANCE_COUNT"
echo "  - 单个Prefill卡数: $PREFILL_CARDS_PER_INSTANCE"
echo "  - 单个Decode卡数: $DECODE_CARDS_PER_INSTANCE"
echo "  - 单节点卡数: $NODE_CARDS_COUNT"
echo "  - 网卡名称：$SOCKET_IFNAME"
echo "  - 总共需要的卡数: $TOTAL_REQUIRED_CARDS"


echo "### 2. 读取集群环境变量..."

if [ -z "$VC_TASK_HOSTS" ] || [ -z "$THIS_POD_IP" ] || [ -z "$VC_TASK_INDEX" ]; then
    echo "错误：缺少关键环境变量 (VC_TASK_HOSTS, THIS_POD_IP, VC_TASK_INDEX)"
    exit 1
fi

TASK_HOSTS_DOMAINS=${VC_TASK_HOSTS//,/ }
THIS_POD_IP=$THIS_POD_IP
VC_TASK_INDEX=$VC_TASK_INDEX

echo "  - 所有节点域名：${VC_TASK_HOSTS}"
echo "  - 当前节点IP：$THIS_POD_IP"
echo "  - 当前节点编号 (VC_TASK_INDEX)：${VC_TASK_INDEX}"

echo "  - 正在将域名转换为 IP 列表（NODE_IP_LIST)..."
NODE_IP_LIST=""
for domain in ${TASK_HOSTS_DOMAINS}; do
    IP_ADDRESS=$(python -c "import socket; print(socket.gethostbyname('$domain'))")
    if [ $? -ne 0 ] || [ -z "$IP_ADDRESS" ]; then
        echo "错误：无法解析域名 $domain"
        exit 1
    fi
    NODE_IP_LIST="${NODE_IP_LIST} ${IP_ADDRESS}"
done

NODE_IP_ARRAY=($NODE_IP_LIST)
NODE_COUNT=${#NODE_IP_ARRAY[@]}

echo "  - 节点IP列表 (NODE_IP_LIST): ${NODE_IP_LIST}"
echo "  - 节点总数 (NODE_COUNT): ${NODE_COUNT}"


echo "### 3. 执行参数校验..."

TOTAL_AVAILABLE_CARDS=$((NODE_COUNT * NODE_CARDS_COUNT))

if [ "$TOTAL_AVAILABLE_CARDS" -lt "$TOTAL_REQUIRED_CARDS" ]; then
    echo "错误：集群总卡数不足！"
    echo "  - 总可用卡数: $TOTAL_AVAILABLE_CARDS"
    echo "  - 总需求卡数: $TOTAL_REQUIRED_CARDS"
    exit 1
fi
echo "  - 校验通过：总可用卡数 ($TOTAL_AVAILABLE_CARDS) >= 总需求卡数 ($TOTAL_REQUIRED_CARDS)"

PREFILL_NODES_PER_INSTANCE=$(( (PREFILL_CARDS_PER_INSTANCE + NODE_CARDS_COUNT - 1) / NODE_CARDS_COUNT ))
DECODE_NODES_PER_INSTANCE=$(( (DECODE_CARDS_PER_INSTANCE + NODE_CARDS_COUNT - 1) / NODE_CARDS_COUNT ))

PREFILL_TOTAL_NODES=$((PREFILL_INSTANCE_COUNT * PREFILL_NODES_PER_INSTANCE))
DECODE_TOTAL_NODES=$((DECODE_INSTANCE_COUNT * DECODE_NODES_PER_INSTANCE))
TOTAL_USED_NODES=$((PREFILL_TOTAL_NODES + DECODE_TOTAL_NODES))

echo "  - 单个Prefill实例占用节点数: ${PREFILL_NODES_PER_INSTANCE}"
echo "  - 单个Decode实例占用节点数: ${DECODE_NODES_PER_INSTANCE}"
echo "  - Prefill总共占用节点数: ${PREFILL_TOTAL_NODES}"
echo "  - Decode总共占用节点数: ${DECODE_TOTAL_NODES}"


echo "### 4. 生成ranktable.json..."

LOCAL_RANKTABLE_SCRIPT="./gen_ranktable.sh"
LOCAL_RANKTABLE_PYTHON="./gen_ranktable.py"
TARGET_RANKTABLE_SCRIPT="/gen_ranktable.sh"
TARGET_RANKTABLE_PYTHON="/gen_ranktable.py"

if [ -f "$LOCAL_RANKTABLE_SCRIPT" ]; then
    echo "  - 正在调用 ${LOCAL_RANKTABLE_SCRIPT} 生成 ranktable..."

    CURRENT_DIR=$(pwd)

    echo "  - 复制脚本文件到根目录："

    cp "$LOCAL_RANKTABLE_SCRIPT" "$TARGET_RANKTABLE_SCRIPT"
    if [ $? -ne 0 ]; then
        echo "错误：复制 ${LOCAL_RANKTABLE_SCRIPT} 到根目录失败，请检查权限。"
        cd "$CURRENT_DIR"
        exit 1
    fi

    if [ -f "$LOCAL_RANKTABLE_PYTHON" ]; then
        cp "$LOCAL_RANKTABLE_PYTHON" "$TARGET_RANKTABLE_PYTHON"
        if [ $? -ne 0 ]; then
            echo "错误：复制 ${LOCAL_RANKTABLE_PYTHON} 到根目录失败，请检查权限。"
            rm -f "$TARGET_RANKTABLE_SCRIPT"
            cd "$CURRENT_DIR"
            exit 1
        fi
        echo "  - ${LOCAL_RANKTABLE_PYTHON} 复制完成。"
    else
        echo "  - ${LOCAL_RANKTABLE_PYTHON} 不存在，跳过复制。"
    fi

    echo "  - 切换到执行目录：/"
    cd /

    IPS_ARG=$(echo "${NODE_IP_LIST}" | sed 's/  */ /g')

    echo "--- 🔑 KEY COMMAND: RANKTABLE GENERATION (Executed in /) ---"
    echo "sh $TARGET_RANKTABLE_SCRIPT \\"
    echo "  --ips ${IPS_ARG} \\"
    echo "  --prefill-device-cnt ${PREFILL_TOTAL_CARDS} \\"
    echo "  --decode-device-cnt ${DECODE_TOTAL_CARDS} \\"
    echo "  --network-card-name ${SOCKET_IFNAME}"
    echo "-------------------------------------------------------------"

    sh "$TARGET_RANKTABLE_SCRIPT" \
      --ips ${IPS_ARG} \
      --prefill-device-cnt ${PREFILL_TOTAL_CARDS} \
      --decode-device-cnt ${DECODE_TOTAL_CARDS} \
      --network-card-name ${SOCKET_IFNAME}
    
    RETURN_CODE=$?

    echo "  - 清理：删除 $TARGET_RANKTABLE_SCRIPT 和 $TARGET_RANKTABLE_PYTHON"
    rm -f "$TARGET_RANKTABLE_SCRIPT"
    rm -f "$TARGET_RANKTABLE_PYTHON"

    echo "  - 返回原目录：$CURRENT_DIR"
    cd "$CURRENT_DIR"

    if [ $RETURN_CODE -ne 0 ]; then
        echo "错误：生成 ranktable失败。请检查 ${LOCAL_RANKTABLE_SCRIPT} 脚本及其依赖。"
        exit 1
    fi
    echo "  - ranktable.json 生成完成。"
else
    echo "警告：ranktable 生成脚本 ${LOCAL_RANKTABLE_SCRIPT} 不存在，跳过生成。"
fi


echo "### 5. 确定当前节点角色并启动 vLLM 实例..."

NODE_IP_ARRAY_INDEX=$VC_TASK_INDEX

if [ "$NODE_IP_ARRAY_INDEX" -lt "$PREFILL_TOTAL_NODES" ]; then
    ROLE="prefill"
    INSTANCE_INDEX=$((NODE_IP_ARRAY_INDEX / PREFILL_NODES_PER_INSTANCE))
    MASTER_NODE_INDEX=$((INSTANCE_INDEX * PREFILL_NODES_PER_INSTANCE))
    MASTER_ADDR=${NODE_IP_ARRAY[MASTER_NODE_INDEX]}
    local_node_rank=$((NODE_IP_ARRAY_INDEX - MASTER_NODE_INDEX))

elif [ "$NODE_IP_ARRAY_INDEX" -lt "$TOTAL_USED_NODES" ]; then
    ROLE="decode"
    DECODE_INDEX_OFFSET=$((NODE_IP_ARRAY_INDEX - PREFILL_TOTAL_NODES))
    INSTANCE_INDEX=$((DECODE_INDEX_OFFSET / DECODE_NODES_PER_INSTANCE))
    MASTER_NODE_INDEX=$((PREFILL_TOTAL_NODES + INSTANCE_INDEX * DECODE_NODES_PER_INSTANCE))
    MASTER_ADDR=${NODE_IP_ARRAY[MASTER_NODE_INDEX]}
    local_node_rank=$((NODE_IP_ARRAY_INDEX - MASTER_NODE_INDEX))

else
    echo "信息：当前节点（VC_TASK_INDEX=$VC_TASK_INDEX）不在任何 VLLM 实例的分配范围内，退出。"
    exit 0
fi

HOST=$THIS_POD_IP
PORT=$VLLM_PORT

echo "  - 节点角色（ROLE): ${ROLE}"
echo "  - 实例 Master 地址（MASTER_ADDR): ${MASTER_ADDR}"
echo "  - 实例内局部序号（local_node_rank): ${local_node_rank}"

if [ "$VC_TASK_INDEX" -eq 0 ]; then
    echo "### 5.2 启动负载均衡代理（仅在 VC_TASK_INDEX=0 时执行）..."

    PREFILLER_HOSTS_LIST=()
    DECODER_HOSTS_LIST=()

    for i in $(seq 0 $((PREFILL_INSTANCE_COUNT - 1))); do
        NODE_INDEX=$((i * PREFILL_NODES_PER_INSTANCE))
        PREFILLER_HOSTS_LIST+=("${NODE_IP_ARRAY[NODE_INDEX]}")
    done
    PREFILLER_HOSTS_ARG=$(IFS=' '; echo "${PREFILLER_HOSTS_LIST[*]}")
    PREFILLER_PORTS_ARG=$(seq 1 $PREFILL_INSTANCE_COUNT | xargs -I {} echo -n "$VLLM_PORT ")

    for i in $(seq 0 $((DECODE_INSTANCE_COUNT - 1))); do
        NODE_INDEX=$((PREFILL_TOTAL_NODES + i * DECODE_NODES_PER_INSTANCE))
        DECODER_HOSTS_LIST+=("${NODE_IP_ARRAY[NODE_INDEX]}")
    done
    DECODER_HOSTS_ARG=$(IFS=' '; echo "${DECODER_HOSTS_LIST[*]}")
    DECODER_PORTS_ARG=$(seq 1 $DECODE_INSTANCE_COUNT | xargs -I {} echo -n "$VLLM_PORT ")

    echo $PREFILLER_HOSTS_ARG
    echo $DECODER_PORTS_ARG

    PROXY_SCRIPT="./load_banlance_proxy_server.py"

    if [ -f "$PROXY_SCRIPT" ]; then
        echo "  - 正在后台启动代理: ${PROXY_SCRIPT}"
        python $PROXY_SCRIPT \
          --host 0.0.0.0 \
          --port ${PROXY_PORT} \
          --prefiller-hosts ${PREFILLER_HOSTS_ARG} \
          --prefiller-ports ${PREFILLER_PORTS_ARG} \
          --decoder-hosts ${DECODER_HOSTS_ARG} \
          --decoder-ports ${DECODER_PORTS_ARG} &
        
        PROXY_PID=$!
        echo "  - 负载均衡代理已在后台启动（PID：${PROXY_PID})。端口：${PROXY_PORT}"
        echo $PROXY_PID > /tmp/proxy_server.pid
    else
        echo "警告：负载均衡代理脚本 ${PROXY_SCRIPT} 不存在，跳过代理启动。"
    fi
fi

echo "### 5.3 启动 vLLM 实例（阻塞执行）..."

VLLM_SERVE_SCRIPT="./vllm_serve.sh"

if [ -f "$VLLM_SERVE_SCRIPT" ]; then
    echo "  - 正在启动 vLLM 服务：${VLLM_SERVE_SCRIPT}"

    sh $VLLM_SERVE_SCRIPT \
      --host ${HOST} \
      --port ${PORT} \
      --master_addr ${MASTER_ADDR} \
      --local_node_rank ${local_node_rank} \
      --role ${ROLE}

    if [ $? -ne 0 ]; then
        echo "错误：vLLM 服务启动失败或异常退出。"
        exit 1
    fi
    echo "信息：vLLM 服务已正常退出。"
else
    echo "错误：vLLM 启动脚本 ${VLLM_SERVE_SCRIPT} 不存在。"
    exit 1
fi