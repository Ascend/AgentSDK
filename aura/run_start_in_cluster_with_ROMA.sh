#!/bin/bash

# 确保脚本在遇到任何错误时立即停止执行
set -e

export VLLM_ASCEND_ENABLE_NZ=0
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

sed -i '619s|return JSONResponse(content=generator.model_dump())|try:\n            return JSONResponse(content=generator.model_dump())\n        except Exception as e:\n            content = generator.model_dump()\n            logger.error(f"JSON Serialization Error: {e}, Content: {content}")\n            return JSONResponse(content=handle_inf(content))|' /vllm/vllm/entrypoints/openai/api_server.py
sed -i "119r ${SCRIPT_DIR}/clean_json_nan.py" /vllm/vllm/entrypoints/openai/api_server.py

# --- 1. 读取输入参数 --config-name ---
echo "### 1. 读取输入参数 --config-name"

CONFIG_NAME=""
# 遍历所有参数，查找 --config-name
for i in "$@"; do
    case $i in
        --config-name=*)
            CONFIG_NAME="${i#*=}" # 提取等号后的值
            shift # 移除已处理的参数
            ;;
        *)
            # 忽略其他参数
            ;;
    esac
done

if [ -z "$CONFIG_NAME" ]; then
    echo "错误: 缺少必要的输入参数 --config-name"
    exit 1
fi

echo "   - 配置名称 (CONFIG_NAME): ${CONFIG_NAME}"

# --- 2. 读取集群环境变量和准备 IP 列表 ---
echo "### 2. 读取集群环境变量..."

# 校验关键环境变量
if [ -z "$VC_WORKER_HOSTS" ] || [ -z "$MA_CURRENT_IP" ] || [ -z "$VC_TASK_INDEX" ]; then
    echo "错误: 缺少关键环境变量 (VC_WORKER_HOSTS, MA_CURRENT_IP, VC_TASK_INDEX)。"
    exit 1
fi

# 替换逗号为空格
NODE_DOMAIN_LIST=$(echo "$VC_WORKER_HOSTS" | tr ',' ' ' | awk '{$1=$1};1')
THIS_NODE_IP=$MA_CURRENT_IP
THIS_NODE_INDEX=$VC_TASK_INDEX

echo "   - 所有节点域名 (VC_TASK_HOSTS): ${NODE_DOMAIN_LIST}"
echo "   - 当前节点 IP (THIS_POD_IP): ${THIS_NODE_IP}"
echo "   - 当前节点编号 (THIS_TASK_INDEX): ${THIS_NODE_INDEX}"

# 将 VC_TASK_HOSTS 域名转换为 IP 列表
echo "   - 正在将域名转换为 IP 列表 (NODE_IP_LIST)..."

# 初始化一个 Bash 数组来存储 IP 地址
NODE_IP_LIST_ARRAY=()

for domain in ${NODE_DOMAIN_LIST}; do
    # 使用 python3 的 socket 模块解析域名
    IP_ADDRESS=$(python -c "import socket; print(socket.gethostbyname('$domain'))")

    # 检查解析是否成功
    if [ $? -ne 0 ] || [ -z "$IP_ADDRESS" ]; then
        echo "错误: 无法解析域名 $domain"
        exit 1
    fi
    NODE_IP_LIST_ARRAY+=("${IP_ADDRESS}")
done

# 第一个 IP 始终作为 Master 节点的 IP
MASTER_NODE_IP=${NODE_IP_LIST_ARRAY[0]}
NODE_IP_LIST="${NODE_IP_LIST_ARRAY[@]}"

echo "   - 最终 IP 列表: ${NODE_IP_LIST}"
echo "   - Master 节点 IP (第一个): ${MASTER_NODE_IP}"

# --- 3. 启动任务：设置端口和执行启动命令 ---
echo "### 3. 启动任务..."

# 使用 ${VAR:-DEFAULT_VALUE} 语法，优先使用环境变量，否则使用默认值
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

RAY_PORT=${RAY_PORT:-9777}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-9778}

echo "   - Ray 节点通信端口 (RAY_PORT): ${RAY_PORT}"
echo "   - Ray Dashboard 端口 (RAY_DASHBOARD_PORT): ${RAY_DASHBOARD_PORT}"

# 构造 Master 节点的地址
MASTER_ADDRESS="${MASTER_NODE_IP}:${RAY_PORT}"

# 根据 THIS_NODE_INDEX 决定启动模式
if [ "$THIS_NODE_INDEX" -eq 0 ]; then
    # --- Master 节点启动 ---
    echo "   -> 当前节点 (Index 0) 作为 Master 节点启动 Ray Cluster Head..."

    # 打印启动命令 (Master)
    echo "      $ sh run_start_in_local.sh \\"
    echo "        --config-name=\"${CONFIG_NAME}\" \\"
    echo "        --is-master=true \\"
    echo "        --master-addr=\"${MASTER_ADDRESS}\" \\"
    echo "        --ray-port=${RAY_PORT} \\"
    echo "        --dashboard-port=${RAY_DASHBOARD_PORT}"

    # 执行 Master 节点启动命令
    bash ${SCRIPT_DIR}/run_start_in_local.sh \
        --config-name "${CONFIG_NAME}" \
        --is-master true \
        --master-addr "${MASTER_ADDRESS}" \
        --ray-port ${RAY_PORT} \
        --dashboard-port ${RAY_DASHBOARD_PORT}
else
    # --- Worker 节点启动 ---
    echo "   -> 当前节点 (Index ${THIS_NODE_INDEX}) 作为 Worker 节点启动 Ray Cluster Node..."

    # 打印启动命令 (Worker)
    echo "      $ sh run_start_in_local.sh \\"
    echo "        --config-name=\"${CONFIG_NAME}\" \\"
    echo "        --is-master=false \\"
    echo "        --master-addr=\"${MASTER_ADDRESS}\""

    # 执行 Worker 节点启动命令
    bash ${SCRIPT_DIR}/run_start_in_local.sh \
        --config-name "${CONFIG_NAME}" \
        --is-master false \
        --master-addr "${MASTER_ADDRESS}"
fi

# --- 4. 无限睡眠命令，保持任务进程存活 ---
echo "--- 保持任务进程存活 (tail -f /dev/null) ---"
tail -f /dev/null

echo "### 任务启动命令已执行完成."