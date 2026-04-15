#!/bin/bash

# ---------------------------------------------------
# 脚本帮助信息
# ---------------------------------------------------
function show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "启动 Ray 节点，可以是 Head 节点（主节点）或 Worker 节点（工作节点）。"
    echo ""
    echo "Parameters:"
    echo "  --config-name <name>  必需。指定要运行的配置文件的名称 (例如: base_serve_1node_qwen7b_infer.yaml)。"
    echo "  --is-master <bool>    可选。如果为 'true' 或未指定，则启动 Head 节点并运行 Python 脚本。"
    echo "                        如果为 'false'，则启动 Worker 节点。"
    echo "                        (默认: true)"
    echo "  --master-addr <addr>  当 --is-master 为 'false' 时必需。指定 Head 节点的 IP 地址和端口 (例如: 192.168.1.100:6379)。"
    echo "  --ray-port <port>     可选。指定 Ray 服务的端口 (默认: 7890)。"
    echo "  --dashboard-port <port> 可选。指定 Ray Dashboard 的端口 (默认: 7891)。"
    echo "  -h, --help            显示此帮助信息。"
}

# ---------------------------------------------------
# 脚本参数和默认值设置
# ---------------------------------------------------

CONFIG_NAME=""
IS_MASTER="true"   # 默认设置为 true
MASTER_ADDR=""
RAY_PORT="7890"
DASHBOARD_PORT="7891"
REQUIRED_RAY_VERSION="2.53.0"

# 使用命名参数解析
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config-name)
            CONFIG_NAME="$2"
            shift # 跳过参数值
            ;;
        --is-master)
            IS_MASTER="$2"
            # 将输入转换为小写，方便判断
            IS_MASTER=$(echo "$IS_MASTER" | tr '[:upper:]' '[:lower:]')
            shift
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift
            ;;
        --ray-port)
            RAY_PORT="$2"
            shift
            ;;
        --dashboard-port)
            DASHBOARD_PORT="$2"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "🚨 错误：未知参数 $1"
            show_help
            exit 1
            ;;
    esac
    shift # 跳过参数名
done

# ---------------------------------------------------
# 核心参数检查
# ---------------------------------------------------

if [ -z "$CONFIG_NAME" ]; then
    echo "🚨 错误：请提供配置文件名 (--config-name)。"
    show_help
    exit 1
fi

# 检查 Worker 节点模式下的 MASTER_ADDR
if [ "$IS_MASTER" == "false" ] && [ -z "$MASTER_ADDR" ]; then
    echo "🚨 错误：当 --is-master 为 'false' 时，必须通过 --master-addr 指定主节点地址 (格式: IP:Port)。"
    show_help
    exit 1
fi

# ---------------------------------------------------
# 路径设置
# ---------------------------------------------------

# 获取当前脚本的绝对路径 (AgenticRL 路径)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "✅ 脚本当前路径 (AgenticRL): ${SCRIPT_DIR}"

# ---------------------------------------------------
# 依赖检查和安装
# ---------------------------------------------------

echo "🛠️ 检查 Ray 版本..."
INSTALLED_RAY_VERSION=$(pip show ray 2>/dev/null | grep Version | awk '{print $2}')

if [ "$INSTALLED_RAY_VERSION" == "$REQUIRED_RAY_VERSION" ]; then
    echo "👍 Ray 版本 ${REQUIRED_RAY_VERSION} 已安装且正确。"
else
    echo "⚠️ Ray 版本不匹配 (当前: ${INSTALLED_RAY_VERSION:-未安装}，要求: ${REQUIRED_RAY_VERSION})。正在安装..."
    pip install ray=="$REQUIRED_RAY_VERSION"
    if [ $? -ne 0 ]; then
        echo "❌ Ray 安装失败。请检查错误信息。"
        exit 1
    fi
    echo "✅ Ray 版本 ${REQUIRED_RAY_VERSION} 安装完成。"
fi

# ---------------------------------------------------
# 环境变量设置 (Worker 和 Master 节点都需要)
# ---------------------------------------------------

echo "🔄 设置 PYTHONPATH..."
# 设置 PYTHONPATH，使用计算出的相对路径
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/third_party/agent_engine/rllm/:${SCRIPT_DIR}/third_party/rl/mindspeed_rl/:$PYTHONPATH"
echo "✅ PYTHONPATH 设置完成。"

echo "🔄 设置 VLLM_VERSION..."
export VLLM_VERSION=0.11.0
echo "✅ VLLM_VERSION 设置为：${VLLM_VERSION}"


# ---------------------------------------------------
# Ray 服务启动逻辑
# ---------------------------------------------------

if [ "$IS_MASTER" == "true" ] || [ -z "$IS_MASTER" ]; then
    # --- Master/Head 节点逻辑 ---
    echo "☁️ 停止并启动 Ray Head 节点..."
    ray stop > /dev/null 2>&1
    echo "ℹ️ 尝试停止 Ray 服务（忽略停止失败，可能未运行）。"

    # 启动 Head 节点
    echo "➡️ 执行 Ray Head 节点启动命令..."
    ray start --head --port="$RAY_PORT" --dashboard-port="$DASHBOARD_PORT"

    if [ $? -ne 0 ]; then
        echo "❌ Ray Head 节点启动失败。请检查错误信息。"
        exit 1
    fi
    echo "✅ Ray Head 节点启动成功。服务端口: ${RAY_PORT}，Dashboard: ${DASHBOARD_PORT}。"

    # ---------------------------------------------------
    # 启动 Python 脚本
    # ---------------------------------------------------

    echo "🚀 启动 AgenticRL Python 脚本..."

    python ${SCRIPT_DIR}/agentic_rl/start.py --config-path="${SCRIPT_DIR}/configs/" --config-name="${CONFIG_NAME}"

    if [ $? -ne 0 ]; then
        echo "❌ Python 脚本执行失败。"
        exit 1
    fi
    echo "✅ Python 脚本执行完成。"

else
    # --- Worker 节点逻辑 ---
    echo "☁️ 启动 Ray Worker 节点，连接主节点 ${MASTER_ADDR}..."
    ray stop > /dev/null 2>&1
    echo "ℹ️ 尝试停止 Ray 服务（忽略停止失败，可能未运行）。"

    # 环境变量设置 (Worker 节点必需)
    # 这里的 MASTER_ADDR 格式应为 <IP>:<PORT>，Ray 启动时需要这个格式
    MASTER_IP=$(echo "$MASTER_ADDR" | cut -d: -f1)
    MASTER_PORT=$(echo "$MASTER_ADDR" | cut -d: -f2)

    if [ -z "$MASTER_PORT" ]; then
        # 如果 MASTER_ADDR 中没有端口，则假定使用默认的 Ray 端口
        MASTER_PORT="$RAY_PORT"
        MASTER_ADDR="${MASTER_IP}:${MASTER_PORT}"
    fi

    # 启动 Worker 节点，连接到主节点
    echo "➡️ 执行 Ray Worker 节点启动命令..."
    # 注意：Worker 节点不再使用 --port 和 --dashboard-port，它会自己选择可用端口
    ray start --address="${MASTER_ADDR}"

    if [ $? -ne 0 ]; then
        echo "❌ Ray Worker 节点启动失败。请检查错误信息或主节点地址。"
        exit 1
    fi
    echo "✅ Ray Worker 节点启动成功，已连接到主节点 ${MASTER_ADDR}。"

    echo "ℹ️ Worker 节点启动完毕，等待 Master 节点分配任务..."

fi

# ---------------------------------------------------