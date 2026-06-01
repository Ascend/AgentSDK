#!/bin/bash
# =============================================================================
# 参数解析 + 子命令提取
# =============================================================================

# 子命令和版本管理子命令
SUBCOMMAND=""
VERSION_SUBCMD=""
VERSION_TARGET=""
OPENCLAW_TOKEN=""
OPENCLAW_TOKEN_PER_INSTANCE=false
GUARDIAN_PORT=""
SKIP_DOWN=false
START_INDEX=1

# 自动检测已有实例的最大编号，返回下一个可用编号
detect_scale_start() {
    local max=0
    local prefix="${INSTANCE_PREFIX:-openclaw}"
    local containers
    containers=$(docker ps -a --filter "name=${prefix}-" --format "{{.Names}}" 2>/dev/null)
    if [ -n "$containers" ]; then
        while IFS= read -r name; do
            local num="${name#${prefix}-}"
            if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -gt "$max" ]; then
                max=$num
            fi
        done <<< "$containers"
    fi
    START_INDEX=$((max + 1))
    log_info "检测到已有实例最大编号: ${max}，新实例从 ${START_INDEX} 开始"
}

# 显示帮助
show_help() {
    cat << 'EOF'
OpenClaw 部署工具

用法: deploy.sh <命令> [选项]

命令:
  up        完整部署：生成配置 + 启动集群（默认）
  scale     扩容部署：自动检测已有实例，仅新增容器，不影响已有实例
  quick     单实例快速部署
  config    仅生成配置文件（不启动容器）
  start     启动已有集群
  stop      停止集群
  version   版本管理（list/current/switch/rollback）
选项:
  -n, --count NUM       实例数量（默认: 1）
  -p, --port BASE       起始端口（默认: 8080）
  -m, --model MODEL       模型名称（默认: Qwen3-30B-A3B）
  --model-provider NAME  模型供应商/提供商名称（默认: local）
  -u, --url URL           推理服务 URL（默认: http://0.0.0.0:8000）
  -c, --config-dir DIR  配置目录（默认: ./openclaw-configs）
  -i, --image IMAGE     Docker 镜像（默认: openclaw:custom）
  --name PREFIX         实例名称前缀（默认: openclaw）
  --token TOKEN         Gateway Token（不设置则自动生成）
  --skills              挂载宿主机 skills/ 目录到容器（默认: 不挂载）
  --hermes              安装并配置 Hermes Agent（默认: 不安装）
  --guardian-port PORT  API Guardian 中间件监听端口
  --memex-kb-volume DIR  Memex 知识库挂载路径（默认: 不挂载）
  --no-down             扩容模式：不关闭已有实例，仅启动新增容器
  --start-index NUM     实例编号起始值（默认: 1）
  --subnet CIDR        Docker 网络子网（默认: 172.30.0.0/16）
  -h, --help            显示帮助

示例:
  deploy.sh up -n 3 -p 18789                  # 部署3个实例
  deploy.sh scale -n 2 -p 18789               # 自动检测已有实例，扩容2个（不影响已有）
  deploy.sh scale -n 3 --image <镜像名>        # 扩容3个实例，指定镜像
  deploy.sh quick -p 18789                     # 快速部署单实例
  deploy.sh config -p 18789 -c ./my-config     # 仅生成配置
  deploy.sh version list                       # 查看可用镜像
  deploy.sh version switch openclaw:2026.4.1   # 切换版本
  deploy.sh -p 18789 -n 2                      # 兼容旧用法，等同于 up
  deploy.sh up -n 1 --guardian-port 19200       # 部署1实例 + API Guardian
EOF
}

# 显示版本管理帮助
show_version_help() {
    cat << 'EOF'
版本管理

用法: deploy.sh version <子命令>

子命令:
  list                 列出本地所有 OpenClaw 镜像
  current              显示当前使用的镜像版本
  switch <image:tag>   切换到指定镜像版本
  rollback             回滚到上一个版本

示例:
  deploy.sh version list
  deploy.sh version switch openclaw:2026.4.1-sftp-docx-browser-ccb
  deploy.sh version rollback
EOF
}

# 交互式引导：提示用户输入关键参数
interactive_prompt() {
    echo ""
    echo "OpenClaw 交互式简易部署向导"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # 推理服务 URL
    read -p "请输入推理服务 URL (无需包含 /v1 或 /anthropic 后缀) [默认: ${INFER_URL}]: " input
    if [ -n "$input" ]; then
        INFER_URL="$input"
    fi

    # 模型名称
    read -p "请输入模型名称 [默认: ${MODEL_NAME}]: " input
    if [ -n "$input" ]; then
        MODEL_NAME="$input"
    fi

    # 推理服务 API Key（可留空，使用 -s 静默输入）
    echo ""
    echo "注意：推理服务 API Key 将以静默方式输入"
    echo ""
    read -s -p "请输入推理服务 API Key [默认: 空，回车跳过]: " input
    echo ""
    API_KEY="$input"

    # 镜像名称 IMAGE（默认为openclaw:custom）
    read -p "请输入镜像名称 [默认: ${IMAGE}]: " input
    if [ -n "$input" ]; then
        IMAGE="$input"
    fi

    # 实例数量（默认 1）
    read -p "请输入实例数量 [默认: ${COUNT}]: " input
    if [ -n "$input" ]; then
        COUNT="$input"
    fi

    # 基础端口
    read -p "请输入基础端口 (Gateway 端口，后续实例依次 +4) [默认: ${BASE_PORT}]: " input
    if [ -n "$input" ]; then
        BASE_PORT="$input"
    fi

    echo ""
    echo "   确认配置："
    echo "   基础端口: ${BASE_PORT} (Gateway: ${BASE_PORT}, SFTP: +1, mDNS: +2)"
    echo "   推理服务 URL: ${INFER_URL}"
    echo "   模型名称: ${MODEL_NAME}"
    echo "   API Key: ${API_KEY:+$(mask_token "$API_KEY")}"
    echo "   镜像名称: ${IMAGE}"
    echo "   实例数量: ${COUNT}"
    echo ""

    read -p "确认开始部署？ [Y/n]: " confirm
    if [[ "$confirm" =~ ^[Nn] ]]; then
        echo "已取消部署。"
        exit 0
    fi

    SUBCOMMAND="up"
}

# 解析参数
parse_args() {
    # 无参数时进入交互式引导
    if [ $# -eq 0 ]; then
        interactive_prompt
        # 交互模式下不指定 Gateway Token，为每个实例独立生成
        OPENCLAW_TOKEN_PER_INSTANCE=true
        return
    fi

    # 提取子命令：第一个非 - 开头的参数
    case "${1:-}" in
        up|quick|config|start|stop|version|scale)
            SUBCOMMAND="$1"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            # 以 - 开头，不是子命令，默认走 up（向后兼容）
            SUBCOMMAND="up"
            ;;
        "")
            show_help
            exit 0
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac

    # version 子命令单独处理
    if [ "$SUBCOMMAND" = "version" ]; then
        VERSION_SUBCMD="${1:-}"
        shift 2>/dev/null || true
        VERSION_TARGET="${1:-}"
        shift 2>/dev/null || true
        return
    fi

    # 解析选项
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--count)   COUNT="$2"; shift 2 ;;
            -p|--port)    BASE_PORT="$2"; shift 2 ;;
            -m|--model)   MODEL_NAME="$2"; shift 2 ;;
            --model-provider) MODEL_PROVIDER="$2"; shift 2 ;;
            -u|--url)     INFER_URL="$2"; shift 2 ;;
            -c|--config-dir)
                if [[ "$2" = /* ]]; then
                    CONFIG_BASE="$2"
                else
                    CONFIG_BASE="$(pwd)/$2"
                fi
                shift 2
                ;;
            -i|--image)   IMAGE="$2"; shift 2 ;;
            --name)       INSTANCE_PREFIX="$2"; shift 2 ;;
            --token)      OPENCLAW_TOKEN="$2"; shift 2 ;;
            --skills)     MOUNT_SKILLS=true; shift ;;
            --hermes)     INSTALL_HERMES=true; shift ;;
            --guardian-port) GUARDIAN_PORT="$2"; shift 2 ;;
            --memex-kb-volume) MEMEX_KB_VOLUME="$2"; shift 2 ;;
            --no-down)    SKIP_DOWN=true; shift ;;
            --start-index) START_INDEX="$2"; shift 2 ;;
            --subnet)      SUBNET="$2"; shift 2 ;;
            --sandbox)    SANDBOX_ENABLED=true; shift ;;
            -h|--help)    show_help; exit 0 ;;
            *)            log_error "未知选项: $1"; exit 1 ;;
        esac
    done

    # Gateway Token 处理：命令行参数 > 环境变量 > 各实例独立生成
    if [ -z "$OPENCLAW_TOKEN" ]; then
        OPENCLAW_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"
    fi
    if [ -z "$OPENCLAW_TOKEN" ]; then
        OPENCLAW_TOKEN_PER_INSTANCE=true
        log_info "未指定 Gateway Token，将为每个实例独立生成随机 Token"
    fi
}
