#!/bin/bash
# =============================================================================
# OpenClaw 部署工具
# =============================================================================
# 用法: deploy.sh <命令> [选项]
#
# 命令:
#   up       完整部署（生成配置 + 启动集群）【默认】
#   quick    单实例快速部署
#   config   仅生成配置文件
#   start    启动已有集群
#   stop     停止集群
#   version  版本管理（list/current/switch/rollback）
#
# 详细帮助: deploy.sh -h
# =============================================================================

set -e

# 定位脚本目录（支持符号链接）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载模块
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/args.sh"
source "$SCRIPT_DIR/lib/config.sh"
source "$SCRIPT_DIR/lib/compose.sh"
source "$SCRIPT_DIR/lib/version.sh"

# 加载默认值
source "$SCRIPT_DIR/defaults.env"

# 解析参数
parse_args "$@"

# 确保 API_KEY 等关键变量导出（供 envsubst 使用）
export API_KEY
export MODEL_NAME MODEL_PROVIDER INFER_URL OPENCLAW_TOKEN

# 子命令分发
case "$SUBCOMMAND" in
    up)
        generate_all_configs
        generate_compose_file
        cluster_up
        ;;
    scale)
        detect_scale_start
        SKIP_DOWN=true
        generate_all_configs
        generate_compose_file
        cluster_up
        ;;
    quick)
        COUNT=1
        generate_all_configs
        generate_compose_file
        cluster_up
        ;;
    config)
        generate_all_configs
        generate_compose_file
        log_ok "配置生成完成（未启动容器）"
        log_info "配置目录: $CONFIG_BASE"
        log_info "启动集群: $0 start"
        ;;
    start)
        cluster_up
        ;;
    stop)
        cluster_down
        ;;
    version)
        case "$VERSION_SUBCMD" in
            list)     version_list ;;
            current)  version_current ;;
            switch)   version_switch "$VERSION_TARGET" ;;
            rollback) version_rollback ;;
            *)        show_version_help ;;
        esac
        ;;
    *)
        show_help
        exit 1
        ;;
esac
