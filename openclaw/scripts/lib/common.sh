#!/bin/bash
# =============================================================================
# 公共函数：日志、密码生成、docker-compose 检测
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 掩码敏感信息：显示前后各3个字符，中间用 * 代替
mask_token() {
    local token="$1"
    if [ -z "$token" ] || [ ${#token} -le 6 ]; then
        echo "******"
    else
        echo "${token:0:3}***${token: -3}"
    fi
}

# 生成随机密码（16位字母数字）
generate_random_password() {
    openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 16
}

# 获取 Docker 网关地址
get_docker_gateway() {
    local docker_version
    docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "0.0.0")
    local major minor
    major=$(echo "$docker_version" | cut -d. -f1)
    minor=$(echo "$docker_version" | cut -d. -f2)
    if [ "$major" -gt 20 ] || { [ "$major" -eq 20 ] && [ "$minor" -ge 1 ]; }; then
        echo "host-gateway"
    else
        docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null || echo "172.17.0.1"
    fi
}

# 检测 docker compose 命令（优先使用新版 docker compose 插件）
detect_docker_compose() {
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
    elif [ -x "/opt/homebrew/bin/docker-compose" ]; then
        DOCKER_COMPOSE_CMD="/opt/homebrew/bin/docker-compose"
    elif command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    else
        log_error "未找到 docker-compose 或 docker"
        exit 1
    fi
}

# 检查 envsubst 是否可用
ensure_envsubst() {
    if ! command -v envsubst &> /dev/null; then
        log_error "envsubst 未找到，请安装 gettext 包"
        log_info "  Ubuntu/Debian: apt-get install gettext-base"
        log_info "  CentOS/RHEL:   yum install gettext"
        log_info "  macOS:         brew install gettext"
        exit 1
    fi
}
