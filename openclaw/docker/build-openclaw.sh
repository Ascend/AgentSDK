#!/bin/bash

# =============================================================================
# OpenClaw 三层架构构建脚本
# =============================================================================
# 功能：在离线环境中基于本地源码构建三层镜像
#   Layer 1 - openclaw-base     → SSH/uv/pip/npm全局包/playwright 等基础设施
#   Layer 2 - openclaw          → 官方 OpenClaw 构建（使用官方 Dockerfile）
#   Layer 3 - openclaw-overlay  → claude-mem + ccb + subagent-coordinator 定制层
#
# 离线构建（使用本地源码）:
#   OPENCLAW_SRC=/path/to/openclaw-src \
#   CLAUDE_MEM_SRC=/path/to/claude-mem-src \
#   bash ./build-openclaw.sh --offline --skip-base --skip-app \
#     --claude-code-src /path/to/claude-code-best
#
# 参数说明:
#   --offline              仅使用本地源码，不尝试网络克隆
#   --version VERSION      指定版本（默认 2026.4.11）
#   --registry REGISTRY    镜像仓库前缀（默认 localhost）
#   --skip-base            跳过 Layer 1 构建
#   --skip-app             跳过 Layer 2 构建
#   --skip-overlay         跳过 Layer 3 构建
#   --skip-plugins         跳过插件准备
#   --include-claude-code-best  启用 claude-code-best 打包（测试功能）
#   --openclaw-src PATH    openclaw 源码目录
#   --claude-mem-src PATH  claude-mem 源码目录
#   --claude-code-src PATH claude-code-best 源码目录
#   --npmmirror URL        npm 镜像地址
#
# 环境变量:
#   OPENCLAW_SRC           openclaw 源码目录（默认 /tmp/openclaw-src）
#   CLAUDE_MEM_SRC         claude-mem 源码目录（默认 /tmp/claude-mem-src）
#   CLAUDE_CODE_SRC        claude-code-best 源码目录（可选，离线自动创建空 placeholder）
#   DOCKER_BUILD_OPTS       额外传给 docker build 的参数
#
# 镜像 CLI 命令:
#   ccb                 claude-code-best CLI
#   openclaw                 OpenClaw 主 CLI
#   claude-mem              claude-mem CLI
# =============================================================================

set -e

# ── 颜色定义 ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── 默认配置 ────────────────────────────────────────────────────────────────
VERSION="2026.4.11"
REGISTRY="localhost"
OFFLINE=false
SKIP_BASE=false
SKIP_APP=false
SKIP_OVERLAY=false
SKIP_PLUGINS=false
INCLUDE_CLAUDE_CODE_BEST=false
OPENCLAW_SRC="${OPENCLAW_SRC:-/tmp/openclaw-src}"
CLAUDE_MEM_SRC="${CLAUDE_MEM_SRC:-/tmp/claude-mem-src}"
CLAUDE_CODE_SRC=""
DOCKER_REGISTRY_NPM=""

# ── 参数解析 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --offline) OFFLINE=true; shift ;;
        --no-cache) DOCKER_BUILD_OPTS="${DOCKER_BUILD_OPTS} --no-cache"; shift ;;
        --version) VERSION="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        --skip-base) SKIP_BASE=true; shift ;;
        --skip-app) SKIP_APP=true; shift ;;
        --skip-overlay) SKIP_OVERLAY=true; shift ;;
        --skip-plugins) SKIP_PLUGINS=true; shift ;;
        --include-claude-code-best) INCLUDE_CLAUDE_CODE_BEST=true; shift ;;
        --openclaw-src) OPENCLAW_SRC="$2"; shift 2 ;;
        --claude-mem-src) CLAUDE_MEM_SRC="$2"; shift 2 ;;
        --claude-code-src) CLAUDE_CODE_SRC="$2"; shift 2 ;;
        --npmmirror) DOCKER_REGISTRY_NPM="$2"; shift 2 ;;
        -h|--help)
            echo "OpenClaw 三层架构构建脚本（支持离线模式）"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --offline              仅使用本地源码，不尝试网络克隆"
            echo "  --version VERSION      指定版本（默认 2026.4.11）"
            echo "  --registry REGISTRY    镜像仓库前缀（默认 localhost）"
            echo "  --skip-base            跳过 Layer 1 构建"
            echo "  --skip-app             跳过 Layer 2 构建"
            echo "  --skip-overlay         跳过 Layer 3 构建"
            echo "  --skip-plugins         跳过插件准备"
            echo "  --include-claude-code-best  启用 claude-code-best 打包（测试功能）"
            echo "  --openclaw-src PATH    openclaw 源码目录"
            echo "  --claude-mem-src PATH  claude-mem 源码目录"
            echo "  --claude-code-src PATH claude-code-best 源码目录"
            echo "  --npmmirror URL        npm 镜像地址"
            echo ""
            echo "环境变量:"
            echo "  OPENCLAW_SRC           openclaw 源码目录（默认 /tmp/openclaw-src）"
            echo "  CLAUDE_MEM_SRC         claude-mem 源码目录（默认 /tmp/claude-mem-src）"
            echo "  CLAUDE_CODE_SRC        claude-code-best 源码目录"
            echo "  DOCKER_BUILD_OPTS       额外传给 docker build 的参数"
            echo ""
            echo "离线构建示例:"
            echo "  OPENCLAW_SRC=/path/to/openclaw-src \\"
            echo "  CLAUDE_MEM_SRC=/path/to/claude-mem-src \\"
            echo "  ./build-openclaw.sh --offline --version 2026.4.11"
            exit 0
            ;;
        -*)
            log_error "未知选项: $1"
            exit 1
            ;;
    esac
done

# ── 镜像标签 ────────────────────────────────────────────────────────────────
BASE_IMAGE="${REGISTRY}/openclaw-base:${VERSION}"
APP_OFFICIAL_TAG="${REGISTRY}/openclaw:${VERSION}"
APP_IMAGE="${REGISTRY}/openclaw:${VERSION}"
FINAL_IMAGE="${REGISTRY}/openclaw:${VERSION}-sftp-docx-browser-ccb"

log_info "构建配置:"
echo "  版本:        ${VERSION}"
echo "  镜像仓库:    ${REGISTRY}"
echo "  Base 镜像:   ${BASE_IMAGE}"
echo "  App 镜像:    ${APP_OFFICIAL_TAG} (${APP_IMAGE})"
echo "  Final 镜像: ${FINAL_IMAGE}"
echo "  离线模式:    ${OFFLINE}"
echo "  打包 claude-code-best: ${INCLUDE_CLAUDE_CODE_BEST}"
echo ""

# ── 准备工作目录 ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 检查 Docker ─────────────────────────────────────────────────────────────
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        exit 1
    fi
    if ! docker info &> /dev/null; then
        log_error "Docker 未运行或无权限"
        exit 1
    fi
    log_ok "Docker 检查通过"
}

# ── 安装 bun（如果需要）──────────────────────────────────────────────────────
ensure_bun() {
    if [[ ":$PATH:" != *":$HOME/.bun/bin:"* ]]; then
        export PATH="$HOME/.bun/bin:$PATH"
    fi
    if command -v bun &> /dev/null; then
        return
    fi
    log_info "安装 bun..."
    curl -fsSL https://bun.sh/install | bash -s -- >&2
}

# ── 构建 claude-code-best 预编译包 ─────────────────────────────────────────
prepare_claude_code() {
    # 如果不需要打包 claude-code-best，直接创建空 placeholder
    if [[ "$INCLUDE_CLAUDE_CODE_BEST" != "true" ]]; then
        if [[ -d "claude-code-best-dist" ]]; then
            log_info "claude-code-best-dist 已存在但 INCLUDE_CLAUDE_CODE_BEST=false，移除"
            rm -rf claude-code-best-dist
        fi
        mkdir -p "${SCRIPT_DIR}/claude-code-best-dist"
        touch "${SCRIPT_DIR}/claude-code-best-dist/.placeholder"
        log_info "claude-code-best 已禁用，创建空 placeholder"
        return
    fi

    if [[ -d "claude-code-best-dist" ]]; then
        log_info "claude-code-best-dist 已存在，跳过"
        return
    fi

    log_info "准备 claude-code-best..."

    # 解析源码目录（离线优先）
    local src_dir=""
    if [[ -n "$CLAUDE_CODE_SRC" && -d "$CLAUDE_CODE_SRC" ]]; then
        src_dir="$CLAUDE_CODE_SRC"
        log_info "使用本地 claude-code-best 源码: ${src_dir}"
    elif [[ "$OFFLINE" == "true" ]]; then
        log_warn "离线模式：CLAUDE_CODE_SRC 未指定或目录不存在，创建空 placeholder"
        mkdir -p "${SCRIPT_DIR}/claude-code-best-dist"
        touch "${SCRIPT_DIR}/claude-code-best-dist/.placeholder"
        log_info "claude-code-best-dist placeholder 已创建"
        return
    fi

    ensure_bun

    if [[ -z "$src_dir" ]]; then
        log_info "克隆 claude-code-best 源码..."
        src_dir="${SCRIPT_DIR}/.claude_code-src-tmp"
        if [[ ! -d "${src_dir}" ]]; then
            git clone --depth 1 https://github.com/claude-code-best/claude-code.git "${src_dir}" 2>&1 | tail -5
        fi
    fi

    [[ ! -d "${src_dir}" ]] && { log_error "claude-code-best 源码不存在"; exit 1; }

    cd "${src_dir}"
    bun install --legacy-peer-deps >&2 | tail -5
    bun run build >&2 | tail -5

    mkdir -p "${SCRIPT_DIR}/claude-code-best-dist"
    cp -r "${src_dir}/dist"/* "${SCRIPT_DIR}/claude-code-best-dist/" 2>/dev/null || true
    cp -r "${src_dir}/.claude"/* "${SCRIPT_DIR}/claude-code-best-dist/" 2>/dev/null || true

    # ws 包在打包时被遗漏，需额外复制到产物目录
    if [[ -d "${src_dir}/node_modules/ws" ]]; then
        mkdir -p "${SCRIPT_DIR}/claude-code-best-dist/node_modules"
        cp -r "${src_dir}/node_modules/ws" "${SCRIPT_DIR}/claude-code-best-dist/node_modules/"
        log_info "ws 依赖已补充到 claude-code-best-dist"
    fi

    cd "${SCRIPT_DIR}"
    log_ok "claude-code-best 编译完成"
}

# ── 构建 claude-mem 预编译包 ────────────────────────────────────────────────
prepare_claude_mem() {
    if [[ -d "claude-mem-dist/plugin/npx-cli" && -d "claude-mem-dist/openclaw" ]]; then
        log_info "claude-mem-dist 已存在，跳过"
        return
    fi

    log_info "准备 claude-mem..."

    # ── 解析源码目录（优先级: CLAUDE_MEM_SRC > docker/ 目录 > git clone）─────────
    local src_dir=""
    # 优先级 1: CLAUDE_MEM_SRC 环境变量
    if [[ -n "$CLAUDE_MEM_SRC" && -d "$CLAUDE_MEM_SRC" ]]; then
        src_dir="$CLAUDE_MEM_SRC"
        log_info "使用 CLAUDE_MEM_SRC 源码: ${src_dir}"
    # 优先级 2: docker/ 目录下的 claude-mem 源码
    elif [[ -d "${SCRIPT_DIR}/../claude-mem" ]]; then
        src_dir="${SCRIPT_DIR}/../claude-mem"
        log_info "使用 docker 目录下的 claude-mem 源码: ${src_dir}"
    # 优先级 3: git clone（仅非离线模式）
    elif [[ "$OFFLINE" != "true" ]]; then
        src_dir="${SCRIPT_DIR}/.claude_mem-src-tmp"
        if [[ ! -d "${src_dir}" ]]; then
            git clone --depth 1 https://github.com/thedotmack/claude-mem.git "${src_dir}" 2>&1 | tail -5
            log_info "已克隆 claude-mem 源码: ${src_dir}"
        else
            log_info "使用缓存的 claude-mem 源码: ${src_dir}"
        fi
    fi

    # 离线模式且无源码 → 错误
    if [[ -z "$src_dir" ]]; then
        log_error "离线模式：未找到 claude-mem 源码（CLAUDE_MEM_SRC 或 docker/../claude-mem）"
        exit 1
    fi

    [[ ! -d "${src_dir}" ]] && { log_error "claude-mem 源码不存在: ${src_dir}"; exit 1; }

    ensure_bun

    cd "${src_dir}"
    bun install --legacy-peer-deps >&2 | tail -5
    bun run build >&2 | tail -5

    # ── 复制编译产物 ───────────────────────────────────────────────────────────
    mkdir -p "${SCRIPT_DIR}/claude-mem-dist/plugin" "${SCRIPT_DIR}/claude-mem-dist/openclaw"
    [[ -d "${src_dir}/dist/plugin" ]] && cp -r "${src_dir}/dist/plugin"/* "${SCRIPT_DIR}/claude-mem-dist/plugin/" || cp -r "${src_dir}/dist"/* "${SCRIPT_DIR}/claude-mem-dist/plugin/"
    [[ -d "${src_dir}/openclaw" ]] && cp -r "${src_dir}/openclaw"/* "${SCRIPT_DIR}/claude-mem-dist/openclaw/"

    # ── 复制 skills 目录 ───────────────────────────────────────────────────────
    mkdir -p "${SCRIPT_DIR}/claude-mem-dist/plugin/skills"
    if [[ -d "${src_dir}/dist/plugin/skills" ]]; then
        cp -r "${src_dir}/dist/plugin/skills"/* "${SCRIPT_DIR}/claude-mem-dist/plugin/skills/"
    elif [[ -d "${src_dir}/plugin/skills" ]]; then
        cp -r "${src_dir}/plugin/skills"/* "${SCRIPT_DIR}/claude-mem-dist/plugin/skills/"
        log_info "plugin/skills 已复制"
    fi

    # ── 复制 scripts 目录（worker-service.cjs 等运行时脚本）────────────────────
    if [[ -d "${src_dir}/plugin/scripts" ]]; then
        mkdir -p "${SCRIPT_DIR}/claude-mem-dist/plugin/scripts"
        cp -r "${src_dir}/plugin/scripts"/* "${SCRIPT_DIR}/claude-mem-dist/plugin/scripts/"
        log_info "plugin/scripts 已复制"
    fi

    # ── 补充 package.json（npx-cli bundle 运行时依赖 package.json 定位模块）─────
    if [[ -f "${src_dir}/package.json" ]]; then
        cp "${src_dir}/package.json" "${SCRIPT_DIR}/claude-mem-dist/plugin/package.json"
        log_info "package.json 已补充"
    fi

    # ── 复制 modes 目录（如果源码中存在）────────────────────────────────────────
    # 优先级 1: 源码中有 modes 目录 → 直接复制
    # 优先级 2: 源码中无 modes 目录 → 自动生成
    mkdir -p "${SCRIPT_DIR}/claude-mem-dist/plugin/modes"
    if [[ -d "${src_dir}/dist/plugin/modes" ]]; then
        cp -r "${src_dir}/dist/plugin/modes"/* "${SCRIPT_DIR}/claude-mem-dist/plugin/modes/"
        log_info "modes 目录已从源码复制"
    elif [[ -d "${src_dir}/plugin/modes" ]]; then
        cp -r "${src_dir}/plugin/modes"/* "${SCRIPT_DIR}/claude-mem-dist/plugin/modes/"
        log_info "modes 目录已从源码复制"
    else
        # 源码中无 modes 目录 → 自动生成（修复镜像缺失问题）
        cat > "${SCRIPT_DIR}/claude-mem-dist/plugin/modes/code.json" << 'MODE_JSON'
{
  "name": "code",
  "description": "Default code mode",
  "observation_types": [
    {"id": "bugfix", "label": "Bug Fix", "emoji": "🐛"},
    {"id": "feature", "label": "Feature", "emoji": "✨"},
    {"id": "refactor", "label": "Refactor", "emoji": "♻️"},
    {"id": "discovery", "label": "Discovery", "emoji": "💡"},
    {"id": "decision", "label": "Decision", "emoji": "Decision"},
    {"id": "change", "label": "Change", "emoji": "📝"}
  ],
  "observation_concepts": [
    {"id": "how-it-works", "label": "How it works"},
    {"id": "why-it-exists", "label": "Why it exists"},
    {"id": "what-changed", "label": "What changed"},
    {"id": "problem-solution", "label": "Problem/Solution"},
    {"id": "gotcha", "label": "Gotcha"},
    {"id": "pattern", "label": "Pattern"},
    {"id": "trade-off", "label": "Trade-off"}
  ]
}
MODE_JSON
        log_info "modes/code.json 已自动生成"
    fi

    cd "${SCRIPT_DIR}"
    log_ok "claude-mem 编译完成"
}

# ── 构建 openclaw monorepo（生成 plugin-sdk 等编译产物）────────────────────
prepare_openclaw_monorepo() {
    if [[ "$SKIP_APP" == "true" ]]; then
        log_warn "跳过 openclaw monorepo 准备（--skip-app）"
        return
    fi

    local src_dir="${OPENCLAW_SRC}"

    if [[ ! -d "${src_dir}" ]]; then
        log_error "openclaw 源码目录不存在: ${src_dir}"
        exit 1
    fi

    if [[ -f "${src_dir}/packages/plugin-sdk/dist/src/plugin-sdk/index.d.ts" ]]; then
        log_info "openclaw monorepo 已构建，跳过"
        return
    fi

    log_info "=== 构建 openclaw monorepo ==="
    cd "${src_dir}"
    pnpm install --frozen-lockfile 2>&1 | tail -10
    pnpm build 2>&1 | tail -15
    cd "${SCRIPT_DIR}"
    log_ok "openclaw monorepo 构建完成"
}

# ── 构建 openclaw 官方 Layer 2 镜像 ──────────────────────────────────────────
build_app_image() {
    if [[ "$SKIP_APP" == "true" ]]; then
        log_warn "跳过 Layer 2 构建（--skip-app）"
        return
    fi

    if ! docker image inspect "${BASE_IMAGE}" &> /dev/null; then
        log_error "Layer 1 镜像不存在: ${BASE_IMAGE}"
        exit 1
    fi

    log_info "=== 构建 Layer 2: openclaw-app ==="

    # 准备 openclaw 源码到 docker build 上下文
    local openclaw_context="${SCRIPT_DIR}/.openclaw-src-tmp"
    if [[ -d "${OPENCLAW_SRC}" ]] && [[ ! -d "${openclaw_context}" ]]; then
        cp -r "${OPENCLAW_SRC}" "${openclaw_context}"
        log_info "openclaw 源码已复制到构建上下文"
    fi

    local npmmirror="${DOCKER_REGISTRY_NPM:-https://registry.npmmirror.com}"

    docker build \
        -t "${APP_OFFICIAL_TAG}" \
        -t "${APP_IMAGE}" \
        -f Dockerfile.openclaw-app \
        --build-arg OPENCLAW_BASE_IMAGE=${BASE_IMAGE} \
        --build-arg OPENCLAW_VERSION=${VERSION} \
        --build-arg OPENCLAW_INSTALL_BROWSER=1 \
        --build-arg NPM_REGISTRY=${npmmirror} \
        ${DOCKER_BUILD_OPTS} \
        "${openclaw_context}"

    log_ok "Layer 2 构建完成: ${APP_OFFICIAL_TAG}"
}

# ── 构建 subagent-coordinator 插件预编译包 ──────────────────────────────────
prepare_subagent_coordinator() {
    local dist_dst="${SCRIPT_DIR}/subagent-coordinator-dist"

    if [[ -d "${dist_dst}" ]]; then
        log_info "subagent-coordinator-dist 已存在，跳过"
        return
    fi

    log_info "=== 准备 subagent-coordinator 插件 ==="

    # 调用插件自带的 install_to_image.sh（会执行 pnpm build 并正确复制产物）
    local install_script="${SCRIPT_DIR}/../plugins/subagent-coordinator/install_to_image.sh"
    if [[ ! -f "$install_script" ]]; then
        log_error "install_to_image.sh 不存在: $install_script"
        exit 1
    fi

    OPENCLAW_VERSION="${VERSION}" \
    OPENCLAW_SRC="${OPENCLAW_SRC}" \
    bash "$install_script" || {
        log_error "subagent-coordinator 插件准备失败"
        exit 1
    }

    # install_to_image.sh 输出到 plugins/subagent-coordinator/dist，复制到 docker context
    cp -r "${SCRIPT_DIR}/../plugins/subagent-coordinator/dist/." "${dist_dst}/"

    log_ok "subagent-coordinator 准备完成"
}

# ── 构建 Layer 1: openclaw-base ──────────────────────────────────────────────
build_base_image() {
    if [[ "$SKIP_BASE" == "true" ]]; then
        log_warn "跳过 Layer 1 构建（--skip-base）"
        return
    fi

    log_info "=== 构建 Layer 1: openclaw-base ==="

    docker build \
        -t "${BASE_IMAGE}" \
        -f Dockerfile.openclaw-base \
        ${DOCKER_BUILD_OPTS} \
        .

    log_ok "Layer 1 构建完成: ${BASE_IMAGE}"
}

# ── 构建 Layer 3: openclaw-overlay ──────────────────────────────────────────
build_overlay_image() {
    if [[ "$SKIP_OVERLAY" == "true" ]]; then
        log_warn "跳过 Layer 3 构建（--skip-overlay）"
        return
    fi

    if ! docker image inspect "${APP_OFFICIAL_TAG}" &> /dev/null; then
        log_error "Layer 2 镜像不存在: ${APP_OFFICIAL_TAG}"
        exit 1
    fi

    log_info "=== 构建 Layer 3: openclaw-overlay ==="

    # 准备 openclaw 源码到 docker build 上下文（Layer 3 容器内需要访问）
    local openclaw_context="${SCRIPT_DIR}/.openclaw-src-tmp"
    if [[ -d "${OPENCLAW_SRC}" ]] && [[ ! -d "${openclaw_context}" ]]; then
        cp -r "${OPENCLAW_SRC}" "${openclaw_context}"
    fi

    # 检查 playwright 缓存
    local browser_src=""
    for cache_dir in "/root/.cache/ms-playwright" "$HOME/.cache/ms-playwright"; do
        if [[ -d "${cache_dir}" ]]; then
            browser_src="${cache_dir}"
            log_info "使用 playwright 缓存: ${browser_src}"
            break
        fi
    done

    # 使用临时构建目录避免 .dockerignore 导致 dist 目录被排除
    local overlay_build_dir="${SCRIPT_DIR}/.overlay-build-tmp"
    rm -rf "${overlay_build_dir}"
    mkdir -p "${overlay_build_dir}"

    # 显式复制所有需要的文件到临时构建目录
    cp "${SCRIPT_DIR}/Dockerfile.openclaw-overlay" "${overlay_build_dir}/"
    cp -r "${SCRIPT_DIR}/claude-mem-dist" "${overlay_build_dir}/"
    cp -r "${SCRIPT_DIR}/claude-code-best-dist" "${overlay_build_dir}/"
    cp -r "${SCRIPT_DIR}/subagent-coordinator-dist" "${overlay_build_dir}/"

    # 通过显式绝对路径复制项目 skills（避免构建上下文解析 ./skills 的歧义）
    local mindclaw_skills_src="${SCRIPT_DIR}/../skills"
    local mindclaw_skills_count
    mindclaw_skills_count=$(ls -1 "${mindclaw_skills_src}/" 2>/dev/null | wc -l | tr -d ' ')
    mkdir -p "${overlay_build_dir}/skills"
    cp -r "${mindclaw_skills_src}/." "${overlay_build_dir}/skills/"
    log_info "mindclaw skills: ${mindclaw_skills_count} 项已复制到构建上下文"

    docker build \
        -t "${FINAL_IMAGE}" \
        -f "${overlay_build_dir}/Dockerfile.openclaw-overlay" \
        --build-arg OPENCLAW_BASE_IMAGE=${APP_OFFICIAL_TAG} \
        --build-arg INCLUDE_CLAUDE_CODE_BEST=${INCLUDE_CLAUDE_CODE_BEST} \
        --build-arg BROWSER_SRC=${browser_src} \
        ${DOCKER_BUILD_OPTS} \
        "${overlay_build_dir}"

    rm -rf "${overlay_build_dir}"
    log_ok "Layer 3 构建完成: ${FINAL_IMAGE}"
}

# ── 显示结果 ────────────────────────────────────────────────────────────────
show_summary() {
    echo ""
    echo "=============================================="
    log_ok "全部构建完成!"
    echo "=============================================="
    echo ""
    echo "镜像标签:"
    echo "  Base:    ${BASE_IMAGE}"
    echo "  App:     ${APP_IMAGE}"
    echo "  Final:   ${FINAL_IMAGE}"
    echo ""
    echo "启动示例:"
    echo "  docker run -d --name openclaw-test \\"
    echo "    -p 18080:18080 -p 18081:18081 \\"
    echo "    ${FINAL_IMAGE}"
}

# ── 主流程 ───────────────────────────────────────────────────────────────────
main() {
    log_info "OpenClaw 构建脚本（离线模式: ${OFFLINE}）"
    echo ""

    check_docker

    # Layer 1 + Layer 3 预编译准备
    if [[ "$SKIP_BASE" != "true" ]] || [[ "$SKIP_OVERLAY" != "true" ]]; then
        prepare_claude_code
        prepare_claude_mem
        if [[ "$SKIP_OVERLAY" != "true" ]]; then
            prepare_subagent_coordinator
        fi
    fi

    # Layer 1
    build_base_image

    # Layer 2（需要 openclaw monorepo 先构建）
    if [[ "$SKIP_APP" != "true" ]]; then
        prepare_openclaw_monorepo
        build_app_image
    fi

    # Layer 3
    if [[ "$SKIP_OVERLAY" != "true" ]]; then
        build_overlay_image
    fi

    show_summary
}

main "$@"
