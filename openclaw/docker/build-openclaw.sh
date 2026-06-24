#!/bin/bash
# =============================================================================
# OpenClaw 三层架构构建脚本
# =============================================================================
# Layer 1: openclaw-base → 基础设施
# Layer 2: openclaw-app   → 官方构建
# Layer 3: openclaw-overlay → 插件 + Hermes + 定制技能
#
# 离线构建: ./build-openclaw.sh --offline --version 2026.5.22
# 统一构建: ./build-openclaw.sh --version 2026.5.22
# =============================================================================

set -e

# ── 颜色 ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── 默认配置 ─────────────────────────────────────────────────
VERSION="2026.5.22"
REGISTRY="localhost"
OFFLINE=false
SKIP_BASE=false
SKIP_APP=false
SKIP_OVERLAY=false
SKIP_PLUGINS=false
CHECK_PLUGINS=false

# 源码路径（已存在则跳过克隆）
OPENCLAW_SRC="${OPENCLAW_SRC:-./.openclaw}"
HERMES_SRC="${HERMES_SRC:-./.hermes}"
PLUGINS_SRC="${PLUGINS_SRC:-./.plugins}"

# ── Git 仓库地址 ──────────────────────────────────────────────
OPENCLAW_GIT_URL="${OPENCLAW_GIT_URL:-https://github.com/openclaw/openclaw.git}"
HERMES_GIT_URL="${HERMES_GIT_URL:-https://github.com/NousResearch/hermes-agent.git}"

DOCKER_REGISTRY_NPM=""

# ── 参数解析 ─────────────────────────────────────────────────
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
        --check-plugins) CHECK_PLUGINS=true; shift ;;
        --openclaw-src) OPENCLAW_SRC="$2"; shift 2 ;;
        --hermes-src) HERMES_SRC="$2"; shift 2 ;;
        --plugins-src) PLUGINS_SRC="$2"; shift 2 ;;
        --npmmirror) DOCKER_REGISTRY_NPM="$2"; shift 2 ;;
        -h|--help)
            echo "OpenClaw 三层架构构建脚本（支持离线模式）"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --offline              仅使用本地源码，不尝试网络克隆"
            echo "  --version VERSION      指定版本（默认 2026.5.22）"
            echo "  --registry REGISTRY    镜像仓库前缀（默认 localhost）"
            echo "  --skip-base            跳过 Layer 1 构建"
            echo "  --skip-app             跳过 Layer 2 构建"
            echo "  --skip-overlay         跳过 Layer 3 构建"
            echo "  --skip-plugins         跳过插件准备"
            echo "  --check-plugins        检查插件集成状态（不执行构建）"
            echo "  --openclaw-src PATH    openclaw 源码目录（默认 ./.openclaw）"
            echo "  --hermes-src PATH      Hermes Agent 源码目录（默认 ./.hermes）"
            echo "  --plugins-src PATH     自定义插件目录（默认 ./.plugins）"
            echo "  --npmmirror URL        npm 镜像地址"
            echo ""
            echo "环境变量:"
            echo "  OPENCLAW_SRC           openclaw 源码目录（默认 ./.openclaw）"
            echo "  HERMES_SRC             Hermes Agent 源码目录（默认 ./.hermes）"
            echo "  PLUGINS_SRC            自定义插件目录（默认 ./.plugins）"
            echo "  OPENCLAW_GIT_URL       OpenClaw git 仓库地址"
            echo "  HERMES_GIT_URL         Hermes git 仓库地址"
            echo "  DOCKER_BUILD_OPTS      额外传给 docker build 的参数"
            echo ""
            echo "源码路径（docker/ 目录下的 . 前缀目录）:"
            echo "  - ./.openclaw           → OpenClaw 源码（app 层构建上下文）"
            echo "  - ./.hermes             → Hermes Agent 源码"
            echo "  - ./.plugins            → 自定义插件目录"
            echo "  - ../skills/            → 自定义技能目录（overlay 层使用）"
            echo ""
            echo "离线构建示例:"
            echo "  ./build-openclaw.sh --offline --version 2026.5.22"
            echo ""
            echo "插件检查示例:"
            echo "  ./build-openclaw.sh --check-plugins"
            exit 0
            ;;
        -*)
            log_error "未知选项: $1"
            exit 1
            ;;
    esac
done

# ── 镜像标签 ────────────────────────────────────────────────
BASE_IMAGE="${REGISTRY}/openclaw:base-${VERSION}"
APP_IMAGE="${REGISTRY}/openclaw:app-${VERSION}"
FINAL_IMAGE="${REGISTRY}/openclaw:${VERSION}"

log_info "构建配置:"
echo "  版本:        ${VERSION}"
echo "  镜像仓库:    ${REGISTRY}"
echo "  Base 镜像:   ${BASE_IMAGE}"
echo "  App 镜像:    ${APP_IMAGE}"
echo "  Final 镜像:  ${FINAL_IMAGE}"
echo "  离线模式:    ${OFFLINE}"
echo "  OpenClaw 源码: ${OPENCLAW_SRC}"
echo "  Hermes 源码:   ${HERMES_SRC}"
echo "  插件目录:      ${PLUGINS_SRC}"
echo ""

# ── 准备工作目录 ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 检查 Docker ─────────────────────────────────────────────
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

# ── 确保 OpenClaw 源码存在 ──────────────────────────────────
ensure_openclaw_src() {
    if [[ -d "${OPENCLAW_SRC}" ]]; then
        log_ok "OpenClaw 源码已存在: ${OPENCLAW_SRC}"
        return
    fi

    if [[ "$OFFLINE" == "true" ]]; then
        log_error "离线模式下无法克隆 OpenClaw 源码: ${OPENCLAW_SRC} 不存在"
        log_error "请手动将 OpenClaw 源码放置到 ${OPENCLAW_SRC}，或取消 --offline"
        exit 1
    fi

    log_info "克隆 OpenClaw 源码到 ${OPENCLAW_SRC} ..."
    git clone --depth 1 "${OPENCLAW_GIT_URL}" "${OPENCLAW_SRC}"
    log_ok "OpenClaw 源码克隆完成: ${OPENCLAW_SRC}"
}

# ── 确保 Hermes 源码存在 ────────────────────────────────────
ensure_hermes_src() {
    if [[ -d "${HERMES_SRC}" ]]; then
        log_ok "Hermes 源码已存在: ${HERMES_SRC}"
        return
    fi

    if [[ "$OFFLINE" == "true" ]]; then
        log_warn "离线模式下无法克隆 Hermes 源码: ${HERMES_SRC} 不存在，跳过"
        return
    fi

    log_info "克隆 Hermes 源码到 ${HERMES_SRC} ..."
    git clone --depth 1 "${HERMES_GIT_URL}" "${HERMES_SRC}"
    log_ok "Hermes 源码克隆完成: ${HERMES_SRC}"
}

# ── 确保自定义插件目录存在 ──────────────────────────────────
ensure_plugins_src() {
    if [[ -d "${PLUGINS_SRC}" ]]; then
        log_ok "插件目录已存在: ${PLUGINS_SRC}"
        return
    fi

    local src_plugins="${SCRIPT_DIR}/../plugins"
    if [[ -d "${src_plugins}" ]]; then
        log_info "从 ${src_plugins} 复制插件到 ${PLUGINS_SRC} ..."
        cp -r "${src_plugins}" "${PLUGINS_SRC}"
        log_ok "插件复制完成: ${PLUGINS_SRC}"
        return
    fi

    log_warn "插件目录不存在: ${PLUGINS_SRC}，且无本地源可复制（${src_plugins} 也不存在）"
}

# ── 安装 bun ────────────────────────────────────────────────
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

# ── 清理旧插件 ──────────────────────────────────────────────
cleanup_old_plugins() {
    if [[ -d "claude-mem-dist" ]]; then
        log_info "移除旧的 claude-mem-dist 目录（已不再使用 claude-mem 插件）"
        rm -rf claude-mem-dist
    fi
    if [[ -d "claude-code-best-dist" ]]; then
        log_info "移除旧的 claude-code-best-dist 目录（已不再使用 ccb 插件）"
        rm -rf claude-code-best-dist
    fi
}

# ── 构建 openclaw monorepo ───────────────────────────────────
prepare_openclaw_monorepo() {
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

# ── 构建 Layer 2: openclaw-app ───────────────────────────────
build_app_image() {
    if [[ "$SKIP_APP" == "true" ]]; then
        log_warn "跳过 Layer 2 构建（--skip-app）"
        return
    fi

    if ! docker image inspect "${BASE_IMAGE}" &> /dev/null; then
        log_error "Layer 1 镜像不存在: ${BASE_IMAGE}"
        exit 1
    fi

    if [[ ! -d "${OPENCLAW_SRC}" ]]; then
        log_error "OpenClaw 源码目录不存在: ${OPENCLAW_SRC}"
        log_error "请先运行 ensure_openclaw_src 或手动将源码放置到 ${OPENCLAW_SRC}"
        exit 1
    fi

    log_info "=== 构建 Layer 2: openclaw-app ==="
    log_info "构建上下文: ${OPENCLAW_SRC}"

    local npmmirror="${DOCKER_REGISTRY_NPM:-https://registry.npmmirror.com}"

    # 使用临时构建目录排除宿主机 node_modules（.dockerignore 在 NTFS 上不可靠）
    local app_build_dir="${SCRIPT_DIR}/.app-build-tmp"
    rm -rf "${app_build_dir}"
    mkdir -p "${app_build_dir}"
    rsync -a --exclude='node_modules' "${OPENCLAW_SRC}/" "${app_build_dir}/"
    log_info "源码已复制到临时构建目录（排除 node_modules）: ${app_build_dir}"

    # 预下载 matrix-sdk-crypto 原生二进制（ARM64 构建时 DOCKERFILE 会按 arch 重新下载）
    local arch_placeholder_name="matrix-sdk-crypto.linux-$(node -p 'process.arch')-gnu.node.placeholder"
    touch "${app_build_dir}/${arch_placeholder_name}"
    local crypto_pkg="${OPENCLAW_SRC}/node_modules/@matrix-org/matrix-sdk-crypto-nodejs"
    if [[ -d "${crypto_pkg}" ]] && [[ -f "${crypto_pkg}/package.json" ]]; then
        local crypto_version=$(node -p "require('${crypto_pkg}/package.json').version" 2>/dev/null || echo "")
        if [[ -n "${crypto_version}" ]]; then
            local arch_suffix
            arch_suffix="linux-$(node -p 'process.arch')-gnu"
            local crypto_file="${app_build_dir}/matrix-sdk-crypto.${arch_suffix}.node"
            if [[ ! -f "${crypto_file}" ]]; then
                local crypto_url="https://github.com/matrix-org/matrix-rust-sdk-crypto-nodejs/releases/download/v${crypto_version}/matrix-sdk-crypto.${arch_suffix}.node"
                log_info "预下载 matrix-sdk-crypto v${crypto_version} (${arch_suffix}) ..."
                curl -fsSL -o "${crypto_file}" "${crypto_url}" 2>/dev/null || {
                    log_warn "预下载失败（Matrix 聊天功能将不可用）"
                    rm -f "${crypto_file}"
                }
            fi
        fi
    else
        log_warn "宿主机无 matrix-sdk-crypto-nodejs 包，跳过预下载"
    fi

    docker build \
        -t "${APP_IMAGE}" \
        -f Dockerfile.openclaw-app \
        --build-arg OPENCLAW_BASE_IMAGE=${BASE_IMAGE} \
        --build-arg OPENCLAW_VERSION=${VERSION} \
        --build-arg NPM_REGISTRY=${npmmirror} \
        ${DOCKER_BUILD_OPTS} \
        "${app_build_dir}"

    rm -rf "${app_build_dir}"
    log_ok "Layer 2 构建完成: ${APP_IMAGE}"
}

# ── 构建 subagent-coordinator 插件预编译包 ────────────────────
prepare_subagent_coordinator() {
    local dist_dst="${SCRIPT_DIR}/subagent-coordinator-dist"

    if [[ -d "${dist_dst}" ]]; then
        log_info "subagent-coordinator-dist 已存在，跳过"
        return
    fi

    log_info "=== 准备 subagent-coordinator 插件 ==="

    # 优先从 .plugins 目录查找，回退到 ../plugins
    local plugin_src="${PLUGINS_SRC}/subagent-coordinator"
    if [[ ! -d "${plugin_src}" ]]; then
        plugin_src="${SCRIPT_DIR}/../plugins/subagent-coordinator"
    fi

    if [[ ! -d "${plugin_src}" ]]; then
        log_error "subagent-coordinator 插件源码不存在（尝试过 ${PLUGINS_SRC}/subagent-coordinator 和 ${SCRIPT_DIR}/../plugins/subagent-coordinator）"
        exit 1
    fi

    # 调用插件自带的 install_to_image.sh（会执行 pnpm build 并正确复制产物）
    local install_script="${plugin_src}/install_to_image.sh"
    if [[ ! -f "$install_script" ]]; then
        log_error "install_to_image.sh 不存在: $install_script"
        exit 1
    fi

    # install_to_image.sh 运行时会 cd 到插件目录，需转绝对路径
    local abs_plugin_sdk_src
    abs_plugin_sdk_src="$(cd "${SCRIPT_DIR}" && cd "${OPENCLAW_SRC}" 2>/dev/null && pwd)/packages/plugin-sdk" || abs_plugin_sdk_src="${OPENCLAW_SRC}/packages/plugin-sdk"
    PLUGIN_SDK_SRC="${abs_plugin_sdk_src}" \
    OPENCLAW_VERSION="${VERSION}" \
    OPENCLAW_SRC="${OPENCLAW_SRC}" \
    bash "$install_script" || {
        log_error "subagent-coordinator 插件准备失败"
        exit 1
    }

    # install_to_image.sh 输出到插件源码的 dist 子目录，复制到 docker context
    cp -r "${plugin_src}/dist/." "${dist_dst}/"

    log_ok "subagent-coordinator 准备完成"
}

# ── 构建 Layer 1: openclaw-base ───────────────────────────────
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

# ── 构建 Layer 3: openclaw-overlay ─────────────────────────────
build_overlay_image() {
    if [[ "$SKIP_OVERLAY" == "true" ]]; then
        log_warn "跳过 Layer 3 构建（--skip-overlay）"
        return
    fi

    if ! docker image inspect "${APP_IMAGE}" &> /dev/null; then
        log_error "Layer 2 镜像不存在: ${APP_IMAGE}"
        exit 1
    fi

    log_info "=== 构建 Layer 3: openclaw-overlay ==="

    # 清理旧插件目录
    cleanup_old_plugins

    # 使用临时构建目录避免 .dockerignore 导致 dist 目录被排除
    local overlay_build_dir="${SCRIPT_DIR}/.overlay-build-tmp"
    rm -rf "${overlay_build_dir}"
    mkdir -p "${overlay_build_dir}"

    # 显式复制所需文件到临时构建目录
    cp "${SCRIPT_DIR}/Dockerfile.openclaw-overlay" "${overlay_build_dir}/"
    cp -r "${SCRIPT_DIR}/subagent-coordinator-dist" "${overlay_build_dir}/" 2>/dev/null || true

    if [[ -d "${HERMES_SRC}" ]]; then
        cp -r "${HERMES_SRC}" "${overlay_build_dir}/hermes-src"
        log_info "Hermes 源码已复制到构建上下文"
    else
        mkdir -p "${overlay_build_dir}/hermes-src"
        log_warn "Hermes 源码不存在，overlay 层将跳过 Hermes 安装"
    fi

    # 复制 skills 到构建上下文（Dockerfile 会将 ./skills COPY 到 skills-shared）
    local mindclaw_skills_src="${SCRIPT_DIR}/../skills"
    local mindclaw_skills_count
    mindclaw_skills_count=$(ls -1 "${mindclaw_skills_src}/" 2>/dev/null | wc -l | tr -d ' ')
    mkdir -p "${overlay_build_dir}/skills"
    cp -r "${mindclaw_skills_src}/." "${overlay_build_dir}/skills/"
    log_info "mindclaw skills: ${mindclaw_skills_count} 项已复制到构建上下文（将打包到 skills-shared）"

    # 使用相对路径（Docker Desktop 无法解析 /mnt/d/ 绝对路径）
    local overlay_rel_dir=".overlay-build-tmp"
    docker build \
        -t "${FINAL_IMAGE}" \
        -f "${overlay_rel_dir}/Dockerfile.openclaw-overlay" \
        --build-arg OPENCLAW_APP_IMAGE=${APP_IMAGE} \
        ${DOCKER_BUILD_OPTS} \
        "${overlay_rel_dir}"

    rm -rf "${overlay_build_dir}"
    log_ok "Layer 3 构建完成: ${FINAL_IMAGE}"
}

# ── 显示结果 ────────────────────────────────────────────────
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

# ── 主流程 ───────────────────────────────────────────────────
main() {
    log_info "OpenClaw 构建脚本（离线模式: ${OFFLINE}）"
    echo ""

    check_docker

    # 按层按需准备源码
    if [[ "$SKIP_APP" != "true" ]] || [[ "$SKIP_OVERLAY" != "true" ]]; then
        ensure_openclaw_src
    fi

    if [[ "$SKIP_OVERLAY" != "true" ]]; then
        ensure_hermes_src
        ensure_plugins_src
    fi

    # Layer 3 预编译准备（需要 openclaw monorepo 产物来编译 subagent-coordinator 插件）
    if [[ "$SKIP_OVERLAY" != "true" ]]; then
        prepare_openclaw_monorepo
        prepare_subagent_coordinator
    fi

    # Layer 1
    build_base_image

    # Layer 2（Dockerfile 内部自行 pnpm install + build）
    if [[ "$SKIP_APP" != "true" ]]; then
        build_app_image
    fi

    # Layer 3
    if [[ "$SKIP_OVERLAY" != "true" ]]; then
        build_overlay_image
    fi

    show_summary
}

main "$@"
