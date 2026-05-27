#!/bin/bash
# install-sc-local.sh - SC插件本地安装脚本
#
# 用法：
#   ./install-sc-local.sh [选项] [SC源目录]
#
# 选项：
#   --openclaw-home <path>  OpenClaw主目录（默认 ~/.openclaw）
#   --copy                  使用复制代替符号链接（默认使用符号链接）
#   --build                 安装前执行构建（pnpm install + build）
#
# 环境变量：
#   OPENCLAW_HOME        OpenClaw主目录（默认 ~/.openclaw）

set -e

# ============ 配置 ============
OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"

# 如果未指定源目录，默认使用脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SC_SOURCE_DIR="${SCRIPT_DIR}"

# 安装选项
USE_COPY=false
DO_BUILD=false

# ============ 颜色输出 ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()      { echo -e "${GREEN}[SC]${NC} $1" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_step() { echo -e "${CYAN}[STEP]${NC} ${BOLD}$1${NC}" >&2; }
log_error(){ echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }

# ============ 参数解析 ============
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --openclaw-home)
                OPENCLAW_HOME="$2"
                shift 2
                ;;
            --copy)
                USE_COPY=true
                shift
                ;;
            --build)
                DO_BUILD=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            -*)
                log_error "未知选项: $1"
                ;;
            *)
                SC_SOURCE_DIR="$(cd "$1" && pwd)"
                shift
                ;;
        esac
    done
}

show_help() {
    sed -n '2,13p' "$0"
}

# ============ 环境检查 ============
check_environment() {
    log "检查环境..."

    if ! command -v openclaw &>/dev/null; then
        log_error "未找到 openclaw 命令，请确保 OpenClaw 已安装"
    fi

    if [[ ! -d "$OPENCLAW_HOME" ]]; then
        log_error "未找到 OpenClaw 目录: $OPENCLAW_HOME"
    fi

    log "OpenClaw 目录: $OPENCLAW_HOME"
}

# ============ 检查源目录 ============
check_source() {
    log_step "检查源目录: ${SC_SOURCE_DIR}"

    [[ -d "${SC_SOURCE_DIR}" ]] || log_error "源目录不存在: ${SC_SOURCE_DIR}"
    [[ -d "${SC_SOURCE_DIR}/plugins" ]] || log_error "plugins 目录不存在"
    [[ -f "${SC_SOURCE_DIR}/skill/SKILL.md" ]] || log_warn "skill/SKILL.md 不存在"

    log "源目录检查通过"
    echo ""
    echo "将安装以下插件:"
    for plugin in "${SC_SOURCE_DIR}/plugins"/*/; do
        if [[ -d "$plugin" ]]; then
            local name
            name=$(basename "$plugin")
            echo "  - $name"
        fi
    done
    echo ""
}

# ============ 构建 ============
build_project() {
    if [[ "$DO_BUILD" != true ]]; then
        return 0
    fi

    log_step "构建 SC 项目..."
    cd "${SC_SOURCE_DIR}"

    if [[ -f "pnpm-lock.yaml" ]] && command -v pnpm &>/dev/null; then
        log "使用 pnpm 安装依赖..."
        pnpm install
        if [[ -f "package.json" ]] && grep -q '"build"' package.json 2>/dev/null; then
            log "执行 pnpm build..."
            pnpm build
        fi
    elif [[ -f "package-lock.json" ]] && command -v npm &>/dev/null; then
        log "使用 npm 安装依赖..."
        npm install
        if [[ -f "package.json" ]] && grep -q '"build"' package.json 2>/dev/null; then
            log "执行 npm run build..."
            npm run build
        fi
    else
        log_warn "未找到 pnpm/npm 或 lock 文件，跳过构建"
    fi
}

# ============ 安装插件 ============
install_plugins() {
    local target_plugins_dir="$OPENCLAW_HOME/workspace/plugins"
    local target_skills_dir="$OPENCLAW_HOME/workspace/skills/subagent-coordinator"

    mkdir -p "$target_plugins_dir" "$target_skills_dir"

    for plugin in "${SC_SOURCE_DIR}/plugins"/*/; do
        if [[ -d "$plugin" ]]; then
            local name
            name=$(basename "$plugin")
            local target="$target_plugins_dir/$name"

            if [[ "$USE_COPY" == true ]]; then
                log "复制插件: $name"
                rm -rf "$target"
                cp -R "$plugin" "$target"
            else
                log "创建符号链接: $name"
                rm -rf "$target"
                ln -s "$plugin" "$target"
            fi
        fi
    done

    # 安装 skill
    if [[ "$USE_COPY" == true ]]; then
        log "复制技能文件..."
        rm -rf "$target_skills_dir"
        cp -R "${SC_SOURCE_DIR}/skill" "$target_skills_dir"
    else
        log "创建技能符号链接..."
        rm -rf "$target_skills_dir"
        ln -s "${SC_SOURCE_DIR}/skill" "$target_skills_dir"
    fi
}

# ============ 配置更新 ============
generate_python_config_script() {
    local plugins_dir=$1

    cat << 'PYTHON_SCRIPT'
import json
import os
import sys

plugins_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/.openclaw/workspace/plugins')
config_path = os.environ.get('OPENCLAW_CONFIG', os.path.expanduser('~/.openclaw/openclaw.json'))

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(f"配置文件读取失败: {e}")
    sys.exit(1)

# === 添加主代理子代理权限 ===
agents_list = config.get('agents', {}).get('list', [])
for agent in agents_list:
    if agent.get('id') == 'main':
        if 'subagents' not in agent:
            agent['subagents'] = {}
        if 'allowAgents' not in agent['subagents']:
            agent['subagents']['allowAgents'] = []
        if 'worker' not in agent['subagents']['allowAgents']:
            agent['subagents']['allowAgents'].append('worker')
            print("  - 已添加 main -> worker 权限")

# === 动态添加插件加载路径 ===
import glob
plugin_paths = sorted(glob.glob(os.path.join(plugins_dir, '*/')))

if 'plugins' not in config:
    config['plugins'] = {}
if 'load' not in config['plugins']:
    config['plugins']['load'] = {}
if 'paths' not in config['plugins']['load']:
    config['plugins']['load']['paths'] = []

for path in plugin_paths:
    path = path.rstrip('/')
    if path not in config['plugins']['load']['paths']:
        config['plugins']['load']['paths'].append(path)
        print(f"  - 已添加插件路径: {path}")

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("配置更新成功")
PYTHON_SCRIPT
}

configure_openclaw() {
    log_step "配置 OpenClaw..."

    local target_plugins_dir="$OPENCLAW_HOME/workspace/plugins"

    # 1. 创建 worker 子代理
    log "创建 worker 子代理..."
    openclaw agents add worker \
        --model ollama/gemma-4-e4b-heretic \
        --non-interactive \
        --workspace "$OPENCLAW_HOME/workspace/worker" 2>/dev/null \
        || log_warn "worker 可能已存在，跳过"

    # 2. 更新配置
    log "更新配置文件..."
    generate_python_config_script "$target_plugins_dir" | python3 - "$target_plugins_dir"

    # 3. 配置 exec allowlist
    log "配置 exec 权限..."
    openclaw approvals allowlist add --agent worker "**" 2>/dev/null || log_warn "allowlist 配置可能已存在"
}

# ============ 验证 ============
verify_installation() {
    log_step "验证安装..."

    echo ""
    echo "--- Agents ---"
    openclaw agents list 2>/dev/null | grep -E "^-" || echo "  (无输出)"

    echo ""
    echo "--- 插件目录 ---"
    ls -d "$OPENCLAW_HOME/workspace/plugins/"*/ 2>/dev/null | xargs -n1 basename || echo "  (无插件)"

    echo ""
    echo "--- 符号链接详情 ---"
    ls -la "$OPENCLAW_HOME/workspace/plugins/" 2>/dev/null | grep -E "^l" || echo "  (无符号链接)"
}

# ============ 显示结果 ============
show_result() {
    echo ""
    echo "=========================================="
    echo -e "${GREEN}${BOLD}SC 插件安装完成！${NC}"
    echo "=========================================="
    echo ""
    echo "OpenClaw 目录: $OPENCLAW_HOME"
    echo ""
    echo "后续步骤:"
    echo "  1. 重启 Gateway:"
    echo "     openclaw gateway restart"
    echo ""
    echo "  2. 验证安装:"
    echo "     openclaw agents list"
    echo ""
    echo "  3. 测试子代理:"
    echo "     openclaw agent --agent worker --message 'Hello'"
    echo ""
    echo "=========================================="
}

# ============ 主流程 ============
main() {
    parse_args "$@"

    echo ""
    echo "=========================================="
    echo -e "${BOLD}SC Plugin 本地安装脚本${NC}"
    echo "=========================================="
    echo ""

    # 1. 检查环境
    check_environment

    # 2. 检查源目录
    check_source

    # 3. 构建（可选）
    build_project

    # 4. 安装与配置
    install_plugins
    configure_openclaw
    verify_installation
    show_result
}

main "$@"
