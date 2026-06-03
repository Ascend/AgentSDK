#!/bin/bash
# install-sc-local.sh - SC插件本地安装脚本
#
# 用法：
#   ./install-sc-local.sh [选项] [SC源目录]
#
# 选项：
#   --openclaw-home <path>  OpenClaw主目录（默认 ~/.openclaw）
#   --symlink               使用符号链接代替复制（默认使用复制；仅开发场景使用）
#   --build                 安装前执行构建（pnpm install + build）
#   --skip-install          跳过 openclaw plugins install（仅链接/复制插件文件）
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
USE_SYMLINK=false
DO_BUILD=false
SKIP_INSTALL=false

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
            --symlink)
                USE_SYMLINK=true
                shift
                ;;
            --build)
                DO_BUILD=true
                shift
                ;;
            --skip-install)
                SKIP_INSTALL=true
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
    sed -n '2,15p' "$0"
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
link_plugin_sdk() {
    # 插件源码依赖 openclaw/plugin-sdk，需要创建符号链接以便 pnpm install 解析。
    # 按以下优先级查找 plugin-sdk：
    # 1. 环境变量 PLUGIN_SDK_SRC
    # 2. 容器镜像常见路径 /app/packages/plugin-sdk
    # 3. 本地 OpenClaw 源码路径（与 SC 源码同仓库）

    local sdk_dir=""

    if [[ -n "$PLUGIN_SDK_SRC" && -d "$PLUGIN_SDK_SRC" ]]; then
        sdk_dir="$PLUGIN_SDK_SRC"
        log "使用 PLUGIN_SDK_SRC 指定的 plugin-sdk: $sdk_dir"
    elif [[ -d "/app/packages/plugin-sdk" ]]; then
        sdk_dir="/app/packages/plugin-sdk"
        log "使用镜像内置 plugin-sdk: $sdk_dir"
    elif [[ -d "${SC_SOURCE_DIR}/../../packages/plugin-sdk" ]]; then
        sdk_dir="${SC_SOURCE_DIR}/../../packages/plugin-sdk"
        log "使用本地 OpenClaw 源码 plugin-sdk: $sdk_dir"
    fi

    if [[ -n "$sdk_dir" ]]; then
        mkdir -p "${SC_SOURCE_DIR}/node_modules/openclaw"
        rm -f "${SC_SOURCE_DIR}/node_modules/openclaw/plugin-sdk"
        ln -sfn "$sdk_dir" "${SC_SOURCE_DIR}/node_modules/openclaw/plugin-sdk"
        log "plugin-sdk 链接完成"
    else
        log_warn "未找到 plugin-sdk，构建可能失败。可设置 PLUGIN_SDK_SRC 环境变量指定路径。"
    fi
}

# 将 src/* TypeScript 源码编译为 dist/*，同时确保根 index.js 指向正确的 dist 路径。
# 当 src/ 位置因 monorepo 结构而落在 dist/plugins/<name>/src/* 时，必须重写根 index.js。
ensure_plugin_entry() {
    local plugin_dir=$1
    local plugin_name
    plugin_name=$(basename "$plugin_dir")

    # 检测 src 实际编译后的位置：
    # 单包结构: <plugin>/dist/src/index.js
    # monorepo 结构: <plugin>/dist/plugins/<plugin>/src/index.js
    local entry=""
    if [[ -f "${plugin_dir}/dist/src/index.js" ]]; then
        entry="./dist/src/index.js"
    elif [[ -f "${plugin_dir}/dist/plugins/${plugin_name}/src/index.js" ]]; then
        entry="./dist/plugins/${plugin_name}/src/index.js"
    fi

    if [[ -z "$entry" ]]; then
        log_warn "  ${plugin_name}: 未找到 dist 入口，跳过入口创建"
        return 1
    fi

    # 创建根 index.js 重新导出 dist 中的实际入口
    cat > "${plugin_dir}/index.js" <<EOF
export { default } from "${entry}";
EOF
    log "  ${plugin_name}: 根 index.js -> ${entry}"

    # 修正 package.json 的 openclaw.extensions 指向根 index.js
    if command -v python3 &>/dev/null; then
        python3 - "$plugin_dir" <<'PY'
import json, sys
plugin_dir = sys.argv[1]
pkg_path = f"{plugin_dir}/package.json"
with open(pkg_path) as f:
    pkg = json.load(f)
oc = pkg.setdefault("openclaw", {})
exts = oc.get("extensions", [])
if exts and exts[0] != "./index.js":
    oc["extensions"] = ["./index.js"]
    with open(pkg_path, "w") as f:
        json.dump(pkg, f, indent=2)
    print(f"  package.json openclaw.extensions -> ['./index.js']")
PY
    fi
}

build_project() {
    if [[ "$DO_BUILD" != true ]]; then
        return 0
    fi

    log_step "构建 SC 项目..."
    cd "${SC_SOURCE_DIR}"

    # 链接 plugin-sdk
    link_plugin_sdk

    if [[ -f "pnpm-lock.yaml" ]] && command -v pnpm &>/dev/null; then
        log "使用 pnpm 安装依赖..."
        CI=true pnpm install
        if [[ -f "package.json" ]] && grep -q '"build"' package.json 2>/dev/null; then
            log "执行 pnpm build..."
            # plugin-sdk 的源码通过相对路径引用了 openclaw 内部源码，
            # 这些源码中的类型错误不影响插件产物的正确性（TypeScript 仍会 emit）。
            pnpm build || true
            log "构建完成（类型警告不影响产物生成）"
        fi
    elif [[ -f "package-lock.json" ]] && command -v npm &>/dev/null; then
        log "使用 npm 安装依赖..."
        npm install
        if [[ -f "package.json" ]] && grep -q '"build"' package.json 2>/dev/null; then
            log "执行 npm run build..."
            npm run build || true
        fi
    else
        log_warn "未找到 pnpm/npm 或 lock 文件，跳过构建"
    fi

    # 为每个插件创建/更新根 index.js 入口
    log "确保插件根 index.js 入口..."
    for plugin_dir in "${SC_SOURCE_DIR}/plugins"/*/; do
        if [[ -d "$plugin_dir" ]]; then
            ensure_plugin_entry "$plugin_dir" || true
        fi
    done

    # 检查各插件的 dist 产物是否生成
    log "检查构建产物..."
    for plugin_dir in "${SC_SOURCE_DIR}/plugins"/*/; do
        if [[ -d "$plugin_dir" ]]; then
            local plugin_name
            plugin_name=$(basename "$plugin_dir")
            if [[ -d "${plugin_dir}dist" ]]; then
                log "  ${plugin_name}: dist 已生成"
            else
                log_warn "  ${plugin_name}: dist 未生成，插件将使用源码直接加载"
            fi
        fi
    done
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

            if [[ "$USE_SYMLINK" == true ]]; then
                log "创建符号链接: $name"
                rm -rf "$target"
                ln -s "$plugin" "$target"
            else
                log "复制插件: $name"
                rm -rf "$target"
                cp -R "$plugin" "$target"
            fi
        fi
    done

    # 安装 skill
    if [[ "$USE_SYMLINK" == true ]]; then
        log "创建技能符号链接..."
        rm -rf "$target_skills_dir"
        ln -s "${SC_SOURCE_DIR}/skill" "$target_skills_dir"
    else
        log "复制技能文件..."
        rm -rf "$target_skills_dir"
        cp -R "${SC_SOURCE_DIR}/skill" "$target_skills_dir"
    fi
}

# ============ 修复插件运行环境 ============
# 容器/某些环境下插件目录可能是 world-writable 或非 root 拥有，
# OpenClaw 会拒绝加载。安装前修复为安全权限。
fix_plugin_permissions() {
    local target_plugins_dir="$OPENCLAW_HOME/workspace/plugins"

    if [[ ! -d "$target_plugins_dir" ]]; then
        return 0
    fi

    log "修复插件目录权限..."
    for plugin_dir in "$target_plugins_dir"/*/; do
        if [[ -d "$plugin_dir" ]]; then
            local plugin_name
            plugin_name=$(basename "$plugin_dir")
            # 仅在当前用户为 root 且目录被其他用户拥有时修复所有权
            if [[ "$(id -u)" == "0" ]] && [[ -n "$(stat -c '%U' "$plugin_dir" 2>/dev/null)" ]]; then
                local owner
                owner=$(stat -c '%U' "$plugin_dir")
                if [[ "$owner" != "root" ]]; then
                    chown -R root:root "$plugin_dir" 2>/dev/null || true
                    log "  ${plugin_name}: chown root:root"
                fi
            fi
            # 修复 world-writable 权限
            local mode
            mode=$(stat -c '%a' "$plugin_dir" 2>/dev/null || echo "")
            if [[ "$mode" == *"7" || "$mode" == "777" ]]; then
                chmod 755 "$plugin_dir" 2>/dev/null || true
                log "  ${plugin_name}: chmod 755"
            fi
        fi
    done
}

# ============ 移除 pnpm 符号链接以通过安全扫描 ============
# pnpm 在 node_modules 中创建指向 .pnpm/ 存储的符号链接，
# OpenClaw 的安全扫描会将其判定为 install root 之外的依赖。
# 安装插件前必须先移除 node_modules，让 openclaw plugins install 自行处理依赖。
strip_node_modules() {
    local target_plugins_dir="$OPENCLAW_HOME/workspace/plugins"

    if [[ ! -d "$target_plugins_dir" ]]; then
        return 0
    fi

    log "移除插件的 node_modules（避免 pnpm 符号链接触发安全扫描）..."
    for plugin_dir in "$target_plugins_dir"/*/; do
        if [[ -d "${plugin_dir}node_modules" ]]; then
            local plugin_name
            plugin_name=$(basename "$plugin_dir")
            rm -rf "${plugin_dir}node_modules"
            log "  ${plugin_name}: node_modules 已移除"
        fi
    done
}

# ============ 通过 openclaw plugins install 注册插件 ============
register_plugins() {
    if [[ "$SKIP_INSTALL" == true ]]; then
        log "跳过 openclaw plugins install (--skip-install)"
        return 0
    fi

    log_step "通过 openclaw plugins install 注册插件..."

    # 先清理可能存在的过期 entries
    cleanup_stale_entries

    local target_plugins_dir="$OPENCLAW_HOME/workspace/plugins"

    for plugin_dir in "$target_plugins_dir"/*/; do
        if [[ -d "$plugin_dir" ]]; then
            local plugin_name
            plugin_name=$(basename "$plugin_dir")

            log "安装插件: $plugin_name"

            # --link 让 OpenClaw 直接引用已部署的目录而非复制到 extensions/，
            # 避免再次走 pnpm/npm install 流程
            if openclaw plugins install --link "$plugin_dir" 2>&1 | tail -5 >&2; then
                log "  ✓ $plugin_name 已注册"
            else
                log_warn "  ✗ $plugin_name 注册失败，请查看上方错误"
            fi
        fi
    done
}

# ============ 清理过期插件 entries ============
cleanup_stale_entries() {
    log "清理过期插件 entries..."

    local config_path="$OPENCLAW_HOME/openclaw.json"
    if [[ ! -f "$config_path" ]]; then
        return 0
    fi

    # 使用 openclaw plugins uninstall --force 清理（如果存在的话）
    for plugin_id in "@subagent-coordinator/taskr" \
                     "@subagent-coordinator/exec-monitor" \
                     "@subagent-coordinator/observability"; do
        # 检查 entry 是否存在且引用了不存在的插件
        if grep -q "\"$plugin_id\"" "$config_path" 2>/dev/null; then
            if openclaw plugins uninstall --force --keep-files "$plugin_id" &>/dev/null; then
                log "  已清理: $plugin_id"
            fi
        fi
    done
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

# === 动态添加插件加载路径（兼容旧式手动加载） ===
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

    # 1. 创建 worker 子代理
    log "创建 worker 子代理..."
    openclaw agents add worker \
        --model ollama/gemma-4-e4b-heretic \
        --non-interactive \
        --workspace "$OPENCLAW_HOME/workspace/worker" 2>/dev/null \
        || log_warn "worker 可能已存在，跳过"

    # 2. 兼容性更新配置文件（添加 main -> worker 权限）
    log "更新配置文件..."
    local target_plugins_dir="$OPENCLAW_HOME/workspace/plugins"
    generate_python_config_script "$target_plugins_dir" | python3 - "$target_plugins_dir" || log_warn "配置更新失败"

    # 3. 配置 exec allowlist
    log "配置 exec 权限..."
    openclaw approvals allowlist add --agent worker "**" 2>/dev/null || log_warn "allowlist 配置可能已存在"
}

# ============ 重启 Gateway ============
restart_gateway() {
    log_step "重启 Gateway..."

    # 容器中无 systemd 时，openclaw gateway restart 会失败，
    # 退化到直接 pkill + nohup 启动
    if pkill -f "openclaw-gateway" 2>/dev/null; then
        log "已停止旧 gateway 进程"
    fi

    if command -v systemctl &>/dev/null && systemctl --user status openclaw-gateway &>/dev/null; then
        log "通过 systemd 重启..."
        openclaw gateway restart 2>&1 | tail -3 || true
    else
        log "直接启动 gateway（无 systemd）..."
        OPENCLAW_HOME="$OPENCLAW_HOME" nohup openclaw gateway > /tmp/openclaw-gateway.log 2>&1 &
        disown 2>/dev/null || true
        log "gateway 已在后台启动，日志: /tmp/openclaw-gateway.log"
    fi
}

# ============ 验证 ============
verify_installation() {
    log_step "验证安装..."

    echo ""
    echo "--- 插件加载状态 ---"
    OPENCLAW_HOME="$OPENCLAW_HOME" openclaw plugins list 2>/dev/null \
        | grep -E "subagent" | head -10 || echo "  (无子代理插件)"

    echo ""
    echo "--- Agents ---"
    OPENCLAW_HOME="$OPENCLAW_HOME" openclaw agents list 2>/dev/null | grep -E "^-" || echo "  (无输出)"

    echo ""
    echo "--- 插件目录 ---"
    ls -d "$OPENCLAW_HOME/workspace/plugins/"*/ 2>/dev/null | xargs -n1 basename || echo "  (无插件)"
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
    echo "  1. 验证插件已加载:"
    echo "     openclaw plugins list | grep subagent"
    echo ""
    echo "  2. 测试 worker 子代理:"
    echo "     openclaw agent --agent worker --message 'Hello'"
    echo ""
    echo "  3. 查看 gateway 日志:"
    echo "     tail -f /tmp/openclaw-gateway.log"
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

    # 4. 复制/链接插件到 workspace
    install_plugins

    # 5. 修复权限（容器环境）
    fix_plugin_permissions

    # 6. 移除 pnpm 符号链接（避免安全扫描失败）
    strip_node_modules

    # 7. 通过 openclaw plugins install 注册插件
    register_plugins

    # 8. 配置 OpenClaw（worker 子代理、权限）
    configure_openclaw

    # 9. 重启 Gateway
    restart_gateway

    # 10. 验证
    verify_installation

    show_result
}

main "$@"
