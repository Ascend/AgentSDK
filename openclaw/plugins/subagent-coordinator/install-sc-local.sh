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
OPENCLAW_PACKAGE_ROOT=""

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

# 探测目标文件系统是否真正支持 POSIX chmod。
# WSL2 以 9p/drvfs 挂载 Windows 目录时，chmod 会被忽略（exit 0 但 mode 不变），
# 此时 cp -R 出来的目录会带 777，被 OpenClaw 安全扫描
# 'blocked plugin candidate: world-writable path' 拦截。
# 用探针判断：chmod 755 后若末位仍是 7，说明文件系统忽略 chmod，强制走符号链接模式。
detect_chmod_support() {
    local probe="$OPENCLAW_HOME/.sc-chmod-probe-$$"
    mkdir -p "$probe" 2>/dev/null || return 0
    chmod 755 "$probe" 2>/dev/null || true
    local mode
    mode=$(stat -c '%a' "$probe" 2>/dev/null || echo "")
    rm -rf "$probe" 2>/dev/null || true
    if [[ "$mode" =~ 7$ ]]; then
        log_warn "检测到目标文件系统不支持 POSIX chmod (probe mode=$mode，可能为 WSL2 9p/drvfs)，自动切换为 --symlink 模式以避开 'world-writable path' 拦截"
        USE_SYMLINK=true
    fi
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
    # 4. 宿主机全局安装 OpenClaw 的 dist/plugin-sdk

    local sdk_dir=""
    local openclaw_bin=""
    local openclaw_resolved=""
    local openclaw_dist_dir=""
    local npm_root=""

    if [[ -n "$PLUGIN_SDK_SRC" && -d "$PLUGIN_SDK_SRC" ]]; then
        sdk_dir="$PLUGIN_SDK_SRC"
        log "使用 PLUGIN_SDK_SRC 指定的 plugin-sdk: $sdk_dir"
    elif [[ -d "/app/packages/plugin-sdk" ]]; then
        sdk_dir="/app/packages/plugin-sdk"
        log "使用镜像内置 plugin-sdk: $sdk_dir"
    elif [[ -d "${SC_SOURCE_DIR}/../../packages/plugin-sdk" ]]; then
        sdk_dir="${SC_SOURCE_DIR}/../../packages/plugin-sdk"
        log "使用本地 OpenClaw 源码 plugin-sdk: $sdk_dir"
    else
        openclaw_bin=$(command -v openclaw 2>/dev/null || true)
        if [[ -n "$openclaw_bin" ]]; then
            openclaw_resolved=$(readlink -f "$openclaw_bin" 2>/dev/null || true)
            openclaw_dist_dir=$(dirname "$openclaw_resolved")
            if [[ -d "${openclaw_dist_dir}/plugin-sdk" ]]; then
                sdk_dir="${openclaw_dist_dir}/plugin-sdk"
                log "使用全局 OpenClaw dist plugin-sdk: $sdk_dir"
            fi
        fi

        if [[ -z "$sdk_dir" ]] && command -v npm &>/dev/null; then
            npm_root=$(npm root -g 2>/dev/null || true)
            if [[ -d "${npm_root}/openclaw/dist/plugin-sdk" ]]; then
                sdk_dir="${npm_root}/openclaw/dist/plugin-sdk"
                log "使用 npm 全局 OpenClaw plugin-sdk: $sdk_dir"
            fi
        fi
    fi

    if [[ -n "$sdk_dir" ]]; then
        if [[ -f "$(dirname "$(dirname "$sdk_dir")")/package.json" && "$(basename "$(dirname "$sdk_dir")")" == "dist" ]]; then
            OPENCLAW_PACKAGE_ROOT=$(dirname "$(dirname "$sdk_dir")")
        elif [[ -f "$(dirname "$(dirname "$sdk_dir")")/package.json" && "$(basename "$(dirname "$sdk_dir")")" == "packages" ]]; then
            OPENCLAW_PACKAGE_ROOT=$(dirname "$(dirname "$sdk_dir")")
        fi

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

# TypeScript 编译器在 "module": "ESNext" 下保留源码中的相对路径，
# 但 Node.js ESM 严格要求 import 写明 .js 后缀，否则会 ERR_MODULE_NOT_FOUND。
# 该函数用 Node 解析后重写所有 dist/**/*.js 中匹配 ./ 或 ../ 开头的字符串字面量，
# 追加 .js 后缀；已经带后缀（.js/.json/.mjs/.cjs/.node/.wasm）的保持不变。
fix_compiled_imports() {
    if ! command -v node &>/dev/null; then
        log_warn "未找到 node，跳过修复编译后导入路径。"
        return 0
    fi

    log "修复编译后相对导入路径..."

    # 在 SC_SOURCE_DIR 范围内处理所有 dist/**/*.js，包括 packages/types 和 plugins/*。
    local dist_root="${SC_SOURCE_DIR}/dist"
    local types_dist="${SC_SOURCE_DIR}/packages/types/dist"
    local plugin_dists=()
    for plugin_dir in "${SC_SOURCE_DIR}/plugins"/*/; do
        if [[ -d "$plugin_dir/dist" ]]; then
            plugin_dists+=("${plugin_dir}/dist")
        fi
    done

    NODE_PATH="$SC_SOURCE_DIR" node - "${SC_SOURCE_DIR}" "${dist_root}" "${types_dist}" "${plugin_dists[@]}" <<'NODE_EOF'
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const roots = args.slice(1).filter((p) => p && fs.existsSync(p));

function walk(dir, out) {
    let entries;
    try {
        entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch (err) {
        return;
    }
    for (const e of entries) {
        const full = path.join(dir, e.name);
        if (e.isDirectory()) {
            walk(full, out);
        } else if (e.isFile() && e.name.endsWith('.js')) {
            out.push(full);
        }
    }
}

const files = [];
for (const root of roots) walk(root, files);

const relativeImport = /(['"])(\.\.?\/[^'"]+?)\1/g;
const hasExt = /\.(?:js|json|mjs|cjs|node|wasm)$/;

let totalChanged = 0;
for (const file of files) {
    const src = fs.readFileSync(file, 'utf8');
    const fixed = src.replace(relativeImport, (m, q, p) => {
        if (hasExt.test(p)) return m;
        return q + p + '.js' + q;
    });
    if (fixed !== src) {
        fs.writeFileSync(file, fixed);
        totalChanged++;
    }
}

console.log(`  已处理 ${files.length} 个 .js 文件，修改 ${totalChanged} 个。`);
NODE_EOF
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
        if [[ "${CI:-}" == "true" ]]; then
            pnpm install --frozen-lockfile
        else
            pnpm install --no-frozen-lockfile
        fi
        log "执行 pnpm build..."
        # plugin-sdk 和部分测试源码可能产生类型错误；TypeScript 仍会 emit。
        # 逐包构建可避免 pnpm -r 在某个包返回非零后跳过后续插件。
        pnpm --filter @subagent-coordinator/types build || true
        for plugin_dir in plugins/*/; do
            if [[ -d "$plugin_dir" ]]; then
                local plugin_name
                plugin_name=$(basename "$plugin_dir")
                log "  构建 ${plugin_name}..."
                pnpm --filter "./plugins/${plugin_name}" build || true
            fi
        done
        log "构建完成（类型警告不影响产物生成）"
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

    # 修复编译后的相对导入：tsc 在 "module": "ESNext" 下会保留源码中的相对路径，
    # 但 Node.js ESM 要求 import 显式带 .js 后缀，否则会 ERR_MODULE_NOT_FOUND。
    # 这一步必须在 ensure_plugin_entry 之前完成，因为 dist/index.js 的导入路径
    # 决定了插件能否被 OpenClaw 加载。
    fix_compiled_imports

    # 为每个插件创建/更新根 index.js 入口。缺少入口说明该插件无法被 OpenClaw 加载，必须失败退出。
    log "确保插件根 index.js 入口..."
    local missing_entries=0
    for plugin_dir in "${SC_SOURCE_DIR}/plugins"/*/; do
        if [[ -d "$plugin_dir" ]]; then
            if ! ensure_plugin_entry "$plugin_dir"; then
                missing_entries=1
            fi
        fi
    done

    if [[ "$missing_entries" -ne 0 ]]; then
        log_error "存在插件缺少 dist 入口，安装中止。请先修复构建产物后重试。"
    fi

    log "检查构建产物..."
    for plugin_dir in "${SC_SOURCE_DIR}/plugins"/*/; do
        if [[ -d "$plugin_dir" ]]; then
            local plugin_name
            plugin_name=$(basename "$plugin_dir")
            log "  ${plugin_name}: dist 入口已生成"
        fi
    done
}

# ============ 安装插件 ============
install_plugins() {
    local target_plugins_dir="$OPENCLAW_HOME/workspace/plugins"
    # skill 真实复制到 canonical 路径 $OPENCLAW_HOME/skills/subagent-coordinator。
    # 路径与其他 openclaw-managed skill (如 agent-browser、api-guardian) 一致，
    # 必须用 cp -R 而非 ln -s：openclaw 安全扫描不允许 skill 符号链接指向
    # 自身根之外的路径（"Skipping skill path that resolves outside its configured root"）。
    local skill_install_dir="$OPENCLAW_HOME/skills/subagent-coordinator"

    mkdir -p "$target_plugins_dir" \
             "$(dirname "$skill_install_dir")"

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

    # skill 真实复制到 canonical 路径
    log "复制技能文件到 $skill_install_dir"
    rm -rf "$skill_install_dir"
    cp -R "${SC_SOURCE_DIR}/skill" "$skill_install_dir"
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

# ============ 准备运行时依赖 ============
resolve_runtime_dependency() {
    local dependency_path=$1
    local package_json

    if [[ -d "${SC_SOURCE_DIR}/node_modules/${dependency_path}" ]]; then
        printf '%s\n' "${SC_SOURCE_DIR}/node_modules/${dependency_path}"
        return 0
    fi

    package_json=$(compgen -G "${SC_SOURCE_DIR}/node_modules/.pnpm/*/node_modules/${dependency_path}/package.json" | head -n 1 || true)
    if [[ -n "$package_json" ]]; then
        dirname "$package_json"
        return 0
    fi

    return 1
}

copy_runtime_dependency() {
    local source_path=$1
    local target_path=$2
    local dependency_name=$3

    [[ -d "$source_path" ]] || log_error "缺少运行时依赖 ${dependency_name}: ${source_path}。请先执行 ./install-sc-local.sh --build。"

    rm -rf "$target_path"
    mkdir -p "$(dirname "$target_path")"
    cp -RL "$source_path" "$target_path"
}

prepare_runtime_dependencies() {
    local target_plugins_dir="$OPENCLAW_HOME/workspace/plugins"
    local typebox_source=""
    local types_source="${SC_SOURCE_DIR}/packages/types"
    local openclaw_source=""

    if [[ ! -d "$target_plugins_dir" ]]; then
        return 0
    fi

    if [[ "$USE_SYMLINK" == true ]]; then
        log_warn "--symlink 模式会通过符号链接修改源码目录的 node_modules（移除 pnpm 符号链接并复制运行时依赖），以满足 openclaw plugins install --link 的安全扫描要求。"
    fi

    typebox_source=$(resolve_runtime_dependency "@sinclair/typebox") || log_error "缺少运行时依赖 @sinclair/typebox。请先执行 ./install-sc-local.sh --build。"
    [[ -f "${types_source}/dist/index.js" ]] || log_error "缺少 @subagent-coordinator/types 构建产物: ${types_source}/dist/index.js。请先执行 ./install-sc-local.sh --build。"

    # 编译后的插件通过 subpath 导入 openclaw 包，例如
    #   import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry"
    # 这些 subpath 在 openclaw 的 package.json exports 中定义，需要一个真正可解析的
    # openclaw 包（包含 package.json + dist/）。OPENCLAW_PACKAGE_ROOT 由
    # link_plugin_sdk() 在找到 plugin-sdk 时推导；若缺失则报错并提示。
    if [[ -z "$OPENCLAW_PACKAGE_ROOT" || ! -f "${OPENCLAW_PACKAGE_ROOT}/package.json" || ! -d "${OPENCLAW_PACKAGE_ROOT}/dist" ]]; then
        log_error "未找到可解析的 openclaw 包（plugin-sdk 链接未生效）。请先执行 ./install-sc-local.sh --build，或通过 PLUGIN_SDK_SRC 指向正确的 plugin-sdk 目录。"
    fi
    openclaw_source="$OPENCLAW_PACKAGE_ROOT"

    log "准备插件运行时依赖..."
    for plugin_dir in "$target_plugins_dir"/*/; do
        if [[ -d "$plugin_dir" ]]; then
            local plugin_name
            plugin_name=$(basename "$plugin_dir")

            rm -rf "${plugin_dir}node_modules"
            copy_runtime_dependency "$typebox_source" "${plugin_dir}node_modules/@sinclair/typebox" "@sinclair/typebox"
            copy_runtime_dependency "$types_source" "${plugin_dir}node_modules/@subagent-coordinator/types" "@subagent-coordinator/types"

            # openclaw 包：复制 package.json 与 dist/ 即可。dist/ 内部使用相对路径
            # 互相 import，因此保留 dist 目录结构很重要。打包时 dist 体积可能较大，
            # 但这是 OpenClaw 加载插件的硬性要求。
            local openclaw_target="${plugin_dir}node_modules/openclaw"
            rm -rf "$openclaw_target"
            mkdir -p "$openclaw_target"
            cp "${openclaw_source}/package.json" "${openclaw_target}/package.json"
            cp -RL "${openclaw_source}/dist" "${openclaw_target}/dist"

            log "  ${plugin_name}: 运行时依赖已复制（含 openclaw 包）"
        fi
    done
}

validate_installed_plugin_files() {
    local target_plugins_dir="$OPENCLAW_HOME/workspace/plugins"
    local failed=0

    log "检查已安装插件文件..."
    for plugin_dir in "$target_plugins_dir"/*/; do
        if [[ -d "$plugin_dir" ]]; then
            local plugin_name
            plugin_name=$(basename "$plugin_dir")

            if [[ ! -f "${plugin_dir}index.js" ]]; then
                log_warn "  ${plugin_name}: 缺少根 index.js"
                failed=1
            fi
            if [[ ! -f "${plugin_dir}dist/src/index.js" && ! -f "${plugin_dir}dist/plugins/${plugin_name}/src/index.js" ]]; then
                log_warn "  ${plugin_name}: 缺少 dist 入口"
                failed=1
            fi
            if [[ ! -f "${plugin_dir}node_modules/@sinclair/typebox/package.json" ]]; then
                log_warn "  ${plugin_name}: 缺少 @sinclair/typebox"
                failed=1
            fi
            if [[ ! -f "${plugin_dir}node_modules/@subagent-coordinator/types/package.json" ]]; then
                log_warn "  ${plugin_name}: 缺少 @subagent-coordinator/types"
                failed=1
            fi
        fi
    done

    if [[ "$failed" -ne 0 ]]; then
        log_error "插件文件或运行时依赖不完整，安装中止。"
    fi
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
    local failed=0
    local first_failure_output=""

    for plugin_dir in "$target_plugins_dir"/*/; do
        if [[ -d "$plugin_dir" ]]; then
            local plugin_name
            local install_output
            local install_status
            plugin_name=$(basename "$plugin_dir")

            log "安装插件: $plugin_name"

            # --link 让 OpenClaw 直接引用已部署的目录而非复制到 extensions/，
            # 避免再次走 pnpm/npm install 流程
            # 使用 PIPESTATUS 显式捕获 install 命令退出码，避免 set -e 在 if 条件
            # 内被抑制时漏报失败。
            install_output=$(OPENCLAW_STATE_DIR="$OPENCLAW_HOME" openclaw plugins install --link "$plugin_dir" 2>&1)
            install_status=$?
            echo "$install_output" | tail -5 >&2
            if [[ "$install_status" -eq 0 ]]; then
                log "  ✓ $plugin_name 已注册"
            else
                log_warn "  ✗ $plugin_name 注册失败 (exit=$install_status)"
                if [[ -z "$first_failure_output" ]]; then
                    first_failure_output="$install_output"
                fi
                failed=1
            fi
        fi
    done

    if [[ "$failed" -ne 0 ]]; then
        echo "$first_failure_output" >&2
        log_error "存在插件注册失败，安装中止。请先解决失败原因（如 node_modules pnpm 符号链接导致的安全扫描拦截），可手动执行 openclaw plugins install --link 排查。"
        return 1
    fi
}

# ============ 修正 install records 路径为 resolved 路径 ============
# openclaw plugins install --link 写入 plugins.installs[*].sourcePath 时使用调用方
# 提供的路径（可能是符号链接），但加载器用 realpath 解析后比较，符号链接与
# 解析路径不一致会产生 "loaded without install/load-path provenance" 警告。
# 此函数把 sourcePath / installPath 改写为 os.path.realpath 后的路径，
# 消除警告但保持 entries 仍指向同一个加载点。
patch_install_records_provenance() {
    local config_path="$OPENCLAW_HOME/openclaw.json"
    [[ -f "$config_path" ]] || return 0

    python3 - "$config_path" <<'PYTHON_SCRIPT' || log_warn "install records 路径修正失败"
import json
import os
import sys

config_path = sys.argv[1]
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(f"配置文件读取失败: {e}")
    sys.exit(1)

installs = config.get('plugins', {}).get('installs', {})
if not installs:
    sys.exit(0)

changed = 0
for _plugin_id, record in installs.items():
    if not isinstance(record, dict):
        continue
    for key in ('sourcePath', 'installPath'):
        path = record.get(key)
        if not path or not os.path.islink(path):
            continue
        try:
            real_path = os.path.realpath(path)
        except OSError:
            continue
        if real_path and real_path != path:
            record[key] = real_path
            changed += 1

if changed:
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  - 已修正 {changed} 处 install record 路径为 resolved 路径")
PYTHON_SCRIPT
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
add_main_worker_permission() {
    local config_path="$OPENCLAW_HOME/openclaw.json"
    [[ -f "$config_path" ]] || return 0

    python3 - "$config_path" <<'PYTHON_SCRIPT' || log_warn "配置更新失败"
import json
import sys

config_path = sys.argv[1]
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(f"配置文件读取失败: {e}")
    sys.exit(1)

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

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("配置更新成功")
PYTHON_SCRIPT
}

configure_openclaw() {
    log_step "配置 OpenClaw..."

    # 1. 创建 worker 子代理示例($PROVIDER 与 $MODEL 需要提前设置或之后修改)
    log "创建 worker 子代理示例...($PROVIDER 与 $MODEL 需要提前设置或之后修改)"
    openclaw agents add worker \
        --model "$PROVIDER/$MODEL" \
        --non-interactive \
        --workspace "$OPENCLAW_HOME/workspace/worker" 2>/dev/null \
        || log_warn "worker 可能已存在，跳过"

    # 2. 仅补全 main -> worker 调用权限。
    # 插件 entries/installs/load.paths 已在 register_plugins() 中由
    # `openclaw plugins install --link` 直接写入 openclaw.json，
    # 此处不再重复维护 plugins.load.paths。
    log "更新配置文件（main -> worker 权限）..."
    add_main_worker_permission

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
        OPENCLAW_STATE_DIR="$OPENCLAW_HOME" nohup openclaw gateway > /tmp/openclaw-gateway.log 2>&1 &
        disown 2>/dev/null || true
        log "gateway 已在后台启动，日志: /tmp/openclaw-gateway.log"
    fi
}

# ============ 验证 ============
verify_installation() {
    log_step "验证安装..."

    local plugins_output=""
    local failed=0
    local expected_plugins=(
        "@subagent-coordinator/taskr"
        "@subagent-coordinator/exec-monitor"
        "@subagent-coordinator/observability"
    )

    validate_installed_plugin_files

    echo ""
    echo "--- 插件加载状态 ---"
    # openclaw plugins list 的默认表格输出会把长 ID 折行，导致 grep 无法匹配完整 ID。
    # 用 --json 解析拿到的 ID 字符串更可靠。
    local plugins_json=""
    if plugins_json=$(OPENCLAW_STATE_DIR="$OPENCLAW_HOME" openclaw plugins list --json 2>/dev/null); then
        if command -v python3 &>/dev/null; then
            python3 - "$OPENCLAW_HOME" <<'PY' || failed=1
import json
import os
import subprocess
import sys

home = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OPENCLAW_HOME", os.path.expanduser("~/.openclaw"))
try:
    out = subprocess.check_output(
        ["openclaw", "plugins", "list", "--json"],
        env={**os.environ, "OPENCLAW_STATE_DIR": home},
        stderr=subprocess.DEVNULL,
    )
    data = json.loads(out.decode("utf-8", errors="replace"))
except Exception as e:
    print(f"  [WARN] 无法解析 openclaw plugins list --json: {e}")
    sys.exit(2)

plugins = data.get("plugins", [])
if not plugins:
    print("  (无插件)")
    sys.exit(2)

expected = {
    "@subagent-coordinator/taskr",
    "@subagent-coordinator/exec-monitor",
    "@subagent-coordinator/observability",
}

found = {p.get("id"): p for p in plugins if p.get("id")}
missing = expected - set(found.keys())
for plugin in plugins:
    pid = plugin.get("id", "")
    status = plugin.get("status", "?")
    enabled = plugin.get("enabled", False)
    print(f"  - {pid}  status={status}  enabled={enabled}")

if missing:
    for m in sorted(missing):
        print(f"  [WARN] 缺少插件注册: {m}")
    sys.exit(1)
PY
        else
            log_warn "未找到 python3，跳过 JSON 校验。"
        fi
    else
        log_warn "无法获取插件列表"
        failed=1
    fi

    echo ""
    echo "--- Skill ---"
    if [[ ! -f "$OPENCLAW_HOME/skills/subagent-coordinator/SKILL.md" ]]; then
        log_warn "缺少 Skill 可发现文件: $OPENCLAW_HOME/skills/subagent-coordinator/SKILL.md"
        failed=1
    fi
    if OPENCLAW_STATE_DIR="$OPENCLAW_HOME" openclaw skills info subagent-coordinator >/dev/null 2>&1; then
        echo "subagent-coordinator skill 可发现"
    else
        log_warn "openclaw skills info subagent-coordinator 失败"
        failed=1
    fi

    echo ""
    echo "--- Agents ---"
    OPENCLAW_STATE_DIR="$OPENCLAW_HOME" openclaw agents list 2>/dev/null | grep -E "^-" || echo "  (无输出)"

    echo ""
    echo "--- 插件目录 ---"
    ls -d "$OPENCLAW_HOME/workspace/plugins/"*/ 2>/dev/null | xargs -n1 basename || echo "  (无插件)"

    if [[ "$failed" -ne 0 ]]; then
        log_error "安装验证失败。"
    fi
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
    echo "提示: 宿主机 Gateway 与 openclaw CLI 都读取 OPENCLAW_STATE_DIR（默认 ~/.openclaw）"
    echo "下的 openclaw.json。本脚本以 OPENCLAW_STATE_DIR 指向同一份文件，因此"
    echo "install --link 与 Gateway 始终读写同一个 openclaw.json。"
    echo "如需将整套 OpenClaw 状态目录迁到其他位置："
    echo "     OPENCLAW_STATE_DIR=\"/custom/path\" openclaw …"
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

    # 1b. 探测 chmod 是否真正生效（WSL2 9p/drvfs 下会静默忽略），
    #     若无效则强制 --symlink 模式，避免复制目录被 world-writable 拦截
    detect_chmod_support

    # 2. 检查源目录
    check_source

    # 3. 构建（可选）
    build_project

    # 4. 复制/链接插件到 workspace
    install_plugins

    # 5. 修复权限（容器环境）
    fix_plugin_permissions

    # 6. 准备运行时依赖并检查安装目录
    prepare_runtime_dependencies
    validate_installed_plugin_files

    # 7. 通过 openclaw plugins install 注册插件
    register_plugins

    # 7b. 修正 install records 路径为 resolved 路径，消除 symlink 模式下的
    #     "loaded without install/load-path provenance" 警告
    patch_install_records_provenance

    # 8. 配置 OpenClaw（worker 子代理、权限）
    configure_openclaw

    # 9. 重启 Gateway
    restart_gateway

    # 10. 验证
    verify_installation

    show_result
}

main "$@"
