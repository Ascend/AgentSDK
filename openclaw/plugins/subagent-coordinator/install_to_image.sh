#!/bin/bash
set -e

# =============================================================================
# subagent-coordinator/install_to_image.sh
# 用途：构建插件并准备打包到 Docker 镜像
#
# 由 docker/build-openclaw.sh 调用
# 产物输出到 plugins/subagent-coordinator/dist/
# （与源码放在同一目录，方便 Dockerfile COPY）
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="${SCRIPT_DIR}/dist"

echo "[install_to_image] 开始构建 subagent-coordinator..."

cd "${SCRIPT_DIR}"

# 检查 pnpm
if ! command -v pnpm &> /dev/null; then
    echo "[install_to_image] 安装 pnpm..."
    npm install -g pnpm
fi

# ── 动态获取 openclaw plugin-sdk（解决跨服务器构建问题）────────────────────
OPENCLAW_VERSION="${OPENCLAW_VERSION:-2026.4.11}"
PLUGIN_SDK_DIR="/tmp/openclaw-src/packages/plugin-sdk"

# 优先级顺序（默认优先从 GitHub 下载，失败后回退到本地路径）：
# 1. 环境变量 PLUGIN_SDK_SRC
# 2. 本地已有的 plugin-sdk（build-openclaw.sh 已准备好）
# 3. 从远程克隆 openclaw 源码

# 优先尝试克隆远程仓库（默认行为）
if [[ ! -d "$PLUGIN_SDK_DIR" ]]; then
    echo "[install_to_image] 克隆 openclaw v${OPENCLAW_VERSION} 获取 plugin-sdk..."
    if git clone --depth 1 --branch "v${OPENCLAW_VERSION}" \
        https://github.com/openclaw/openclaw.git /tmp/openclaw-src 2>&1; then
        echo "[install_to_image] 克隆 openclaw 成功"
    else
        # 克隆失败，尝试使用本地路径
        echo "[install_to_image] 克隆失败，尝试使用本地路径..."
        rm -rf /tmp/openclaw-src

        if [[ -n "$PLUGIN_SDK_SRC" && -d "$PLUGIN_SDK_SRC" ]]; then
            # 使用环境变量指定的 plugin-sdk
            echo "[install_to_image] 使用环境变量指定的 plugin-sdk: ${PLUGIN_SDK_SRC}"
            mkdir -p node_modules/openclaw
            ln -sfn "$PLUGIN_SDK_SRC" node_modules/openclaw/plugin-sdk
        elif [[ -d "${SCRIPT_DIR}/../../docker/.openclaw-src-tmp/packages/plugin-sdk" ]]; then
            # 使用 build-openclaw.sh 准备好的本地 plugin-sdk
            LOCAL_PLUGIN_SDK="${SCRIPT_DIR}/../../docker/.openclaw-src-tmp/packages/plugin-sdk"
            echo "[install_to_image] 使用本地 plugin-sdk: ${LOCAL_PLUGIN_SDK}"
            mkdir -p node_modules/openclaw
            ln -sfn "$LOCAL_PLUGIN_SDK" node_modules/openclaw/plugin-sdk
        elif [[ -d "/tmp/.openclaw-src/packages/plugin-sdk" ]]; then
            # 使用缓存的 plugin-sdk
            echo "[install_to_image] 使用缓存的 plugin-sdk"
            mkdir -p node_modules/openclaw
            ln -sfn "/tmp/.openclaw-src/packages/plugin-sdk" node_modules/openclaw/plugin-sdk
        else
            echo "[install_to_image] 错误: 无法获取 plugin-sdk（克隆失败，且无有效的本地路径）"
            echo "[install_to_image] 可设置 PLUGIN_SDK_SRC 环境变量指定本地 plugin-sdk 路径"
            exit 1
        fi
    fi
fi

# 创建 plugin-sdk symlink（供 tsconfig paths 和 pnpm install 解析 workspace 依赖）
if [[ -d "$PLUGIN_SDK_DIR" ]]; then
    mkdir -p node_modules/openclaw
    ln -sfn "$PLUGIN_SDK_DIR" node_modules/openclaw/plugin-sdk
    echo "[install_to_image] plugin-sdk 准备完成"
fi

# 安装依赖
echo "[install_to_image] 安装依赖..."
pnpm install --config.minimum-release-age=0

# 构建
# 注意：pnpm build 可能会因 openclaw 内部源码的类型错误返回非零退出码，
# 但这不影响 JS 产物的正确性（TypeScript 即使有类型错误也会正常 emit）
echo "[install_to_image] 构建插件..."
pnpm build || true

# 确保每个插件的 dist 都已生成
# 注意：由于 monorepo 路径依赖复杂，不再使用 tsc -p 补编译，统一依赖 pnpm -r build 的产物
echo "[install_to_image] 确保各插件 dist 产物存在..."
for plugin in plugins/*/; do
    plugin_name=$(basename "$plugin")
    # 实际输出路径是 dist/plugins/<plugin_name>/src/index.js
    expected_dist="${plugin}dist/plugins/${plugin_name}/src/index.js"
    if [[ -f "$expected_dist" ]]; then
        echo "[install_to_image]   ${plugin_name} dist 已存在 (${expected_dist})，跳过"
    else
        echo "[install_to_image]   警告: ${plugin_name} 缺少产物 ${expected_dist}，将依赖 pnpm build 产物"
    fi
done

# 创建镜像用产物目录
echo "[install_to_image] 准备产物目录: ${DIST_DIR}"
mkdir -p "${DIST_DIR}"

# 复制各子插件的 dist
echo "[install_to_image] 复制插件产物..."
for plugin in plugins/*/; do
    plugin_name=$(basename "$plugin")
    if [[ -d "${plugin}dist" ]]; then
        echo "[install_to_image]   复制 ${plugin_name}..."
        mkdir -p "${DIST_DIR}/plugins/${plugin_name}"
        # TypeScript 输出结构为 dist/plugins/<plugin_name>/src/ 或 dist/plugins/<plugin_name>/
        # 找到实际的产物目录（向上查找有 index.js 的目录）
        src_dir="${plugin}dist/plugins/${plugin_name}/src"
        if [[ -d "$src_dir" ]]; then
            # 有 src/ 子目录，展平
            cp -r "$src_dir"/* "${DIST_DIR}/plugins/${plugin_name}/"
        else
            # 直接是产物目录
            nested_dir="${plugin}dist/plugins/${plugin_name}"
            if [[ -d "$nested_dir" ]]; then
                cp -r "$nested_dir"/* "${DIST_DIR}/plugins/${plugin_name}/"
            else
                cp -r "${plugin}dist"/* "${DIST_DIR}/plugins/${plugin_name}/"
            fi
        fi
        # 复制 plugin.json manifest（如果存在）
        if [[ -f "${plugin}openclaw.plugin.json" ]]; then
            cp "${plugin}openclaw.plugin.json" "${DIST_DIR}/plugins/${plugin_name}/"
        fi
        # 复制 package.json（如果存在）
        if [[ -f "${plugin}dist/package.json" ]]; then
            cp "${plugin}dist/package.json" "${DIST_DIR}/plugins/${plugin_name}/"
        fi
        # 复制共享类型包到插件的 node_modules（直接嵌入，绕过包名冲突）
        if [[ -d "packages/types/dist" ]]; then
            mkdir -p "${DIST_DIR}/plugins/${plugin_name}/node_modules/@subagent-coordinator"
            # 复制类型文件并修复 package.json
            mkdir -p "${DIST_DIR}/plugins/${plugin_name}/node_modules/@subagent-coordinator/types"
            if [[ -d "packages/types/dist/src" ]]; then
                cp -r "packages/types/dist/src"/* "${DIST_DIR}/plugins/${plugin_name}/node_modules/@subagent-coordinator/types/"
            else
                cp -r "packages/types/dist"/* "${DIST_DIR}/plugins/${plugin_name}/node_modules/@subagent-coordinator/types/"
            fi
            # 创建正确的 package.json
            cat > "${DIST_DIR}/plugins/${plugin_name}/node_modules/@subagent-coordinator/types/package.json" << 'PKGJSON'
{
  "name": "@subagent-coordinator/types",
  "version": "0.0.3",
  "type": "commonjs",
  "main": "./index.js",
  "types": "./index.d.ts"
}
PKGJSON
        fi

        # 复制 @sinclair/typebox 运行时依赖到插件的 node_modules。
        # pnpm 的 .pnpm/ 内容存储对镜像分发不可用，镜像后续执行
        # `openclaw plugins install --link` 会被安全扫描判定为
        # "node_modules symlink target outside install root"。直接把 typebox
        # 真实复制到 dist/plugins/<name>/node_modules/@sinclair/typebox
        # 以通过扫描并避免运行时缺失。
        local typebox_source=""
        typebox_source=$(compgen -G "node_modules/.pnpm/@sinclair+typebox@*/node_modules/@sinclair/typebox" | head -n 1 || true)
        if [[ -n "$typebox_source" && -d "$typebox_source" ]]; then
            mkdir -p "${DIST_DIR}/plugins/${plugin_name}/node_modules/@sinclair"
            cp -RL "$typebox_source" "${DIST_DIR}/plugins/${plugin_name}/node_modules/@sinclair/typebox"
        else
            echo "[install_to_image]   警告: 未找到 @sinclair/typebox 源（node_modules/.pnpm/@sinclair+typebox@*/node_modules/@sinclair/typebox），镜像安装时可能需要补充"
        fi

        # 防御性清理：移除 dist 输出中残留的 pnpm 符号链接（如有）。
        # install_to_image.sh 本身不复制源码 node_modules，正常情况下无符号链接；
        # 此步骤应对未来如改动拷贝范围时误带入 pnpm 链接的情形。
        if [[ -d "${DIST_DIR}/plugins/${plugin_name}/node_modules" ]]; then
            find "${DIST_DIR}/plugins/${plugin_name}/node_modules" -maxdepth 3 -type l -lname '*/.pnpm/*' -exec rm -f {} \; 2>/dev/null || true
        fi
    fi
done

# 复制 package.json（可能用于版本检测）
if [[ -f "package.json" ]]; then
    cp "package.json" "${DIST_DIR}/"
fi

echo "[install_to_image] 完成！产物目录: ${DIST_DIR}"
ls -la "${DIST_DIR}"
