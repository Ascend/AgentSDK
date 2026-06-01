#!/bin/bash
# Health monitor script - 2026.4.1 版本
# 变更: claude-mem 改为内置插件模式，由 OpenClaw gateway 自动加载，无需单独启动 worker

HEALTH_CHECK_PORT=${HEALTH_CHECK_PORT:-8080}
HEALTH_CHECK_INTERVAL=${HEALTH_CHECK_INTERVAL:-30}
HEALTH_CHECK_FAILURE_THRESHOLD=${HEALTH_CHECK_FAILURE_THRESHOLD:-3}
OPENCLAW_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"

# ── 修复 volume 挂载目录的所有权 ────────────────────────────────────────────
mkdir -p /home/node/.openclaw/data/.claude-mem/logs
chown -R node:node /home/node/.openclaw/data 2>/dev/null || true
chown -R node:node /home/node/.claude 2>/dev/null || true

# ── 修复 Docker socket 权限 ──────────────────────────────────────────────────
# 确保 node 用户可以访问 Docker socket
if [ -S /var/run/docker.sock ]; then
    # 获取 docker.sock 的 GID
    DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
    # 创建 docker 组（使用宿主机的 GID，关键！）
    if ! getent group docker >/dev/null 2>&1; then
        groupadd -g "$DOCKER_GID" docker
    fi
    # 将 node 用户加入 docker 组
    usermod -aG docker node
    echo "[Health Monitor] Docker socket fixed (GID match: $DOCKER_GID)"
fi

echo "[Health Monitor] Starting health check monitor for port $HEALTH_CHECK_PORT"
echo "[Health Monitor] Check interval: ${HEALTH_CHECK_INTERVAL}s, Failure threshold: $HEALTH_CHECK_FAILURE_THRESHOLD"

pkill_wrapper() {
    # 只终止明确管理的进程：gateway 和 claude-mem worker
    for pid_file in /tmp/gateway_pid /tmp/claude_mem_worker_pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file" 2>/dev/null)
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null
            fi
            rm -f "$pid_file"
        fi
    done
    sleep 1
    # 强制终止仍然存在的进程
    for pid_file in /tmp/gateway_pid /tmp/claude_mem_worker_pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file" 2>/dev/null)
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null
            fi
            rm -f "$pid_file"
        fi
    done
}

save_pids() {
    # 只保存 health_monitor 自身的 PID，不保存子进程
    # 子进程由单独的 PID 文件管理
    echo $$ > /tmp/health_monitor_pids
}

cleanup() {
    echo "[Health Monitor] Shutting down services..."
    pkill_wrapper
    echo "[Health Monitor] All services stopped"
}

trap cleanup EXIT

# 启动 SSH
(/home/node/.openclaw/ssh/start_sshd.sh) &

# 兼容新镜像：若没有 ccb 命令但存在 cli.js，创建 bun 包装脚本
if [ ! -f /usr/local/bin/ccb ] && [ -f /home/node/.claude/cli.js ]; then
    cat > /usr/local/bin/ccb << 'CCBEOF'
#!/bin/bash
exec bun /home/node/.claude/cli.js "$@"
CCBEOF
    chmod +x /usr/local/bin/ccb
fi

# Hermes Agent：镜像已预装，仅确保 symlink 存在
if [ -f /home/node/.hermes/hermes-agent/venv/bin/hermes ]; then
    ln -sf /home/node/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
fi

# 运行 doctor --fix 修复启动项
node /app/dist/index.js doctor --fix
chown -R node:node /home/node/.openclaw 2>/dev/null || true

# 清理 openclaw.json 中引用了不存在插件路径的条目
OPENCLAW_JSON="/home/node/.openclaw/openclaw.json"
if [ -f "$OPENCLAW_JSON" ]; then
    for missing_plugin in claude-mem self-improvement-monitor; do
        if grep -q "$missing_plugin" "$OPENCLAW_JSON" 2>/dev/null; then
            # 移除 plugins.load.paths 中对不存在插件的引用（完整的 JSON path 条目）
            python3 -c "
import json, sys
with open('$OPENCLAW_JSON', 'r') as f:
    data = json.load(f)
if 'plugins' in data and 'load' in data['plugins'] and 'paths' in data['plugins']['load']:
    data['plugins']['load']['paths'] = [p for p in data['plugins']['load']['paths'] if '$missing_plugin' not in p]
# Also remove slots.memory reference for claude-mem
if 'plugins' in data and 'slots' in data['plugins'] and data['plugins']['slots'].get('memory') == '$missing_plugin':
    del data['plugins']['slots']['memory']
with open('$OPENCLAW_JSON', 'w') as f:
    json.dump(data, f, indent=2)
print('cleaned $missing_plugin')
" 2>/dev/null || true
        fi
    done
fi

# =============================================================================
# self-improvement-monitor 插件挂载
# /plugins/self-improvement-monitor 挂载到 /tmp/openclaw-self-improvement-monitor (ro)
# 复制到 /app/extensions/self-improvement-monitor 并修复权限
# =============================================================================
SIM_PATCH_SRC="/tmp/openclaw-self-improvement-monitor"
SIM_PATCH_TARGET="/app/extensions/self-improvement-monitor"
if [ -d "$SIM_PATCH_SRC" ]; then
    echo "[Health Monitor] Copying self-improvement-monitor plugin..."
    rm -rf "$SIM_PATCH_TARGET"
    mkdir -p "$SIM_PATCH_TARGET"
    cp -r "$SIM_PATCH_SRC/"* "$SIM_PATCH_TARGET/"
    chown -R root:root "$SIM_PATCH_TARGET"
    chmod -R go-w "$SIM_PATCH_TARGET"
    echo "[Health Monitor] self-improvement-monitor plugin patched and permissions fixed"
fi

# =============================================================================
# claude-mem 插件补丁: 修复插件文件权限
# 补丁文件挂载到 /tmp/openclaw-claude-mem-patch/index.js (ro)
# 由于 9p 挂载不支持 Unix 权限，需要复制到 overlay 层再设置权限
# 此修复必须在 Gateway 启动前执行
# 版本: v2026.4.11
# =============================================================================
CLAUDE_MEM_PATCH_SRC="/tmp/openclaw-claude-mem-patch/index.js"
CLAUDE_MEM_PATCH_TARGET="/app/extensions/claude-mem/dist/index.js"
if [ -f "$CLAUDE_MEM_PATCH_SRC" ]; then
    mkdir -p "$(dirname "$CLAUDE_MEM_PATCH_TARGET")"
    if cp "$CLAUDE_MEM_PATCH_SRC" "$CLAUDE_MEM_PATCH_TARGET" 2>/dev/null; then
        chmod 755 "$CLAUDE_MEM_PATCH_TARGET"
        chown root:root "$CLAUDE_MEM_PATCH_TARGET"
        echo "[Health Monitor] claude-mem plugin patched and permissions fixed"
    else
        echo "[Health Monitor] claude-mem plugin copy failed, skipping chmod/chown"
    fi
else
    echo "[Health Monitor] claude-mem patch source not found at $CLAUDE_MEM_PATCH_SRC"
fi

# =============================================================================
# claude-mem openclaw.plugin.json 补丁: 修复 kind 冲突
# 镜像内的 openclaw.plugin.json 声明 kind: "memory"，会与内置 memory-core 冲突
# 导致 claude-mem 被自动禁用（不报错不警告）
# 修复: 将 kind 改为 "extension"，使其不抢占 memory slot
# 此修复必须在 Gateway 启动前执行
# 版本: v2026.4.11
# =============================================================================
CLAUDE_MEM_PLUGIN_JSON="/app/extensions/claude-mem/openclaw.plugin.json"
if [ -f "$CLAUDE_MEM_PLUGIN_JSON" ]; then
    if grep -q '"kind": "memory"' "$CLAUDE_MEM_PLUGIN_JSON" 2>/dev/null; then
        sed -i 's/"kind": "memory"/"kind": "extension"/' "$CLAUDE_MEM_PLUGIN_JSON"
        echo "[Health Monitor] claude-mem openclaw.plugin.json patched: kind "memory" -> "extension""
    elif grep -q '"kind": "extension"' "$CLAUDE_MEM_PLUGIN_JSON" 2>/dev/null; then
        echo "[Health Monitor] claude-mem openclaw.plugin.json already has kind: extension (no patch needed)"
    else
        echo "[Health Monitor] claude-mem openclaw.plugin.json kind field not found, skipping patch"
    fi
else
    echo "[Health Monitor] claude-mem openclaw.plugin.json not found at $CLAUDE_MEM_PLUGIN_JSON"
fi

# =============================================================================
# claude-mem worker modes 目录修复
# 镜像中缺少 modes/code.json 文件，导致 worker 初始化失败返回 503
# 需要在 worker 启动前创建该文件
# 版本: v2026.4.11
# =============================================================================
CLAUDE_MEM_MODES_SRC="/tmp/openclaw-claude-mem-patch/modes/code.json"
CLAUDE_MEM_MODES_TARGET="/usr/local/lib/node_modules/claude-mem/modes"
if [ -f "$CLAUDE_MEM_MODES_SRC" ]; then
    mkdir -p "$CLAUDE_MEM_MODES_TARGET"
    cp "$CLAUDE_MEM_MODES_SRC" "$CLAUDE_MEM_MODES_TARGET/code.json"
    chmod 644 "$CLAUDE_MEM_MODES_TARGET/code.json"
    echo "[Health Monitor] claude-mem modes directory fixed"
else
    # Fallback: 使用内联 JSON 创建 code.json
    mkdir -p "$CLAUDE_MEM_MODES_TARGET"
    cat > "$CLAUDE_MEM_MODES_TARGET/code.json" << 'MODE_JSON_EOF'
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
MODE_JSON_EOF
    echo "[Health Monitor] claude-mem modes directory created (fallback)"
fi

# 沙箱启用时修复 docker 组权限（供 sandbox 使用）
if [ "${SANDBOX_ENABLED:-true}" = "true" ]; then
    groupadd -g 117 docker 2>/dev/null || true
    usermod -aG docker node 2>/dev/null || true
fi

# 启动 Gateway
GATEWAY_CMD="node /app/dist/index.js gateway --port $OPENCLAW_GATEWAY_PORT --bind lan"
if [ -n "$OPENCLAW_TOKEN" ]; then
    GATEWAY_CMD="$GATEWAY_CMD"
fi
GATEWAY_CMD="$GATEWAY_CMD --allow-unconfigured"

# 根据当前用户选择合适的执行方式，捕获 PID
if [ "$(id -u)" = "0" ]; then
    if command -v gosu &> /dev/null; then
        (gosu node bash -c "$GATEWAY_CMD") &
        echo $! > /tmp/gateway_pid
    elif command -v su-exec &> /dev/null; then
        (su-exec node bash -c "$GATEWAY_CMD") &
        echo $! > /tmp/gateway_pid
    else
        (su -s /bin/bash node -c "$GATEWAY_CMD") &
        echo $! > /tmp/gateway_pid
    fi
else
    (bash -c "$GATEWAY_CMD") &
    echo $! > /tmp/gateway_pid
fi

# ── 启动 claude-mem worker ──────────────────────────────────────────────────
echo "[Health Monitor] Starting claude-mem worker..."
CLAUDE_MEM_WORKER="/usr/local/lib/node_modules/claude-mem/scripts/worker-service.cjs"
# 确保日志目录存在，并清理旧容器残留的 stale PID 文件
mkdir -p /home/node/.openclaw/data/.claude-mem/logs
rm -f /home/node/.openclaw/data/.claude-mem/worker.pid
if [ -f "$CLAUDE_MEM_WORKER" ]; then
    if [ "$(id -u)" = "0" ]; then
        if command -v gosu &> /dev/null; then
            (gosu node bash -c "nohup bun $CLAUDE_MEM_WORKER > /home/node/.openclaw/data/.claude-mem/logs/worker-startup.log 2>&1 &") &
            echo $! > /tmp/claude_mem_worker_pid
        elif command -v su-exec &> /dev/null; then
            (su-exec node bun $CLAUDE_MEM_WORKER &) &
            echo $! > /tmp/claude_mem_worker_pid
        else
            (su -s /bin/bash node -c "nohup bun $CLAUDE_MEM_WORKER > /home/node/.openclaw/data/.claude-mem/logs/worker-startup.log 2>&1 &") &
            echo $! > /tmp/claude_mem_worker_pid
        fi
    else
        (nohup bun "$CLAUDE_MEM_WORKER" > /home/node/.openclaw/data/.claude-mem/logs/worker-startup.log 2>&1 &) &
        echo $! > /tmp/claude_mem_worker_pid
    fi
else
    echo "[Health Monitor] claude-mem worker not found at $CLAUDE_MEM_WORKER, skipping..."
fi

save_pids

# ── 启动 memex-openclaw-server ─────────────────────────────────────────────
echo "[Health Monitor] Starting memex-openclaw-server..."
MEMEX_KB_ROOT="${MEMEX_KB_ROOT:-/home/node/wiki}"
MEMEX_SERVER_PORT="${MEMEX_SERVER_PORT:-8080}"
MEMEX_SERVER_LOG="/home/node/.openclaw/data/memex-server.log"
mkdir -p "$(dirname "$MEMEX_SERVER_LOG")" /home/node/wiki
chown -R node:node /home/node/wiki 2>/dev/null || true

if command -v memex-openclaw-server &> /dev/null; then
    if [ "$(id -u)" = "0" ]; then
        if command -v gosu &> /dev/null; then
            gosu node bash -c "MEMEX_KB_ROOT=${MEMEX_KB_ROOT} nohup memex-openclaw-server --kb-root ${MEMEX_KB_ROOT} --port ${MEMEX_SERVER_PORT} --host 0.0.0.0 > ${MEMEX_SERVER_LOG} 2>&1 &"
        elif command -v su-exec &> /dev/null; then
            su-exec node bash -c "MEMEX_KB_ROOT=${MEMEX_KB_ROOT} nohup memex-openclaw-server --kb-root ${MEMEX_KB_ROOT} --port ${MEMEX_SERVER_PORT} --host 0.0.0.0 > ${MEMEX_SERVER_LOG} 2>&1 &"
        else
            su -s /bin/bash node -c "MEMEX_KB_ROOT=${MEMEX_KB_ROOT} nohup memex-openclaw-server --kb-root ${MEMEX_KB_ROOT} --port ${MEMEX_SERVER_PORT} --host 0.0.0.0 > ${MEMEX_SERVER_LOG} 2>&1 &"
        fi
    else
        MEMEX_KB_ROOT=${MEMEX_KB_ROOT} nohup memex-openclaw-server --kb-root ${MEMEX_KB_ROOT} --port ${MEMEX_SERVER_PORT} --host 0.0.0.0 > ${MEMEX_SERVER_LOG} 2>&1 &
    fi
    echo "[Health Monitor] memex-openclaw-server started (kb: ${MEMEX_KB_ROOT}, port: ${MEMEX_SERVER_PORT})"
else
    echo "[Health Monitor] memex-openclaw-server not found, skipping..."
fi

# ── 获取 Gateway 实际绑定地址 ─────────────────────────────────────────────
get_gateway_bind_address() {
    local port=${OPENCLAW_GATEWAY_PORT:-12345}
    local max_attempts=3
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        local addr=""

        # 方法1: 使用 lsof（如果可用）
        if command -v lsof &>/dev/null; then
            addr=$(lsof -i ":${port}" -sTCP:LISTEN -t 2>/dev/null | head -1)
            if [ -n "$addr" ]; then
                # lsof 返回 PID，尝试获取实际的地址
                addr=$(lsof -i ":${port}" -sTCP:LISTEN 2>/dev/null | grep ":${port}" | awk '{print $9}' | head -1)
            fi
        fi

        # 方法2: 使用 /proc/net/tcp（Linux 内核接口，最可靠）
        if [ -z "$addr" ] && [ -f /proc/net/tcp ]; then
            # 将端口转换为十六进制（小写）
            local port_hex=$(printf '%04x' $port)
            # 读取 /proc/net/tcp 和 /proc/net/tcp6，查找 LISTEN 状态
            local tcp_addr=$(cat /proc/net/tcp /proc/net/tcp6 2>/dev/null | grep -i "${port_hex}.*0A" | head -1)
            if [ -n "$tcp_addr" ]; then
                # 格式: local_address:port (hex)
                local ip_hex=$(echo "$tcp_addr" | awk '{print $2}' | cut -d':' -f1)
                # 转换十六进制 IP 为点分十进制（小端序，需要反转字节顺序）
                if [ ${#ip_hex} -eq 8 ]; then
                    # IPv4: 4 bytes in reverse order
                    local a=$((16#${ip_hex:6:2}))
                    local b=$((16#${ip_hex:4:2}))
                    local c=$((16#${ip_hex:2:2}))
                    local d=$((16#${ip_hex:0:2}))
                    addr="${d}.${c}.${b}.${a}:${port}"
                elif [ ${#ip_hex} -eq 32 ]; then
                    # IPv6: 简化处理，显示原始格式
                    addr="[$(echo "$ip_hex" | sed 's/../&:/g' | cut -c1-39)]:${port}"
                fi
            fi
        fi

        # 方法3: 使用 netstat（备选）
        if [ -z "$addr" ] && command -v netstat &>/dev/null; then
            addr=$(netstat -tlnp 2>/dev/null | grep ":${port}" | awk '{print $4}' | head -1)
        fi

        # 方法4: 使用 ss（备选）
        if [ -z "$addr" ] && command -v ss &>/dev/null; then
            addr=$(ss -tlnp 2>/dev/null | grep ":${port}" | awk '{print $4}' | head -1)
        fi

        # 如果获取到了有效地址，直接返回
        if [ -n "$addr" ] && echo "$addr" | grep -q ":"; then
            echo "$addr"
            return 0
        fi

        # 短暂等待后重试（gateway 可能还在启动）
        if [ $attempt -lt $max_attempts ]; then
            sleep 1
        fi
        attempt=$((attempt + 1))
    done

    # 多次重试后仍无法获取，返回空
    echo ""
}

# ── 检查 Gateway 绑定地址是否可外部访问 ───────────────────────────────────
is_gateway_externally_accessible() {
    local port=${OPENCLAW_GATEWAY_PORT:-12345}
    local bind_addr=$(get_gateway_bind_address)

    # 如果无法获取绑定地址，可能是 gateway 还在启动，跳过本次检查（不计入失败）
    if [ -z "$bind_addr" ]; then
        echo "SKIP: cannot determine gateway bind address yet (gateway may still be starting)"
        return 2  # 使用返回值 2 表示跳过，不计入失败
    fi

    # 提取 IP 部分（去掉端口）
    local ip=$(echo "$bind_addr" | sed 's/.*://' | sed 's/\]//' | cut -d':' -f1)

    # 127.0.0.1 不可外部访问
    if [ "$ip" = "127.0.0.1" ]; then
        echo "UNHEALTHY: gateway bound to 127.0.0.1 (not externally accessible)"
        return 1
    fi

    # 0.0.0.0 (所有接口) 或具体 LAN IP 都算可访问
    echo "HEALTHY: gateway bound to $bind_addr"
    return 0
}

failure_count=0
while true; do
    sleep "$HEALTH_CHECK_INTERVAL"
    if wget -q --spider --timeout=10 "http://127.0.0.1:${HEALTH_CHECK_PORT}/health" 2>/dev/null; then
        # 本地 health endpoint 正常，进一步检查 gateway 绑定地址
        check_result=$(is_gateway_externally_accessible)
        ret=$?

        if [ $ret -eq 1 ]; then
            # 返回 1 = 不健康（绑定到 127.0.0.1）
            failure_count=$((failure_count + 1))
            echo "$(date '+%Y-%m-%d %H:%M:%S') [Health Monitor] $check_result ($failure_count/$HEALTH_CHECK_FAILURE_THRESHOLD)"
            if [ "$failure_count" -ge "$HEALTH_CHECK_FAILURE_THRESHOLD" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') [Health Monitor] External accessibility check failed threshold reached. Exiting container to trigger restart..."
                exit 1
            fi
            continue
        elif [ $ret -eq 2 ]; then
            # 返回 2 = 跳过（gateway 还在启动），不计入失败
            echo "$(date '+%Y-%m-%d %H:%M:%S') [Health Monitor] $check_result"
            continue
        fi

        # ret = 0，健康
        if [ "$failure_count" -gt 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') [Health Monitor] Health check recovered"
        fi
        failure_count=0
    else
        failure_count=$((failure_count + 1))
        echo "$(date '+%Y-%m-%d %H:%M:%S') [Health Monitor] Health check failed ($failure_count/$HEALTH_CHECK_FAILURE_THRESHOLD)"
        if [ "$failure_count" -ge "$HEALTH_CHECK_FAILURE_THRESHOLD" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') [Health Monitor] Health check failed threshold reached. Exiting container to trigger restart..."
            exit 1
        fi
    fi
done
