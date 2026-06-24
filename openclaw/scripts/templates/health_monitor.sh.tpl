#!/bin/bash
# Health monitor script

HEALTH_CHECK_PORT=${HEALTH_CHECK_PORT:-8080}
HEALTH_CHECK_INTERVAL=${HEALTH_CHECK_INTERVAL:-30}
HEALTH_CHECK_FAILURE_THRESHOLD=${HEALTH_CHECK_FAILURE_THRESHOLD:-3}
OPENCLAW_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"

# 修复 volume 挂载目录的所有权
chown -R node:node /home/node/.openclaw/data 2>/dev/null || true
chown -R node:node /home/node/.claude 2>/dev/null || true

# 修复 Docker socket 权限（确保 node 用户可访问）
if [ -S /var/run/docker.sock ]; then
    # 获取宿主机 docker.sock 的 GID
    DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
    # 创建 docker 组（使用宿主机 GID）
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
    # 只终止管理的 gateway 进程
    for pid_file in /tmp/gateway_pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file" 2>/dev/null)
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null
            fi
            rm -f "$pid_file"
        fi
    done
    sleep 1
    # 强制终止残留进程
    for pid_file in /tmp/gateway_pid; do
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
    # 只保存自身 PID，子进程由单独 PID 文件管理
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
    for missing_plugin in self-improvement-monitor; do
        if grep -q "$missing_plugin" "$OPENCLAW_JSON" 2>/dev/null; then
            # 移除 plugins.load.paths 中对不存在插件的引用（完整的 JSON path 条目）
            python3 -c "
import json, sys
with open('$OPENCLAW_JSON', 'r') as f:
    data = json.load(f)
if 'plugins' in data and 'load' in data['plugins'] and 'paths' in data['plugins']['load']:
    data['plugins']['load']['paths'] = [p for p in data['plugins']['load']['paths'] if '$missing_plugin' not in p]
# Also remove slots.memory reference for the missing plugin
if 'plugins' in data and 'slots' in data['plugins'] and data['plugins']['slots'].get('memory') == '$missing_plugin':
    del data['plugins']['slots']['memory']
with open('$OPENCLAW_JSON', 'w') as f:
    json.dump(data, f, indent=2)
print('cleaned $missing_plugin')
" 2>/dev/null || true
        fi
    done
fi

# 恢复 memory 插件状态（doctor --fix 会重置 plugin 启用状态）
python3 -c "
import json
with open('$OPENCLAW_JSON', 'r') as f:
    data = json.load(f)

# restore memory plugin entries if they exist but were disabled
for entry_name in ['memory-core', 'memory-wiki']:
    if entry_name in data.get('plugins', {}).get('entries', {}):
        data['plugins']['entries'][entry_name]['enabled'] = True

# restore memory slot
if 'plugins' in data and 'slots' in data['plugins']:
    data['plugins']['slots']['memory'] = 'memory-core'

# restore memory backend
if 'memory' not in data:
    data['memory'] = {}
data['memory']['backend'] = 'builtin'

# restore memorySearch
if 'agents' in data and 'defaults' in data['agents']:
    ms = data['agents']['defaults'].get('memorySearch', {})
    ms['enabled'] = True
    ms['provider'] = 'none'
    missing = {'sync': {'watch': True, 'watchDebounceMs': 1500}, 'query': {'maxResults': 8, 'hybrid': {'enabled': True, 'vectorWeight': 0, 'textWeight': 1, 'candidateMultiplier': 4, 'mmr': {'enabled': False, 'lambda': 0.7}, 'temporalDecay': {'enabled': True, 'halfLifeDays': 30}}}, 'cache': {'enabled': True, 'maxEntries': 50000}}
    for k, v in missing.items():
        if k not in ms:
            ms[k] = v

with open('$OPENCLAW_JSON', 'w') as f:
    json.dump(data, f, indent=2)
print('[Health Monitor] memory plugin state restored after doctor --fix')
" 2>/dev/null || echo '[Health Monitor] memory plugin restoration failed'

# self-improvement-monitor 插件挂载（从 /tmp 复制到 /app/extensions）
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

save_pids

# 获取 Gateway 实际绑定地址
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
                    addr="${a}.${b}.${c}.${d}:${port}"
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

        # 有效地址直接返回
        if [ -n "$addr" ] && echo "$addr" | grep -q ":"; then
            echo "$addr"
            return 0
        fi

        # gateway 可能还在启动，等待后重试
        if [ $attempt -lt $max_attempts ]; then
            sleep 1
        fi
        attempt=$((attempt + 1))
    done

    # 返回空（多次重试后仍无法获取）
    echo ""
}

# 检查 Gateway 绑定地址是否可外部访问
is_gateway_externally_accessible() {
    local port=${OPENCLAW_GATEWAY_PORT:-12345}
    local bind_addr=$(get_gateway_bind_address)

    # gateway 可能还在启动，跳过本次检查（不计入失败）
    if [ -z "$bind_addr" ]; then
        echo "SKIP: cannot determine gateway bind address yet (gateway may still be starting)"
        return 2  # 使用返回值 2 表示跳过，不计入失败
    fi

    # 提取 IP（去端口）
    local ip=$(echo "$bind_addr" | sed 's/:[^:]*$//' | sed 's/^\[//; s/\]$//')

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
