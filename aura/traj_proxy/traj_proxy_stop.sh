#!/bin/bash
set -e

file_dir=$(realpath $(dirname $0))
root_dir=${file_dir}

TRAJ_PROXY_DATA="${TRAJ_PROXY_DATA:-/traj_proxy/data}"
PG_VERSION=14
PGDATA="${PGDATA:-${TRAJ_PROXY_DATA}/postgresql}"

echo ">>> 停止 traj_proxy..."
if pgrep -f "python3 -m traj_proxy.app" > /dev/null 2>&1; then
    pkill -TERM -f "python3 -m traj_proxy.app" || true
    for i in $(seq 1 10); do
        if ! pgrep -f "python3 -m traj_proxy.app" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if pgrep -f "python3 -m traj_proxy.app" > /dev/null 2>&1; then
        echo "traj_proxy 未响应 SIGTERM，发送 SIGKILL..."
        pkill -KILL -f "python3 -m traj_proxy.app" || true
    fi
    echo "traj_proxy 已停止"
else
    echo "traj_proxy 未运行，跳过停止"
fi

echo ">>> 停止 Ray 集群..."
if pgrep -f "ray" > /dev/null 2>&1; then
    python3 -m ray stop 2>/dev/null || true
    for i in $(seq 1 10); do
        if ! pgrep -f "ray" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if pgrep -f "ray" > /dev/null 2>&1; then
        echo "Ray 进程未完全退出，强制杀死..."
        pkill -KILL -f "ray" || true
    fi
    echo "Ray 集群已停止"
else
    echo "Ray 集群未运行，跳过停止"
fi

echo ">>> 停止 LiteLLM..."
if pgrep -f "litellm" > /dev/null 2>&1; then
    pkill -TERM -f "litellm" || true
    for i in $(seq 1 5); do
        if ! pgrep -f "litellm" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if pgrep -f "litellm" > /dev/null 2>&1; then
        echo "LiteLLM 未响应 SIGTERM，发送 SIGKILL..."
        pkill -KILL -f "litellm" || true
    fi
    echo "LiteLLM 已停止"
else
    echo "LiteLLM 未运行，跳过停止"
fi

echo ">>> 停止 Nginx..."
if pidof nginx > /dev/null 2>&1; then
    /usr/sbin/nginx -s stop || true
    echo "Nginx 已停止"
else
    echo "Nginx 未运行，跳过停止"
fi

echo ">>> 停止 PostgreSQL..."
if su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_isready -q" 2>/dev/null; then
    su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl stop -D \"${PGDATA}\" -m fast -w" || true
    echo "PostgreSQL 已停止"
else
    echo "PostgreSQL 未运行，跳过停止"
fi

echo "=== 所有服务已停止 ==="
