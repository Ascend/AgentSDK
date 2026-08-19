#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# TrajProxy stop
set -e

file_dir=$(realpath $(dirname $0))
root_dir=${file_dir}

TRAJ_PROXY_DATA="${TRAJ_PROXY_DATA:-/traj_proxy/data}"
PG_VERSION=14
PGDATA="${PGDATA:-${TRAJ_PROXY_DATA}/postgresql}"

# Stop traj_proxy and its Ray child processes
echo ">>> Stopping traj_proxy..."
if pgrep -f "python3 -m traj_proxy.app" > /dev/null 2>&1; then
    pkill -TERM -f "python3 -m traj_proxy.app" || true
    for i in $(seq 1 10); do
        if ! pgrep -f "python3 -m traj_proxy.app" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if pgrep -f "python3 -m traj_proxy.app" > /dev/null 2>&1; then
        echo "traj_proxy did not respond to SIGTERM, sending SIGKILL..."
        pkill -KILL -f "python3 -m traj_proxy.app" || true
    fi
    echo "traj_proxy stopped"
else
    echo "traj_proxy not running, skipping stop"
fi

# Stop Ray cluster (Ray child processes started by traj_proxy)
echo ">>> Stopping Ray cluster..."
if pgrep -f "ray" > /dev/null 2>&1; then
    python3 -m ray stop 2>/dev/null || true
    for i in $(seq 1 10); do
        if ! pgrep -f "ray" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if pgrep -f "ray" > /dev/null 2>&1; then
        echo "Ray processes did not exit completely, force killing..."
        pkill -KILL -f "ray" || true
    fi
    echo "Ray cluster stopped"
else
    echo "Ray cluster not running, skipping stop"
fi

# Stop LiteLLM
echo ">>> Stopping LiteLLM..."
if pgrep -f "litellm" > /dev/null 2>&1; then
    pkill -TERM -f "litellm" || true
    for i in $(seq 1 5); do
        if ! pgrep -f "litellm" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if pgrep -f "litellm" > /dev/null 2>&1; then
        echo "LiteLLM did not respond to SIGTERM, sending SIGKILL..."
        pkill -KILL -f "litellm" || true
    fi
    echo "LiteLLM stopped"
else
    echo "LiteLLM not running, skipping stop"
fi

# Stop Nginx
echo ">>> Stopping Nginx..."
if pidof nginx > /dev/null 2>&1; then
    /usr/sbin/nginx -s stop || true
    echo "Nginx stopped"
else
    echo "Nginx not running, skipping stop"
fi

# Stop PostgreSQL
echo ">>> Stopping PostgreSQL..."
if su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_isready -q" 2>/dev/null; then
    su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl stop -D \"${PGDATA}\" -m fast -w" || true
    echo "PostgreSQL stopped"
else
    echo "PostgreSQL not running, skipping stop"
fi

echo "=== All services stopped ==="
