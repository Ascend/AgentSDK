#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# TrajProxy start
set -ex

export RAY_raylet_start_wait_time_s=120

file_dir=$(realpath $(dirname $0))
root_dir=${file_dir}
traj_proxy_dir=${file_dir}/app

TRAJ_PROXY_DATA="${TRAJ_PROXY_DATA:-/traj_proxy/data}"
PG_VERSION=14
PGDATA="${PGDATA:-${TRAJ_PROXY_DATA}/postgresql}"

# Start PostgreSQL
echo ">>> Starting PostgreSQL..."
if su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_isready -q"; then
  echo "PostgreSQL is already running, skipping startup"
else
  su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl start -D \"${PGDATA}\" -l ${TRAJ_PROXY_DATA}/logs/postgresql_init.log -o \"-c max_connections=300\" -w -t 60"
  # Wait for PostgreSQL to start
  echo ">>> Waiting for PostgreSQL to be ready..."
  until su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_isready -q"; do
      echo "PostgreSQL not ready yet, waiting..."
      sleep 1
  done
  echo "PostgreSQL is ready"
fi

# Start Nginx
echo ">>> Starting Nginx..."
cp -f ${traj_proxy_dir}/dockers/allinone/configs/nginx.conf /etc/nginx/conf.d/default.conf
if pidof nginx > /dev/null; then
    echo "Nginx is already running, reloading configuration..."
    /usr/sbin/nginx -s reload
else
    /usr/sbin/nginx
fi

# Start LiteLLM
echo ">>> Starting LiteLLM..."
export PYTHONUNBUFFERED="1"
export DATABASE_URL="postgresql://llmproxy:dbpassword9090@127.0.0.1:5432/litellm"
${root_dir}/litellm-venv/bin/litellm --config ${traj_proxy_dir}/dockers/allinone/configs/litellm.yaml 2>&1 | tee ${TRAJ_PROXY_DATA}/logs/litellm.log &
LITELLM_PID=$!
sleep 10

# Start traj_proxy
echo ">>> Starting traj_proxy..."
export DATABASE_URL="postgresql://llmproxy:dbpassword9090@127.0.0.1:5432/traj_proxy"
export TRAJ_PROXY_CONFIG=${traj_proxy_dir}/dockers/allinone/configs/config.yaml
export LOG_DIR=${TRAJ_PROXY_DATA}/logs
export PYTHONPATH=${traj_proxy_dir}/:$PYTHONPATH
export RAY_WORKING_DIR=${traj_proxy_dir}
export RAY_PYTHONPATH=${traj_proxy_dir}
export RAY_ADDRESS="local"
python3 -m traj_proxy.app 2>&1 | tee ${TRAJ_PROXY_DATA}/logs/traj_proxy_stdout.log &
TRAJ_PROXY_PID=$!

echo "=== All services started ==="
echo "PostgreSQL: port 5432"
echo "LiteLLM: port 4000"
echo "Nginx: port 12345"
echo "TrajProxy: ports 12300-12320"
