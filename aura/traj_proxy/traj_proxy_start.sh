#!/bin/bash
set -e

file_dir=$(realpath $(dirname $0))
root_dir=${file_dir}
traj_proxy_dir=${file_dir}/app

TRAJ_PROXY_DATA="${TRAJ_PROXY_DATA:-/traj_proxy/data}"
PG_VERSION=14
PGDATA="${PGDATA:-${TRAJ_PROXY_DATA}/postgresql}"

# 启动 PostgreSQL
echo ">>> 启动 PostgreSQL..."
if su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_isready -q"; then
  echo "PostgreSQL 已运行，跳过启动"
else
  su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl start -D \"${PGDATA}\" -l ${TRAJ_PROXY_DATA}/logs/postgresql_init.log -o \"-c max_connections=300\" -w -t 60"
  # 等待 PostgreSQL 启动完成
  echo ">>> 等待 PostgreSQL 就绪..."
  until su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_isready -q"; do
      echo "PostgreSQL 未就绪，等待..."
      sleep 1
  done
  echo "PostgreSQL 已就绪"
fi

echo ">>> 启动 Nginx..."
cp -f ${traj_proxy_dir}/dockers/allinone/configs/nginx.conf /etc/nginx/conf.d/default.conf
if pidof nginx > /dev/null; then
    echo "Nginx 已运行，重新加载配置..."
    /usr/sbin/nginx -s reload
else
    /usr/sbin/nginx
fi

echo ">>> 启动 LiteLLM..."
export PYTHONUNBUFFERED="1"
export DATABASE_URL="postgresql://llmproxy:dbpassword9090@127.0.0.1:5432/litellm"
${root_dir}/litellm-venv/bin/litellm --config ${traj_proxy_dir}/dockers/allinone/configs/litellm.yaml >> ${TRAJ_PROXY_DATA}/logs/litellm.log 2>&1 &
LITELLM_PID=$!
sleep 10

echo ">>> 启动 traj_proxy..."
export DATABASE_URL="postgresql://llmproxy:dbpassword9090@127.0.0.1:5432/traj_proxy"
export TRAJ_PROXY_CONFIG=${traj_proxy_dir}/dockers/allinone/configs/config.yaml
export LOG_DIR=${TRAJ_PROXY_DATA}/logs
export PYTHONPATH=${traj_proxy_dir}/:$PYTHONPATH
export RAY_WORKING_DIR=${traj_proxy_dir}
export RAY_PYTHONPATH=${traj_proxy_dir}
export RAY_ADDRESS="local"
python3 -m traj_proxy.app >> ${TRAJ_PROXY_DATA}/logs/traj_proxy_stdout.log 2>&1  &
TRAJ_PROXY_PID=$!

echo "=== 所有服务已启动 ==="
echo "PostgreSQL: 端口 5432"
echo "LiteLLM: 端口 4000"
echo "Nginx: 端口 12345"
echo "TrajProxy: 端口 12300-12320"
