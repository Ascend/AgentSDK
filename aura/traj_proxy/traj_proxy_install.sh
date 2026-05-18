#!/bin/bash
set -e

echo "7.223.219.58 mirrors.tools.huawei.com" | tee -a /etc/hosts

file_dir=$(realpath $(dirname $0))
root_dir=${file_dir}
traj_proxy_dir=${file_dir}/app

# System Dependency Installation
apt-get update
apt-get install -y --no-install-recommends nginx postgresql postgresql-client
rm -rf /var/lib/apt/lists/*

# traj_proxy dependency
pip install --no-cache-dir -r ${traj_proxy_dir}/requirements.txt --trusted-host mirrors.tools.huawei.com -i https://mirrors.tools.huawei.com/pypi/simple/
# litellm dependencies (use a separate virtual environment to avoid dependency conflicts)
python -m venv ${root_dir}/litellm-venv && \
    ${root_dir}/litellm-venv/bin/pip install --no-cache-dir 'litellm[proxy]' 'prisma' \
    'opentelemetry-api' 'opentelemetry-sdk' 'opentelemetry-exporter-otlp' \
    --trusted-host mirrors.tools.huawei.com -i https://mirrors.tools.huawei.com/pypi/simple/

wget --no-check-certificate https://mirrors.tools.huawei.com/nodejs/v24.14.0/node-v24.14.0-linux-arm64.tar.xz
tar -xf node-v24.14.0-linux-arm64.tar.xz -C /usr/local --strip-components=1
rm -fr node-v24.14.0-linux-arm64.tar.xz

npm config set strict-ssl false
npm config set registry https://mirrors.tools.huawei.com/npm/
npm install prisma

# Execute `prisma generate` during build to bake the engine binary into the image
# This way, running `prisma db push` at runtime does not require downloading the engine from the network
export HTTP_PROXY=http://10.50.113.120:3128
export HTTPS_PROXY=http://10.50.113.120:3128
export no_proxy="localhost,127.0.0.1,*.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.huawei.com"
export NO_PROXY="$no_proxy"
export NODE_TLS_REJECT_UNAUTHORIZED=0
export PATH="${root_dir}/litellm-venv/bin:$PATH"
${root_dir}/litellm-venv/bin/prisma generate \
    --schema ${root_dir}/litellm-venv/lib/python3.11/site-packages/litellm/proxy/schema.prisma
