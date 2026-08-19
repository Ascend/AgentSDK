#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# TrajProxy download
set -ex

file_dir=$(realpath $(dirname $0))
root_dir=${file_dir}
traj_proxy_dir=${file_dir}/app

# Install system dependencies
apt-get update
apt-get install -y --no-install-recommends nginx postgresql postgresql-client
rm -rf /var/lib/apt/lists/*


# Install traj_proxy dependencies
pip install --no-cache-dir -r ${traj_proxy_dir}/requirements.txt \
   -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# litellm dependencies (use a separate virtual environment to avoid conflicts)
python -m venv ${root_dir}/litellm-venv && \
   ${root_dir}/litellm-venv/bin/pip install --no-cache-dir 'litellm[proxy]' 'prisma' \
   'opentelemetry-api' 'opentelemetry-sdk' 'opentelemetry-exporter-otlp' 'fastapi==0.114.2' \
   -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# Download nodejs
wget --no-check-certificate https://mirrors.tools.huawei.com/nodejs/v24.14.0/node-v24.14.0-linux-arm64.tar.xz
tar -xf node-v24.14.0-linux-arm64.tar.xz -C /usr/local --strip-components=1
rm -fr node-v24.14.0-linux-arm64.tar.xz

npm config set strict-ssl false
npm config set registry https://registry.npmmirror.com
npm install prisma

# Run prisma generate at build time to bake the engine binary into the image,
# so prisma db push works without downloading the engine at runtime
export NODE_TLS_REJECT_UNAUTHORIZED=0
export PATH="${root_dir}/litellm-venv/bin:$PATH"
${root_dir}/litellm-venv/bin/prisma generate \
   --schema ${root_dir}/litellm-venv/lib/python3.11/site-packages/litellm/proxy/schema.prisma
