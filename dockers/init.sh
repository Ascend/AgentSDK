#!/bin/bash

HOSTS_FLAG="/tmp/.hosts_modified"
BASHRC_FLAG="/tmp/.bashrc_modified"

# 写入 /etc/hosts（仅一次，避免重复添加）
if [ ! -f "$HOSTS_FLAG" ]; then
  echo "writting in /etc/hosts..."
  add_host() {
    grep -q "$2" /etc/hosts || echo "$1 $2" >> /etc/hosts
  }

  add_host "7.223.217.116" "cmc-cd-mirror.rnd.huawei.com"
  add_host "10.50.113.123" "registry.fusionstage.local"
  add_host "10.50.113.123" "csms-storagemgr.fst-manage.svc.cluster.local"
  add_host "7.222.218.218" "binaryartget.cmc.szv.dragon.tools.huawei.com"
  add_host "7.223.218.199" "cmcsearch.cmc.szv.dragon.tools.huawei.com"
  add_host "7.223.199.227" "mirrors.tools.huawei.com"
  add_host "100.95.17.174" "codehub-dg-y.huawei.com"

  touch "$HOSTS_FLAG"
else
  echo "/etc/hosts skipped"
fi

# 修改 ~/.bashrc（仅一次，避免重复添加）
if [ ! -f "$BASHRC_FLAG" ]; then
  echo "writting in /root/.bashrc..."

  append_if_missing() {
    grep -qF "$1" /root/.bashrc || echo "$1" >> /root/.bashrc
  }

  append_if_missing 'export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH'
  append_if_missing 'export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH'
  append_if_missing 'export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH'
  append_if_missing 'source /usr/local/Ascend/ascend-toolkit/set_env.sh'
  append_if_missing 'source /usr/local/Ascend/nnal/atb/set_env.sh'
  append_if_missing 'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python'

  touch "$BASHRC_FLAG"
else
  echo "~/.bashrc skipped"
fi

# 继续执行原始命令
exec "$@"
