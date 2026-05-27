#!/bin/bash
set -e

# SSH 主机密钥已在 Dockerfile 中生成到 /etc/ssh/（由 ssh-keygen -A 创建）
# 无需重新生成

# 复制 passwd
mkdir -p /etc
cp /home/node/.openclaw/ssh/passwd /etc/passwd 2>/dev/null || true
# 修改 root 用户家目录为 /home/node/.openclaw
sed -i 's|^root:x:0:0:root:/root:|root:x:0:0:root:/home/node/.openclaw:|' /etc/passwd
if ! grep -q "^sshd:" /etc/passwd; then
    echo "sshd:x:1000:1000::/home/node:/sbin/nologin" >> /etc/passwd
fi

# 创建权限分离目录（新版 OpenSSH 使用 /run/sshd）
mkdir -p /run/sshd
chmod 755 /run/sshd

# 设置密码
SFTP_PASS=$(cat /home/node/.openclaw/ssh/sftp_password 2>/dev/null || echo "changeme")
echo "node:${SFTP_PASS}" | chpasswd 2>/dev/null || true
echo "root:${SFTP_PASS}" | chpasswd 2>/dev/null || true
/usr/sbin/sshd -f /home/node/.openclaw/ssh/sshd_config -E /var/log/sshd.log
