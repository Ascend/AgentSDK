# OpenClaw Subagent-Coordinator 插件安装指南

## 概述

本文档描述如何安装 subagent-coordinator（SC）插件，实现主子协同功能。

安装脚本 `install-sc-local.sh` 会自动完成：

- 创建 worker 子代理
- 配置 main → worker 子代理权限
- 安装插件（符号链接或复制）
- 配置插件加载路径
- 配置 exec allowlist

## 部署模式

插件安装脚本不感知 OpenClaw 的部署方式，无论 OpenClaw 运行在宿主机还是容器中，安装流程一致。

| 模式 | 说明 |
|------|------|
| 宿主机模式 | OpenClaw 直接安装在本地，`openclaw` 命令可直接使用 |
| 容器模式 | OpenClaw 运行在容器中，需先通过 volume 挂载将 SC 项目映射到容器内，再执行安装 |

---

## 快速开始

### 前提条件

- `openclaw` 命令可用（宿主机模式）或容器内 `openclaw` 命令可用（容器模式）
- `~/.openclaw` 或指定目录存在

### 基本用法

```bash
# 默认安装（使用符号链接）
./install-sc-local.sh

# 指定 SC 源目录
./install-sc-local.sh /path/to/subagent-coordinator

# 使用复制代替符号链接
./install-sc-local.sh --copy

# 指定 OpenClaw 主目录
./install-sc-local.sh --openclaw-home /custom/path/.openclaw

```

---

## 容器模式安装

由于安装脚本不感知容器，容器模式需要用户自行处理 volume 挂载。

### 方式一：在容器内执行安装脚本

```bash
# 1. 将 SC 项目挂载到容器内
docker run -d \
  --name openclaw \
  -v /path/to/subagent-coordinator:/workspace/sc \
  openclaw-image

# 2. 在容器内执行安装脚本
docker exec openclaw bash -c "cd /workspace/sc && ./install-sc-local.sh"

```

### 方式二：使用 --openclaw-home 指定容器内路径

```bash
docker exec openclaw ./install-sc-local.sh \
  --openclaw-home /home/node/.openclaw

```

---

## 命令行选项

| 选项 | 说明 |
|------|------|
| `--openclaw-home <path>` | OpenClaw 主目录（默认 `~/.openclaw`） |
| `--copy` | 使用复制代替符号链接 |
| `--build` | 安装前执行 `pnpm install + build` |
| `--help` | 显示帮助信息 |

### 环境变量

| 变量 | 说明 |
|------|------|
| `OPENCLAW_HOME` | 同 `--openclaw-home`，优先级更低 |

---

## 安装后验证

### 1. 重启 Gateway

```bash
openclaw gateway restart

```

### 2. 检查 agents

```bash
openclaw agents list

```

**预期输出**：

```bash

Agents:
- main (default)
  Subagents: worker (allowed)
- worker

```

### 3. 测试子代理

```bash
openclaw agent --agent worker --message 'Hello'

```

**预期响应**：worker agent 回复 "Hello! How can I help you today?"

### 4. 检查插件加载

重启后观察日志，确认以下插件已加载：

```bash

@subagent-coordinator/exec-monitor
@subagent-coordinator/taskr
@subagent-coordinator/observability

```

---

## 目录结构

安装后，`~/.openclaw/workspace/` 结构如下：

```text

~/.openclaw/workspace/
├── plugins/
│   ├── subagent-exec-monitor/   # → 符号链接或复制
│   ├── subagent-taskr/          # → 符号链接或复制
│   └── subagent-observability/   # → 符号链接或复制
└── skills/
    └── subagent-coordinator/    # → 符号链接或复制

```

---

## 故障排查

### 1. "未找到 openclaw 命令"

**问题**：宿主机模式下 `openclaw` 未安装或不在 PATH 中

**解决**：

```bash
which openclaw
# 或确认 OpenClaw 已启动
openclaw gateway status

```

### 2. "未找到 OpenClaw 目录"

**问题**：`~/.openclaw` 或指定的 `--openclaw-home` 目录不存在

**解决**：确认 OpenClaw 已初始化

```bash
ls -la ~/.openclaw/

```

### 3. 插件未加载

**问题**：重启后插件未生效

**解决**：

1.确认 `~/.openclaw/openclaw.json` 中已添加插件路径：

```bash

cat ~/.openclaw/openclaw.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
paths = d.get('plugins', {}).get('load', {}).get('paths', [])
print('Plugin paths:', paths)
"

```

2.重启 Gateway：

```bash
openclaw gateway restart

```

### 4. suspicious ownership 警告

**问题**：插件目录所有权不是 root

**解决**（如有必要）：

```bash
sudo chown -R root:root ~/.openclaw/workspace/plugins/

```

### 5. 子代理调用失败

**问题**：main agent 无法调用 worker

**解决**：确认 main agent 的 `allowAgents` 配置：

```bash
cat ~/.openclaw/openclaw.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
for a in d['agents']['list']:
    if a['id'] == 'main':
        print('main allowAgents:', a.get('subagents', {}).get('allowAgents'))
"

```

---

## 相关资源

- 架构文档：`ARCHITECTURE.md`
- 插件 API：`PLUGIN_API.md`
- 用户指南：`USER_GUIDE.md`
- 迁移指南：`MIGRATION.md`
