# OpenClaw Subagent-Coordinator 插件安装指南

## 概述

本文档描述如何安装 subagent-coordinator（SC）插件，实现主子协同功能。

安装脚本 `install-sc-local.sh` 会自动完成：

- 创建 worker 子代理
- 配置 main → worker 子代理权限
- 安装插件（符号链接或复制）
- 将 Skill 同步到工作区目录和 OpenClaw 可发现目录
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
- 如需使用 `--build`，构建环境需要能解析 OpenClaw `plugin-sdk`。完整 OpenClaw 源码树中通常是 `packages/plugin-sdk`，容器镜像中可能是 `/app/packages/plugin-sdk`，宿主机全局安装通常使用 OpenClaw npm 包中的 `dist/plugin-sdk`。脚本会自动尝试发现，无法发现时可设置 `PLUGIN_SDK_SRC=/path/to/plugin-sdk`。

### 基本用法

```bash
# 默认安装（复制文件）
./install-sc-local.sh

# 指定 SC 源目录
./install-sc-local.sh /path/to/subagent-coordinator

# 使用符号链接代替复制（开发调试时使用）
./install-sc-local.sh --symlink

# 指定 OpenClaw 主目录
./install-sc-local.sh --openclaw-home /custom/path/.openclaw

# 宿主机无法自动发现 plugin-sdk 时显式指定
PLUGIN_SDK_SRC=/path/to/openclaw/dist/plugin-sdk ./install-sc-local.sh --build

```

如果 `pnpm install` 提示 lockfile 与 `package.json` 不一致，应先更新 `pnpm-lock.yaml`，确保依赖声明和锁文件同步后再安装。

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
| `--symlink` | 使用符号链接代替复制（开发调试时使用） |
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

重启后检查插件列表，确认以下三个插件 ID 都出现：

```bash
openclaw plugins list | grep subagent
```

```text
@subagent-coordinator/exec-monitor
@subagent-coordinator/taskr
@subagent-coordinator/observability
```

> **宿主机上裸跑 `openclaw` 命令前注意设置 `OPENCLAW_HOME`**。OpenClaw CLI 在宿主机上有两个 `openclaw.json`：
>
> - `~/.openclaw/openclaw.json`：宿主机 Gateway 和**默认** CLI 读取的 OUTER 配置。
> - `~/.openclaw/.openclaw/openclaw.json`：`openclaw plugins install --link` 写入的 INNER 配置。
>
> `openclaw plugins install --link` 只会更新 INNER 配置；OUTER 配置的 `plugins.entries` 缺条目时，加载器会**静默丢弃**部分插件。
>
> `install-sc-local.sh` 已在配置阶段把三个插件的 entries 同步写入 OUTER 配置，因此默认 CLI 也能看到完整三个插件 ID。
>
> 如果宿主机是 OpenClaw systemd service（`HOME=/home/chad`，未设置 `OPENCLAW_HOME`），Gateway 也会用 OUTER 配置加载插件。重跑一次 `./install-sc-local.sh --build` 可修复历史脏状态。

### 5. 检查 Skill 可发现性

```bash
openclaw skills info subagent-coordinator
```

该命令应能显示 `subagent-coordinator` 的 Skill 信息。脚本会把 Skill 同步到 `~/.openclaw/workspace/skills/subagent-coordinator/` 和 `~/.openclaw/skills/subagent-coordinator/`；如果只复制到工作区目录而 CLI 仍提示 `Skill not found`，请确认可发现目录中也存在 `SKILL.md`。

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

~/.openclaw/skills/
└── subagent-coordinator/        # OpenClaw CLI 可发现的 Skill 副本

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
