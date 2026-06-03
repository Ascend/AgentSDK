# AgentSDK Openclaw

<p align="center">
  <b>高易用性的多领域 Agent 框架与服务</b><br>
  基于 OpenClaw 构建，集成代码生成、研究分析(开发中)等垂直领域 Agent 能力
</p>

---

## 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [特性](#特性)
- [使用指南](#使用指南)
- [技能详解](#当前实际包含的-skill)

---

## 项目概述

本项目构建一个**高易用性的 Agent 框架**，并提供相应的 **Agent 服务**。该框架以 OpenClaw 为基础，扩展多个**垂直领域的子 Agent**，包括但不限于：

- **代码生成（Coding）** - 智能代码补全、代码审查、Bug 修复(开发中)
- **研究分析（Research）** - 深度调研与报告生成(开发中)

### 核心优势

| 特性 | 说明 |
|------|------|
| **一键部署** | 单脚本启动多个容器实例 |
| **容器化交付** | 镜像形式提供，快速部署 |
| **插件化架构** | 支持垂直领域 Agent 即插即用 |
| **预置配置** | 无需交互式 onboard，启动即生效 |
| **SkillHub** | 可挂载的技能仓库，支持本地扩展 |

---

## 快速开始

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- 已部署的大模型服务（提供 API 端点）

### 1. 拉取或构建镜像

```bash
# 方式一：访问昇腾社区镜像仓库获取预构建镜像
# https://www.hiascend.com/developer/ascendhub/detail/5faf337534c847f0b135a52af924bbf4

# 方式二：基于 Dockerfile 构建（三层架构）
cd AgentSDK/openclaw/docker

# 一键构建全部三层镜像
bash ./build-openclaw.sh

# 仅构建最终镜像（跳过 base 和 app，使用已有镜像）
bash ./build-openclaw.sh --skip-base --skip-app

# 离线构建（使用本地源码）
OPENCLAW_SRC=/path/to/openclaw-src \
CLAUDE_MEM_SRC=/path/to/claude-mem-src \
bash ./build-openclaw.sh --offline --skip-base --skip-app \
  --claude-code-src /path/to/claude-code-best
```

#### 三层镜像架构

| 层级 | 镜像名称 | 说明 |
|------|---------|------|
| **Layer 1** | `openclaw-base` | SSH/uv/pip/npm 全局包/Playwright 等基础设施 |
| **Layer 2** | `openclaw` | 官方 OpenClaw 构建 |
| **Layer 3** | `openclaw-overlay` | 插件 + skills 定制层 |

#### 构建脚本参数

| 参数 | 说明 |
|------|------|
| `--offline` | 仅使用本地源码，不尝试网络克隆 |
| `--version VERSION` | 指定镜像版本号 |
| `--registry REGISTRY` | 镜像仓库前缀（默认 localhost） |
| `--skip-base` | 跳过 Layer 1 构建 |
| `--skip-app` | 跳过 Layer 2 构建 |
| `--skip-overlay` | 跳过 Layer 3 构建 |
| `--skip-plugins` | 跳过插件准备 |
| `--include-claude-code-best` | 启用 claude-code-best 打包 |
| `--openclaw-src PATH` | openclaw 源码目录 |
| `--claude-mem-src PATH` | claude-mem 源码目录 |
| `--claude-code-src PATH` | claude-code-best 源码目录 |
| `--npmmirror URL` | npm 镜像地址 |

### 2. 执行部署

```bash
# 进入项目目录
cd AgentSDK/openclaw

# 完整部署（生成配置 + 启动集群）
bash ./scripts/deploy.sh up -p 18789 -i openclaw:custom  # `-p` 表示 OpenClaw 初始容器实例的 gateway 起始端口，`-i` 表示部署使用的 OpenClaw 镜像名称与版本。

# 或使用单实例快速部署
bash ./scripts/deploy.sh quick -p 18789 -i openclaw:custom
```

其中 `-p` 表示 OpenClaw 初始容器实例的 gateway 起始端口，`-i` 表示部署使用的 OpenClaw 镜像名称与版本。

### 3. 其他部署命令

| 命令 | 说明 |
|------|------|
| `deploy.sh up` | 完整部署（生成配置 + 启动集群） |
| `deploy.sh scale` | 扩容部署（自动检测已有实例，仅新增容器） |
| `deploy.sh quick` | 单实例快速部署 |
| `deploy.sh config` | 仅生成配置文件 |
| `deploy.sh start` | 启动已有集群 |
| `deploy.sh start-one <name>` | 启动指定容器（如 `start-one openclaw-2`） |
| `deploy.sh stop` | 停止集群 |
| `deploy.sh stop-one <name>` | 停止指定容器（如 `stop-one openclaw-2`） |
| `deploy.sh version` | 版本管理 |

### 4. 访问 Web 界面

部署完成后，访问 `http://{节点IP}:18789` 开始与大模型对话并触发 Agent 任务。

### 5. 在 WebUI 中重启容器

若需要在 OpenClaw WebUI 页面中重启容器，可在交互对话框中输入 `/restart`，此外还可 kill gateway 进程（node 用户），而非 PID 1 的守护进程（root 用户），健康探针探测到异常后触发重启。

```bash
# kill gateway 进程，容器会自动重启
pkill -u node -f gateway
```

> 容器入口为 root 运行的守护进程（PID 1），负责拉起 gateway 等子进程。杀死 gateway 进程后健康探针检测到退出并触发容器重启。

---

## 项目结构

```text
openclaw/
│
├── docker/                           # Docker 相关配置
│   ├── Dockerfile.openclaw-app          # OpenClaw 主镜像构建文件
│   ├── Dockerfile.openclaw-base         # OpenClaw 基础依赖镜像构建文件
│   ├── Dockerfile.openclaw-overlay      # OpenClaw Overlay 层构建文件
│   ├── OVERVIEW.md                      # Docker 构建说明
│   ├── OVERVIEW.zh.md                   # Docker 构建中文说明
│   ├── image-checklist-deps.md          # 镜像依赖检查清单
│   ├── build-openclaw.sh                # 镜像构建脚本
│   └── .dockerignore                    # Docker 构建忽略文件
│
├── scripts/                          # 部署脚本
│   ├── deploy.sh                        # 一键部署脚本（主入口）
│   ├── defaults.env                     # 默认配置值
│   ├── openclaw-health-check.sh         # 独立健康检查脚本
│   ├── lib/                             # 部署脚本库
│   ├── patches/                         # 构建补丁
│   └── templates/                       # 配置模板
│
├── plugins/                          # OpenClaw 插件 Plugins
│   └── subagent-coordinator/            # 子 Agent 协调插件
│
├── skills/                           # Agent Skill（基于 OpenClaw 架构）
│   ├── subagent-coordinator/            # 主子协同 Skill（任务分解与多 Agent 协作）
│   ├── skill-vetter/                    # Skill 安全审查（权限审计/恶意代码检测）
│   ├── summarize-pro/                   # URL/文件/YouTube 摘要
│   ├── tuanziguardianclaw/              # 安全内核
│   └── mineru-to-markdown/              # 文档转 Markdown（PDF/图片/Office）
│
├── skillhub/                         # SkillHub CLI 与服务实现
│   ├── cli.py                           # 命令行入口
│   ├── config.py                        # 配置管理
│   ├── pyproject.toml                   # Python 项目配置
│   ├── adapters/                        # 适配层
│   ├── commands/                        # CLI 命令
│   ├── exceptions/                      # 异常定义
│   ├── interfaces/                      # 接口定义
│   ├── models/                          # 数据模型
│   ├── services/                        # 业务服务
│   └── utils/                           # 工具函数
│
├── tests/                            # 测试用例
│   └── skillhub/                        # SkillHub 测试
│
├── .gitignore                           # Git 忽略文件
├── pytest.ini                           # pytest 配置
└── README.md                            # 本文件
```

---

## 特性

### 1. OpenClaw 基础框架集成

| 功能点 | 描述 |
|--------|------|
| 框架派生 | 基于 OpenClaw 官方代码仓库派生，继承其核心 Agent 能力 |
| 配置化启动 | 通过挂载配置文件实现免交互 onboard，容器启动即生效 |
| 插件系统 | 支持动态加载/卸载插件，扩展框架功能 |
| 多模型支持 | 兼容本地模型等多种 LLM 后端 |

### 2. 垂直领域子 Agent (开发中)

#### 2.1 代码生成 Agent（Coding Agent）

| 功能点 | 描述 |
|--------|------|
| 智能代码补全 | 根据上下文生成代码片段，支持多种编程语言 |
| 代码审查 | 自动检测代码中的潜在问题，提供优化建议 |
| Bug 修复 | 分析错误信息，生成修复方案 |
| 代码重构 | 根据最佳实践重构代码结构 |
| 测试生成 | 自动生成单元测试用例 |

### 3. 通用 Agent Skill

AgentSDK Openclaw 当前预置的通用 Agent Skill 覆盖多 Agent 协作、安全审查、内容总结、文档转 Markdown 等能力。这些 Skill 基于 OpenClaw 架构开发，可直接挂载到容器中使用。

#### 当前实际包含的 Skill

下表仅列出当前仓库 `openclaw/skills/` 目录下实际包含的 Skill：

| Skill 名称 | 类型 | 功能描述 | 关键特性 | 来源 |
|------------|------|----------|----------|------|
| **subagent-coordinator** | 协同 | 主/子 Agent 任务编排与协调 | 多 Agent 协作、任务分解、状态共享 | `本仓特性` |
| **skill-vetter** | 安全 | Skill 安全审查 | 权限审计、恶意代码检测、信任分级 | `本仓特性` |
| **summarize-pro** | 工具 | 长篇内容清晰、简洁、可操作总结 | 多格式支持、关键信息提取 | `本仓特性` |
| **tuanziguardianclaw** | 安全 | 安全内核 | 拦截系统操作 | `本仓特性` |
| **mineru-to-markdown** | 文档 | 文档转 Markdown | PDF/图片/Office 转 MD，OCR，批量处理 | `本仓特性` |

---

### 4. SkillHub 技能仓库

| 功能点 | 描述 |
|--------|------|
| 本地挂载 | 支持将 SkillHub 挂载至容器内 |
| 动态安装 | 运行时安装 Skill，无需重建镜像 |
| 版本管理 | 支持 Skill 版本控制与依赖管理 |
| 自定义扩展 | 用户可开发并添加自定义 Skill |

### 5. 容器化与部署

| 功能点 | 描述 |
|--------|------|
| 分层镜像 | 支持 base/app/overlay 分层构建，灵活定制 |
| 多实例部署 | 一键脚本启动多个 OpenClaw 容器实例 |
| 配置挂载 | 配置文件挂载至容器，灵活调整 |
| Hermes 内置 | Hermes Agent 已集成至镜像（内置模式） |
| 健康监控 | 内置 health_monitor.sh 守护进程 + Docker Compose 原生 healthcheck，`docker ps` 直接展示健康状态 |
| 单实例启停 | 支持 `start-one`/`stop-one` 精确控制单个容器，不影响其他实例 |
| SFTP 支持 | 集成 SFTP 服务，支持文件传输 |
| 版本管理 | 支持镜像版本切换与回滚 |

## 使用指南

### 步骤一：清理旧环境（若存在）

部署前按需清理上一次遗留的容器和配置，使用部署脚本部署时若存在需配置的路径则会跳过。

> **说明**：以下示例中 `openclaw-configs` 为此前部署时设置的配置路径，`openclaw-1` 为此前部署时设置的容器名称，请按需设置。

```bash
cd AgentSDK/openclaw

# 停止并删除旧容器
if [ -f openclaw-configs/docker-compose.yml ]; then
    cd openclaw-configs && docker compose down && cd ..
fi
docker rm -f openclaw-1 2>/dev/null

# 删除旧配置目录
rm -rf openclaw-configs
```

### 步骤二：设置环境变量

```bash
# 推理服务 API Key（注入到容器的 ANTHROPIC_API_KEY 和 OPENAI_API_KEY）
export API_KEY=<your-api-key>
```

> **说明**：`API_KEY` 由用户根据自己使用的推理服务获取，部署脚本会将其注入到容器环境变量中。若对应推理服务无需 API_KEY 则可不用设置。

### 步骤三：执行部署

以下参数均为示例，用户需根据实际环境替换：

- **模型名称**（`-m, --model`）：取决于推理服务部署的模型，如 `Qwen3-30B-A3B`、`minimax-2.5` 等
- **推理服务 URL**（`-u, --url`）：取决于推理服务的实际部署地址和端口
- **起始端口**（`-p, --port`）：OpenClaw 初始容器实例的 gateway 起始端口
- **Docker 镜像**（`-i, --image`）：部署使用的 OpenClaw 镜像名称和版本

```bash
# 部署示例（生成配置 + 启动容器）
bash ./scripts/deploy.sh up \
  -n <实例数量> \
  -p <起始端口> \
  -m <模型名称> \
  --model-provider <供应商> \
  -u <推理服务URL> \
  -i <Docker镜像名称:版本>

# 示例：单实例部署，使用本地 Qwen3 模型
bash ./scripts/deploy.sh up \
  -n 1 \
  -p 8080 \
  -m Qwen3-30B-A3B \
  --model-provider local \
  -u http://0.0.0.0:8000 \
  -i openclaw:custom
```

该命令会在 `openclaw-configs/` 下自动生成：

- `docker-compose.yml` — 容器编排文件
- `instance-N/openclaw.json` — OpenClaw 配置（含模型、端口、插件等）
- `instance-N/health_monitor.sh` — 健康检查守护脚本
- `instance-N/.claude/settings.json` — Claude 配置
- `instance-N/.hermes/` — Hermes 配置
- `instance-N/plugins/` — 插件补丁

### 步骤四：等待 Gateway 初始化

Gateway 启动需要加载插件，约 60 秒完成初始化。

### 步骤五：确认部署成功

```bash
# 查看容器状态（应显示 healthy）
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep openclaw

# 确认 Gateway 已 ready
docker logs openclaw-1 2>&1 | grep 'gateway.*ready'

# 验证健康检查端点
curl -s http://localhost:8080/health
# 预期输出: {"ok":true,"status":"live"}
```

### 步骤六：日常运维操作

```bash
# 停止整个集群
bash ./scripts/deploy.sh stop

# 启动已有集群
bash ./scripts/deploy.sh start

# 只停止某个实例（不影响其他实例）
bash ./scripts/deploy.sh stop-one openclaw-2

# 只启动某个实例
bash ./scripts/deploy.sh start-one openclaw-2

# 扩容（自动检测已有实例，仅新增容器）
bash ./scripts/deploy.sh scale -n 2 -i <Docker镜像名称:版本>

# 版本管理
bash ./scripts/deploy.sh version list
bash ./scripts/deploy.sh version switch <目标镜像标签>
```

### 部署脚本参数参考

| 参数 | 描述 |
|------|------|
| `<命令>` | `up`：完整部署；`scale`：扩容；`quick`：单实例快速部署；`config`：仅生成配置；`start`/`stop`：启停集群；`start-one`/`stop-one`：单实例启停；`version`：版本管理 |
| `-n, --count NUM` | 实例数量（默认: 1） |
| `-p, --port BASE` | 起始端口（默认: 8080），每实例递增 4（Gateway、SFTP、MDNS、Guardian） |
| `-m, --model MODEL` | 模型名称（默认: Qwen3-30B-A3B） |
| `--model-provider NAME` | 模型供应商（默认: local） |
| `-u, --url URL` | 推理服务 URL（默认: http://0.0.0.0:8000） |
| `-c, --config-dir DIR` | 配置目录（默认: ./openclaw-configs） |
| `-i, --image IMAGE` | Docker 镜像（默认: openclaw:custom） |
| `--name PREFIX` | 实例名称前缀（默认: openclaw） |
| `--token TOKEN` | Gateway Token（不设置则自动生成） |
| `--skills` | 挂载宿主机 skills/ 目录到容器 |
| `--hermes` | 安装并配置 Hermes Agent |
| `--guardian-port PORT` | API Guardian 中间件监听端口 |
| `--no-down` | 扩容模式：不关闭已有实例 |
| `--start-index NUM` | 实例编号起始值（默认: 1） |
| `--subnet CIDR` | Docker 网络子网（默认: 172.30.0.0/16） |
| `--sandbox` | 启用 Docker 沙箱环境（默认: 不启用） |

### 环境变量

| 变量 | 描述 |
|------|------|
| `API_KEY` | 推理服务 API Key，注入到容器的 ANTHROPIC_API_KEY 和 OPENAI_API_KEY |
| `OPENCLAW_GATEWAY_TOKEN` | Gateway 认证 Token |

---
