# 1. 概述

## 1.1 简介

OpenClaw当前安装部署流程复杂，需手动配置依赖、SKILL、运行环境等，对用户技术能力要求高。本提案旨在通过标准化Docker镜像构建与部署方案，将常见依赖预置打包，通用SKILL集成至镜像，实现配置文件外挂与命令行部署。通过多层镜像架构与版本化管理，显著降低OpenClaw部署门槛，支持开发测试、生产部署等多场景快速上线。

## 1.2 动机

当前OpenClaw部署面临以下核心痛点：

- **部署门槛高**：用户需自行安装系统依赖（Git、Python运行时、各类库），非技术用户难以独立完成
- **配置复杂**：需手动克隆配置SKILL仓库、编写配置文件、调试运行环境问题
- **环境不一致**：不同用户环境差异大，导致运行结果不一致
- **升级困难**：版本更新时需重复执行复杂的手动配置流程
- **耗时过长**：整个部署流程耗时较长，严重影响用户体验

**用户案例：**

- 某企业运维人员首次部署OpenClaw，花费大量时间解决Python依赖冲突问题
- 某边缘设备现场部署时，因无外网环境无法下载依赖，导致部署失败
- 某开发团队成员各自配置环境，因环境差异导致运行结果不一致

**不做此提案的影响：**

- 高部署门槛严重阻碍OpenClaw的用户增长与商业化推广
- 环境不一致导致问题难以复现与定位，增加技术支持成本
- 现场部署困难限制边缘计算场景的应用落地

## 1.3 目标

**目标：**

- 提供标准化Docker镜像，采用多层架构打包常见依赖，实现开箱即用
- 预置通用SKILL与插件至镜像，兼容官方与社区扩展
- 支持配置文件外挂（openclaw.json、workspace目录），保障配置数据持久化
- 提供命令行构建工具，支持镜像构建与版本管理
- 建立版本化管理机制，支持版本切换与回滚
- 支持离线构建模式，满足无外网环境的部署需求

**非目标：**

- 不替代源码安装方式，高级用户仍可自定义构建
- 不涉及Kubernetes集群部署（由容器编排平台方案覆盖）
- 不提供在线IDE或Web界面（专注运行时环境）
- 不覆盖镜像仓库平台本身的运维与运营

# 2. 用例分析

## 2.1 用例1：新用户快速部署

**场景描述：** 新用户首次使用OpenClaw，希望快速完成部署并运行第一个Agent。

**功能点：**

- 通过构建脚本生成本地镜像
- 执行部署命令自动完成配置生成与容器启动
- 默认配置文件通过模板自动生成，可根据需要修改
- 部署完成后自动启动服务

**安全隐私要求：**

- 部署过程中不收集用户数据

**DFX要求：**

- **兼容性**：支持Docker环境
- **易用性**：构建脚本提供清晰的命令行帮助与默认值
- **可维护性**：支持平滑升级至新版本
- **可靠性**：部署失败时提供清晰的错误信息与修复建议

## 2.2 用例2：边缘设备现场部署

**场景描述：** 在无外网或弱网环境的边缘设备上部署OpenClaw。

**功能点：**

- 基于下载好的镜像包离线部署
- 镜像预置SKILL包
- 镜像导出导入（docker save/load）

**DFX要求：**

- **兼容性**：支持主流服务器架构
- **可靠性**：离线环境可依赖镜像包部署

## 2.3 用例3：企业多环境标准化部署

**场景描述：** 企业在开发、测试、生产多环境部署OpenClaw，要求环境一致性。

**功能点：**

- 配置文件外挂
- 多实例部署支持（单宿主机运行多个OpenClaw实例）
- 部署配置版本化管理

**DFX要求：**

- **兼容性**：支持私有镜像仓库
- **可重复性**：相同环境配置产生相同部署结果
- **可追溯性**：部署操作记录日志

# 3. 方案设计

## 3.1 总体方案

OpenClaw镜像发布采用三层镜像架构，核心设计如下：

```text
┌─────────────────────────────────────────────────────────────────┐
│                    三层镜像架构                                  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: 定制层 (Overlay Layer)                                │
│  ┌──────────────┬──────────────┬──────────────────────────────┐ │
│  │  claude-mem  │  subagent    │  项目SKILL                   │ │
│  │  插件        │  协调器插件   │  (官方+社区)                  │ │
│  ├──────────────┼──────────────┼──────────────────────────────┤ │
│  │  ccb CLI     │  Hermes Agent│  Playwright Chromium         │ │
│  │  (可选)      │  (预装)      │  (浏览器自动化)               │ │
│  └──────────────┴──────────────┴──────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: 应用层 (App Layer)                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  OpenClaw源码构建 / pnpm install / UI构建 / 插件SDK构建      │ │
│  └────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: 基础层 (Base Layer)                                   │
│  ┌──────────────┬──────────────┬──────────────────────────────┐ │
│  │  Node.js     │  系统依赖     │  Python运行时                 │ │
│  │  (官方镜像)   │  (Git/SSH/   │  (pip包/uv工具)               │ │
│  │              │   vim/curl)  │                              │ │
│  ├──────────────┼──────────────┼──────────────────────────────┤ │
│  │  Bun/pnpm    │  npm全局包    │  SSH服务                     │ │
│  │  (包管理器)   │  (文档处理)   │  (端口2222)                  │ │
│  └──────────────┴──────────────┴──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**镜像产品线规划：**

| 镜像名称 | 基础层 | 应用层 | 定制层 | 适用场景 |
|----------|--------|--------|--------|----------|
| openclaw-base | √ | 无 | 无 | 基础设施，作为后续构建基础 |
| openclaw-app | √ | √ | 无 | 官方OpenClaw运行时 |
| openclaw-overlay | √ | √ | √ | 完整功能，含插件与SKILL |

**核心设计思路：**

1. **分层构建**：三层Dockerfile独立构建，每层可单独缓存与复用
2. **离线支持**：构建脚本支持 `--offline` 模式，仅使用本地源码
3. **配置外挂**：配置文件与数据目录挂载至宿主机，镜像无状态化
4. **版本对齐**：镜像tag与OpenClaw版本号对齐，便于追溯

**技术平台选择：**

- **基础镜像**：node:24-bookworm（Debian Bookworm）
- **包管理器**：pnpm（Workspace管理）+ bun（插件构建）+ uv（Python工具）
- **构建工具**：Docker + Docker Compose
- **配置渲染**：envsubst（环境变量模板替换）+ Bash脚本

## 3.2 技术选型

### 3.2.1 基础镜像选型

| 方案 | 优势 | 劣势 | 结论 |
|------|------|------|------|
| node:24-bookworm | 官方维护，Node生态原生支持 | 体积相对较大 | **采用** |
| Alpine Linux | 体积极小 | musl libc兼容性问题，部分依赖不支持 | 放弃 |
| Ubuntu 22.04 LTS | 生态成熟，用户熟悉 | 与node官方镜像重复，增加维护成本 | 放弃 |

### 3.2.2 多阶段构建策略

```dockerfile
# Stage 1: 工具下载阶段（uv安装）
FROM node:24-bookworm AS uv-install
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install uv

# Stage 2: 基础层（系统依赖 + 工具链）
FROM node:24-bookworm AS openclaw-base
COPY --from=uv-install /root/.local/bin/uv /usr/local/bin/uv
RUN apt-get install -y git openssh-server python3 python3-pip
RUN npm install -g bun pnpm

# Stage 3: 应用层（源码构建）
FROM openclaw-base AS openclaw-app
COPY . /app
RUN pnpm install && pnpm build

# Stage 4: 定制层（插件注入）
FROM openclaw-app AS openclaw-overlay
COPY plugins/ /app/extensions/
COPY skills/ /home/node/.openclaw/skills/
```

### 3.2.3 放弃方案说明

| 方案 | 放弃理由 |
|------|----------|
| 单一全量镜像 | 构建时间长，缓存效率低，任何改动需重新构建全部 |
| 源码构建镜像 | 构建时间长，需要网络环境，不适用于离线部署 |
| 非Docker方案（如snap/flatpak） | 与容器生态不兼容，边缘设备支持差 |
| 多架构并行构建（buildx） | 当前阶段聚焦单架构优化，多架构作为后续扩展 |

## 3.3 功能与性能设计

### 3.3.1 Dockerfile设计

**基础层 Dockerfile 核心设计：**

```dockerfile
# 阶段1: 工具安装（uv等）
FROM node:24-bookworm AS uv-install
# 使用国内镜像源加速
RUN apt-get update && apt-get install -y curl python3-pip && \
    pip3 install -i https://mirrors.aliyun.com/pypi/simple uv

# 阶段2: 基础镜像
FROM node:24-bookworm AS openclaw-base
# 系统依赖（Git/SSH/curl等）
RUN apt-get install -y procps git openssh-server python3 python3-pip
# SSH安全加固（非标准端口，禁用root登录）
RUN sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
# 包管理器（bun/pnpm）
RUN npm install -g bun pnpm
# npm全局包（文档处理相关）
RUN npm install -g pptxgenjs sharp docx xlsx react react-dom
# pip包（文档解析相关）
RUN python3 -m pip install markitdown[pptx] python-pptx python-docx lxml
# 运行时用户（非root）
RUN useradd -m -u 1000 -s /bin/bash node
USER node
```

**应用层 Dockerfile 核心设计：**

```dockerfile
FROM openclaw-base AS openclaw-app
WORKDIR /app
# npm镜像配置
RUN npm config set registry https://registry.npmmirror.com
# 复制包管理文件
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
# pnpm install
RUN pnpm install --frozen-lockfile
# 复制全部源码
COPY . .
# OpenClaw构建
RUN pnpm build:docker && pnpm ui:build
# 裁剪生产依赖
RUN pnpm prune --prod
# 安装Playwright Chromium
RUN npx playwright install chromium
USER node
CMD ["node", "/app/dist/index.js"]
```

**定制层 Dockerfile 核心设计：**

```dockerfile
FROM openclaw-app AS openclaw-overlay
USER root
# 安装额外系统依赖（gosu/ffmpeg等）
RUN apt-get install -y gosu ffmpeg rsync
# 注入claude-mem插件
COPY claude-mem-dist/ /app/extensions/claude-mem/
# 注入subagent-coordinator插件
COPY subagent-coordinator-dist/ /app/extensions/
# 注入项目SKILL
COPY skills/ /home/node/.openclaw/skills/
# 预装Hermes Agent
RUN git clone hermes-agent && python3 -m venv venv && pip install -e "."
# 修复文件所有权
RUN chown -R node:node /app/dist /app/docs /home/node
# CLI命令注册
RUN echo '#!/bin/sh\nexec node /app/dist/index.js "$@"' > /usr/local/bin/openclaw
EXPOSE 2222 18792
CMD ["node", "/app/dist/index.js"]
```

### 3.3.2 配置文件外挂设计

**挂载点设计：**

```text
宿主机路径                      容器内路径                    说明
─────────────────────────────────────────────────────────────────────────
./openclaw-configs/instance-N/  /home/node/.openclaw/         主配置目录
  ├─ openclaw.json             →  OpenClaw主配置              网关/模型/Agent配置
  ├─ agents/main/agent/        →  Agent配置                   模型选择
  ├─ .hermes/config.yaml       →  Hermes Agent配置            多Agent支持
  ├─ .hermes/.env              →  API Key环境变量             敏感配置
  ├─ ssh/sshd_config           →  SSH服务配置                 端口/认证
  ├─ ssh/start_sshd.sh         →  SSH启动脚本                 容器内服务
  └─ health_monitor.sh         →  健康监控脚本                容器入口
```

**docker-compose 服务配置示例：**

```yaml
services:
  openclaw-1:
    image: openclaw:2026.4.11-overlay
    container_name: openclaw-1
    user: root
    ports:
      - "8080:8080"
      - "8081:8081"
    volumes:
      - ./openclaw-configs/instance-1:/home/node/.openclaw
      - openclaw-skills-1:/home/node/.openclaw/skills
    environment:
      - HOME=/home/node
      - OPENCLAW_HOME=/home/node/.openclaw
      - OPENCLAW_GATEWAY_TOKEN=<token>
      - ANTHROPIC_API_KEY=<key>
      - OPENAI_API_KEY=<key>
    command: /home/node/.openclaw/health_monitor.sh
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    restart: unless-stopped
```

### 3.3.3 构建脚本设计

**构建脚本核心功能：**

```bash
./build-openclaw.sh [选项]

# 选项:
#   --offline              仅使用本地源码，不尝试网络克隆
#   --version VERSION      指定版本（默认 2026.4.11）
#   --registry REGISTRY    镜像仓库前缀（默认 localhost）
#   --skip-base            跳过 Layer 1 构建
#   --skip-app             跳过 Layer 2 构建
#   --skip-overlay         跳过 Layer 3 构建
#   --openclaw-src PATH    openclaw 源码目录
```

**构建流程：**

```text
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 准备插件  │ -> │ 构建基础层 │ -> │ 构建应用层 │ -> │ 构建定制层 │
│ 预编译包  │    │ (Base)   │    │ (App)    │    │ (Overlay)│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │                │                │                │
     ▼                ▼                ▼                ▼
  claude-mem      系统依赖+        OpenClaw        插件注入+
  subagent        工具链安装       源码构建         SKILL注入+
  预编译          SSH配置                          Hermes预装
```

### 3.3.5 版本化管理设计

**镜像Tag规范：**

```text
[registry/]openclaw[:version][-overlay]

version: 版本号，如 2026.4.11
-overlay: 定制层后缀（含插件与SKILL）

示例：
  openclaw:2026.4.11              # 应用层镜像
  openclaw:2026.4.11-overlay      # 完整定制层镜像
  localhost/openclaw:2026.4.11    # 本地仓库镜像
```

### 3.3.6 性能指标设计

| 指标项 | 目标值 | 测试方法 |
|--------|--------|----------|
| 镜像构建时间 | 可接受范围内 | CI流水线完整构建 |
| 容器启动时间 | 较快 | 冷启动到服务就绪 |
| 首次部署耗时 | 显著低于手动部署 | 从零到Agent可运行 |
| 运行时内存占用 | 合理范围 | 容器资源监控 |

## 3.4 安全隐私与DFX设计

### 3.4.1 安全设计

- **镜像安全**：基础镜像使用官方维护的node镜像，定期更新基础依赖
- **最小权限**：基础镜像创建非root用户（node, UID=1000），应用运行时建议以此用户运行
- **SSH安全加固**：SSH服务监听非标准端口(2222)，禁用root密码登录，禁用密码认证
- **配置隔离**：敏感配置（API Key）通过外挂文件注入，不硬编码在镜像中
- **健康检查**：容器配置健康检查机制，异常自动重启

### 3.4.2 隐私设计

- 镜像内不预置任何用户数据或敏感信息
- API Key等敏感配置通过环境变量或外挂文件注入
- 支持配置加密卷挂载

### 3.4.3 DFX设计

**兼容性：**

- Docker环境支持
- 主流服务器架构支持
- Debian Bookworm基础镜像兼容主流Linux宿主机

**可维护性：**

- Dockerfile分层清晰，便于定位问题层
- 镜像历史透明（docker history）
- 构建脚本支持分层跳过，便于调试

**可测试性：**

- 提供镜像依赖检查清单
- 提供一键批量检查脚本
- 构建后可快速验证关键依赖完整性

**可靠性：**

- 容器异常自动重启（docker-compose restart策略）
- 健康检查机制（health_monitor.sh）

## 3.5 编程与调用设计

### 3.5.1 编程模型基本设计

**开发环境设计：**

- **硬件环境**：主流服务器架构
- **软件环境**：Docker、Bash
- **开发工具链**：Docker CLI、Make

**开发约束：**

- Dockerfile需遵循Docker最佳实践（层缓存、最小化层数）
- 基础镜像版本需明确指定，避免使用latest标签
- 构建上下文需包含完整的源码与插件产物

**可验收设计：**

- 镜像构建成功且关键依赖完整
- 构建脚本在目标环境执行通过

### 3.5.2 接口定义与设计

#### 3.5.2.1 image.build（镜像构建接口）

**接口描述：** 构建指定版本的OpenClaw镜像

**接口原型（构建脚本）：**

```bash
./build-openclaw.sh --version 2026.4.11 [--offline] [--skip-base] [--skip-app]
```

**输入参数：**

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
|----------|-----------|------|------|----------|
| --version | 输入 | str | OpenClaw版本号 | 版本号，必填 |
| --registry | 输入 | str | 镜像仓库前缀 | 默认 "localhost" |
| --offline | 输入 | bool | 仅使用本地源码 | true / false |
| --skip-base | 输入 | bool | 跳过基础层构建 | true / false |
| --skip-app | 输入 | bool | 跳过应用层构建 | true / false |
| --skip-overlay | 输入 | bool | 跳过定制层构建 | true / false |

**返回结果：**

- 成功：生成三层镜像（openclaw-base、openclaw、openclaw-overlay）
- 失败：返回错误信息（依赖缺失、构建失败等）

# 4. 缺点和风险

## 4.1 潜在风险

| 风险项 | 风险描述 | 影响等级 | 应对措施 |
|--------|----------|----------|----------|
| 镜像体积大 | 预置依赖与插件导致镜像体积较大 | 中 | 分层构建，基础层可复用缓存 |
| 依赖过时 | 镜像内预置依赖版本滞后 | 中 | 建立定期更新机制，提供更新指南 |
| 安全漏洞 | 基础镜像或依赖存在CVE | 高 | 及时更新基础镜像版本 |
| 构建失败 | 插件预编译或源码构建失败 | 中 | 构建脚本支持分层跳过，便于定位问题 |
| 宿主机兼容 | 不同宿主机Docker版本差异 | 低 | 明确最低版本要求，兼容性测试 |

## 4.2 负面影响

- **存储开销**：镜像占用一定磁盘空间
- **网络带宽**：首次拉取或构建镜像消耗网络带宽
- **灵活性降低**：标准化镜像可能无法满足高度定制化需求

## 4.3 实现成本

- **开发工作量**：Dockerfile设计、构建脚本、配置模板
- **测试验证**：多场景构建测试、镜像依赖完整性验证
- **维护成本**：跟进基础镜像版本更新、依赖版本升级

## 4.4 兼容性考虑

- **版本兼容**：镜像版本与OpenClaw版本号对齐，保证版本一致性
- **配置兼容**：新版本向后兼容旧版本配置文件
- **升级路径**：支持从源码部署平滑迁移至容器部署
- **数据兼容**：外挂数据目录格式保持稳定，升级不丢失数据

# 5. 现有技术

## 5.1 参考项目

### 5.1.1 Docker Official Images

- **借鉴点**：多阶段构建、安全最佳实践、官方镜像维护流程
- **差异点**：OpenClaw镜像针对AI Agent场景定制，预置SKILL生态与插件体系

### 5.1.2 Node.js Docker Image

- **借鉴点**：Node运行时环境配置、npm镜像源配置
- **差异点**：OpenClaw扩展了Python运行时、SSH服务、浏览器自动化等能力

### 5.1.3 Docker Compose

- **借鉴点**：多服务编排、配置文件管理、服务发现
- **差异点**：OpenClaw针对单宿主机多实例场景优化，自动生成compose配置

## 5.2 技术差异优势

| 维度 | 源码手动部署 | OpenClaw标准化镜像 |
|------|-------------|-------------------|
| 部署时间 | 数小时 | 显著缩短 |
| 环境一致性 | 低（各环境差异大） | 高（镜像标准化） |
| 离线支持 | 困难 | 支持（本地构建） |
| 升级难度 | 高（重复手动配置） | 低（替换镜像即可） |
| 依赖管理 | 手动解决冲突 | 镜像内预置兼容版本 |
| 回退能力 | 困难 | 简单（版本切换命令） |

# 6. 未解决问题

1. **镜像仓库集成**：当前为本地构建模式，与AscendHub等镜像仓库的集成发布方案
2. **镜像签名验证**：镜像完整性验证与签名机制
3. **多架构支持**：x86_64/aarch64多架构镜像并行构建

---

附录

- **术语表：**
  - **多层镜像**：通过多阶段构建将镜像分为基础层/应用层/定制层
  - **镜像定制层**：在官方镜像基础上注入插件与SKILL的扩展层
  - **配置外挂**：将配置文件和数据目录挂载到宿主机，实现镜像无状态化
  - **SKILL**：OpenClaw的能力扩展单元
  - **envsubst**：环境变量模板替换工具
